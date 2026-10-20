"""Center-based local Lagrange functions on operator input samples.

One footprint per training center, not a query-dependent nearest-stencil
interpolant. With a nugget these are localized regularized inverse columns,
not exact cardinal functions. No decay theorem is asserted for arbitrary
high-dimensional input clouds.
"""
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np
from .fastol import TrainingNormalizer, wendland_c4_matrix


def matern_c4_matrix(x, y, epsilon):
    """Scalar C^4 Matern matrix exp(-z)(3+3z+z^2), z=epsilon*|x-y| (dense control kernel)."""
    r = jnp.sqrt(jnp.maximum(jnp.sum(x*x, 1)[:, None] + jnp.sum(y*y, 1)[None, :]
        - 2*jnp.matmul(x, y.T, precision=jax.lax.Precision.HIGHEST), 0))
    z = epsilon*r
    return jnp.exp(-z)*(3 + 3*z + z*z)


class LocalLagrangeOperator:
    def __init__(self, inputs, epsilon, footprint, *, nugget=0.,
                 kernel='matern', center_batch_size=16, gram=None):
        raw=jnp.asarray(inputs)
        if raw.ndim!=2 or not len(raw) or not bool(jnp.all(jnp.isfinite(raw))):
            raise ValueError('inputs must be a nonempty finite matrix')
        if int(footprint)!=footprint or not 1<=footprint<=len(raw):
            raise ValueError('footprint must be an integer from 1 to N')
        if epsilon<=0 or nugget<0 or center_batch_size<1:
            raise ValueError('positive epsilon and batch size; nonnegative nugget required')
        if kernel not in ('matern','wendland'):
            raise ValueError('kernel must be matern or wendland')
        self.kernel=matern_c4_matrix if kernel=='matern' else wendland_c4_matrix
        self.normalizer=TrainingNormalizer.fit(raw)
        self.inputs=self.normalizer.transform(raw)
        self.epsilon=float(epsilon); self.nugget=float(nugget)
        self.footprint=int(footprint); self.center_batch_size=int(center_batch_size)
        self._corrected_state=None
        n=len(raw); x=self.inputs
        # Exact pilot geometry. This currently costs O(N^2) memory/work.
        distances=jnp.maximum(jnp.sum(x*x,1)[:,None]+jnp.sum(x*x,1)[None,:]-2*x@x.T,0.)
        distances=distances.at[jnp.diag_indices(n)].set(jnp.inf)
        near=jnp.argsort(distances,axis=1,stable=True)[:,:self.footprint-1]
        self.indices=jnp.concatenate([jnp.arange(n)[:,None],near],axis=1)
        if gram is not None and np.shape(gram)!=(n,n):
            raise ValueError('gram must be unregularized N by N matrix for these normalized inputs')
        weights=[]; residuals=[]
        for start in range(0,n,self.center_batch_size):
            ids=self.indices[start:start+self.center_batch_size]
            if gram is None:
                local=jax.vmap(lambda z:self.kernel(z,z,self.epsilon))(x[ids])
            else:
                local=jnp.asarray(gram)[ids[:,:,None],ids[:,None,:]]
            regularized=local+self.nugget*jnp.eye(self.footprint)
            lower=jnp.linalg.cholesky(regularized)
            if not bool(jnp.all(jnp.isfinite(lower))):
                raise RuntimeError('Local footprint Cholesky failed; no hidden nugget/fallback')
            rhs=jnp.zeros((len(ids),self.footprint)).at[:,0].set(1.)
            solve=lambda l,b:jl.solve_triangular(l.T,jl.solve_triangular(l,b,lower=True),lower=False)
            w=jax.vmap(solve)(lower,rhs)
            jax.block_until_ready(w)
            residuals.append(jnp.max(jnp.linalg.norm(jnp.einsum('bij,bj->bi',regularized,w)-rhs,axis=1)))
            weights.append(w)
        self.weights=jnp.concatenate(weights)
        self.max_local_solve_residual=float(jnp.max(jnp.asarray(residuals)))

    def lagrange_map(self, queries):
        q=self.normalizer.transform(queries)
        kq=self.kernel(q,self.inputs,self.epsilon)
        parts=[]
        for start in range(0,len(self.inputs),self.center_batch_size):
            ids=self.indices[start:start+self.center_batch_size]
            w=self.weights[start:start+self.center_batch_size]
            parts.append(jnp.einsum('qbm,bm->qb',kq[:,ids],w))
        return jnp.concatenate(parts,axis=1)

    def predict(self, queries, coefficients):
        """Apply the same scalar input weights to every output coefficient."""
        c=jnp.asarray(coefficients)
        if c.shape[0]!=len(self.inputs):
            raise ValueError('leading coefficient axis must be training examples')
        return (self.lagrange_map(queries)@c.reshape(len(c),-1)).reshape((len(queries),)+c.shape[1:])

    def coefficient_matrix(self):
        """Sparse columns: rows are kernel centers, columns are basis centers."""
        from jax.experimental.sparse import BCOO
        centers=jnp.broadcast_to(jnp.arange(len(self.inputs))[:,None],self.indices.shape)
        ids=jnp.stack([self.indices.ravel(),centers.ravel()],axis=1)
        return BCOO((self.weights.ravel(),ids),shape=(len(self.inputs),len(self.inputs)))

    def corrected_lagrange_map(self, queries, *, operator_nugget=None,
                               gram=None, tolerance=1e-11, maxiter=30):
        """Solve in the local basis with GMRES, preserving the global model.

        Bare local weights give a quasi-interpolant. The corrected map solves
        (Kreg A)^T H^T = (Kquery A)^T. This pilot uses full Gram assembly;
        the local coefficient matrix A itself has only N*footprint entries.
        Returns the map and measured relative original operator residual.
        """
        from jax.scipy.sparse.linalg import gmres
        tau=self.nugget if operator_nugget is None else float(operator_nugget)
        key=(tau,float(tolerance),int(maxiter))
        state=self._corrected_state
        if state is not None and state[0] is gram and state[1]==key:
            _,_,solver,matvec=state
        else:
            k=self.kernel(self.inputs,self.inputs,self.epsilon) if gram is None else jnp.asarray(gram)
            kreg=k+tau*jnp.eye(len(k));a=self.coefficient_matrix()
            matvec=lambda z:kreg.T@z
            precondition=lambda z:a.T@z
            solve=lambda b:gmres(matvec,b,M=precondition,tol=tolerance,atol=0.,
                restart=min(30,len(k)),maxiter=maxiter,solve_method='incremental')[0]
            solver=jax.jit(jax.vmap(solve))
            # JAX arrays are immutable; do not mutate a NumPy gram after passing it.
            self._corrected_state=(gram,key,solver,matvec)
        # Apply the same local-basis left preconditioner, but terminate on
        # the original operator residual instead of a cancellation-prone
        # explicitly transformed basis matrix.
        rhs=self.kernel(self.normalizer.transform(queries),self.inputs,self.epsilon)
        h=solver(rhs)
        residual=jax.vmap(matvec)(h)-rhs
        relative=float(jnp.linalg.norm(residual)/jnp.maximum(jnp.linalg.norm(rhs),1e-300))
        if not bool(jnp.all(jnp.isfinite(h))) or relative>max(10*tolerance,1e-12):
            raise RuntimeError(f'Local-basis GMRES did not converge: residual={relative}')
        return h,relative
