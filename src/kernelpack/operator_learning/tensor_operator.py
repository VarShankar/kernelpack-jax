"""Dimension-safe compact operator covariance with an optional affine mean.

Sparse GPU Cholesky-QR with an explicit dense reference backend.
"""
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
import math
import time
from functools import partial
from .fastol import TrainingNormalizer
from kernelpack.kernels import squared_distances, squared_norm, safe_sqrt, wendland_c4_squared, matern52_squared
from kernelpack.domain.support import support_counts as _support_counts, compact_support as _compact_support


def _distance(x, y):
    return safe_sqrt(squared_distances(x, y))


def _gate(s):
    t = jnp.clip(2*(1-s), 0., 1.)
    value = t**5*(126+t*(-420+t*(540+t*(-315+70*t))))
    return jnp.where(s <= .5, 1., jnp.where(s >= 1., 0., value))


@jax.jit
def _normalized_svd(x):
    mean=jnp.mean(x,axis=0)
    sigma=jnp.std(x,axis=0,ddof=0)
    safe=jnp.where(sigma==0,1.,sigma);zero=sigma==0
    inputs=jnp.where(zero,0.,(x-mean)/safe)
    center=jnp.mean(inputs,axis=0)
    _,singular,vh=jnp.linalg.svd(inputs-center,full_matrices=False)
    threshold=jnp.maximum(1e-10,jnp.finfo(x.dtype).eps*max(x.shape))
    rank=jnp.minimum(jnp.count_nonzero(singular>singular[0]*threshold),len(x)-1)
    return mean,safe,zero,inputs,center,singular,vh,rank


@partial(jax.jit,static_argnames=("mean_dimension","parallel_dimension","include_constant"))
def _training_coordinates(inputs,center,vh,*,mean_dimension,parallel_dimension,include_constant=True):
    centered=inputs-center
    basis=vh[:mean_dimension].T;parallel=vh[:parallel_dimension].T
    zmean=centered@basis;scale=jnp.std(zmean,axis=0,ddof=1)
    p=_mean_values(centered,basis,scale,include_constant=include_constant)
    z=centered@parallel;perp=centered-z@parallel.T
    delta=jnp.maximum(jnp.maximum(.01*jnp.max(jnp.linalg.norm(centered,axis=1)),
        2.2*jnp.max(jnp.linalg.norm(perp,axis=1))),1e-8)
    gate=_gate(safe_sqrt(squared_norm(perp))/delta)
    return basis,parallel,scale,p,z,perp,delta,gate


def _mean_values(centered,basis,scale,*,include_constant):
    """Evaluate the declared mean space; disabled means zero columns."""
    if not include_constant:
        return jnp.empty((len(centered),0),dtype=centered.dtype)
    return jnp.concatenate([jnp.ones((len(centered),1),dtype=centered.dtype),
                            (centered@basis)/scale],axis=1)


@jax.jit
def _dense_kernel_factor(z,perp,gate,radius,ratio,delta,nugget,p):
    kernel=wendland_c4_squared(squared_distances(z,z)/radius**2)*matern52_squared(
        squared_distances(perp,perp)*(ratio/delta)**2)*gate[:,None]*gate[None,:]
    kernel=(kernel+kernel.T)/2
    kernel=kernel.at[jnp.diag_indices(len(z))].set(gate**2)
    matrix=kernel+nugget*jnp.eye(len(z),dtype=z.dtype)
    lower=jnp.linalg.cholesky(matrix)
    whitened=jl.solve_triangular(lower,p,lower=True) if p.shape[1] else jnp.empty_like(p)
    return matrix,jnp.linalg.norm(matrix),lower,whitened,jnp.count_nonzero(kernel)


@jax.jit
def _qr_left(whitened):
    if whitened.shape[1]>whitened.shape[0]:
        raise ValueError("Polynomial basis has more columns than training functions")
    if not whitened.shape[1]:
        return jnp.empty((0,len(whitened)),dtype=whitened.dtype)
    q,r=jnp.linalg.qr(whitened,mode="reduced")
    return jl.solve_triangular(r,q.T,lower=False)


@partial(jax.jit,static_argnames=("chunk",))
def _assemble_supported(rows,cols,z,perp,gate,radius,ratio,delta,nugget,*,chunk):
    count=len(rows);padding=(-count)%chunk
    ri=jnp.pad(rows,(0,padding)).reshape(-1,chunk)
    ci=jnp.pad(cols,(0,padding)).reshape(-1,chunk)
    def block(_,ij):
        i,j=ij
        v=wendland_c4_squared(squared_norm(z[i]-z[j])/radius**2)*matern52_squared(
            squared_norm(perp[i]-perp[j])*(ratio/delta)**2)*gate[i]*gate[j]
        return None,v+nugget*(i==j)
    return jax.lax.scan(block,None,(ri,ci))[1].ravel()[:count]


class TensorWendlandOperator:
    """Wendland on <=3 coordinates times Matérn-5/2 on the orthogonal part.

    A compact gate bounds support in the entire ambient space. The affine
    mean uses up to 15 training PCA coordinates by default. ``mean_dimension=0``
    disables the complete polynomial space, including its constant column;
    positive values include a constant and up to that many training PCA
    coordinates. Radius and normalized nugget
    must be supplied explicitly; this class does no parameter selection.
    """
    def __init__(self, inputs, *, radius, nugget, mean_dimension=15,
                 parallel_dimension=3, perpendicular_ratio=1., log_parallel_scales=None,
                 backend="cudss", rhs_block_size=128, cudss_reordering="default", profile_setup=False, _geometry=None):
        if not all(math.isfinite(v) and v > 0 for v in (radius,nugget,perpendicular_ratio)):
            raise ValueError('radius, nugget, and perpendicular_ratio must be positive')
        if (int(parallel_dimension) != parallel_dimension or int(mean_dimension) != mean_dimension
                or not 1 <= parallel_dimension <= 3 or mean_dimension < 0):
            raise ValueError('parallel_dimension must be 1..3 and mean_dimension nonnegative')
        self.profile_setup=bool(profile_setup)
        self.setup_timings={}
        self._setup_clock=time.perf_counter()
        x = jnp.asarray(inputs)
        if x.device.platform!="gpu":
            raise ValueError("Tensor operator requires GPU arrays; no CPU fallback is enabled")
        mean_dimension, parallel_dimension = int(mean_dimension), int(parallel_dimension)
        if (x.ndim != 2 or not x.size or x.dtype not in (jnp.float32,jnp.float64)
                or not bool(jnp.all(jnp.isfinite(x)))):
            raise ValueError('inputs must be a nonempty finite floating matrix')
        if backend not in ("dense", "cudss"):
            raise ValueError("backend must be dense or cudss")
        if int(rhs_block_size) != rhs_block_size or rhs_block_size < 1:
            raise ValueError("rhs_block_size must be positive")
        self.rhs_block_size = int(rhs_block_size)
        if cudss_reordering not in ("default", "none", "amd"):
            raise ValueError("cudss_reordering must be default, none or amd")
        self.cudss_reordering = cudss_reordering
        self.backend = backend
        self.sparse_factor = None
        self.training_inputs = x
        if _geometry is None:
            mean,safe,zero,self.inputs,self.center,singular,vh,rank=_normalized_svd(x)
            self.normalizer=TrainingNormalizer(mean,safe,zero)
            rank=int(rank);self.affine_rank=rank
            (self.basis,self.parallel_basis,self.scale,self.P,self.z,self.perp,self.delta,self.gate)=_training_coordinates(
                self.inputs,self.center,vh,mean_dimension=min(rank,mean_dimension),
                parallel_dimension=min(rank,parallel_dimension),include_constant=mean_dimension>0)
        else:
            # Internal handoff from training: retain the exact PCA coordinates,
            # including their orientation within repeated singular spaces.
            if (x.shape!=_geometry.inputs.shape or x.dtype!=_geometry.inputs.dtype or x.device!=_geometry.inputs.device
                    or not bool(jnp.all(x==_geometry.inputs))
                    or mean_dimension!=_geometry.config.mean_dimension
                    or parallel_dimension!=_geometry.config.parallel_dimension):
                raise ValueError("Frozen geometry must match the ordered training inputs and configuration")
            self.affine_rank=_geometry.affine_rank
            self.normalizer=_geometry.normalizer
            self.inputs=self.normalizer.transform(x)
            self.center=_geometry.center
            self.basis,self.parallel_basis,self.scale,self.P=(_geometry.basis,
                _geometry.parallel_basis,_geometry.mean_scale,_geometry.P)
            self.z,self.perp,self.delta,self.gate=(_geometry.z,_geometry.perp,_geometry.delta,_geometry.gate)
        self.log_parallel_scales=(jnp.zeros(self.z.shape[1],dtype=x.dtype,device=x.device)
            if log_parallel_scales is None else jnp.asarray(log_parallel_scales,dtype=x.dtype,device=x.device))
        if (self.log_parallel_scales.shape!=(self.z.shape[1],)
                or not bool(jnp.all(jnp.isfinite(self.log_parallel_scales))
                    &jnp.all(jnp.isfinite(jnp.exp(self.log_parallel_scales)))
                    &jnp.all(jnp.exp(self.log_parallel_scales)>0))):
            raise ValueError("log_parallel_scales must give one finite positive scale per parallel coordinate")
        self.z=self.z*jnp.exp(self.log_parallel_scales)
        self.radius = float(radius)
        self.nugget = float(nugget)
        self.perpendicular_ratio = float(perpendicular_ratio)
        self._mark_setup("normalization_svd_geometry" if _geometry is None else "frozen_geometry_handoff",
            self.P,self.z,self.perp,self.gate)
        self.refit(radius=radius, nugget=nugget)

    def _mark_setup(self,name,*arrays):
        if self.profile_setup:
            for array in arrays:
                array.block_until_ready()
            now=time.perf_counter()
            self.setup_timings[name]=now-self._setup_clock
            self._setup_clock=now

    def refit(self, *, radius, nugget):
        """Refactor fixed training geometry; recreate prepared predictors after this."""
        if self.profile_setup:
            self._setup_clock=time.perf_counter()
        if not all(math.isfinite(v) and v > 0 for v in (radius,nugget)):
            raise ValueError('radius and nugget must be positive')
        self.radius, self.nugget = float(radius), float(nugget)
        x = self.training_inputs
        if self.sparse_factor is not None:
            self.sparse_factor.close()
            self.sparse_factor = None
        if self.backend == "cudss":
            from ..sparse.cudss import CudssFactor
            # Two bounded passes, one scalar allocation count, one compiled
            # extraction for all row blocks. No numerical arrays leave GPU.
            n=len(x)
            radius_value=jnp.asarray(self.radius,dtype=x.dtype)
            indptr=_support_counts(self.z,radius_value)
            count=int(indptr[-1])
            if count<1:
                raise ValueError("Sparse operator exceeds cuDSS int32 CSR limits")
            rows,cols=_compact_support(self.z,radius_value,indptr,
                jnp.zeros(count,dtype=jnp.int32,device=x.device),jnp.zeros(count,dtype=jnp.int32,device=x.device))
            self._mark_setup("support_compaction",rows,cols)
            chunk=min(count,max(1,(32*1024**2)//(x.shape[1]*x.dtype.itemsize*3)))
            values=_assemble_supported(rows,cols,self.z,self.perp,self.gate,radius_value,
                jnp.asarray(self.perpendicular_ratio,dtype=x.dtype),self.delta,
                jnp.asarray(self.nugget,dtype=x.dtype),chunk=chunk)
            self.kernel_nnz=int(jnp.count_nonzero(values))
            self._mark_setup("kernel_assembly",values,indptr)
            self.sparse_factor=CudssFactor(indptr,cols,values,(n,n),matrix_type="spd",
                                          rhs_block_size=self.rhs_block_size,row_indices=rows,profile_setup=self.profile_setup,
                                          reordering=self.cudss_reordering)
            self.sparse_factor.enable_cuda_graph()
            self._mark_setup("csr_validation_and_factorization",self.sparse_factor._values)
            t=(self.sparse_factor.triangular_solve(self.P) if self.P.shape[1]
               else jnp.empty_like(self.P))
            self._mark_setup("forward_whitening",t)
        else:
            self.matrix,self.matrix_norm,self.lower,t,count=_dense_kernel_factor(
                self.z,self.perp,self.gate,jnp.asarray(self.radius,dtype=x.dtype),
                jnp.asarray(self.perpendicular_ratio,dtype=x.dtype),self.delta,
                jnp.asarray(self.nugget,dtype=x.dtype),self.P)
            self.kernel_nnz=int(count)
            if not bool(jnp.all(jnp.isfinite(self.lower))):
                raise RuntimeError('Tensor operator Cholesky failed')
        if self.backend == "dense":
            self._mark_setup("dense_assembly_factorization_whitening",t)
        left = _qr_left(t)
        self._mark_setup("qr_and_small_triangular_solve",left)
        self.mean_correction = (jnp.empty((0,len(self.P)),dtype=self.P.dtype) if not self.P.shape[1]
            else self.sparse_factor.triangular_solve(left.T,backward=True).T
            if self.backend == "cudss" else jl.solve_triangular(self.lower.T, left.T, lower=False).T)
        if not bool(jnp.all(jnp.isfinite(self.mean_correction))):
            raise RuntimeError("Nonfinite Cholesky-QR polynomial correction")
        reproduction=self.mean_correction@self.P
        tolerance=5e-5 if self.P.dtype==jnp.float32 else 1e-8
        if self.P.shape[1] and not bool(jnp.all(jnp.isfinite(reproduction)) &
                    (jnp.max(jnp.abs(reproduction-jnp.eye(self.P.shape[1],dtype=self.P.dtype)))<=tolerance)):
            raise RuntimeError("Cholesky-QR correction failed polynomial reproduction")
        self._mark_setup("backward_correction",self.mean_correction)

    def projected_distances(self, queries):
        queries=jnp.asarray(queries,dtype=self.inputs.dtype,device=self.inputs.device)
        centered = self.normalizer.transform(queries)-self.center
        return _distance((centered@self.parallel_basis)*jnp.exp(self.log_parallel_scales), self.z)

    def device_state(self):
        """Return every numerical inference buffer for explicit JIT arguments."""
        state=dict(mean=self.normalizer.mean,std_safe=self.normalizer.std_safe,
            zero=self.normalizer.zero,center=self.center,basis=self.basis,
            scale=self.scale,parallel_basis=self.parallel_basis,z=self.z,
            perp=self.perp,delta=self.delta,gate=self.gate,
            log_parallel_scales=self.log_parallel_scales,P=self.P,
            mean_correction=self.mean_correction,
            radius=jnp.asarray(self.radius,dtype=self.inputs.dtype,device=self.inputs.device),
            perpendicular_ratio=jnp.asarray(self.perpendicular_ratio,dtype=self.inputs.dtype,device=self.inputs.device))
        if self.backend == "cudss":
            state['factor']=self.sparse_factor.device_state()
        else:
            state.update(lower=self.lower,matrix=self.matrix,matrix_norm=self.matrix_norm)
        return state

    def kernel_matrix(self, queries, *, state=None):
        state=self.device_state() if state is None else state
        queries=jnp.asarray(queries,dtype=self.inputs.dtype,device=self.inputs.device)
        centered=jnp.where(state['zero'],0.,(queries-state['mean'])/state['std_safe'])-state['center']
        z = centered@state['parallel_basis']
        perp = centered-z@state['parallel_basis'].T
        z=z*jnp.exp(state['log_parallel_scales'])
        w = wendland_c4_squared(squared_distances(z,state['z'])/state['radius']**2)
        matern = matern52_squared(squared_distances(perp,state['perp'])*(state['perpendicular_ratio']/state['delta'])**2)
        gate = _gate(safe_sqrt(squared_norm(perp))/state['delta'])
        return w*matern*gate[:,None]*state['gate'][None,:]

    def solve(self, rhs, *, state=None):
        """Kernel solve with the same per-column backward-error check on both backends."""
        state=self.device_state() if state is None else state
        if self.backend == "cudss":
            return self.sparse_factor.solve_with_state(rhs,state=state['factor'])
        from ..sparse.cudss import _check_solution
        result=jl.cho_solve((state['lower'],True),rhs)
        return _check_solution(result,rhs,lambda x:state['matrix']@x,anorm=state['matrix_norm'])

    def prepare_coefficients(self,targets):
        """Optional fitted-coefficient evaluation; may lose Lagrange numerical accuracy."""
        y=jnp.asarray(targets,dtype=self.inputs.dtype)
        if y.shape[0]!=len(self.inputs):
            raise ValueError("targets must have the training axis first")
        output_shape=y.shape[1:];flat=y.reshape(len(y),-1)
        polynomial=self.mean_correction@flat
        coefficients=self.solve(flat-self.P@polynomial)
        if not bool(jnp.all(jnp.isfinite(coefficients))):
            raise RuntimeError("Nonfinite fitted operator coefficients")
        def evaluate(queries):
            q=jnp.asarray(queries,dtype=self.inputs.dtype)
            leading=q.shape[:-1];q=q.reshape(-1,q.shape[-1])
            centered=self.normalizer.transform(q)-self.center
            pq=_mean_values(centered,self.basis,self.scale,include_constant=bool(self.P.shape[1]))
            values=self.kernel_matrix(q)@coefficients+pq@polynomial
            return values.reshape(leading+output_shape)
        return jax.jit(evaluate)

    def lagrange_map(self, queries, *, state=None):
        state=self.device_state() if state is None else state
        q = jnp.asarray(queries,dtype=self.inputs.dtype,device=self.inputs.device)
        if q.ndim < 2 or not q.size or q.shape[-1] != self.inputs.shape[1]:
            raise ValueError('queries must have shape (...,query,training_features)')
        query_shape = q.shape[:-1]
        q = q.reshape(-1,q.shape[-1])
        centered=jnp.where(state['zero'],0.,(q-state['mean'])/state['std_safe'])-state['center']
        cross = self.kernel_matrix(q,state=state)
        ordinary = self.solve(cross.T,state=state).T
        if not state['P'].shape[1]:
            # The kernel solve above retains its independent original-matrix
            # backward-error check. Polynomial reproduction is vacuous here.
            result=jnp.where(jnp.all(jnp.isfinite(ordinary),axis=1)[:,None],ordinary,jnp.nan)
            return result.reshape(query_shape+(len(self.inputs),))
        pq = _mean_values(centered,state['basis'],state['scale'],include_constant=True)
        result = ordinary+(pq-ordinary@state['P'])@state['mean_correction']
        # Verify the actual query weights, including cancellation in the
        # Lagrange correction, rather than checking finiteness alone.
        residual=result@state['P']-pq
        tolerance=5e-5 if q.dtype==jnp.float32 else 1e-8
        valid=jnp.all(jnp.isfinite(residual),axis=1) & (
            jnp.max(jnp.abs(residual),axis=1)<=tolerance*(1+jnp.max(jnp.abs(pq),axis=1)))
        result=jnp.where(valid[:,None],result,jnp.nan)
        return result.reshape(query_shape+(len(self.inputs),))

    def close(self):
        if self.sparse_factor is not None:
            self.sparse_factor.close()
