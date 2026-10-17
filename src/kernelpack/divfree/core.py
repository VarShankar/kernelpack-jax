from __future__ import annotations

from dataclasses import dataclass
from functools import partial, lru_cache
import math

import jax
from jax import lax, vmap
import jax.numpy as jnp
from jax.custom_batching import custom_vmap

from kernelpack.accelerators import WarpUnavailableError, warp_available, warp_exact_knn
from kernelpack.geometry import distance_matrix
from kernelpack import poly
from kernelpack.poly.core import _tensor_evaluate_with_recurrence
from kernelpack.domain.stencils import exact_knn


def _as_points(x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] not in (2, 3):
        raise ValueError("points must have shape (n, 2) or (n, 3)")
    return x


@partial(jax.jit, static_argnames=("degree",))
def divfree_gram_matrix(x: jnp.ndarray, y: jnp.ndarray, degree: int) -> jnp.ndarray:
    """Assemble the divergence-free PHS block Gram matrix.

    The kernel is ``(-Delta I + grad grad^T) r^m`` with component-major
    stacking: all x-components, then y-components, then z-components.
    """
    x = jnp.asarray(x, dtype=float)
    y = jnp.asarray(y, dtype=float)
    dim = x.shape[1]
    if dim not in (2, 3) or y.shape[1] != dim:
        raise ValueError("x and y must have matching 2D or 3D point dimensions")
    if degree <= 0 or degree % 2 == 0:
        raise ValueError("divergence-free PHS expects a positive odd degree")

    diff = x[:, None, :] - y[None, :, :]
    r = jnp.sqrt(jnp.maximum(jnp.sum(diff * diff, axis=-1), 0.0)) + jnp.finfo(float).eps
    r_m2 = r ** (degree - 2)
    cross_coeff = degree * (degree - 2) * r ** (degree - 4)
    full_diag = -degree * (degree + dim - 2) * r_m2

    def block(a: int, b: int) -> jnp.ndarray:
        hess = cross_coeff * diff[:, :, a] * diff[:, :, b]
        return lax.cond(
            a == b,
            lambda h: h + degree * r_m2 + full_diag,
            lambda h: h,
            hess,
        )

    blocks = [[block(a, b) for b in range(dim)] for a in range(dim)]
    return jnp.block(blocks)


def DFPHS(dim: int, degree: int) -> tuple[int, int]:
    """Return a lightweight kernel token accepted by ``DivFreeGram``.

    The Python package exposes callable block entries for Matlab parity. In
    JAX, the traceable object is the direct block assembly, so this token keeps
    API familiarity without carrying Python callables into compiled code.
    """
    dim = int(dim)
    degree = int(degree)
    if dim not in (2, 3):
        raise ValueError("divergence-free PHS is only supported in 2D and 3D")
    if degree <= 0 or degree % 2 == 0:
        raise ValueError("divergence-free PHS expects a positive odd degree")
    return dim, degree


def DivFreeGram(kernel: tuple[int, int] | int, x: jnp.ndarray, y: jnp.ndarray, _r: jnp.ndarray | None = None) -> jnp.ndarray:
    del _r
    degree = int(kernel[1] if isinstance(kernel, tuple) else kernel)
    return divfree_gram_matrix(x, y, degree=degree)


def _stack_field(u: jnp.ndarray) -> jnp.ndarray:
    u = jnp.asarray(u, dtype=float)
    return jnp.ravel(u.T)


def _unstack_field(u: jnp.ndarray, dim: int) -> jnp.ndarray:
    u = jnp.asarray(u, dtype=float).reshape(dim, -1)
    return u.T


@partial(jax.jit, static_argnames=("poly_degree",))
def divfree_polynomial_stack(x: jnp.ndarray, poly_degree: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate the static divergence-free Legendre polynomial stack."""
    x = jnp.asarray(x, dtype=float)
    dim = x.shape[1]
    alpha = poly.total_degree_indices(dim, int(poly_degree) + 1)
    recurrence_a, recurrence_b = _legendre_recurrence_static(int(poly_degree) + 2)

    def eval_derivative(orders: jnp.ndarray) -> jnp.ndarray:
        values = _tensor_evaluate_with_recurrence(x, alpha, orders, recurrence_a, recurrence_b)
        return values[:, :, 0]

    if dim == 2:
        dphidx = eval_derivative(jnp.array([[1, 0]], dtype=int))
        dphidy = eval_derivative(jnp.array([[0, 1]], dtype=int))
        values = jnp.concatenate([dphidy, -dphidx], axis=0)
    elif dim == 3:
        dphidx = eval_derivative(jnp.array([[1, 0, 0]], dtype=int))
        dphidy = eval_derivative(jnp.array([[0, 1, 0]], dtype=int))
        dphidz = eval_derivative(jnp.array([[0, 0, 1]], dtype=int))
        zeros = jnp.zeros_like(dphidx)
        curl_a = jnp.concatenate([zeros, dphidz, -dphidy], axis=0)
        curl_b = jnp.concatenate([-dphidz, zeros, dphidx], axis=0)
        curl_c = jnp.concatenate([dphidy, -dphidx, zeros], axis=0)
        values = jnp.concatenate([curl_a, curl_b, curl_c], axis=1)
    else:
        raise ValueError("divergence-free polynomial stack is only supported in 2D and 3D")
    return values, alpha


def _legendre_recurrence_static(n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    idx = jnp.arange(int(n), dtype=float)
    a = jnp.zeros((int(n),), dtype=float)
    b_tail = idx * idx / jnp.maximum(4.0 * idx * idx - 1.0, 1.0)
    b = b_tail.at[0].set(2.0)
    return a, b


def _poly_count(dim: int, degree: int) -> int:
    if degree < 0:
        return 0
    return math.comb(dim + degree, dim)


def _divfree_poly_rank(dim: int, degree: int, row_count: int) -> int:
    rank = dim * _poly_count(dim, degree) - _poly_count(dim, degree - 1)
    return min(int(rank), int(row_count))


def _reduce_sampled_svd(raw,s,vh,rank,tolerance,scale_floor,basis=None):
    active=s[...,:rank]>tolerance*jnp.maximum(s[...,0,None],scale_floor)
    vectors=jnp.swapaxes(vh[...,:rank,:],-1,-2)
    if basis is not None:
        vectors=basis@vectors
    transform=vectors/jnp.where(active,s[...,:rank],1.)[...,None,:]
    transform=transform*active[...,None,:]
    ratio=jnp.min(jnp.where(active,s[...,:rank]/jnp.where(s[...,0,None]>0,s[...,0,None],1.),jnp.inf),axis=-1)
    ratio=jnp.where(jnp.any(active,axis=-1),ratio,0.)
    return raw@transform,transform,active,ratio


@partial(jax.jit,static_argnames=("dim","degree"))
def _coefficient_row_basis(alpha,beta,*,dim,degree):
    """GPU row space of the normalized-Legendre curl coefficient map."""
    derivatives=[]
    for axis in range(dim):
        match=jnp.all(jnp.where(jnp.arange(dim)==axis,True,
            beta[:,None,:]==alpha[None,:,:]),axis=-1)
        gap=alpha[None,:,axis]-beta[:,None,axis]
        value=jnp.sqrt(((2*alpha[None,:,axis]+1)*(2*beta[:,None,axis]+1)).astype(float))
        derivatives.append(jnp.where(match&(gap>0)&(gap%2==1),value,0.))
    if dim==2:
        coefficients=jnp.concatenate([derivatives[1],-derivatives[0]],axis=0)
    else:
        dx,dy,dz=derivatives
        zero=jnp.zeros_like(dx)
        coefficients=jnp.block([[zero,-dz,dy],[dz,zero,-dx],[-dy,dx,zero]])
    _,_,vh=jnp.linalg.svd(coefficients,full_matrices=False)
    rank=_divfree_poly_rank(dim,degree,coefficients.shape[0])
    basis=vh[:rank].T
    projection=jnp.linalg.norm(coefficients-(coefficients@basis)@basis.T)/jnp.maximum(jnp.linalg.norm(coefficients),1.)
    orthogonality=jnp.linalg.norm(basis.T@basis-jnp.eye(rank,dtype=basis.dtype))
    return basis,projection,orthogonality


@lru_cache(maxsize=32)
def _cached_polynomial_row_basis(dim,degree,dtype,device):
    # Only static degree/index bookkeeping is host work. Coefficients, SVD and
    # all validation norms execute on the requested device, once per cache key.
    with jax.default_device(device):
        alpha=poly.total_degree_indices(dim,degree+1)
        beta=poly.total_degree_indices(dim,degree)
        basis,projection,orthogonality=_coefficient_row_basis(alpha,beta,dim=dim,degree=degree)
        basis=basis.astype(dtype)
    tolerance=5e-5 if basis.dtype==jnp.float32 else 1e-12
    if not bool(jnp.isfinite(projection)&jnp.isfinite(orthogonality)
                &(projection<=tolerance)&(orthogonality<=tolerance)):
        raise RuntimeError("Divergence-free polynomial coefficient row-space check failed")
    return basis


def divfree_polynomial_row_basis(dim,degree,*,dtype,device):
    """Cached GPU curl row basis; pass it dynamically into compiled builders.

    For raw sampled stack R, R=R S S.T in exact arithmetic and S.T S=I.
    The smaller R S therefore retains every nonzero singular value of R.
    This eager setup helper must be called outside JIT/vmap; no sampled data
    or numerical linear algebra are moved to the host.
    """
    if dim not in (2,3) or int(degree)!=degree or degree<0:
        raise ValueError("Expected dimension 2/3 and nonnegative polynomial degree")
    return _cached_polynomial_row_basis(int(dim),int(degree),jnp.dtype(dtype),device)


def _sampled_polynomial_svd(matrix):
    # Keep the native batched Jacobi path for small matrices. Larger matrices
    # otherwise use serial Jacobi dispatch, which is expensive near rank loss.
    algorithm=(None if max(matrix.shape[-2:])<=32 else lax.linalg.SvdAlgorithm.QR)
    return lax.linalg.svd(matrix,full_matrices=False,algorithm=algorithm)


def _compressed_rank_candidate(raw,basis,tolerance,scale_floor):
    projected=raw@basis
    # Thin QR preserves every singular value and right singular vector. Only
    # R is retained; GPU lowering eliminates the unused Householder product Q.
    # A square/wide sampled stack cannot shrink further; retain its direct
    # SVD (also preserving JAX's existing wide-matrix derivative support).
    triangular=(jnp.linalg.qr(projected,mode="r")
        if projected.shape[-2]>projected.shape[-1] else projected)
    # These matrices exceed cuSolver's <=32 native batched Jacobi regime for
    # typical 3D stencils. Select its QR SVD directly; all algebra stays on GPU.
    _,s,vh=_sampled_polynomial_svd(triangular)
    rank=min(raw.shape[-2],basis.shape[-1])
    compressed=_reduce_sampled_svd(raw,s,vh,rank,tolerance,scale_floor,basis)
    # Weyl's bound, including floating row-space/orthogonality defects. Near
    # the unchanged sampled-rank threshold use the original full-stack SVD.
    residual=jnp.linalg.norm(raw-projected@basis.T,axis=(-2,-1))
    defect=jnp.linalg.norm(basis.T@basis-jnp.eye(basis.shape[-1],dtype=basis.dtype))
    roundoff=8*jnp.finfo(raw.dtype).eps*(
        max(raw.shape[-2:])*jnp.linalg.norm(raw,axis=(-2,-1))
        +max(projected.shape[-2:])*jnp.linalg.norm(projected,axis=(-2,-1)))
    bound=residual+jnp.linalg.norm(projected,axis=(-2,-1))*defect+roundoff
    threshold=tolerance*jnp.maximum(s[...,0],scale_floor)
    near=jnp.any(jnp.abs(s[...,:rank]-threshold[...,None])
        <=(1+tolerance)*bound[...,None],axis=-1)
    return compressed,near


def _compressed_rank_reduce_impl(raw,basis,tolerance,scale_floor):
    compressed,near=_compressed_rank_candidate(raw,basis,tolerance,scale_floor)
    rank=min(raw.shape[-2],basis.shape[-1])

    def reference(one_raw):
        # An ambiguous boundary still uses every column of the full sampled
        # stack and the original rank threshold; only its GPU algorithm varies.
        _,full_s,full_vh=_sampled_polynomial_svd(one_raw)
        return _reduce_sampled_svd(one_raw,full_s,full_vh,rank,tolerance,scale_floor)

    if raw.ndim==2:
        return lax.cond(near,reference,lambda _:compressed,raw)

    # A scalar conditional inside lax.map is intentional. A vmapped cond would
    # become select and factor every raw stack, including healthy stencils.
    # The outer conditional skips this serial exceptional path for a healthy
    # batch; the custom_vmap rule packs primal stencil axes before this path.
    batch_shape=raw.shape[:-2]
    flat_raw=raw.reshape((-1,)+raw.shape[-2:])
    flat_compressed=tuple(value.reshape((-1,)+value.shape[len(batch_shape):])
        for value in compressed)

    def refine(_):
        def one_stencil(values):
            one_near,one_raw,*one_compressed=values
            return lax.cond(one_near,reference,lambda _:tuple(one_compressed),one_raw)
        refined=lax.map(one_stencil,(near.reshape(-1),flat_raw,*flat_compressed))
        return tuple(value.reshape(original.shape)
            for value,original in zip(refined,compressed))

    return lax.cond(jnp.any(near),refine,lambda _:compressed,operand=None)


_compressed_rank_reduce_batched=custom_vmap(_compressed_rank_reduce_impl)


@_compressed_rank_reduce_batched.def_vmap
def _compressed_rank_reduce_batch(axis_size,in_batched,raw,basis,tolerance,scale_floor):
    if any(in_batched[1:]):
        raise ValueError("Polynomial row basis and rank policy must be shared across stencils")
    # Re-enter the wrapper to consume any remaining outer vmap level before
    # evaluating scalar exceptional-path conditionals in the implementation.
    return _compressed_rank_reduce_batched(raw,basis,tolerance,scale_floor), (in_batched[0],)*4


@jax.custom_jvp
def _compressed_rank_reduce(raw,basis,tolerance,scale_floor):
    return _compressed_rank_reduce_batched(raw,basis,tolerance,scale_floor)


@_compressed_rank_reduce.defjvp
def _compressed_rank_reduce_jvp(primals,tangents):
    # Keep the differentiable implementation visible to reverse-mode AD.
    # custom_vmap primitives have no transpose rule. Mapped derivative paths
    # retain their existing mathematics, without a sparse-fallback speed claim.
    return jax.jvp(_compressed_rank_reduce_impl,primals,tangents)


def _rank_reduce_polynomial_stack(raw,dim,degree,tolerance,scale_floor=0.,coefficient_basis=None):
    """Shared sampled-basis reduction with the caller's unchanged rank policy."""
    rank=_divfree_poly_rank(dim,degree,raw.shape[0])
    if coefficient_basis is not None:
        if coefficient_basis.shape!=(raw.shape[1],_divfree_poly_rank(dim,degree,dim*_poly_count(dim,degree))):
            raise ValueError("Coefficient row basis does not match the polynomial stack")
        return _compressed_rank_reduce(raw,coefficient_basis,tolerance,scale_floor)
    _,s,vh=jnp.linalg.svd(raw,full_matrices=False)
    return _reduce_sampled_svd(raw,s,vh,rank,tolerance,scale_floor)


@partial(jax.jit,static_argnames=("poly_degree","phs_degree"))
def prepare_divfree_local_system(nodes,queries,*,poly_degree,phs_degree,coefficient_basis=None):
    """Rank-reduced PHS/polynomial saddle system and evaluation matrix.

    Preserve the FastOL 1e-12 sampled-rank policy and inactive-column padding.
    This shared builder is also available to other local interpolators.
    """
    center=jnp.mean(nodes,axis=0)
    width=jnp.max(jnp.linalg.norm(nodes-center,axis=1))
    width=jnp.where(width>0,width,1.)
    raw,_=divfree_polynomial_stack((nodes-center)/width,poly_degree=poly_degree)
    p,transform,active,ratio=_rank_reduce_polynomial_stack(raw,nodes.shape[1],poly_degree,1e-12,
        coefficient_basis=coefficient_basis)
    gram=divfree_gram_matrix(nodes,nodes,degree=phs_degree)
    system=jnp.block([[gram,p],[p.T,jnp.diag((~active).astype(nodes.dtype))]])
    qraw,_=divfree_polynomial_stack((queries-center)/width,poly_degree=poly_degree)
    evaluation=jnp.concatenate([divfree_gram_matrix(queries,nodes,degree=phs_degree),qraw@transform],axis=1)
    return system,evaluation,ratio


@partial(jax.jit, static_argnames=("poly_degree",))
def df_poly_basis_from_legendre(
    x: jnp.ndarray,
    poly_degree: int,
    center: jnp.ndarray | None = None,
    scale: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x = _as_points(x)
    if center is None:
        center_arr = jnp.mean(x, axis=0)
    else:
        center_arr = jnp.asarray(center, dtype=float).reshape((x.shape[1],))
    shifted = x - center_arr
    if scale is None:
        width = jnp.max(jnp.linalg.norm(shifted, axis=1), initial=1.0)
        scale_arr = jnp.maximum(width, 1.0)
    else:
        scale_arr = jnp.maximum(jnp.asarray(scale, dtype=float), jnp.finfo(float).eps)
    xi = shifted / scale_arr
    raw_values, alpha = divfree_polynomial_stack(xi, poly_degree=poly_degree)
    raw_values = raw_values / scale_arr
    values,transform,_,_=_rank_reduce_polynomial_stack(
        raw_values,x.shape[1],int(poly_degree),1e-10,jnp.finfo(float).eps)
    return values, center_arr, scale_arr, alpha, transform


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class DivFreePHSInterpolant:
    nodes: jnp.ndarray
    rbf_coeffs: jnp.ndarray
    poly_coeffs: jnp.ndarray
    center: jnp.ndarray
    scale: jnp.ndarray
    poly_transform: jnp.ndarray
    poly_degree: int
    phs_degree: int

    def tree_flatten(self):
        children = (self.nodes, self.rbf_coeffs, self.poly_coeffs, self.center, self.scale, self.poly_transform)
        aux = (self.poly_degree, self.phs_degree)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        poly_degree, phs_degree = aux
        nodes, rbf_coeffs, poly_coeffs, center, scale, poly_transform = children
        return cls(nodes, rbf_coeffs, poly_coeffs, center, scale, poly_transform, poly_degree, phs_degree)

    @staticmethod
    def fit(x: jnp.ndarray, u: jnp.ndarray, poly_degree: int, phs_degree: int) -> "DivFreePHSInterpolant":
        return fit_divfree_phs(x, u, poly_degree=poly_degree, phs_degree=phs_degree)

    def evaluate(self, xq: jnp.ndarray) -> jnp.ndarray:
        return evaluate_divfree_phs(self, xq)


@partial(jax.jit, static_argnames=("poly_degree", "phs_degree"))
def fit_divfree_phs(
    x: jnp.ndarray,
    u: jnp.ndarray,
    *,
    poly_degree: int,
    phs_degree: int,
) -> DivFreePHSInterpolant:
    x = _as_points(x)
    u = jnp.asarray(u, dtype=float)
    if u.shape != x.shape:
        raise ValueError("x and u must both have shape (n, dim)")

    gram = divfree_gram_matrix(x, x, degree=phs_degree)
    pmat, center, scale, _alpha, transform = df_poly_basis_from_legendre(x, poly_degree=poly_degree)
    zeros_pp = jnp.zeros((pmat.shape[1], pmat.shape[1]), dtype=x.dtype)
    saddle = jnp.block([[gram, pmat], [pmat.T, zeros_pp]])
    rhs = jnp.concatenate([_stack_field(u), jnp.zeros((pmat.shape[1],), dtype=x.dtype)])
    coeffs = jnp.linalg.solve(saddle, rhs)
    n_rbf = gram.shape[1]
    return DivFreePHSInterpolant(
        nodes=x,
        rbf_coeffs=coeffs[:n_rbf],
        poly_coeffs=coeffs[n_rbf:],
        center=center,
        scale=scale,
        poly_transform=transform,
        poly_degree=poly_degree,
        phs_degree=phs_degree,
    )


@jax.jit
def evaluate_divfree_phs(model: DivFreePHSInterpolant, xq: jnp.ndarray) -> jnp.ndarray:
    xq = _as_points(xq)
    gram = divfree_gram_matrix(xq, model.nodes, degree=model.phs_degree)
    xiq = (xq - model.center) / model.scale
    raw_pmat, _ = divfree_polynomial_stack(xiq, poly_degree=model.poly_degree)
    pmat = (raw_pmat / model.scale) @ model.poly_transform
    stacked = gram @ model.rbf_coeffs + pmat @ model.poly_coeffs
    return _unstack_field(stacked, xq.shape[1])


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class LocalDivFreeInterpolator:
    nodes: jnp.ndarray
    center_indices: jnp.ndarray
    stencil_indices: jnp.ndarray
    rbf_coeffs: jnp.ndarray
    poly_coeffs: jnp.ndarray
    centers: jnp.ndarray
    scales: jnp.ndarray
    poly_transforms: jnp.ndarray
    poly_degree: int
    phs_degree: int

    def tree_flatten(self):
        children = (
            self.nodes,
            self.center_indices,
            self.stencil_indices,
            self.rbf_coeffs,
            self.poly_coeffs,
            self.centers,
            self.scales,
            self.poly_transforms,
        )
        aux = (self.poly_degree, self.phs_degree)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        poly_degree, phs_degree = aux
        nodes, center_indices, stencil_indices, rbf_coeffs, poly_coeffs, centers, scales, poly_transforms = children
        return cls(nodes, center_indices, stencil_indices, rbf_coeffs, poly_coeffs, centers, scales, poly_transforms, poly_degree, phs_degree)

    @staticmethod
    def fit(
        x: jnp.ndarray,
        u: jnp.ndarray,
        poly_degree: int,
        phs_degree: int,
        stencil_size: int,
        *,
        center_indices: jnp.ndarray | None = None,
        stencil_indices: jnp.ndarray | None = None,
    ) -> "LocalDivFreeInterpolator":
        return fit_local_divfree(
            x,
            u,
            poly_degree=poly_degree,
            phs_degree=phs_degree,
            stencil_size=stencil_size,
            center_indices=center_indices,
            stencil_indices=stencil_indices,
        )

    def evaluate(self, xq: jnp.ndarray) -> jnp.ndarray:
        return evaluate_local_divfree(self, xq)


@partial(jax.jit, static_argnames=("k",))
def knn_indices_jax(points: jnp.ndarray, query_points: jnp.ndarray, k: int) -> jnp.ndarray:
    points = _as_points(points)
    query_points = jnp.atleast_2d(jnp.asarray(query_points, dtype=float))
    return exact_knn(query_points,points,jnp.ones(len(points),dtype=bool),k=k)[0]


def build_local_divfree_stencil_indices(
    x: jnp.ndarray,
    stencil_size: int,
    *,
    center_indices: jnp.ndarray | None = None,
    backend: str = "auto",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build fixed local stencil indices using the repo's shared KNN backend.

    Warp is intentionally confined to this setup function. The returned arrays
    are then consumed by pure JAX fit/evaluate kernels.
    """
    points = _as_points(x)
    active = jnp.arange(points.shape[0], dtype=int) if center_indices is None else jnp.asarray(center_indices, dtype=int).reshape(-1)
    centers = points[active]
    k = min(int(stencil_size), int(points.shape[0]))
    backend_l = str(backend).strip().lower()
    if backend_l == "auto":
        backend_l = "warp" if warp_available() else "cpu"
    if backend_l == "warp":
        try:
            ids, _ = warp_exact_knn(centers, points, k)
            return active, jnp.asarray(ids, dtype=int)
        except WarpUnavailableError:
            if str(backend).strip().lower() != "auto":
                raise
    if backend_l not in {"auto", "cpu", "jax", "warp"}:
        raise ValueError("backend must be 'auto', 'cpu', 'jax', or 'warp'")
    return active, knn_indices_jax(points, centers, k)


@partial(jax.jit, static_argnames=("poly_degree", "phs_degree", "stencil_size"))
def fit_local_divfree(
    x: jnp.ndarray,
    u: jnp.ndarray,
    *,
    poly_degree: int,
    phs_degree: int,
    stencil_size: int,
    center_indices: jnp.ndarray | None = None,
    stencil_indices: jnp.ndarray | None = None,
) -> LocalDivFreeInterpolator:
    x = _as_points(x)
    u = jnp.asarray(u, dtype=float)
    if u.shape != x.shape:
        raise ValueError("x and u must both have shape (n, dim)")
    if center_indices is None:
        active = jnp.arange(x.shape[0], dtype=int)
    else:
        active = jnp.asarray(center_indices, dtype=int).reshape(-1)
    centers = x[active]
    k = min(int(stencil_size), x.shape[0])
    if stencil_indices is None:
        stencils = knn_indices_jax(x, centers, k)
    else:
        stencils = jnp.asarray(stencil_indices, dtype=int)
        if stencils.ndim != 2 or stencils.shape[0] != active.shape[0] or stencils.shape[1] != k:
            raise ValueError("stencil_indices must have shape (num_centers, stencil_size)")

    def fit_one(idx: jnp.ndarray):
        local = fit_divfree_phs(x[idx], u[idx], poly_degree=poly_degree, phs_degree=phs_degree)
        return local.rbf_coeffs, local.poly_coeffs, local.center, local.scale, local.poly_transform

    rbf_coeffs, poly_coeffs, local_centers, local_scales, poly_transforms = vmap(fit_one)(stencils)
    return LocalDivFreeInterpolator(
        nodes=x,
        center_indices=active,
        stencil_indices=stencils,
        rbf_coeffs=rbf_coeffs,
        poly_coeffs=poly_coeffs,
        centers=local_centers,
        scales=local_scales,
        poly_transforms=poly_transforms,
        poly_degree=poly_degree,
        phs_degree=phs_degree,
    )


def fit_local_divfree_with_backend(
    x: jnp.ndarray,
    u: jnp.ndarray,
    *,
    poly_degree: int,
    phs_degree: int,
    stencil_size: int,
    center_indices: jnp.ndarray | None = None,
    backend: str = "auto",
) -> LocalDivFreeInterpolator:
    active, stencils = build_local_divfree_stencil_indices(x, stencil_size, center_indices=center_indices, backend=backend)
    return fit_local_divfree(
        x,
        u,
        poly_degree=poly_degree,
        phs_degree=phs_degree,
        stencil_size=stencil_size,
        center_indices=active,
        stencil_indices=stencils,
    )


@jax.jit
def evaluate_local_divfree(model: LocalDivFreeInterpolator, xq: jnp.ndarray) -> jnp.ndarray:
    xq = _as_points(xq)
    center_points = model.nodes[model.center_indices]
    nearest = jnp.argmin(distance_matrix(xq, center_points), axis=1)

    def eval_one(q: jnp.ndarray, slot: jnp.ndarray) -> jnp.ndarray:
        idx = model.stencil_indices[slot]
        nodes = model.nodes[idx]
        dim = nodes.shape[1]
        q2 = q.reshape(1, dim)
        gram = divfree_gram_matrix(q2, nodes, degree=model.phs_degree)
        xiq = (q2 - model.centers[slot]) / model.scales[slot]
        raw_pmat, _ = divfree_polynomial_stack(xiq, poly_degree=model.poly_degree)
        pmat = (raw_pmat / model.scales[slot]) @ model.poly_transforms[slot]
        stacked = gram @ model.rbf_coeffs[slot] + pmat @ model.poly_coeffs[slot]
        return _unstack_field(stacked, dim)[0]

    return vmap(eval_one)(xq, nearest)
