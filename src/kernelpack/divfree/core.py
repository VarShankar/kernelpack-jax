from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math

import jax
from jax import lax, vmap
import jax.numpy as jnp

from kernelpack.accelerators import WarpUnavailableError, warp_available, warp_exact_knn
from kernelpack.geometry import distance_matrix
from kernelpack import poly
from kernelpack.poly.core import _tensor_evaluate_with_recurrence


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
    rank = _divfree_poly_rank(x.shape[1], int(poly_degree), raw_values.shape[0])
    _u, singular_values, vh = jnp.linalg.svd(raw_values, full_matrices=False)
    kept_singular_values = singular_values[:rank]
    rank_tol = 1e-10 * jnp.maximum(singular_values[0], jnp.finfo(float).eps)
    inv_singular_values = jnp.where(kept_singular_values > rank_tol, 1.0 / kept_singular_values, 0.0)
    transform = vh[:rank].T * inv_singular_values
    values = raw_values @ transform
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
    distances = distance_matrix(query_points, points)
    return jnp.argsort(distances, axis=1)[:, :k]


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
