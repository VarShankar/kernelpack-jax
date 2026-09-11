from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache, partial
from math import ceil
from typing import Callable

from jax import jit, vmap
from jax import scipy as jsp
import jax.numpy as jnp

from kernelpack.domain import DomainDescriptor
from kernelpack.geometry import distance_matrix
from kernelpack.poly import PolynomialBasis, total_degree_indices
from kernelpack.poly.core import _tensor_evaluate_with_recurrence


def _unit_multi_index(dim: int, selectdim: int) -> jnp.ndarray:
    d = jnp.zeros((1, dim), dtype=int)
    return d.at[0, selectdim].set(1)


def _first_derivative_multi_indices(dim: int) -> jnp.ndarray:
    return jnp.eye(dim, dtype=int)


def _second_derivative_multi_indices(dim: int) -> jnp.ndarray:
    return 2 * jnp.eye(dim, dtype=int)


@jit
def _stable_solve(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.solve(a, b)


def _build_legendre_basis(dim: int, ell: int) -> PolynomialBasis:
    return PolynomialBasis.from_total_degree(dim, ell, family="legendre", center=jnp.zeros(dim), scale=1.0)


@lru_cache(maxsize=None)
def _build_legendre_basis_data(dim: int, ell: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    basis = _build_legendre_basis(dim, ell)
    return basis.index_set, basis.recurrence_a, basis.recurrence_b


@jit
def _batched_pairwise_distances(x_stencils: jnp.ndarray) -> jnp.ndarray:
    diff = x_stencils[:, :, None, :] - x_stencils[:, None, :, :]
    return jnp.linalg.norm(diff, axis=3)


@jit
def _batched_basis_eval(
    index_set: jnp.ndarray,
    recurrence_a: jnp.ndarray,
    recurrence_b: jnp.ndarray,
    x_batch: jnp.ndarray,
    d: jnp.ndarray,
) -> jnp.ndarray:
    x_batch = jnp.asarray(x_batch, dtype=float)
    d = jnp.atleast_2d(jnp.asarray(d, dtype=int))
    lead_shape = x_batch.shape[:-1]
    x_flat = x_batch.reshape(-1, x_batch.shape[-1])
    values = _tensor_evaluate_with_recurrence(x_flat, index_set, d, recurrence_a, recurrence_b)
    values = values.reshape(*lead_shape, index_set.shape[0], d.shape[0])
    return values[..., 0] if d.shape[0] == 1 else values


@partial(jit, static_argnames=("spline_degree",))
def _prepare_rbf_system(
    x_stencils: jnp.ndarray,
    index_set: jnp.ndarray,
    recurrence_a: jnp.ndarray,
    recurrence_b: jnp.ndarray,
    spline_degree: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    dim = x_stencils.shape[2]
    r = _batched_pairwise_distances(x_stencils)
    width = jnp.maximum(r.max(axis=(1, 2)), 1.0)
    xm = x_stencils.mean(axis=1)
    xc = (x_stencils - xm[:, None, :]) / width[:, None, None]
    p = _batched_basis_eval(index_set, recurrence_a, recurrence_b, xc, jnp.zeros((1, dim), dtype=int))
    k = RBFStencil.phs_rbf(r, spline_degree)
    n = x_stencils.shape[1]
    npoly = index_set.shape[0]
    lhs = jnp.zeros((x_stencils.shape[0], n + npoly, n + npoly), dtype=float)
    lhs = lhs.at[:, :n, :n].set(k)
    lhs = lhs.at[:, :n, n:].set(p)
    lhs = lhs.at[:, n:, :n].set(jnp.swapaxes(p, 1, 2))
    return lhs, width, xc, r


@partial(jit, static_argnames=("spline_degree",))
def _batched_rbf_lap_weights(
    x_stencils: jnp.ndarray,
    index_set: jnp.ndarray,
    recurrence_a: jnp.ndarray,
    recurrence_b: jnp.ndarray,
    spline_degree: int,
) -> jnp.ndarray:
    dim = x_stencils.shape[2]
    lhs, width, xc, r = _prepare_rbf_system(x_stencils, index_set, recurrence_a, recurrence_b, spline_degree)
    rhs_poly = jnp.sum(
        _batched_basis_eval(
            index_set,
            recurrence_a,
            recurrence_b,
            xc[:, :1, :],
            _second_derivative_multi_indices(dim),
        ),
        axis=3,
    ).transpose(0, 2, 1) / (width[:, None, None] ** 2)
    rhs = jnp.concatenate(
        [jnp.swapaxes(RBFStencil.phs_lap(r[:, :1, :], spline_degree, dim), 1, 2), rhs_poly],
        axis=1,
    )
    return jnp.linalg.solve(lhs, rhs)[:, : x_stencils.shape[1], :]


@partial(jit, static_argnames=("spline_degree",))
def _batched_rbf_interp_weights(
    x_stencils: jnp.ndarray,
    index_set: jnp.ndarray,
    recurrence_a: jnp.ndarray,
    recurrence_b: jnp.ndarray,
    spline_degree: int,
) -> jnp.ndarray:
    dim = x_stencils.shape[2]
    lhs, _width, xc, r = _prepare_rbf_system(x_stencils, index_set, recurrence_a, recurrence_b, spline_degree)
    rhs_poly = _batched_basis_eval(
        index_set,
        recurrence_a,
        recurrence_b,
        xc[:, :1, :],
        jnp.zeros((1, dim), dtype=int),
    ).transpose(0, 2, 1)
    rhs = jnp.concatenate([jnp.swapaxes(RBFStencil.phs_rbf(r[:, :1, :], spline_degree), 1, 2), rhs_poly], axis=1)
    return jnp.linalg.solve(lhs, rhs)[:, : x_stencils.shape[1], :]


@partial(jit, static_argnames=("spline_degree", "selectdim"))
def _batched_rbf_grad_weights(
    x_stencils: jnp.ndarray,
    index_set: jnp.ndarray,
    recurrence_a: jnp.ndarray,
    recurrence_b: jnp.ndarray,
    spline_degree: int,
    selectdim: int,
) -> jnp.ndarray:
    dim = x_stencils.shape[2]
    lhs, width, xc, r = _prepare_rbf_system(x_stencils, index_set, recurrence_a, recurrence_b, spline_degree)
    diff = x_stencils[:, :1, selectdim] - x_stencils[:, :, selectdim]
    rhs_poly = _batched_basis_eval(
        index_set,
        recurrence_a,
        recurrence_b,
        xc[:, :1, :],
        _unit_multi_index(dim, selectdim),
    ).transpose(0, 2, 1) / width[:, None, None]
    rhs_rbf = (diff[:, None, :] * RBFStencil.phs_dr_over_r(r[:, :1, :], spline_degree)).swapaxes(1, 2)
    rhs = jnp.concatenate([rhs_rbf, rhs_poly], axis=1)
    return jnp.linalg.solve(lhs, rhs)[:, : x_stencils.shape[1], :]


@partial(jit, static_argnames=("spline_degree", "selectdim"))
def _batched_rbf_grad_weights_at_points(
    x_stencils: jnp.ndarray,
    xe: jnp.ndarray,
    index_set: jnp.ndarray,
    recurrence_a: jnp.ndarray,
    recurrence_b: jnp.ndarray,
    spline_degree: int,
    selectdim: int,
) -> jnp.ndarray:
    dim = x_stencils.shape[2]
    lhs, width, _xc, _r = _prepare_rbf_system(x_stencils, index_set, recurrence_a, recurrence_b, spline_degree)
    xm = x_stencils.mean(axis=1)
    xec = (xe - xm) / width[:, None]
    r_rhs = jnp.linalg.norm(xe[:, None, :] - x_stencils, axis=2)
    diff = xe[:, None, selectdim] - x_stencils[:, :, selectdim]
    rhs_poly = _batched_basis_eval(
        index_set,
        recurrence_a,
        recurrence_b,
        xec[:, None, :],
        _unit_multi_index(dim, selectdim),
    ).transpose(0, 2, 1) / width[:, None, None]
    rhs = jnp.concatenate(
        [(diff * RBFStencil.phs_dr_over_r(r_rhs, spline_degree))[:, :, None], rhs_poly],
        axis=1,
    )
    return jnp.linalg.solve(lhs, rhs)[:, : x_stencils.shape[1], :]


@partial(jit, static_argnames=("spline_degree",))
def _batched_rbf_interp_weights_at_points(
    x_stencils: jnp.ndarray,
    xe: jnp.ndarray,
    index_set: jnp.ndarray,
    recurrence_a: jnp.ndarray,
    recurrence_b: jnp.ndarray,
    spline_degree: int,
) -> jnp.ndarray:
    dim = x_stencils.shape[2]
    lhs, width, _xc, _r = _prepare_rbf_system(x_stencils, index_set, recurrence_a, recurrence_b, spline_degree)
    xm = x_stencils.mean(axis=1)
    xec = (xe - xm) / width[:, None]
    r_rhs = jnp.linalg.norm(xe[:, None, :] - x_stencils, axis=2)
    rhs_poly = _batched_basis_eval(
        index_set,
        recurrence_a,
        recurrence_b,
        xec[:, None, :],
        jnp.zeros((1, dim), dtype=int),
    ).transpose(0, 2, 1)
    rhs = jnp.concatenate([RBFStencil.phs_rbf(r_rhs, spline_degree)[:, :, None], rhs_poly], axis=1)
    return jnp.linalg.solve(lhs, rhs)[:, : x_stencils.shape[1], :]


@partial(jit, static_argnames=("spline_degree",))
def _factor_batched_rbf_interpolants(
    x_stencils: jnp.ndarray,
    index_set: jnp.ndarray,
    recurrence_a: jnp.ndarray,
    recurrence_b: jnp.ndarray,
    spline_degree: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Factor augmented local interpolation systems once per owner stencil."""

    lhs, width, _xc, _r = _prepare_rbf_system(
        x_stencils,
        index_set,
        recurrence_a,
        recurrence_b,
        spline_degree,
    )
    lu, pivots = jsp.linalg.lu_factor(lhs)
    return lu, pivots, width, x_stencils.mean(axis=1)


@partial(jit, static_argnames=("spline_degree", "stencil_size"))
def _evaluate_factored_rbf_interpolants(
    lu: jnp.ndarray,
    pivots: jnp.ndarray,
    width: jnp.ndarray,
    centers: jnp.ndarray,
    stencil_points: jnp.ndarray,
    query_points: jnp.ndarray,
    owner_indices: jnp.ndarray,
    index_set: jnp.ndarray,
    recurrence_a: jnp.ndarray,
    recurrence_b: jnp.ndarray,
    spline_degree: int,
    stencil_size: int,
) -> jnp.ndarray:
    """Evaluate local cardinal weights using cached pivoted LU factors."""

    owner_lu = lu[owner_indices]
    owner_pivots = pivots[owner_indices]
    owner_width = width[owner_indices]
    owner_centers = centers[owner_indices]
    xec = (query_points - owner_centers) / owner_width[:, None]
    r_rhs = jnp.linalg.norm(query_points[:, None, :] - stencil_points, axis=2)
    rhs_poly = _batched_basis_eval(
        index_set,
        recurrence_a,
        recurrence_b,
        xec[:, None, :],
        jnp.zeros((1, query_points.shape[1]), dtype=int),
    ).transpose(0, 2, 1)
    rhs = jnp.concatenate(
        [RBFStencil.phs_rbf(r_rhs, spline_degree)[:, :, None], rhs_poly],
        axis=1,
    )
    solution = jsp.linalg.lu_solve((owner_lu, owner_pivots), rhs)
    return solution[:, :stencil_size, 0]


@partial(jit, static_argnames=("spline_degree",))
def _batched_rbf_bc_weights(
    x_stencils: jnp.ndarray,
    index_set: jnp.ndarray,
    recurrence_a: jnp.ndarray,
    recurrence_b: jnp.ndarray,
    spline_degree: int,
    normals: jnp.ndarray,
    neu_coeff: jnp.ndarray,
    dir_coeff: jnp.ndarray,
) -> jnp.ndarray:
    dim = x_stencils.shape[2]
    lhs, width, xc, r = _prepare_rbf_system(x_stencils, index_set, recurrence_a, recurrence_b, spline_degree)
    grad_poly = _batched_basis_eval(
        index_set,
        recurrence_a,
        recurrence_b,
        xc[:, :1, :],
        _first_derivative_multi_indices(dim),
    ) / width[:, None, None, None]
    diff = x_stencils[:, :1, None, :] - x_stencils[:, None, :, :]
    grad_rbf = RBFStencil.phs_dr_over_r(r[:, :1, :], spline_degree)[..., None] * diff
    total_rbf = neu_coeff[:, None, None] * jnp.sum(jnp.swapaxes(grad_rbf, 1, 2) * normals[:, None, None, :], axis=3)
    total_poly = neu_coeff[:, None, None] * jnp.sum(grad_poly[:, 0, :, :] * normals[:, None, :], axis=2)[:, :, None]
    total_rbf = total_rbf + dir_coeff[:, None, None] * jnp.swapaxes(RBFStencil.phs_rbf(r[:, :1, :], spline_degree), 1, 2)
    total_poly = total_poly + dir_coeff[:, None, None] * _batched_basis_eval(
        index_set,
        recurrence_a,
        recurrence_b,
        xc[:, :1, :],
        jnp.zeros((1, dim), dtype=int),
    ).transpose(0, 2, 1)
    rhs = jnp.concatenate([total_rbf, total_poly], axis=1)
    return jnp.linalg.solve(lhs, rhs)[:, : x_stencils.shape[1], :]


@jit
def _prepare_wls_system(
    x_stencils: jnp.ndarray,
    index_set: jnp.ndarray,
    recurrence_a: jnp.ndarray,
    recurrence_b: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    dim = x_stencils.shape[2]
    xm = x_stencils[:, :1, :]
    r2 = jnp.sum((x_stencils - xm) ** 2, axis=2)
    width = jnp.maximum(jnp.sqrt(r2.max(axis=1)), 1.0)
    xc = (x_stencils - xm) / width[:, None, None]
    p = _batched_basis_eval(index_set, recurrence_a, recurrence_b, xc, jnp.zeros((1, dim), dtype=int))
    node_weights = jnp.clip(jnp.exp(-4.0 * r2 / (width[:, None] ** 2)), 1e-10, 1.0)
    sqrtw = jnp.sqrt(node_weights)
    reconstructor = jnp.linalg.pinv(p * sqrtw[:, :, None]) @ vmap(jnp.diag)(sqrtw)
    return reconstructor, width, xc


@jit
def _batched_wls_lap_weights(
    x_stencils: jnp.ndarray,
    index_set: jnp.ndarray,
    recurrence_a: jnp.ndarray,
    recurrence_b: jnp.ndarray,
) -> jnp.ndarray:
    dim = x_stencils.shape[2]
    reconstructor, width, xc = _prepare_wls_system(x_stencils, index_set, recurrence_a, recurrence_b)
    rhs = jnp.sum(
        _batched_basis_eval(
            index_set,
            recurrence_a,
            recurrence_b,
            xc[:, :1, :],
            _second_derivative_multi_indices(dim),
        ),
        axis=3,
    ).transpose(0, 2, 1) / (width[:, None, None] ** 2)
    return jnp.swapaxes(reconstructor, 1, 2) @ rhs


@jit
def _batched_wls_interp_weights(
    x_stencils: jnp.ndarray,
    index_set: jnp.ndarray,
    recurrence_a: jnp.ndarray,
    recurrence_b: jnp.ndarray,
) -> jnp.ndarray:
    dim = x_stencils.shape[2]
    reconstructor, _width, xc = _prepare_wls_system(x_stencils, index_set, recurrence_a, recurrence_b)
    rhs = _batched_basis_eval(
        index_set,
        recurrence_a,
        recurrence_b,
        xc[:, :1, :],
        jnp.zeros((1, dim), dtype=int),
    ).transpose(0, 2, 1)
    return jnp.swapaxes(reconstructor, 1, 2) @ rhs


@partial(jit, static_argnames=("selectdim",))
def _batched_wls_grad_weights(
    x_stencils: jnp.ndarray,
    index_set: jnp.ndarray,
    recurrence_a: jnp.ndarray,
    recurrence_b: jnp.ndarray,
    selectdim: int,
) -> jnp.ndarray:
    dim = x_stencils.shape[2]
    reconstructor, width, xc = _prepare_wls_system(x_stencils, index_set, recurrence_a, recurrence_b)
    rhs = _batched_basis_eval(
        index_set,
        recurrence_a,
        recurrence_b,
        xc[:, :1, :],
        _unit_multi_index(dim, selectdim),
    ).transpose(0, 2, 1) / width[:, None, None]
    return jnp.swapaxes(reconstructor, 1, 2) @ rhs


@partial(jit, static_argnames=("selectdim",))
def _batched_wls_grad_weights_at_points(
    x_stencils: jnp.ndarray,
    xe: jnp.ndarray,
    index_set: jnp.ndarray,
    recurrence_a: jnp.ndarray,
    recurrence_b: jnp.ndarray,
    selectdim: int,
) -> jnp.ndarray:
    reconstructor, width, _xc = _prepare_wls_system(x_stencils, index_set, recurrence_a, recurrence_b)
    xm = x_stencils[:, :1, :]
    xec = (xe[:, None, :] - xm) / width[:, None, None]
    rhs = _batched_basis_eval(
        index_set,
        recurrence_a,
        recurrence_b,
        xec,
        _unit_multi_index(x_stencils.shape[2], selectdim),
    ).transpose(0, 2, 1) / width[:, None, None]
    return jnp.swapaxes(reconstructor, 1, 2) @ rhs


@jit
def _batched_wls_interp_weights_at_points(
    x_stencils: jnp.ndarray,
    xe: jnp.ndarray,
    index_set: jnp.ndarray,
    recurrence_a: jnp.ndarray,
    recurrence_b: jnp.ndarray,
) -> jnp.ndarray:
    reconstructor, _width, _xc = _prepare_wls_system(x_stencils, index_set, recurrence_a, recurrence_b)
    xm = x_stencils[:, :1, :]
    xec = (xe[:, None, :] - xm) / _width[:, None, None]
    rhs = _batched_basis_eval(
        index_set,
        recurrence_a,
        recurrence_b,
        xec,
        jnp.zeros((1, x_stencils.shape[2]), dtype=int),
    ).transpose(0, 2, 1)
    return jnp.swapaxes(reconstructor, 1, 2) @ rhs


@jit
def _batched_wls_bc_weights(
    x_stencils: jnp.ndarray,
    index_set: jnp.ndarray,
    recurrence_a: jnp.ndarray,
    recurrence_b: jnp.ndarray,
    normals: jnp.ndarray,
    neu_coeff: jnp.ndarray,
    dir_coeff: jnp.ndarray,
) -> jnp.ndarray:
    dim = x_stencils.shape[2]
    reconstructor, width, xc = _prepare_wls_system(x_stencils, index_set, recurrence_a, recurrence_b)
    grad = _batched_basis_eval(
        index_set,
        recurrence_a,
        recurrence_b,
        xc[:, :1, :],
        _first_derivative_multi_indices(dim),
    ) / width[:, None, None, None]
    total = neu_coeff[:, None, None] * jnp.sum(grad[:, 0, :, :] * normals[:, None, :], axis=2)[:, :, None]
    total = total + dir_coeff[:, None, None] * _batched_basis_eval(
        index_set,
        recurrence_a,
        recurrence_b,
        xc[:, :1, :],
        jnp.zeros((1, dim), dtype=int),
    ).transpose(0, 2, 1)
    return jnp.swapaxes(reconstructor, 1, 2) @ total


@jit
def _assemble_operator_entries(
    active_row_ids: jnp.ndarray,
    stencil_globals: jnp.ndarray,
    knn_indices: jnp.ndarray,
    batched_weights: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    rows = jnp.broadcast_to((active_row_ids - 1)[:, None], knn_indices.shape)
    cols = stencil_globals[knn_indices - 1] - 1
    flat_rows = rows.reshape(-1)
    flat_cols = cols.reshape(-1)
    flat_vals = batched_weights[:, :, 0].reshape(-1)
    return flat_rows, flat_cols, flat_vals, jnp.column_stack([flat_rows + 1, flat_cols + 1])


@partial(jit, static_argnames=("n1", "n2"))
def _scatter_operator_matrix(flat_rows: jnp.ndarray, flat_cols: jnp.ndarray, flat_vals: jnp.ndarray, *, n1: int, n2: int) -> jnp.ndarray:
    return jnp.zeros((n1, n2), dtype=flat_vals.dtype).at[flat_rows, flat_cols].add(flat_vals)


def _canonical_op_name(op_name: str) -> str:
    name = str(op_name).lower()
    aliases = {
        "lap": "lap",
        "laplacian": "lap",
        "interp": "interp",
        "interpolation": "interp",
        "grad": "grad",
        "gradient": "grad",
        "bc": "bc",
        "boundary": "bc",
    }
    if name not in aliases:
        raise ValueError(f"unsupported batched operator {op_name}")
    return aliases[name]


def _batched_rbf_weights(
    x_stencils: jnp.ndarray,
    sp: StencilProperties,
    op_name: str,
    op: OpProperties,
    normals: jnp.ndarray | None = None,
    neu_coeff: jnp.ndarray | None = None,
    dir_coeff: jnp.ndarray | None = None,
) -> jnp.ndarray:
    dim = x_stencils.shape[2]
    index_set, recurrence_a, recurrence_b = _build_legendre_basis_data(dim, sp.ell)
    name = _canonical_op_name(op_name)
    if name == "lap":
        return _batched_rbf_lap_weights(x_stencils, index_set, recurrence_a, recurrence_b, sp.spline_degree)
    if name == "interp":
        return _batched_rbf_interp_weights(x_stencils, index_set, recurrence_a, recurrence_b, sp.spline_degree)
    if name == "grad":
        return _batched_rbf_grad_weights(x_stencils, index_set, recurrence_a, recurrence_b, sp.spline_degree, op.selectdim)
    batch_size = x_stencils.shape[0]
    return _batched_rbf_bc_weights(
        x_stencils,
        index_set,
        recurrence_a,
        recurrence_b,
        sp.spline_degree,
        normals if normals is not None else jnp.zeros((batch_size, dim), dtype=float),
        neu_coeff if neu_coeff is not None else jnp.zeros((batch_size,), dtype=float),
        dir_coeff if dir_coeff is not None else jnp.zeros((batch_size,), dtype=float),
    )


def _batched_wls_weights(
    x_stencils: jnp.ndarray,
    sp: StencilProperties,
    op_name: str,
    op: OpProperties,
    normals: jnp.ndarray | None = None,
    neu_coeff: jnp.ndarray | None = None,
    dir_coeff: jnp.ndarray | None = None,
) -> jnp.ndarray:
    dim = x_stencils.shape[2]
    index_set, recurrence_a, recurrence_b = _build_legendre_basis_data(dim, sp.ell)
    name = _canonical_op_name(op_name)
    if name == "lap":
        return _batched_wls_lap_weights(x_stencils, index_set, recurrence_a, recurrence_b)
    if name == "interp":
        return _batched_wls_interp_weights(x_stencils, index_set, recurrence_a, recurrence_b)
    if name == "grad":
        return _batched_wls_grad_weights(x_stencils, index_set, recurrence_a, recurrence_b, op.selectdim)
    batch_size = x_stencils.shape[0]
    return _batched_wls_bc_weights(
        x_stencils,
        index_set,
        recurrence_a,
        recurrence_b,
        normals if normals is not None else jnp.zeros((batch_size, dim), dtype=float),
        neu_coeff if neu_coeff is not None else jnp.zeros((batch_size,), dtype=float),
        dir_coeff if dir_coeff is not None else jnp.zeros((batch_size,), dtype=float),
    )


def _batched_weights_at_points(
    approx: object,
    x_stencils: jnp.ndarray,
    xe: jnp.ndarray,
    sp: StencilProperties,
    op_name: str,
    op: OpProperties,
) -> jnp.ndarray | None:
    dim = x_stencils.shape[2]
    index_set, recurrence_a, recurrence_b = _build_legendre_basis_data(dim, sp.ell)
    name = _canonical_op_name(op_name)
    if isinstance(approx, RBFStencil):
        if name == "grad":
            return _batched_rbf_grad_weights_at_points(
                x_stencils,
                xe,
                index_set,
                recurrence_a,
                recurrence_b,
                sp.spline_degree,
                op.selectdim,
            )
        if name == "interp":
            return _batched_rbf_interp_weights_at_points(
                x_stencils,
                xe,
                index_set,
                recurrence_a,
                recurrence_b,
                sp.spline_degree,
            )
    if isinstance(approx, WeightedLeastSquaresStencil):
        if name == "grad":
            return _batched_wls_grad_weights_at_points(x_stencils, xe, index_set, recurrence_a, recurrence_b, op.selectdim)
        if name == "interp":
            return _batched_wls_interp_weights_at_points(x_stencils, xe, index_set, recurrence_a, recurrence_b)
    return None


@dataclass
class StencilProperties:
    n: int = 0
    dim: int = 0
    ell: int = 0
    spline_degree: int = 3
    npoly: int = 0
    width: float = 1.0
    tree_mode: str = "all"
    point_set: str = "interior_boundary"

    def __post_init__(self) -> None:
        self.tree_mode = self.normalize_tree_mode(self.tree_mode)
        self.point_set = self.normalize_point_set(self.point_set)
        if self.npoly == 0 and self.dim > 0:
            self.npoly = total_degree_indices(self.dim, self.ell).shape[0]

    @classmethod
    def from_accuracy(
        cls,
        *,
        operator: str = "lap",
        convergence_order: int | None = None,
        diff_op_order: int | None = None,
        dimension: int,
        approximation: str = "rbf",
        stencil_factor: float | None = None,
        spline_degree: int | None = None,
        tree_mode: str = "all",
        point_set: str = "interior_boundary",
    ) -> "StencilProperties":
        q = cls.default_diff_order(operator) if diff_op_order is None else diff_op_order
        p = 2 if convergence_order is None else convergence_order
        ell = max(p + q - 1, 0)
        npoly = total_degree_indices(dimension, ell).shape[0]
        approx = approximation.lower()
        if stencil_factor is None:
            if approx in {"rbf", "rbf-fd", "rbffd"}:
                stencil_factor = 2.0
            elif approx in {"wls", "weighted_least_squares", "weightedleastsquares"}:
                stencil_factor = 1.5
            else:
                raise ValueError(f"unknown approximation {approximation}")
        if spline_degree is None:
            spline_degree = max(2 * q + 1, 3)
        if spline_degree % 2 == 0:
            spline_degree -= 1
        n = max(npoly + 1, ceil(stencil_factor * npoly))
        return cls(n=n, dim=dimension, ell=ell, spline_degree=spline_degree, npoly=npoly, tree_mode=tree_mode, point_set=point_set)

    @staticmethod
    def normalize_tree_mode(mode: str | int) -> str:
        if isinstance(mode, (int, jnp.integer)):
            return ["all", "interior_boundary", "boundary"][int(mode)]
        mode = str(mode).lower()
        aliases = {
            "all": "all",
            "all_nodes": "all",
            "full": "all",
            "interior_boundary": "interior_boundary",
            "interior+boundary": "interior_boundary",
            "int_bdry": "interior_boundary",
            "intboundary": "interior_boundary",
            "owned": "interior_boundary",
            "boundary": "boundary",
            "bdry": "boundary",
            "boundary_only": "boundary",
        }
        if mode not in aliases:
            raise ValueError(f"unknown tree mode {mode}")
        return aliases[mode]

    @staticmethod
    def normalize_point_set(mode: str | int) -> str:
        return StencilProperties.normalize_tree_mode(mode)

    @staticmethod
    def default_diff_order(op: str) -> int:
        op = str(op).lower()
        if op in {"interp", "interpolation", "identity"}:
            return 0
        if op in {"grad", "gradient", "dx", "dy", "dz"}:
            return 1
        if op in {"lap", "laplacian", "bc", "boundary"}:
            return 2
        raise ValueError(f"unknown operator {op}")


@dataclass
class OpProperties:
    selectdim: int = 0
    decompose: bool = True
    store_weights: bool = True
    record_stencils: bool = False
    nosolve: bool = False
    overlap_load: float = 0.5
    use_parallel: bool = False


@dataclass(frozen=True)
class FrozenStencilGraph:
    tree_mode: str
    point_set: str
    stencil_size: int
    active_rows: jnp.ndarray
    knn_indices: jnp.ndarray


@dataclass
class RBFStencil:
    a: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    solve_lhs: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    x_stencil: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    xc: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    xm: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    width: float = 1.0
    s_dim: int = 0
    n: int = 0
    ell: int = 0
    npoly: int = 0
    basis: PolynomialBasis | None = None
    wt: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    l: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    bc: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    gx: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))

    def initialize_geometry(self, x: jnp.ndarray, sp: StencilProperties) -> None:
        self.s_dim = x.shape[1]
        self.n = x.shape[0]
        self.x_stencil = x
        self.ell = sp.ell
        self.npoly = sp.npoly
        r = distance_matrix(x, x)
        self.width = max(float(r.max(initial=0.0)), 1.0)
        self.xm = x.mean(axis=0)
        self.xc = (x - self.xm) / self.width
        self.basis = PolynomialBasis.from_total_degree(self.s_dim, self.ell, family="legendre", center=jnp.zeros(self.s_dim), scale=1.0)
        p = self.basis.evaluate(self.xc, jnp.zeros((1, self.s_dim), dtype=int), True)
        self.a = jnp.zeros((self.n + self.npoly, self.n + self.npoly))
        self.a = self.a.at[: self.n, : self.n].set(self.phs_rbf(r, sp.spline_degree))
        self.a = self.a.at[: self.n, self.n :].set(p)
        self.a = self.a.at[self.n :, : self.n].set(p.T)
        self.solve_lhs = self.a

    def compute_weights(self, x: jnp.ndarray, *args: object) -> jnp.ndarray:
        if len(args) >= 7 and isinstance(args[0], jnp.ndarray) and args[0].shape[1] == x.shape[1]:
            nr, neu_coeff, dir_coeff, sp, op, apply_op, rhs_indices = args[:7]
            return self._compute_weights_boundary(x, nr, float(neu_coeff), float(dir_coeff), sp, op, apply_op, rhs_indices)
        sp, op, apply_op, rhs_indices = args[:4]
        return self._compute_weights_interior(x, sp, op, apply_op, rhs_indices)

    def eval_weights(self, sp: StencilProperties, xe: jnp.ndarray) -> jnp.ndarray:
        xe = jnp.asarray(xe, dtype=float)
        if xe.size == 0:
            return jnp.zeros((0, self.n))
        re = distance_matrix(xe, self.x_stencil)
        xec = (xe - self.xm) / self.width
        pe = self.basis.evaluate(xec, jnp.zeros((1, self.s_dim), dtype=int), True)
        rt = jnp.vstack([self.phs_rbf(re, sp.spline_degree).T, pe.T])
        lagrange = self.stable_solve(self.solve_lhs, rt)
        return lagrange[: self.n].T

    def get_interp_mat(self) -> jnp.ndarray:
        return self.a[: self.n, : self.n]

    def lap_op(self, sp: StencilProperties, _op: OpProperties, r_rhs: jnp.ndarray, _x_subset: jnp.ndarray, _x: jnp.ndarray, x_at_origin_subset: jnp.ndarray, _x_at_origin: jnp.ndarray) -> jnp.ndarray:
        d2 = _second_derivative_multi_indices(self.s_dim)
        bpoly = jnp.sum(self.basis.evaluate(x_at_origin_subset, d2, True), axis=2).T / (self.width**2)
        return jnp.vstack([self.phs_lap(r_rhs, sp.spline_degree, self.s_dim).T, bpoly])

    def grad_op(self, sp: StencilProperties, op: OpProperties, r_rhs: jnp.ndarray, x_subset: jnp.ndarray, x: jnp.ndarray, x_at_origin_subset: jnp.ndarray, _x_at_origin: jnp.ndarray) -> jnp.ndarray:
        dim = op.selectdim
        diff = x_subset[:, dim : dim + 1] - x[None, :, dim]
        bpoly = self.basis.evaluate(x_at_origin_subset, _unit_multi_index(self.s_dim, dim), True).T / self.width
        return jnp.vstack([(diff * self.phs_dr_over_r(r_rhs, sp.spline_degree)).T, bpoly])

    def bc_op(self, sp: StencilProperties, op: OpProperties, neu_coeff: float, dir_coeff: float, r_rhs: jnp.ndarray, x_subset: jnp.ndarray, x: jnp.ndarray, x_at_origin_subset: jnp.ndarray, x_at_origin: jnp.ndarray, nr_subset: jnp.ndarray) -> jnp.ndarray:
        total = jnp.zeros((self.n + self.npoly, x_at_origin_subset.shape[0]))
        if neu_coeff != 0:
            radial = self.phs_dr_over_r(r_rhs, sp.spline_degree)[..., None]
            diff = x_subset[:, None, :] - x[None, :, :]
            grad_rbf = (radial * diff) * nr_subset[:, None, :]
            grad_rbf = neu_coeff * jnp.sum(grad_rbf, axis=2).T
            grad_poly = self.basis.evaluate(x_at_origin_subset, _first_derivative_multi_indices(self.s_dim), True) / self.width
            grad_poly = neu_coeff * jnp.sum(grad_poly * nr_subset[:, None, :], axis=2).T
            total = total + jnp.vstack([grad_rbf, grad_poly])
        if dir_coeff != 0:
            total = total + dir_coeff * self.interp_op(sp, op, r_rhs, x_subset, x, x_at_origin_subset, x_at_origin)
        if dir_coeff == 0 and neu_coeff == 0:
            raise ValueError("both boundary coefficients cannot be zero")
        return total

    def interp_op(self, sp: StencilProperties, _op: OpProperties, r_rhs: jnp.ndarray, _x_subset: jnp.ndarray, _x: jnp.ndarray, x_at_origin_subset: jnp.ndarray, _x_at_origin: jnp.ndarray) -> jnp.ndarray:
        bpoly = self.basis.evaluate(x_at_origin_subset, jnp.zeros((1, self.s_dim), dtype=int), True).T
        return jnp.vstack([self.phs_rbf(r_rhs, sp.spline_degree).T, bpoly])

    def _compute_weights_interior(self, x: jnp.ndarray, sp: StencilProperties, op: OpProperties, apply_op: str | Callable[..., jnp.ndarray], rhs_indices: int | jnp.ndarray) -> jnp.ndarray:
        self.initialize_geometry(x, sp)
        rhs_inds = jnp.atleast_1d(rhs_indices).astype(int) - 1
        x_subset = x[rhs_inds]
        x_at_origin_subset = self.xc[rhs_inds]
        r = distance_matrix(x, x)
        r_rhs = r[rhs_inds]
        b = self._apply_operator(apply_op, sp, op, r_rhs, x_subset, x, x_at_origin_subset, self.xc)
        w = b if op.nosolve else self.stable_solve(self.solve_lhs, b)[: self.n]
        if rhs_inds.size == 1 and int(rhs_inds[0]) == 0:
            name = str(apply_op).lower()
            if name in {"lap", "laplacian"}:
                self.l = w
            elif name in {"interp", "interpolation"}:
                self.wt = w
            elif name in {"grad", "gradient"}:
                self.gx = w
        return w

    def _compute_weights_boundary(self, x: jnp.ndarray, nr: jnp.ndarray, neu_coeff: float, dir_coeff: float, sp: StencilProperties, op: OpProperties, apply_op: str | Callable[..., jnp.ndarray], rhs_indices: int | jnp.ndarray) -> jnp.ndarray:
        self.initialize_geometry(x, sp)
        rhs_inds = jnp.atleast_1d(rhs_indices).astype(int) - 1
        x_subset = x[rhs_inds]
        x_at_origin_subset = self.xc[rhs_inds]
        nr_subset = nr[rhs_inds]
        r = distance_matrix(x, x)
        r_rhs = r[rhs_inds]
        b = self._apply_boundary_operator(apply_op, sp, op, neu_coeff, dir_coeff, r_rhs, x_subset, x, x_at_origin_subset, self.xc, nr_subset)
        w = self.stable_solve(self.solve_lhs, b)[: self.n]
        if rhs_inds.size == 1 and int(rhs_inds[0]) == 0:
            self.bc = w
        return w

    def _apply_operator(self, apply_op: str | Callable[..., jnp.ndarray], sp: StencilProperties, op: OpProperties, r_rhs: jnp.ndarray, x_subset: jnp.ndarray, x: jnp.ndarray, x_at_origin_subset: jnp.ndarray, x_at_origin: jnp.ndarray) -> jnp.ndarray:
        if callable(apply_op):
            return apply_op(self, sp, op, r_rhs, x_subset, x, x_at_origin_subset, x_at_origin)
        name = str(apply_op).lower()
        if name in {"lap", "laplacian"}:
            return self.lap_op(sp, op, r_rhs, x_subset, x, x_at_origin_subset, x_at_origin)
        if name in {"grad", "gradient"}:
            return self.grad_op(sp, op, r_rhs, x_subset, x, x_at_origin_subset, x_at_origin)
        if name in {"interp", "interpolation"}:
            return self.interp_op(sp, op, r_rhs, x_subset, x, x_at_origin_subset, x_at_origin)
        raise ValueError(f"unknown operator {apply_op}")

    def _apply_boundary_operator(self, apply_op: str | Callable[..., jnp.ndarray], sp: StencilProperties, op: OpProperties, neu_coeff: float, dir_coeff: float, r_rhs: jnp.ndarray, x_subset: jnp.ndarray, x: jnp.ndarray, x_at_origin_subset: jnp.ndarray, x_at_origin: jnp.ndarray, nr_subset: jnp.ndarray) -> jnp.ndarray:
        if callable(apply_op):
            return apply_op(self, sp, op, neu_coeff, dir_coeff, r_rhs, x_subset, x, x_at_origin_subset, x_at_origin, nr_subset)
        name = str(apply_op).lower()
        if name in {"bc", "boundary"}:
            return self.bc_op(sp, op, neu_coeff, dir_coeff, r_rhs, x_subset, x, x_at_origin_subset, x_at_origin, nr_subset)
        raise ValueError(f"unknown boundary operator {apply_op}")

    @staticmethod
    def phs_rbf(r: jnp.ndarray, degree: int) -> jnp.ndarray:
        return jnp.where((degree % 2 == 0) & (r > 0), r**degree * jnp.log(r + 2e-16), r**degree)

    @staticmethod
    def phs_dr_over_r(r: jnp.ndarray, degree: int) -> jnp.ndarray:
        if degree % 2 == 0:
            d = r ** (degree - 2) * (degree * jnp.log(r + 2e-16) + 1)
        else:
            d = degree * r ** (degree - 2)
        return jnp.where(jnp.isfinite(d), d, 0.0)

    @staticmethod
    def phs_lap(r: jnp.ndarray, degree: int, dim: int) -> jnp.ndarray:
        if degree % 2 == 0:
            logt = jnp.log(r + 2e-16)
            l = r ** (degree - 2) * (dim + 2 * degree + degree**2 * logt - 2 * degree * logt + dim * degree * logt - 2)
        else:
            l = degree * (dim + degree - 2) * r ** (degree - 2)
        return jnp.where(jnp.isfinite(l), l, 0.0)

    @staticmethod
    def stable_solve(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return _stable_solve(a, b)


@dataclass
class WeightedLeastSquaresStencil:
    x_stencil: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    xc: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    xm: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    width: float = 1.0
    s_dim: int = 0
    n: int = 0
    fit_ell: int = 0
    fit_npoly: int = 0
    node_weights: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    interp_metric: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    reconstructor: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    basis: PolynomialBasis | None = None
    wt: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    l: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    bc: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))

    def initialize_geometry(self, x: jnp.ndarray, sp: StencilProperties) -> None:
        self.s_dim = x.shape[1]
        self.n = x.shape[0]
        self.fit_ell = sp.ell
        self.fit_npoly = sp.npoly
        self.x_stencil = x
        self.xm = x[0]
        r2 = jnp.sum((x - self.xm) ** 2, axis=1)
        self.width = max(float(jnp.sqrt(r2.max(initial=0.0))), 1.0)
        self.xc = (x - self.xm) / self.width
        self.basis = PolynomialBasis.from_total_degree(self.s_dim, self.fit_ell, family="legendre", center=jnp.zeros(self.s_dim), scale=1.0)
        p = self.basis.evaluate(self.xc, jnp.zeros((1, self.s_dim), dtype=int), True)
        self.node_weights = jnp.clip(jnp.exp(-4.0 * r2 / (self.width**2)), 1e-10, 1.0)
        sqrtw = jnp.sqrt(self.node_weights)
        weighted_p = p * sqrtw[:, None]
        weighted_identity = jnp.diag(sqrtw)
        gram = weighted_p.T @ weighted_p
        self.reconstructor = jnp.linalg.pinv(weighted_p) @ weighted_identity
        self.interp_metric = gram

    def compute_weights(self, x: jnp.ndarray, *args: object) -> jnp.ndarray:
        if len(args) >= 7 and isinstance(args[0], jnp.ndarray) and args[0].shape[1] == x.shape[1]:
            nr, neu_coeff, dir_coeff, sp, op, apply_op, rhs_indices = args[:7]
            return self._compute_weights_boundary(x, nr, float(neu_coeff), float(dir_coeff), sp, op, apply_op, rhs_indices)
        sp, op, apply_op, rhs_indices = args[:4]
        return self._compute_weights_interior(x, sp, op, apply_op, rhs_indices)

    def lap_op(self, _sp: StencilProperties, _op: OpProperties, _r_rhs: jnp.ndarray | None, _x_subset: jnp.ndarray, _x: jnp.ndarray, x_at_origin_subset: jnp.ndarray, _x_at_origin: jnp.ndarray) -> jnp.ndarray:
        total = jnp.sum(self.basis.evaluate(x_at_origin_subset, _second_derivative_multi_indices(self.s_dim), True), axis=2) / (self.width**2)
        return total.T

    def grad_op(self, _sp: StencilProperties, op: OpProperties, _r_rhs: jnp.ndarray | None, _x_subset: jnp.ndarray, _x: jnp.ndarray, x_at_origin_subset: jnp.ndarray, _x_at_origin: jnp.ndarray) -> jnp.ndarray:
        return self.basis.evaluate(x_at_origin_subset, _unit_multi_index(self.s_dim, op.selectdim), True).T / self.width

    def bc_op(self, _sp: StencilProperties, _op: OpProperties, neu_coeff: float, dir_coeff: float, _r_rhs: jnp.ndarray | None, _x_subset: jnp.ndarray, _x: jnp.ndarray, x_at_origin_subset: jnp.ndarray, _x_at_origin: jnp.ndarray, nr_subset: jnp.ndarray) -> jnp.ndarray:
        total = jnp.zeros((self.fit_npoly, x_at_origin_subset.shape[0]))
        if neu_coeff != 0:
            grad = self.basis.evaluate(x_at_origin_subset, _first_derivative_multi_indices(self.s_dim), True) / self.width
            total = total + neu_coeff * jnp.sum(grad * nr_subset[:, None, :], axis=2).T
        if dir_coeff != 0:
            total = total + dir_coeff * self.basis.evaluate(x_at_origin_subset, jnp.zeros((1, self.s_dim), dtype=int), True).T
        if neu_coeff == 0 and dir_coeff == 0:
            raise ValueError("both boundary coefficients cannot be zero")
        return total

    def interp_op(self, _sp: StencilProperties, _op: OpProperties, _r_rhs: jnp.ndarray | None, _x_subset: jnp.ndarray, _x: jnp.ndarray, x_at_origin_subset: jnp.ndarray, _x_at_origin: jnp.ndarray) -> jnp.ndarray:
        return self.basis.evaluate(x_at_origin_subset, jnp.zeros((1, self.s_dim), dtype=int), True).T

    def _compute_weights_interior(self, x: jnp.ndarray, sp: StencilProperties, op: OpProperties, apply_op: str | Callable[..., jnp.ndarray], rhs_indices: int | jnp.ndarray) -> jnp.ndarray:
        self.initialize_geometry(x, sp)
        rhs_inds = jnp.atleast_1d(rhs_indices).astype(int) - 1
        x_subset = x[rhs_inds]
        x_at_origin_subset = self.xc[rhs_inds]
        bpoly = self._apply_operator(apply_op, sp, op, None, x_subset, x, x_at_origin_subset, self.xc)
        w = self.reconstructor.T @ bpoly
        if rhs_inds.size == 1 and int(rhs_inds[0]) == 0:
            name = str(apply_op).lower()
            if name in {"lap", "laplacian"}:
                self.l = w
            elif name in {"interp", "interpolation"}:
                self.wt = w
        return w

    def _compute_weights_boundary(self, x: jnp.ndarray, nr: jnp.ndarray, neu_coeff: float, dir_coeff: float, sp: StencilProperties, op: OpProperties, apply_op: str | Callable[..., jnp.ndarray], rhs_indices: int | jnp.ndarray) -> jnp.ndarray:
        self.initialize_geometry(x, sp)
        rhs_inds = jnp.atleast_1d(rhs_indices).astype(int) - 1
        x_subset = x[rhs_inds]
        x_at_origin_subset = self.xc[rhs_inds]
        nr_subset = nr[rhs_inds]
        bpoly = self._apply_boundary_operator(apply_op, sp, op, neu_coeff, dir_coeff, None, x_subset, x, x_at_origin_subset, self.xc, nr_subset)
        w = self.reconstructor.T @ bpoly
        if rhs_inds.size == 1 and int(rhs_inds[0]) == 0:
            self.bc = w
        return w

    def _apply_operator(self, apply_op: str | Callable[..., jnp.ndarray], sp: StencilProperties, op: OpProperties, r_rhs: jnp.ndarray | None, x_subset: jnp.ndarray, x: jnp.ndarray, x_at_origin_subset: jnp.ndarray, x_at_origin: jnp.ndarray) -> jnp.ndarray:
        if callable(apply_op):
            return apply_op(self, sp, op, r_rhs, x_subset, x, x_at_origin_subset, x_at_origin)
        name = str(apply_op).lower()
        if name in {"lap", "laplacian"}:
            return self.lap_op(sp, op, r_rhs, x_subset, x, x_at_origin_subset, x_at_origin)
        if name in {"grad", "gradient"}:
            return self.grad_op(sp, op, r_rhs, x_subset, x, x_at_origin_subset, x_at_origin)
        if name in {"interp", "interpolation"}:
            return self.interp_op(sp, op, r_rhs, x_subset, x, x_at_origin_subset, x_at_origin)
        raise ValueError(f"unknown operator {apply_op}")

    def _apply_boundary_operator(self, apply_op: str | Callable[..., jnp.ndarray], sp: StencilProperties, op: OpProperties, neu_coeff: float, dir_coeff: float, r_rhs: jnp.ndarray | None, x_subset: jnp.ndarray, x: jnp.ndarray, x_at_origin_subset: jnp.ndarray, x_at_origin: jnp.ndarray, nr_subset: jnp.ndarray) -> jnp.ndarray:
        if callable(apply_op):
            return apply_op(self, sp, op, neu_coeff, dir_coeff, r_rhs, x_subset, x, x_at_origin_subset, x_at_origin, nr_subset)
        name = str(apply_op).lower()
        if name in {"bc", "boundary"}:
            return self.bc_op(sp, op, neu_coeff, dir_coeff, r_rhs, x_subset, x, x_at_origin_subset, x_at_origin, nr_subset)
        raise ValueError(f"unknown boundary operator {apply_op}")


@dataclass
class FDDiffOp:
    approx_factory: Callable[[], object] = RBFStencil
    locations: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 2), dtype=int))
    values: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    op_matrix: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    n1: int = 0
    n2: int = 0
    stencils: list[dict[str, object]] = field(default_factory=list)
    recorded_stencil_centers: list[jnp.ndarray] = field(default_factory=list)
    recorded_stencil_globals: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=int))

    def _batched_weights(
        self,
        x_stencils: jnp.ndarray,
        op_name: str,
        st_props: StencilProperties,
        op_props: OpProperties,
        *,
        normals: jnp.ndarray | None = None,
        neu_coeff: jnp.ndarray | None = None,
        dir_coeff: jnp.ndarray | None = None,
    ) -> jnp.ndarray | None:
        approx = self.approx_factory()
        if isinstance(approx, RBFStencil):
            return _batched_rbf_weights(x_stencils, st_props, op_name, op_props, normals=normals, neu_coeff=neu_coeff, dir_coeff=dir_coeff)
        if isinstance(approx, WeightedLeastSquaresStencil):
            return _batched_wls_weights(x_stencils, st_props, op_name, op_props, normals=normals, neu_coeff=neu_coeff, dir_coeff=dir_coeff)
        return None

    def assemble_op(self, domain: DomainDescriptor, op_name: str, st_props: StencilProperties, op_props: OpProperties, *, neu_coeff: jnp.ndarray | None = None, dir_coeff: jnp.ndarray | None = None, active_rows: jnp.ndarray | None = None, stencil_graph: FrozenStencilGraph | None = None) -> None:
        center_points, center_row_ids, center_col_globals, center_normals = _pick_centers(domain, st_props.point_set)
        stencil_globals = domain.get_tree_globals(st_props.tree_mode)
        stencil_points = domain.get_tree_points(st_props.tree_mode)
        if stencil_graph is not None:
            expected_tree_mode = StencilProperties.normalize_tree_mode(st_props.tree_mode)
            expected_point_set = StencilProperties.normalize_point_set(st_props.point_set)
            if stencil_graph.tree_mode != expected_tree_mode or stencil_graph.point_set != expected_point_set:
                raise ValueError("stencil graph is incompatible with the requested tree/point set")
            active_rows = jnp.asarray(stencil_graph.active_rows, dtype=int) if active_rows is None else jnp.asarray(active_rows, dtype=int)
            if active_rows.shape != stencil_graph.active_rows.shape or not bool(jnp.all(active_rows == stencil_graph.active_rows)):
                raise ValueError("active_rows must match the supplied stencil graph")
            knn_indices = jnp.asarray(stencil_graph.knn_indices, dtype=int)
        else:
            if active_rows is None:
                active_rows = jnp.arange(1, center_points.shape[0] + 1)
            active_rows = jnp.asarray(active_rows, dtype=int)
            active_centers = center_points[active_rows - 1]
            knn_indices, _ = domain.query_knn(st_props.tree_mode, active_centers, st_props.n)
            knn_indices = knn_indices + 1
        active_centers = center_points[active_rows - 1]
        self.n1 = int(center_points.shape[0])
        self.n2 = _tree_column_count(domain, st_props.tree_mode)
        self.op_matrix = jnp.zeros((0, 0), dtype=float)
        self.stencils = []
        self.recorded_stencil_centers = []
        self.recorded_stencil_globals = jnp.zeros(active_rows.size, dtype=int)
        use_boundary = neu_coeff is not None or dir_coeff is not None
        active_row_ids = center_row_ids[active_rows - 1]
        active_globals = center_col_globals[active_rows - 1]
        if op_props.record_stencils:
            self.recorded_stencil_centers = [pt for pt in active_centers]
        self.recorded_stencil_globals = active_globals

        fast_path_ok = (
            not op_props.record_stencils
            and isinstance(op_name, str)
            and str(op_name).lower() in {"lap", "laplacian", "interp", "interpolation", "grad", "gradient", "bc", "boundary"}
        )
        if fast_path_ok:
            x_stencils = stencil_points[knn_indices - 1]
            batched_normals = center_normals[active_rows - 1] if use_boundary and center_normals is not None else None
            active_neu = neu_coeff[active_rows - 1] if neu_coeff is not None else None
            active_dir = dir_coeff[active_rows - 1] if dir_coeff is not None else None
            batched_weights = self._batched_weights(
                x_stencils,
                str(op_name).lower(),
                st_props,
                op_props,
                normals=batched_normals,
                neu_coeff=active_neu,
                dir_coeff=active_dir,
            )
            if batched_weights is not None:
                flat_rows, flat_cols, flat_vals, locations = _assemble_operator_entries(
                    active_row_ids,
                    stencil_globals,
                    knn_indices,
                    batched_weights,
                )
                self.locations = locations
                self.values = flat_vals
                return

        locs = []
        vals = []
        for k, local_row in enumerate(active_rows):
            indices, w, stencil, center_point, row_id, global_id = _assemble_one(
                self.approx_factory,
                stencil_points,
                center_points,
                center_row_ids,
                center_col_globals,
                center_normals,
                int(local_row),
                st_props,
                op_props,
                op_name,
                use_boundary,
                neu_coeff,
                dir_coeff,
                knn_indices[k],
            )
            rows = jnp.full(indices.shape[0], row_id - 1, dtype=int)
            cols = stencil_globals[indices - 1] - 1
            locs.append(jnp.column_stack([rows + 1, cols + 1]))
            vals.append(w[:, 0])
            if op_props.record_stencils:
                self.stencils.append({"Approx": stencil, "Indices": stencil_globals[indices - 1]})
        self.locations = jnp.vstack(locs) if locs else jnp.zeros((0, 2), dtype=int)
        self.values = jnp.concatenate(vals) if vals else jnp.zeros(0)

    def get_op(self) -> jnp.ndarray:
        if self.op_matrix.shape != (self.n1, self.n2):
            if self.locations.size == 0:
                self.op_matrix = jnp.zeros((self.n1, self.n2), dtype=self.values.dtype)
            else:
                self.op_matrix = _scatter_operator_matrix(
                    self.locations[:, 0] - 1,
                    self.locations[:, 1] - 1,
                    self.values,
                    n1=self.n1,
                    n2=self.n2,
                )
        return self.op_matrix


@dataclass
class CrossNodeDiffOp:
    approx_factory: Callable[[], object] = RBFStencil
    locations: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 2), dtype=int))
    values: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    op_matrix: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    n1: int = 0
    n2: int = 0

    def assemble_op(
        self,
        source_domain: DomainDescriptor,
        target_domain: DomainDescriptor,
        op_name: str,
        st_props: StencilProperties,
        op_props: OpProperties,
        *,
        active_rows: jnp.ndarray | None = None,
    ) -> None:
        target_points, target_row_ids, _target_col_globals, _target_normals = _pick_centers(target_domain, st_props.point_set)
        if active_rows is None:
            active_rows = jnp.arange(1, target_points.shape[0] + 1)
        active_rows = jnp.asarray(active_rows, dtype=int).reshape(-1)
        active_targets = target_points[active_rows - 1]
        source_points = source_domain.get_tree_points(st_props.tree_mode)
        source_globals = source_domain.get_tree_globals(st_props.tree_mode)
        self.n1 = int(target_points.shape[0])
        self.n2 = _tree_column_count(source_domain, st_props.tree_mode)
        knn_indices, _ = source_domain.query_knn(st_props.tree_mode, active_targets, st_props.n)
        knn_indices = knn_indices + 1

        approx = self.approx_factory()
        x_stencils = source_points[knn_indices - 1]
        batched_weights = _batched_weights_at_points(approx, x_stencils, active_targets, st_props, op_name, op_props)
        if batched_weights is None:
            raise ValueError("CrossNodeDiffOp currently supports batched RBF/WLS interpolation and gradient operators")
        active_row_ids = target_row_ids[active_rows - 1]
        flat_rows, flat_cols, flat_vals, locations = _assemble_operator_entries(
            active_row_ids,
            source_globals,
            knn_indices,
            batched_weights,
        )
        self.op_matrix = jnp.zeros((0, 0), dtype=float)
        self.locations = locations
        self.values = flat_vals

    def get_op(self) -> jnp.ndarray:
        if self.op_matrix.shape != (self.n1, self.n2):
            if self.locations.size == 0:
                self.op_matrix = jnp.zeros((self.n1, self.n2), dtype=self.values.dtype)
            else:
                self.op_matrix = _scatter_operator_matrix(
                    self.locations[:, 0] - 1,
                    self.locations[:, 1] - 1,
                    self.values,
                    n1=self.n1,
                    n2=self.n2,
                )
        return self.op_matrix


@dataclass
class FDODiffOp(FDDiffOp):
    pass


def build_frozen_stencil_graph(
    domain: DomainDescriptor,
    st_props: StencilProperties,
    *,
    active_rows: jnp.ndarray | None = None,
) -> FrozenStencilGraph:
    domain.build_structs()
    center_points, _center_row_ids, _center_col_globals, _center_normals = _pick_centers(domain, st_props.point_set)
    if active_rows is None:
        active_rows = jnp.arange(1, center_points.shape[0] + 1)
    active_rows = jnp.asarray(active_rows, dtype=int)
    active_centers = center_points[active_rows - 1]
    knn_indices, _ = domain.query_knn(st_props.tree_mode, active_centers, st_props.n)
    return FrozenStencilGraph(
        tree_mode=st_props.tree_mode,
        point_set=st_props.point_set,
        stencil_size=int(knn_indices.shape[1]),
        active_rows=active_rows,
        knn_indices=knn_indices + 1,
    )


def _assemble_one(approx_factory: Callable[[], object], stencil_points: jnp.ndarray, center_points: jnp.ndarray, center_row_ids: jnp.ndarray, center_col_globals: jnp.ndarray, center_normals: jnp.ndarray | None, local_row: int, st_props: StencilProperties, op_props: OpProperties, op_name: str, use_boundary: bool, neu_coeff: jnp.ndarray | None, dir_coeff: jnp.ndarray | None, indices: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, object, jnp.ndarray, int, int]:
    center_point = center_points[local_row - 1]
    center_row_id = int(center_row_ids[local_row - 1])
    center_col_global = int(center_col_globals[local_row - 1])
    loc_x = stencil_points[indices - 1]
    stencil = approx_factory()
    if use_boundary:
        loc_nr = center_normals[local_row - 1 : local_row]
        w = stencil.compute_weights(loc_x, loc_nr, float(neu_coeff[local_row - 1]), float(dir_coeff[local_row - 1]), st_props, op_props, op_name, 1)
    else:
        w = stencil.compute_weights(loc_x, st_props, op_props, op_name, 1)
    return indices, w, stencil, center_point, center_row_id, center_col_global


def _pick_centers(domain: DomainDescriptor, point_set: str) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
    normals = None
    mode = StencilProperties.normalize_point_set(point_set)
    if mode == "all":
        points = domain.get_all_nodes()
        row_ids = jnp.arange(1, points.shape[0] + 1)
        col_globals = row_ids
    elif mode == "interior_boundary":
        points = domain.get_int_bdry_nodes()
        row_ids = jnp.arange(1, points.shape[0] + 1)
        col_globals = row_ids
    elif mode == "boundary":
        points = domain.get_bdry_nodes()
        ni = domain.get_num_interior_nodes()
        row_ids = jnp.arange(1, points.shape[0] + 1)
        col_globals = ni + jnp.arange(1, points.shape[0] + 1)
        normals = domain.get_nrmls()
    else:
        raise ValueError("unknown point set")
    return points, row_ids, col_globals, normals


def _tree_column_count(domain: DomainDescriptor, tree_mode: str) -> int:
    mode = StencilProperties.normalize_tree_mode(tree_mode)
    if mode == "boundary":
        return domain.get_num_interior_nodes() + domain.get_num_bdry_nodes()
    return int(domain.get_tree_points(mode).shape[0])
