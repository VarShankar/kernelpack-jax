from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from jax import lax
from jax.errors import TracerArrayConversionError
import jax.numpy as jnp
import numpy as np

from kernelpack.accelerators import WarpUnavailableError, warp_available, warp_exact_ball
from kernelpack.domain import DomainDescriptor
from kernelpack.geometry import distance_matrix
from kernelpack.rbffd import RBFStencil, StencilProperties
from kernelpack.rbffd.core import _batched_basis_eval, _build_legendre_basis_data
from ._common import (
    FrozenILUPreconditioner,
    SparseCOOMatrix,
    build_fixed_ilu_preconditioner,
    solve_sparse_system_gmres,
)
from ._pu import (
    PUQueryGroup,
    PUQueryGrouping,
    _batched_padded_local_weights,
    build_padded_patch_arrays,
    build_patch_node_ids,
    build_patch_stencil_properties,
    choose_minimum_patch_nodes,
    choose_patch_centers,
    choose_patch_radius,
    choose_patch_spacing,
    pu_patch_weight,
)


@dataclass(frozen=True)
class AdvectionBoundaryCondition:
    mode: str = "unspecified"
    normal_velocity_tolerance: float = 1.0e-10
    periodic_patches: object | None = None
    inflow_value: Callable[..., jnp.ndarray] | None = None


@dataclass(frozen=True)
class LocalizedAdvectionPatchData:
    nodes: jnp.ndarray
    centers: jnp.ndarray
    radius: float
    spacing: float
    node_ids: tuple[jnp.ndarray, ...]
    node_counts: jnp.ndarray
    padded_node_ids: jnp.ndarray
    padded_nodes: jnp.ndarray
    padded_valid_mask: jnp.ndarray
    padded_self_eval: jnp.ndarray
    padded_coeff_indices: jnp.ndarray
    padded_coeff_valid_mask: jnp.ndarray
    patch_nodes: tuple[jnp.ndarray, ...]
    centroids: tuple[jnp.ndarray, ...]
    widths: tuple[float, ...]
    lhs_systems: tuple[jnp.ndarray, ...]
    padded_lhs_systems: jnp.ndarray
    coeff_offsets: tuple[int, ...]
    total_coeffs: int
    stencil_props: StencilProperties
    index_set: jnp.ndarray
    recurrence_a: jnp.ndarray
    recurrence_b: jnp.ndarray


@dataclass(frozen=True)
class ForwardNodalData:
    local_cardinal_coeffs: tuple[jnp.ndarray, ...]
    nodal_mass_weights: jnp.ndarray
    local_constant_column: jnp.ndarray
    fixed_collocation: SparseCOOMatrix
    fixed_collocation_mass_augmented: SparseCOOMatrix
    fixed_collocation_preconditioner: FrozenILUPreconditioner
    fixed_collocation_mass_augmented_preconditioner: FrozenILUPreconditioner


@dataclass(frozen=True)
class LocalizedQueryMap:
    patch_ids: jnp.ndarray
    patch_distances: jnp.ndarray
    valid_mask: jnp.ndarray


def rk4sl_step(t: float, x: jnp.ndarray, dt: float, velocity: Callable[..., jnp.ndarray]) -> jnp.ndarray:
    k1 = jnp.asarray(velocity(t, x), dtype=float)
    k2 = jnp.asarray(velocity(t + 0.5 * dt, x + 0.5 * dt * k1), dtype=float)
    k3 = jnp.asarray(velocity(t + 0.5 * dt, x + 0.5 * dt * k2), dtype=float)
    k4 = jnp.asarray(velocity(t + dt, x + dt * k3), dtype=float)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _build_localized_patch_data(
    nodes: jnp.ndarray,
    xi: int,
    h: float,
    patch_spacing_factor: float,
    patch_radius_factor: float,
) -> LocalizedAdvectionPatchData:
    nodes = jnp.asarray(nodes, dtype=float)
    nodes_np = np.asarray(nodes, dtype=float)
    dim = int(nodes.shape[1])
    spacing = choose_patch_spacing(h, patch_spacing_factor)
    radius = choose_patch_radius(h, patch_radius_factor)
    min_nodes = choose_minimum_patch_nodes(dim, xi)
    centers = choose_patch_centers(nodes, spacing, backend="auto")
    centers_np = np.asarray(centers, dtype=float)
    local_domain = DomainDescriptor()
    local_domain.set_nodes(nodes, jnp.zeros((0, dim), dtype=float))
    local_domain.set_normals(jnp.zeros((0, dim), dtype=float))
    local_domain.set_sep_rad(float(h))
    local_domain.build_structs()
    node_ids = build_patch_node_ids(
        local_domain,
        centers,
        radius,
        min_nodes,
        backend="auto",
    )
    node_ids_list: list[jnp.ndarray] = []
    patch_nodes: list[jnp.ndarray] = []
    centroids: list[jnp.ndarray] = []
    widths: list[float] = []
    lhs_systems: list[jnp.ndarray] = []
    coeff_offsets = [0]

    sp = build_patch_stencil_properties(dim, xi)
    index_set, recurrence_a, recurrence_b = _build_legendre_basis_data(dim, sp.ell)

    zero_multi = jnp.zeros((1, dim), dtype=int)
    for p in range(int(centers.shape[0])):
        ids_np = np.asarray(node_ids[p], dtype=int)
        xloc_np = nodes_np[ids_np]
        xm_np = np.mean(xloc_np, axis=0)
        diff_np = xloc_np[:, None, :] - xloc_np[None, :, :]
        r_np = np.linalg.norm(diff_np, axis=2)
        width = float(max(float(np.max(r_np)), 1.0))
        xc = jnp.asarray((xloc_np - xm_np) / width, dtype=float)
        r = jnp.asarray(r_np, dtype=float)
        k = np.asarray(RBFStencil.phs_rbf(r, sp.spline_degree), dtype=float)
        pmat = np.asarray(_batched_basis_eval(index_set, recurrence_a, recurrence_b, xc, zero_multi), dtype=float)
        lhs = np.zeros((xloc_np.shape[0] + sp.npoly, xloc_np.shape[0] + sp.npoly), dtype=float)
        lhs[: xloc_np.shape[0], : xloc_np.shape[0]] = k
        lhs[: xloc_np.shape[0], xloc_np.shape[0] :] = pmat
        lhs[xloc_np.shape[0] :, : xloc_np.shape[0]] = np.swapaxes(pmat, 0, 1)
        node_ids_list.append(jnp.asarray(ids_np, dtype=int))
        patch_nodes.append(jnp.asarray(xloc_np, dtype=float))
        centroids.append(jnp.asarray(xm_np, dtype=float))
        widths.append(width)
        lhs_systems.append(jnp.asarray(lhs, dtype=float))
        coeff_offsets.append(coeff_offsets[-1] + int(xloc_np.shape[0] + sp.npoly))

    node_ids_tuple = tuple(node_ids_list)
    node_counts, padded_node_ids, padded_nodes = build_padded_patch_arrays(nodes, node_ids_tuple)
    padded_valid_mask = padded_node_ids >= 0
    max_nodes = int(padded_node_ids.shape[1]) if padded_node_ids.ndim == 2 else 0
    system_size = max_nodes + sp.npoly
    padded_lhs_systems = np.zeros((len(node_ids_tuple), system_size, system_size), dtype=float)
    padded_self_eval = np.zeros((len(node_ids_tuple), max_nodes, system_size), dtype=float)
    padded_coeff_indices = np.zeros((len(node_ids_tuple), system_size), dtype=int)
    padded_coeff_valid_mask = np.zeros((len(node_ids_tuple), system_size), dtype=bool)
    for patch_idx, lhs in enumerate(lhs_systems):
        lhs_np = np.asarray(lhs, dtype=float)
        nloc = int(node_counts[patch_idx])
        begin = int(coeff_offsets[patch_idx])
        end = int(coeff_offsets[patch_idx + 1])
        padded_lhs_systems[patch_idx, :max_nodes, :max_nodes] = np.eye(max_nodes, dtype=float)
        padded_lhs_systems[patch_idx, :nloc, :nloc] = lhs_np[:nloc, :nloc]
        padded_lhs_systems[patch_idx, :nloc, max_nodes:] = lhs_np[:nloc, nloc:]
        padded_lhs_systems[patch_idx, max_nodes:, :nloc] = lhs_np[nloc:, :nloc]
        padded_self_eval[patch_idx, :nloc, :nloc] = lhs_np[:nloc, :nloc]
        padded_self_eval[patch_idx, :nloc, max_nodes:] = lhs_np[:nloc, nloc:]
        padded_coeff_indices[patch_idx, :nloc] = np.arange(begin, begin + nloc, dtype=int)
        padded_coeff_valid_mask[patch_idx, :nloc] = True
        padded_coeff_indices[patch_idx, max_nodes : max_nodes + sp.npoly] = np.arange(begin + nloc, end, dtype=int)
        padded_coeff_valid_mask[patch_idx, max_nodes : max_nodes + sp.npoly] = True

    return LocalizedAdvectionPatchData(
        nodes=nodes,
        centers=centers,
        radius=float(radius),
        spacing=float(spacing),
        node_ids=node_ids_tuple,
        node_counts=node_counts,
        padded_node_ids=padded_node_ids,
        padded_nodes=padded_nodes,
        padded_valid_mask=padded_valid_mask,
        padded_self_eval=jnp.asarray(padded_self_eval, dtype=float),
        padded_coeff_indices=jnp.asarray(padded_coeff_indices, dtype=int),
        padded_coeff_valid_mask=jnp.asarray(padded_coeff_valid_mask, dtype=bool),
        patch_nodes=tuple(patch_nodes),
        centroids=tuple(centroids),
        widths=tuple(widths),
        lhs_systems=tuple(lhs_systems),
        padded_lhs_systems=jnp.asarray(padded_lhs_systems, dtype=float),
        coeff_offsets=tuple(coeff_offsets),
        total_coeffs=int(coeff_offsets[-1]),
        stencil_props=sp,
        index_set=index_set,
        recurrence_a=recurrence_a,
        recurrence_b=recurrence_b,
    )


def _local_basis_values(
    patch_data: LocalizedAdvectionPatchData,
    patch_idx: int,
    xe: jnp.ndarray,
) -> jnp.ndarray:
    xe = jnp.atleast_2d(jnp.asarray(xe, dtype=float))
    xloc = patch_data.patch_nodes[patch_idx]
    xm = patch_data.centroids[patch_idx]
    width = patch_data.widths[patch_idx]
    re = distance_matrix(xe, xloc)
    xec = (xe - xm) / width
    rbf_part = RBFStencil.phs_rbf(re, patch_data.stencil_props.spline_degree)
    poly_part = _batched_basis_eval(
        patch_data.index_set,
        patch_data.recurrence_a,
        patch_data.recurrence_b,
        xec,
        jnp.zeros((1, xloc.shape[1]), dtype=int),
    )
    return jnp.concatenate([rbf_part, poly_part], axis=1)


def _as_coeff_matrix(coeffs: jnp.ndarray) -> tuple[jnp.ndarray, bool]:
    arr = jnp.asarray(coeffs, dtype=float)
    return (arr[:, None], True) if arr.ndim == 1 else (arr, False)


def _restore_coeff_shape(coeffs: jnp.ndarray, squeezed: bool) -> jnp.ndarray:
    return coeffs[:, 0] if squeezed else coeffs


def _pad_local_coeff_blocks(
    patch_data: LocalizedAdvectionPatchData,
    coeff_blocks: tuple[jnp.ndarray, ...],
) -> jnp.ndarray:
    num_patches = int(patch_data.node_counts.shape[0])
    max_nodes = int(patch_data.padded_node_ids.shape[1])
    npoly = int(patch_data.stencil_props.npoly)
    ncols = int(coeff_blocks[0].shape[1]) if coeff_blocks else 1
    padded = jnp.zeros((num_patches, max_nodes + npoly, ncols), dtype=float)
    for patch_idx, block in enumerate(coeff_blocks):
        nloc = int(patch_data.node_counts[patch_idx])
        padded = padded.at[patch_idx, :nloc].set(block[:nloc])
        padded = padded.at[patch_idx, max_nodes : max_nodes + npoly].set(block[nloc:])
    return padded


def _coeff_matrix_to_padded(
    patch_data: LocalizedAdvectionPatchData,
    coeffs: jnp.ndarray,
) -> jnp.ndarray:
    coeffs_2d = jnp.asarray(coeffs, dtype=float)
    safe_ids = jnp.where(patch_data.padded_coeff_valid_mask, patch_data.padded_coeff_indices, 0)
    gathered = coeffs_2d[safe_ids]
    return gathered * patch_data.padded_coeff_valid_mask[:, :, None]


def _assemble_coeff_matrix_from_padded(
    patch_data: LocalizedAdvectionPatchData,
    padded_coeffs: jnp.ndarray,
) -> jnp.ndarray:
    coeffs = jnp.zeros((patch_data.total_coeffs, padded_coeffs.shape[2]), dtype=float)
    flat_ids = patch_data.padded_coeff_indices.reshape(-1)
    flat_mask = patch_data.padded_coeff_valid_mask.reshape(-1)
    flat_vals = padded_coeffs.reshape(-1, padded_coeffs.shape[2])
    safe_ids = jnp.where(flat_mask, flat_ids, 0)
    return coeffs.at[safe_ids].add(flat_vals * flat_mask[:, None])


def _local_nodal_values_from_coeff_blocks(
    patch_data: LocalizedAdvectionPatchData,
    coeff_blocks: tuple[jnp.ndarray, ...],
) -> jnp.ndarray:
    padded_coeffs = _pad_local_coeff_blocks(patch_data, coeff_blocks)
    local_values = jnp.matmul(patch_data.padded_self_eval, padded_coeffs)
    return local_values * patch_data.padded_valid_mask[:, :, None]


def _local_nodal_values_from_coeff_matrix(
    patch_data: LocalizedAdvectionPatchData,
    coeffs: jnp.ndarray,
) -> jnp.ndarray:
    padded_coeffs = _coeff_matrix_to_padded(patch_data, coeffs)
    local_values = jnp.matmul(patch_data.padded_self_eval, padded_coeffs)
    return local_values * patch_data.padded_valid_mask[:, :, None]


def _build_localized_query_grouping(
    patch_data: LocalizedAdvectionPatchData,
    xq: jnp.ndarray,
) -> PUQueryGrouping:
    xq = jnp.atleast_2d(jnp.asarray(xq, dtype=float))
    centers = np.asarray(patch_data.centers, dtype=float)
    xq_np = np.asarray(xq, dtype=float)
    if centers.size == 0:
        return PUQueryGrouping(groups=tuple())
    try:
        ids_rows, _ = warp_exact_ball(xq_np, centers, float(patch_data.radius))
    except WarpUnavailableError:
        d = np.linalg.norm(xq_np[:, None, :] - centers[None, :, :], axis=2)
        ids_rows = [np.flatnonzero(d[q] < patch_data.radius).astype(np.int32, copy=False) for q in range(xq_np.shape[0])]
    count_to_queries: dict[int, list[int]] = {}
    patch_rows: dict[int, list[np.ndarray]] = {}
    d_fallback = np.linalg.norm(xq_np[:, None, :] - centers[None, :, :], axis=2)
    for q, ids in enumerate(ids_rows):
        ids_np = np.asarray(ids, dtype=int)
        if ids_np.size == 0:
            ids_np = np.asarray([int(np.argmin(d_fallback[q]))], dtype=int)
        count = int(ids_np.size)
        count_to_queries.setdefault(count, []).append(q)
        patch_rows.setdefault(count, []).append(ids_np)
    groups = []
    for count in sorted(count_to_queries):
        query_ids = jnp.asarray(count_to_queries[count], dtype=int)
        patch_ids = jnp.asarray(np.stack(patch_rows[count], axis=0), dtype=int)
        groups.append(PUQueryGroup(query_ids=query_ids, patch_ids=patch_ids))
    return PUQueryGrouping(groups=tuple(groups))


def _build_localized_query_map(
    patch_data: LocalizedAdvectionPatchData,
    xq: jnp.ndarray,
) -> LocalizedQueryMap:
    xq = jnp.atleast_2d(jnp.asarray(xq, dtype=float))
    xq_np = np.asarray(xq, dtype=float)
    centers = np.asarray(patch_data.centers, dtype=float)
    if xq_np.shape[0] == 0:
        return LocalizedQueryMap(
            patch_ids=jnp.zeros((0, 0), dtype=int),
            patch_distances=jnp.zeros((0, 0), dtype=float),
            valid_mask=jnp.zeros((0, 0), dtype=bool),
        )
    if centers.size == 0:
        raise ValueError("localized query map requires at least one patch center")
    try:
        ids_rows, dist_rows = warp_exact_ball(xq_np, centers, float(patch_data.radius))
    except WarpUnavailableError:
        d = np.linalg.norm(xq_np[:, None, :] - centers[None, :, :], axis=2)
        ids_rows = [np.flatnonzero(d[q] < patch_data.radius).astype(np.int32, copy=False) for q in range(xq_np.shape[0])]
        dist_rows = [d[q, ids_rows[q]].astype(np.float32, copy=False) for q in range(xq_np.shape[0])]
    fallback_dist = np.linalg.norm(xq_np[:, None, :] - centers[None, :, :], axis=2)
    max_count = 1
    for q, ids in enumerate(ids_rows):
        if np.asarray(ids).size == 0:
            max_count = max(max_count, 1)
        else:
            max_count = max(max_count, int(np.asarray(ids).size))
    patch_ids = np.zeros((xq_np.shape[0], max_count), dtype=np.int32)
    patch_distances = np.zeros((xq_np.shape[0], max_count), dtype=np.float32)
    valid_mask = np.zeros((xq_np.shape[0], max_count), dtype=bool)
    for q, ids in enumerate(ids_rows):
        ids_np = np.asarray(ids, dtype=np.int32)
        d_np = np.asarray(dist_rows[q], dtype=np.float32)
        if ids_np.size == 0:
            nearest = int(np.argmin(fallback_dist[q]))
            ids_np = np.asarray([nearest], dtype=np.int32)
            d_np = np.asarray([fallback_dist[q, nearest]], dtype=np.float32)
        count = int(ids_np.size)
        patch_ids[q, :count] = ids_np
        patch_distances[q, :count] = d_np
        valid_mask[q, :count] = True
    return LocalizedQueryMap(
        patch_ids=jnp.asarray(patch_ids, dtype=int),
        patch_distances=jnp.asarray(patch_distances, dtype=float),
        valid_mask=jnp.asarray(valid_mask, dtype=bool),
    )


def _localized_group_patch_weights_from_distances(
    patch_distances: jnp.ndarray,
    valid_mask: jnp.ndarray,
    radius: float,
) -> jnp.ndarray:
    d = jnp.asarray(patch_distances, dtype=float)
    valid = jnp.asarray(valid_mask, dtype=bool)
    raw = pu_patch_weight(d / radius) * valid
    denom = raw.sum(axis=1, keepdims=True)
    nearest = jnp.argmin(jnp.where(valid, d, jnp.inf), axis=1)
    fallback = jnp.eye(d.shape[1], dtype=float)[nearest]
    return jnp.where(denom > 1.0e-14, raw / denom, fallback)


def _dense_patch_weights(
    patch_data: LocalizedAdvectionPatchData,
    xq: jnp.ndarray,
) -> jnp.ndarray:
    xq = jnp.atleast_2d(jnp.asarray(xq, dtype=float))
    d = jnp.linalg.norm(xq[:, None, :] - patch_data.centers[None, :, :], axis=2)
    raw = pu_patch_weight(d / patch_data.radius)
    denom = raw.sum(axis=1, keepdims=True)
    nearest = jnp.argmin(d, axis=1)
    fallback = jnp.eye(patch_data.centers.shape[0], dtype=float)[nearest]
    return jnp.where(denom > 1.0e-14, raw / denom, fallback)


def _evaluate_coeff_field_grouped(
    patch_data: LocalizedAdvectionPatchData,
    coeff_blocks: tuple[jnp.ndarray, ...],
    xq: jnp.ndarray,
    *,
    query_grouping: PUQueryGrouping | None = None,
) -> jnp.ndarray:
    xq = jnp.atleast_2d(jnp.asarray(xq, dtype=float))
    if xq.size == 0:
        ncols = int(coeff_blocks[0].shape[1]) if coeff_blocks else 1
        return jnp.zeros((0, ncols), dtype=float)
    ncols = int(coeff_blocks[0].shape[1]) if coeff_blocks else 1
    local_values = _local_nodal_values_from_coeff_blocks(patch_data, coeff_blocks)
    dim = int(patch_data.nodes.shape[1])
    query_map = _build_localized_query_map(patch_data, xq)
    patch_ids = query_map.patch_ids
    patch_mask = query_map.valid_mask
    query_weights = _localized_group_patch_weights_from_distances(
        query_map.patch_distances,
        patch_mask,
        patch_data.radius,
    )
    patch_nodes = patch_data.padded_nodes[patch_ids]
    patch_valid_nodes = patch_data.padded_valid_mask[patch_ids]
    nq, patch_count, nmax, _ = patch_nodes.shape
    flat_nodes = patch_nodes.reshape(nq * patch_count, nmax, dim)
    flat_valid = patch_valid_nodes.reshape(nq * patch_count, nmax)
    flat_queries = jnp.repeat(xq, patch_count, axis=0)
    flat_zero_normals = jnp.zeros((flat_queries.shape[0], dim), dtype=float)
    flat_zero_coeff = jnp.zeros((flat_queries.shape[0],), dtype=float)
    flat_interp = _batched_padded_local_weights(
        flat_nodes,
        flat_valid,
        flat_queries,
        flat_zero_normals,
        flat_zero_coeff,
        flat_zero_coeff,
        patch_data.index_set,
        patch_data.recurrence_a,
        patch_data.recurrence_b,
        patch_data.stencil_props.spline_degree,
        0,
    ).reshape(nq, patch_count, nmax)
    patch_local_values = local_values[patch_ids]
    weighted_interp = flat_interp * query_weights[:, :, None] * patch_valid_nodes * patch_mask[:, :, None]
    return jnp.sum(weighted_interp[:, :, :, None] * patch_local_values, axis=(1, 2))


def _evaluate_coeff_field_grouped_from_matrix(
    patch_data: LocalizedAdvectionPatchData,
    coeffs: jnp.ndarray,
    xq: jnp.ndarray,
) -> jnp.ndarray:
    xq = jnp.atleast_2d(jnp.asarray(xq, dtype=float))
    coeffs_2d = jnp.asarray(coeffs, dtype=float)
    if xq.size == 0:
        return jnp.zeros((0, coeffs_2d.shape[1]), dtype=float)
    local_values = _local_nodal_values_from_coeff_matrix(patch_data, coeffs_2d)
    dim = int(patch_data.nodes.shape[1])
    query_map = _build_localized_query_map(patch_data, xq)
    patch_ids = query_map.patch_ids
    patch_mask = query_map.valid_mask
    query_weights = _localized_group_patch_weights_from_distances(
        query_map.patch_distances,
        patch_mask,
        patch_data.radius,
    )
    patch_nodes = patch_data.padded_nodes[patch_ids]
    patch_valid_nodes = patch_data.padded_valid_mask[patch_ids]
    nq, patch_count, nmax, _ = patch_nodes.shape
    flat_nodes = patch_nodes.reshape(nq * patch_count, nmax, dim)
    flat_valid = patch_valid_nodes.reshape(nq * patch_count, nmax)
    flat_queries = jnp.repeat(xq, patch_count, axis=0)
    flat_zero_normals = jnp.zeros((flat_queries.shape[0], dim), dtype=float)
    flat_zero_coeff = jnp.zeros((flat_queries.shape[0],), dtype=float)
    flat_interp = _batched_padded_local_weights(
        flat_nodes,
        flat_valid,
        flat_queries,
        flat_zero_normals,
        flat_zero_coeff,
        flat_zero_coeff,
        patch_data.index_set,
        patch_data.recurrence_a,
        patch_data.recurrence_b,
        patch_data.stencil_props.spline_degree,
        0,
    ).reshape(nq, patch_count, nmax)
    patch_local_values = local_values[patch_ids]
    weighted_interp = flat_interp * query_weights[:, :, None] * patch_valid_nodes * patch_mask[:, :, None]
    return jnp.sum(weighted_interp[:, :, :, None] * patch_local_values, axis=(1, 2))


def _evaluate_coeff_field_dense_from_matrix(
    patch_data: LocalizedAdvectionPatchData,
    coeffs: jnp.ndarray,
    xq: jnp.ndarray,
) -> jnp.ndarray:
    xq = jnp.atleast_2d(jnp.asarray(xq, dtype=float))
    coeffs_2d = jnp.asarray(coeffs, dtype=float)
    if xq.size == 0:
        return jnp.zeros((0, coeffs_2d.shape[1]), dtype=float)
    local_values = _local_nodal_values_from_coeff_matrix(patch_data, coeffs_2d)
    query_weights = _dense_patch_weights(patch_data, xq)
    dim = int(patch_data.nodes.shape[1])
    npatches = int(patch_data.padded_nodes.shape[0])
    nmax = int(patch_data.padded_nodes.shape[1])
    patch_nodes = jnp.broadcast_to(patch_data.padded_nodes[None, :, :, :], (xq.shape[0], npatches, nmax, dim))
    patch_valid_nodes = jnp.broadcast_to(patch_data.padded_valid_mask[None, :, :], (xq.shape[0], npatches, nmax))
    flat_nodes = patch_nodes.reshape(xq.shape[0] * npatches, nmax, dim)
    flat_valid = patch_valid_nodes.reshape(xq.shape[0] * npatches, nmax)
    flat_queries = jnp.repeat(xq, npatches, axis=0)
    flat_zero_normals = jnp.zeros((flat_queries.shape[0], dim), dtype=float)
    flat_zero_coeff = jnp.zeros((flat_queries.shape[0],), dtype=float)
    flat_interp = _batched_padded_local_weights(
        flat_nodes,
        flat_valid,
        flat_queries,
        flat_zero_normals,
        flat_zero_coeff,
        flat_zero_coeff,
        patch_data.index_set,
        patch_data.recurrence_a,
        patch_data.recurrence_b,
        patch_data.stencil_props.spline_degree,
        0,
    ).reshape(xq.shape[0], npatches, nmax)
    patch_local_values = jnp.broadcast_to(local_values[None, :, :, :], (xq.shape[0], npatches, nmax, coeffs_2d.shape[1]))
    weighted_interp = flat_interp * query_weights[:, :, None] * patch_valid_nodes
    return jnp.sum(weighted_interp[:, :, :, None] * patch_local_values, axis=(1, 2))


def _evaluate_coeff_field_auto_from_matrix(
    patch_data: LocalizedAdvectionPatchData,
    coeffs: jnp.ndarray,
    xq: jnp.ndarray,
) -> jnp.ndarray:
    try:
        return _evaluate_coeff_field_grouped_from_matrix(patch_data, coeffs, xq)
    except TracerArrayConversionError:
        return _evaluate_coeff_field_dense_from_matrix(patch_data, coeffs, xq)


def _solve_padded_local_systems(
    patch_data: LocalizedAdvectionPatchData,
    local_samples: jnp.ndarray,
) -> jnp.ndarray:
    samples = jnp.asarray(local_samples, dtype=float)
    samples_2d = samples[:, :, None] if samples.ndim == 2 else samples
    num_patches = int(patch_data.node_counts.shape[0])
    max_nodes = int(patch_data.padded_node_ids.shape[1])
    npoly = int(patch_data.stencil_props.npoly)
    rhs = jnp.zeros((num_patches, max_nodes + npoly, samples_2d.shape[2]), dtype=float)
    rhs = rhs.at[:, :max_nodes, :].set(samples_2d)
    return jnp.linalg.solve(patch_data.padded_lhs_systems, rhs)


def _compact_padded_local_coeffs(
    patch_data: LocalizedAdvectionPatchData,
    padded_coeffs: jnp.ndarray,
) -> tuple[jnp.ndarray, ...]:
    coeff_blocks: list[jnp.ndarray] = []
    max_nodes = int(patch_data.padded_node_ids.shape[1])
    npoly = int(patch_data.stencil_props.npoly)
    for patch_idx in range(int(patch_data.node_counts.shape[0])):
        nloc = int(patch_data.node_counts[patch_idx])
        top = padded_coeffs[patch_idx, :nloc]
        poly = padded_coeffs[patch_idx, max_nodes : max_nodes + npoly]
        coeff_blocks.append(jnp.concatenate([top, poly], axis=0))
    return tuple(coeff_blocks)


def _patch_overlap_weights(patch_data: LocalizedAdvectionPatchData, xq: jnp.ndarray) -> jnp.ndarray:
    xq = jnp.atleast_2d(jnp.asarray(xq, dtype=float))
    query_map = _build_localized_query_map(patch_data, xq)
    local_weights = _localized_group_patch_weights_from_distances(
        query_map.patch_distances,
        query_map.valid_mask,
        patch_data.radius,
    )
    weights = jnp.zeros((xq.shape[0], patch_data.centers.shape[0]), dtype=float)
    scatter_rows = jnp.broadcast_to(jnp.arange(xq.shape[0], dtype=int)[:, None], query_map.patch_ids.shape).reshape(-1)
    scatter_cols = query_map.patch_ids.reshape(-1)
    scatter_vals = (local_weights * query_map.valid_mask).reshape(-1)
    return weights.at[scatter_rows, scatter_cols].add(scatter_vals)


def _evaluate_nodal_field(
    patch_data: LocalizedAdvectionPatchData,
    forward_nodal_data: ForwardNodalData,
    xq: jnp.ndarray,
    nodal_values: jnp.ndarray,
) -> jnp.ndarray:
    del forward_nodal_data
    xq = jnp.atleast_2d(jnp.asarray(xq, dtype=float))
    nodal_2d = jnp.asarray(nodal_values, dtype=float)
    nodal_2d = nodal_2d[:, None] if nodal_2d.ndim == 1 else nodal_2d
    dim = int(patch_data.nodes.shape[1])
    query_map = _build_localized_query_map(patch_data, xq)
    patch_ids = query_map.patch_ids
    patch_mask = query_map.valid_mask
    query_weights = _localized_group_patch_weights_from_distances(
        query_map.patch_distances,
        patch_mask,
        patch_data.radius,
    )
    patch_nodes = patch_data.padded_nodes[patch_ids]
    patch_valid_nodes = patch_data.padded_valid_mask[patch_ids]
    nq, patch_count, nmax, _ = patch_nodes.shape
    flat_nodes = patch_nodes.reshape(nq * patch_count, nmax, dim)
    flat_valid = patch_valid_nodes.reshape(nq * patch_count, nmax)
    flat_queries = jnp.repeat(xq, patch_count, axis=0)
    flat_zero_normals = jnp.zeros((flat_queries.shape[0], dim), dtype=float)
    flat_zero_coeff = jnp.zeros((flat_queries.shape[0],), dtype=float)
    flat_interp = _batched_padded_local_weights(
        flat_nodes,
        flat_valid,
        flat_queries,
        flat_zero_normals,
        flat_zero_coeff,
        flat_zero_coeff,
        patch_data.index_set,
        patch_data.recurrence_a,
        patch_data.recurrence_b,
        patch_data.stencil_props.spline_degree,
        0,
    ).reshape(nq, patch_count, nmax)
    group_node_ids = jnp.where(patch_valid_nodes, patch_data.padded_node_ids[patch_ids], 0)
    patch_values = nodal_2d[group_node_ids]
    weighted_interp = flat_interp * query_weights[:, :, None] * patch_valid_nodes * patch_mask[:, :, None]
    return jnp.sum(weighted_interp[:, :, :, None] * patch_values, axis=(1, 2))


def _localized_interp_operator_sparse(
    patch_data: LocalizedAdvectionPatchData,
    xq: jnp.ndarray,
    *,
    query_grouping: PUQueryGrouping | None = None,
) -> SparseCOOMatrix:
    xq = jnp.atleast_2d(jnp.asarray(xq, dtype=float))
    if xq.size == 0:
        return SparseCOOMatrix(
            indices=jnp.zeros((0, 2), dtype=int),
            values=jnp.zeros((0,), dtype=float),
            shape=(0, int(patch_data.nodes.shape[0])),
        )
    dim = int(patch_data.nodes.shape[1])
    query_map = _build_localized_query_map(patch_data, xq)
    patch_ids = query_map.patch_ids
    patch_mask = query_map.valid_mask
    query_weights = _localized_group_patch_weights_from_distances(
        query_map.patch_distances,
        patch_mask,
        patch_data.radius,
    )
    patch_nodes = patch_data.padded_nodes[patch_ids]
    patch_valid_nodes = patch_data.padded_valid_mask[patch_ids]
    nq, patch_count, nmax, _ = patch_nodes.shape
    flat_nodes = patch_nodes.reshape(nq * patch_count, nmax, dim)
    flat_valid = patch_valid_nodes.reshape(nq * patch_count, nmax)
    flat_queries = jnp.repeat(xq, patch_count, axis=0)
    flat_zero_normals = jnp.zeros((flat_queries.shape[0], dim), dtype=float)
    flat_zero_coeff = jnp.zeros((flat_queries.shape[0],), dtype=float)
    flat_interp = _batched_padded_local_weights(
        flat_nodes,
        flat_valid,
        flat_queries,
        flat_zero_normals,
        flat_zero_coeff,
        flat_zero_coeff,
        patch_data.index_set,
        patch_data.recurrence_a,
        patch_data.recurrence_b,
        patch_data.stencil_props.spline_degree,
        0,
    ).reshape(nq, patch_count, nmax)
    weighted_interp = flat_interp * query_weights[:, :, None] * patch_valid_nodes * patch_mask[:, :, None]
    keep_np = np.asarray(patch_valid_nodes & patch_mask[:, :, None], dtype=bool)
    group_node_ids = np.asarray(patch_data.padded_node_ids, dtype=np.int32)[np.asarray(patch_ids, dtype=np.int32)]
    rows = np.broadcast_to(np.arange(xq.shape[0], dtype=np.int32)[:, None, None], group_node_ids.shape)[keep_np]
    cols = group_node_ids[keep_np]
    values = np.asarray(weighted_interp, dtype=float)[keep_np]
    indices = np.column_stack([rows, cols]) if rows.size else np.zeros((0, 2), dtype=int)
    return SparseCOOMatrix(
        indices=jnp.asarray(indices, dtype=int),
        values=jnp.asarray(values, dtype=float),
        shape=(int(xq.shape[0]), int(patch_data.nodes.shape[0])),
    )


def _precompute_forward_nodal_data(
    patch_data: LocalizedAdvectionPatchData,
    output_nodes: jnp.ndarray,
    domain_measure: float,
) -> ForwardNodalData:
    fixed_collocation_sparse = _localized_interp_operator_sparse(
        patch_data,
        output_nodes,
        query_grouping=_build_localized_query_grouping(patch_data, output_nodes),
    )

    nodal_mass_weights = (float(domain_measure) / max(int(output_nodes.shape[0]), 1)) * jnp.ones(
        (output_nodes.shape[0],), dtype=float
    )

    local_constant_column = jnp.zeros((output_nodes.shape[0],), dtype=float).at[
        fixed_collocation_sparse.indices[:, 0]
    ].add(fixed_collocation_sparse.values)
    n = int(output_nodes.shape[0])
    mass_aug_indices = jnp.concatenate(
        [
            fixed_collocation_sparse.indices,
            jnp.column_stack([jnp.arange(n, dtype=int), jnp.full((n,), n, dtype=int)]),
            jnp.column_stack([jnp.full((n,), n, dtype=int), jnp.arange(n, dtype=int)]),
        ],
        axis=0,
    )
    mass_aug_values = jnp.concatenate([fixed_collocation_sparse.values, local_constant_column, nodal_mass_weights], axis=0)
    fixed_collocation_mass_augmented_sparse = SparseCOOMatrix(
        indices=mass_aug_indices,
        values=mass_aug_values,
        shape=(n + 1, n + 1),
    )
    return ForwardNodalData(
        local_cardinal_coeffs=tuple(),
        nodal_mass_weights=nodal_mass_weights,
        local_constant_column=local_constant_column,
        fixed_collocation=fixed_collocation_sparse,
        fixed_collocation_mass_augmented=fixed_collocation_mass_augmented_sparse,
        fixed_collocation_preconditioner=build_fixed_ilu_preconditioner(fixed_collocation_sparse),
        fixed_collocation_mass_augmented_preconditioner=build_fixed_ilu_preconditioner(
            fixed_collocation_mass_augmented_sparse
        ),
    )


@dataclass
class PUSLAdvectionSolver:
    domain: DomainDescriptor = field(default_factory=DomainDescriptor)
    xi: int = 0
    dt: float = 0.0
    output_nodes: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    output_range: tuple[int, int] = (1, 0)
    patch_data: LocalizedAdvectionPatchData | None = None
    domain_measure: float = 0.0
    boundary_condition: AdvectionBoundaryCondition = field(default_factory=AdvectionBoundaryCondition)
    constant_mode: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 1)))
    constant_mode_mass: float = 0.0
    forward_nodal_data: ForwardNodalData | None = None
    inflow_forward_fallback_reported_: bool = False
    solve_stats: dict[str, object] = field(
        default_factory=lambda: {
            "num_backward_steps": 0,
            "num_forward_steps": 0,
            "num_forward_fallbacks": 0,
            "last_forward_iterations": 0,
            "last_forward_max_residual": 0.0,
            "last_forward_max_mass_residual": 0.0,
        }
    )

    def clear_advection_boundary_condition(self) -> None:
        self.boundary_condition = AdvectionBoundaryCondition()

    def set_tangential_flow_boundary(self, normal_velocity_tolerance: float = 1.0e-10) -> None:
        self.boundary_condition = AdvectionBoundaryCondition(
            mode="tangential",
            normal_velocity_tolerance=float(normal_velocity_tolerance),
        )

    def set_periodic_boundary(self, periodic_patches: object, normal_velocity_tolerance: float = 1.0e-10) -> None:
        self.boundary_condition = AdvectionBoundaryCondition(
            mode="periodic",
            normal_velocity_tolerance=float(normal_velocity_tolerance),
            periodic_patches=periodic_patches,
        )

    def set_inflow_dirichlet_boundary(
        self,
        inflow_value: Callable[..., jnp.ndarray],
        normal_velocity_tolerance: float = 1.0e-10,
    ) -> None:
        self.boundary_condition = AdvectionBoundaryCondition(
            mode="inflow_dirichlet",
            normal_velocity_tolerance=float(normal_velocity_tolerance),
            inflow_value=inflow_value,
        )

    def get_advection_boundary_condition(self) -> AdvectionBoundaryCondition:
        return self.boundary_condition

    def returns_distributed_state(self) -> bool:
        return False

    def get_output_range(self) -> tuple[int, int]:
        return self.output_range

    def init(
        self,
        domain: DomainDescriptor,
        xi: int,
        dlt: float,
        patch_spacing_factor: float = 0.0,
        patch_radius_factor: float = 0.0,
    ) -> None:
        if self.boundary_condition.mode == "periodic":
            raise ValueError(
                "PUSL periodic advection is not yet supported robustly. "
                "The localized PU basis itself still needs true periodic patch support, not just wrapped traces."
            )
        self.domain = domain
        self.domain.build_structs()
        self.xi = xi
        self.dt = dlt
        self.output_nodes = self.domain.get_int_bdry_nodes()
        self.output_range = (0, int(self.output_nodes.shape[0]))
        self.patch_data = _build_localized_patch_data(
            self.output_nodes,
            self.xi,
            float(self.domain.get_sep_rad()),
            patch_spacing_factor,
            patch_radius_factor,
        )
        self.domain_measure = float(self._estimate_domain_measure())
        self.forward_nodal_data = _precompute_forward_nodal_data(
            self.patch_data,
            self.output_nodes,
            self.domain_measure,
        )
        self.constant_mode = self.project_samples(jnp.ones((self.output_nodes.shape[0], 1), dtype=float))
        self.constant_mode_mass = self.total_mass(self.constant_mode, 0)
        self.inflow_forward_fallback_reported_ = False
        self.reset_solve_stats()

    def get_output_nodes(self) -> jnp.ndarray:
        return self.output_nodes

    def project_initial(self, rho0: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
        return self.project_samples(jnp.asarray(rho0(self.output_nodes), dtype=float).reshape(self.output_nodes.shape[0], -1))

    def project_constant(self, value: float, n_cols: int = 1) -> jnp.ndarray:
        if self.constant_mode.size == 0:
            raise ValueError("solver must be initialized before projection")
        return jnp.asarray(value, dtype=float) * jnp.tile(self.constant_mode, (1, int(n_cols)))

    def project_samples(self, nodal_samples: jnp.ndarray) -> jnp.ndarray:
        if self.patch_data is None:
            raise ValueError("solver must be initialized before projection")
        samples = jnp.asarray(nodal_samples, dtype=float)
        squeezed = samples.ndim == 1
        samples_2d = samples[:, None] if squeezed else samples
        if int(samples_2d.shape[0]) != int(self.output_nodes.shape[0]):
            raise ValueError("project_samples expects one sample per output node")
        coeffs = jnp.zeros((self.patch_data.total_coeffs, samples_2d.shape[1]), dtype=float)
        local_values = samples_2d[self.patch_data.padded_node_ids.clip(min=0)] * self.patch_data.padded_valid_mask[:, :, None]
        padded_coeffs = _solve_padded_local_systems(self.patch_data, local_values)
        local_coeff_blocks = _compact_padded_local_coeffs(self.patch_data, padded_coeffs)
        for patch_idx, local_coeffs in enumerate(local_coeff_blocks):
            begin = self.patch_data.coeff_offsets[patch_idx]
            end = self.patch_data.coeff_offsets[patch_idx + 1]
            coeffs = coeffs.at[begin:end].set(local_coeffs)
        return _restore_coeff_shape(coeffs, squeezed)

    def evaluate_at_nodes(self, coeffs: jnp.ndarray) -> jnp.ndarray:
        return self.evaluate_at_points(coeffs, self.output_nodes)

    def evaluate_at_points(self, coeffs: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        if self.patch_data is None:
            raise ValueError("solver must be initialized before evaluation")
        coeffs_2d, squeezed = _as_coeff_matrix(coeffs)
        xq = jnp.atleast_2d(jnp.asarray(x, dtype=float))
        values = _evaluate_coeff_field_auto_from_matrix(self.patch_data, coeffs_2d, xq)
        return values[:, 0] if squeezed else values

    def backward_trace_evaluate_at_points(
        self,
        t_start: float,
        num_steps: int,
        coeffs_old: jnp.ndarray,
        arrival_points: jnp.ndarray,
        velocity: Callable[..., jnp.ndarray],
        rk: Callable[[float, jnp.ndarray, float, Callable[..., jnp.ndarray]], jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        if self.patch_data is None:
            raise ValueError("solver must be initialized before trace evaluation")
        if num_steps <= 0:
            raise ValueError("PUSLAdvectionSolver backward trace evaluation requires a positive step count")
        coeffs_old_2d, squeezed = _as_coeff_matrix(coeffs_old)
        tracer = rk4sl_step if rk is None else rk
        traced = jnp.atleast_2d(jnp.asarray(arrival_points, dtype=float))
        for step in range(int(num_steps) - 1, -1, -1):
            t_arrival = jnp.asarray(t_start, dtype=float) + (step + 1) * self.dt
            traced = tracer(t_arrival, traced, -self.dt, velocity)
        values = _evaluate_coeff_field_auto_from_matrix(self.patch_data, coeffs_old_2d, traced)
        values = self._apply_boundary_condition(values, traced, t_start + int(num_steps) * self.dt, velocity)
        return values[:, 0] if squeezed else values

    def _backward_sl_step_coeff_matrix(
        self,
        tn: float | jnp.ndarray,
        coeffs_old_2d: jnp.ndarray,
        velocity: Callable[..., jnp.ndarray],
        tracer: Callable[[float, jnp.ndarray, float, Callable[..., jnp.ndarray]], jnp.ndarray],
    ) -> jnp.ndarray:
        if self.patch_data is None:
            raise ValueError("solver must be initialized before stepping")
        t_arrival = jnp.asarray(tn, dtype=float) + self.dt
        target_mass = self._mass_vector(coeffs_old_2d) if self.boundary_condition.mode != "inflow_dirichlet" else None
        num_patches = int(self.patch_data.node_counts.shape[0])
        max_nodes = int(self.patch_data.padded_node_ids.shape[1])
        padded_nodes = self.patch_data.padded_nodes
        traced_all = tracer(
            t_arrival,
            padded_nodes.reshape(-1, padded_nodes.shape[-1]),
            -self.dt,
            velocity,
        ).reshape(num_patches, max_nodes, padded_nodes.shape[-1])
        traced_flat = traced_all.reshape(-1, traced_all.shape[-1])
        sample_all = _evaluate_coeff_field_auto_from_matrix(self.patch_data, coeffs_old_2d, traced_flat)
        sample_all = self._apply_boundary_condition(sample_all, traced_flat, t_arrival, velocity)
        sample_all = sample_all[:, None] if sample_all.ndim == 1 else sample_all
        sample_tensor = sample_all.reshape(num_patches, max_nodes, sample_all.shape[1]) * self.patch_data.padded_valid_mask[:, :, None]
        padded_coeffs = _solve_padded_local_systems(self.patch_data, sample_tensor)
        coeffs_new = _assemble_coeff_matrix_from_padded(self.patch_data, padded_coeffs)
        if target_mass is not None:
            coeffs_new = self._enforce_mass_target(coeffs_new, target_mass)
        return coeffs_new

    def backward_sl_step(
        self,
        tn: float,
        coeffs_old: jnp.ndarray,
        velocity: Callable[..., jnp.ndarray],
        rk: Callable[[float, jnp.ndarray, float, Callable[..., jnp.ndarray]], jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        if self.patch_data is None:
            raise ValueError("solver must be initialized before stepping")
        self.solve_stats["num_backward_steps"] = int(self.solve_stats["num_backward_steps"]) + 1
        coeffs_old_2d, squeezed = _as_coeff_matrix(coeffs_old)
        tracer = rk4sl_step if rk is None else rk
        coeffs_new = self._backward_sl_step_coeff_matrix(tn, coeffs_old_2d, velocity, tracer)
        return _restore_coeff_shape(coeffs_new, squeezed)

    def backward_sl_rollout(
        self,
        t0: float,
        coeffs0: jnp.ndarray,
        velocity: Callable[..., jnp.ndarray],
        nsteps: int,
        rk: Callable[[float, jnp.ndarray, float, Callable[..., jnp.ndarray]], jnp.ndarray] | None = None,
        *,
        return_history: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        if self.patch_data is None:
            raise ValueError("solver must be initialized before rollout")
        coeffs0_2d, squeezed = _as_coeff_matrix(coeffs0)
        tracer = rk4sl_step if rk is None else rk
        nsteps_i = int(nsteps)

        def _scan_step(carry, _):
            tcur, coeffs_cur = carry
            coeffs_next = self._backward_sl_step_coeff_matrix(tcur, coeffs_cur, velocity, tracer)
            return (tcur + self.dt, coeffs_next), coeffs_next

        (tfinal, coeffs_final), coeff_hist = lax.scan(
            _scan_step,
            (jnp.asarray(t0, dtype=float), coeffs0_2d),
            xs=None,
            length=nsteps_i,
        )
        del tfinal
        if return_history:
            hist_out = coeff_hist[:, :, 0] if squeezed else coeff_hist
            return _restore_coeff_shape(coeffs_final, squeezed), hist_out
        return _restore_coeff_shape(coeffs_final, squeezed)

    def forward_sl_step(
        self,
        tn: float,
        coeffs_old: jnp.ndarray,
        velocity: Callable[..., jnp.ndarray],
        rk: Callable[[float, jnp.ndarray, float, Callable[..., jnp.ndarray]], jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        if self.forward_nodal_data is None or self.patch_data is None:
            raise ValueError("solver must be initialized before stepping")
        if self.boundary_condition.mode == "inflow_dirichlet":
            self.inflow_forward_fallback_reported_ = True
            self.solve_stats["num_forward_fallbacks"] = int(self.solve_stats["num_forward_fallbacks"]) + 1
            return self.backward_sl_step(tn, coeffs_old, velocity, rk)
        self.solve_stats["num_forward_steps"] = int(self.solve_stats["num_forward_steps"]) + 1
        coeffs_old_2d, squeezed = _as_coeff_matrix(coeffs_old)
        tracer = rk4sl_step if rk is None else rk
        preserve_mass = self.boundary_condition.mode != "inflow_dirichlet"
        target_mass = self._mass_vector(coeffs_old_2d) if preserve_mass else None

        nodal_old = self.evaluate_at_nodes(coeffs_old_2d)
        nodal_old = nodal_old[:, None] if nodal_old.ndim == 1 else nodal_old
        rhs_local = nodal_old
        nodal_current = nodal_old
        traced_points = tracer(tn, self.output_nodes, self.dt, velocity)
        traced_points = jnp.asarray(traced_points, dtype=float)

        if self.boundary_condition.mode == "inflow_dirichlet":
            adjusted = self._apply_boundary_condition(rhs_local, traced_points, tn + self.dt, velocity)
            rhs_local = adjusted[:, None] if adjusted.ndim == 1 else adjusted

        forward_iterations = 0
        last_resid_norm = 0.0
        last_mass_resid_norm = 0.0
        for iteration in range(4):
            forward_values = _evaluate_nodal_field(self.patch_data, self.forward_nodal_data, traced_points, nodal_current)
            residual = rhs_local - forward_values
            resid_norm = float(jnp.max(jnp.abs(residual))) if residual.size else 0.0
            mass_resid_norm = 0.0
            if preserve_mass:
                current_mass = self.forward_nodal_data.nodal_mass_weights @ nodal_current
                mass_resid = target_mass - current_mass
                mass_resid_norm = float(jnp.max(jnp.abs(mass_resid)))
            last_resid_norm = resid_norm
            last_mass_resid_norm = mass_resid_norm
            forward_iterations = iteration + 1
            if resid_norm <= 1.0e-10 and (not preserve_mass or mass_resid_norm <= 1.0e-12):
                break

            if preserve_mass:
                rhs_aug = jnp.zeros((rhs_local.shape[0] + 1, rhs_local.shape[1]), dtype=float)
                rhs_aug = rhs_aug.at[: rhs_local.shape[0]].set(residual)
                rhs_aug = rhs_aug.at[-1].set(mass_resid)
                delta_aug = jnp.column_stack(
                    [
                        solve_sparse_system_gmres(
                            self.forward_nodal_data.fixed_collocation_mass_augmented,
                            rhs_aug[:, col],
                            preconditioner=self.forward_nodal_data.fixed_collocation_mass_augmented_preconditioner,
                        )
                        for col in range(rhs_aug.shape[1])
                    ]
                )
                delta = delta_aug[: rhs_local.shape[0]]
            else:
                delta = jnp.column_stack(
                    [
                        solve_sparse_system_gmres(
                            self.forward_nodal_data.fixed_collocation,
                            residual[:, col],
                            preconditioner=self.forward_nodal_data.fixed_collocation_preconditioner,
                        )
                        for col in range(residual.shape[1])
                    ]
                )
            nodal_current = nodal_current + delta

        coeffs_new = self.project_samples(nodal_current)
        coeffs_new_2d, _ = _as_coeff_matrix(coeffs_new)
        if preserve_mass:
            coeffs_new_2d = self._enforce_mass_target(coeffs_new_2d, target_mass)
        self.solve_stats["last_forward_iterations"] = forward_iterations
        self.solve_stats["last_forward_max_residual"] = last_resid_norm
        self.solve_stats["last_forward_max_mass_residual"] = last_mass_resid_norm
        return _restore_coeff_shape(coeffs_new_2d, squeezed)

    def set_step_size(self, dlt: float) -> None:
        self.dt = dlt

    def reset_solve_stats(self) -> None:
        self.solve_stats = {
            "num_backward_steps": 0,
            "num_forward_steps": 0,
            "num_forward_fallbacks": 0,
            "last_forward_iterations": 0,
            "last_forward_max_residual": 0.0,
            "last_forward_max_mass_residual": 0.0,
        }

    def get_solve_stats(self) -> dict[str, object]:
        return dict(self.solve_stats)

    def get_num_patches(self) -> int:
        return 0 if self.patch_data is None else int(self.patch_data.centers.shape[0])

    def get_num_dofs(self) -> int:
        return 0 if self.patch_data is None else int(self.patch_data.total_coeffs)

    def get_patch_radius(self) -> float:
        return 0.0 if self.patch_data is None else float(self.patch_data.radius)

    def get_patch_spacing(self) -> float:
        return 0.0 if self.patch_data is None else float(self.patch_data.spacing)

    def total_mass(self, coeffs: jnp.ndarray, col: int = 0) -> float:
        values = self.evaluate_at_nodes(coeffs)
        values_2d = values[:, None] if jnp.asarray(values).ndim == 1 else jnp.asarray(values, dtype=float)
        data = values_2d[:, col]
        return float(self.domain_measure * jnp.mean(data))

    def get_domain_measure(self) -> float:
        return float(self.domain_measure)

    def _mass_vector(self, coeffs: jnp.ndarray) -> jnp.ndarray:
        values = self.evaluate_at_nodes(coeffs)
        values_2d = values[:, None] if values.ndim == 1 else values
        return self.domain_measure * jnp.mean(values_2d, axis=0)

    def _enforce_mass_target(self, coeffs: jnp.ndarray, target_mass: jnp.ndarray) -> jnp.ndarray:
        denom = max(abs(self.constant_mode_mass), 1.0e-14)
        current_mass = self._mass_vector(coeffs)
        alpha = (target_mass - current_mass) / denom
        return coeffs + self.constant_mode * alpha[None, :]

    def _apply_boundary_condition(
        self,
        values: jnp.ndarray,
        xdep: jnp.ndarray,
        tnext: float,
        velocity: Callable[..., jnp.ndarray],
    ) -> jnp.ndarray:
        bc = self.boundary_condition
        del velocity
        if bc.mode in {"unspecified", "tangential"}:
            return values
        if bc.mode == "inflow_dirichlet":
            phi = self.domain.get_outer_level_set().evaluate(xdep)
            outside = phi > 0
            if bc.inflow_value is None:
                raise ValueError("inflow_dirichlet boundary requires inflow_value")
            inflow_all = jnp.asarray(bc.inflow_value(tnext, xdep), dtype=float)
            out = jnp.asarray(values, dtype=float)
            out = out[:, None] if out.ndim == 1 else out
            inflow_all = inflow_all[:, None] if inflow_all.ndim == 1 else inflow_all
            out = jnp.where(outside[:, None], inflow_all, out)
            return out[:, 0] if values.ndim == 1 else out
        if bc.mode == "periodic":
            raise ValueError("Periodic PU-SL transport is not yet supported in kernelpack-jax")
        return values

    def _estimate_domain_measure(self) -> float:
        phi = self.domain.get_outer_level_set()
        if phi is None:
            return float(self.domain.get_num_interior_nodes() * self.domain.get_sep_rad() ** self.domain.get_dim())
        xall = self.domain.get_int_bdry_nodes()
        xmin = jnp.min(xall, axis=0) - self.domain.get_sep_rad()
        xmax = jnp.max(xall, axis=0) + self.domain.get_sep_rad()
        dim = int(xall.shape[1])
        ngrid = 32
        if dim == 2:
            xs = jnp.linspace(float(xmin[0]), float(xmax[0]), ngrid)
            ys = jnp.linspace(float(xmin[1]), float(xmax[1]), ngrid)
            xg, yg = jnp.meshgrid(xs, ys, indexing="ij")
            pts = jnp.column_stack([xg.reshape(-1), yg.reshape(-1)])
        else:
            raise ValueError("domain measure estimate currently supports 2D only")
        inside = phi.evaluate(pts) <= 0
        box_measure = float(jnp.prod(xmax - xmin))
        return float(box_measure * jnp.mean(inside.astype(float)))
