from __future__ import annotations

from dataclasses import dataclass
from math import comb

from functools import partial

from jax import jit, vmap
import jax.numpy as jnp
import numpy as np

from kernelpack.accelerators import WarpUnavailableError, warp_available, warp_exact_ball
from kernelpack.domain import DomainDescriptor
from kernelpack.geometry import distance_matrix
from kernelpack.rbffd.core import _batched_basis_eval, _build_legendre_basis_data
from kernelpack.rbffd import OpProperties, RBFStencil, StencilProperties
from ._common import SparseCOOMatrix


@dataclass(frozen=True)
class PUPatchData:
    centers: jnp.ndarray
    radius: float
    spacing: float
    min_patch_nodes: int
    node_ids: tuple[jnp.ndarray, ...]
    node_counts: jnp.ndarray
    padded_node_ids: jnp.ndarray
    padded_nodes: jnp.ndarray
    padded_valid_mask: jnp.ndarray
    stencil_props: StencilProperties


@dataclass(frozen=True)
class PUQueryGroup:
    query_ids: jnp.ndarray
    patch_ids: jnp.ndarray


@dataclass(frozen=True)
class PUQueryGrouping:
    groups: tuple[PUQueryGroup, ...]


def _resolve_pu_backend(backend: str) -> str:
    backend_l = str(backend).strip().lower()
    if backend_l == "auto":
        return "warp" if warp_available() else "cpu"
    if backend_l not in {"cpu", "warp"}:
        raise ValueError("backend must be 'cpu', 'warp', or 'auto'")
    return backend_l


def build_pu_query_grouping(patch_data: PUPatchData, xq: jnp.ndarray, *, backend: str = "auto") -> PUQueryGrouping:
    xq = jnp.atleast_2d(jnp.asarray(xq, dtype=float))
    patch_ids_per_query = _query_patch_ids(patch_data, xq, patch_data.radius, backend=backend)
    return _group_queries_by_patch_count(patch_ids_per_query)


@jit
def pu_patch_weight(r: jnp.ndarray) -> jnp.ndarray:
    r = jnp.asarray(r, dtype=float)
    t = jnp.clip(1.0 - r, 0.0, None)
    return jnp.where(r < 1.0, t**8 * (32.0 * r**3 + 25.0 * r**2 + 8.0 * r + 1.0), 0.0)


@jit
def _normalize_patch_weights(alpha: jnp.ndarray) -> jnp.ndarray:
    alpha_sum = alpha.sum()
    safe_alpha = jnp.where(alpha_sum <= 1.0e-14, jnp.ones_like(alpha), alpha)
    return safe_alpha / safe_alpha.sum()


@jit
def _group_patch_weights(centers: jnp.ndarray, xq_group: jnp.ndarray, patch_ids: jnp.ndarray, radius: float) -> jnp.ndarray:
    selected_centers = centers[patch_ids]
    d = jnp.linalg.norm(selected_centers - xq_group[:, None, :], axis=2)
    raw = pu_patch_weight(d / radius)
    return vmap(_normalize_patch_weights)(raw)


def _all_patch_weights(patch_data: PUPatchData, xq: jnp.ndarray) -> jnp.ndarray:
    xq = jnp.atleast_2d(jnp.asarray(xq, dtype=float))
    d = jnp.linalg.norm(xq[:, None, :] - patch_data.centers[None, :, :], axis=2)
    raw = pu_patch_weight(d / patch_data.radius)
    denom = raw.sum(axis=1, keepdims=True)
    nearest = jnp.argmin(d, axis=1)
    fallback = jnp.eye(patch_data.centers.shape[0], dtype=float)[nearest]
    return jnp.where(denom > 1.0e-14, raw / denom, fallback)


@partial(jit, static_argnums=(0, 1, 2, 3))
def _operator_stencil_properties(dim: int, xi: int, theta: int, boundary_ell: int) -> tuple[int, int, int]:
    ell = boundary_ell if theta == 1 else max(xi + theta - 1, 2)
    npoly = int(comb(dim + ell, dim))
    spline_degree = max(5, ell)
    if spline_degree % 2 == 0:
        spline_degree -= 1
    return ell, npoly, spline_degree


def choose_patch_spacing(h: float, patch_spacing_factor: float = 0.0) -> float:
    return float(patch_spacing_factor * h if patch_spacing_factor > 0 else 2.0 * h)


def choose_patch_radius(h: float, patch_radius_factor: float = 0.0) -> float:
    return float(patch_radius_factor * h if patch_radius_factor > 0 else 3.0 * h)


def choose_minimum_patch_nodes(dim: int, xi: int) -> int:
    ell = max(xi + 1, 2)
    npoly = int(comb(dim + ell, dim))
    return 2 * npoly + 1


def build_patch_stencil_properties(dim: int, xi: int) -> StencilProperties:
    ell = max(xi + 1, 2)
    npoly = int(comb(dim + ell, dim))
    spline_degree = max(5, ell)
    if spline_degree % 2 == 0:
        spline_degree -= 1
    return StencilProperties(
        dim=dim,
        ell=ell,
        npoly=npoly,
        spline_degree=spline_degree,
        tree_mode="all",
        point_set="interior_boundary",
    )


def choose_patch_centers(x: jnp.ndarray, spacing: float, *, backend: str = "auto") -> jnp.ndarray:
    x = jnp.asarray(x, dtype=float)
    if x.size == 0:
        return jnp.zeros((0, 0), dtype=float)
    x_np = np.asarray(x, dtype=float)
    selected_backend = _resolve_pu_backend(backend)
    centers: list[np.ndarray] = []
    remaining = np.ones((x_np.shape[0],), dtype=bool)
    for i in range(x_np.shape[0]):
        if not remaining[i]:
            continue
        xi = x_np[i]
        centers.append(xi)
        if selected_backend == "warp":
            try:
                ids_rows, _ = warp_exact_ball(xi[None, :], x_np, float(spacing))
                remaining[np.asarray(ids_rows[0], dtype=int)] = False
                continue
            except WarpUnavailableError:
                selected_backend = "cpu"
        d = np.linalg.norm(x_np - xi, axis=1)
        remaining = np.logical_and(remaining, d > spacing)
    return jnp.asarray(np.stack(centers, axis=0), dtype=float) if centers else jnp.zeros((0, x.shape[1]), dtype=float)


def build_patch_node_ids(
    domain: DomainDescriptor,
    centers: jnp.ndarray,
    radius: float,
    min_patch_nodes: int,
    *,
    backend: str = "auto",
) -> tuple[jnp.ndarray, ...]:
    del backend
    if centers.size == 0 or domain.get_num_total_nodes() == 0:
        return tuple()
    num_all = int(domain.get_num_total_nodes())
    kmin = min(int(min_patch_nodes), num_all)
    ids, _dist, mask = domain.query_ball_padded("all", centers, radius, k_min=kmin)
    ids_np = np.asarray(ids, dtype=int)
    mask_np = np.asarray(mask, dtype=bool)
    return tuple(jnp.asarray(ids_np[p, mask_np[p]], dtype=int) for p in range(ids_np.shape[0]))


def build_padded_patch_arrays(xf: jnp.ndarray, node_ids: tuple[jnp.ndarray, ...]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if not node_ids:
        dim = int(xf.shape[1]) if xf.ndim == 2 and xf.size else 0
        return (
            jnp.zeros((0,), dtype=int),
            jnp.zeros((0, 0), dtype=int),
            jnp.zeros((0, 0, dim), dtype=float),
        )
    xf_np = np.asarray(xf, dtype=float)
    counts_np = np.asarray([int(ids.shape[0]) for ids in node_ids], dtype=int)
    max_nodes = int(counts_np.max(initial=0))
    padded_ids = np.full((len(node_ids), max_nodes), -1, dtype=int)
    padded_nodes = np.zeros((len(node_ids), max_nodes, xf.shape[1]), dtype=float)
    for p, ids in enumerate(node_ids):
        nloc = int(ids.shape[0])
        if nloc == 0:
            continue
        ids_np = np.asarray(ids, dtype=int)
        padded_ids[p, :nloc] = ids_np
        padded_nodes[p, :nloc, :] = xf_np[ids_np]
    return jnp.asarray(counts_np, dtype=int), jnp.asarray(padded_ids, dtype=int), jnp.asarray(padded_nodes, dtype=float)


def pu_patch_geometry(
    domain: DomainDescriptor,
    xi: int,
    patch_spacing_factor: float = 0.0,
    patch_radius_factor: float = 0.0,
    *,
    backend: str = "auto",
) -> PUPatchData:
    selected_backend = _resolve_pu_backend(backend)
    domain.build_structs()
    xf = domain.get_all_nodes()
    h = domain.get_sep_rad()
    spacing = choose_patch_spacing(h, patch_spacing_factor)
    radius = choose_patch_radius(h, patch_radius_factor)
    min_nodes = choose_minimum_patch_nodes(xf.shape[1], xi)
    centers = choose_patch_centers(xf, spacing, backend=selected_backend)
    node_ids = build_patch_node_ids(domain, centers, radius, min_nodes, backend=selected_backend)
    node_counts, padded_node_ids, padded_nodes = build_padded_patch_arrays(xf, node_ids)
    padded_valid_mask = padded_node_ids >= 0
    sp = build_patch_stencil_properties(xf.shape[1], xi)
    return PUPatchData(
        centers=centers,
        radius=radius,
        spacing=spacing,
        min_patch_nodes=min_nodes,
        node_ids=node_ids,
        node_counts=node_counts,
        padded_node_ids=padded_node_ids,
        padded_nodes=padded_nodes,
        padded_valid_mask=padded_valid_mask,
        stencil_props=sp,
    )


def _query_patch_ids(patch_data: PUPatchData, xq: jnp.ndarray, radius: float, *, backend: str = "auto") -> list[jnp.ndarray]:
    xq = jnp.atleast_2d(jnp.asarray(xq, dtype=float))
    centers = patch_data.centers
    if centers.size == 0:
        return [jnp.zeros((0,), dtype=int) for _ in range(xq.shape[0])]
    selected_backend = _resolve_pu_backend(backend)
    if selected_backend == "warp":
        try:
            xq_np = np.asarray(xq, dtype=float)
            centers_np = np.asarray(centers, dtype=float)
            ids_rows, _ = warp_exact_ball(xq_np, centers_np, float(radius))
            patch_ids: list[jnp.ndarray] = []
            for q, ids in enumerate(ids_rows):
                ids_np = np.asarray(ids, dtype=int)
                if ids_np.size == 0:
                    d = np.linalg.norm(xq_np[q : q + 1] - centers_np, axis=1)
                    ids_np = np.asarray([int(np.argmin(d))], dtype=int)
                patch_ids.append(jnp.asarray(ids_np, dtype=int))
            return patch_ids
        except WarpUnavailableError:
            selected_backend = "cpu"
    xq_np = np.asarray(xq, dtype=float)
    centers_np = np.asarray(centers, dtype=float)
    d = np.linalg.norm(xq_np[:, None, :] - centers_np[None, :, :], axis=2)
    patch_ids: list[jnp.ndarray] = []
    for q in range(xq.shape[0]):
        ids = np.flatnonzero(d[q] < radius)
        if ids.size == 0:
            ids = np.asarray([int(np.argmin(d[q]))], dtype=int)
        patch_ids.append(jnp.asarray(ids, dtype=int))
    return patch_ids


def _group_queries_by_patch_count(patch_ids_per_query: list[jnp.ndarray]) -> PUQueryGrouping:
    count_to_queries: dict[int, list[int]] = {}
    for q, patch_ids in enumerate(patch_ids_per_query):
        count_to_queries.setdefault(int(patch_ids.size), []).append(q)
    groups: list[PUQueryGroup] = []
    for count in sorted(count_to_queries):
        query_ids = jnp.asarray(count_to_queries[count], dtype=int)
        patch_rows = jnp.stack([patch_ids_per_query[int(q)] for q in query_ids.tolist()], axis=0)
        groups.append(PUQueryGroup(query_ids=query_ids, patch_ids=patch_rows))
    return PUQueryGrouping(groups=tuple(groups))


def _local_operator_weights(
    stencil: RBFStencil,
    xq: jnp.ndarray,
    sp: StencilProperties,
    op_name: str,
    normal: jnp.ndarray | None = None,
    neu_coeff: float = 0.0,
    dir_coeff: float = 0.0,
) -> jnp.ndarray:
    xloc = stencil.x_stencil
    xc = (xq - stencil.xm) / stencil.width
    r = distance_matrix(xq[None, :], xloc)
    op = OpProperties(nosolve=False, selectdim=0)
    name = str(op_name).lower()
    if name in {"interp", "interpolation"}:
        b = stencil.interp_op(sp, op, r, xq[None, :], xloc, xc[None, :], stencil.xc)
    elif name in {"lap", "laplacian"}:
        b = stencil.lap_op(sp, op, r, xq[None, :], xloc, xc[None, :], stencil.xc)
    elif name in {"bc", "boundary"}:
        if normal is None:
            raise ValueError("boundary operator requires normals")
        b = stencil.bc_op(sp, op, neu_coeff, dir_coeff, r, xq[None, :], xloc, xc[None, :], stencil.xc, normal[None, :])
    else:
        raise ValueError(f"unknown PU operator {op_name}")
    w = stencil.stable_solve(stencil.solve_lhs, b)
    return w[: xloc.shape[0], 0]


@jit
def _masked_pairwise_distance_matrix(x: jnp.ndarray, valid_mask: jnp.ndarray) -> jnp.ndarray:
    diff = x[:, :, None, :] - x[:, None, :, :]
    d = jnp.linalg.norm(diff, axis=3)
    valid_pairs = valid_mask[:, :, None] & valid_mask[:, None, :]
    return jnp.where(valid_pairs, d, 0.0)


@partial(jit, static_argnums=(5,))
def _prepare_padded_rbf_system(
    x_stencils: jnp.ndarray,
    valid_mask: jnp.ndarray,
    index_set: jnp.ndarray,
    recurrence_a: jnp.ndarray,
    recurrence_b: jnp.ndarray,
    spline_degree: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    counts = jnp.maximum(valid_mask.sum(axis=1), 1)
    xm = jnp.sum(x_stencils * valid_mask[:, :, None], axis=1) / counts[:, None]
    r = _masked_pairwise_distance_matrix(x_stencils, valid_mask)
    width = jnp.maximum(r.max(axis=(1, 2)), 1.0)
    xc = (x_stencils - xm[:, None, :]) / width[:, None, None]
    dim = x_stencils.shape[2]
    p = _batched_basis_eval(index_set, recurrence_a, recurrence_b, xc, jnp.zeros((1, dim), dtype=int))
    k = RBFStencil.phs_rbf(r, spline_degree)
    nmax = x_stencils.shape[1]
    npoly = index_set.shape[0]
    lhs = jnp.zeros((x_stencils.shape[0], nmax + npoly, nmax + npoly), dtype=float)
    lhs = lhs.at[:, :nmax, :nmax].set(k)
    lhs = lhs.at[:, :nmax, nmax:].set(p * valid_mask[:, :, None])
    lhs = lhs.at[:, nmax:, :nmax].set(jnp.swapaxes(p * valid_mask[:, :, None], 1, 2))
    invalid = ~valid_mask
    lhs = lhs.at[:, :nmax, :nmax].add(vmap(jnp.diag)(invalid.astype(float)))
    return lhs, width, xm


@partial(jit, static_argnums=(9, 10))
def _batched_padded_local_weights(
    x_stencils: jnp.ndarray,
    valid_mask: jnp.ndarray,
    queries: jnp.ndarray,
    normals: jnp.ndarray,
    neu_coeff: jnp.ndarray,
    dir_coeff: jnp.ndarray,
    index_set: jnp.ndarray,
    recurrence_a: jnp.ndarray,
    recurrence_b: jnp.ndarray,
    spline_degree: int,
    theta: int,
) -> jnp.ndarray:
    lhs, width, xm = _prepare_padded_rbf_system(
        x_stencils,
        valid_mask,
        index_set,
        recurrence_a,
        recurrence_b,
        spline_degree,
    )
    dim = x_stencils.shape[2]
    nmax = x_stencils.shape[1]
    qn = (queries - xm) / width[:, None]
    r_rhs = jnp.linalg.norm(queries[:, None, :] - x_stencils, axis=2)
    if theta == 0:
        rhs_top = RBFStencil.phs_rbf(r_rhs, spline_degree) * valid_mask
        rhs_poly = _batched_basis_eval(index_set, recurrence_a, recurrence_b, qn[:, None, :], jnp.zeros((1, dim), dtype=int))[:, 0, :]
    elif theta == 2:
        rhs_top = RBFStencil.phs_lap(r_rhs, spline_degree, dim) * valid_mask
        rhs_poly = (
            jnp.sum(
                _batched_basis_eval(
                    index_set,
                    recurrence_a,
                    recurrence_b,
                    qn[:, None, :],
                    2 * jnp.eye(dim, dtype=int),
                )[:, 0, :, :],
                axis=2,
            )
            / (width[:, None] ** 2)
        )
    elif theta == 1:
        diff = queries[:, None, :] - x_stencils
        radial = RBFStencil.phs_dr_over_r(r_rhs, spline_degree)[..., None]
        grad_rbf = radial * diff
        rhs_top = neu_coeff[:, None] * jnp.sum(grad_rbf * normals[:, None, :], axis=2) + dir_coeff[:, None] * RBFStencil.phs_rbf(r_rhs, spline_degree)
        rhs_top = rhs_top * valid_mask
        grad_poly = _batched_basis_eval(
            index_set,
            recurrence_a,
            recurrence_b,
            qn[:, None, :],
            jnp.eye(dim, dtype=int),
        )[:, 0, :, :] / width[:, None, None]
        rhs_poly = neu_coeff[:, None] * jnp.sum(grad_poly * normals[:, None, :], axis=2) + dir_coeff[:, None] * _batched_basis_eval(
            index_set,
            recurrence_a,
            recurrence_b,
            qn[:, None, :],
            jnp.zeros((1, dim), dtype=int),
        )[:, 0, :]
    else:
        raise ValueError(f"unsupported theta {theta}")
    rhs = jnp.concatenate([rhs_top[:, :, None], rhs_poly[:, :, None]], axis=1)
    w = jnp.linalg.solve(lhs, rhs)[:, :nmax, 0]
    return w * valid_mask


def pu_localized_operator(
    domain: DomainDescriptor,
    patch_data: PUPatchData,
    xi: int,
    xq: jnp.ndarray,
    op_name: str,
    *,
    normals: jnp.ndarray | None = None,
    neu_coeff: jnp.ndarray | None = None,
    dir_coeff: jnp.ndarray | None = None,
    query_grouping: PUQueryGrouping | None = None,
) -> jnp.ndarray:
    del query_grouping
    xq = jnp.atleast_2d(jnp.asarray(xq, dtype=float))
    if xq.size == 0:
        return jnp.zeros((0, domain.get_num_total_nodes()), dtype=float)
    xnodes = domain.get_all_nodes()
    num_all = xnodes.shape[0]
    dim = xnodes.shape[1]
    normals = jnp.zeros((xq.shape[0], dim), dtype=float) if normals is None else jnp.asarray(normals, dtype=float)
    neu_coeff = jnp.zeros((xq.shape[0],), dtype=float) if neu_coeff is None else jnp.asarray(neu_coeff, dtype=float).reshape(-1)
    dir_coeff = jnp.zeros((xq.shape[0],), dtype=float) if dir_coeff is None else jnp.asarray(dir_coeff, dtype=float).reshape(-1)

    theta_map = {"interp": 0, "interpolation": 0, "lap": 2, "laplacian": 2, "bc": 1, "boundary": 1}
    key = str(op_name).lower()
    if key not in theta_map:
        raise ValueError(f"unknown PU operator {op_name}")
    theta = theta_map[key]
    dim = xnodes.shape[1]
    ell, npoly, spline_degree = _operator_stencil_properties(dim, xi, theta, patch_data.stencil_props.ell)
    ell = int(ell)
    npoly = int(npoly)
    spline_degree = int(spline_degree)
    index_set, recurrence_a, recurrence_b = _build_legendre_basis_data(dim, ell)

    npatches = int(patch_data.padded_nodes.shape[0])
    nmax = int(patch_data.padded_nodes.shape[1])
    patch_weights = _all_patch_weights(patch_data, xq)
    patch_nodes = jnp.broadcast_to(patch_data.padded_nodes[None, :, :, :], (xq.shape[0], npatches, nmax, dim))
    patch_valid = jnp.broadcast_to(patch_data.padded_valid_mask[None, :, :], (xq.shape[0], npatches, nmax))
    flat_nodes = patch_nodes.reshape(xq.shape[0] * npatches, nmax, dim)
    flat_valid = patch_valid.reshape(xq.shape[0] * npatches, nmax)
    flat_queries = jnp.repeat(xq, npatches, axis=0)
    flat_normals = jnp.repeat(normals, npatches, axis=0)
    flat_neu = jnp.repeat(neu_coeff, npatches, axis=0)
    flat_dir = jnp.repeat(dir_coeff, npatches, axis=0)
    flat_local_weights = _batched_padded_local_weights(
        flat_nodes,
        flat_valid,
        flat_queries,
        flat_normals,
        flat_neu,
        flat_dir,
        index_set,
        recurrence_a,
        recurrence_b,
        spline_degree,
        theta,
    ).reshape(xq.shape[0], npatches, nmax)
    weighted_local = flat_local_weights * patch_weights[:, :, None] * patch_valid
    node_ids = jnp.broadcast_to(patch_data.padded_node_ids[None, :, :], (xq.shape[0], npatches, nmax))
    scatter_rows = jnp.broadcast_to(jnp.arange(xq.shape[0], dtype=int)[:, None, None], node_ids.shape).reshape(-1)
    scatter_cols = jnp.where(patch_valid, node_ids, 0).reshape(-1)
    scatter_vals = weighted_local.reshape(-1)
    return jnp.zeros((xq.shape[0], num_all), dtype=float).at[scatter_rows, scatter_cols].add(scatter_vals)


def pu_localized_operator_sparse(
    domain: DomainDescriptor,
    patch_data: PUPatchData,
    xi: int,
    xq: jnp.ndarray,
    op_name: str,
    *,
    normals: jnp.ndarray | None = None,
    neu_coeff: jnp.ndarray | None = None,
    dir_coeff: jnp.ndarray | None = None,
    query_grouping: PUQueryGrouping | None = None,
    value_tol: float = 0.0,
) -> SparseCOOMatrix:
    del query_grouping
    xq = jnp.atleast_2d(jnp.asarray(xq, dtype=float))
    if xq.size == 0:
        return SparseCOOMatrix(
            indices=jnp.zeros((0, 2), dtype=int),
            values=jnp.zeros((0,), dtype=float),
            shape=(0, int(domain.get_num_total_nodes())),
        )
    xnodes = domain.get_all_nodes()
    num_all = int(xnodes.shape[0])
    dim = int(xnodes.shape[1])
    normals = jnp.zeros((xq.shape[0], dim), dtype=float) if normals is None else jnp.asarray(normals, dtype=float)
    neu_coeff = jnp.zeros((xq.shape[0],), dtype=float) if neu_coeff is None else jnp.asarray(neu_coeff, dtype=float).reshape(-1)
    dir_coeff = jnp.zeros((xq.shape[0],), dtype=float) if dir_coeff is None else jnp.asarray(dir_coeff, dtype=float).reshape(-1)

    theta_map = {"interp": 0, "interpolation": 0, "lap": 2, "laplacian": 2, "bc": 1, "boundary": 1}
    key = str(op_name).lower()
    if key not in theta_map:
        raise ValueError(f"unknown PU operator {op_name}")
    theta = theta_map[key]
    ell, _, spline_degree = _operator_stencil_properties(dim, xi, theta, patch_data.stencil_props.ell)
    ell = int(ell)
    spline_degree = int(spline_degree)
    index_set, recurrence_a, recurrence_b = _build_legendre_basis_data(dim, ell)

    npatches = int(patch_data.padded_nodes.shape[0])
    nmax = int(patch_data.padded_nodes.shape[1])
    patch_weights = _all_patch_weights(patch_data, xq)
    patch_nodes = jnp.broadcast_to(patch_data.padded_nodes[None, :, :, :], (xq.shape[0], npatches, nmax, dim))
    patch_valid = jnp.broadcast_to(patch_data.padded_valid_mask[None, :, :], (xq.shape[0], npatches, nmax))
    flat_nodes = patch_nodes.reshape(xq.shape[0] * npatches, nmax, dim)
    flat_valid = patch_valid.reshape(xq.shape[0] * npatches, nmax)
    flat_queries = jnp.repeat(xq, npatches, axis=0)
    flat_normals = jnp.repeat(normals, npatches, axis=0)
    flat_neu = jnp.repeat(neu_coeff, npatches, axis=0)
    flat_dir = jnp.repeat(dir_coeff, npatches, axis=0)
    flat_local_weights = _batched_padded_local_weights(
        flat_nodes,
        flat_valid,
        flat_queries,
        flat_normals,
        flat_neu,
        flat_dir,
        index_set,
        recurrence_a,
        recurrence_b,
        spline_degree,
        theta,
    ).reshape(xq.shape[0], npatches, nmax)
    weighted_local = flat_local_weights * patch_weights[:, :, None] * patch_valid
    keep = patch_valid
    if value_tol > 0.0:
        keep = keep & (jnp.abs(weighted_local) > value_tol)
    keep_np = np.asarray(keep, dtype=bool)
    node_ids = np.broadcast_to(np.asarray(patch_data.padded_node_ids, dtype=int)[None, :, :], keep_np.shape)
    rows = np.broadcast_to(np.arange(xq.shape[0], dtype=int)[:, None, None], keep_np.shape)[keep_np]
    cols = node_ids[keep_np]
    values = np.asarray(weighted_local, dtype=float)[keep_np]
    indices = np.column_stack([rows, cols]) if rows.size else np.zeros((0, 2), dtype=int)
    return SparseCOOMatrix(
        indices=jnp.asarray(indices, dtype=int),
        values=jnp.asarray(values, dtype=float),
        shape=(int(xq.shape[0]), num_all),
    )


def pu_localized_evaluate(
    domain: DomainDescriptor,
    patch_data: PUPatchData,
    xi: int,
    coeffs: jnp.ndarray,
    xq: jnp.ndarray,
    *,
    query_grouping: PUQueryGrouping | None = None,
) -> jnp.ndarray:
    del xi, query_grouping
    xq = jnp.atleast_2d(jnp.asarray(xq, dtype=float))
    coeffs = jnp.asarray(coeffs, dtype=float)
    if xq.size == 0:
        return jnp.zeros((0, coeffs.shape[1] if coeffs.ndim > 1 else 1), dtype=float)
    coeffs_2d = coeffs if coeffs.ndim == 2 else coeffs[:, None]
    index_set, recurrence_a, recurrence_b = _build_legendre_basis_data(xq.shape[1], patch_data.stencil_props.ell)
    dim = int(xq.shape[1])
    npatches = int(patch_data.padded_nodes.shape[0])
    nmax = int(patch_data.padded_nodes.shape[1])
    patch_weights = _all_patch_weights(patch_data, xq)
    patch_nodes = jnp.broadcast_to(patch_data.padded_nodes[None, :, :, :], (xq.shape[0], npatches, nmax, dim))
    patch_valid = jnp.broadcast_to(patch_data.padded_valid_mask[None, :, :], (xq.shape[0], npatches, nmax))
    flat_nodes = patch_nodes.reshape(xq.shape[0] * npatches, nmax, dim)
    flat_valid = patch_valid.reshape(xq.shape[0] * npatches, nmax)
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
        index_set,
        recurrence_a,
        recurrence_b,
        patch_data.stencil_props.spline_degree,
        0,
    ).reshape(xq.shape[0], npatches, nmax)
    node_ids = jnp.broadcast_to(patch_data.padded_node_ids[None, :, :], (xq.shape[0], npatches, nmax))
    coeff_ids = jnp.where(patch_valid, node_ids, 0)
    patch_coeffs = coeffs_2d[coeff_ids]
    weighted_interp = flat_interp * patch_weights[:, :, None] * patch_valid
    values = jnp.sum(weighted_interp[:, :, :, None] * patch_coeffs, axis=(1, 2))
    if coeffs.ndim == 1:
        return values[:, 0]
    return values
