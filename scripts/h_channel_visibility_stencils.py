from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from kernelpack.domain import DomainDescriptor
from kernelpack.rbffd import FrozenStencilGraph
from kernelpack.solvers import ReactionDiffusionSolver
from kernelpack.solvers._common import (
    FixedDiscretization,
    assemble_operator,
    assemble_operator_sparse,
    build_domain_state,
    build_stencil_properties,
)


H_CHANNEL_CURVE_AMPLITUDE = 0.06


@dataclass(frozen=True)
class VisibilityFilterStats:
    lap_rejected_edges: int
    lap_fallback_rows: int
    boundary_rejected_edges: int
    boundary_fallback_rows: int


def h_channel_bounds(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = np.clip(x / 2.0, 0.0, 1.0)
    bottom = -H_CHANNEL_CURVE_AMPLITUDE * np.sin(np.pi * u)
    top = 1.0 + H_CHANNEL_CURVE_AMPLITUDE * np.sin(np.pi * u)
    return bottom, top


def h_channel_level_set(points: np.ndarray) -> np.ndarray:
    """Positive inside the saved curved H-channel, negative outside.

    The saved domains do not store the generating level-set function, so this is
    the matching analytic reconstruction used only by the experimental stencil
    filter. The H-channel is the union of the lower channel, upper channel, and
    central throat, clipped by the curved outer top/bottom boundaries.
    """

    pts = np.asarray(points, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]
    bottom, top = h_channel_bounds(x)
    outer_margin = np.minimum.reduce([x, 2.0 - x, y - bottom, top - y])
    lower_channel_margin = 0.4 - y
    upper_channel_margin = y - 0.6
    throat_margin = np.minimum.reduce([x - 0.9, 1.1 - x, y - 0.4, 0.6 - y])
    h_shape_margin = np.maximum.reduce([lower_channel_margin, upper_channel_margin, throat_margin])
    return np.minimum(outer_margin, h_shape_margin)


def segment_stays_in_h_channel(p: np.ndarray, q: np.ndarray, sample_count: int, tolerance: float) -> bool:
    alpha = np.linspace(0.0, 1.0, int(sample_count) + 2)
    samples = p[None, :] + alpha[:, None] * (q - p)[None, :]
    return bool(np.all(h_channel_level_set(samples) >= -float(tolerance)))


def build_visibility_filtered_graph(
    domain: DomainDescriptor,
    st_props,
    *,
    candidate_factor: int = 8,
    segment_samples: int = 8,
    level_set_tolerance: float = 1.0e-10,
    max_distance_factor: float = 2.0,
    allow_ghost_neighbors: bool = True,
) -> tuple[FrozenStencilGraph, int, int]:
    domain.build_structs()
    if st_props.point_set == "interior_boundary":
        centers = np.asarray(domain.get_int_bdry_nodes(), dtype=float)
    elif st_props.point_set == "boundary":
        centers = np.asarray(domain.get_bdry_nodes(), dtype=float)
    else:
        raise ValueError(f"Unsupported filtered point set '{st_props.point_set}'")

    tree_points = np.asarray(domain.get_tree_points(st_props.tree_mode), dtype=float)
    tree_globals = np.asarray(domain.get_tree_globals(st_props.tree_mode), dtype=int)
    physical_count = int(domain.get_num_int_bdry_nodes())
    stencil_size = min(int(st_props.n), int(tree_points.shape[0]))
    candidate_count = min(max(stencil_size * int(candidate_factor), stencil_size), int(tree_points.shape[0]))

    tree = cKDTree(tree_points)
    dist, candidate_idx = tree.query(centers, k=candidate_count)
    dist = np.atleast_2d(dist)
    candidate_idx = np.atleast_2d(candidate_idx)
    if candidate_idx.shape[0] == 1 and centers.shape[0] != 1:
        candidate_idx = candidate_idx.T
        dist = dist.T

    rejected_edges = 0
    fallback_rows = 0
    knn_rows = np.empty((centers.shape[0], stencil_size), dtype=np.int64)
    for row, (center, candidates) in enumerate(zip(centers, candidate_idx)):
        original_nearest = np.asarray(candidates[:stencil_size], dtype=np.int64)
        original_radius = float(np.max(dist[row, :stencil_size]))
        max_visible_radius = float(max_distance_factor) * max(original_radius, 1.0e-14)
        accepted: list[int] = []
        rejected: list[int] = []
        for candidate, candidate_distance in zip(np.atleast_1d(candidates), np.atleast_1d(dist[row])):
            local_idx = int(candidate)
            if float(candidate_distance) > max_visible_radius:
                rejected_edges += max(0, stencil_size - len(accepted))
                break
            neighbor = tree_points[local_idx]
            neighbor_is_ghost = int(tree_globals[local_idx]) > physical_count
            visible = neighbor_is_ghost and allow_ghost_neighbors
            if not visible:
                visible = segment_stays_in_h_channel(center, neighbor, segment_samples, level_set_tolerance)
            if visible:
                accepted.append(local_idx)
            else:
                rejected.append(local_idx)
                rejected_edges += 1
            if len(accepted) == stencil_size:
                break
        if len(accepted) < stencil_size:
            fallback_rows += 1
            # A partially filtered row can be much worse conditioned than the
            # unfiltered nearest-neighbor row. If visibility cannot fill the
            # stencil from the candidate pool, leave that row unchanged.
            accepted = list(original_nearest)
        knn_rows[row, :] = np.asarray(accepted[:stencil_size], dtype=np.int64) + 1

    active_rows = np.arange(1, centers.shape[0] + 1, dtype=np.int64)
    return (
        FrozenStencilGraph(
            tree_mode=st_props.tree_mode,
            point_set=st_props.point_set,
            stencil_size=stencil_size,
            active_rows=jnp.asarray(active_rows, dtype=int),
            knn_indices=jnp.asarray(knn_rows, dtype=int),
        ),
        rejected_edges,
        fallback_rows,
    )


def build_h_channel_visibility_solver_template(
    domain: DomainDescriptor,
    *,
    order: int,
    dt: float,
    nu: float,
    stencil: str,
    linear_solver: str,
    candidate_factor: int = 8,
    segment_samples: int = 8,
    level_set_tolerance: float = 1.0e-10,
    max_distance_factor: float = 2.0,
    allow_ghost_neighbors: bool = True,
) -> tuple[ReactionDiffusionSolver, VisibilityFilterStats]:
    solver = ReactionDiffusionSolver(
        lap_assembler="fd",
        bc_assembler="fd",
        lap_stencil=stencil,
        bc_stencil=stencil,
        linear_solver=linear_solver,
    )
    solver.domain = domain
    solver.xi = order
    solver.dt = dt
    solver.nu = nu
    state = build_domain_state(solver.domain)
    solver.x = state.x
    solver.xb = state.xb
    solver.nr = state.nr
    solver.n = state.n
    solver.nf = state.nf
    solver.lap_stencil_properties = build_stencil_properties(solver.domain, solver.xi, 2, "interior_boundary")
    solver.bc_stencil_properties = build_stencil_properties(solver.domain, solver.xi, 1, "boundary")

    lap_graph, lap_rejected, lap_fallback = build_visibility_filtered_graph(
        solver.domain,
        solver.lap_stencil_properties,
        candidate_factor=candidate_factor,
        segment_samples=segment_samples,
        level_set_tolerance=level_set_tolerance,
        max_distance_factor=max_distance_factor,
        allow_ghost_neighbors=allow_ghost_neighbors,
    )
    bc_graph, bc_rejected, bc_fallback = build_visibility_filtered_graph(
        solver.domain,
        solver.bc_stencil_properties,
        candidate_factor=candidate_factor,
        segment_samples=segment_samples,
        level_set_tolerance=level_set_tolerance,
        max_distance_factor=max_distance_factor,
        allow_ghost_neighbors=allow_ghost_neighbors,
    )
    solver.fixed_discretization = FixedDiscretization(laplacian_graph=lap_graph, boundary_graph=bc_graph)
    solver.lap = assemble_operator(
        solver.domain,
        solver.lap_assembler,
        solver.lap_stencil,
        "lap",
        solver.lap_stencil_properties,
        solver.lap_op_properties,
        stencil_graph=solver.fixed_discretization.laplacian_graph,
    )
    solver.lap_sparse = assemble_operator_sparse(
        solver.domain,
        solver.lap_assembler,
        solver.lap_stencil,
        "lap",
        solver.lap_stencil_properties,
        solver.lap_op_properties,
        stencil_graph=solver.fixed_discretization.laplacian_graph,
    )
    solver.bc = jnp.zeros((0, solver.nf))
    solver.bc_sparse = None
    solver.cnm2 = jnp.zeros(solver.n)
    solver.cnm1 = jnp.zeros(solver.n)
    solver.cn = jnp.zeros(solver.n)
    solver.completed_steps_ = 0
    solver.fixed_bc_operator_ready_ = False
    solver.fixed_bc_coefficients_ready_ = False
    solver.cached_neu_coeff_ = jnp.zeros(0)
    solver.cached_dir_coeff_ = jnp.zeros(0)
    solver.fixed_preconditioner = None
    return solver, VisibilityFilterStats(
        lap_rejected_edges=lap_rejected,
        lap_fallback_rows=lap_fallback,
        boundary_rejected_edges=bc_rejected,
        boundary_fallback_rows=bc_fallback,
    )
