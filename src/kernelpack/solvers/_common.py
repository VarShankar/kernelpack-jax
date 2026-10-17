from __future__ import annotations

from dataclasses import dataclass, replace
from math import comb
from typing import Callable
import time
import warnings

from functools import partial

import jax
from jax import jit, lax, vmap
from jax import scipy as jsp
import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, gmres as scipy_gmres, spilu, splu

from kernelpack.domain import DomainDescriptor
from kernelpack.rbffd import FDODiffOp, FDDiffOp, FrozenStencilGraph, RBFStencil, StencilProperties, WeightedLeastSquaresStencil, build_frozen_stencil_graph


@dataclass
class DomainState:
    x: jnp.ndarray
    xb: jnp.ndarray
    nr: jnp.ndarray
    n: int
    nf: int


@dataclass(frozen=True)
class FixedDiscretization:
    laplacian_graph: FrozenStencilGraph
    boundary_graph: FrozenStencilGraph


@dataclass(frozen=True)
class SparseCOOMatrix:
    indices: jnp.ndarray
    values: jnp.ndarray
    shape: tuple[int, int]


@dataclass(frozen=True)
class FrozenILUPreconditioner:
    """Frozen setup-time preconditioner for traced sparse solves.

    The ILU factorization itself is intentionally a setup-time CPU operation
    through SciPy. Small cases can materialize dense approximate inverse
    matrices on-device. Large forward-only cases can instead keep the SuperLU
    object on the host and apply it through a callback to avoid dense N-by-N
    storage.
    """

    apply_matrix: jnp.ndarray | None
    transpose_apply_matrix: jnp.ndarray | None
    host_ilu: object | None = None
    host_row_scaling: np.ndarray | None = None
    application: str = "dense"
    drop_tol: float = 1e-4
    fill_factor: float = 10.0

    def apply(self, x: jnp.ndarray, *, transpose: bool = False) -> jnp.ndarray:
        if self.application == "diagonal":
            if self.apply_matrix is None:
                raise ValueError("diagonal preconditioner is missing its inverse diagonal")
            return self.apply_matrix * x
        if self.application == "host_callback":
            if self.host_ilu is None:
                raise ValueError("host_callback ILU preconditioner is missing its host factorization")

            result_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)

            def callback(x_host: np.ndarray) -> np.ndarray:
                solve_mode = "T" if transpose else "N"
                x_array = np.asarray(x_host, dtype=np.float64)
                if transpose:
                    out = self.host_ilu.solve(x_array, solve_mode)
                    if self.host_row_scaling is not None:
                        out = self.host_row_scaling * out
                else:
                    if self.host_row_scaling is not None:
                        x_array = self.host_row_scaling * x_array
                    out = self.host_ilu.solve(x_array, solve_mode)
                return np.asarray(out, dtype=np.asarray(x_host).dtype)

            return jax.pure_callback(callback, result_shape, x, vmap_method="sequential")

        matrix = self.transpose_apply_matrix if transpose else self.apply_matrix
        if matrix is None:
            raise ValueError("dense ILU preconditioner is missing its apply matrix")
        return matrix @ x


def _normalize_node_values(values: jnp.ndarray | float, row_count: int, label: str) -> jnp.ndarray:
    arr = jnp.asarray(values, dtype=float)
    if arr.ndim == 0:
        return jnp.broadcast_to(arr, (row_count,))
    if arr.ndim == 1:
        if arr.size == 1:
            return jnp.broadcast_to(arr[0], (row_count,))
        if arr.size != row_count:
            raise ValueError(f"{label} values must match the node count")
        return arr
    if arr.ndim == 2:
        if arr.shape[0] == row_count:
            return arr
        if arr.shape[0] == 1:
            return jnp.broadcast_to(arr, (row_count, arr.shape[1]))
        raise ValueError(f"{label} values must match the node count on axis 0")
    raise ValueError(f"{label} values must be scalar, vector, or matrix-valued")


def _ensure_unbatched_operator_input(values: jnp.ndarray, label: str) -> jnp.ndarray:
    arr = jnp.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{label} must be shared across the batch; batched operator-defining inputs are not supported")
    return arr


def _slice_sample(values: jnp.ndarray, sample_idx: int) -> jnp.ndarray:
    arr = jnp.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr
    return arr[:, sample_idx]


def build_stencil_properties(domain: DomainDescriptor, xi: int, theta: int, point_set: str) -> StencilProperties:
    dim = domain.get_dim()
    ell = max(xi + theta - 1, 2)
    sp = StencilProperties()
    sp.dim = dim
    sp.ell = ell
    sp.npoly = int(comb(dim + ell, dim))
    sp.n = 2 * sp.npoly + 1
    sp.spline_degree = ell
    if sp.spline_degree % 2 == 0:
        sp.spline_degree -= 1
    sp.spline_degree = max(sp.spline_degree, 5)
    sp.tree_mode = "all"
    sp.point_set = point_set
    return sp


def resolve_stencil_factory(stencil_spec: str | Callable[[], object]) -> Callable[[], object]:
    if callable(stencil_spec):
        return stencil_spec
    name = str(stencil_spec).lower()
    if name in {"rbf", "rbffd", "rbf-fd"}:
        return lambda: RBFStencil()
    if name in {"wls", "weightedleastsquares", "weighted_least_squares"}:
        return lambda: WeightedLeastSquaresStencil()
    raise ValueError(f"unknown stencil backend {stencil_spec}")


def make_assembler(assembler_spec: str, stencil_spec: str | Callable[[], object]) -> FDDiffOp | FDODiffOp:
    factory = resolve_stencil_factory(stencil_spec)
    name = str(assembler_spec).lower()
    if name in {"fd", "fddiffop", "standard"}:
        return FDDiffOp(factory)
    if name in {"fdo", "fdodiffop", "overlapped", "overlap"}:
        return FDODiffOp(factory)
    raise ValueError(f"unknown assembler {assembler_spec}")


def build_domain_state(domain: DomainDescriptor) -> DomainState:
    domain.build_structs()
    x = domain.get_int_bdry_nodes()
    xb = domain.get_bdry_nodes()
    nr = domain.get_nrmls()
    return DomainState(
        x=x,
        xb=xb,
        nr=nr,
        n=int(x.shape[0]),
        nf=int(domain.get_num_total_nodes()),
    )


def build_fixed_discretization(
    domain: DomainDescriptor,
    lap_stencil_properties: StencilProperties,
    bc_stencil_properties: StencilProperties,
) -> FixedDiscretization:
    domain.build_structs()
    return FixedDiscretization(
        laplacian_graph=build_frozen_stencil_graph(domain, lap_stencil_properties),
        boundary_graph=build_frozen_stencil_graph(domain, bc_stencil_properties),
    )


def build_boundary_paired_ghost_stencil_graph(
    domain: DomainDescriptor,
    bc_stencil_properties: StencilProperties,
    *,
    active_rows: jnp.ndarray | None = None,
) -> FrozenStencilGraph:
    """Build a boundary stencil graph that always includes the paired ghost node.

    KernelPack domains are ordered as [interior; boundary; ghost]. This helper is
    an opt-in diagnostic/experimental closure for ghost-node systems: each
    boundary row keeps the standard nearest-neighbor stencil except that, if its
    paired ghost point was not selected, the farthest selected neighbor is
    replaced by that paired ghost point.
    """

    domain.build_structs()
    point_set = StencilProperties.normalize_point_set(bc_stencil_properties.point_set)
    tree_mode = StencilProperties.normalize_tree_mode(bc_stencil_properties.tree_mode)
    if point_set != "boundary":
        raise ValueError("paired ghost stencil graph is only defined for boundary-row operators")
    if tree_mode != "all":
        raise ValueError("paired ghost stencil graph requires tree_mode='all' so ghost nodes can be selected")

    nb = int(domain.get_num_bdry_nodes())
    nf = int(domain.get_num_total_nodes())
    n_phys = int(domain.get_num_int_bdry_nodes())
    if nb <= 0:
        return FrozenStencilGraph(
            tree_mode=tree_mode,
            point_set=point_set,
            stencil_size=0,
            active_rows=jnp.zeros((0,), dtype=int),
            knn_indices=jnp.zeros((0, 0), dtype=int),
            periodic_bounds=domain.periodic_bounds,
            periodic_dims=domain.periodic_dims,
        )
    if nf - n_phys < nb:
        raise ValueError("paired ghost stencil graph requires at least one ghost node per boundary node")

    if active_rows is None:
        active_rows = jnp.arange(1, nb + 1, dtype=int)
    active_rows_np = np.asarray(active_rows, dtype=np.int64).reshape(-1)
    if np.any(active_rows_np < 1) or np.any(active_rows_np > nb):
        raise ValueError("boundary active rows are out of range")

    centers = domain.get_bdry_nodes()[jnp.asarray(active_rows_np, dtype=int) - 1]
    knn_indices, _ = domain.query_knn(tree_mode, centers, bc_stencil_properties.n,
        center_indices=domain.tree_indices_for_globals(tree_mode,
            domain.get_num_interior_nodes()+jnp.asarray(active_rows_np,dtype=int)))
    knn_np = np.asarray(knn_indices, dtype=np.int64) + 1

    stencil_globals = np.asarray(domain.get_tree_globals(tree_mode), dtype=np.int64)
    global_to_tree_local = {int(global_id): i + 1 for i, global_id in enumerate(stencil_globals)}
    paired_global_ids = n_phys + active_rows_np
    paired_tree_indices = np.asarray(
        [global_to_tree_local[int(global_id)] for global_id in paired_global_ids],
        dtype=np.int64,
    )

    replaced = 0
    for row, paired_tree_index in enumerate(paired_tree_indices):
        if paired_tree_index not in knn_np[row]:
            knn_np[row, -1] = paired_tree_index
            replaced += 1
    if replaced:
        print(
            "[assembly] paired ghost boundary stencil graph: "
            f"replaced_farthest_neighbors={replaced}/{len(active_rows_np)}, stencil_size={knn_np.shape[1]}",
            flush=True,
        )

    return FrozenStencilGraph(
        tree_mode=tree_mode,
        point_set=point_set,
        stencil_size=int(knn_np.shape[1]),
        active_rows=jnp.asarray(active_rows_np, dtype=int),
        knn_indices=jnp.asarray(knn_np, dtype=int),
        periodic_bounds=domain.periodic_bounds,
        periodic_dims=domain.periodic_dims,
    )


def assemble_operator(
    domain: DomainDescriptor,
    assembler_spec: str,
    stencil_spec: str | Callable[[], object],
    op_name: str,
    stencil_properties: StencilProperties,
    op_properties: object,
    *,
    neu_coeff: jnp.ndarray | None = None,
    dir_coeff: jnp.ndarray | None = None,
    stencil_graph: FrozenStencilGraph | None = None,
) -> jnp.ndarray:
    assembler = make_assembler(assembler_spec, stencil_spec)
    assembler.assemble_op(
        domain,
        op_name,
        stencil_properties,
        op_properties,
        neu_coeff=neu_coeff,
        dir_coeff=dir_coeff,
        stencil_graph=stencil_graph,
    )
    return assembler.get_op()


def assemble_operator_sparse(
    domain: DomainDescriptor,
    assembler_spec: str,
    stencil_spec: str | Callable[[], object],
    op_name: str,
    stencil_properties: StencilProperties,
    op_properties: object,
    *,
    neu_coeff: jnp.ndarray | None = None,
    dir_coeff: jnp.ndarray | None = None,
    stencil_graph: FrozenStencilGraph | None = None,
    chunk_size: int | None = None,
) -> SparseCOOMatrix:
    if chunk_size is not None and int(chunk_size) > 0:
        return assemble_operator_sparse_chunked(
            domain,
            assembler_spec,
            stencil_spec,
            op_name,
            stencil_properties,
            op_properties,
            neu_coeff=neu_coeff,
            dir_coeff=dir_coeff,
            stencil_graph=stencil_graph,
            chunk_size=int(chunk_size),
        )
    assembler = make_assembler(assembler_spec, stencil_spec)
    sparse_op_properties = replace(op_properties, sparse_only=True) if hasattr(op_properties, "sparse_only") else op_properties
    assembler.assemble_op(
        domain,
        op_name,
        stencil_properties,
        sparse_op_properties,
        neu_coeff=neu_coeff,
        dir_coeff=dir_coeff,
        stencil_graph=stencil_graph,
    )
    indices = np.asarray(assembler.locations, dtype=np.int64) - 1
    values = np.asarray(assembler.values, dtype=np.float64)
    return SparseCOOMatrix(
        indices=jnp.asarray(indices, dtype=int),
        values=jnp.asarray(values, dtype=float),
        shape=(int(assembler.n1), int(assembler.n2)),
    )


def _slice_stencil_graph(stencil_graph: FrozenStencilGraph, start: int, stop: int) -> FrozenStencilGraph:
    return FrozenStencilGraph(
        tree_mode=stencil_graph.tree_mode,
        point_set=stencil_graph.point_set,
        stencil_size=stencil_graph.stencil_size,
        active_rows=stencil_graph.active_rows[start:stop],
        knn_indices=stencil_graph.knn_indices[start:stop],
        periodic_bounds=stencil_graph.periodic_bounds,
        periodic_dims=stencil_graph.periodic_dims,
    )


def assemble_operator_sparse_chunked(
    domain: DomainDescriptor,
    assembler_spec: str,
    stencil_spec: str | Callable[[], object],
    op_name: str,
    stencil_properties: StencilProperties,
    op_properties: object,
    *,
    neu_coeff: jnp.ndarray | None = None,
    dir_coeff: jnp.ndarray | None = None,
    stencil_graph: FrozenStencilGraph | None = None,
    chunk_size: int,
) -> SparseCOOMatrix:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if stencil_graph is None:
        center_points = domain.get_int_bdry_nodes()
        point_set = StencilProperties.normalize_point_set(stencil_properties.point_set)
        if point_set == "all":
            center_count = int(domain.get_num_total_nodes())
        elif point_set == "interior_boundary":
            center_count = int(domain.get_num_int_bdry_nodes())
        elif point_set == "boundary":
            center_count = int(domain.get_num_bdry_nodes())
        else:
            center_count = int(center_points.shape[0])
        row_count = center_count
        graph_mode = "incremental_knn"
    else:
        row_count = int(stencil_graph.active_rows.shape[0])
        graph_mode = "supplied"

    index_chunks: list[np.ndarray] = []
    value_chunks: list[np.ndarray] = []
    n1 = 0
    n2 = 0
    sparse_op_properties = replace(op_properties, sparse_only=True) if hasattr(op_properties, "sparse_only") else op_properties
    chunk_count = (row_count + chunk_size - 1) // chunk_size
    print(
        "[assembly] chunked sparse assembly start: "
        f"op={op_name}, point_set={stencil_properties.point_set}, tree_mode={stencil_properties.tree_mode}, "
        f"rows={row_count}, stencil_size={stencil_properties.n}, chunk_size={chunk_size}, chunks={chunk_count}, "
        f"graph_mode={graph_mode}",
        flush=True,
    )
    assembly_start = time.perf_counter()
    for start in range(0, row_count, chunk_size):
        stop = min(start + chunk_size, row_count)
        chunk_index = start // chunk_size + 1
        chunk_start = time.perf_counter()
        chunk_rows = stop - start
        augmented_size = int(stencil_properties.n + stencil_properties.npoly)
        lhs_mib = chunk_rows * augmented_size * augmented_size * 8 / (1024**2)
        print(
            "[assembly] chunk start: "
            f"op={op_name}, chunk={chunk_index}/{chunk_count}, rows={start}:{stop}, "
            f"chunk_rows={chunk_rows}, stencil_size={stencil_properties.n}, "
            f"npoly={stencil_properties.npoly}, augmented_size={augmented_size}, "
            f"lhs_estimate={lhs_mib:.2f}MiB",
            flush=True,
        )
        if stencil_graph is None:
            active_rows = jnp.arange(start + 1, stop + 1, dtype=int)
            chunk_graph = build_frozen_stencil_graph(domain, stencil_properties, active_rows=active_rows)
        else:
            chunk_graph = _slice_stencil_graph(stencil_graph, start, stop)
        assembler = make_assembler(assembler_spec, stencil_spec)
        assembler.assemble_op(
            domain,
            op_name,
            stencil_properties,
            sparse_op_properties,
            neu_coeff=neu_coeff,
            dir_coeff=dir_coeff,
            stencil_graph=chunk_graph,
        )
        n1 = max(n1, int(assembler.n1))
        n2 = max(n2, int(assembler.n2))
        # Materialize the solve-dependent values before locations. This blocks
        # the current chunk and permits its temporary factorization buffers to
        # be reclaimed before the next chunk is dispatched.
        value_chunks.append(np.asarray(jax.device_get(assembler.values), dtype=np.float64))
        index_chunks.append(np.asarray(jax.device_get(assembler.locations), dtype=np.int64) - 1)
        del assembler
        print(
            "[assembly] chunk done: "
            f"op={op_name}, chunk={chunk_index}/{chunk_count}, nnz={index_chunks[-1].shape[0]}, "
            f"elapsed={time.perf_counter() - chunk_start:.3f}s",
            flush=True,
        )

    concat_start = time.perf_counter()
    print(f"[assembly] concatenate start: op={op_name}, chunks={chunk_count}", flush=True)
    if index_chunks:
        indices = np.concatenate(index_chunks, axis=0)
        values = np.concatenate(value_chunks, axis=0)
    else:
        indices = np.zeros((0, 2), dtype=np.int64)
        values = np.zeros((0,), dtype=np.float64)
    print(
        "[assembly] chunked sparse assembly done: "
        f"op={op_name}, nnz={values.shape[0]}, concat_elapsed={time.perf_counter() - concat_start:.3f}s, "
        f"total_elapsed={time.perf_counter() - assembly_start:.3f}s",
        flush=True,
    )
    return SparseCOOMatrix(
        indices=jnp.asarray(indices, dtype=int),
        values=jnp.asarray(values, dtype=float),
        shape=(n1, n2),
    )


def evaluate_node_callback(func: Callable[..., jnp.ndarray] | jnp.ndarray | float, x: jnp.ndarray, label: str) -> jnp.ndarray:
    values = func(x) if callable(func) else func
    return _normalize_node_values(values, int(x.shape[0]), label)


def evaluate_boundary_values(
    func: Callable[..., jnp.ndarray] | jnp.ndarray | float,
    neu_coeff: jnp.ndarray,
    dir_coeff: jnp.ndarray,
    nr: jnp.ndarray,
    xb: jnp.ndarray,
) -> jnp.ndarray:
    if callable(func):
        try:
            values = func(neu_coeff, dir_coeff, nr, xb)
        except TypeError:
            values = func(xb)
    else:
        values = func
    return _normalize_node_values(values, int(xb.shape[0]), "boundary")


@partial(jit, static_argnums=(2, 3))
def build_system_matrix(lap: jnp.ndarray, bc: jnp.ndarray, n_cols: int, pure_neumann: bool) -> jnp.ndarray:
    system = jnp.vstack([-lap, bc])
    if pure_neumann:
        ones_col = jnp.ones((system.shape[0], 1))
        ones_row = jnp.ones((1, n_cols))
        system = jnp.vstack([jnp.hstack([system, ones_col]), jnp.hstack([ones_row, jnp.array([[0.0]])])])
    return system


@partial(jit, static_argnums=(2, 5, 6, 7))
def _build_system_matrix_sparse(
    lap_indices: jnp.ndarray,
    lap_values: jnp.ndarray,
    lap_rows_count: int,
    bc_indices: jnp.ndarray,
    bc_values: jnp.ndarray,
    bc_rows_count: int,
    n_cols: int,
    pure_neumann: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, int, int]:
    lap_rows = lap_indices[:, 0]
    lap_cols = lap_indices[:, 1]
    bc_rows = bc_indices[:, 0] + lap_rows_count
    bc_cols = bc_indices[:, 1]
    indices = jnp.column_stack(
        [
            jnp.concatenate([lap_rows, bc_rows]),
            jnp.concatenate([lap_cols, bc_cols]),
        ]
    )
    values = jnp.concatenate([-lap_values, bc_values])
    n_rows = lap_rows_count + bc_rows_count
    final_cols = n_cols
    if pure_neumann:
        aug_rows = jnp.concatenate([jnp.arange(n_rows, dtype=int), jnp.full((n_cols,), n_rows, dtype=int)])
        aug_cols = jnp.concatenate([jnp.full((n_rows,), n_cols, dtype=int), jnp.arange(n_cols, dtype=int)])
        aug_vals = jnp.ones((n_rows + n_cols,), dtype=values.dtype)
        indices = jnp.concatenate([indices, jnp.column_stack([aug_rows, aug_cols])], axis=0)
        values = jnp.concatenate([values, aug_vals])
        n_rows += 1
        final_cols += 1
    return indices, values, n_rows, final_cols


def build_system_matrix_sparse(lap: SparseCOOMatrix, bc: SparseCOOMatrix, n_cols: int, pure_neumann: bool) -> SparseCOOMatrix:
    indices, values, n_rows, final_cols = _build_system_matrix_sparse(
        lap.indices,
        lap.values,
        lap.shape[0],
        bc.indices,
        bc.values,
        bc.shape[0],
        n_cols,
        pure_neumann,
    )
    return SparseCOOMatrix(indices=indices, values=values, shape=(int(n_rows), int(final_cols)))


@partial(jit, static_argnums=(2,))
def build_system_rhs(rhs_target: jnp.ndarray, rhs_boundary: jnp.ndarray, pure_neumann: bool) -> jnp.ndarray:
    rhs_target = jnp.asarray(rhs_target, dtype=float)
    rhs_boundary = jnp.asarray(rhs_boundary, dtype=float)
    if rhs_target.ndim != rhs_boundary.ndim:
        raise ValueError("target and boundary right-hand sides must have matching rank")
    rhs = jnp.concatenate([rhs_target, rhs_boundary], axis=0)
    if pure_neumann:
        if rhs.ndim == 1:
            rhs = jnp.concatenate([rhs, jnp.array([0.0])], axis=0)
        else:
            rhs = jnp.concatenate([rhs, jnp.zeros((1, rhs.shape[1]), dtype=rhs.dtype)], axis=0)
    return rhs


def build_initial_guess(
    initial_guess: jnp.ndarray,
    n_targets: int,
    n_cols: int,
    rhs_boundary: jnp.ndarray,
    pure_neumann: bool,
) -> jnp.ndarray | None:
    guess = jnp.asarray(initial_guess, dtype=float)
    if guess.size == 0:
        return None
    if guess.ndim > 2:
        raise ValueError("initial guesses must be vector- or matrix-valued")
    if guess.ndim == 1:
        batch_size = None
    else:
        batch_size = int(guess.shape[1])
    rhs_boundary_2d = rhs_boundary[:, None] if jnp.asarray(rhs_boundary).ndim == 1 else jnp.asarray(rhs_boundary, dtype=float)
    if batch_size is not None and rhs_boundary_2d.shape[1] != batch_size:
        if rhs_boundary_2d.shape[1] == 1:
            rhs_boundary_2d = jnp.broadcast_to(rhs_boundary_2d, (rhs_boundary_2d.shape[0], batch_size))
        else:
            raise ValueError("initial guess batch size must match the boundary right-hand side batch size")
    if pure_neumann:
        if guess.ndim == 1:
            if guess.size == n_targets:
                return jnp.concatenate([guess, rhs_boundary_2d[:, 0], jnp.array([0.0])])
            if guess.size == n_cols:
                return jnp.concatenate([guess, jnp.array([0.0])])
            if guess.size == n_cols + 1:
                return guess
        else:
            if guess.shape[0] == n_targets:
                return jnp.concatenate([guess, rhs_boundary_2d, jnp.zeros((1, batch_size), dtype=guess.dtype)], axis=0)
            if guess.shape[0] == n_cols:
                return jnp.concatenate([guess, jnp.zeros((1, batch_size), dtype=guess.dtype)], axis=0)
            if guess.shape[0] == n_cols + 1:
                return guess
        raise ValueError(f"pure-Neumann Poisson guess must have length {n_targets}, {n_cols}, or {n_cols + 1}")
    if guess.ndim == 1:
        if guess.size == n_targets:
            return jnp.concatenate([guess, rhs_boundary_2d[:, 0]])
        if guess.size == n_cols:
            return guess
    else:
        if guess.shape[0] == n_targets:
            return jnp.concatenate([guess, rhs_boundary_2d], axis=0)
        if guess.shape[0] == n_cols:
            return guess
    raise ValueError(f"Poisson guess must have length {n_targets} or {n_cols}")


def gmres_with_fallback(system: jnp.ndarray, rhs: jnp.ndarray, guess: jnp.ndarray) -> jnp.ndarray:
    del guess
    return jnp.linalg.solve(system, rhs)


@jit
def solve_dense_system(system: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.solve(system, rhs)


@partial(jit, static_argnums=(2,))
def sparse_matvec(indices: jnp.ndarray, values: jnp.ndarray, n_rows: int, x: jnp.ndarray) -> jnp.ndarray:
    rows = indices[:, 0]
    cols = indices[:, 1]
    contributions = values * x[cols]
    return jnp.zeros((n_rows,), dtype=values.dtype).at[rows].add(contributions)


@partial(jit, static_argnums=(2,))
def sparse_todense(indices: jnp.ndarray, values: jnp.ndarray, shape: tuple[int, int]) -> jnp.ndarray:
    rows = indices[:, 0]
    cols = indices[:, 1]
    return jnp.zeros(shape, dtype=values.dtype).at[rows, cols].add(values)


def sparse_matrix_to_dense(matrix: SparseCOOMatrix) -> jnp.ndarray:
    return sparse_todense(matrix.indices, matrix.values, (int(matrix.shape[0]), int(matrix.shape[1])))


def dense_matrix_to_sparse(matrix: jnp.ndarray, *, tol: float = 0.0) -> SparseCOOMatrix:
    arr = np.asarray(matrix, dtype=np.float64)
    if tol > 0:
        rows, cols = np.nonzero(np.abs(arr) > tol)
    else:
        rows, cols = np.nonzero(arr)
    values = arr[rows, cols]
    indices = jnp.asarray(np.column_stack([rows, cols]), dtype=int)
    return SparseCOOMatrix(indices=indices, values=jnp.asarray(values, dtype=float), shape=(int(arr.shape[0]), int(arr.shape[1])))


def build_fixed_ilu_preconditioner(
    system: SparseCOOMatrix,
    *,
    drop_tol: float = 1e-4,
    fill_factor: float = 10.0,
    application: str = "dense",
) -> FrozenILUPreconditioner:
    """Build a frozen preconditioner outside differentiated rollouts.

    We prefer sparse ILU because it is cheap to factor and usually gives a good
    approximate inverse for GMRES. The default application path materializes a
    dense approximate inverse because that is JAX-friendly for small problems.
    For large forward-only problems, application="host_callback" keeps the
    SuperLU factorization on the host and applies it without dense N-by-N
    storage.
    """

    if application not in ("dense", "host_callback"):
        raise ValueError("application must be 'dense' or 'host_callback'")
    indices = np.asarray(system.indices, dtype=np.int32)
    values = np.asarray(system.values, dtype=np.float64)
    scipy_matrix = coo_matrix((values, (indices[:, 0], indices[:, 1])), shape=system.shape).tocsc()
    scipy_matrix.sum_duplicates()
    scipy_matrix.eliminate_zeros()
    scipy_matrix.sort_indices()
    if not np.all(np.isfinite(scipy_matrix.data)):
        raise ValueError("cannot build ILU preconditioner for a matrix containing nonfinite values")
    row_nnz = np.diff(scipy_matrix.tocsr().indptr)
    column_nnz = np.diff(scipy_matrix.indptr)
    empty_rows = int(np.count_nonzero(row_nnz == 0))
    empty_columns = int(np.count_nonzero(column_nnz == 0))
    zero_diagonal = int(np.count_nonzero(scipy_matrix.diagonal() == 0))
    print(
        "[ilu] matrix diagnostics: "
        f"shape={scipy_matrix.shape}, nnz={scipy_matrix.nnz}, "
        f"empty_rows={empty_rows}, empty_columns={empty_columns}, "
        f"zero_diagonal={zero_diagonal}",
        flush=True,
    )
    if empty_rows or empty_columns:
        raise ValueError(
            "cannot build a preconditioner for a structurally singular sparse system: "
            f"empty_rows={empty_rows}, empty_columns={empty_columns}"
        )
    row_maximum = np.asarray(np.abs(scipy_matrix).max(axis=1).toarray(), dtype=np.float64).reshape(-1)
    row_scaling = 1.0 / row_maximum
    equilibrated_matrix = scipy_matrix.multiply(row_scaling[:, None]).tocsc()
    equilibrated_matrix.sum_duplicates()
    equilibrated_matrix.eliminate_zeros()
    equilibrated_matrix.sort_indices()
    print(
        "[ilu] row equilibration: "
        f"scale_min={float(np.min(row_scaling)):.6e}, "
        f"scale_max={float(np.max(row_scaling)):.6e}",
        flush=True,
    )
    try:
        if application == "host_callback":
            ilu_attempts = (
                {
                    "drop_tol": drop_tol,
                    "fill_factor": fill_factor,
                },
                {
                    "drop_tol": min(drop_tol, 1e-6),
                    "fill_factor": max(fill_factor, 20.0),
                    "permc_spec": "COLAMD",
                    "diag_pivot_thresh": 0.0,
                },
            )
            ilu = None
            failures = []
            probe = np.random.default_rng(0).standard_normal(system.shape[1])
            probe_rhs = equilibrated_matrix @ probe
            for attempt_index, kwargs in enumerate(ilu_attempts, start=1):
                try:
                    candidate = spilu(equilibrated_matrix, **kwargs)
                    probe_reconstruction = candidate.solve(probe_rhs)
                    quality = float(
                        np.linalg.norm(probe_reconstruction - probe)
                        / max(np.linalg.norm(probe), np.finfo(np.float64).eps)
                    )
                    if not np.isfinite(quality) or quality > 1e2:
                        raise RuntimeError(
                            f"numerically unstable ILU preconditioner (probe relative error={quality:.6e})"
                        )
                    ilu = candidate
                    if attempt_index > 1:
                        print(
                            f"[ilu] sparse retry {attempt_index}/{len(ilu_attempts)} succeeded: "
                            f"probe_relative_error={quality:.6e}, settings={kwargs}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[ilu] sparse attempt 1/{len(ilu_attempts)} succeeded: "
                            f"probe_relative_error={quality:.6e}",
                            flush=True,
                        )
                    break
                except RuntimeError as exc:
                    failures.append(str(exc))
                    print(
                        f"[ilu] sparse attempt {attempt_index}/{len(ilu_attempts)} failed: {exc}",
                        flush=True,
                    )

            if ilu is None:
                diagonal = np.asarray(equilibrated_matrix.diagonal(), dtype=np.float64)
                diagonal_scale = max(float(np.max(np.abs(diagonal), initial=0.0)), 1.0)
                shifted_matrix = equilibrated_matrix.copy()
                diagonal_shift = 1e-10 * diagonal_scale
                shifted_matrix.setdiag(diagonal + diagonal_shift)
                shifted_kwargs = {
                    "drop_tol": min(drop_tol, 1e-6),
                    "fill_factor": max(fill_factor, 20.0),
                    "permc_spec": "COLAMD",
                    "diag_pivot_thresh": 0.0,
                }
                try:
                    ilu = spilu(shifted_matrix, **shifted_kwargs)
                    probe_reconstruction = ilu.solve(probe_rhs)
                    quality = float(
                        np.linalg.norm(probe_reconstruction - probe)
                        / max(np.linalg.norm(probe), np.finfo(np.float64).eps)
                    )
                    if not np.isfinite(quality) or quality > 1e2:
                        raise RuntimeError(
                            f"numerically unstable shifted ILU preconditioner "
                            f"(probe relative error={quality:.6e})"
                        )
                    print(
                        f"[ilu] sparse shifted retry succeeded: shift={diagonal_shift:.6e}, "
                        f"probe_relative_error={quality:.6e}, settings={shifted_kwargs}",
                        flush=True,
                    )
                except RuntimeError as exc:
                    failures.append(str(exc))
                    print(f"[ilu] sparse shifted retry failed: {exc}", flush=True)

            if ilu is None:
                raise RuntimeError(
                    "Sparse ILU factorization failed after all retries; refusing to use "
                    "the ineffective Jacobi fallback. "
                    f"Last SuperLU error: {failures[-1]}"
                )
        else:
            ilu = spilu(scipy_matrix, drop_tol=drop_tol, fill_factor=fill_factor)
        if application == "host_callback":
            return FrozenILUPreconditioner(
                apply_matrix=None,
                transpose_apply_matrix=None,
                host_ilu=ilu,
                host_row_scaling=row_scaling,
                application=application,
                drop_tol=drop_tol,
                fill_factor=fill_factor,
            )
        eye = np.eye(system.shape[0], dtype=values.dtype)
        apply_matrix = np.column_stack([ilu.solve(eye[:, i]) for i in range(system.shape[0])])
        transpose_apply_matrix = np.column_stack([ilu.solve(eye[:, i], "T") for i in range(system.shape[0])])
    except RuntimeError:
        if application == "host_callback":
            raise
        dense_matrix = scipy_matrix.toarray()
        eye = np.eye(system.shape[0], dtype=values.dtype)
        try:
            apply_matrix = np.linalg.solve(dense_matrix, eye)
            transpose_apply_matrix = np.linalg.solve(dense_matrix.T, eye)
        except np.linalg.LinAlgError:
            apply_matrix = np.linalg.pinv(dense_matrix)
            transpose_apply_matrix = np.linalg.pinv(dense_matrix.T)
    return FrozenILUPreconditioner(
        apply_matrix=lax.stop_gradient(jnp.asarray(apply_matrix, dtype=float)),
        transpose_apply_matrix=lax.stop_gradient(jnp.asarray(transpose_apply_matrix, dtype=float)),
        host_ilu=None,
        application=application,
        drop_tol=drop_tol,
        fill_factor=fill_factor,
    )


def build_fixed_sparse_lu_preconditioner(system: SparseCOOMatrix) -> FrozenILUPreconditioner:
    """Build an exact row-equilibrated sparse-LU preconditioner on the host."""

    indices = np.asarray(system.indices, dtype=np.int32)
    values = np.asarray(system.values, dtype=np.float64)
    scipy_matrix = coo_matrix(
        (values, (indices[:, 0], indices[:, 1])),
        shape=system.shape,
    ).tocsc()
    scipy_matrix.sum_duplicates()
    scipy_matrix.eliminate_zeros()
    scipy_matrix.sort_indices()
    row_maximum = np.asarray(
        np.abs(scipy_matrix).max(axis=1).toarray(),
        dtype=np.float64,
    ).reshape(-1)
    if np.any(row_maximum == 0.0):
        raise ValueError("cannot factor a sparse system containing an empty row")
    row_scaling = 1.0 / row_maximum
    equilibrated_matrix = scipy_matrix.multiply(row_scaling[:, None]).tocsc()
    print(
        "[sparse-lu-preconditioner] factorizing: "
        f"shape={system.shape}, nnz={scipy_matrix.nnz}, "
        f"scale_min={float(np.min(row_scaling)):.6e}, "
        f"scale_max={float(np.max(row_scaling)):.6e}",
        flush=True,
    )
    factor = splu(equilibrated_matrix, permc_spec="COLAMD")
    print(
        "[sparse-lu-preconditioner] ready: "
        f"factor_nnz={int(factor.L.nnz + factor.U.nnz)}",
        flush=True,
    )
    return FrozenILUPreconditioner(
        apply_matrix=None,
        transpose_apply_matrix=None,
        host_ilu=factor,
        host_row_scaling=row_scaling,
        application="host_callback",
        drop_tol=0.0,
        fill_factor=0.0,
    )


def solve_sparse_system_gmres(
    system: SparseCOOMatrix,
    rhs: jnp.ndarray,
    *,
    guess: jnp.ndarray | None = None,
    preconditioner: FrozenILUPreconditioner | None = None,
    tol: float = 1e-8,
    atol: float = 1e-10,
    restart: int = 40,
    maxiter: int | None = None,
) -> jnp.ndarray:
    """Solve sparse systems with JAX GMRES under `custom_linear_solve`.

    Matrix-vector products, preconditioner application, batching, and the
    transpose solve all run through JAX arrays.  A supplied ILU is treated as a
    frozen setup artifact, so gradients propagate through the linear solve but
    not through the preconditioner construction.
    """

    # TODO: Add an experimental hybrid AD mode: use sparse GMRES for the
    # primal solve, but use a Lineax solve for tangent/transpose solves during
    # Jacobian formation. This may preserve sparse forward performance while
    # avoiding GPU memory blowups from AD through the custom GMRES path.

    rhs = jnp.asarray(rhs, dtype=float)
    if rhs.ndim == 2:
        guess_matrix = None if guess is None else jnp.asarray(guess, dtype=rhs.dtype)
        if guess_matrix is None:
            solve_cols = vmap(
                lambda rhs_col: solve_sparse_system_gmres(
                    system,
                    rhs_col,
                    guess=None,
                    preconditioner=preconditioner,
                    tol=tol,
                    atol=atol,
                    restart=restart,
                    maxiter=maxiter,
                ),
                in_axes=1,
                out_axes=1,
            )
            return solve_cols(rhs)
        solve_cols = vmap(
            lambda rhs_col, guess_col: solve_sparse_system_gmres(
                system,
                rhs_col,
                guess=guess_col,
                preconditioner=preconditioner,
                tol=tol,
                atol=atol,
                restart=restart,
                maxiter=maxiter,
            ),
            in_axes=(1, 1),
            out_axes=1,
        )
        return solve_cols(rhs, guess_matrix)
    if guess is None:
        guess = jnp.zeros(system.shape[1], dtype=rhs.dtype)
    else:
        guess = jnp.asarray(guess, dtype=rhs.dtype).reshape(-1)

    def matvec(x: jnp.ndarray) -> jnp.ndarray:
        return sparse_matvec(system.indices, system.values, system.shape[0], x)

    def vecmat(x: jnp.ndarray) -> jnp.ndarray:
        transposed = jnp.column_stack([system.indices[:, 1], system.indices[:, 0]])
        return sparse_matvec(transposed, system.values, system.shape[1], x)

    def precond(x: jnp.ndarray) -> jnp.ndarray:
        if preconditioner is None:
            return x
        return preconditioner.apply(x)

    def transpose_precond(x: jnp.ndarray) -> jnp.ndarray:
        if preconditioner is None:
            return x
        return preconditioner.apply(x, transpose=True)

    def solve(operator: Callable[[jnp.ndarray], jnp.ndarray], target: jnp.ndarray) -> jnp.ndarray:
        sol, _ = jsp.sparse.linalg.gmres(
            operator,
            target,
            x0=guess,
            tol=tol,
            atol=atol,
            restart=restart,
            maxiter=maxiter,
            M=precond,
            solve_method="batched",
        )
        return sol

    def transpose_solve(operator: Callable[[jnp.ndarray], jnp.ndarray], target: jnp.ndarray) -> jnp.ndarray:
        sol, _ = jsp.sparse.linalg.gmres(
            operator,
            target,
            x0=jnp.zeros_like(target),
            tol=tol,
            atol=atol,
            restart=restart,
            maxiter=maxiter,
            M=transpose_precond,
            solve_method="batched",
        )
        return sol

    del vecmat #why is this being cleared?
    return lax.custom_linear_solve(matvec, rhs, solve, transpose_solve=transpose_solve, symmetric=False)


def solve_sparse_system_scipy_gmres(
    system: SparseCOOMatrix,
    rhs: jnp.ndarray,
    *,
    guess: jnp.ndarray | None = None,
    preconditioner: FrozenILUPreconditioner | None = None,
    tol: float = 1e-8,
    atol: float = 1e-10,
    restart: int = 40,
    maxiter: int | None = 500,
) -> jnp.ndarray:
    """Solve a forward-only sparse system with SciPy and report convergence."""

    indices = np.asarray(system.indices, dtype=np.int32)
    values = np.asarray(system.values, dtype=np.float64)
    scipy_matrix = coo_matrix(
        (values, (indices[:, 0], indices[:, 1])),
        shape=system.shape,
    ).tocsr()
    scipy_matrix.sum_duplicates()
    scipy_matrix.eliminate_zeros()
    scipy_matrix.sort_indices()

    preconditioner_operator = None
    if preconditioner is not None:
        if preconditioner.application == "host_callback":
            if preconditioner.host_ilu is None:
                raise ValueError("host_callback ILU preconditioner is missing its host factorization")

            def apply_host_ilu(vector: np.ndarray) -> np.ndarray:
                vector_host = np.asarray(vector, dtype=np.float64)
                if preconditioner.host_row_scaling is not None:
                    vector_host = preconditioner.host_row_scaling * vector_host
                return preconditioner.host_ilu.solve(vector_host)

            preconditioner_operator = LinearOperator(
                system.shape,
                matvec=apply_host_ilu,
                dtype=np.float64,
            )
        elif preconditioner.application == "diagonal":
            inverse_diagonal = np.asarray(preconditioner.apply_matrix, dtype=np.float64)
            preconditioner_operator = LinearOperator(
                system.shape,
                matvec=lambda vector: inverse_diagonal * vector,
                dtype=np.float64,
            )
        elif preconditioner.apply_matrix is not None:
            apply_matrix = np.asarray(preconditioner.apply_matrix, dtype=np.float64)
            preconditioner_operator = LinearOperator(
                system.shape,
                matvec=lambda vector: apply_matrix @ vector,
                dtype=np.float64,
            )

    rhs_host = np.asarray(rhs, dtype=np.float64)
    if rhs_host.ndim == 2:
        columns = [
            solve_sparse_system_scipy_gmres(
                system,
                jnp.asarray(rhs_host[:, column]),
                guess=None if guess is None else jnp.asarray(guess)[:, column],
                preconditioner=preconditioner,
                tol=tol,
                atol=atol,
                restart=restart,
                maxiter=maxiter,
            )
            for column in range(rhs_host.shape[1])
        ]
        return jnp.column_stack(columns)

    guess_host = None
    if guess is not None:
        guess_host = np.asarray(guess, dtype=np.float64).reshape(-1)
        if guess_host.size == 0:
            guess_host = None

    iteration_count = 0

    def report_iteration(preconditioned_residual: float) -> None:
        nonlocal iteration_count
        iteration_count += 1
        if iteration_count == 1 or iteration_count % 25 == 0:
            print(
                f"[scipy-gmres] iteration={iteration_count}, "
                f"preconditioned_residual={float(preconditioned_residual):.6e}",
                flush=True,
            )

    solution, info = scipy_gmres(
        scipy_matrix,
        rhs_host,
        x0=guess_host,
        rtol=tol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        M=preconditioner_operator,
        callback=report_iteration,
        callback_type="pr_norm",
    )
    residual = scipy_matrix @ solution - rhs_host
    relative_residual = float(np.linalg.norm(residual) / max(np.linalg.norm(rhs_host), atol))
    print(
        f"[scipy-gmres] done: info={info}, iterations={iteration_count}, "
        f"relative_residual={relative_residual:.6e}",
        flush=True,
    )
    if info != 0:
        raise RuntimeError(
            f"SciPy GMRES did not converge: info={info}, iterations={iteration_count}, "
            f"relative_residual={relative_residual:.6e}"
        )
    if not np.all(np.isfinite(solution)) or not np.isfinite(relative_residual):
        raise FloatingPointError("SciPy GMRES produced a nonfinite solution or residual")
    if relative_residual > max(10.0 * tol, atol):
        raise RuntimeError(
            "SciPy GMRES reported convergence but the verified relative residual was "
            f"{relative_residual:.6e}"
        )
    return jnp.asarray(solution, dtype=float)


def solve_sparse_system_scipy_direct(
    system: SparseCOOMatrix,
    rhs: jnp.ndarray,
) -> jnp.ndarray:
    """Solve a forward-only sparse system using row-equilibrated SuperLU."""

    indices = np.asarray(system.indices, dtype=np.int32)
    values = np.asarray(system.values, dtype=np.float64)
    scipy_matrix = coo_matrix(
        (values, (indices[:, 0], indices[:, 1])),
        shape=system.shape,
    ).tocsc()
    scipy_matrix.sum_duplicates()
    scipy_matrix.eliminate_zeros()
    scipy_matrix.sort_indices()
    rhs_host = np.asarray(rhs, dtype=np.float64)

    row_maximum = np.asarray(
        np.abs(scipy_matrix).max(axis=1).toarray(),
        dtype=np.float64,
    ).reshape(-1)
    if np.any(row_maximum == 0.0):
        raise ValueError("cannot directly solve a sparse system containing an empty row")
    row_scaling = 1.0 / row_maximum
    equilibrated_matrix = scipy_matrix.multiply(row_scaling[:, None]).tocsc()
    equilibrated_rhs = row_scaling[:, None] * rhs_host if rhs_host.ndim == 2 else row_scaling * rhs_host

    print(
        "[scipy-direct] factorizing row-equilibrated sparse system: "
        f"shape={scipy_matrix.shape}, nnz={scipy_matrix.nnz}, "
        f"scale_min={float(np.min(row_scaling)):.6e}, "
        f"scale_max={float(np.max(row_scaling)):.6e}",
        flush=True,
    )
    factor = splu(equilibrated_matrix, permc_spec="COLAMD")
    factor_nnz = int(factor.L.nnz + factor.U.nnz)
    print(
        f"[scipy-direct] factorization ready: factor_nnz={factor_nnz}",
        flush=True,
    )
    solution = factor.solve(equilibrated_rhs)
    residual = scipy_matrix @ solution - rhs_host
    relative_residual = float(np.linalg.norm(residual) / max(np.linalg.norm(rhs_host), 1e-10))
    print(
        f"[scipy-direct] solve done: relative_residual={relative_residual:.6e}",
        flush=True,
    )
    if not np.all(np.isfinite(solution)) or not np.isfinite(relative_residual):
        raise FloatingPointError("SciPy sparse direct solve produced a nonfinite solution or residual")
    if relative_residual > 1e-8:
        raise RuntimeError(
            "SciPy sparse direct solve produced an excessive verified relative residual: "
            f"{relative_residual:.6e}"
        )
    return jnp.asarray(solution, dtype=float)


def validate_physical_state(state: jnp.ndarray, n: int) -> jnp.ndarray:
    state = jnp.asarray(state, dtype=float)
    if state.ndim == 1:
        if state.size != n:
            raise ValueError(f"expected a physical state of length {n}")
        return state
    if state.ndim == 2 and state.shape[0] == n:
        return state
    raise ValueError(f"expected a physical state with leading dimension {n}")


def is_fixed_boundary_callback(func: object) -> bool:
    if not callable(func):
        return False
    try:
        return func.__code__.co_argcount == 1
    except AttributeError:
        return False


def evaluate_boundary_coefficient(
    func: Callable[..., jnp.ndarray] | jnp.ndarray | float,
    x: jnp.ndarray,
    t: float | None = None,
) -> jnp.ndarray:
    if callable(func):
        if t is None:
            values = func(x)
        else:
            try:
                values = func(t, x)
            except TypeError:
                values = func(x)
    else:
        values = func
    return _normalize_node_values(values, int(x.shape[0]), "boundary coefficient")


def evaluate_forcing_callback(
    func: Callable[..., jnp.ndarray] | jnp.ndarray | float,
    nu: float,
    t: float,
    x: jnp.ndarray,
) -> jnp.ndarray:
    if callable(func):
        try:
            values = func(nu, t, x)
        except TypeError:
            try:
                values = func(t, x)
            except TypeError:
                values = func(x)
    else:
        values = func
    return _normalize_node_values(values, int(x.shape[0]), "forcing")


def evaluate_transient_boundary_values(
    func: Callable[..., jnp.ndarray] | jnp.ndarray | float,
    neu_coeff: jnp.ndarray,
    dir_coeff: jnp.ndarray,
    nr: jnp.ndarray,
    t: float,
    xb: jnp.ndarray,
) -> jnp.ndarray:
    if callable(func):
        try:
            values = func(neu_coeff, dir_coeff, nr, t, xb)
        except TypeError:
            try:
                values = func(t, xb)
            except TypeError:
                values = func(xb)
    else:
        values = func
    return _normalize_node_values(values, int(xb.shape[0]), "boundary")


@partial(jit, static_argnums=(2,))
def build_implicit_system(lap: jnp.ndarray, bc: jnp.ndarray, n_physical: int, lap_scale: float) -> jnp.ndarray:
    system = lap_scale * lap
    system = system.at[:n_physical, :n_physical].add(jnp.eye(n_physical))
    return jnp.vstack([system, bc])


@partial(jit, static_argnums=(2, 3, 6))
def _build_implicit_system_sparse(
    lap_indices: jnp.ndarray,
    lap_values: jnp.ndarray,
    lap_rows_count: int,
    lap_cols_count: int,
    bc_indices: jnp.ndarray,
    bc_values: jnp.ndarray,
    n_physical: int,
    lap_scale: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    lap_rows = lap_indices[:, 0]
    lap_cols = lap_indices[:, 1]
    diag_idx = jnp.arange(n_physical, dtype=int)
    bc_rows = bc_indices[:, 0] + lap_rows_count
    bc_cols = bc_indices[:, 1]
    indices = jnp.concatenate(
        [
            jnp.column_stack([lap_rows, lap_cols]),
            jnp.column_stack([diag_idx, diag_idx]),
            jnp.column_stack([bc_rows, bc_cols]),
        ],
        axis=0,
    )
    values = jnp.concatenate(
        [
            lap_scale * lap_values,
            jnp.ones((n_physical,), dtype=lap_values.dtype),
            bc_values,
        ]
    )
    return indices, values


def build_implicit_system_sparse(lap: SparseCOOMatrix, bc: SparseCOOMatrix, n_physical: int, lap_scale: float) -> SparseCOOMatrix:
    indices, values = _build_implicit_system_sparse(
        lap.indices,
        lap.values,
        lap.shape[0],
        lap.shape[1],
        bc.indices,
        bc.values,
        n_physical,
        lap_scale,
    )
    return SparseCOOMatrix(indices=indices, values=values, shape=(int(lap.shape[0] + bc.shape[0]), int(lap.shape[1])))


@jit
def build_implicit_rhs(rhs_physical: jnp.ndarray, rhs_boundary: jnp.ndarray) -> jnp.ndarray:
    rhs_physical = jnp.asarray(rhs_physical, dtype=float)
    rhs_boundary = jnp.asarray(rhs_boundary, dtype=float)
    if rhs_physical.ndim != rhs_boundary.ndim:
        raise ValueError("physical and boundary right-hand sides must have matching rank")
    return jnp.concatenate([rhs_physical, rhs_boundary], axis=0)


@jit
def push_completed_step(
    cnm2: jnp.ndarray,
    cnm1: jnp.ndarray,
    cn: jnp.ndarray,
    completed_steps: int,
    next_state: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    def case0(_: None) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
        return cnm2, next_state, cn, completed_steps + 1

    def case1(_: None) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
        return cnm2, cnm1, next_state, completed_steps + 1

    def case2(_: None) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
        return cnm1, cn, next_state, completed_steps + 1

    return lax.switch(jnp.minimum(completed_steps, 2), [case0, case1, case2], None)
