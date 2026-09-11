from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Callable

from functools import partial

from jax import jit, lax, vmap
from jax import scipy as jsp
import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spilu

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
    """Host-built, device-applied preconditioner for traced sparse solves.

    The ILU factorization itself is intentionally a setup-time CPU operation
    through SciPy.  The resulting approximate inverse matrices are frozen with
    `lax.stop_gradient` and used only as fixed JAX arrays inside GMRES and its
    transpose solve.
    """

    apply_matrix: jnp.ndarray
    transpose_apply_matrix: jnp.ndarray
    drop_tol: float = 1e-4
    fill_factor: float = 10.0


def _normalize_node_values(values: jnp.ndarray | float, row_count: int, label: str) -> jnp.ndarray:
    arr = jnp.asarray(values, dtype=float)
    if arr.ndim == 0:
        return jnp.full((row_count,), float(arr))
    if arr.ndim == 1:
        if arr.size == 1:
            return jnp.full((row_count,), float(arr[0]))
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
) -> SparseCOOMatrix:
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
    return SparseCOOMatrix(
        indices=jnp.asarray(assembler.locations - 1, dtype=int),
        values=jnp.asarray(assembler.values, dtype=float),
        shape=(int(assembler.n1), int(assembler.n2)),
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
) -> FrozenILUPreconditioner:
    """Build a frozen preconditioner outside differentiated rollouts.

    We prefer sparse ILU because it is cheap to factor and usually gives a good
    approximate inverse for GMRES. If SuperLU reports a singular factorization,
    fall back to a dense approximate inverse so the preconditioner can still be
    used on awkward matrices.
    """

    indices = np.asarray(system.indices, dtype=np.int32)
    values = np.asarray(system.values, dtype=np.float64)
    scipy_matrix = coo_matrix((values, (indices[:, 0], indices[:, 1])), shape=system.shape).tocsc()
    eye = np.eye(system.shape[0], dtype=values.dtype)
    try:
        ilu = spilu(scipy_matrix, drop_tol=drop_tol, fill_factor=fill_factor)
        apply_matrix = np.column_stack([ilu.solve(eye[:, i]) for i in range(system.shape[0])])
        transpose_apply_matrix = np.column_stack([ilu.solve(eye[:, i], "T") for i in range(system.shape[0])])
    except RuntimeError:
        dense_matrix = scipy_matrix.toarray()
        try:
            apply_matrix = np.linalg.solve(dense_matrix, eye)
            transpose_apply_matrix = np.linalg.solve(dense_matrix.T, eye)
        except np.linalg.LinAlgError:
            apply_matrix = np.linalg.pinv(dense_matrix)
            transpose_apply_matrix = np.linalg.pinv(dense_matrix.T)
    return FrozenILUPreconditioner(
        apply_matrix=lax.stop_gradient(jnp.asarray(apply_matrix, dtype=float)),
        transpose_apply_matrix=lax.stop_gradient(jnp.asarray(transpose_apply_matrix, dtype=float)),
        drop_tol=drop_tol,
        fill_factor=fill_factor,
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
        return preconditioner.apply_matrix @ x

    def transpose_precond(x: jnp.ndarray) -> jnp.ndarray:
        if preconditioner is None:
            return x
        return preconditioner.transpose_apply_matrix @ x

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
