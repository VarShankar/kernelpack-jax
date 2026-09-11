from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax.numpy as jnp
from jax import core as jax_core, vmap
import lineax as lx

from kernelpack.domain import DomainDescriptor
from kernelpack.rbffd import OpProperties, StencilProperties
from ._common import (
    assemble_operator,
    assemble_operator_sparse,
    build_domain_state,
    build_initial_guess,
    build_fixed_ilu_preconditioner,
    build_stencil_properties,
    build_system_rhs,
    solve_sparse_system_gmres,
    sparse_matrix_to_dense,
    SparseCOOMatrix,
    FrozenILUPreconditioner,
    evaluate_boundary_values,
    evaluate_node_callback,
    gmres_with_fallback,
    _ensure_unbatched_operator_input,
    solve_dense_system,
)


def _solve_lineax_system(system: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    operator = lx.MatrixLinearOperator(jnp.asarray(system, dtype=float))
    solver = lx.AutoLinearSolver(well_posed=False)
    rhs_arr = jnp.asarray(rhs, dtype=float)
    if rhs_arr.ndim == 1:
        return lx.linear_solve(operator, rhs_arr, solver=solver).value
    return vmap(
        lambda rhs_col: lx.linear_solve(operator, rhs_col, solver=solver).value,
        in_axes=1,
        out_axes=1,
    )(rhs_arr)

def _build_variable_pde_operator(lap: jnp.ndarray, grad_ops: tuple[jnp.ndarray, ...], coeff_all: jnp.ndarray, n_rows: int) -> jnp.ndarray:
    coeff_local = coeff_all[:n_rows]
    pde = -coeff_local[:, None] * lap
    for grad in grad_ops:
        grad_coeff = grad @ coeff_all
        pde = pde - grad_coeff[:, None] * grad
    return pde


def _build_variable_system_matrix(pde: jnp.ndarray, bc: jnp.ndarray, n_cols: int, pure_neumann: bool) -> jnp.ndarray:
    system = jnp.vstack([pde, bc])
    if pure_neumann:
        ones_col = jnp.ones((system.shape[0], 1))
        ones_row = jnp.ones((1, n_cols))
        system = jnp.vstack([jnp.hstack([system, ones_col]), jnp.hstack([ones_row, jnp.array([[0.0]])])])
    return system


def _scale_sparse_rows(matrix: SparseCOOMatrix, row_scalars: jnp.ndarray) -> SparseCOOMatrix:
    rows = matrix.indices[:, 0]
    return SparseCOOMatrix(indices=matrix.indices, values=matrix.values * row_scalars[rows], shape=matrix.shape)


def _add_sparse_matrices(*matrices: SparseCOOMatrix) -> SparseCOOMatrix:
    first = matrices[0]
    indices = jnp.concatenate([matrix.indices for matrix in matrices], axis=0)
    values = jnp.concatenate([matrix.values for matrix in matrices], axis=0)
    return SparseCOOMatrix(indices=indices, values=values, shape=first.shape)


def _build_variable_pde_operator_sparse(
    lap_sparse: SparseCOOMatrix,
    grad_sparse: tuple[SparseCOOMatrix, ...],
    coeff_all: jnp.ndarray,
    n_rows: int,
) -> SparseCOOMatrix:
    coeff_local = coeff_all[:n_rows]
    pieces = [_scale_sparse_rows(lap_sparse, -coeff_local)]
    for grad in grad_sparse:
        grad_coeff = jnp.asarray(
            sparse_matrix_to_dense(grad) @ coeff_all if grad.shape[0] == 0 else jnp.zeros((grad.shape[0],), dtype=coeff_all.dtype),
            dtype=float,
        )
        if grad.shape[0] > 0:
            grad_coeff = sparse_matvec_from_coo(grad, coeff_all)
        pieces.append(_scale_sparse_rows(grad, -grad_coeff))
    return _add_sparse_matrices(*pieces)


def sparse_matvec_from_coo(matrix: SparseCOOMatrix, x: jnp.ndarray) -> jnp.ndarray:
    rows = matrix.indices[:, 0]
    cols = matrix.indices[:, 1]
    return jnp.zeros((matrix.shape[0],), dtype=matrix.values.dtype).at[rows].add(matrix.values * x[cols])


def _build_variable_system_matrix_sparse(pde: SparseCOOMatrix, bc: SparseCOOMatrix, n_cols: int, pure_neumann: bool) -> SparseCOOMatrix:
    pde_rows = pde.indices[:, 0]
    pde_cols = pde.indices[:, 1]
    bc_rows = bc.indices[:, 0] + pde.shape[0]
    bc_cols = bc.indices[:, 1]
    indices = jnp.column_stack([jnp.concatenate([pde_rows, bc_rows]), jnp.concatenate([pde_cols, bc_cols])])
    values = jnp.concatenate([pde.values, bc.values])
    n_rows = pde.shape[0] + bc.shape[0]
    final_cols = n_cols
    if pure_neumann:
        aug_rows = jnp.concatenate([jnp.arange(n_rows, dtype=int), jnp.full((n_cols,), n_rows, dtype=int)])
        aug_cols = jnp.concatenate([jnp.full((n_rows,), n_cols, dtype=int), jnp.arange(n_cols, dtype=int)])
        aug_vals = jnp.ones((n_rows + n_cols,), dtype=values.dtype)
        indices = jnp.concatenate([indices, jnp.column_stack([aug_rows, aug_cols])], axis=0)
        values = jnp.concatenate([values, aug_vals])
        n_rows += 1
        final_cols += 1
    return SparseCOOMatrix(indices=indices, values=values, shape=(int(n_rows), int(final_cols)))


@dataclass
class VariablePoissonSolver:
    lap_assembler: str = "fd"
    bc_assembler: str = "fd"
    lap_stencil: str = "rbf"
    bc_stencil: str = "rbf"
    domain: DomainDescriptor = field(default_factory=DomainDescriptor)
    xi: int = 0
    num_omp_threads: int = 1
    linear_solver: str = "gmres_ilu"
    x: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    xb: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    xf: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    nr: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    n: int = 0
    nf: int = 0
    lap: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    lap_sparse: SparseCOOMatrix | None = None
    grad: tuple[jnp.ndarray, ...] = field(default_factory=tuple)
    grad_sparse: tuple[SparseCOOMatrix, ...] = field(default_factory=tuple)
    pde: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    pde_sparse: SparseCOOMatrix | None = None
    bc: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    bc_sparse: SparseCOOMatrix | None = None
    lap_stencil_properties: StencilProperties = field(default_factory=StencilProperties)
    bc_stencil_properties: StencilProperties = field(default_factory=StencilProperties)
    lap_op_properties: OpProperties = field(default_factory=lambda: OpProperties(decompose=False, store_weights=True, record_stencils=False))
    bc_op_properties: OpProperties = field(default_factory=lambda: OpProperties(decompose=False, store_weights=True, record_stencils=False))
    fixed_preconditioner: FrozenILUPreconditioner | None = None
    fixed_bc_operator_ready_: bool = False
    fixed_bc_coefficients_ready_: bool = False
    cached_neu_coeff_: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    cached_dir_coeff_: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    cached_pure_neumann_: bool = False
    last_solve_used_nullspace_: bool = False

    def init(self, domain: DomainDescriptor, xi: int, num_omp_threads: int = 1) -> None:
        self.domain = domain
        self.xi = xi
        self.num_omp_threads = num_omp_threads
        state = build_domain_state(self.domain)
        self.x = state.x
        self.xb = state.xb
        self.nr = state.nr
        self.n = state.n
        self.nf = state.nf
        self.xf = self.domain.get_all_nodes()
        self.lap_stencil_properties = build_stencil_properties(self.domain, self.xi, 2, "interior_boundary")
        self.bc_stencil_properties = build_stencil_properties(self.domain, self.xi, 1, "boundary")

        self.lap = assemble_operator(
            self.domain,
            self.lap_assembler,
            self.lap_stencil,
            "lap",
            self.lap_stencil_properties,
            self.lap_op_properties,
        )
        self.lap_sparse = assemble_operator_sparse(
            self.domain,
            self.lap_assembler,
            self.lap_stencil,
            "lap",
            self.lap_stencil_properties,
            self.lap_op_properties,
        )
        grad_ops = []
        grad_sparse_ops = []
        for d in range(self.domain.get_dim()):
            grad_props = OpProperties(decompose=False, store_weights=True, record_stencils=False)
            grad_props.selectdim = d
            grad_ops.append(
                assemble_operator(
                    self.domain,
                    self.lap_assembler,
                    self.lap_stencil,
                    "grad",
                    self.lap_stencil_properties,
                    grad_props,
                )
            )
            grad_sparse_ops.append(
                assemble_operator_sparse(
                    self.domain,
                    self.lap_assembler,
                    self.lap_stencil,
                    "grad",
                    self.lap_stencil_properties,
                    grad_props,
                )
            )
        self.grad = tuple(grad_ops)
        self.grad_sparse = tuple(grad_sparse_ops)
        self.pde = jnp.zeros((0, self.nf), dtype=float)
        self.pde_sparse = None
        self.bc = jnp.zeros((0, self.nf), dtype=float)
        self.bc_sparse = None
        self.fixed_preconditioner = None
        self.fixed_bc_operator_ready_ = False
        self.fixed_bc_coefficients_ready_ = False
        self.cached_neu_coeff_ = jnp.zeros(0)
        self.cached_dir_coeff_ = jnp.zeros(0)
        self.cached_pure_neumann_ = False
        self.last_solve_used_nullspace_ = False

    def _get_boundary_coefficients(
        self,
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if not callable(neu_coeff_func) and not callable(dir_coeff_func):
            if not self.fixed_bc_coefficients_ready_:
                self.cached_neu_coeff_ = _ensure_unbatched_operator_input(
                    evaluate_node_callback(neu_coeff_func, self.xb, "boundary coefficient"),
                    "boundary coefficient",
                )
                self.cached_dir_coeff_ = _ensure_unbatched_operator_input(
                    evaluate_node_callback(dir_coeff_func, self.xb, "boundary coefficient"),
                    "boundary coefficient",
                )
                self.fixed_bc_coefficients_ready_ = True
            return self.cached_neu_coeff_, self.cached_dir_coeff_
        return (
            _ensure_unbatched_operator_input(evaluate_node_callback(neu_coeff_func, self.xb, "boundary coefficient"), "boundary coefficient"),
            _ensure_unbatched_operator_input(evaluate_node_callback(dir_coeff_func, self.xb, "boundary coefficient"), "boundary coefficient"),
        )

    def _ensure_boundary_operator(self, neu_coeff: jnp.ndarray, dir_coeff: jnp.ndarray) -> None:
        if self.fixed_bc_operator_ready_ and self.bc.shape[0] > 0:
            return
        self.bc = assemble_operator(
            self.domain,
            self.bc_assembler,
            self.bc_stencil,
            "bc",
            self.bc_stencil_properties,
            self.bc_op_properties,
            neu_coeff=neu_coeff,
            dir_coeff=dir_coeff,
        )
        self.bc_sparse = assemble_operator_sparse(
            self.domain,
            self.bc_assembler,
            self.bc_stencil,
            "bc",
            self.bc_stencil_properties,
            self.bc_op_properties,
            neu_coeff=neu_coeff,
            dir_coeff=dir_coeff,
        )
        if self.fixed_bc_coefficients_ready_:
            self.fixed_bc_operator_ready_ = True


    def prepare_boundary_operator(
        self,
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
    ) -> None:
        """Assemble and cache the fixed boundary operator outside a traced solve."""

        neu_coeff, dir_coeff = self._get_boundary_coefficients(neu_coeff_func, dir_coeff_func)
        self._ensure_boundary_operator(neu_coeff, dir_coeff)

    def solve(
        self,
        forcing: Callable[..., jnp.ndarray] | jnp.ndarray,
        coeff: Callable[..., jnp.ndarray] | jnp.ndarray,
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        bc: Callable[..., jnp.ndarray] | jnp.ndarray,
        initial_guess: jnp.ndarray | None = None,
    ) -> dict[str, object]:
        if initial_guess is None:
            initial_guess = jnp.zeros(0)
        coeff_all = evaluate_node_callback(coeff, self.xf, "coefficient")
        coeff_all = _ensure_unbatched_operator_input(coeff_all, "coefficient")
        if not isinstance(coeff_all, jax_core.Tracer) and bool(jnp.any(coeff_all <= 0)):
            raise ValueError("VariablePoissonSolver expects a positive scalar coefficient field.")

        self.pde = _build_variable_pde_operator(self.lap, self.grad, coeff_all, self.n)
        if self.lap_sparse is not None:
            self.pde_sparse = _build_variable_pde_operator_sparse(self.lap_sparse, self.grad_sparse, coeff_all, self.n)

        neu_coeff, dir_coeff = self._get_boundary_coefficients(neu_coeff_func, dir_coeff_func)
        self._ensure_boundary_operator(neu_coeff, dir_coeff)

        rhs_target = evaluate_node_callback(forcing, self.x, "forcing")
        rhs_boundary = evaluate_boundary_values(bc, neu_coeff, dir_coeff, self.nr, self.xb)
        pure_neumann = self.cached_pure_neumann_ if self.fixed_bc_coefficients_ready_ else (
            True if isinstance(dir_coeff, jax_core.Tracer) else float(jnp.max(jnp.abs(dir_coeff))) <= 1e-13
        )
        self.last_solve_used_nullspace_ = pure_neumann

        system = _build_variable_system_matrix(self.pde, self.bc, self.nf, pure_neumann)
        sparse_system = None if self.pde_sparse is None or self.bc_sparse is None else _build_variable_system_matrix_sparse(self.pde_sparse, self.bc_sparse, self.nf, pure_neumann)
        rhs = build_system_rhs(rhs_target, rhs_boundary, pure_neumann)
        guess = build_initial_guess(initial_guess, self.n, self.nf, rhs_boundary, pure_neumann)

        if self.linear_solver == "dense":
            sol = solve_dense_system(system, rhs) if guess is None else gmres_with_fallback(system, rhs, guess)
            self.fixed_preconditioner = None
        elif self.linear_solver == "lineax":
            dense_system = system if system.size else sparse_matrix_to_dense(sparse_system)
            sol = _solve_lineax_system(dense_system, rhs)
            self.fixed_preconditioner = None
        elif self.linear_solver == "gmres_ilu":
            if sparse_system is None:
                raise ValueError("sparse solve requires sparse operator assembly")
            if self.fixed_preconditioner is None:
                if isinstance(sparse_system.values, jax_core.Tracer):
                    raise ValueError(
                        "VariablePoissonSolver linear_solver='gmres_ilu' requires a frozen preconditioner when solving inside a differentiated trace. "
                        "Call prepare_preconditioner(...) outside the differentiated path first."
                    )
                self.fixed_preconditioner = build_fixed_ilu_preconditioner(sparse_system)
            sol = solve_sparse_system_gmres(sparse_system, rhs, guess=guess, preconditioner=self.fixed_preconditioner)
        else:
            raise ValueError(f'unknown linear solver {self.linear_solver}')

        if pure_neumann:
            full_state = sol[: self.nf]
            lagrange_multiplier = sol[-1] if sol.ndim == 1 else sol[-1, :]
        else:
            full_state = sol
            lagrange_multiplier = None

        return {
            "u": jnp.asarray(full_state[: self.n], dtype=float),
            "full_state": jnp.asarray(full_state, dtype=float),
            "coefficient": coeff_all,
            "L": self.lap,
            "Grad": self.grad,
            "PDE": self.pde,
            "PDE_sparse": self.pde_sparse,
            "BC": self.bc,
            "BC_sparse": self.bc_sparse,
            "system_matrix": system,
            "system_matrix_sparse": sparse_system,
            "system_matrix_dense_from_sparse": None if sparse_system is None else sparse_matrix_to_dense(sparse_system),
            "rhs": rhs,
            "target_rhs": rhs_target,
            "boundary_rhs": rhs_boundary,
            "used_nullspace_augmentation": pure_neumann,
            "lagrange_multiplier": lagrange_multiplier,
            "linear_solver": self.linear_solver,
        }

    def get_laplacian(self) -> jnp.ndarray:
        return self.lap

    def get_gradient_ops(self) -> tuple[jnp.ndarray, ...]:
        return self.grad

    def get_last_pde_operator(self) -> jnp.ndarray:
        return self.pde

    def get_bc_op(self) -> jnp.ndarray:
        return self.bc

    def last_solve_used_nullspace(self) -> bool:
        return self.last_solve_used_nullspace_

    def clear_preconditioner(self) -> None:
        self.fixed_preconditioner = None

    def prepare_preconditioner(
        self,
        coeff: Callable[..., jnp.ndarray] | jnp.ndarray,
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
    ) -> FrozenILUPreconditioner:
        coeff_all = evaluate_node_callback(coeff, self.xf, "coefficient")
        coeff_all = _ensure_unbatched_operator_input(coeff_all, "coefficient")
        if bool(jnp.any(coeff_all <= 0)):
            raise ValueError("VariablePoissonSolver expects a positive scalar coefficient field.")
        pde_sparse = _build_variable_pde_operator_sparse(self.lap_sparse, self.grad_sparse, coeff_all, self.n)
        neu_coeff = evaluate_node_callback(neu_coeff_func, self.xb, "boundary coefficient")
        dir_coeff = evaluate_node_callback(dir_coeff_func, self.xb, "boundary coefficient")
        neu_coeff = _ensure_unbatched_operator_input(neu_coeff, "boundary coefficient")
        dir_coeff = _ensure_unbatched_operator_input(dir_coeff, "boundary coefficient")
        bc_dense = assemble_operator(
            self.domain,
            self.bc_assembler,
            self.bc_stencil,
            "bc",
            self.bc_stencil_properties,
            self.bc_op_properties,
            neu_coeff=neu_coeff,
            dir_coeff=dir_coeff,
        )
        bc_sparse = assemble_operator_sparse(
            self.domain,
            self.bc_assembler,
            self.bc_stencil,
            "bc",
            self.bc_stencil_properties,
            self.bc_op_properties,
            neu_coeff=neu_coeff,
            dir_coeff=dir_coeff,
        )
        # Reuse the same fixed boundary operator during traced solve calls so
        # autodiff never falls back into dynamic operator assembly.
        self.cached_neu_coeff_ = neu_coeff
        self.cached_dir_coeff_ = dir_coeff
        self.cached_pure_neumann_ = float(jnp.max(jnp.abs(dir_coeff))) <= 1e-13
        self.fixed_bc_coefficients_ready_ = True
        self.bc = bc_dense
        self.bc_sparse = bc_sparse
        self.fixed_bc_operator_ready_ = True
        pure_neumann = self.cached_pure_neumann_
        sparse_system = _build_variable_system_matrix_sparse(pde_sparse, bc_sparse, self.nf, pure_neumann)
        self.fixed_preconditioner = build_fixed_ilu_preconditioner(sparse_system)
        return self.fixed_preconditioner

    def freeze(self) -> "StatelessVariablePoissonSolver":
        """Snapshot the prepared solver into a stateless, solve-only wrapper."""

        if self.bc.shape[0] == 0 or self.bc_sparse is None:
            raise ValueError("Call prepare_boundary_operator(...) before freeze()")
        if self.linear_solver == "gmres_ilu" and self.fixed_preconditioner is None:
            raise ValueError("Call prepare_preconditioner(...) before freeze() when using gmres_ilu")
        if not self.fixed_bc_coefficients_ready_:
            raise ValueError("freeze() requires fixed boundary coefficients to be prepared first")
        return StatelessVariablePoissonSolver.from_solver(self)


@dataclass(frozen=True)
class StatelessVariablePoissonSolver:
    """Frozen variable Poisson solve context with no in-solve mutation.

    All fixed spatial operators and boundary data are captured once up front,
    so repeated differentiable evaluations only rebuild the
    coefficient-dependent PDE block and the linear system. This remains an
    implementation detail rather than part of the public solver namespace.
    """

    domain: DomainDescriptor
    lap_assembler: str
    bc_assembler: str
    lap_stencil: str
    bc_stencil: str
    linear_solver: str
    x: jnp.ndarray
    xb: jnp.ndarray
    xf: jnp.ndarray
    nr: jnp.ndarray
    n: int
    nf: int
    lap: jnp.ndarray
    lap_sparse: SparseCOOMatrix | None
    grad: tuple[jnp.ndarray, ...]
    grad_sparse: tuple[SparseCOOMatrix, ...]
    bc: jnp.ndarray
    bc_sparse: SparseCOOMatrix | None
    fixed_preconditioner: FrozenILUPreconditioner | None
    cached_neu_coeff_: jnp.ndarray
    cached_dir_coeff_: jnp.ndarray
    cached_pure_neumann_: bool

    @classmethod
    def from_solver(cls, solver: VariablePoissonSolver) -> "StatelessVariablePoissonSolver":
        if solver.n <= 0:
            raise ValueError("StatelessVariablePoissonSolver requires an initialized VariablePoissonSolver")
        if solver.bc.shape[0] == 0 or solver.bc_sparse is None:
            raise ValueError("Call prepare_boundary_operator(...) on the base solver before freezing it")
        return cls(
            domain=solver.domain,
            lap_assembler=solver.lap_assembler,
            bc_assembler=solver.bc_assembler,
            lap_stencil=solver.lap_stencil,
            bc_stencil=solver.bc_stencil,
            linear_solver=solver.linear_solver,
            x=solver.x,
            xb=solver.xb,
            xf=solver.xf,
            nr=solver.nr,
            n=solver.n,
            nf=solver.nf,
            lap=solver.lap,
            lap_sparse=solver.lap_sparse,
            grad=solver.grad,
            grad_sparse=solver.grad_sparse,
            bc=solver.bc,
            bc_sparse=solver.bc_sparse,
            fixed_preconditioner=solver.fixed_preconditioner,
            cached_neu_coeff_=solver.cached_neu_coeff_,
            cached_dir_coeff_=solver.cached_dir_coeff_,
            cached_pure_neumann_=solver.cached_pure_neumann_,
        )

    def solve(
        self,
        forcing: Callable[..., jnp.ndarray] | jnp.ndarray,
        coeff: Callable[..., jnp.ndarray] | jnp.ndarray,
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        bc: Callable[..., jnp.ndarray] | jnp.ndarray,
        initial_guess: jnp.ndarray | None = None,
    ) -> dict[str, object]:
        del neu_coeff_func, dir_coeff_func
        if initial_guess is None:
            initial_guess = jnp.zeros(0)
        coeff_all = evaluate_node_callback(coeff, self.xf, "coefficient")
        coeff_all = _ensure_unbatched_operator_input(coeff_all, "coefficient")
        if not isinstance(coeff_all, jax_core.Tracer) and bool(jnp.any(coeff_all <= 0)):
            raise ValueError("VariablePoissonSolver expects a positive scalar coefficient field.")

        pde = _build_variable_pde_operator(self.lap, self.grad, coeff_all, self.n)
        pde_sparse = None if self.lap_sparse is None else _build_variable_pde_operator_sparse(self.lap_sparse, self.grad_sparse, coeff_all, self.n)
        pure_neumann = self.cached_pure_neumann_
        system = _build_variable_system_matrix(pde, self.bc, self.nf, pure_neumann)
        sparse_system = None if pde_sparse is None or self.bc_sparse is None else _build_variable_system_matrix_sparse(pde_sparse, self.bc_sparse, self.nf, pure_neumann)
        rhs_target = evaluate_node_callback(forcing, self.x, "forcing")
        rhs_boundary = evaluate_boundary_values(bc, self.cached_neu_coeff_, self.cached_dir_coeff_, self.nr, self.xb)
        rhs = build_system_rhs(rhs_target, rhs_boundary, pure_neumann)
        guess = build_initial_guess(initial_guess, self.n, self.nf, rhs_boundary, pure_neumann)

        if self.linear_solver == "dense":
            sol = solve_dense_system(system, rhs) if guess is None else gmres_with_fallback(system, rhs, guess)
        elif self.linear_solver == "lineax":
            dense_system = system if system.size else sparse_matrix_to_dense(sparse_system)
            sol = _solve_lineax_system(dense_system, rhs)
        elif self.linear_solver == "gmres_ilu":
            if sparse_system is None:
                raise ValueError("sparse solve requires sparse operator assembly")
            if self.fixed_preconditioner is None:
                raise ValueError("StatelessVariablePoissonSolver requires a frozen preconditioner for gmres_ilu")
            sol = solve_sparse_system_gmres(sparse_system, rhs, guess=guess, preconditioner=self.fixed_preconditioner)
        else:
            raise ValueError(f"unknown linear solver {self.linear_solver}")

        if pure_neumann:
            full_state = sol[: self.nf]
            lagrange_multiplier = sol[-1] if sol.ndim == 1 else sol[-1, :]
        else:
            full_state = sol
            lagrange_multiplier = None

        return {
            "u": jnp.asarray(full_state[: self.n], dtype=float),
            "full_state": jnp.asarray(full_state, dtype=float),
            "coefficient": coeff_all,
            "L": self.lap,
            "Grad": self.grad,
            "PDE": pde,
            "PDE_sparse": pde_sparse,
            "BC": self.bc,
            "BC_sparse": self.bc_sparse,
            "system_matrix": system,
            "system_matrix_sparse": sparse_system,
            "system_matrix_dense_from_sparse": None,
            "rhs": rhs,
            "target_rhs": rhs_target,
            "boundary_rhs": rhs_boundary,
            "used_nullspace_augmentation": pure_neumann,
            "lagrange_multiplier": lagrange_multiplier,
            "linear_solver": self.linear_solver,
        }

    def solve_from_fields(
        self,
        coeff_all: jnp.ndarray,
        rhs_target: jnp.ndarray,
        rhs_boundary: jnp.ndarray,
        initial_guess: jnp.ndarray | None = None,
    ) -> dict[str, object]:
        if initial_guess is None:
            initial_guess = jnp.zeros(0)
        coeff_all = _ensure_unbatched_operator_input(jnp.asarray(coeff_all, dtype=float), "coefficient")
        rhs_target = jnp.asarray(rhs_target, dtype=float)
        rhs_boundary = jnp.asarray(rhs_boundary, dtype=float)
        if not isinstance(coeff_all, jax_core.Tracer) and bool(jnp.any(coeff_all <= 0)):
            raise ValueError("VariablePoissonSolver expects a positive scalar coefficient field.")

        pure_neumann = self.cached_pure_neumann_
        rhs = build_system_rhs(rhs_target, rhs_boundary, pure_neumann)
        guess = build_initial_guess(initial_guess, self.n, self.nf, rhs_boundary, pure_neumann)
        pde = None
        pde_sparse = None
        system = None
        sparse_system = None

        if self.linear_solver == "dense":
            pde = _build_variable_pde_operator(self.lap, self.grad, coeff_all, self.n)
            system = _build_variable_system_matrix(pde, self.bc, self.nf, pure_neumann)
            sol = solve_dense_system(system, rhs) if guess is None else gmres_with_fallback(system, rhs, guess)
        elif self.linear_solver == "lineax":
            pde = _build_variable_pde_operator(self.lap, self.grad, coeff_all, self.n)
            system = _build_variable_system_matrix(pde, self.bc, self.nf, pure_neumann)
            sol = _solve_lineax_system(system, rhs)
        elif self.linear_solver == "gmres_ilu":
            if self.lap_sparse is None or self.bc_sparse is None:
                raise ValueError("sparse solve requires sparse operator assembly")
            pde_sparse = _build_variable_pde_operator_sparse(self.lap_sparse, self.grad_sparse, coeff_all, self.n)
            sparse_system = _build_variable_system_matrix_sparse(pde_sparse, self.bc_sparse, self.nf, pure_neumann)
            if sparse_system is None:
                raise ValueError("sparse solve requires sparse operator assembly")
            if self.fixed_preconditioner is None:
                raise ValueError("StatelessVariablePoissonSolver requires a frozen preconditioner for gmres_ilu")
            sol = solve_sparse_system_gmres(sparse_system, rhs, guess=guess, preconditioner=self.fixed_preconditioner)
        else:
            raise ValueError(f"unknown linear solver {self.linear_solver}")

        if pure_neumann:
            full_state = sol[: self.nf]
            lagrange_multiplier = sol[-1] if sol.ndim == 1 else sol[-1, :]
        else:
            full_state = sol
            lagrange_multiplier = None

        return {
            "u": jnp.asarray(full_state[: self.n], dtype=float),
            "full_state": jnp.asarray(full_state, dtype=float),
            "coefficient": coeff_all,
            "L": self.lap,
            "Grad": self.grad,
            "PDE": pde,
            "PDE_sparse": pde_sparse,
            "BC": self.bc,
            "BC_sparse": self.bc_sparse,
            "system_matrix": system,
            "system_matrix_sparse": sparse_system,
            "system_matrix_dense_from_sparse": None,
            "rhs": rhs,
            "target_rhs": rhs_target,
            "boundary_rhs": rhs_boundary,
            "used_nullspace_augmentation": pure_neumann,
            "lagrange_multiplier": lagrange_multiplier,
            "linear_solver": self.linear_solver,
        }

    def solve_u_from_fields(
        self,
        coeff_all: jnp.ndarray,
        rhs_target: jnp.ndarray,
        rhs_boundary: jnp.ndarray,
        initial_guess: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        if initial_guess is None:
            initial_guess = jnp.zeros(0)
        coeff_all = _ensure_unbatched_operator_input(jnp.asarray(coeff_all, dtype=float), "coefficient")
        rhs_target = jnp.asarray(rhs_target, dtype=float)
        rhs_boundary = jnp.asarray(rhs_boundary, dtype=float)
        if not isinstance(coeff_all, jax_core.Tracer) and bool(jnp.any(coeff_all <= 0)):
            raise ValueError("VariablePoissonSolver expects a positive scalar coefficient field.")

        pure_neumann = self.cached_pure_neumann_
        rhs = build_system_rhs(rhs_target, rhs_boundary, pure_neumann)
        guess = build_initial_guess(initial_guess, self.n, self.nf, rhs_boundary, pure_neumann)

        if self.linear_solver == "dense":
            pde = _build_variable_pde_operator(self.lap, self.grad, coeff_all, self.n)
            system = _build_variable_system_matrix(pde, self.bc, self.nf, pure_neumann)
            sol = solve_dense_system(system, rhs) if guess is None else gmres_with_fallback(system, rhs, guess)
        elif self.linear_solver == "lineax":
            pde = _build_variable_pde_operator(self.lap, self.grad, coeff_all, self.n)
            system = _build_variable_system_matrix(pde, self.bc, self.nf, pure_neumann)
            sol = _solve_lineax_system(system, rhs)
        elif self.linear_solver == "gmres_ilu":
            if self.lap_sparse is None or self.bc_sparse is None:
                raise ValueError("sparse solve requires sparse operator assembly")
            if self.fixed_preconditioner is None:
                raise ValueError("StatelessVariablePoissonSolver requires a frozen preconditioner for gmres_ilu")
            pde_sparse = _build_variable_pde_operator_sparse(self.lap_sparse, self.grad_sparse, coeff_all, self.n)
            sparse_system = _build_variable_system_matrix_sparse(pde_sparse, self.bc_sparse, self.nf, pure_neumann)
            sol = solve_sparse_system_gmres(sparse_system, rhs, guess=guess, preconditioner=self.fixed_preconditioner)
        else:
            raise ValueError(f"unknown linear solver {self.linear_solver}")

        full_state = sol[: self.nf] if pure_neumann else sol
        return jnp.asarray(full_state[: self.n], dtype=float)
