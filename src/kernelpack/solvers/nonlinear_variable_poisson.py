from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from jax import jvp, lax, vjp
from jax import scipy as jsp
import jax.numpy as jnp
from jax import core as jax_core

from kernelpack.domain import DomainDescriptor
from kernelpack.rbffd import OpProperties, StencilProperties
from ._common import (
    assemble_operator,
    assemble_operator_sparse,
    build_domain_state,
    build_fixed_ilu_preconditioner,
    build_stencil_properties,
    solve_sparse_system_gmres,
    sparse_matrix_to_dense,
    SparseCOOMatrix,
    FrozenILUPreconditioner,
    evaluate_boundary_values,
    evaluate_node_callback,
)
from .variable_poisson import _add_sparse_matrices, _build_variable_pde_operator, _build_variable_pde_operator_sparse, _build_variable_system_matrix_sparse, _scale_sparse_rows


def _diagonal_sparse(diag_values: jnp.ndarray, n_rows: int, n_cols: int) -> SparseCOOMatrix:
    count = min(int(diag_values.shape[0]), n_rows, n_cols)
    idx = jnp.arange(count, dtype=int)
    return SparseCOOMatrix(
        indices=jnp.column_stack([idx, idx]),
        values=jnp.asarray(diag_values[:count], dtype=float),
        shape=(n_rows, n_cols),
    )


def _build_preconditioner_pde_sparse(
    lap_sparse: SparseCOOMatrix,
    grad_sparse: tuple[SparseCOOMatrix, ...],
    coeff_all: jnp.ndarray,
    forcing_u_local: jnp.ndarray,
    n_rows: int,
) -> SparseCOOMatrix:
    pde = _build_variable_pde_operator_sparse(lap_sparse, grad_sparse, coeff_all, n_rows)
    diag = _diagonal_sparse(-forcing_u_local, n_rows, lap_sparse.shape[1])
    return _add_sparse_matrices(pde, diag)


def _build_nonlinear_initial_state(initial_guess: jnp.ndarray, n: int, nf: int, pure_neumann: bool) -> jnp.ndarray:
    guess = jnp.asarray(initial_guess, dtype=float).reshape(-1)
    if guess.size == 0:
        return jnp.zeros((nf + (1 if pure_neumann else 0),), dtype=float)
    if pure_neumann:
        if guess.size == n:
            return jnp.concatenate([guess, jnp.zeros((nf - n + 1,), dtype=float)])
        if guess.size == nf:
            return jnp.concatenate([guess, jnp.array([0.0])])
        if guess.size == nf + 1:
            return guess
        raise ValueError(f"nonlinear variable Poisson guess must have length {n}, {nf}, or {nf + 1}")
    if guess.size == n:
        return jnp.concatenate([guess, jnp.zeros((nf - n,), dtype=float)])
    if guess.size == nf:
        return guess
    raise ValueError(f"nonlinear variable Poisson guess must have length {n} or {nf}")


def _solve_linearized_operator_gmres(
    state: jnp.ndarray,
    rhs: jnp.ndarray,
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    preconditioner: FrozenILUPreconditioner | None,
    tol: float,
    atol: float,
) -> jnp.ndarray:
    def matvec(x: jnp.ndarray) -> jnp.ndarray:
        return jvp(residual_fn, (state,), (x,))[1]

    _, pullback = vjp(residual_fn, state)

    def vecmat(x: jnp.ndarray) -> jnp.ndarray:
        return pullback(x)[0]

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
            x0=jnp.zeros_like(target),
            tol=tol,
            atol=atol,
            restart=40,
            maxiter=None,
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
            restart=40,
            maxiter=None,
            M=transpose_precond,
            solve_method="batched",
        )
        return sol

    return lax.custom_linear_solve(matvec, rhs, solve, transpose_solve=transpose_solve, symmetric=False)


@dataclass
class NonlinearVariablePoissonSolver:
    lap_assembler: str = "fd"
    bc_assembler: str = "fd"
    lap_stencil: str = "rbf"
    bc_stencil: str = "rbf"
    domain: DomainDescriptor = field(default_factory=DomainDescriptor)
    xi: int = 0
    num_omp_threads: int = 1
    linear_solver: str = "gmres_ilu"
    nonlinear_tol: float = 1.0e-9
    linear_tol: float = 1.0e-9
    max_nonlinear_iterations: int = 20
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
    last_solve_used_nullspace_: bool = False
    last_nonlinear_iterations_: int = 0
    last_residual_norm_: float = 0.0

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
        self.last_solve_used_nullspace_ = False
        self.last_nonlinear_iterations_ = 0
        self.last_residual_norm_ = 0.0

    def _assemble_boundary_operator(
        self,
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        neu_coeff = evaluate_node_callback(neu_coeff_func, self.xb, "boundary coefficient")
        dir_coeff = evaluate_node_callback(dir_coeff_func, self.xb, "boundary coefficient")
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
        return neu_coeff, dir_coeff

    def _evaluate_residual(
        self,
        state_all: jnp.ndarray,
        forcing: Callable[..., jnp.ndarray],
        coeff: Callable[..., jnp.ndarray],
        bc_rhs: jnp.ndarray,
        pure_neumann: bool,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        coeff_all = jnp.asarray(coeff(self.xf, state_all), dtype=float).reshape(-1)
        if coeff_all.size != self.nf:
            raise ValueError("NonlinearVariablePoissonSolver coefficient callback must return one value per all-node unknown.")
        self.pde = _build_variable_pde_operator(self.lap, self.grad, coeff_all, self.n)
        self.pde_sparse = _build_variable_pde_operator_sparse(self.lap_sparse, self.grad_sparse, coeff_all, self.n)
        state_target = state_all[: self.n]
        forcing_local = jnp.asarray(forcing(self.x, state_target), dtype=float).reshape(-1)
        if forcing_local.size != self.n:
            raise ValueError("NonlinearVariablePoissonSolver forcing callback returned the wrong number of target-row values.")
        residual_target = self.pde @ state_all - forcing_local
        residual_boundary = self.bc @ state_all - bc_rhs
        if pure_neumann:
            lambda_value = state_all[-1]
            residual_target = residual_target + lambda_value
            residual_boundary = residual_boundary + lambda_value
            gauge = jnp.sum(state_all[:-1])
            residual = jnp.concatenate([residual_target, residual_boundary, jnp.array([gauge])])
        else:
            residual = jnp.concatenate([residual_target, residual_boundary])
        return residual, coeff_all, residual_target, residual_boundary

    def _residual_only(
        self,
        state: jnp.ndarray,
        forcing: Callable[..., jnp.ndarray],
        coeff: Callable[..., jnp.ndarray],
        bc_rhs: jnp.ndarray,
        pure_neumann: bool,
    ) -> jnp.ndarray:
        physical_state = state[:-1] if pure_neumann else state
        coeff_all = jnp.asarray(coeff(self.xf, physical_state), dtype=float).reshape(-1)
        pde = _build_variable_pde_operator(self.lap, self.grad, coeff_all, self.n)
        forcing_local = jnp.asarray(forcing(self.x, physical_state[: self.n]), dtype=float).reshape(-1)
        residual_target = pde @ physical_state - forcing_local
        residual_boundary = self.bc @ physical_state - bc_rhs
        if pure_neumann:
            lambda_value = state[-1]
            residual_target = residual_target + lambda_value
            residual_boundary = residual_boundary + lambda_value
            gauge = jnp.sum(physical_state)
            return jnp.concatenate([residual_target, residual_boundary, jnp.array([gauge])])
        return jnp.concatenate([residual_target, residual_boundary])

    def _assemble_preconditioner_matrix(
        self,
        state_all: jnp.ndarray,
        coeff: Callable[..., jnp.ndarray],
        forcing_u: Callable[..., jnp.ndarray],
        pure_neumann: bool,
    ) -> SparseCOOMatrix:
        coeff_all = jnp.asarray(coeff(self.xf, state_all), dtype=float).reshape(-1)
        if coeff_all.size != self.nf:
            raise ValueError("NonlinearVariablePoissonSolver coefficient callback must return one value per all-node unknown.")
        forcing_u_local = jnp.asarray(forcing_u(self.x, state_all[: self.n]), dtype=float).reshape(-1)
        if forcing_u_local.size != self.n:
            raise ValueError("NonlinearVariablePoissonSolver forcing derivative callback returned the wrong number of target-row values.")
        pde_prec = _build_preconditioner_pde_sparse(self.lap_sparse, self.grad_sparse, coeff_all, forcing_u_local, self.n)
        return _build_variable_system_matrix_sparse(pde_prec, self.bc_sparse, self.nf, pure_neumann)

    def solve(
        self,
        forcing: Callable[..., jnp.ndarray],
        forcing_u: Callable[..., jnp.ndarray],
        coeff: Callable[..., jnp.ndarray],
        coeff_u: Callable[..., jnp.ndarray],
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        bc: Callable[..., jnp.ndarray] | jnp.ndarray,
        initial_guess: jnp.ndarray | None = None,
    ) -> dict[str, object]:
        if initial_guess is None:
            initial_guess = jnp.zeros(0)
        neu_coeff, dir_coeff = self._assemble_boundary_operator(neu_coeff_func, dir_coeff_func)
        bc_rhs = evaluate_boundary_values(bc, neu_coeff, dir_coeff, self.nr, self.xb)
        pure_neumann = float(jnp.max(jnp.abs(dir_coeff))) <= 1.0e-13
        self.last_solve_used_nullspace_ = pure_neumann

        state = _build_nonlinear_initial_state(initial_guess, self.n, self.nf, pure_neumann)
        residual, coeff_all, residual_target, residual_boundary = self._evaluate_residual(
            state[:-1] if pure_neumann else state,
            forcing,
            coeff,
            bc_rhs,
            pure_neumann,
        )
        residual_norm = jnp.linalg.norm(residual)

        if self.linear_solver == "gmres_ilu" and self.fixed_preconditioner is None:
            preconditioner_matrix = self._assemble_preconditioner_matrix(state[:-1] if pure_neumann else state, coeff, forcing_u, pure_neumann)
            if isinstance(preconditioner_matrix.values, jax_core.Tracer):
                raise ValueError(
                    "NonlinearVariablePoissonSolver linear_solver='gmres_ilu' requires a frozen preconditioner when solving inside a differentiated trace. "
                    "Call prepare_preconditioner(...) outside the differentiated path first."
                )
            self.fixed_preconditioner = build_fixed_ilu_preconditioner(preconditioner_matrix)

        iterations = jnp.asarray(0, dtype=int)
        for _ in range(self.max_nonlinear_iterations):
            rhs = -residual
            if self.linear_solver != "gmres_ilu":
                raise ValueError("NonlinearVariablePoissonSolver currently supports only linear_solver='gmres_ilu'")
            residual_fn = lambda s: self._residual_only(s, forcing, coeff, bc_rhs, pure_neumann)
            delta = _solve_linearized_operator_gmres(
                state,
                rhs,
                residual_fn,
                preconditioner=self.fixed_preconditioner,
                tol=self.linear_tol,
                atol=min(self.linear_tol, 1.0e-12),
            )
            trial_state = state + delta
            trial_residual, trial_coeff, trial_target, trial_boundary = self._evaluate_residual(
                trial_state[:-1] if pure_neumann else trial_state,
                forcing,
                coeff,
                bc_rhs,
                pure_neumann,
            )
            active = residual_norm > self.nonlinear_tol
            state = jnp.where(active, trial_state, state)
            residual = jnp.where(active, trial_residual, residual)
            coeff_all = jnp.where(active, trial_coeff, coeff_all)
            residual_target = jnp.where(active, trial_target, residual_target)
            residual_boundary = jnp.where(active, trial_boundary, residual_boundary)
            iterations = iterations + active.astype(int)
            residual_norm = jnp.linalg.norm(residual)

        if not isinstance(iterations, jax_core.Tracer):
            self.last_nonlinear_iterations_ = int(iterations)
        if not isinstance(residual_norm, jax_core.Tracer):
            self.last_residual_norm_ = float(residual_norm)

        if pure_neumann:
            full_state = state[:-1]
            lagrange_multiplier = state[-1]
        else:
            full_state = state
            lagrange_multiplier = None

        return {
            "u": jnp.asarray(full_state[: self.n], dtype=float),
            "full_state": jnp.asarray(full_state, dtype=float),
            "coefficient": coeff_all,
            "PDE": self.pde,
            "PDE_sparse": self.pde_sparse,
            "BC": self.bc,
            "BC_sparse": self.bc_sparse,
            "residual": residual,
            "target_residual": residual_target,
            "boundary_residual": residual_boundary,
            "used_nullspace_augmentation": pure_neumann,
            "lagrange_multiplier": lagrange_multiplier,
            "linear_solver": self.linear_solver,
            "nonlinear_iterations": iterations,
            "residual_norm": residual_norm,
        }

    def clear_preconditioner(self) -> None:
        self.fixed_preconditioner = None

    def prepare_preconditioner(
        self,
        forcing_u: Callable[..., jnp.ndarray],
        coeff: Callable[..., jnp.ndarray],
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        initial_guess: jnp.ndarray | None = None,
    ) -> FrozenILUPreconditioner:
        if initial_guess is None:
            initial_guess = jnp.zeros(0)
        neu_coeff, dir_coeff = self._assemble_boundary_operator(neu_coeff_func, dir_coeff_func)
        pure_neumann = float(jnp.max(jnp.abs(dir_coeff))) <= 1.0e-13
        state = _build_nonlinear_initial_state(initial_guess, self.n, self.nf, pure_neumann)
        preconditioner_matrix = self._assemble_preconditioner_matrix(state[:-1] if pure_neumann else state, coeff, forcing_u, pure_neumann)
        self.fixed_preconditioner = build_fixed_ilu_preconditioner(preconditioner_matrix)
        return self.fixed_preconditioner

    def last_solve_used_nullspace(self) -> bool:
        return self.last_solve_used_nullspace_

    def get_last_nonlinear_iterations(self) -> int:
        return self.last_nonlinear_iterations_

    def get_last_residual_norm(self) -> float:
        return self.last_residual_norm_
