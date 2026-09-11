from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax.numpy as jnp

from kernelpack.domain import DomainDescriptor
from kernelpack.rbffd import OpProperties, StencilProperties
from ._common import (
    assemble_operator,
    assemble_operator_sparse,
    build_domain_state,
    build_fixed_discretization,
    build_initial_guess,
    build_fixed_ilu_preconditioner,
    build_stencil_properties,
    build_system_matrix,
    build_system_matrix_sparse,
    build_system_rhs,
    evaluate_boundary_values,
    evaluate_node_callback,
    FixedDiscretization,
    FrozenILUPreconditioner,
    gmres_with_fallback,
    _ensure_unbatched_operator_input,
    SparseCOOMatrix,
    sparse_matrix_to_dense,
    solve_sparse_system_gmres,
    solve_dense_system,
)


@dataclass
class PoissonSolver:
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
    nr: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    n: int = 0
    nf: int = 0
    lap: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    bc: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    lap_sparse: SparseCOOMatrix | None = None
    bc_sparse: SparseCOOMatrix | None = None
    lap_stencil_properties: StencilProperties = field(default_factory=StencilProperties)
    bc_stencil_properties: StencilProperties = field(default_factory=StencilProperties)
    lap_op_properties: OpProperties = field(default_factory=lambda: OpProperties(decompose=False, store_weights=True, record_stencils=False))
    bc_op_properties: OpProperties = field(default_factory=lambda: OpProperties(decompose=False, store_weights=True, record_stencils=False))
    fixed_discretization: FixedDiscretization | None = None
    fixed_preconditioner: FrozenILUPreconditioner | None = None
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
        self.lap_stencil_properties = build_stencil_properties(self.domain, self.xi, 2, "interior_boundary")
        self.bc_stencil_properties = build_stencil_properties(self.domain, self.xi, 1, "boundary")
        self.fixed_discretization = build_fixed_discretization(
            self.domain,
            self.lap_stencil_properties,
            self.bc_stencil_properties,
        )
        self.lap = assemble_operator(
            self.domain,
            self.lap_assembler,
            self.lap_stencil,
            "lap",
            self.lap_stencil_properties,
            self.lap_op_properties,
            stencil_graph=self.fixed_discretization.laplacian_graph,
        )
        self.lap_sparse = assemble_operator_sparse(
            self.domain,
            self.lap_assembler,
            self.lap_stencil,
            "lap",
            self.lap_stencil_properties,
            self.lap_op_properties,
            stencil_graph=self.fixed_discretization.laplacian_graph,
        )
        self.bc = jnp.zeros((0, self.nf))
        self.bc_sparse = None
        self.fixed_preconditioner = None
        self.last_solve_used_nullspace_ = False

    def solve(
        self,
        forcing: Callable[..., jnp.ndarray] | jnp.ndarray,
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        bc: Callable[..., jnp.ndarray] | jnp.ndarray,
        initial_guess: jnp.ndarray | None = None,
    ) -> dict[str, object]:
        if initial_guess is None:
            initial_guess = jnp.zeros(0)
        neu_coeff = evaluate_node_callback(neu_coeff_func, self.xb, "boundary coefficient")
        dir_coeff = evaluate_node_callback(dir_coeff_func, self.xb, "boundary coefficient")
        neu_coeff = _ensure_unbatched_operator_input(neu_coeff, "boundary coefficient")
        dir_coeff = _ensure_unbatched_operator_input(dir_coeff, "boundary coefficient")
        self.bc = assemble_operator(
            self.domain,
            self.bc_assembler,
            self.bc_stencil,
            "bc",
            self.bc_stencil_properties,
            self.bc_op_properties,
            neu_coeff=neu_coeff,
            dir_coeff=dir_coeff,
            stencil_graph=None if self.fixed_discretization is None else self.fixed_discretization.boundary_graph,
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
            stencil_graph=None if self.fixed_discretization is None else self.fixed_discretization.boundary_graph,
        )
        rhs_target = evaluate_node_callback(forcing, self.x, "forcing")
        rhs_boundary = evaluate_boundary_values(bc, neu_coeff, dir_coeff, self.nr, self.xb)
        pure_neumann = float(jnp.max(jnp.abs(dir_coeff))) <= 1e-13
        self.last_solve_used_nullspace_ = pure_neumann
        system = build_system_matrix(self.lap, self.bc, self.nf, pure_neumann)
        sparse_system = None if self.lap_sparse is None or self.bc_sparse is None else build_system_matrix_sparse(self.lap_sparse, self.bc_sparse, self.nf, pure_neumann)
        rhs = build_system_rhs(rhs_target, rhs_boundary, pure_neumann)
        guess = build_initial_guess(initial_guess, self.n, self.nf, rhs_boundary, pure_neumann)
        if self.linear_solver == "dense":
            sol = solve_dense_system(system, rhs) if guess is None else gmres_with_fallback(system, rhs, guess)
            self.fixed_preconditioner = None
        elif self.linear_solver == "gmres_ilu":
            if sparse_system is None:
                raise ValueError("sparse solve requires sparse operator assembly")
            self.fixed_preconditioner = build_fixed_ilu_preconditioner(sparse_system)
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
            "L": self.lap,
            "BC": self.bc,
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

    def get_bc_op(self) -> jnp.ndarray:
        return self.bc

    def last_solve_used_nullspace(self) -> bool:
        return self.last_solve_used_nullspace_
