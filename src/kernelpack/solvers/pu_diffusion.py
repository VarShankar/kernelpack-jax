from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import Callable

from jax import lax
import jax.numpy as jnp

from kernelpack.domain import DomainDescriptor
from ._common import (
    build_implicit_rhs,
    build_implicit_system,
    build_implicit_system_sparse,
    build_fixed_ilu_preconditioner,
    evaluate_boundary_coefficient,
    evaluate_forcing_callback,
    evaluate_transient_boundary_values,
    is_fixed_boundary_callback,
    push_completed_step,
    solve_sparse_system_gmres,
    SparseCOOMatrix,
    FrozenILUPreconditioner,
    solve_dense_system,
    _ensure_unbatched_operator_input,
    validate_physical_state,
)
from ._pu import (
    PUPatchData,
    PUQueryGrouping,
    build_pu_query_grouping,
    pu_localized_operator,
    pu_localized_operator_sparse,
    pu_patch_geometry,
)


@dataclass
class PUDiffusionSolver:
    domain: DomainDescriptor = field(default_factory=DomainDescriptor)
    xi: int = 0
    dt: float = jnp.nan
    nu: float = jnp.nan
    num_omp_threads: int = 1
    linear_solver: str = "gmres_ilu"
    x: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    xb: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    nr: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    n: int = 0
    nf: int = 0
    patch_data: PUPatchData | None = None
    interior_query_grouping: PUQueryGrouping | None = None
    boundary_query_grouping: PUQueryGrouping | None = None
    lap: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    lap_sparse: SparseCOOMatrix | None = None
    bc: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    bc_sparse: SparseCOOMatrix | None = None
    cnm2: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    cnm1: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    cn: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    completed_steps_: int = 0
    fixed_bc_operator_ready_: bool = False
    fixed_bc_coefficients_ready_: bool = False
    cached_neu_coeff_: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    cached_dir_coeff_: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    fixed_preconditioner: FrozenILUPreconditioner | None = None

    def spawn_rollout_copy(self) -> PUDiffusionSolver:
        clone = copy(self)
        clone.cnm2 = jnp.zeros((self.n,), dtype=float)
        clone.cnm1 = jnp.zeros((self.n,), dtype=float)
        clone.cn = jnp.zeros((self.n,), dtype=float)
        clone.completed_steps_ = 0
        clone.fixed_bc_operator_ready_ = self.fixed_bc_operator_ready_
        clone.fixed_bc_coefficients_ready_ = self.fixed_bc_coefficients_ready_
        clone.cached_neu_coeff_ = self.cached_neu_coeff_
        clone.cached_dir_coeff_ = self.cached_dir_coeff_
        clone.fixed_preconditioner = self.fixed_preconditioner
        return clone

    def init(self, domain: DomainDescriptor, xi: int, dlt: float, d_coeff: float, num_omp_threads: int = 1) -> None:
        self.domain = domain
        self.xi = xi
        self.dt = dlt
        self.nu = d_coeff
        self.num_omp_threads = num_omp_threads

        self.domain.build_structs()
        self.x = self.domain.get_int_bdry_nodes()
        self.xb = self.domain.get_bdry_nodes()
        self.nr = self.domain.get_nrmls()
        self.n = int(self.x.shape[0])
        self.nf = int(self.domain.get_num_total_nodes())
        if self.nf != self.n + self.domain.get_num_bdry_nodes():
            raise ValueError("PUDiffusionSolver expects one ghost node per boundary node")

        self.patch_data = pu_patch_geometry(self.domain, self.xi)
        self.interior_query_grouping = build_pu_query_grouping(self.patch_data, self.x)
        self.boundary_query_grouping = build_pu_query_grouping(self.patch_data, self.xb)
        if self.linear_solver == "dense":
            self.lap = pu_localized_operator(
                self.domain,
                self.patch_data,
                self.xi,
                self.x,
                "lap",
                query_grouping=self.interior_query_grouping,
            )
            self.lap_sparse = None
        else:
            self.lap = jnp.zeros((0, 0), dtype=float)
            self.lap_sparse = pu_localized_operator_sparse(
                self.domain,
                self.patch_data,
                self.xi,
                self.x,
                "lap",
                query_grouping=self.interior_query_grouping,
            )
        self.bc = jnp.zeros((0, 0), dtype=float)
        self.bc_sparse = None
        self.cnm2 = jnp.zeros(0, dtype=float)
        self.cnm1 = jnp.zeros(0, dtype=float)
        self.cn = jnp.zeros(0, dtype=float)
        self.completed_steps_ = 0
        self.fixed_bc_operator_ready_ = False
        self.fixed_bc_coefficients_ready_ = False
        self.cached_neu_coeff_ = jnp.zeros(0, dtype=float)
        self.cached_dir_coeff_ = jnp.zeros(0, dtype=float)
        self.fixed_preconditioner = None

    def set_step_size(self, dlt: float) -> None:
        self.dt = dlt
        self.fixed_bc_operator_ready_ = False
        self.fixed_bc_coefficients_ready_ = False
        self.fixed_preconditioner = None

    def set_initial_state(self, c0: jnp.ndarray) -> None:
        self.cnm2 = validate_physical_state(c0, self.n)
        self.cnm1 = jnp.zeros_like(self.cnm2)
        self.cn = jnp.zeros_like(self.cnm2)
        self.completed_steps_ = 0

    def set_state_history(self, *states: jnp.ndarray) -> None:
        if len(states) == 1:
            self.set_initial_state(states[0])
        elif len(states) == 2:
            self.cnm2 = validate_physical_state(states[0], self.n)
            self.cnm1 = validate_physical_state(states[1], self.n)
            self.cn = jnp.zeros_like(self.cnm1)
            self.completed_steps_ = 1
        elif len(states) == 3:
            self.cnm2 = validate_physical_state(states[0], self.n)
            self.cnm1 = validate_physical_state(states[1], self.n)
            self.cn = validate_physical_state(states[2], self.n)
            self.completed_steps_ = 2
        else:
            raise ValueError("set_state_history expects one, two, or three physical states")

    def current_physical_state(self) -> jnp.ndarray:
        steps = jnp.minimum(jnp.asarray(self.completed_steps_, dtype=int), 2)
        return lax.switch(
            steps,
            [
                lambda _: self.cnm2,
                lambda _: self.cnm1,
                lambda _: self.cn,
            ],
            operand=None,
        )

    def prepare_preconditioner(
        self,
        t: float,
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        *,
        nu_ref: float | None = None,
        bdf_order: int = 1,
    ) -> FrozenILUPreconditioner:
        neu_coeff, dir_coeff = self._get_boundary_coefficients(t, neu_coeff_func, dir_coeff_func)
        self._ensure_boundary_operator(neu_coeff, dir_coeff)
        if self.lap_sparse is None or self.bc_sparse is None:
            raise ValueError("prepare_preconditioner requires sparse PU operator assembly")
        nu_value = self.nu if nu_ref is None else float(nu_ref)
        if bdf_order == 1:
            lap_scale = -nu_value * self.dt
        elif bdf_order == 2:
            lap_scale = -(2.0 / 3.0) * nu_value * self.dt
        elif bdf_order == 3:
            lap_scale = -(6.0 / 11.0) * nu_value * self.dt
        else:
            raise ValueError("bdf_order must be 1, 2, or 3")
        sparse_system = build_implicit_system_sparse(self.lap_sparse, self.bc_sparse, self.n, lap_scale)
        self.fixed_preconditioner = build_fixed_ilu_preconditioner(sparse_system)
        return self.fixed_preconditioner

    def returns_distributed_state(self) -> bool:
        return False

    def get_output_range(self) -> tuple[int, int]:
        return (0, int(self.x.shape[0]))

    def get_output_nodes(self) -> jnp.ndarray:
        return self.x

    def bdf1_step(
        self,
        t: float,
        forcing: Callable[..., jnp.ndarray] | jnp.ndarray,
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        bc: Callable[..., jnp.ndarray] | jnp.ndarray,
    ) -> jnp.ndarray:
        if self.cnm2.size == 0:
            raise ValueError("PUDiffusionSolver.bdf1_step requires set_initial_state first")
        previous = self.current_physical_state()
        rhs_physical = previous + self.dt * evaluate_forcing_callback(forcing, self.nu, t, self.x)
        return self._take_step(rhs_physical, t, neu_coeff_func, dir_coeff_func, bc, -self.nu * self.dt)

    def bdf2_step(
        self,
        t: float,
        forcing: Callable[..., jnp.ndarray] | jnp.ndarray,
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        bc: Callable[..., jnp.ndarray] | jnp.ndarray,
    ) -> jnp.ndarray:
        if self.completed_steps_ < 1:
            raise ValueError("PUDiffusionSolver.bdf2_step requires one prior step in the state history")
        rhs_physical = (4.0 / 3.0) * self.cnm1 - (1.0 / 3.0) * self.cnm2 + (2.0 / 3.0) * self.dt * evaluate_forcing_callback(forcing, self.nu, t, self.x)
        return self._take_step(rhs_physical, t, neu_coeff_func, dir_coeff_func, bc, -(2.0 / 3.0) * self.nu * self.dt)

    def bdf3_step(
        self,
        t: float,
        forcing: Callable[..., jnp.ndarray] | jnp.ndarray,
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        bc: Callable[..., jnp.ndarray] | jnp.ndarray,
    ) -> jnp.ndarray:
        if self.completed_steps_ < 2:
            raise ValueError("PUDiffusionSolver.bdf3_step requires two prior steps in the state history")
        rhs_physical = (18.0 / 11.0) * self.cn - (9.0 / 11.0) * self.cnm1 + (2.0 / 11.0) * self.cnm2 + (6.0 / 11.0) * self.dt * evaluate_forcing_callback(forcing, self.nu, t, self.x)
        return self._take_step(rhs_physical, t, neu_coeff_func, dir_coeff_func, bc, -(6.0 / 11.0) * self.nu * self.dt)

    def _take_step(
        self,
        rhs_physical: jnp.ndarray,
        t: float,
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        bc: Callable[..., jnp.ndarray] | jnp.ndarray,
        lap_scale: float,
    ) -> jnp.ndarray:
        neu_coeff, dir_coeff = self._get_boundary_coefficients(t, neu_coeff_func, dir_coeff_func)
        self._ensure_boundary_operator(neu_coeff, dir_coeff)
        rhs_boundary = evaluate_transient_boundary_values(bc, neu_coeff, dir_coeff, self.nr, t, self.xb)
        system = build_implicit_system(self.lap, self.bc, self.n, lap_scale)
        sparse_system = None if self.lap_sparse is None or self.bc_sparse is None else build_implicit_system_sparse(self.lap_sparse, self.bc_sparse, self.n, lap_scale)
        rhs = build_implicit_rhs(rhs_physical, rhs_boundary)
        if self.linear_solver == "dense":
            sol = solve_dense_system(system, rhs)
            self.fixed_preconditioner = None
        elif self.linear_solver == "gmres_ilu":
            if sparse_system is None:
                raise ValueError("sparse solve requires sparse PU operator assembly")
            if self.fixed_preconditioner is None:
                self.fixed_preconditioner = build_fixed_ilu_preconditioner(sparse_system)
            sol = solve_sparse_system_gmres(sparse_system, rhs, preconditioner=self.fixed_preconditioner)
        else:
            raise ValueError(f"unknown linear solver {self.linear_solver}")
        next_state = jnp.asarray(sol[: self.n], dtype=float)
        self._push_completed_step(next_state)
        return self.current_physical_state()

    def _get_boundary_coefficients(
        self,
        t: float,
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if is_fixed_boundary_callback(neu_coeff_func) and is_fixed_boundary_callback(dir_coeff_func):
            if not self.fixed_bc_coefficients_ready_:
                self.cached_neu_coeff_ = _ensure_unbatched_operator_input(
                    evaluate_boundary_coefficient(neu_coeff_func, self.xb),
                    "boundary coefficient",
                )
                self.cached_dir_coeff_ = _ensure_unbatched_operator_input(
                    evaluate_boundary_coefficient(dir_coeff_func, self.xb),
                    "boundary coefficient",
                )
                self.fixed_bc_coefficients_ready_ = True
            return self.cached_neu_coeff_, self.cached_dir_coeff_
        return (
            _ensure_unbatched_operator_input(evaluate_boundary_coefficient(neu_coeff_func, self.xb, t), "boundary coefficient"),
            _ensure_unbatched_operator_input(evaluate_boundary_coefficient(dir_coeff_func, self.xb, t), "boundary coefficient"),
        )

    def _ensure_boundary_operator(self, neu_coeff: jnp.ndarray, dir_coeff: jnp.ndarray) -> None:
        if self.fixed_bc_operator_ready_ and ((self.linear_solver == "dense" and self.bc.shape[0] > 0) or (self.linear_solver != "dense" and self.bc_sparse is not None)):
            return
        if self.patch_data is None:
            raise ValueError("solver must be initialized before building PU boundary operators")
        if self.linear_solver == "dense":
            self.bc = pu_localized_operator(
                self.domain,
                self.patch_data,
                self.xi,
                self.xb,
                "bc",
                normals=self.nr,
                neu_coeff=neu_coeff,
                dir_coeff=dir_coeff,
                query_grouping=self.boundary_query_grouping,
            )
            self.bc_sparse = None
        else:
            self.bc = jnp.zeros((0, 0), dtype=float)
            self.bc_sparse = pu_localized_operator_sparse(
                self.domain,
                self.patch_data,
                self.xi,
                self.xb,
                "bc",
                normals=self.nr,
                neu_coeff=neu_coeff,
                dir_coeff=dir_coeff,
                query_grouping=self.boundary_query_grouping,
            )
        if self.fixed_bc_coefficients_ready_:
            self.fixed_bc_operator_ready_ = True
        self.fixed_preconditioner = None

    def _push_completed_step(self, next_state: jnp.ndarray) -> None:
        self.cnm2, self.cnm1, self.cn, self.completed_steps_ = push_completed_step(
            self.cnm2 if self.cnm2.size else jnp.zeros(self.n, dtype=float),
            self.cnm1 if self.cnm1.size else jnp.zeros(self.n, dtype=float),
            self.cn if self.cn.size else jnp.zeros(self.n, dtype=float),
            self.completed_steps_,
            next_state,
        )
