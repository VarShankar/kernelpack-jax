from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import Callable

from jax import core as jax_core, lax
import jax.numpy as jnp
import lineax as lx
from jax import vmap

from kernelpack.domain import DomainDescriptor
from kernelpack.rbffd import OpProperties, StencilProperties
from ._cudss import prepare_transient_cudss, close_transient_cudss, solve_transient_cudss, get_transient_cudss_factor
from ._common import (
    assemble_operator,
    assemble_operator_sparse,
    build_domain_state,
    build_fixed_discretization,
    build_fixed_ilu_preconditioner,
    build_implicit_rhs,
    build_implicit_system,
    build_implicit_system_sparse,
    build_stencil_properties,
    evaluate_boundary_coefficient,
    evaluate_forcing_callback,
    evaluate_transient_boundary_values,
    FixedDiscretization,
    FrozenILUPreconditioner,
    is_fixed_boundary_callback,
    push_completed_step,
    SparseCOOMatrix,
    solve_dense_system,
    solve_sparse_system_gmres,
    sparse_matrix_to_dense,
    _ensure_unbatched_operator_input,
    validate_physical_state,
)


def _evaluate_reaction_callback(
    reaction: Callable[..., jnp.ndarray] | jnp.ndarray,
    t: float,
    state: jnp.ndarray,
    x: jnp.ndarray,
) -> jnp.ndarray:
    """Normalize the reaction callback API to a single nodal field.

    The solver accepts a few convenient calling conventions so the inverse
    examples can pass either `reaction(t, state, X)` or simpler closures such as
    `reaction(state)`.
    """

    if not callable(reaction):
        values = reaction
    else:
        errors: list[Exception] = []
        for call in (
            lambda: reaction(t, state, x),
            lambda: reaction(state, x),
            lambda: reaction(t, state),
            lambda: reaction(state),
        ):
            try:
                values = call()
                break
            except TypeError as exc:
                errors.append(exc)
        else:
            raise TypeError(
                "Reaction callback must be callable as reaction(t, state, X), reaction(state, X), reaction(t, state), or reaction(state)."
            ) from errors[-1]
    arr = jnp.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = jnp.full((state.shape[0],), arr, dtype=float)
    arr = arr.reshape(-1)
    if int(arr.shape[0]) != int(state.shape[0]):
        raise ValueError("reaction callback returned the wrong number of physical-node values")
    return arr


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


@dataclass
class ReactionDiffusionSolver:
    """Semilinear reaction-diffusion solver on a fixed ghost-node discretization.

    This is intentionally a light fork of `DiffusionSolver`:

    - diffusion stays implicit,
    - reaction is extrapolated explicitly in time,
    - boundary handling and sparse/dense solve paths stay the same.

    That makes it a good fit for inverse problems where we want to learn a
    reaction law through repeated rollouts without dragging in advection or the
    PU transport machinery.
    """

    lap_assembler: str = "fd"
    bc_assembler: str = "fd"
    lap_stencil: str = "rbf"
    bc_stencil: str = "rbf"
    domain: DomainDescriptor = field(default_factory=DomainDescriptor)
    xi: int = 0
    dt: float = jnp.nan
    nu: float = jnp.nan
    num_omp_threads: int = 1
    linear_solver: str = "gmres_ilu"
    gmres_tol: float = 1.0e-8
    gmres_atol: float = 1.0e-10
    gmres_restart: int = 40
    gmres_maxiter: int | None = None
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
    cnm2: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    cnm1: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    cn: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    completed_steps_: int = 0
    fixed_bc_operator_ready_: bool = False
    fixed_bc_coefficients_ready_: bool = False
    cached_neu_coeff_: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    cached_dir_coeff_: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    fixed_preconditioner: FrozenILUPreconditioner | None = None
    assemble_dense_operators: bool = True
    sparse_assembly_chunk_size: int | None = None

    def prepare_cudss(self):
        prepare_transient_cudss(self)

    def close_cudss(self):
        close_transient_cudss(self)

    def spawn_rollout_copy(self) -> ReactionDiffusionSolver:
        """Reuse assembled operators while resetting only the time history."""

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

    def init(
        self,
        domain: DomainDescriptor,
        xi: int,
        dlt: float,
        d_coeff: float,
        num_omp_threads: int = 1,
        *,
        sparse_assembly_chunk_size: int | None = None,
        assemble_dense_operators: bool | None = None,
    ) -> None:
        """Assemble the fixed spatial discretization once."""

        self.domain = domain
        self.xi = xi
        self.dt = dlt
        self.nu = d_coeff
        self.num_omp_threads = num_omp_threads
        self.sparse_assembly_chunk_size = sparse_assembly_chunk_size
        self.assemble_dense_operators = (self.linear_solver != "cudss") if assemble_dense_operators is None else bool(assemble_dense_operators)
        state = build_domain_state(self.domain)
        self.x = state.x
        self.xb = state.xb
        self.nr = state.nr
        self.n = state.n
        self.nf = state.nf
        self.lap_stencil_properties = build_stencil_properties(self.domain, self.xi, 2, "interior_boundary")
        self.bc_stencil_properties = build_stencil_properties(self.domain, self.xi, 1, "boundary")
        # Sparse chunked setup must not first materialize full KNN graphs.
        # The sparse assembler constructs graph rows incrementally in this mode.
        use_incremental_sparse_graphs = (
            self.sparse_assembly_chunk_size is not None
            and self.sparse_assembly_chunk_size > 0
            and not self.assemble_dense_operators
        )
        self.fixed_discretization = (
            None
            if use_incremental_sparse_graphs
            else build_fixed_discretization(self.domain, self.lap_stencil_properties, self.bc_stencil_properties)
        )
        if self.assemble_dense_operators:
            self.lap = assemble_operator(
                self.domain,
                self.lap_assembler,
                self.lap_stencil,
                "lap",
                self.lap_stencil_properties,
                self.lap_op_properties,
                stencil_graph=self.fixed_discretization.laplacian_graph,
            )
        else:
            self.lap = jnp.zeros((0, 0), dtype=float)
        self.lap_sparse = assemble_operator_sparse(
            self.domain,
            self.lap_assembler,
            self.lap_stencil,
            "lap",
            self.lap_stencil_properties,
            self.lap_op_properties,
            stencil_graph=None if self.fixed_discretization is None else self.fixed_discretization.laplacian_graph,
            chunk_size=self.sparse_assembly_chunk_size,
        )
        self.bc = jnp.zeros((0, self.nf))
        self.bc_sparse = None
        self.cnm2 = jnp.zeros(self.n)
        self.cnm1 = jnp.zeros(self.n)
        self.cn = jnp.zeros(self.n)
        self.completed_steps_ = 0
        self.fixed_bc_operator_ready_ = False
        self.fixed_bc_coefficients_ready_ = False
        self.cached_neu_coeff_ = jnp.zeros(0)
        self.cached_dir_coeff_ = jnp.zeros(0)
        self.fixed_preconditioner = None

    def set_step_size(self, dlt: float) -> None:
        """Changing the timestep changes the implicit matrix scaling."""

        self.dt = dlt
        self.fixed_bc_operator_ready_ = False
        self.fixed_bc_coefficients_ready_ = False
        self.fixed_preconditioner = None

    def set_initial_state(self, c0: jnp.ndarray) -> None:
        """Start a fresh rollout from one physical state."""

        self.cnm2 = validate_physical_state(c0, self.n)
        self.cnm1 = jnp.zeros_like(self.cnm2)
        self.cn = jnp.zeros_like(self.cnm2)
        self.completed_steps_ = 0

    def set_state_history(self, *states: jnp.ndarray) -> None:
        """Seed BDF1/BDF2/BDF3 with one, two, or three physical states."""

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
        """Return the newest physical state currently stored in the solver."""

        return lax.switch(
            jnp.minimum(jnp.asarray(self.completed_steps_, dtype=int), 2),
            [
                lambda _: self.cnm2,
                lambda _: self.cnm1,
                lambda _: self.cn,
            ],
            None,
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
        """Freeze an ILU at a reference diffusion coefficient and BDF order.

        This is the same compromise we use in the inverse examples: the traced
        parameter may change the matrix, but the preconditioner itself is built
        once outside the differentiated path and then reused.
        """

        neu_coeff, dir_coeff = self._get_boundary_coefficients(t, neu_coeff_func, dir_coeff_func)
        self._ensure_boundary_operator(neu_coeff, dir_coeff)
        if self.lap_sparse is None or self.bc_sparse is None:
            raise ValueError("prepare_preconditioner requires sparse operator assembly")
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
        self.fixed_preconditioner = build_fixed_ilu_preconditioner(
            sparse_system,
            application=self._ilu_application(sparse_system),
        )
        return self.fixed_preconditioner

    def prepare_boundary_operator(
        self,
        t: float,
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
    ) -> None:
        """Assemble and cache the fixed boundary operator outside a traced rollout."""

        neu_coeff, dir_coeff = self._get_boundary_coefficients(t, neu_coeff_func, dir_coeff_func)
        self._ensure_boundary_operator(neu_coeff, dir_coeff)
    def bdf1_step(
        self,
        t: float,
        reaction: Callable[..., jnp.ndarray] | jnp.ndarray,
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        bc: Callable[..., jnp.ndarray] | jnp.ndarray,
        forcing: Callable[..., jnp.ndarray] | jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        if self.cnm2.size == 0:
            raise ValueError("bdf1_step requires set_initial_state first")
        # BDF1 uses the current state directly in the explicit reaction term.
        previous = self.current_physical_state()
        forcing_term = self._evaluate_forcing_if_present(forcing, t)
        reaction_term = _evaluate_reaction_callback(reaction, t, previous, self.x)
        rhs_physical = previous + self.dt * (forcing_term + reaction_term)
        return self._take_step(rhs_physical, t, neu_coeff_func, dir_coeff_func, bc, -self.nu * self.dt)

    def bdf2_step(
        self,
        t: float,
        reaction: Callable[..., jnp.ndarray] | jnp.ndarray,
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        bc: Callable[..., jnp.ndarray] | jnp.ndarray,
        forcing: Callable[..., jnp.ndarray] | jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        if not isinstance(self.completed_steps_, jax_core.Tracer) and self.completed_steps_ < 1:
            raise ValueError("bdf2_step requires one prior step in the state history")
        # AB2-style extrapolation of the semilinear reaction term.
        reaction_nm2 = _evaluate_reaction_callback(reaction, t, self.cnm2, self.x)
        reaction_nm1 = _evaluate_reaction_callback(reaction, t, self.cnm1, self.x)
        reaction_extrapolated = 2.0 * reaction_nm1 - reaction_nm2
        rhs_physical = (4.0 / 3.0) * self.cnm1 - (1.0 / 3.0) * self.cnm2 + (2.0 / 3.0) * self.dt * (
            self._evaluate_forcing_if_present(forcing, t) + reaction_extrapolated
        )
        return self._take_step(rhs_physical, t, neu_coeff_func, dir_coeff_func, bc, -(2.0 / 3.0) * self.nu * self.dt, bdf_order=2)

    def bdf3_step(
        self,
        t: float,
        reaction: Callable[..., jnp.ndarray] | jnp.ndarray,
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        bc: Callable[..., jnp.ndarray] | jnp.ndarray,
        forcing: Callable[..., jnp.ndarray] | jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        if not isinstance(self.completed_steps_, jax_core.Tracer) and self.completed_steps_ < 2:
            raise ValueError("bdf3_step requires two prior steps in the state history")
        # AB3-style extrapolation of the semilinear reaction term.
        reaction_nm2 = _evaluate_reaction_callback(reaction, t, self.cnm2, self.x)
        reaction_nm1 = _evaluate_reaction_callback(reaction, t, self.cnm1, self.x)
        reaction_n = _evaluate_reaction_callback(reaction, t, self.cn, self.x)
        reaction_extrapolated = 3.0 * reaction_n - 3.0 * reaction_nm1 + reaction_nm2
        rhs_physical = (18.0 / 11.0) * self.cn - (9.0 / 11.0) * self.cnm1 + (2.0 / 11.0) * self.cnm2 + (6.0 / 11.0) * self.dt * (
            self._evaluate_forcing_if_present(forcing, t) + reaction_extrapolated
        )
        return self._take_step(rhs_physical, t, neu_coeff_func, dir_coeff_func, bc, -(6.0 / 11.0) * self.nu * self.dt, bdf_order=3)

    def _evaluate_forcing_if_present(
        self,
        forcing: Callable[..., jnp.ndarray] | jnp.ndarray | None,
        t: float,
    ) -> jnp.ndarray:
        """Support optional external forcing in addition to the reaction term."""

        if forcing is None:
            return jnp.zeros((self.n,), dtype=float)
        return evaluate_forcing_callback(forcing, self.nu, t, self.x)

    def _take_step(
        self,
        rhs_physical: jnp.ndarray,
        t: float,
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        bc: Callable[..., jnp.ndarray] | jnp.ndarray,
        lap_scale: float,
        *,
        bdf_order: int = 1,
    ) -> jnp.ndarray:
        """Solve the implicit diffusion block with explicit reaction/forcing RHS."""

        neu_coeff, dir_coeff = self._get_boundary_coefficients(t, neu_coeff_func, dir_coeff_func)
        self._ensure_boundary_operator(neu_coeff, dir_coeff)
        rhs_boundary = evaluate_transient_boundary_values(bc, neu_coeff, dir_coeff, self.nr, t, self.xb)
        prepared = get_transient_cudss_factor(self, bdf_order) if self.linear_solver == "cudss" else None
        sparse_system = None if prepared is not None or self.lap_sparse is None or self.bc_sparse is None else build_implicit_system_sparse(self.lap_sparse, self.bc_sparse, self.n, lap_scale)
        rhs = build_implicit_rhs(rhs_physical, rhs_boundary)
        if self.linear_solver == "cudss":
            if prepared is not None:
                sol = prepared.solve(rhs)
            elif sparse_system is None:
                raise ValueError("cuDSS requires sparse operator assembly")
            else:
                sol = solve_transient_cudss(self, sparse_system, rhs, bdf_order)
        elif self.linear_solver == "dense":
            system = build_implicit_system(self.lap, self.bc, self.n, lap_scale)
            sol = solve_dense_system(system, rhs)
            self.fixed_preconditioner = None
        elif self.linear_solver == "lineax":
            system = build_implicit_system(self.lap, self.bc, self.n, lap_scale) if self.lap.size and self.bc.size else jnp.zeros((0, 0), dtype=float)
            dense_system = system if system.size else sparse_matrix_to_dense(sparse_system)
            sol = _solve_lineax_system(dense_system, rhs)
            self.fixed_preconditioner = None
        elif self.linear_solver in ("gmres", "gmres_ilu"):
            if sparse_system is None:
                raise ValueError("sparse solve requires sparse operator assembly")
            # If the caller already froze a reference preconditioner, reuse it.
            if self.linear_solver == "gmres_ilu" and self.fixed_preconditioner is None:
                self.fixed_preconditioner = build_fixed_ilu_preconditioner(
                    sparse_system,
                    application=self._ilu_application(sparse_system),
                )
            preconditioner = self.fixed_preconditioner if self.linear_solver == "gmres_ilu" else None
            sol = solve_sparse_system_gmres(
                sparse_system,
                rhs,
                preconditioner=preconditioner,
                tol=self.gmres_tol,
                atol=self.gmres_atol,
                restart=self.gmres_restart,
                maxiter=self.gmres_maxiter,
            )
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
        """Cache fixed boundary coefficients when the callback is time-independent."""

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

    @staticmethod
    def _ilu_application(sparse_system: SparseCOOMatrix) -> str:
        """Avoid a dense approximate inverse for large sparse systems."""

        if max(sparse_system.shape) >= 8192:
            print(
                f"[ilu] using host_callback application for sparse system shape={sparse_system.shape}",
                flush=True,
            )
            return "host_callback"
        return "dense"

    def _ensure_boundary_operator(self, neu_coeff: jnp.ndarray, dir_coeff: jnp.ndarray) -> None:
        """Assemble the boundary operator once when coefficients are fixed."""

        dense_ready = self.bc.shape[0] > 0
        sparse_ready = self.bc_sparse is not None and self.bc_sparse.shape[0] > 0
        if self.fixed_bc_operator_ready_ and (dense_ready or sparse_ready):
            return
        if self.assemble_dense_operators:
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
        else:
            self.bc = jnp.zeros((0, self.nf), dtype=float)
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
            chunk_size=self.sparse_assembly_chunk_size,
        )
        if self.fixed_bc_coefficients_ready_:
            self.fixed_bc_operator_ready_ = True
        self.fixed_preconditioner = None

    def _push_completed_step(self, next_state: jnp.ndarray) -> None:
        """Advance the BDF history registers."""

        self.cnm2, self.cnm1, self.cn, self.completed_steps_ = push_completed_step(
            self.cnm2,
            self.cnm1,
            self.cn,
            self.completed_steps_,
            next_state,
        )

