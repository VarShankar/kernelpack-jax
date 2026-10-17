from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Callable

from jax import jit, vmap
import jax.numpy as jnp

from kernelpack.domain import DomainDescriptor, DualNodeDomainDescriptor
from kernelpack.rbffd import CrossNodeDiffOp, OpProperties, RBFStencil, StencilProperties

from ._common import FrozenILUPreconditioner, SparseCOOMatrix, build_fixed_ilu_preconditioner, dense_matrix_to_sparse, solve_sparse_system_gmres
from .pu_sl_advection import PUSLAdvectionSolver


def _strip_ghosts(velocity_domain: DomainDescriptor) -> DomainDescriptor:
    dim = velocity_domain.get_dim()
    domain = DomainDescriptor()
    domain.set_nodes(velocity_domain.get_interior_nodes(), velocity_domain.get_bdry_nodes(), jnp.zeros((0, dim)))
    domain.set_normals(velocity_domain.get_nrmls())
    domain.set_sep_rad(velocity_domain.get_sep_rad())
    domain.set_outer_level_set(velocity_domain.get_outer_level_set())
    domain.set_boundary_level_sets(velocity_domain.get_boundary_level_sets())
    domain.build_structs()
    return domain


def _with_point_set_tree(sp: StencilProperties, point_set: str, tree_mode: str) -> StencilProperties:
    return StencilProperties(
        n=sp.n,
        dim=sp.dim,
        ell=sp.ell,
        spline_degree=sp.spline_degree,
        npoly=sp.npoly,
        width=sp.width,
        tree_mode=tree_mode,
        point_set=point_set,
    )


@partial(jit, static_argnames=("n_interior", "n_phys"))
def _boundary_normal_matrix(nr: jnp.ndarray, n_interior: int, n_phys: int) -> jnp.ndarray:
    nub, dim = nr.shape
    mat = jnp.zeros((nub, dim * n_phys), dtype=nr.dtype)
    bidx = jnp.arange(nub)
    for d in range(dim):
        mat = mat.at[bidx, d * n_phys + n_interior + bidx].set(nr[:, d])
    return mat


@partial(jit, static_argnames=("n_interior", "n_phys"))
def _boundary_lambda_matrix(nr: jnp.ndarray, n_interior: int, n_phys: int) -> jnp.ndarray:
    nub, dim = nr.shape
    mat = jnp.zeros((dim * n_phys, nub), dtype=nr.dtype)
    bidx = jnp.arange(nub)
    for d in range(dim):
        mat = mat.at[d * n_phys + n_interior + bidx, bidx].set(nr[:, d])
    return mat


@jit
def _block_divergence_matrix(div_ops: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([div_ops[d] for d in range(div_ops.shape[0])], axis=1)


@jit
def _block_gradient_matrix(grad_ops: jnp.ndarray, scale: float) -> jnp.ndarray:
    return jnp.concatenate([scale * grad_ops[d] for d in range(grad_ops.shape[0])], axis=0)


@partial(jit, static_argnames=("completed_steps",))
def _history_rhs(unm2: jnp.ndarray, unm1: jnp.ndarray, un: jnp.ndarray, completed_steps: int, coeffs: jnp.ndarray, scale: float) -> jnp.ndarray:
    h0 = jnp.where(completed_steps <= 0, unm2, jnp.where(completed_steps == 1, unm1, un))
    h1 = jnp.where(completed_steps <= 0, jnp.zeros_like(unm2), jnp.where(completed_steps == 1, unm2, unm1))
    h2 = jnp.where(completed_steps <= 1, jnp.zeros_like(unm2), unm2)
    return scale * (coeffs[0] * h0 + coeffs[1] * h1 + coeffs[2] * h2)


@jit
def _divergence_diagnostics(div_ops: jnp.ndarray, velocity: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    div = jnp.sum(jnp.stack([div_ops[d] @ velocity[:, d] for d in range(div_ops.shape[0])], axis=0), axis=0)
    return jnp.linalg.norm(div) / jnp.sqrt(jnp.maximum(div.size, 1)), jnp.max(jnp.abs(div), initial=0.0)


@partial(jit, static_argnames=("n_interior",))
def _wall_diagnostics(nr: jnp.ndarray, velocity: jnp.ndarray, n_interior: int, boundary_normal_velocity: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    nub = nr.shape[0]
    vals = jnp.sum(nr * velocity[n_interior : n_interior + nub, :], axis=1)
    resid = vals - boundary_normal_velocity.reshape(-1)
    return jnp.linalg.norm(resid) / jnp.sqrt(jnp.maximum(nub, 1)), jnp.max(jnp.abs(resid), initial=0.0)


@partial(jit, static_argnames=("dim", "nphys"))
def _pack_velocity(velocity: jnp.ndarray, *, dim: int, nphys: int) -> jnp.ndarray:
    return velocity.reshape(nphys, dim).T.reshape(-1)


@partial(jit, static_argnames=("dim", "nphys"))
def _unpack_velocity(flat: jnp.ndarray, *, dim: int, nphys: int) -> jnp.ndarray:
    return flat[: dim * nphys].reshape(dim, nphys).T


@partial(jit, static_argnames=("npib", "add_gauge"))
def _pack_projection_rhs(velocity_rhs: jnp.ndarray, boundary_normal_velocity: jnp.ndarray, *, npib: int, add_gauge: bool) -> jnp.ndarray:
    rhs = jnp.concatenate([velocity_rhs, jnp.zeros(npib, dtype=velocity_rhs.dtype), boundary_normal_velocity.reshape(-1)])
    if add_gauge:
        rhs = jnp.concatenate([rhs, jnp.zeros(1, dtype=velocity_rhs.dtype)])
    return rhs


@dataclass
class IncompressibleEulerBDFBackend:
    domain: DualNodeDomainDescriptor = field(default_factory=DualNodeDomainDescriptor)
    velocity_stencil_properties: StencilProperties = field(default_factory=StencilProperties)
    pressure_stencil_properties: StencilProperties = field(default_factory=StencilProperties)
    dt: float = 0.0
    velocity_physical_domain: DomainDescriptor = field(default_factory=DomainDescriptor)
    xphys: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    xb: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    nr: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    xp: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    grad_ops: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0, 0)))
    div_ops: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0, 0)))
    velocity_node_range_: tuple[int, int] = (1, 0)
    pressure_node_range_: tuple[int, int] = (1, 0)
    unm2: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    unm1: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    un: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    completed_steps_: int = 0
    fixed_preconditioners: dict[tuple[float, bool], FrozenILUPreconditioner] = field(default_factory=dict)
    sparse_systems: dict[tuple[float, bool], SparseCOOMatrix] = field(default_factory=dict)

    def init(
        self,
        dual_domain: DualNodeDomainDescriptor,
        velocity_sp: StencilProperties,
        pressure_sp: StencilProperties,
        dt: float,
        num_omp_threads: int = 1,
    ) -> None:
        del num_omp_threads
        if not (dt > 0):
            raise ValueError("IncompressibleEulerBDFBackend requires dt > 0")
        self.domain = dual_domain
        self.domain.build_structs()
        self.velocity_stencil_properties = velocity_sp
        self.pressure_stencil_properties = pressure_sp
        self.dt = float(dt)
        self.velocity_physical_domain = _strip_ghosts(self.domain.get_velocity_domain())
        self.xphys = self.velocity_physical_domain.get_int_bdry_nodes()
        self.xb = self.velocity_physical_domain.get_bdry_nodes()
        self.nr = self.velocity_physical_domain.get_nrmls()
        self.xp = self.domain.get_pressure_domain().get_int_bdry_nodes()
        self.velocity_node_range_ = (1, int(self.xphys.shape[0]))
        self.pressure_node_range_ = (1, int(self.xp.shape[0]))
        self._assemble_operators()
        self.unm2 = jnp.zeros((self.xphys.shape[0], self.xphys.shape[1]))
        self.unm1 = jnp.zeros_like(self.unm2)
        self.un = jnp.zeros_like(self.unm2)
        self.completed_steps_ = 0
        self.fixed_preconditioners.clear()
        self.sparse_systems.clear()

    def set_step_size(self, dt: float) -> None:
        if not (dt > 0):
            raise ValueError("IncompressibleEulerBDFBackend requires dt > 0")
        self.dt = float(dt)
        self.fixed_preconditioners.clear()
        self.sparse_systems.clear()

    def set_initial_velocity(self, u0: jnp.ndarray) -> None:
        velocity = _validate_velocity_state(self, u0, "set_initial_velocity")
        self.unm2 = velocity
        self.unm1 = jnp.zeros_like(velocity)
        self.un = jnp.zeros_like(velocity)
        self.completed_steps_ = 0

    def set_initial_velocity_owned(self, u0_local: jnp.ndarray) -> None:
        self.set_initial_velocity(u0_local)

    def set_velocity_history(self, *states: jnp.ndarray) -> None:
        self._set_velocity_history_impl(*states)

    def set_velocity_history_owned(self, *states: jnp.ndarray) -> None:
        self._set_velocity_history_impl(*states)

    def get_owned_pressure_range(self) -> tuple[int, int]:
        return self.pressure_node_range_

    def gather_pressure_samples(self, pressure: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(pressure, dtype=float)

    def bdf1_step(self, forcing: Callable[[jnp.ndarray], jnp.ndarray], problem: dict[str, object] | None = None) -> dict[str, object]:
        return self._solve_bdf_step(1.0, jnp.array([1.0, 0.0, 0.0]), forcing, problem)

    def bdf2_step(self, forcing: Callable[[jnp.ndarray], jnp.ndarray], problem: dict[str, object] | None = None) -> dict[str, object]:
        if self.completed_steps_ < 1:
            raise ValueError("IncompressibleEulerBDFBackend.bdf2_step requires one prior step")
        return self._solve_bdf_step(1.5, jnp.array([2.0, -0.5, 0.0]), forcing, problem)

    def bdf3_step(self, forcing: Callable[[jnp.ndarray], jnp.ndarray], problem: dict[str, object] | None = None) -> dict[str, object]:
        if self.completed_steps_ < 2:
            raise ValueError("IncompressibleEulerBDFBackend.bdf3_step requires two prior steps")
        return self._solve_bdf_step(11.0 / 6.0, jnp.array([3.0, -1.5, 1.0 / 3.0]), forcing, problem)

    @staticmethod
    def stationary_slip_wall(boundary_indices: jnp.ndarray) -> dict[str, object]:
        return {
            "boundary_indices": jnp.asarray(boundary_indices, dtype=int).reshape(-1),
            "normal_velocity": lambda x, nr: jnp.zeros(x.shape[0], dtype=nr.dtype),
        }

    @staticmethod
    def default_problem_definition() -> dict[str, object]:
        return {
            "slip_walls": [],
            "gauge_options": {
                "mode": "automatic",
                "nullspace_tol": 1.0e-7,
                "row_scale_detection": True,
            },
        }

    def _set_velocity_history_impl(self, *states: jnp.ndarray) -> None:
        if len(states) == 0:
            raise ValueError("set_velocity_history requires at least one state")
        checked = [_validate_velocity_state(self, state, "set_velocity_history") for state in states]
        self.unm2 = checked[0]
        self.unm1 = checked[1] if len(checked) >= 2 else jnp.zeros_like(self.unm2)
        self.un = checked[2] if len(checked) >= 3 else jnp.zeros_like(self.unm2)
        self.completed_steps_ = min(len(checked) - 1, 2)

    def _assemble_operators(self) -> None:
        dim = self.domain.get_dim()
        grad_ops = []
        div_ops = []
        for d in range(dim):
            op_props = OpProperties(decompose=False, store_weights=True, record_stencils=False, selectdim=d)
            grad_op = CrossNodeDiffOp(lambda: RBFStencil())
            grad_op.assemble_op(
                self.domain.get_pressure_domain(),
                self.velocity_physical_domain,
                "grad",
                _with_point_set_tree(self.pressure_stencil_properties, "interior_boundary", "interior_boundary"),
                op_props,
            )
            grad_ops.append(grad_op.get_op())

            div_op = CrossNodeDiffOp(lambda: RBFStencil())
            div_op.assemble_op(
                self.velocity_physical_domain,
                self.domain.get_pressure_domain(),
                "grad",
                _with_point_set_tree(self.velocity_stencil_properties, "interior_boundary", "interior_boundary"),
                op_props,
            )
            div_ops.append(div_op.get_op())
        self.grad_ops = jnp.stack(grad_ops, axis=0)
        self.div_ops = jnp.stack(div_ops, axis=0)

    def _solve_bdf_step(self, alpha0: float, history_coeffs: jnp.ndarray, forcing: Callable[[jnp.ndarray], jnp.ndarray], problem: dict[str, object] | None) -> dict[str, object]:
        problem = _normalize_problem(problem)
        _validate_problem(self, problem)

        dim = self.domain.get_dim()
        nui = self.velocity_physical_domain.get_num_interior_nodes()
        nub = self.velocity_physical_domain.get_num_bdry_nodes()
        nphys = self.velocity_physical_domain.get_num_int_bdry_nodes()
        npib = self.domain.get_pressure_domain().get_num_int_bdry_nodes()

        history_rhs = _history_rhs(self.unm2, self.unm1, self.un, self.completed_steps_, history_coeffs, 1.0 / alpha0)
        forcing_u = jnp.asarray(forcing(self.xphys), dtype=float)
        if forcing_u.shape != self.xphys.shape:
            raise ValueError("IncompressibleEulerBDFBackend forcing callback returned invalid data")
        forcing_u = (self.dt / alpha0) * forcing_u
        boundary_normal_velocity = _prepare_wall_data(self, problem)

        add_gauge = _pressure_constant_is_null(self, problem["gauge_options"])
        system, preconditioner = self._get_cached_system(alpha0, add_gauge)
        velocity_rhs = _pack_velocity(forcing_u + history_rhs, dim=dim, nphys=nphys)
        rhs = _pack_projection_rhs(velocity_rhs, boundary_normal_velocity, npib=npib, add_gauge=add_gauge)
        guess = self._initial_guess(nphys, npib, nub, add_gauge)
        x = solve_sparse_system_gmres(system, rhs, guess=guess, preconditioner=preconditioner)
        velocity = _unpack_velocity(x, dim=dim, nphys=nphys)
        pressure = x[dim * nphys : dim * nphys + npib]

        if self.completed_steps_ <= 0:
            self.unm1 = velocity
            self.completed_steps_ = 1
        elif self.completed_steps_ == 1:
            self.un = velocity
            self.completed_steps_ = 2
        else:
            self.unm2 = self.unm1
            self.unm1 = self.un
            self.un = velocity
            self.completed_steps_ = max(self.completed_steps_, 3)

        div_rms, div_max = _divergence_diagnostics(self.div_ops, velocity)
        wall_rms, wall_max = _wall_diagnostics(self.nr, velocity, nui, boundary_normal_velocity)
        return {
            "velocity": velocity,
            "pressure": pressure,
            "pressure_is_local": False,
            "pressure_begin": 1,
            "pressure_end": npib,
            "divergence_rms": div_rms,
            "divergence_max": div_max,
            "wall_normal_rms": wall_rms,
            "wall_normal_max": wall_max,
        }

    def _get_cached_system(self, alpha0: float, add_gauge: bool) -> tuple[SparseCOOMatrix, FrozenILUPreconditioner]:
        key = (float(alpha0), bool(add_gauge))
        if key not in self.sparse_systems:
            dense = self._assemble_system_dense(alpha0, add_gauge)
            sparse = dense_matrix_to_sparse(dense)
            self.sparse_systems[key] = sparse
            self.fixed_preconditioners[key] = build_fixed_ilu_preconditioner(sparse)
        return self.sparse_systems[key], self.fixed_preconditioners[key]

    def make_jitted_projection_solver(self, alpha0: float, *, add_gauge: bool = True):
        """Return compiled single-RHS and batched projection solvers.

        This is the differentiable hot path for inverse/learning loops: setup
        happens once, then RHS changes stay on JAX arrays and can be jitted or
        vmapped without rebuilding stencils, matrices, or ILU factors.
        """

        system, preconditioner = self._get_cached_system(alpha0, add_gauge)
        dim = self.domain.get_dim()
        nphys = self.velocity_physical_domain.get_num_int_bdry_nodes()
        npib = self.domain.get_pressure_domain().get_num_int_bdry_nodes()
        nub = self.velocity_physical_domain.get_num_bdry_nodes()
        tail_size = npib + nub + int(add_gauge)

        @jit
        def solve_one(
            velocity_rhs: jnp.ndarray,
            boundary_normal_velocity: jnp.ndarray,
            velocity_guess: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            rhs = _pack_projection_rhs(
                _pack_velocity(velocity_rhs, dim=dim, nphys=nphys),
                boundary_normal_velocity,
                npib=npib,
                add_gauge=add_gauge,
            )
            guess = jnp.concatenate([_pack_velocity(velocity_guess, dim=dim, nphys=nphys), jnp.zeros(tail_size, dtype=velocity_rhs.dtype)])
            sol = solve_sparse_system_gmres(system, rhs, guess=guess, preconditioner=preconditioner)
            return _unpack_velocity(sol, dim=dim, nphys=nphys), sol[dim * nphys : dim * nphys + npib]

        solve_batch = jit(vmap(solve_one, in_axes=(0, 0, 0), out_axes=(0, 0)))
        return solve_one, solve_batch

    def solve_projection_batch(
        self,
        alpha0: float,
        velocity_rhs_batch: jnp.ndarray,
        boundary_normal_velocity_batch: jnp.ndarray | None = None,
        velocity_guess_batch: jnp.ndarray | None = None,
        *,
        add_gauge: bool = True,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        velocity_rhs_batch = jnp.asarray(velocity_rhs_batch, dtype=float)
        if velocity_rhs_batch.ndim != 3:
            raise ValueError("velocity_rhs_batch must have shape (batch, nphys, dim)")
        batch_size = velocity_rhs_batch.shape[0]
        nub = self.velocity_physical_domain.get_num_bdry_nodes()
        if boundary_normal_velocity_batch is None:
            boundary_normal_velocity_batch = jnp.zeros((batch_size, nub), dtype=velocity_rhs_batch.dtype)
        else:
            boundary_normal_velocity_batch = jnp.asarray(boundary_normal_velocity_batch, dtype=velocity_rhs_batch.dtype)
        if velocity_guess_batch is None:
            velocity_guess_batch = jnp.zeros_like(velocity_rhs_batch)
        else:
            velocity_guess_batch = jnp.asarray(velocity_guess_batch, dtype=velocity_rhs_batch.dtype)
        _solve_one, solve_batch = self.make_jitted_projection_solver(alpha0, add_gauge=add_gauge)
        return solve_batch(velocity_rhs_batch, boundary_normal_velocity_batch, velocity_guess_batch)

    def _assemble_system_dense(self, alpha0: float, add_gauge: bool) -> jnp.ndarray:
        dim = self.domain.get_dim()
        nui = self.velocity_physical_domain.get_num_interior_nodes()
        nub = self.velocity_physical_domain.get_num_bdry_nodes()
        nphys = self.velocity_physical_domain.get_num_int_bdry_nodes()
        npib = self.domain.get_pressure_domain().get_num_int_bdry_nodes()
        vel_eye = jnp.eye(dim * nphys, dtype=float)
        grad_block = _block_gradient_matrix(self.grad_ops, self.dt / alpha0)
        div_block = _block_divergence_matrix(self.div_ops)
        lambda_block = _boundary_lambda_matrix(self.nr, nui, nphys)
        wall_block = _boundary_normal_matrix(self.nr, nui, nphys)

        top = jnp.concatenate([vel_eye, grad_block, lambda_block], axis=1)
        middle = jnp.concatenate([div_block, jnp.zeros((npib, npib + nub), dtype=float)], axis=1)
        bottom = jnp.concatenate([wall_block, jnp.zeros((nub, npib + nub), dtype=float)], axis=1)
        system = jnp.concatenate([top, middle, bottom], axis=0)
        if add_gauge:
            gauge_col = jnp.concatenate(
                [jnp.zeros((dim * nphys, 1), dtype=float), jnp.ones((npib, 1), dtype=float), jnp.zeros((nub, 1), dtype=float)],
                axis=0,
            )
            gauge_row = jnp.concatenate(
                [jnp.zeros((1, dim * nphys), dtype=float), jnp.ones((1, npib), dtype=float), jnp.zeros((1, nub + 1), dtype=float)],
                axis=1,
            )
            system = jnp.concatenate([jnp.concatenate([system, gauge_col], axis=1), gauge_row], axis=0)
        return system

    def _initial_guess(self, nphys: int, npib: int, nub: int, add_gauge: bool) -> jnp.ndarray:
        dim = self.domain.get_dim()
        state = self.un if self.completed_steps_ >= 2 else (self.unm1 if self.completed_steps_ >= 1 else self.unm2)
        velocity_guess = _pack_velocity(state, dim=dim, nphys=nphys) if state.size else jnp.zeros(dim * nphys, dtype=float)
        return jnp.concatenate([velocity_guess, jnp.zeros(npib + nub + int(add_gauge), dtype=float)])


@dataclass
class PUSLIncompressibleEulerSolver:
    domain: DualNodeDomainDescriptor = field(default_factory=DualNodeDomainDescriptor)
    xi_sl: int = 0
    dt: float = 0.0
    advection: PUSLAdvectionSolver = field(default_factory=PUSLAdvectionSolver)
    backend: IncompressibleEulerBDFBackend = field(default_factory=IncompressibleEulerBDFBackend)
    advection_domain: DomainDescriptor = field(default_factory=DomainDescriptor)
    physical_nodes: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    interior_nodes: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    coeff_nm2: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    coeff_nm1: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    coeff_n: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    completed_steps_: int = 0

    def clear_advection_boundary_condition(self) -> None:
        self.advection.clear_advection_boundary_condition()

    def set_tangential_flow_boundary(self, normal_velocity_tolerance: float = 1.0e-10) -> None:
        self.advection.set_tangential_flow_boundary(normal_velocity_tolerance)

    def set_inflow_dirichlet_boundary(self, inflow_value: Callable[..., jnp.ndarray], normal_velocity_tolerance: float = 1.0e-10) -> None:
        self.advection.set_inflow_dirichlet_boundary(inflow_value, normal_velocity_tolerance)

    def init(
        self,
        domain: DualNodeDomainDescriptor,
        xi_sl: int,
        velocity_sp: StencilProperties,
        pressure_sp: StencilProperties,
        dt: float,
        num_omp_threads: int = 1,
    ) -> None:
        if not (dt > 0):
            raise ValueError("PUSLIncompressibleEulerSolver requires dt > 0")
        self.domain = domain
        self.domain.build_structs()
        self.xi_sl = int(xi_sl)
        self.dt = float(dt)
        self.advection_domain = _strip_ghosts(self.domain.get_velocity_domain())
        self.advection.init(self.advection_domain, xi_sl, dt)
        self.backend.init(self.domain, velocity_sp, pressure_sp, dt, num_omp_threads)
        self.physical_nodes = self.advection.get_output_nodes()
        self.interior_nodes = self.domain.get_velocity_domain().get_interior_nodes()
        self.coeff_nm2 = jnp.zeros((0, self.domain.get_dim()))
        self.coeff_nm1 = jnp.zeros((0, self.domain.get_dim()))
        self.coeff_n = jnp.zeros((0, self.domain.get_dim()))
        self.completed_steps_ = 0

    def set_step_size(self, dt: float) -> None:
        if not (dt > 0):
            raise ValueError("PUSLIncompressibleEulerSolver requires dt > 0")
        self.dt = float(dt)
        self.advection.set_step_size(dt)
        self.backend.set_step_size(dt)

    def set_initial_velocity(self, physical_velocity: jnp.ndarray, _global_velocity: jnp.ndarray | None = None) -> None:
        del _global_velocity
        local_velocity = _normalize_local_physical_velocity(self, physical_velocity, "set_initial_velocity")
        self.coeff_nm2 = self.advection.project_samples(local_velocity)
        self.coeff_nm1 = jnp.zeros((0, local_velocity.shape[1]))
        self.coeff_n = jnp.zeros((0, local_velocity.shape[1]))
        self.completed_steps_ = 0
        self.backend.set_initial_velocity_owned(local_velocity)

    def set_velocity_history(self, *velocities: jnp.ndarray) -> None:
        states = [_normalize_local_physical_velocity(self, velocity, "set_velocity_history") for velocity in velocities]
        self.coeff_nm2 = self.advection.project_samples(states[0])
        self.coeff_nm1 = self.advection.project_samples(states[1]) if len(states) >= 2 else jnp.zeros((0, states[0].shape[1]))
        self.coeff_n = self.advection.project_samples(states[2]) if len(states) >= 3 else jnp.zeros((0, states[0].shape[1]))
        self.completed_steps_ = min(len(states) - 1, 2)
        self.backend.set_velocity_history_owned(*states)

    def get_interior_nodes(self) -> jnp.ndarray:
        return self.interior_nodes

    def get_output_nodes(self) -> jnp.ndarray:
        return self.physical_nodes

    def get_domain(self) -> DualNodeDomainDescriptor:
        return self.domain

    def advection_solver(self) -> PUSLAdvectionSolver:
        return self.advection

    def euler_backend(self) -> IncompressibleEulerBDFBackend:
        return self.backend

    def bdf1_step(self, t_next: float, rk, body_force: Callable[[float, jnp.ndarray], jnp.ndarray], problem: dict[str, object] | None = None) -> dict[str, object]:
        transported_nm2 = _transport_to_physical(self, t_next - self.dt, self.coeff_nm2, 1, _extrapolated_velocity_callback(self, t_next, 1), rk)
        self.backend.set_velocity_history_owned(_normalize_local_physical_velocity(self, transported_nm2, "bdf1_step"))
        sol = self.backend.bdf1_step(_force_at(t_next, body_force), problem)
        self.coeff_nm1 = self.advection.project_samples(sol["velocity"])
        self.completed_steps_ = 1
        return sol

    def bdf2_step(self, t_next: float, rk, body_force: Callable[[float, jnp.ndarray], jnp.ndarray], problem: dict[str, object] | None = None) -> dict[str, object]:
        if self.completed_steps_ < 1 or self.coeff_nm1.size == 0:
            raise ValueError("PUSLIncompressibleEulerSolver.bdf2_step requires one prior step")
        velocity = _extrapolated_velocity_callback(self, t_next, 2)
        transported_nm2 = _transport_to_physical(self, t_next - 2.0 * self.dt, self.coeff_nm2, 2, velocity, rk)
        transported_nm1 = _transport_to_physical(self, t_next - self.dt, self.coeff_nm1, 1, velocity, rk)
        self.backend.set_velocity_history_owned(transported_nm2, transported_nm1)
        sol = self.backend.bdf2_step(_force_at(t_next, body_force), problem)
        self.coeff_n = self.advection.project_samples(sol["velocity"])
        self.completed_steps_ = 2
        return sol

    def bdf3_step(self, t_next: float, rk, body_force: Callable[[float, jnp.ndarray], jnp.ndarray], problem: dict[str, object] | None = None) -> dict[str, object]:
        if self.completed_steps_ < 2 or self.coeff_n.size == 0:
            raise ValueError("PUSLIncompressibleEulerSolver.bdf3_step requires two prior steps")
        velocity = _extrapolated_velocity_callback(self, t_next, 3)
        transported_nm2 = _transport_to_physical(self, t_next - 3.0 * self.dt, self.coeff_nm2, 3, velocity, rk)
        transported_nm1 = _transport_to_physical(self, t_next - 2.0 * self.dt, self.coeff_nm1, 2, velocity, rk)
        transported_n = _transport_to_physical(self, t_next - self.dt, self.coeff_n, 1, velocity, rk)
        self.backend.set_velocity_history_owned(transported_nm2, transported_nm1, transported_n)
        sol = self.backend.bdf3_step(_force_at(t_next, body_force), problem)
        self.coeff_nm2 = self.coeff_nm1
        self.coeff_nm1 = self.coeff_n
        self.coeff_n = self.advection.project_samples(sol["velocity"])
        self.completed_steps_ = max(self.completed_steps_, 3)
        return sol

    def rollout(
        self,
        final_time: float,
        rk,
        body_force: Callable[[float, jnp.ndarray], jnp.ndarray],
        problem: dict[str, object] | None = None,
    ) -> dict[str, object]:
        if self.coeff_nm2.size == 0:
            raise ValueError("PUSLIncompressibleEulerSolver.rollout requires set_initial_velocity first")
        nsteps = int(round(float(final_time) / self.dt))
        if nsteps <= 0:
            raise ValueError("PUSLIncompressibleEulerSolver.rollout requires final_time >= dt")
        if abs(nsteps * self.dt - float(final_time)) > 100.0 * jnp.finfo(float).eps * max(1.0, abs(float(final_time))):
            raise ValueError("final_time must be an integer multiple of dt")
        sol = self.bdf1_step(self.dt, rk, body_force, problem)
        if nsteps >= 2:
            sol = self.bdf2_step(2.0 * self.dt, rk, body_force, problem)
        for step in range(3, nsteps + 1):
            sol = self.bdf3_step(step * self.dt, rk, body_force, problem)
        return sol

    def current_interior_velocity(self) -> jnp.ndarray:
        coeffs = _current_coefficients(self)
        if coeffs.size == 0:
            return jnp.zeros((0, self.interior_nodes.shape[1]))
        return self.advection.evaluate_at_points(coeffs, self.interior_nodes)

    def current_physical_velocity(self) -> jnp.ndarray:
        coeffs = _current_coefficients(self)
        if coeffs.size == 0:
            return jnp.zeros((0, self.physical_nodes.shape[1]))
        return self.advection.evaluate_at_points(coeffs, self.physical_nodes)

    def evaluate_current_velocity_at_points(self, x: jnp.ndarray) -> jnp.ndarray:
        coeffs = _current_coefficients(self)
        xq = jnp.atleast_2d(jnp.asarray(x, dtype=float))
        if coeffs.size == 0:
            return jnp.zeros((xq.shape[0], self.domain.get_dim()))
        return self.advection.evaluate_at_points(coeffs, xq)


def _validate_velocity_state(obj: IncompressibleEulerBDFBackend, velocity: jnp.ndarray, caller: str) -> jnp.ndarray:
    out = jnp.asarray(velocity, dtype=float)
    expected = (obj.velocity_physical_domain.get_num_int_bdry_nodes(), obj.domain.get_dim())
    if out.shape != expected:
        raise ValueError(f"IncompressibleEulerBDFBackend::{caller} received an invalid velocity state")
    return out


def _normalize_problem(problem: dict[str, object] | None) -> dict[str, object]:
    if not problem:
        return IncompressibleEulerBDFBackend.default_problem_definition()
    out = dict(problem)
    out.setdefault("slip_walls", [])
    out.setdefault("gauge_options", {"mode": "automatic", "nullspace_tol": 1.0e-7, "row_scale_detection": True})
    return out


def _validate_problem(obj: IncompressibleEulerBDFBackend, problem: dict[str, object]) -> None:
    nub = obj.velocity_physical_domain.get_num_bdry_nodes()
    covered = jnp.zeros(nub, dtype=int)
    for wall in problem["slip_walls"]:
        idx = jnp.asarray(wall["boundary_indices"], dtype=int).reshape(-1)
        if bool(jnp.any((idx < 1) | (idx > nub))):
            raise ValueError("IncompressibleEulerBDFBackend boundary index out of range")
        covered = covered.at[idx - 1].add(1)
    if bool(jnp.any(covered != 1)):
        raise ValueError("IncompressibleEulerBDFBackend slip walls must cover each boundary node exactly once")


def _prepare_wall_data(obj: IncompressibleEulerBDFBackend, problem: dict[str, object]) -> jnp.ndarray:
    xb = obj.velocity_physical_domain.get_bdry_nodes()
    nr = obj.velocity_physical_domain.get_nrmls()
    boundary_normal_velocity = jnp.zeros(xb.shape[0], dtype=float)
    for wall in problem["slip_walls"]:
        idx = jnp.asarray(wall["boundary_indices"], dtype=int).reshape(-1) - 1
        v = jnp.asarray(wall["normal_velocity"](xb[idx, :], nr[idx, :]), dtype=float).reshape(-1)
        if v.size != idx.size:
            raise ValueError("IncompressibleEulerBDFBackend slip wall callback returned invalid normal velocity data")
        boundary_normal_velocity = boundary_normal_velocity.at[idx].set(v)
    return boundary_normal_velocity


def _pressure_constant_is_null(obj: IncompressibleEulerBDFBackend, gauge_options: dict[str, object]) -> bool:
    mode = str(gauge_options.get("mode", "automatic")).lower()
    if mode in {"forcepressuremean", "force_pressure_mean", "automatic"}:
        return True
    if mode == "none":
        return False
    npib = obj.domain.get_pressure_domain().get_num_int_bdry_nodes()
    ones = jnp.ones(npib, dtype=float)
    residual = jnp.linalg.norm(jnp.concatenate([obj.grad_ops[d] @ ones for d in range(obj.domain.get_dim())])) / jnp.sqrt(jnp.maximum(npib, 1))
    return bool(residual <= float(gauge_options.get("nullspace_tol", 1.0e-7)))


def _normalize_local_physical_velocity(obj: PUSLIncompressibleEulerSolver, velocity: jnp.ndarray, where: str) -> jnp.ndarray:
    out = jnp.asarray(velocity, dtype=float)
    if out.shape != (obj.physical_nodes.shape[0], obj.domain.get_dim()):
        raise ValueError(f"PUSLIncompressibleEulerSolver::{where} received an invalid physical velocity")
    return out


def _force_at(t: float, body_force: Callable[[float, jnp.ndarray], jnp.ndarray]) -> Callable[[jnp.ndarray], jnp.ndarray]:
    return lambda x: body_force(t, x)


def _extrapolation_weights(theta: jnp.ndarray, order: int) -> jnp.ndarray:
    if order <= 1:
        return jnp.array([1.0, 0.0, 0.0], dtype=float)
    if order == 2:
        return jnp.array([theta + 1.0, -theta, 0.0], dtype=float)
    return jnp.array(
        [
            0.5 * (theta + 1.0) * (theta + 2.0),
            -theta * (theta + 2.0),
            0.5 * theta * (theta + 1.0),
        ],
        dtype=float,
    )


def _extrapolated_velocity_callback(obj: PUSLIncompressibleEulerSolver, t_next: float, order: int) -> Callable[[float, jnp.ndarray], jnp.ndarray]:
    coeff0 = obj.coeff_nm2
    coeff1 = obj.coeff_nm1
    coeff2 = obj.coeff_n
    dt = obj.dt

    def velocity(t: float, x: jnp.ndarray) -> jnp.ndarray:
        xq = jnp.atleast_2d(jnp.asarray(x, dtype=float))
        if coeff0.size == 0:
            return jnp.zeros((xq.shape[0], obj.domain.get_dim()))
        if order <= 1 or coeff1.size == 0:
            return obj.advection.evaluate_at_points(coeff0, xq)
        theta = (jnp.asarray(t, dtype=float) - (t_next - dt)) / dt
        w = _extrapolation_weights(theta, 2 if coeff2.size == 0 else order)
        if coeff2.size == 0:
            return w[0] * obj.advection.evaluate_at_points(coeff1, xq) + w[1] * obj.advection.evaluate_at_points(coeff0, xq)
        return (
            w[0] * obj.advection.evaluate_at_points(coeff2, xq)
            + w[1] * obj.advection.evaluate_at_points(coeff1, xq)
            + w[2] * obj.advection.evaluate_at_points(coeff0, xq)
        )

    return velocity


def _transport_to_physical(
    obj: PUSLIncompressibleEulerSolver,
    t_start: float,
    coeff_state: jnp.ndarray,
    num_steps: int,
    velocity: Callable[[float, jnp.ndarray], jnp.ndarray],
    rk,
) -> jnp.ndarray:
    if num_steps <= 0:
        raise ValueError("PUSLIncompressibleEulerSolver transport requires a positive step count")
    return obj.advection.backward_trace_evaluate_at_points(
        t_start,
        int(num_steps),
        coeff_state,
        obj.physical_nodes,
        velocity,
        rk,
    )


def _current_coefficients(obj: PUSLIncompressibleEulerSolver) -> jnp.ndarray:
    if obj.completed_steps_ <= 0:
        return obj.coeff_nm2
    if obj.completed_steps_ == 1:
        return obj.coeff_nm1
    return obj.coeff_n
