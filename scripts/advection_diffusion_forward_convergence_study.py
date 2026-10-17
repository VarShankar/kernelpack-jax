from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import sys
import traceback
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence
from scipy.io import savemat

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

INV = Path(__file__).resolve().parents[1] / "inverse problems"
if str(INV) not in sys.path:
    sys.path.insert(0, str(INV))

from kernelpack.domain import DomainDescriptor
from kernelpack.geometry import RBFLevelSet, distance_matrix
from kernelpack.nodes import DomainNodeGenerator
from kernelpack.solvers import EulerianAdvectionDiffusionReactionSolver, StatelessEulerianAdvectionDiffusionReactionSolver
from kernelpack.solvers._common import FrozenILUPreconditioner, SparseCOOMatrix
from kernelpack.solvers.eulerian_advection_diffusion_reaction import paper_gamma_component

from common import _build_surface_2d, build_domain, default_h, relative_l2
from reaction_diffusion_inverse_convergence_study import SavedDomainMetadata, default_geometry, validate_geometry


DEFAULT_MAT_OUTPUT = Path(__file__).with_name("advection_diffusion_forward_convergence_study.mat")


def progress(message: str) -> None:
    print(f"[progress] {message}", flush=True)


def setup_device_context(setup_device: str):
    if setup_device == "cpu":
        return jax.default_device(jax.devices("cpu")[0])
    return nullcontext()


def _put_array_on_device(value, device):
    if isinstance(value, jax.Array):
        return jax.device_put(value, device)
    return value


def _put_sparse_on_device(matrix: SparseCOOMatrix | None, device) -> SparseCOOMatrix | None:
    if matrix is None:
        return None
    return SparseCOOMatrix(
        indices=jax.device_put(matrix.indices, device),
        values=jax.device_put(matrix.values, device),
        shape=matrix.shape,
    )


def _put_preconditioner_on_device(
    preconditioner: FrozenILUPreconditioner | None,
    device,
) -> FrozenILUPreconditioner | None:
    if preconditioner is None:
        return None
    if preconditioner.application == "host_callback":
        return preconditioner
    return FrozenILUPreconditioner(
        apply_matrix=jax.device_put(preconditioner.apply_matrix, device),
        transpose_apply_matrix=jax.device_put(preconditioner.transpose_apply_matrix, device),
        host_ilu=preconditioner.host_ilu,
        application=preconditioner.application,
        drop_tol=preconditioner.drop_tol,
        fill_factor=preconditioner.fill_factor,
    )


def move_solver_arrays_to_device(
    solver: EulerianAdvectionDiffusionReactionSolver,
    device,
) -> EulerianAdvectionDiffusionReactionSolver:
    solver.x = _put_array_on_device(solver.x, device)
    solver.xall = _put_array_on_device(solver.xall, device)
    solver.xb = _put_array_on_device(solver.xb, device)
    solver.nr = _put_array_on_device(solver.nr, device)
    solver.lap = _put_array_on_device(solver.lap, device)
    solver.hyper = _put_array_on_device(solver.hyper, device)
    solver.bc = _put_array_on_device(solver.bc, device)
    solver.grad = [_put_array_on_device(grad, device) for grad in solver.grad]
    solver.lap_sparse = _put_sparse_on_device(solver.lap_sparse, device)
    solver.hyper_sparse = _put_sparse_on_device(solver.hyper_sparse, device)
    solver.bc_sparse = _put_sparse_on_device(solver.bc_sparse, device)
    solver.grad_sparse = [_put_sparse_on_device(grad, device) for grad in solver.grad_sparse]
    solver.cfnm2 = _put_array_on_device(solver.cfnm2, device)
    solver.cfnm1 = _put_array_on_device(solver.cfnm1, device)
    solver.cfn = _put_array_on_device(solver.cfn, device)
    solver.gamma_components = _put_array_on_device(solver.gamma_components, device)
    solver.cached_neu_coeff_ = _put_array_on_device(solver.cached_neu_coeff_, device)
    solver.cached_dir_coeff_ = _put_array_on_device(solver.cached_dir_coeff_, device)
    solver.fixed_preconditioner = _put_preconditioner_on_device(solver.fixed_preconditioner, device)
    return solver


@dataclass(frozen=True)
class ForwardResult:
    order: int
    resolution: str
    h: float
    dt: float
    steps: int
    final_time: float
    node_count: int
    final_relative_l2_error: float
    mean_relative_l2_error: float
    max_relative_l2_error: float
    elapsed_seconds: float
    preprocess_elapsed_seconds: float
    gamma_status: str
    hyperviscosity_power: int


def load_domain_file(path: Path, *, rebuild_level_set: bool = True) -> tuple[DomainDescriptor, SavedDomainMetadata]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        bdry_nodes = jnp.asarray(data["bdry_nodes"], dtype=float)
        normals = jnp.asarray(data["normals"], dtype=float)
        descriptor = DomainDescriptor()
        descriptor.set_nodes(
            jnp.asarray(data["int_nodes"], dtype=float),
            bdry_nodes,
            jnp.asarray(data["ghost_nodes"], dtype=float),
        )
        descriptor.set_normals(normals)
        descriptor.set_sep_rad(float(np.asarray(data["sep_rad"]).item()))
        descriptor.build_structs()
        if rebuild_level_set:
            level_set = RBFLevelSet()
            level_set.build_level_set_from_cfi(bdry_nodes, normals)
            descriptor.set_outer_level_set(level_set)
            descriptor.set_boundary_level_sets([level_set])
        metadata = SavedDomainMetadata(
            dim=int(np.asarray(data["dim"]).item()),
            geometry=str(np.asarray(data["geometry"]).item()),
            h=float(np.asarray(data["h"]).item()),
            seed=int(np.asarray(data["seed"]).item()),
            sep_rad=float(np.asarray(data["sep_rad"]).item()),
            label=str(np.asarray(data["label"]).item()),
        )
    return descriptor, metadata


def build_forward_domain(dim: int, geometry_name: str, h: float, seed: int, *, mapped_cloud_preferred: bool) -> DomainDescriptor:
    if dim == 2 and mapped_cloud_preferred:
        surface = _build_surface_2d(geometry_name)
        generator = DomainNodeGenerator()
        return generator.build_domain_descriptor_from_geometry(
            surface,
            h,
            seed=seed,
            strip_count=5,
            do_outer_refinement=True,
            outer_fraction_of_h=0.5,
            outer_refinement_zone_size_as_multiple_of_h=2.0,
            mapped_cloud_preferred=True,
        )
    return build_domain(dim, geometry_name, h, seed)


def timestep_from_h(h: float, final_time: float, dt_factor: float, min_steps: int) -> tuple[float, int]:
    if h <= 0.0:
        raise ValueError("h must be positive for h-dependent timestep selection")
    if final_time <= 0.0:
        raise ValueError("final_time must be positive")
    if dt_factor <= 0.0:
        raise ValueError("dt_factor must be positive")
    steps = max(int(min_steps), int(np.ceil(float(final_time) / (float(dt_factor) * float(h)))))
    if steps <= 0:
        raise ValueError("computed timestep count must be positive")
    return float(final_time) / float(steps), steps


def _level_set_values(domain: DomainDescriptor, x: jnp.ndarray) -> jnp.ndarray:
    level_set = domain.get_outer_level_set()
    if level_set is None:
        raise ValueError("Varun-style Eulerian ADR manufactured cases require an outer level set")
    return jnp.asarray(level_set.evaluate(x), dtype=float)


def _level_set_gradients(domain: DomainDescriptor, x: jnp.ndarray) -> jnp.ndarray:
    level_set = domain.get_outer_level_set()
    if level_set is None:
        raise ValueError("Varun-style Eulerian ADR manufactured cases require an outer level set")
    return jnp.asarray(level_set.evaluate_gradient(x), dtype=float)


def incompressible_boundary_vanishing_velocity(domain: DomainDescriptor, speed: float):
    """Varun-style incompressible velocity with zero boundary flux.

    In 2D this is v = (psi_y, -psi_x), and in 3D it is curl(A). The
    phi_+^2 factor makes the potential gradient vanish on the boundary.
    """

    x_phys = jnp.asarray(domain.get_int_bdry_nodes(), dtype=float)
    center = jnp.mean(x_phys, axis=0)
    dim = int(x_phys.shape[1])
    phi_phys = jnp.maximum(_level_set_values(domain, x_phys), 0.0)
    grad_phi_phys = jnp.where(phi_phys[:, None] > 0.0, _level_set_gradients(domain, x_phys), 0.0)
    shifted_phys = x_phys - center

    if dim == 2:
        q_phys = 0.5 * (shifted_phys[:, 0] ** 2 + shifted_phys[:, 1] ** 2)
        grad_q_phys = shifted_phys
    elif dim == 3:
        q_phys = 0.5 * (shifted_phys[:, 0] ** 2 + shifted_phys[:, 2] ** 2)
        grad_q_phys = jnp.column_stack([shifted_phys[:, 0], jnp.zeros(x_phys.shape[0]), shifted_phys[:, 2]])
    else:
        raise ValueError("Only dim=2 or dim=3 is supported")

    grad_potential_phys = 2.0 * phi_phys[:, None] * grad_phi_phys * q_phys[:, None] + (phi_phys**2)[:, None] * grad_q_phys
    if dim == 2:
        base_phys = jnp.column_stack([grad_potential_phys[:, 1], -grad_potential_phys[:, 0]])
    else:
        base_phys = jnp.column_stack([-grad_potential_phys[:, 2], jnp.zeros(x_phys.shape[0]), grad_potential_phys[:, 0]])
    max_base = jnp.maximum(jnp.max(jnp.linalg.norm(base_phys, axis=1)), 1.0e-14)

    def velocity(time: float | jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        xq = jnp.asarray(x, dtype=float)
        phi_q = jnp.maximum(_level_set_values(domain, xq), 0.0)
        grad_phi_q = jnp.where(phi_q[:, None] > 0.0, _level_set_gradients(domain, xq), 0.0)
        shifted = xq - center
        if dim == 2:
            q = 0.5 * (shifted[:, 0] ** 2 + shifted[:, 1] ** 2)
            grad_q = shifted
            grad_psi = 2.0 * phi_q[:, None] * grad_phi_q * q[:, None] + (phi_q**2)[:, None] * grad_q
            base = jnp.column_stack([grad_psi[:, 1], -grad_psi[:, 0]])
        elif dim == 3:
            q = 0.5 * (shifted[:, 0] ** 2 + shifted[:, 2] ** 2)
            grad_q = jnp.column_stack([shifted[:, 0], jnp.zeros(xq.shape[0]), shifted[:, 2]])
            grad_a = 2.0 * phi_q[:, None] * grad_phi_q * q[:, None] + (phi_q**2)[:, None] * grad_q
            base = jnp.column_stack([-grad_a[:, 2], jnp.zeros(xq.shape[0]), grad_a[:, 0]])
        else:
            raise ValueError("Only dim=2 or dim=3 is supported")
        return speed * jnp.sin(jnp.pi * time) * base / max_base

    return velocity


def spatial_mode(x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=float)
    return jnp.prod(jnp.cos(jnp.pi * x), axis=1)


def grad_spatial_mode(x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=float)
    cols = []
    for d in range(int(x.shape[1])):
        factors = []
        for axis in range(int(x.shape[1])):
            if axis == d:
                factors.append(-jnp.pi * jnp.sin(jnp.pi * x[:, axis]))
            else:
                factors.append(jnp.cos(jnp.pi * x[:, axis]))
        cols.append(jnp.prod(jnp.column_stack(factors), axis=1))
    return jnp.column_stack(cols)


def lap_spatial_mode(x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=float)
    return -(jnp.pi**2) * x.shape[1] * spatial_mode(x)


def exact_state(time: float | jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    return 1.0 + 0.15 * jnp.cos(0.8 * time) * spatial_mode(x)


def exact_gradient(time: float | jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    return 0.15 * jnp.cos(0.8 * time) * grad_spatial_mode(x)


def exact_laplacian(time: float | jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    return 0.15 * jnp.cos(0.8 * time) * lap_spatial_mode(x)


def manufactured_laplacian_boundary_trace(order: int, time, x):
    """Continuous prescribed traces for the repeated-Laplacian MMS.

    This is boundary data for each intermediate Laplacian, never a discrete
    forcing correction or an observation generated by the numerical solver.
    """
    if order == 0:
        return exact_state(time, x)
    return (0.15 * jnp.cos(0.8 * time)
            * (-x.shape[1] * jnp.pi**2)**order * spatial_mode(x))


def exact_time_derivative(time: float | jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    return -0.15 * 0.8 * jnp.sin(0.8 * time) * spatial_mode(x)


def manufactured_forcing(nu: float, velocity, rho: float):
    def forcing(_nu_value: float, time: float, x: jnp.ndarray) -> jnp.ndarray:
        u = exact_state(time, x)
        u_t = exact_time_derivative(time, x)
        grad_u = exact_gradient(time, x)
        lap_u = exact_laplacian(time, x)
        vel = velocity(time, x)
        advective = jnp.sum(vel * grad_u, axis=1)
        reaction = rho * u * (1.0 - u)
        return u_t + advective - reaction - nu * lap_u

    return forcing


def make_discrete_bdf_forcing(
    solver: EulerianAdvectionDiffusionReactionSolver,
    gamma: jnp.ndarray,
    *,
    velocity,
    rho: float,
    steps: int,
    dt: float,
    compensate_hyperviscosity: bool = True,
):
    """Build a forcing sequence that removes BDF/AB time truncation error.

    This is a manufactured-solution diagnostic: the exact state is inserted into
    the same BDF1/BDF2/BDF3 time formula, but the target forcing is based on the
    continuous advection-diffusion-reaction operators. Since hyperviscosity is a
    numerical stabilizer rather than part of the target PDE, the discrete
    hyperviscosity term is optionally canceled against the exact state.
    """

    x_phys = jnp.asarray(solver.x, dtype=float)
    x_all = jnp.asarray(solver.xall, dtype=float)
    gamma_sum = jnp.sum(jnp.asarray(gamma, dtype=float))

    def exact_full(time: float) -> jnp.ndarray:
        return exact_state(time, x_all)

    def discrete_hyperviscosity(time: float) -> jnp.ndarray:
        if not compensate_hyperviscosity:
            return jnp.zeros((solver.n,), dtype=float)
        return gamma_sum * solver._apply_physical_operator(solver.hyper, solver.hyper_sparse, exact_full(time))

    def continuous_explicit_without_forcing(time: float) -> jnp.ndarray:
        u = exact_state(time, x_phys)
        vel = velocity(time, x_phys)
        adv = -jnp.sum(vel * exact_gradient(time, x_phys), axis=1)
        reaction = rho * u * (1.0 - u)
        return adv + reaction

    target_explicit: list[jnp.ndarray] = []
    for step in range(1, steps + 1):
        t_next = step * dt
        u_np1 = exact_full(t_next)
        if step == 1:
            u_n = exact_full(0.0)
            derivative = (u_np1[: solver.n] - u_n[: solver.n]) / dt
        elif step == 2:
            u_nm2 = exact_full(0.0)
            u_nm1 = exact_full(dt)
            derivative = (3.0 * u_np1[: solver.n] - 4.0 * u_nm1[: solver.n] + u_nm2[: solver.n]) / (2.0 * dt)
        else:
            u_nm2 = exact_full((step - 3) * dt)
            u_nm1 = exact_full((step - 2) * dt)
            u_n = exact_full((step - 1) * dt)
            derivative = (11.0 * u_np1[: solver.n] - 18.0 * u_n[: solver.n] + 9.0 * u_nm1[: solver.n] - 2.0 * u_nm2[: solver.n]) / (6.0 * dt)
        target_explicit.append(
            derivative
            - solver.nu * exact_laplacian(t_next, x_phys)
            - discrete_hyperviscosity(t_next)
        )

    explicit_values: list[jnp.ndarray] = []
    forcing_values: list[jnp.ndarray] = []
    for history_index in range(steps):
        time = history_index * dt
        if history_index == 0:
            explicit_value = target_explicit[0]
        elif history_index == 1:
            explicit_value = 0.5 * (target_explicit[1] + explicit_values[0])
        else:
            explicit_value = (target_explicit[history_index] + 3.0 * explicit_values[history_index - 1] - explicit_values[history_index - 2]) / 3.0
        explicit_values.append(explicit_value)
        forcing_values.append(explicit_value - continuous_explicit_without_forcing(time))

    forcing_table = jnp.stack(forcing_values, axis=0)

    def forcing(_nu_value: float, time: float, x: jnp.ndarray) -> jnp.ndarray:
        del x
        idx = jnp.rint(jnp.asarray(time, dtype=float) / float(dt)).astype(jnp.int32)
        idx = jnp.clip(idx, 0, int(forcing_table.shape[0]) - 1)
        return jnp.take(forcing_table, idx, axis=0)

    return forcing


def build_solver_template(
    domain: DomainDescriptor,
    *,
    order: int,
    dt: float,
    nu: float,
    stencil: str,
    linear_solver: str,
    velocity_amplitude: float = 0.12,
    gamma_mode: str = "estimate_or_fixed",
    fixed_gamma: float = -1.0e-4,
    fixed_gamma_components=None,
    gamma_scale: float = 1.0,
    gamma_h_power: float = 0.0,
    hyperviscosity_power: str = "2",
    hyperviscosity_method: str = "laplacian_power",
    hyperviscosity_treatment: str = "explicit",
    laplacian_power_solver: str = "sparse",
    hyperviscosity_boundary_traces=None,
    boundary_order: int | None = None,
    hyperviscosity_row_mask: str = "none",
    hyperviscosity_mask_distance_factor: float = 2.0,
    sparse_assembly_chunk_size: int | None = None,
    gamma_arpack_ncv: int | None = None,
    gamma_arpack_maxiter: int | None = None,
    setup_device: str = "default",
    gmres_tol: float = 1.0e-8,
    gmres_atol: float = 1.0e-10,
    gmres_restart: int = 40,
    gmres_maxiter: int | None = None,
    ilu_application: str = "dense",
    linear_solve_diagnostics: bool = False,
    linear_solve_diagnostics_threshold: float = 1.0e-6,
    force_boundary_paired_ghost_stencil: bool = False,
    return_to_default_device: bool = True,
) -> tuple[EulerianAdvectionDiffusionReactionSolver, jnp.ndarray, str]:
    def init_solver(power: int) -> EulerianAdvectionDiffusionReactionSolver:
        solver = EulerianAdvectionDiffusionReactionSolver(linear_solver=linear_solver)
        solver.lap_stencil = stencil
        solver.bc_stencil = stencil
        solver.gmres_tol = gmres_tol
        solver.gmres_atol = gmres_atol
        solver.gmres_restart = gmres_restart
        solver.gmres_maxiter = gmres_maxiter
        solver.linear_solve_diagnostics = linear_solve_diagnostics
        solver.linear_solve_diagnostics_threshold = linear_solve_diagnostics_threshold
        solver.init(
            domain,
            order,
            dt,
            nu,
            boundary_xi=boundary_order,
            hyperviscosity_power=power,
            hyperviscosity_method=hyperviscosity_method,
            hyperviscosity_treatment=hyperviscosity_treatment,
            laplacian_power_solver=laplacian_power_solver,
            # The paper's laplacian_power path uses the full-square sparse
            # matrix power and crops ghost rows after the power is formed.
            # It does not prescribe traces for intermediate Laplacians.
            hyperviscosity_boundary_traces=hyperviscosity_boundary_traces,
            hyperviscosity_row_mask=hyperviscosity_row_mask,
            hyperviscosity_mask_distance_factor=hyperviscosity_mask_distance_factor,
            sparse_assembly_chunk_size=sparse_assembly_chunk_size,
            force_boundary_paired_ghost_stencil=force_boundary_paired_ghost_stencil,
            assemble_dense_operators=linear_solver not in ("gmres", "gmres_ilu", "cudss"),
        )
        return solver

    with setup_device_context(setup_device):
        requested_power = str(hyperviscosity_power).strip().lower()
        chosen_power = 2 if requested_power == "auto" else int(requested_power)
        if chosen_power < 1:
            raise ValueError("--hyperviscosity-power must be a positive integer or 'auto'")
        solver = init_solver(chosen_power)
        auto_q_text = ""
        trial_estimate = None
        if requested_power == "auto":
            if gamma_mode == "fixed":
                raise ValueError("--hyperviscosity-power auto requires gamma estimation; use --gamma-mode estimate or estimate_or_fixed")
            try:
                trial_estimate = solver.estimate_hyperviscosity_from_paper(
                    velocity_scale=velocity_amplitude,
                    arpack_ncv=gamma_arpack_ncv,
                    arpack_maxiter=gamma_arpack_maxiter,
                )
                q_max = float(np.max(np.asarray(trial_estimate.q)))
                chosen_power = max(1, int(np.ceil((float(order) + q_max) / 2.0)))
                auto_q_text = f"_auto_k{chosen_power}_qmax{q_max:.2f}"
                if hyperviscosity_method == "laplacian_power":
                    solver.set_laplacian_power(chosen_power)
                else:
                    solver = init_solver(chosen_power)
            except ArpackNoConvergence:
                if gamma_mode != "estimate_or_fixed":
                    raise
                print(
                    "[warning] hyperviscosity eigenvalue estimate did not converge during auto-power selection; "
                    "using k=2 with fixed gamma fallback",
                    flush=True,
                )
                chosen_power = 2
                auto_q_text = "_auto_failed_k2"
                if hyperviscosity_method == "laplacian_power":
                    solver.set_laplacian_power(chosen_power)
                else:
                    solver = init_solver(chosen_power)
        else:
            chosen_power = int(requested_power)
            if chosen_power < 1:
                raise ValueError("--hyperviscosity-power must be a positive integer or 'auto'")

        if gamma_mode == "fixed":
            if fixed_gamma_components is None:
                gamma = jnp.full((solver.dim,), fixed_gamma, dtype=float)
                if fixed_gamma * (-1.0) ** (chosen_power + 1) < 0:
                    print(f"[warning] fixed gamma {fixed_gamma:.3e} is anti-dissipative for k={chosen_power}", flush=True)
            else:
                gamma = jnp.asarray(fixed_gamma_components, dtype=float)
                if gamma.shape != (solver.dim,) or not bool(jnp.all(jnp.isfinite(gamma))):
                    raise ValueError("fixed_gamma_components must be a finite vector of length solver.dim")
            solver.set_gamma_components(gamma)
            gamma_status = (f"fixed_components_k{chosen_power}{auto_q_text}"
                            if fixed_gamma_components is not None else f"fixed_k{chosen_power}{auto_q_text}")
        else:
            if trial_estimate is not None:
                gamma = jnp.asarray(
                    [
                        paper_gamma_component(
                            float(trial_estimate.h),
                            velocity_amplitude,
                            max(abs(float(tau)), np.finfo(float).tiny),  # |tau|, as in estimate_hyperviscosity_from_paper
                            float(q),
                            chosen_power,
                            dimension=solver.dim,
                        )
                        for tau, q in zip(np.asarray(trial_estimate.tau), np.asarray(trial_estimate.q), strict=True)
                    ],
                    dtype=float,
                )
                solver.set_gamma_components(gamma)
                gamma_status = f"estimated_reused_q_k{chosen_power}{auto_q_text}"
            else:
                try:
                    gamma = solver.estimate_hyperviscosity_from_paper(
                        velocity_scale=velocity_amplitude,
                        arpack_ncv=gamma_arpack_ncv,
                        arpack_maxiter=gamma_arpack_maxiter,
                    ).gamma_components
                    gamma_status = f"estimated_k{chosen_power}{auto_q_text}"
                except ArpackNoConvergence:
                    if gamma_mode != "estimate_or_fixed":
                        raise
                    # Fallback must be dissipative: (-1)^(k+1) Delta^k with gamma > 0.
                    gamma = jnp.full((solver.dim,), (-1.0) ** (chosen_power + 1) * abs(fixed_gamma), dtype=float)
                    solver.set_gamma_components(gamma)
                    gamma_status = f"estimate_failed_fixed_k{chosen_power}{auto_q_text}"
                    print(
                        "[warning] hyperviscosity eigenvalue estimate did not converge; "
                        f"using fixed gamma={fixed_gamma:.3e} for order={order}, k={chosen_power}",
                        flush=True,
                    )
        gamma_multiplier = float(gamma_scale) * float(domain.get_sep_rad()) ** float(gamma_h_power)
        gamma = gamma * gamma_multiplier
        solver.set_gamma_components(gamma)
        if gamma_multiplier != 1.0:
            gamma_status = f"{gamma_status}_scaled"
        neu_coeff = lambda xb: jnp.zeros((xb.shape[0],), dtype=float)
        dir_coeff = lambda xb: jnp.ones((xb.shape[0],), dtype=float)
        solver.prepare_boundary_operator(0.0, neu_coeff, dir_coeff)
        if linear_solver == "gmres_ilu":
            solver.prepare_preconditioner(
                0.0,
                neu_coeff,
                dir_coeff,
                nu_ref=nu,
                gamma_components_ref=gamma,
                bdf_order=3,
                application=ilu_application,
            )

    if setup_device == "cpu" and return_to_default_device:
        target_device = jax.devices()[0]
        solver = move_solver_arrays_to_device(solver, target_device)
        gamma = jax.device_put(gamma, target_device)
        # The preconditioner was already built against this gamma during CPU
        # setup. Calling set_gamma_components here would conservatively clear
        # that frozen ILU, so just move the stored value to the target device.
        solver.gamma_components = gamma
    return solver, gamma, gamma_status


def synchronize_solver_operators(solver: EulerianAdvectionDiffusionReactionSolver) -> None:
    if solver.lap.size:
        jax.block_until_ready(solver.lap)
    if solver.hyper.size:
        jax.block_until_ready(solver.hyper)
    if solver.lap_sparse is not None:
        jax.block_until_ready(solver.lap_sparse.indices)
        jax.block_until_ready(solver.lap_sparse.values)
    if solver.hyper_sparse is not None:
        jax.block_until_ready(solver.hyper_sparse.indices)
        jax.block_until_ready(solver.hyper_sparse.values)
    if solver.bc.size:
        jax.block_until_ready(solver.bc)
    if solver.bc_sparse is not None:
        jax.block_until_ready(solver.bc_sparse.indices)
        jax.block_until_ready(solver.bc_sparse.values)


def rollout_component_norms(
    solver: EulerianAdvectionDiffusionReactionSolver,
    gamma: jnp.ndarray,
    *,
    state_full: jnp.ndarray,
    time: float,
    velocity,
    forcing,
    rho: float,
) -> dict[str, jnp.ndarray]:
    physical = state_full[: solver.n]
    vel = velocity(time, jnp.asarray(solver.x, dtype=float))
    advection = jnp.zeros((solver.n,), dtype=float)
    for d in range(solver.dim):
        grad_d = solver._apply_physical_operator(solver.grad[d], solver.grad_sparse[d], state_full)
        advection = advection - vel[:, d] * grad_d
    reaction = rho * physical * (1.0 - physical)
    forcing_value = jnp.zeros((solver.n,), dtype=float) if forcing is None else forcing(solver.nu, time, jnp.asarray(solver.x, dtype=float))
    diffusion = solver.nu * solver._apply_physical_operator(solver.lap, solver.lap_sparse, state_full)
    hyperviscosity = jnp.sum(jnp.asarray(gamma, dtype=float)) * solver._apply_physical_operator(
        solver.hyper,
        solver.hyper_sparse,
        state_full,
    )
    rhs = advection + reaction + forcing_value + diffusion + hyperviscosity
    return {
        "state_norm": jnp.linalg.norm(physical),
        "state_max_abs": jnp.max(jnp.abs(physical)),
        "advection_norm": jnp.linalg.norm(advection),
        "diffusion_norm": jnp.linalg.norm(diffusion),
        "hyperviscosity_norm": jnp.linalg.norm(hyperviscosity),
        "reaction_norm": jnp.linalg.norm(reaction),
        "forcing_norm": jnp.linalg.norm(forcing_value),
        "rhs_norm": jnp.linalg.norm(rhs),
        "finite": jnp.all(jnp.isfinite(state_full)),
    }


def physical_region_masks(solver: EulerianAdvectionDiffusionReactionSolver, near_boundary_factor: float) -> dict[str, jnp.ndarray]:
    n_interior = int(solver.domain.get_num_interior_nodes())
    n_boundary = int(solver.domain.get_num_bdry_nodes())
    interior = jnp.arange(solver.n) < n_interior
    boundary = (jnp.arange(solver.n) >= n_interior) & (jnp.arange(solver.n) < n_interior + n_boundary)
    if n_boundary == 0 or n_interior == 0:
        near_boundary = boundary
        far_interior = interior
    else:
        x_interior = jnp.asarray(solver.x[:n_interior], dtype=float)
        xb = jnp.asarray(solver.xb, dtype=float)
        nearest = jnp.min(distance_matrix(x_interior, xb), axis=1)
        near_interior = nearest <= float(near_boundary_factor) * float(solver.domain.get_sep_rad())
        near_boundary = jnp.concatenate([near_interior, jnp.ones((n_boundary,), dtype=bool)])
        far_interior = jnp.concatenate([jnp.logical_not(near_interior), jnp.zeros((n_boundary,), dtype=bool)])
    return {
        "interior": interior,
        "boundary": boundary,
        "near_boundary": near_boundary,
        "far_interior": far_interior,
    }


def masked_relative_l2(value: jnp.ndarray, reference: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    count = jnp.sum(mask)
    masked_value = jnp.where(mask, value, 0.0)
    masked_reference = jnp.where(mask, reference, 0.0)
    denom = jnp.linalg.norm(masked_reference)
    err = jnp.linalg.norm(masked_value - masked_reference) / jnp.maximum(denom, jnp.finfo(value.dtype).tiny)
    return jnp.where(count > 0, err, jnp.nan)


def print_rollout_diagnostic(
    solver: EulerianAdvectionDiffusionReactionSolver,
    gamma: jnp.ndarray,
    *,
    state_full: jnp.ndarray,
    step: int,
    steps: int,
    time: float,
    velocity,
    forcing,
    rho: float,
    case_label: str,
    masks: dict[str, jnp.ndarray] | None = None,
) -> float:
    x_phys = jnp.asarray(solver.x, dtype=float)
    exact = exact_state(time, x_phys)
    error = relative_l2(state_full[: solver.n], exact)
    split_errors = {}
    if masks is not None:
        physical = state_full[: solver.n]
        split_errors = {
            name: jax.block_until_ready(masked_relative_l2(physical, exact, mask))
            for name, mask in masks.items()
        }
    norms = rollout_component_norms(
        solver,
        gamma,
        state_full=state_full,
        time=time,
        velocity=velocity,
        forcing=forcing,
        rho=rho,
    )
    values = {name: jax.block_until_ready(value) for name, value in norms.items()}
    error = jax.block_until_ready(error)
    error_float = float(error)
    print(
        "[rollout] "
        f"{case_label}, step={step}/{steps}, t={time:.6e}, rel_l2={error_float:.6e}, "
        f"state_norm={float(values['state_norm']):.6e}, "
        f"state_max_abs={float(values['state_max_abs']):.6e}, "
        f"interior_rel_l2={float(split_errors['interior']):.6e}, " if masks is not None else "",
        f"boundary_rel_l2={float(split_errors['boundary']):.6e}, " if masks is not None else "",
        f"near_boundary_rel_l2={float(split_errors['near_boundary']):.6e}, " if masks is not None else "",
        f"far_interior_rel_l2={float(split_errors['far_interior']):.6e}, " if masks is not None else "",
        f"rhs_norm={float(values['rhs_norm']):.6e}, "
        f"adv_norm={float(values['advection_norm']):.6e}, "
        f"diff_norm={float(values['diffusion_norm']):.6e}, "
        f"hyper_norm={float(values['hyperviscosity_norm']):.6e}, "
        f"reaction_norm={float(values['reaction_norm']):.6e}, "
        f"forcing_norm={float(values['forcing_norm']):.6e}, "
        f"gamma_sum={float(jnp.sum(jnp.asarray(gamma, dtype=float))):.6e}, "
        f"finite={bool(values['finite'])}",
        flush=True,
    )
    return error_float


def rollout_final_state(
    solver: EulerianAdvectionDiffusionReactionSolver,
    gamma: jnp.ndarray,
    *,
    initial_state: jnp.ndarray,
    velocity,
    forcing,
    steps: int,
    rho: float,
    startup_mode: str,
    diagnostics_every: int = 0,
    diagnostics_growth_limit: float = float("inf"),
    diagnostics_near_boundary_factor: float = 2.0,
    case_label: str = "",
) -> jnp.ndarray:
    stateless = StatelessEulerianAdvectionDiffusionReactionSolver.from_solver(solver)
    reaction = lambda time, state, x: rho * state * (1.0 - state)
    bc = lambda _neu_coeffs, _dir_coeffs, _nr, time, xb: exact_state(time, xb)
    if startup_mode == "stepped":
        if diagnostics_every > 0:
            progress("rollout diagnostics are only available for startup_mode=exact_history; running stepped rollout without checkpoints")
        return stateless.march_to_final_physical_state(
            initial_state,
            steps=steps,
            velocity=velocity,
            reaction=reaction,
            bc=bc,
            forcing=forcing,
            gamma_components=gamma,
        )
    if startup_mode != "exact_history":
        raise ValueError(f"unknown startup mode {startup_mode}")
    if steps < 3:
        return exact_state(steps * solver.dt, jnp.asarray(solver.x, dtype=float))

    x_all = jnp.asarray(solver.xall, dtype=float)
    u0 = exact_state(0.0, x_all)
    u1 = exact_state(solver.dt, x_all)
    u2 = exact_state(2.0 * solver.dt, x_all)
    u_nm2, u_nm1, u_n = u0, u1, u2
    masks = physical_region_masks(solver, diagnostics_near_boundary_factor) if diagnostics_every > 0 else None
    if diagnostics_every > 0:
        print_rollout_diagnostic(
            solver,
            gamma,
            state_full=u_n,
            step=2,
            steps=steps,
            time=2.0 * solver.dt,
            velocity=velocity,
            forcing=forcing,
            rho=rho,
            case_label=case_label,
            masks=masks,
        )
    for k in range(2, steps):
        t_nm2 = (k - 2) * solver.dt
        t_nm1 = (k - 1) * solver.dt
        t_n = k * solver.dt
        t_np1 = (k + 1) * solver.dt
        u_np1 = stateless.bdf3_full_state(
            u_nm2,
            u_nm1,
            u_n,
            t_nm2,
            t_nm1,
            t_n,
            t_np1,
            velocity=velocity,
            reaction=reaction,
            bc=bc,
            forcing=forcing,
            gamma_components=gamma,
        )
        u_nm2, u_nm1, u_n = u_nm1, u_n, u_np1
        step = k + 1
        if diagnostics_every > 0 and (step % diagnostics_every == 0 or step == steps):
            error_float = print_rollout_diagnostic(
                solver,
                gamma,
                state_full=u_n,
                step=step,
                steps=steps,
                time=t_np1,
                velocity=velocity,
                forcing=forcing,
                rho=rho,
                case_label=case_label,
                masks=masks,
            )
            if (not np.isfinite(error_float)) or error_float > diagnostics_growth_limit:
                raise RuntimeError(
                    "rollout diagnostics stopped after explosive/nonfinite growth: "
                    f"step={step}, rel_l2={error_float:.6e}, limit={diagnostics_growth_limit:.6e}"
                )
    return u_n[: solver.n]


def build_forcing_for_mode(
    forcing_mode: str,
    solver: EulerianAdvectionDiffusionReactionSolver,
    gamma: jnp.ndarray,
    *,
    nu: float,
    velocity,
    rho: float,
    steps: int,
    dt: float,
    compensate_hyperviscosity: bool,
):
    if forcing_mode == "continuous":
        return manufactured_forcing(nu, velocity, rho)
    if forcing_mode == "discrete_bdf":
        return make_discrete_bdf_forcing(
            solver,
            gamma,
            velocity=velocity,
            rho=rho,
            steps=steps,
            dt=dt,
            compensate_hyperviscosity=compensate_hyperviscosity,
        )
    raise ValueError(f"unknown forcing mode {forcing_mode}")


def run_case(
    *,
    domain: DomainDescriptor,
    metadata: SavedDomainMetadata,
    order: int,
    xi_sl: int,
    stencil: str,
    linear_solver: str,
    final_time: float,
    dt_factor: float,
    min_steps: int,
    fixed_steps: int | None,
    nu: float,
    velocity_amplitude: float,
    rho: float,
    gamma_mode: str,
    fixed_gamma: float,
    gamma_scale: float,
    gamma_h_power: float,
    hyperviscosity_power: str,
    boundary_order: int | None,
    hyperviscosity_row_mask: str,
    hyperviscosity_mask_distance_factor: float,
    sparse_assembly_chunk_size: int | None,
    gamma_arpack_ncv: int | None,
    gamma_arpack_maxiter: int | None,
    forcing_mode: str,
    startup_mode: str,
    discrete_forcing_compensate_hyperviscosity: bool,
    skip_warmup: bool,
    setup_device: str,
    gmres_tol: float,
    gmres_atol: float,
    gmres_restart: int,
    gmres_maxiter: int | None,
    ilu_application: str,
    linear_solve_diagnostics: bool,
    linear_solve_diagnostics_threshold: float,
    force_boundary_paired_ghost_stencil: bool,
    rollout_diagnostics_every: int,
    rollout_diagnostics_growth_limit: float,
    rollout_near_boundary_factor: float,
) -> ForwardResult:
    del xi_sl  # Kept as a CLI compatibility option for older PU-SL runs.
    x_phys = jnp.asarray(domain.get_int_bdry_nodes(), dtype=float)
    x_all = jnp.asarray(domain.get_all_nodes(), dtype=float)
    if fixed_steps is None:
        dt, steps = timestep_from_h(metadata.h, final_time, dt_factor, min_steps)
    else:
        if fixed_steps <= 0:
            raise ValueError("fixed steps must be positive")
        steps = int(fixed_steps)
        dt = float(final_time) / float(steps)

    case_label = (
        f"geometry={metadata.geometry}, resolution={metadata.label}, order={order}, "
        f"h={metadata.h:.5f}, dt={dt:.6e}, steps={steps}, physical_nodes={int(domain.get_num_int_bdry_nodes())}"
    )
    progress(f"case start: {case_label}")
    progress(f"building velocity/exact state: {case_label}")
    velocity = incompressible_boundary_vanishing_velocity(domain, velocity_amplitude)
    initial_state = exact_state(0.0, x_all)
    exact_final = exact_state(final_time, x_phys)

    if skip_warmup:
        preprocess_elapsed_seconds = 0.0
        progress(f"warmup skipped: {case_label}")
    else:
        preprocess_start_time = perf_counter()
        progress(f"warmup solver/gamma setup start: {case_label}")
        warmup_solver, warmup_gamma, _warmup_gamma_status = build_solver_template(
            domain,
            order=order,
            dt=dt,
            nu=nu,
            stencil=stencil,
            linear_solver=linear_solver,
            velocity_amplitude=velocity_amplitude,
            gamma_mode=gamma_mode,
            fixed_gamma=fixed_gamma,
            gamma_scale=gamma_scale,
            gamma_h_power=gamma_h_power,
            hyperviscosity_power=hyperviscosity_power,
            boundary_order=boundary_order,
            hyperviscosity_row_mask=hyperviscosity_row_mask,
            hyperviscosity_mask_distance_factor=hyperviscosity_mask_distance_factor,
            sparse_assembly_chunk_size=sparse_assembly_chunk_size,
            gamma_arpack_ncv=gamma_arpack_ncv,
            gamma_arpack_maxiter=gamma_arpack_maxiter,
            setup_device=setup_device,
            gmres_tol=gmres_tol,
            gmres_atol=gmres_atol,
            gmres_restart=gmres_restart,
            gmres_maxiter=gmres_maxiter,
            ilu_application=ilu_application,
            linear_solve_diagnostics=linear_solve_diagnostics,
            linear_solve_diagnostics_threshold=linear_solve_diagnostics_threshold,
            force_boundary_paired_ghost_stencil=force_boundary_paired_ghost_stencil,
        )
        progress(
            f"warmup solver/gamma setup done: {case_label}, "
            f"gamma_status={_warmup_gamma_status}, k={int(warmup_solver.hyperviscosity_power)}"
        )
        progress(f"warmup operator sync start: {case_label}")
        synchronize_solver_operators(warmup_solver)
        progress(f"warmup operator sync done: {case_label}")
        progress(f"warmup forcing build start: {case_label}, forcing_mode={forcing_mode}")
        warmup_forcing = build_forcing_for_mode(
            forcing_mode,
            warmup_solver,
            warmup_gamma,
            nu=nu,
            velocity=velocity,
            rho=rho,
            steps=steps,
            dt=dt,
            compensate_hyperviscosity=discrete_forcing_compensate_hyperviscosity,
        )
        progress(f"warmup forcing build done: {case_label}")
        progress(f"warmup rollout start: {case_label}")
        warmup_state = rollout_final_state(
            warmup_solver,
            warmup_gamma,
            initial_state=initial_state,
            velocity=velocity,
            forcing=warmup_forcing,
            steps=steps,
            rho=rho,
            startup_mode=startup_mode,
            diagnostics_every=0,
            diagnostics_growth_limit=rollout_diagnostics_growth_limit,
            diagnostics_near_boundary_factor=rollout_near_boundary_factor,
            case_label=case_label,
        )
        jax.block_until_ready(warmup_state)
        preprocess_elapsed_seconds = perf_counter() - preprocess_start_time
        progress(f"warmup rollout done: {case_label}, preprocess_seconds={preprocess_elapsed_seconds:.6f}")

    start_time = perf_counter()
    progress(f"timed solver/gamma setup start: {case_label}")
    solver, gamma, gamma_status = build_solver_template(
        domain,
        order=order,
        dt=dt,
        nu=nu,
        stencil=stencil,
        linear_solver=linear_solver,
        velocity_amplitude=velocity_amplitude,
        gamma_mode=gamma_mode,
        fixed_gamma=fixed_gamma,
        gamma_scale=gamma_scale,
        gamma_h_power=gamma_h_power,
        hyperviscosity_power=hyperviscosity_power,
        boundary_order=boundary_order,
        hyperviscosity_row_mask=hyperviscosity_row_mask,
        hyperviscosity_mask_distance_factor=hyperviscosity_mask_distance_factor,
        sparse_assembly_chunk_size=sparse_assembly_chunk_size,
        gamma_arpack_ncv=gamma_arpack_ncv,
        gamma_arpack_maxiter=gamma_arpack_maxiter,
        setup_device=setup_device,
        gmres_tol=gmres_tol,
        gmres_atol=gmres_atol,
        gmres_restart=gmres_restart,
        gmres_maxiter=gmres_maxiter,
        ilu_application=ilu_application,
        linear_solve_diagnostics=linear_solve_diagnostics,
        linear_solve_diagnostics_threshold=linear_solve_diagnostics_threshold,
        force_boundary_paired_ghost_stencil=force_boundary_paired_ghost_stencil,
    )
    progress(f"timed solver/gamma setup done: {case_label}, gamma_status={gamma_status}, k={int(solver.hyperviscosity_power)}")
    progress(f"timed operator sync start: {case_label}")
    synchronize_solver_operators(solver)
    progress(f"timed operator sync done: {case_label}")
    progress(f"timed forcing build start: {case_label}, forcing_mode={forcing_mode}")
    forcing = build_forcing_for_mode(
        forcing_mode,
        solver,
        gamma,
        nu=nu,
        velocity=velocity,
        rho=rho,
        steps=steps,
        dt=dt,
        compensate_hyperviscosity=discrete_forcing_compensate_hyperviscosity,
    )
    progress(f"timed forcing build done: {case_label}")
    progress(f"timed rollout start: {case_label}")
    final_state = rollout_final_state(
        solver,
        gamma,
        initial_state=initial_state,
        velocity=velocity,
        forcing=forcing,
        steps=steps,
        rho=rho,
        startup_mode=startup_mode,
        diagnostics_every=rollout_diagnostics_every,
        diagnostics_growth_limit=rollout_diagnostics_growth_limit,
        diagnostics_near_boundary_factor=rollout_near_boundary_factor,
        case_label=case_label,
    )
    error = relative_l2(final_state, exact_final)
    progress(f"timed rollout submitted, waiting for readiness: {case_label}")
    jax.block_until_ready(error)
    elapsed_seconds = perf_counter() - start_time
    error_float = float(error)
    progress(f"case done: {case_label}, error={error_float:.6e}, seconds={elapsed_seconds:.6f}")
    return ForwardResult(
        order=order,
        resolution=metadata.label,
        h=metadata.h,
        dt=dt,
        steps=steps,
        final_time=final_time,
        node_count=int(domain.get_num_int_bdry_nodes()),
        final_relative_l2_error=error_float,
        mean_relative_l2_error=error_float,
        max_relative_l2_error=error_float,
        elapsed_seconds=elapsed_seconds,
        preprocess_elapsed_seconds=preprocess_elapsed_seconds,
        gamma_status=gamma_status,
        hyperviscosity_power=int(solver.hyperviscosity_power),
    )


def failed_result(
    *,
    domain: DomainDescriptor,
    metadata: SavedDomainMetadata,
    order: int,
    final_time: float,
    dt_factor: float,
    min_steps: int,
    fixed_steps: int | None,
) -> ForwardResult:
    if fixed_steps is None:
        dt, steps = timestep_from_h(metadata.h, final_time, dt_factor, min_steps)
    else:
        steps = int(fixed_steps)
        dt = float(final_time) / float(steps)
    return ForwardResult(order, metadata.label, metadata.h, dt, steps, final_time, int(domain.get_num_int_bdry_nodes()), -1.0, -1.0, -1.0, -1.0, -1.0, "failed", -1)


def export_forward_mat(
    results: list[ForwardResult],
    *,
    dim: int,
    geometry: str,
    stencil: str,
    linear_solver: str,
    final_time: float,
    dt_factor: float,
    min_steps: int,
    nu: float,
    velocity_amplitude: float,
    rho: float,
    gamma_mode: str,
    fixed_gamma: float,
    gamma_scale: float,
    gamma_h_power: float,
    boundary_order: int | None,
    hyperviscosity_row_mask: str,
    hyperviscosity_mask_distance_factor: float,
    setup_device: str,
    gmres_tol: float,
    gmres_atol: float,
    gmres_restart: int,
    gmres_maxiter: int | None,
    ilu_application: str,
    linear_solve_diagnostics: bool,
    linear_solve_diagnostics_threshold: float,
    force_boundary_paired_ghost_stencil: bool,
    rollout_diagnostics_every: int,
    rollout_diagnostics_growth_limit: float,
    rollout_near_boundary_factor: float,
    forcing_mode: str,
    startup_mode: str,
    discrete_forcing_compensate_hyperviscosity: bool,
    output_path: Path,
) -> None:
    orders = sorted({row.order for row in results})
    labels = []
    seen = set()
    for row in results:
        if row.resolution not in seen:
            labels.append(row.resolution)
            seen.add(row.resolution)
    order_to_idx = {order: i for i, order in enumerate(orders)}
    label_to_idx = {label: i for i, label in enumerate(labels)}
    shape = (len(orders), len(labels))
    relative_l2_errors = np.full(shape, np.nan, dtype=float)
    mean_relative_l2_errors = np.full(shape, np.nan, dtype=float)
    max_relative_l2_errors = np.full(shape, np.nan, dtype=float)
    elapsed_seconds = np.full(shape, np.nan, dtype=float)
    preprocess_elapsed_seconds = np.full(shape, np.nan, dtype=float)
    node_counts = np.full(shape, np.nan, dtype=float)
    dt_values = np.full(shape, np.nan, dtype=float)
    steps_values = np.full(shape, np.nan, dtype=float)
    gamma_status = np.empty(shape, dtype=object)
    gamma_status[:, :] = ""
    hyperviscosity_powers = np.full(shape, np.nan, dtype=float)
    hs = np.full((len(labels),), np.nan, dtype=float)
    for row in results:
        i = order_to_idx[row.order]
        j = label_to_idx[row.resolution]
        relative_l2_errors[i, j] = row.final_relative_l2_error
        mean_relative_l2_errors[i, j] = row.mean_relative_l2_error
        max_relative_l2_errors[i, j] = row.max_relative_l2_error
        elapsed_seconds[i, j] = row.elapsed_seconds
        preprocess_elapsed_seconds[i, j] = row.preprocess_elapsed_seconds
        node_counts[i, j] = row.node_count
        dt_values[i, j] = row.dt
        steps_values[i, j] = row.steps
        gamma_status[i, j] = row.gamma_status
        hyperviscosity_powers[i, j] = row.hyperviscosity_power
        hs[j] = row.h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        output_path,
        {
            "relative_l2_errors": relative_l2_errors,
            "mean_relative_l2_errors": mean_relative_l2_errors,
            "max_relative_l2_errors": max_relative_l2_errors,
            "elapsed_seconds": elapsed_seconds,
            "preprocess_elapsed_seconds": preprocess_elapsed_seconds,
            "orders": np.asarray(orders, dtype=float),
            "hs": hs,
            "point_cloud_numbers": np.arange(1, len(labels) + 1, dtype=float),
            "labels": np.asarray(labels, dtype=object),
            "node_counts": node_counts,
            "dt_values": dt_values,
            "steps_values": steps_values,
            "gamma_status": gamma_status,
            "hyperviscosity_powers": hyperviscosity_powers,
            "geometry": np.asarray([geometry], dtype=object),
            "dimension": np.asarray([[dim]], dtype=float),
            "stencil": np.asarray([stencil], dtype=object),
            "linear_solver": np.asarray([linear_solver], dtype=object),
            "final_time": np.asarray([[final_time]], dtype=float),
            "dt_factor": np.asarray([[dt_factor]], dtype=float),
            "min_steps": np.asarray([[min_steps]], dtype=float),
            "nu": np.asarray([[nu]], dtype=float),
            "velocity_amplitude": np.asarray([[velocity_amplitude]], dtype=float),
            "rho": np.asarray([[rho]], dtype=float),
            "gamma_mode": np.asarray([gamma_mode], dtype=object),
            "fixed_gamma": np.asarray([[fixed_gamma]], dtype=float),
            "gamma_scale": np.asarray([[gamma_scale]], dtype=float),
            "gamma_h_power": np.asarray([[gamma_h_power]], dtype=float),
            "boundary_order": np.asarray([[-1 if boundary_order is None else boundary_order]], dtype=float),
            "hyperviscosity_row_mask": np.asarray([hyperviscosity_row_mask], dtype=object),
            "hyperviscosity_mask_distance_factor": np.asarray([[hyperviscosity_mask_distance_factor]], dtype=float),
            "setup_device": np.asarray([setup_device], dtype=object),
            "gmres_tol": np.asarray([[gmres_tol]], dtype=float),
            "gmres_atol": np.asarray([[gmres_atol]], dtype=float),
            "gmres_restart": np.asarray([[gmres_restart]], dtype=float),
            "gmres_maxiter": np.asarray([[-1 if gmres_maxiter is None else gmres_maxiter]], dtype=float),
            "ilu_application": np.asarray([ilu_application], dtype=object),
            "linear_solve_diagnostics": np.asarray([[float(linear_solve_diagnostics)]], dtype=float),
            "linear_solve_diagnostics_threshold": np.asarray([[linear_solve_diagnostics_threshold]], dtype=float),
            "force_boundary_paired_ghost_stencil": np.asarray([[float(force_boundary_paired_ghost_stencil)]], dtype=float),
            "rollout_diagnostics_every": np.asarray([[rollout_diagnostics_every]], dtype=float),
            "rollout_diagnostics_growth_limit": np.asarray([[rollout_diagnostics_growth_limit]], dtype=float),
            "rollout_near_boundary_factor": np.asarray([[rollout_near_boundary_factor]], dtype=float),
            "forcing_mode": np.asarray([forcing_mode], dtype=object),
            "startup_mode": np.asarray([startup_mode], dtype=object),
            "discrete_forcing_compensate_hyperviscosity": np.asarray([[float(discrete_forcing_compensate_hyperviscosity)]], dtype=float),
            "family": np.asarray(["manufactured_eulerian_adr"], dtype=object),
            "solver_framework": np.asarray(["EulerianAdvectionDiffusionReactionSolver"], dtype=object),
            "timestep_rule": np.asarray(["dt = final_time / max(min_steps, ceil(final_time / (dt_factor * h)))"], dtype=object),
            "equation": np.asarray(["u_t + v dot grad u = nu lap u + rho u(1-u) + f"], dtype=object),
            "error_metric": np.asarray(["final_relative_l2"], dtype=object),
        },
    )


def render_markdown(
    results: list[ForwardResult],
    *,
    dim: int,
    geometry: str,
    stencil: str,
    linear_solver: str,
    final_time: float,
    dt_factor: float,
    min_steps: int,
    fixed_steps: int | None,
    nu: float,
    velocity_amplitude: float,
    rho: float,
    gamma_mode: str,
    fixed_gamma: float,
    gamma_scale: float,
    gamma_h_power: float,
    boundary_order: int | None,
    hyperviscosity_row_mask: str,
    hyperviscosity_mask_distance_factor: float,
    setup_device: str,
    gmres_tol: float,
    gmres_atol: float,
    gmres_restart: int,
    gmres_maxiter: int | None,
    ilu_application: str,
    linear_solve_diagnostics: bool,
    linear_solve_diagnostics_threshold: float,
    force_boundary_paired_ghost_stencil: bool,
    rollout_diagnostics_every: int,
    rollout_diagnostics_growth_limit: float,
    rollout_near_boundary_factor: float,
    forcing_mode: str,
    startup_mode: str,
    discrete_forcing_compensate_hyperviscosity: bool,
) -> str:
    lines = [
        f"# Advection-Diffusion Forward Convergence Study ({stencil.upper()})",
        "",
        f"- dimension: {dim}",
        f"- geometry: {geometry}",
        f"- stencil: {stencil}",
        f"- linear solver: {linear_solver}",
        f"- final time: {final_time}",
        f"- nu: {nu}",
        f"- velocity amplitude: {velocity_amplitude}",
        f"- reaction rho: {rho}",
        f"- hyperviscosity gamma mode: {gamma_mode}",
        f"- fixed gamma fallback: {fixed_gamma}",
        f"- hyperviscosity gamma scale: {gamma_scale}",
        f"- hyperviscosity gamma h power: {gamma_h_power}",
        f"- boundary closure order: {'same as PDE order' if boundary_order is None else boundary_order}",
        f"- hyperviscosity row mask: {hyperviscosity_row_mask}",
        f"- hyperviscosity mask distance factor: {hyperviscosity_mask_distance_factor}",
        f"- setup device: {setup_device}",
        f"- GMRES tol: {gmres_tol}",
        f"- GMRES atol: {gmres_atol}",
        f"- GMRES restart: {gmres_restart}",
        f"- GMRES maxiter: {'default' if gmres_maxiter is None else gmres_maxiter}",
        f"- ILU application: {ilu_application}",
        f"- linear solve diagnostics: {linear_solve_diagnostics}",
        f"- linear solve diagnostics threshold: {linear_solve_diagnostics_threshold}",
        f"- paired boundary ghost stencil: {force_boundary_paired_ghost_stencil}",
        f"- rollout diagnostics every: {rollout_diagnostics_every}",
        f"- rollout diagnostics growth limit: {rollout_diagnostics_growth_limit}",
        f"- rollout near-boundary factor: {rollout_near_boundary_factor}",
        f"- forcing mode: {forcing_mode}",
        f"- startup mode: {startup_mode}",
        f"- discrete forcing compensates hyperviscosity: {discrete_forcing_compensate_hyperviscosity}",
        f"- timestep rule: {'fixed steps' if fixed_steps is not None else 'h-dependent'}",
        f"- dt factor: {dt_factor}",
        f"- minimum steps: {min_steps}",
        f"- JAX backend: `{jax.default_backend()}`",
        f"- timing: `preprocess seconds is first warmup/compile pass; seconds is second pass after warmup`",
        "",
        "| xi | resolution | h | dt | steps | physical nodes | hyperviscosity | k | final relative L2 | mean relative L2 | max relative L2 | preprocess seconds | seconds |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {row.order} | {row.resolution} | {row.h:.5f} | {row.dt:.6e} | {row.steps} | {row.node_count} | "
            f"{row.gamma_status} | {row.hyperviscosity_power} | "
            f"{row.final_relative_l2_error:.6e} | {row.mean_relative_l2_error:.6e} | "
            f"{row.max_relative_l2_error:.6e} | {row.preprocess_elapsed_seconds:.6f} | {row.elapsed_seconds:.6f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advection-diffusion forward convergence study with a manufactured transient.")
    parser.add_argument("--dim", type=int, default=2, choices=(2, 3))
    parser.add_argument("--geometry", type=str, default=None)
    parser.add_argument("--orders", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--hs", type=float, nargs="+", default=[0.45, 0.35, 0.25, 0.15])
    parser.add_argument("--domain-files", type=Path, nargs="+", default=None)
    parser.add_argument("--stencil", type=str, default="rbf", choices=("wls", "rbf"))
    parser.add_argument("--linear-solver", type=str, default="lineax", choices=("gmres", "gmres_ilu", "dense", "lineax", "cudss"))
    parser.add_argument("--xi-sl", type=int, default=4, help="Legacy PU-SL compatibility option; ignored by the Eulerian ADR path.")
    parser.add_argument("--nu", type=float, default=0.05)
    parser.add_argument("--velocity-amplitude", type=float, default=0.12)
    parser.add_argument("--rho", type=float, default=0.0, help="Logistic reaction strength. Use 0 for pure advection-diffusion.")
    parser.add_argument(
        "--gamma-mode",
        choices=("estimate", "estimate_or_fixed", "fixed"),
        default="estimate_or_fixed",
        help="How to choose Eulerian ADR hyperviscosity. estimate_or_fixed falls back if ARPACK does not converge.",
    )
    parser.add_argument("--fixed-gamma", type=float, default=-1.0e-4, help="Component value used by fixed gamma mode/fallback.")
    parser.add_argument(
        "--gamma-scale",
        type=float,
        default=1.0,
        help="Extra multiplicative scale applied after estimating or choosing fixed hyperviscosity gamma.",
    )
    parser.add_argument(
        "--gamma-h-power",
        type=float,
        default=0.0,
        help="Additional h-dependent gamma scaling: final gamma is gamma * gamma_scale * sep_rad**gamma_h_power.",
    )
    parser.add_argument(
        "--hyperviscosity-power",
        type=str,
        default="2",
        help="Laplacian power k in the hyperviscosity operator. Use a positive integer or 'auto' for k=ceil((xi+max(q))/2).",
    )
    parser.add_argument(
        "--boundary-order",
        type=int,
        default=None,
        help="Experimental: use this convergence order only for boundary/ghost closure stencils. Omit to match --orders.",
    )
    parser.add_argument(
        "--hyperviscosity-row-mask",
        choices=("none", "boundary", "near_boundary"),
        default="none",
        help="Optionally zero hyperviscosity rows on boundary or near-boundary physical nodes.",
    )
    parser.add_argument(
        "--hyperviscosity-mask-distance-factor",
        type=float,
        default=2.0,
        help="Near-boundary row mask distance in separation-radius units.",
    )
    parser.add_argument(
        "--sparse-assembly-chunk-size",
        type=int,
        default=None,
        help="Assemble sparse RBF-FD operators in row chunks of this size to reduce peak memory.",
    )
    parser.add_argument("--gamma-arpack-ncv", type=int, default=None, help="Optional ARPACK ncv for paper hyperviscosity eigenvalue estimates.")
    parser.add_argument("--gamma-arpack-maxiter", type=int, default=None, help="Optional ARPACK maxiter for paper hyperviscosity eigenvalue estimates.")
    parser.add_argument("--gmres-tol", type=float, default=1e-8, help="Relative tolerance for sparse JAX GMRES solves.")
    parser.add_argument("--gmres-atol", type=float, default=1e-10, help="Absolute tolerance for sparse JAX GMRES solves.")
    parser.add_argument("--gmres-restart", type=int, default=40, help="Restart length for sparse JAX GMRES solves.")
    parser.add_argument("--gmres-maxiter", type=int, default=None, help="Maximum sparse JAX GMRES iterations; omit for JAX default.")
    parser.add_argument(
        "--ilu-application",
        choices=("dense", "host_callback"),
        default="dense",
        help="How gmres_ilu applies the frozen ILU preconditioner. host_callback avoids dense N-by-N preconditioner storage.",
    )
    parser.add_argument("--linear-solve-diagnostics", action="store_true", help="Print sparse linear solve residual diagnostics at runtime.")
    parser.add_argument(
        "--linear-solve-diagnostics-threshold",
        type=float,
        default=1e-6,
        help="Only print sparse linear diagnostics when relative residual exceeds this value. Use 0 to print every solve.",
    )
    parser.add_argument(
        "--force-boundary-paired-ghost-stencil",
        action="store_true",
        help="Experimental: force each boundary closure stencil to include its paired ghost node.",
    )
    parser.add_argument(
        "--rollout-diagnostics-every",
        type=int,
        default=0,
        help="Print rollout component diagnostics every N timesteps during the timed exact-history rollout. Use 0 to disable.",
    )
    parser.add_argument(
        "--rollout-diagnostics-growth-limit",
        type=float,
        default=float("inf"),
        help="Abort an instrumented rollout when checkpoint relative L2 exceeds this value.",
    )
    parser.add_argument(
        "--rollout-near-boundary-factor",
        type=float,
        default=2.0,
        help="Distance, in separation-radius units, used for near-boundary split rollout diagnostics.",
    )
    parser.add_argument("--final-time", type=float, default=0.5)
    parser.add_argument("--dt-factor", type=float, default=0.25)
    parser.add_argument("--min-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=None, help="Legacy fixed-step mode. If provided, overrides h-dependent timestep selection.")
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip the warmup/compile pass. Timed seconds will include first-run compilation/execution cost.",
    )
    parser.add_argument(
        "--setup-device",
        choices=("default", "cpu"),
        default="default",
        help="Device used for solver/operator/gamma setup. Use cpu to avoid large GPU assembly temporaries on big domains.",
    )
    parser.add_argument(
        "--forcing-mode",
        type=str,
        choices=("continuous", "discrete_bdf"),
        default="continuous",
        help="Use continuous manufactured forcing, or discrete-BDF/AB forcing that removes time truncation from the manufactured test.",
    )
    parser.add_argument(
        "--startup-mode",
        type=str,
        choices=("stepped", "exact_history"),
        default="stepped",
        help="Use normal BDF1/BDF2 startup, or seed BDF3 with exact history to remove startup error in manufactured tests.",
    )
    parser.add_argument(
        "--discrete-forcing-compensate-hyperviscosity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For discrete_bdf forcing, cancel the artificial hyperviscosity term against the exact state.",
    )
    parser.add_argument("--failure-tracebacks", action="store_true")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--mapped-cloud-preferred",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the mapped closed-curve node generator for smooth 2D geometries.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional markdown output path. Omit to write only .mat output.")
    parser.add_argument("--mat-output", type=Path, default=DEFAULT_MAT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    geometry_name = args.geometry or default_geometry(args.dim)
    if args.domain_files:
        domain_entries = [load_domain_file(path) for path in args.domain_files]
    else:
        validate_geometry(args.dim, geometry_name)
        domain_entries = []
        for h in args.hs:
            domain = build_forward_domain(
                args.dim,
                geometry_name,
                h,
                args.seed,
                mapped_cloud_preferred=args.mapped_cloud_preferred,
            )
            metadata = SavedDomainMetadata(
                dim=args.dim,
                geometry=geometry_name,
                h=float(h),
                seed=args.seed,
                sep_rad=float(domain.get_sep_rad()),
                label=f"{geometry_name}_d{args.dim}_h{h:.5f}_seed{args.seed}",
            )
            domain_entries.append((domain, metadata))

    results: list[ForwardResult] = []
    for domain, metadata in domain_entries:
        for order in args.orders:
            try:
                results.append(
                    run_case(
                        domain=domain,
                        metadata=metadata,
                        order=order,
                        xi_sl=args.xi_sl,
                        stencil=args.stencil,
                        linear_solver=args.linear_solver,
                        final_time=args.final_time,
                        dt_factor=args.dt_factor,
                        min_steps=args.min_steps,
                        fixed_steps=args.steps,
                        nu=args.nu,
                        velocity_amplitude=args.velocity_amplitude,
                        rho=args.rho,
                        gamma_mode=args.gamma_mode,
                        fixed_gamma=args.fixed_gamma,
                        gamma_scale=args.gamma_scale,
                        gamma_h_power=args.gamma_h_power,
                        hyperviscosity_power=args.hyperviscosity_power,
                        boundary_order=args.boundary_order,
                        hyperviscosity_row_mask=args.hyperviscosity_row_mask,
                        hyperviscosity_mask_distance_factor=args.hyperviscosity_mask_distance_factor,
                        sparse_assembly_chunk_size=args.sparse_assembly_chunk_size,
                        gamma_arpack_ncv=args.gamma_arpack_ncv,
                        gamma_arpack_maxiter=args.gamma_arpack_maxiter,
                        forcing_mode=args.forcing_mode,
                        startup_mode=args.startup_mode,
                        discrete_forcing_compensate_hyperviscosity=args.discrete_forcing_compensate_hyperviscosity,
                        skip_warmup=args.skip_warmup,
                        setup_device=args.setup_device,
                        gmres_tol=args.gmres_tol,
                        gmres_atol=args.gmres_atol,
                        gmres_restart=args.gmres_restart,
                        gmres_maxiter=args.gmres_maxiter,
                        ilu_application=args.ilu_application,
                        linear_solve_diagnostics=args.linear_solve_diagnostics,
                        linear_solve_diagnostics_threshold=args.linear_solve_diagnostics_threshold,
                        force_boundary_paired_ghost_stencil=args.force_boundary_paired_ghost_stencil,
                        rollout_diagnostics_every=args.rollout_diagnostics_every,
                        rollout_diagnostics_growth_limit=args.rollout_diagnostics_growth_limit,
                        rollout_near_boundary_factor=args.rollout_near_boundary_factor,
                    )
                )
            except Exception as exc:
                exc_lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
                exc_summary = exc_lines[0] if exc_lines else str(exc)
                print(
                    "[failed] "
                    f"geometry={metadata.geometry}, resolution={metadata.label}, order={order}, "
                    f"h={metadata.h:.5f}: {type(exc).__name__}: {exc_summary}",
                    flush=True,
                )
                if args.failure_tracebacks:
                    traceback.print_exc()
                results.append(
                    failed_result(
                        domain=domain,
                        metadata=metadata,
                        order=order,
                        final_time=args.final_time,
                        dt_factor=args.dt_factor,
                        min_steps=args.min_steps,
                        fixed_steps=args.steps,
                    )
                )

    dim = domain_entries[0][1].dim if domain_entries else args.dim
    geometry = domain_entries[0][1].geometry if domain_entries else geometry_name
    md = render_markdown(
        results,
        dim=dim,
        geometry=geometry,
        stencil=args.stencil,
        linear_solver=args.linear_solver,
        final_time=args.final_time,
        dt_factor=args.dt_factor,
        min_steps=args.min_steps,
        fixed_steps=args.steps,
        nu=args.nu,
        velocity_amplitude=args.velocity_amplitude,
        rho=args.rho,
        gamma_mode=args.gamma_mode,
        fixed_gamma=args.fixed_gamma,
        gamma_scale=args.gamma_scale,
        gamma_h_power=args.gamma_h_power,
        boundary_order=args.boundary_order,
        hyperviscosity_row_mask=args.hyperviscosity_row_mask,
        hyperviscosity_mask_distance_factor=args.hyperviscosity_mask_distance_factor,
        setup_device=args.setup_device,
        gmres_tol=args.gmres_tol,
        gmres_atol=args.gmres_atol,
        gmres_restart=args.gmres_restart,
        gmres_maxiter=args.gmres_maxiter,
        ilu_application=args.ilu_application,
        linear_solve_diagnostics=args.linear_solve_diagnostics,
        linear_solve_diagnostics_threshold=args.linear_solve_diagnostics_threshold,
        force_boundary_paired_ghost_stencil=args.force_boundary_paired_ghost_stencil,
        rollout_diagnostics_every=args.rollout_diagnostics_every,
        rollout_diagnostics_growth_limit=args.rollout_diagnostics_growth_limit,
        rollout_near_boundary_factor=args.rollout_near_boundary_factor,
        forcing_mode=args.forcing_mode,
        startup_mode=args.startup_mode,
        discrete_forcing_compensate_hyperviscosity=args.discrete_forcing_compensate_hyperviscosity,
    )
    print(md)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md, encoding="utf-8")
    export_forward_mat(
        results,
        dim=dim,
        geometry=geometry,
        stencil=args.stencil,
        linear_solver=args.linear_solver,
        final_time=args.final_time,
        dt_factor=args.dt_factor,
        min_steps=args.min_steps,
        nu=args.nu,
        velocity_amplitude=args.velocity_amplitude,
        rho=args.rho,
        gamma_mode=args.gamma_mode,
        fixed_gamma=args.fixed_gamma,
        gamma_scale=args.gamma_scale,
        gamma_h_power=args.gamma_h_power,
        boundary_order=args.boundary_order,
        hyperviscosity_row_mask=args.hyperviscosity_row_mask,
        hyperviscosity_mask_distance_factor=args.hyperviscosity_mask_distance_factor,
        setup_device=args.setup_device,
        gmres_tol=args.gmres_tol,
        gmres_atol=args.gmres_atol,
        gmres_restart=args.gmres_restart,
        gmres_maxiter=args.gmres_maxiter,
        ilu_application=args.ilu_application,
        linear_solve_diagnostics=args.linear_solve_diagnostics,
        linear_solve_diagnostics_threshold=args.linear_solve_diagnostics_threshold,
        force_boundary_paired_ghost_stencil=args.force_boundary_paired_ghost_stencil,
        rollout_diagnostics_every=args.rollout_diagnostics_every,
        rollout_diagnostics_growth_limit=args.rollout_diagnostics_growth_limit,
        rollout_near_boundary_factor=args.rollout_near_boundary_factor,
        forcing_mode=args.forcing_mode,
        startup_mode=args.startup_mode,
        discrete_forcing_compensate_hyperviscosity=args.discrete_forcing_compensate_hyperviscosity,
        output_path=args.mat_output,
    )
    print(f"wrote matlab output to {args.mat_output}")


if __name__ == "__main__":
    main()
