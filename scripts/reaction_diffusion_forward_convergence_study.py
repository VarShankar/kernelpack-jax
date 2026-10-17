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
from scipy.io import savemat
from scipy.spatial import cKDTree

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

INV = Path(__file__).resolve().parents[1] / "inverse problems"
if str(INV) not in sys.path:
    sys.path.insert(0, str(INV))

from kernelpack.domain import DomainDescriptor
from kernelpack.nodes import DomainNodeGenerator
from kernelpack.solvers import ReactionDiffusionSolver
from kernelpack.solvers._common import build_implicit_rhs, build_implicit_system

from common import build_domain, default_h, relative_l2
from common import _build_surface_2d
from reaction_diffusion_inverse_convergence_study import (
    FAMILIES,
    SavedDomainMetadata,
    build_solver_template,
    default_geometry,
    exact_state,
    exact_state_laplacian,
    make_forcing,
    reaction_family_value,
    build_reaction_term,
    rollout_states,
    sample_theta,
    solver_settings,
    validate_geometry,
)
from h_channel_visibility_stencils import build_h_channel_visibility_solver_template

DEFAULT_OUTPUT = Path(__file__).with_name("reaction_diffusion_forward_convergence_study.md")
DEFAULT_MAT_OUTPUT = Path(__file__).with_name("reaction_diffusion_forward_convergence_study-egg-affine.mat")


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
    interior_node_count: int
    interior_distance_factor: float
    interior_final_relative_l2_error: float
    interior_mean_relative_l2_error: float
    interior_max_relative_l2_error: float
    elapsed_seconds: float
    preprocess_elapsed_seconds: float


def load_domain_file(path: Path) -> tuple[DomainDescriptor, SavedDomainMetadata]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        descriptor = DomainDescriptor()
        descriptor.set_nodes(
            jnp.asarray(data["int_nodes"], dtype=float),
            jnp.asarray(data["bdry_nodes"], dtype=float),
            jnp.asarray(data["ghost_nodes"], dtype=float),
        )
        descriptor.set_normals(jnp.asarray(data["normals"], dtype=float))
        descriptor.set_sep_rad(float(np.asarray(data["sep_rad"]).item()))
        descriptor.build_structs()
        metadata = SavedDomainMetadata(
            dim=int(np.asarray(data["dim"]).item()),
            geometry=str(np.asarray(data["geometry"]).item()),
            h=float(np.asarray(data["h"]).item()),
            seed=int(np.asarray(data["seed"]).item()),
            sep_rad=float(np.asarray(data["sep_rad"]).item()),
            label=str(np.asarray(data["label"]).item()),
        )
    return descriptor, metadata


def setup_device_context(setup_device: str):
    if setup_device == "cpu":
        cpu_devices = jax.devices("cpu")
        if not cpu_devices:
            raise RuntimeError("requested CPU setup device, but JAX did not report a CPU device")
        return jax.default_device(cpu_devices[0])
    if setup_device == "default":
        return nullcontext()
    raise ValueError(f"unknown setup device {setup_device}")


def copy_domain_to_device(domain: DomainDescriptor, device) -> DomainDescriptor:
    copied = DomainDescriptor()
    copied.set_nodes(
        jax.device_put(domain.get_interior_nodes(), device),
        jax.device_put(domain.get_bdry_nodes(), device),
        jax.device_put(domain.get_ghost_nodes(), device),
    )
    copied.set_normals(jax.device_put(domain.get_nrmls(), device))
    copied.set_sep_rad(float(domain.get_sep_rad()))
    copied.set_outer_level_set(domain.get_outer_level_set())
    copied.set_boundary_level_sets(domain.get_boundary_level_sets())
    copied.build_structs()
    return copied


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


def synchronize_solver_operators(solver: ReactionDiffusionSolver) -> None:
    """Block on assembled dense/sparse operators so timing includes assembly work."""

    jax.block_until_ready(solver.lap)
    if solver.lap_sparse is not None:
        jax.block_until_ready(solver.lap_sparse.indices)
        jax.block_until_ready(solver.lap_sparse.values)
    if solver.bc.size:
        jax.block_until_ready(solver.bc)
    if solver.bc_sparse is not None:
        jax.block_until_ready(solver.bc_sparse.indices)
        jax.block_until_ready(solver.bc_sparse.values)


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


def finite_summary(name: str, value: jnp.ndarray) -> str:
    arr = np.asarray(jax.device_get(jnp.asarray(value)))
    finite = np.isfinite(arr)
    if arr.size == 0:
        return f"{name}: empty"
    if finite.any():
        finite_values = arr[finite]
        detail = f"min={finite_values.min():.6e}, max={finite_values.max():.6e}"
    else:
        detail = "no finite entries"
    return (
        f"{name}: shape={arr.shape}, finite={int(finite.sum())}/{arr.size}, "
        f"nan={int(np.isnan(arr).sum())}, inf={int(np.isinf(arr).sum())}, {detail}"
    )


def assert_finite(name: str, value: jnp.ndarray, context: str) -> None:
    arr = np.asarray(jax.device_get(jnp.asarray(value)))
    if not np.isfinite(arr).all():
        raise FloatingPointError(f"{context}\n{finite_summary(name, arr)}")


def interior_mask_from_boundary(domain: DomainDescriptor, h: float, distance_factor: float) -> jnp.ndarray:
    x_phys = np.asarray(jax.device_get(domain.get_int_bdry_nodes()), dtype=float)
    xb = np.asarray(jax.device_get(domain.get_bdry_nodes()), dtype=float)
    if distance_factor <= 0.0:
        return jnp.ones((x_phys.shape[0],), dtype=bool)
    if xb.size == 0:
        return jnp.ones((x_phys.shape[0],), dtype=bool)
    distances, _ = cKDTree(xb).query(x_phys, k=1)
    return jnp.asarray(distances >= float(distance_factor) * float(h), dtype=bool)


def masked_relative_l2(pred: jnp.ndarray, target: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    mask = jnp.asarray(mask, dtype=bool)
    pred_masked = pred[mask]
    target_masked = target[mask]
    if pred_masked.size == 0:
        return jnp.asarray(jnp.nan, dtype=float)
    numerator = jnp.linalg.norm(pred_masked - target_masked)
    denominator = jnp.linalg.norm(target_masked)
    return numerator / denominator


def check_first_step_inputs(
    *,
    solver: ReactionDiffusionSolver,
    family: str,
    theta_true_raw: jnp.ndarray,
    initial_state: jnp.ndarray,
    forcing,
    dt: float,
    nu: float,
    context: str,
) -> None:
    x_phys = jnp.asarray(solver.x, dtype=float)
    rhs_forcing = forcing(nu, dt, x_phys)
    reaction_term = reaction_family_value(family, theta_true_raw, initial_state, x_phys)
    rhs_physical = initial_state + dt * (rhs_forcing + reaction_term)
    rhs_boundary = exact_state(dt, jnp.asarray(solver.xb, dtype=float))
    system = build_implicit_system(solver.lap, solver.bc, solver.n, -nu * dt)
    rhs = build_implicit_rhs(rhs_physical, rhs_boundary)

    checks = {
        "x_phys": x_phys,
        "x_boundary": solver.xb,
        "normals": solver.nr,
        "lap": solver.lap,
        "bc": solver.bc,
        "initial_state": initial_state,
        "forcing_dt": rhs_forcing,
        "reaction_initial": reaction_term,
        "rhs_physical": rhs_physical,
        "rhs_boundary": rhs_boundary,
        "implicit_system": system,
        "implicit_rhs": rhs,
    }
    print(f"[diagnostics] {context}", flush=True)
    for name, value in checks.items():
        print("[diagnostics] " + finite_summary(name, value), flush=True)
        assert_finite(name, value, context)


def make_discrete_bdf_forcing(family: str, nu: float, theta_true_raw: jnp.ndarray, dt: float):
    """Manufactured forcing consistent with the actual BDF/AB time discretization.

    The usual manufactured forcing closes the continuous PDE. This variant
    closes the discrete BDF1/BDF2/BDF3 formulas used by the solver, so a
    convergence run can isolate spatial discretization error.
    """

    reaction = build_reaction_term(family, theta_true_raw)

    def forcing(_nu_value: float, time: float, x: jnp.ndarray) -> jnp.ndarray:
        step = max(1, int(round(float(time) / float(dt))))
        u_np1 = exact_state(time, x)
        if step == 1:
            u_n = exact_state(time - dt, x)
            bdf_derivative = (u_np1 - u_n) / dt
            reaction_extrapolated = reaction(time, u_n, x)
        elif step == 2:
            u_nm2 = exact_state(time - 2.0 * dt, x)
            u_nm1 = exact_state(time - dt, x)
            bdf_derivative = (3.0 * u_np1 - 4.0 * u_nm1 + u_nm2) / (2.0 * dt)
            reaction_extrapolated = 2.0 * reaction(time, u_nm1, x) - reaction(time, u_nm2, x)
        else:
            u_nm2 = exact_state(time - 3.0 * dt, x)
            u_nm1 = exact_state(time - 2.0 * dt, x)
            u_n = exact_state(time - dt, x)
            bdf_derivative = (11.0 * u_np1 - 18.0 * u_n + 9.0 * u_nm1 - 2.0 * u_nm2) / (6.0 * dt)
            reaction_extrapolated = 3.0 * reaction(time, u_n, x) - 3.0 * reaction(time, u_nm1, x) + reaction(time, u_nm2, x)
        return bdf_derivative - nu * exact_state_laplacian(time, x) - reaction_extrapolated

    return forcing


def rollout_states_for_forward(
    solver_template: ReactionDiffusionSolver,
    *,
    family: str,
    theta_raw: jnp.ndarray,
    initial_state: jnp.ndarray,
    forcing,
    exact_states: list[jnp.ndarray],
    dt: float,
    startup_mode: str,
):
    if startup_mode == "stepped":
        return rollout_states(
            solver_template,
            family=family,
            theta_raw=theta_raw,
            initial_state=initial_state,
            forcing=forcing,
            exact_states=exact_states,
            dt=dt,
        )
    if startup_mode != "exact_history":
        raise ValueError(f"unknown startup mode {startup_mode}")
    if len(exact_states) < 3:
        raise ValueError("--startup-mode exact_history requires at least three timesteps")

    solver = solver_template.spawn_rollout_copy()
    solver.set_state_history(initial_state, exact_states[0], exact_states[1])
    reaction_term = build_reaction_term(family, theta_raw)
    neu_coeff = lambda xb: jnp.zeros((xb.shape[0],), dtype=float)
    dir_coeff = lambda xb: jnp.ones((xb.shape[0],), dtype=float)
    bc = lambda neu_coeffs, dir_coeffs, nr, time, xb: exact_state(time, xb)

    states = [exact_states[0], exact_states[1]]
    for k in range(2, len(exact_states)):
        states.append(solver.bdf3_step((k + 1) * dt, reaction_term, neu_coeff, dir_coeff, bc, forcing))
    return states


def run_case(
    *,
    domain: DomainDescriptor,
    metadata: SavedDomainMetadata,
    family: str,
    order: int,
    stencil: str,
    linear_solver: str,
    final_time: float,
    dt_factor: float,
    min_steps: int,
    fixed_steps: int | None,
    diagnostics: bool,
    forcing_mode: str,
    startup_mode: str,
    setup_device: str,
    sparse_assembly_chunk_size: int | None,
    gmres_tol: float,
    gmres_atol: float,
    gmres_restart: int,
    gmres_maxiter: int | None,
    interior_distance_factor: float,
    h_channel_visibility_filter: bool,
    visibility_candidate_factor: int,
    visibility_segment_samples: int,
    visibility_level_set_tolerance: float,
    visibility_max_distance_factor: float,
) -> ForwardResult:
    if setup_device == "cpu":
        setup_domain = copy_domain_to_device(domain, jax.devices("cpu")[0])
    else:
        setup_domain = domain
    with setup_device_context(setup_device):
        x_phys = jnp.asarray(setup_domain.get_int_bdry_nodes(), dtype=float)
        dim = x_phys.shape[1]
        theta_true_raw = sample_theta(family, dim)
        _default_dt, _, nu = solver_settings(dim)
        if fixed_steps is None:
            dt, steps = timestep_from_h(metadata.h, final_time, dt_factor, min_steps)
        else:
            if fixed_steps <= 0:
                raise ValueError("fixed steps must be positive")
            steps = int(fixed_steps)
            dt = float(final_time) / float(steps)
        if forcing_mode == "continuous":
            forcing = make_forcing(family, nu, theta_true_raw)
        elif forcing_mode == "discrete_bdf":
            forcing = make_discrete_bdf_forcing(family, nu, theta_true_raw, dt)
        else:
            raise ValueError(f"unknown forcing mode {forcing_mode}")
        initial_state = exact_state(0.0, x_phys)
        exact_states = [exact_state((k + 1) * dt, x_phys) for k in range(steps)]
        interior_mask = interior_mask_from_boundary(setup_domain, metadata.h, interior_distance_factor)
        interior_node_count = int(jnp.sum(interior_mask))
        neu_coeff = lambda xb: jnp.zeros((xb.shape[0],), dtype=float)
        dir_coeff = lambda xb: jnp.ones((xb.shape[0],), dtype=float)

        def make_solver() -> ReactionDiffusionSolver:
            if not h_channel_visibility_filter:
                return build_solver_template(
                    setup_domain,
                    order=order,
                    dt=dt,
                    nu=nu,
                    stencil=stencil,
                    linear_solver=linear_solver,
                    sparse_assembly_chunk_size=sparse_assembly_chunk_size,
                    gmres_tol=gmres_tol,
                    gmres_atol=gmres_atol,
                    gmres_restart=gmres_restart,
                    gmres_maxiter=gmres_maxiter,
                )
            if metadata.geometry != "h_channel":
                raise ValueError("--h-channel-visibility-filter is only supported for saved h_channel domains")
            solver, filter_stats = build_h_channel_visibility_solver_template(
                setup_domain,
                order=order,
                dt=dt,
                nu=nu,
                stencil=stencil,
                linear_solver=linear_solver,
                candidate_factor=visibility_candidate_factor,
                segment_samples=visibility_segment_samples,
                level_set_tolerance=visibility_level_set_tolerance,
                max_distance_factor=visibility_max_distance_factor,
            )
            solver.gmres_tol = gmres_tol
            solver.gmres_atol = gmres_atol
            solver.gmres_restart = gmres_restart
            solver.gmres_maxiter = gmres_maxiter
            print(
                "[visibility] "
                f"lap_rejected_edges={filter_stats.lap_rejected_edges}, "
                f"lap_fallback_rows={filter_stats.lap_fallback_rows}, "
                f"boundary_rejected_edges={filter_stats.boundary_rejected_edges}, "
                f"boundary_fallback_rows={filter_stats.boundary_fallback_rows}",
                flush=True,
            )
            return solver

        preprocess_start_time = perf_counter()
        warmup_solver = make_solver()
        warmup_solver.prepare_boundary_operator(dt, neu_coeff, dir_coeff)
        synchronize_solver_operators(warmup_solver)
        context = (
            f"reaction-diffusion forward finite preflight: "
            f"family={family}, geometry={metadata.geometry}, resolution={metadata.label}, "
            f"order={order}, h={metadata.h}, dt={dt}, steps={steps}, linear_solver={linear_solver}"
        )
        if diagnostics:
            check_first_step_inputs(
                solver=warmup_solver,
                family=family,
                theta_true_raw=theta_true_raw,
                initial_state=initial_state,
                forcing=forcing,
                dt=dt,
                nu=nu,
                context=context,
            )
        if linear_solver == "gmres_ilu":
            warmup_solver.prepare_preconditioner(dt, neu_coeff, dir_coeff, nu_ref=nu)
        warmup_states = rollout_states_for_forward(
            warmup_solver,
            family=family,
            theta_raw=theta_true_raw,
            initial_state=initial_state,
            forcing=forcing,
            exact_states=exact_states,
            dt=dt,
            startup_mode=startup_mode,
        )
        jax.block_until_ready(warmup_states[-1])
        preprocess_elapsed_seconds = perf_counter() - preprocess_start_time

        start_time = perf_counter()
        solver_template = make_solver()
        solver_template.prepare_boundary_operator(dt, neu_coeff, dir_coeff)
        synchronize_solver_operators(solver_template)
        if linear_solver == "gmres_ilu":
            solver_template.prepare_preconditioner(dt, neu_coeff, dir_coeff, nu_ref=nu)

        states = rollout_states_for_forward(
            solver_template,
            family=family,
            theta_raw=theta_true_raw,
            initial_state=initial_state,
            forcing=forcing,
            exact_states=exact_states,
            dt=dt,
            startup_mode=startup_mode,
        )
        errors = jnp.asarray([relative_l2(state, target) for state, target in zip(states, exact_states, strict=True)])
        interior_errors = jnp.asarray(
            [masked_relative_l2(state, target, interior_mask) for state, target in zip(states, exact_states, strict=True)]
        )
        jax.block_until_ready(errors)
        jax.block_until_ready(interior_errors)
        elapsed_seconds = perf_counter() - start_time
    return ForwardResult(
        order=order,
        resolution=metadata.label,
        h=metadata.h,
        dt=dt,
        steps=steps,
        final_time=final_time,
        node_count=int(domain.get_num_int_bdry_nodes()),
        final_relative_l2_error=float(errors[-1]),
        mean_relative_l2_error=float(jnp.mean(errors)),
        max_relative_l2_error=float(jnp.max(errors)),
        interior_node_count=interior_node_count,
        interior_distance_factor=interior_distance_factor,
        interior_final_relative_l2_error=float(interior_errors[-1]),
        interior_mean_relative_l2_error=float(jnp.mean(interior_errors)),
        interior_max_relative_l2_error=float(jnp.max(interior_errors)),
        elapsed_seconds=elapsed_seconds,
        preprocess_elapsed_seconds=preprocess_elapsed_seconds,
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
    interior_distance_factor: float,
) -> ForwardResult:
    if fixed_steps is None:
        dt, steps = timestep_from_h(metadata.h, final_time, dt_factor, min_steps)
    else:
        steps = int(fixed_steps)
        dt = float(final_time) / float(steps)
    return ForwardResult(
        order=order,
        resolution=metadata.label,
        h=metadata.h,
        dt=dt,
        steps=steps,
        final_time=final_time,
        node_count=int(domain.get_num_int_bdry_nodes()),
        final_relative_l2_error=-1.0,
        mean_relative_l2_error=-1.0,
        max_relative_l2_error=-1.0,
        interior_node_count=-1,
        interior_distance_factor=interior_distance_factor,
        interior_final_relative_l2_error=-1.0,
        interior_mean_relative_l2_error=-1.0,
        interior_max_relative_l2_error=-1.0,
        elapsed_seconds=-1.0,
        preprocess_elapsed_seconds=-1.0,
    )


def export_forward_mat(
    results: list[ForwardResult],
    *,
    dim: int,
    geometry: str,
    family: str,
    stencil: str,
    linear_solver: str,
    final_time: float,
    dt_factor: float,
    min_steps: int,
    forcing_mode: str,
    startup_mode: str,
    gmres_tol: float,
    gmres_atol: float,
    gmres_restart: int,
    gmres_maxiter: int | None,
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
    interior_relative_l2_errors = np.full(shape, np.nan, dtype=float)
    interior_mean_relative_l2_errors = np.full(shape, np.nan, dtype=float)
    interior_max_relative_l2_errors = np.full(shape, np.nan, dtype=float)
    elapsed_seconds = np.full(shape, np.nan, dtype=float)
    preprocess_elapsed_seconds = np.full(shape, np.nan, dtype=float)
    node_counts = np.full(shape, np.nan, dtype=float)
    interior_node_counts = np.full(shape, np.nan, dtype=float)
    dt_values = np.full(shape, np.nan, dtype=float)
    steps_values = np.full(shape, np.nan, dtype=float)
    hs = np.full((len(labels),), np.nan, dtype=float)

    for row in results:
        i = order_to_idx[row.order]
        j = label_to_idx[row.resolution]
        relative_l2_errors[i, j] = row.final_relative_l2_error
        mean_relative_l2_errors[i, j] = row.mean_relative_l2_error
        max_relative_l2_errors[i, j] = row.max_relative_l2_error
        interior_relative_l2_errors[i, j] = row.interior_final_relative_l2_error
        interior_mean_relative_l2_errors[i, j] = row.interior_mean_relative_l2_error
        interior_max_relative_l2_errors[i, j] = row.interior_max_relative_l2_error
        elapsed_seconds[i, j] = row.elapsed_seconds
        preprocess_elapsed_seconds[i, j] = row.preprocess_elapsed_seconds
        node_counts[i, j] = row.node_count
        interior_node_counts[i, j] = row.interior_node_count
        dt_values[i, j] = row.dt
        steps_values[i, j] = row.steps
        hs[j] = row.h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        output_path,
        {
            "relative_l2_errors": relative_l2_errors,
            "mean_relative_l2_errors": mean_relative_l2_errors,
            "max_relative_l2_errors": max_relative_l2_errors,
            "interior_relative_l2_errors": interior_relative_l2_errors,
            "interior_mean_relative_l2_errors": interior_mean_relative_l2_errors,
            "interior_max_relative_l2_errors": interior_max_relative_l2_errors,
            "elapsed_seconds": elapsed_seconds,
            "preprocess_elapsed_seconds": preprocess_elapsed_seconds,
            "orders": np.asarray(orders, dtype=float),
            "hs": hs,
            "point_cloud_numbers": np.arange(1, len(labels) + 1, dtype=float),
            "labels": np.asarray(labels, dtype=object),
            "node_counts": node_counts,
            "interior_node_counts": interior_node_counts,
            "interior_distance_factor": np.asarray([[results[0].interior_distance_factor if results else np.nan]], dtype=float),
            "dt_values": dt_values,
            "steps_values": steps_values,
            "geometry": np.asarray([geometry], dtype=object),
            "dimension": np.asarray([[dim]], dtype=float),
            "family": np.asarray([family], dtype=object),
            "stencil": np.asarray([stencil], dtype=object),
            "linear_solver": np.asarray([linear_solver], dtype=object),
            "final_time": np.asarray([[final_time]], dtype=float),
            "dt_factor": np.asarray([[dt_factor]], dtype=float),
            "min_steps": np.asarray([[min_steps]], dtype=float),
            "forcing_mode": np.asarray([forcing_mode], dtype=object),
            "startup_mode": np.asarray([startup_mode], dtype=object),
            "gmres_tol": np.asarray([[gmres_tol]], dtype=float),
            "gmres_atol": np.asarray([[gmres_atol]], dtype=float),
            "gmres_restart": np.asarray([[gmres_restart]], dtype=float),
            "gmres_maxiter": np.asarray([[-1 if gmres_maxiter is None else gmres_maxiter]], dtype=float),
            "timestep_rule": np.asarray(["dt = final_time / max(min_steps, ceil(final_time / (dt_factor * h)))"], dtype=object),
            "error_metric": np.asarray(["final_relative_l2"], dtype=object),
        },
    )


def render_markdown(
    results: list[ForwardResult],
    *,
    dim: int,
    geometry: str,
    family: str,
    stencil: str,
    linear_solver: str,
    final_time: float,
    dt_factor: float,
    min_steps: int,
    fixed_steps: int | None,
    forcing_mode: str,
    startup_mode: str,
    gmres_tol: float,
    gmres_atol: float,
    gmres_restart: int,
    gmres_maxiter: int | None,
) -> str:
    lines = [
        f"# Reaction-Diffusion Forward Convergence Study ({family}, {stencil.upper()})",
        "",
        f"- dimension: {dim}",
        f"- geometry: {geometry}",
        f"- family: {family}",
        f"- stencil: {stencil}",
        f"- linear solver: {linear_solver}",
        f"- final time: {final_time}",
        f"- timestep rule: {'fixed steps' if fixed_steps is not None else 'h-dependent'}",
        f"- dt factor: {dt_factor}",
        f"- minimum steps: {min_steps}",
        f"- forcing mode: {forcing_mode}",
        f"- startup mode: {startup_mode}",
        f"- GMRES tolerance: {gmres_tol}",
        f"- GMRES absolute tolerance: {gmres_atol}",
        f"- GMRES restart: {gmres_restart}",
        f"- GMRES max iterations: {'default' if gmres_maxiter is None else gmres_maxiter}",
        f"- interior error mask: nodes with distance to boundary >= {results[0].interior_distance_factor if results else 0.0} h",
        f"- JAX backend: `{jax.default_backend()}`",
        f"- timing: `preprocess seconds is first warmup/compile pass; seconds is second pass after warmup`",
        "",
        "| xi | resolution | h | dt | steps | physical nodes | final relative L2 | mean relative L2 | max relative L2 | interior nodes | interior final relative L2 | interior mean relative L2 | interior max relative L2 | preprocess seconds | seconds |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {row.order} | {row.resolution} | {row.h:.5f} | {row.dt:.6e} | {row.steps} | {row.node_count} | "
            f"{row.final_relative_l2_error:.6e} | {row.mean_relative_l2_error:.6e} | "
            f"{row.max_relative_l2_error:.6e} | {row.interior_node_count} | "
            f"{row.interior_final_relative_l2_error:.6e} | {row.interior_mean_relative_l2_error:.6e} | "
            f"{row.interior_max_relative_l2_error:.6e} | {row.preprocess_elapsed_seconds:.6f} | {row.elapsed_seconds:.6f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reaction-diffusion forward convergence study using inverse-test families.")
    parser.add_argument("--family", type=str, default="semilinear", choices=FAMILIES)
    parser.add_argument("--dim", type=int, default=2, choices=(2, 3))
    parser.add_argument("--geometry", type=str, default=None)
    parser.add_argument("--orders", type=int, nargs="+", default=[2,4,6])
    parser.add_argument("--hs", type=float, nargs="+", default=[.45,.35,.25,.15])
    parser.add_argument("--domain-files", type=Path, nargs="+", default=None)
    parser.add_argument("--stencil", type=str, default="rbf", choices=("wls", "rbf"))
    parser.add_argument("--linear-solver", type=str, default="lineax", choices=("gmres", "gmres_ilu", "dense", "lineax", "cudss"))
    parser.add_argument("--final-time", type=float, default=0.5)
    parser.add_argument("--dt-factor", type=float, default=0.25)
    parser.add_argument("--min-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=None, help="Legacy fixed-step mode. If provided, overrides h-dependent timestep selection.")
    parser.add_argument(
        "--forcing-mode",
        type=str,
        choices=("continuous", "discrete_bdf"),
        default="continuous",
        help="Use continuous manufactured forcing, or discrete-BDF forcing that removes time truncation from the manufactured test.",
    )
    parser.add_argument(
        "--startup-mode",
        type=str,
        choices=("stepped", "exact_history"),
        default="stepped",
        help="Use normal BDF1/BDF2 startup, or seed BDF3 with exact history to remove startup error in manufactured tests.",
    )
    parser.add_argument("--setup-device", choices=("default", "cpu"), default="default")
    parser.add_argument("--sparse-assembly-chunk-size", type=int, default=None)
    parser.add_argument("--gmres-tol", type=float, default=1.0e-8)
    parser.add_argument("--gmres-atol", type=float, default=1.0e-10)
    parser.add_argument("--gmres-restart", type=int, default=40)
    parser.add_argument("--gmres-maxiter", type=int, default=None)
    parser.add_argument(
        "--interior-error-distance-factor",
        type=float,
        default=2.0,
        help="Report an additional relative L2 error using only nodes at least this many h from the boundary.",
    )
    parser.add_argument("--diagnostics", action="store_true", help="Print finite checks for first-step matrix/RHS inputs before rollout.")
    parser.add_argument(
        "--h-channel-visibility-filter",
        action="store_true",
        help="For saved H-channel domains, reject physical stencil edges whose segment leaves the analytic channel geometry.",
    )
    parser.add_argument("--visibility-candidate-factor", type=int, default=8)
    parser.add_argument("--visibility-segment-samples", type=int, default=8)
    parser.add_argument("--visibility-level-set-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--visibility-max-distance-factor", type=float, default=2.0)
    parser.add_argument("--failure-tracebacks", action="store_true", help="Print full tracebacks for failed cases instead of only a concise failure line.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--mapped-cloud-preferred",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the mapped closed-curve node generator for smooth 2D geometries.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mat-output", type=Path, default=DEFAULT_MAT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    geometry_name = args.geometry or default_geometry(args.dim)
    if args.domain_files:
        domain_entries = [load_domain_file(path) for path in args.domain_files]
    else:
        validate_geometry(args.dim, geometry_name)
        hs = args.hs or [default_h(args.dim)]
        domain_entries = []
        for h in hs:
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
        orders = args.orders or ([4] if metadata.dim == 2 else [3])
        for order in orders:
            try:
                results.append(
                    run_case(
                        domain=domain,
                        metadata=metadata,
                        family=args.family,
                        order=order,
                        stencil=args.stencil,
                        linear_solver=args.linear_solver,
                        final_time=args.final_time,
                        dt_factor=args.dt_factor,
                        min_steps=args.min_steps,
                        fixed_steps=args.steps,
                        diagnostics=args.diagnostics,
                        forcing_mode=args.forcing_mode,
                        startup_mode=args.startup_mode,
                        setup_device=args.setup_device,
                        sparse_assembly_chunk_size=args.sparse_assembly_chunk_size,
                        gmres_tol=args.gmres_tol,
                        gmres_atol=args.gmres_atol,
                        gmres_restart=args.gmres_restart,
                        gmres_maxiter=args.gmres_maxiter,
                        interior_distance_factor=args.interior_error_distance_factor,
                        h_channel_visibility_filter=args.h_channel_visibility_filter,
                        visibility_candidate_factor=args.visibility_candidate_factor,
                        visibility_segment_samples=args.visibility_segment_samples,
                        visibility_level_set_tolerance=args.visibility_level_set_tolerance,
                        visibility_max_distance_factor=args.visibility_max_distance_factor,
                    )
                )
            except Exception as exc:
                exc_lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
                preferred_markers = ("received non-finite", "out of memory", "RESOURCE_EXHAUSTED", "nan", "inf")
                exc_summary = next(
                    (line for line in exc_lines if any(marker.lower() in line.lower() for marker in preferred_markers)),
                    exc_lines[0] if exc_lines else str(exc),
                )
                print(
                    "[failure] reaction-diffusion forward case failed; writing -1 sentinels and continuing: "
                    f"family={args.family}, geometry={metadata.geometry}, resolution={metadata.label}, "
                    f"order={order}, stencil={args.stencil}, linear_solver={args.linear_solver}: "
                    f"{type(exc).__name__}: {exc_summary}",
                    file=sys.stderr,
                    flush=True,
                )
                if args.failure_tracebacks:
                    traceback.print_exc(file=sys.stderr)
                results.append(
                    failed_result(
                        domain=domain,
                        metadata=metadata,
                        order=order,
                        final_time=args.final_time,
                        dt_factor=args.dt_factor,
                        min_steps=args.min_steps,
                        fixed_steps=args.steps,
                        interior_distance_factor=args.interior_error_distance_factor,
                    )
                )

    dim = domain_entries[0][1].dim if domain_entries else args.dim
    geometry = domain_entries[0][1].geometry if domain_entries else geometry_name
    markdown = render_markdown(
        results,
        dim=dim,
        geometry=geometry,
        family=args.family,
        stencil=args.stencil,
        linear_solver=args.linear_solver,
        final_time=args.final_time,
        dt_factor=args.dt_factor,
        min_steps=args.min_steps,
        fixed_steps=args.steps,
        forcing_mode=args.forcing_mode,
        startup_mode=args.startup_mode,
        gmres_tol=args.gmres_tol,
        gmres_atol=args.gmres_atol,
        gmres_restart=args.gmres_restart,
        gmres_maxiter=args.gmres_maxiter,
    )
    export_forward_mat(
        results,
        dim=dim,
        geometry=geometry,
        family=args.family,
        stencil=args.stencil,
        linear_solver=args.linear_solver,
        final_time=args.final_time,
        dt_factor=args.dt_factor,
        min_steps=args.min_steps,
        forcing_mode=args.forcing_mode,
        startup_mode=args.startup_mode,
        gmres_tol=args.gmres_tol,
        gmres_atol=args.gmres_atol,
        gmres_restart=args.gmres_restart,
        gmres_maxiter=args.gmres_maxiter,
        output_path=args.mat_output,
    )
    print(markdown, end="")
    print(f"wrote matlab output to {args.mat_output}")


if __name__ == "__main__":
    main()
