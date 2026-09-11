"""Three-dimensional manufactured moving-domain ADR convergence case.

This is the JAX counterpart of
``kernelpack-matlab/examples/moving_domain_adr_convergence_3d_xi4.m``.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time

import jax
import jax.numpy as jnp

from kernelpack import geometry, nodes, solvers


XI = 4
MAXIMUM_VELOCITY = 1.5 * math.sqrt(3.0 / 2.0)


def sphere_surface(center, radius, seed_count, h, *, build_model):
    sites = jnp.asarray(center, dtype=float) + radius * geometry.fibonacci_sphere(seed_count)
    surface = geometry.EmbeddedSurface()
    surface.set_data_sites(sites)
    if build_model:
        sample_count = max(12, math.ceil(4.0 * math.pi * radius**2 / h**2))
        surface.build_closed_geometric_model_ps(3, h, seed_count, sample_count)
        surface.build_level_set_from_geometric_model()
    return surface


def exact_solution(t, points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return 1.0 + jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y) * jnp.cos(jnp.pi * z) * jnp.sin(jnp.pi * t)


def velocity(t, points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    scale = 1.5 * jnp.sin(jnp.pi * (x * x + y * y + z * z)) * jnp.sin(jnp.pi * t)
    return scale[:, None] * jnp.column_stack([y * z, -2.0 * x * z, x * y])


def manufactured_forcing(nu, t, points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    sin_x = jnp.sin(jnp.pi * x)
    cos_x = jnp.cos(jnp.pi * x)
    sin_y = jnp.sin(jnp.pi * y)
    cos_y = jnp.cos(jnp.pi * y)
    sin_z = jnp.sin(jnp.pi * z)
    cos_z = jnp.cos(jnp.pi * z)
    sin_t = jnp.sin(jnp.pi * t)

    time_derivative = jnp.pi * sin_x * cos_y * cos_z * jnp.cos(jnp.pi * t)
    gradient = jnp.pi * sin_t * jnp.column_stack(
        [cos_x * cos_y * cos_z, -sin_x * sin_y * cos_z, -sin_x * cos_y * sin_z]
    )
    laplacian = -3.0 * jnp.pi**2 * sin_x * cos_y * cos_z * sin_t
    return time_derivative + jnp.sum(velocity(t, points) * gradient, axis=1) - nu * laplacian


def exact_dirichlet_data(alpha, beta, normals, t, points):
    del alpha, beta, normals
    return exact_solution(t, points)


def run_case(
    h=0.12,
    peclet=1000.0,
    final_time=0.5,
    seed=2021,
    *,
    report_steps=False,
    profile_phases=False,
    gmres_restart=10,
):
    if h <= 0.0 or peclet <= 0.0 or final_time <= 0.0:
        raise ValueError("h, peclet, and final_time must be positive")

    outer_seed_count = max(420, math.ceil(4.0 * math.pi / h**2))
    outer = sphere_surface([0.0, 0.0, 0.0], 1.0, outer_seed_count, h, build_model=True)
    hole = sphere_surface([0.35, 0.20, 0.15], 0.18, 220, h, build_model=False)
    background = nodes.DomainNodeGenerator().build_domain_descriptor_from_geometry(
        outer,
        h,
        seed=seed,
        strip_count=6,
    )

    background_count = background.get_num_int_bdry_nodes()
    h_measure = background_count ** (-1.0 / 3.0)
    temporal_dt = h_measure ** (XI / 3.0)
    cfl_dt = 0.3 * h_measure / MAXIMUM_VELOCITY
    step_count = max(3, math.ceil(final_time / min(temporal_dt, cfl_dt)))
    dt = final_time / step_count

    solver = solvers.MovingDomainADRSolver(
        gmres_restart=gmres_restart,
        profile_phases=profile_phases,
    )
    setup_start = time.perf_counter()
    solver.init(background, [hole], xi=XI, dt=dt, diffusivity=1.0 / peclet)
    solver.set_initial_state(exact_solution)
    jax.block_until_ready(solver.current_state())
    setup_seconds = time.perf_counter() - setup_start

    step_seconds = []
    step_node_counts = []
    step_recomputed_rows = []
    interpolation_recomputed_rows = []
    maximum_residual = 0.0
    maximum_gmres_info = 0
    fallback_count = 0
    phase_timings = []
    for step in range(step_count):
        step_start = time.perf_counter()
        state = solver.step(
            (step + 1) * dt,
            velocity,
            manufactured_forcing,
            0.0,
            1.0,
            exact_dirichlet_data,
            0.0,
        )
        jax.block_until_ready(state)
        step_seconds.append(time.perf_counter() - step_start)
        step_node_counts.append(int(solver.get_output_nodes().shape[0]))
        step_recomputed_rows.append(
            int(solver.last_update_diagnostics["laplacian_rows_recomputed"])
            + int(solver.last_update_diagnostics["boundary_rows_recomputed"])
        )
        interpolation_recomputed_rows.append(
            int(solver.last_update_diagnostics["interpolation_rows_recomputed"])
        )
        maximum_residual = max(maximum_residual, float(solver.last_linear_diagnostics["relative_residual"]))
        maximum_gmres_info = max(maximum_gmres_info, int(solver.last_linear_diagnostics["info"]))
        fallback_count += int(solver.last_linear_diagnostics["used_dense_fallback"])
        phase_timings.append(dict(solver.last_phase_timings))

    output_nodes = solver.get_output_nodes()
    numerical = solver.current_state()
    exact = exact_solution(final_time, output_nodes)
    error = numerical - exact
    relative_l2 = jnp.linalg.norm(error) / jnp.linalg.norm(exact)
    relative_linf = jnp.linalg.norm(error, ord=jnp.inf) / jnp.linalg.norm(exact, ord=jnp.inf)
    relative_l2, relative_linf = jax.block_until_ready((relative_l2, relative_linf))

    result = {
        "device": str(jax.devices()[0]),
        "h": h,
        "xi": XI,
        "peclet": peclet,
        "background_nodes": int(background_count),
        "final_nodes": int(output_nodes.shape[0]),
        "physical_capacity": int(
            solver.moving_domain.get_discretization_domain().get_num_int_bdry_nodes()
        ),
        "capacity_growths": int(solver.moving_domain.capacity_growth_count),
        "gmres_restart": int(gmres_restart),
        "step_count": int(step_count),
        "dt": float(dt),
        "relative_l2_error": float(relative_l2),
        "relative_linf_error": float(relative_linf),
        "maximum_linear_residual": maximum_residual,
        "maximum_gmres_info": maximum_gmres_info,
        "dense_fallback_count": fallback_count,
        "setup_seconds": setup_seconds,
        "first_step_seconds": step_seconds[0],
        "median_post_first_step_seconds": statistics.median(step_seconds[1:]) if len(step_seconds) > 1 else step_seconds[0],
        "minimum_post_first_step_seconds": min(step_seconds[1:]) if len(step_seconds) > 1 else step_seconds[0],
        "mean_step_seconds": sum(step_seconds) / len(step_seconds),
        "total_step_seconds": sum(step_seconds),
    }
    for key, value in result.items():
        print(f"{key}={value}")
    if report_steps:
        print("step,time_seconds,node_count,recomputed_rows,interpolation_recomputed_rows")
        for index, (elapsed, node_count, recomputed_rows, interp_rows) in enumerate(
            zip(step_seconds, step_node_counts, step_recomputed_rows, interpolation_recomputed_rows), start=1
        ):
            print(f"{index},{elapsed:.9f},{node_count},{recomputed_rows},{interp_rows}")
    if profile_phases:
        print("phase,median_post_first_seconds")
        for phase in phase_timings[0]:
            samples = [timings[phase] for timings in phase_timings[1:]] or [phase_timings[0][phase]]
            print(f"{phase},{statistics.median(samples):.9f}")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h", type=float, default=0.12)
    parser.add_argument("--peclet", type=float, default=1000.0)
    parser.add_argument("--final-time", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--report-steps", action="store_true")
    parser.add_argument("--profile-phases", action="store_true")
    parser.add_argument("--gmres-restart", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_case(
        arguments.h,
        arguments.peclet,
        arguments.final_time,
        arguments.seed,
        report_steps=arguments.report_steps,
        profile_phases=arguments.profile_phases,
        gmres_restart=arguments.gmres_restart,
    )
