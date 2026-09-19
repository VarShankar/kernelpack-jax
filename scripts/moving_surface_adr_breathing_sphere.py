"""Run a short GPU-first moving-surface ADR verification problem."""

import jax
import jax.numpy as jnp

from kernelpack import geometry, manifold, solvers


def main() -> None:
    node_count = 256
    xi = 2
    dt = 0.01
    step_count = 10
    diffusivity = 0.03
    material_sites = geometry.fibonacci_sphere(node_count)
    radius = lambda time: 1.0 + 0.1 * jnp.sin(time)
    radius_rate = lambda time: 0.1 * jnp.cos(time)
    points_at = lambda time: radius(time) * material_sites
    exact = lambda time: jnp.exp(-time) * (2.0 + material_sites[:, 0])

    ell = manifold.surface_polynomial_degree(xi)
    neighbors = manifold.build_surface_stencil_graph(
        material_sites,
        manifold.surface_stencil_size(ell),
        backend="auto",
    )
    history = solvers.initialize_moving_surface_history(points_at(0.0), exact(0.0))
    operators = None
    operator_cache = None
    calibration = None

    for step_index in range(1, step_count + 1):
        time = step_index * dt
        points = points_at(time)
        normals = material_sites
        if operator_cache is None:
            operators, operator_cache = manifold.assemble_tangent_plane_operators(
                points,
                normals,
                neighbors,
                xi=xi,
            )
        else:
            operators, operator_cache, _ = manifold.update_tangent_plane_operators(
                points,
                normals,
                neighbors,
                operator_cache,
                xi=xi,
                tolerance=1.0e-8,
            )

        h = jnp.sqrt(1.0 / node_count)
        if calibration is None:
            calibration = manifold.calibrate_surface_hyperviscosity(
                operators,
                points,
                normals,
                h,
                target_order=xi,
                arnoldi_iterations=12,
            )
        else:
            calibration, _, _ = manifold.refresh_surface_hyperviscosity(
                operators,
                points,
                normals,
                h,
                calibration,
                drift_threshold=0.15,
                arnoldi_iterations=12,
            )

        exact_now = exact(time)
        forcing = (
            -exact_now
            + 2.0 * radius_rate(time) / radius(time) * exact_now
            + 2.0 * diffusivity * jnp.exp(-time) * material_sites[:, 0] / radius(time) ** 2
        )
        quadrature = jnp.full(
            (node_count,),
            4.0 * jnp.pi * radius(time) ** 2 / node_count,
        )
        target_mass = 8.0 * jnp.pi * radius(time) ** 2 * jnp.exp(-time)
        history, info = solvers.moving_surface_adr_step(
            history,
            points,
            operators,
            forcing,
            quadrature,
            target_mass,
            dt,
            diffusivity,
            calibration.gamma,
            1.0e-8,
            order=min(step_index, 3),
            hyperviscosity_power=calibration.power,
        )

    numerical = history.concentration[0]
    relative_error = jnp.linalg.norm(numerical - exact(step_count * dt)) / jnp.linalg.norm(
        exact(step_count * dt)
    )
    relative_error.block_until_ready()
    print(f"device: {jax.devices()[0]}")
    print(f"relative l2 error: {float(relative_error):.6e}")
    print(f"linear relative residual: {float(info.relative_residual):.6e}")
    print(f"mass error after projection: {float(info.mass_after_projection - target_mass):.6e}")


if __name__ == "__main__":
    main()
