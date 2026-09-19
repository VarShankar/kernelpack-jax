"""Remap a sheared Lagrangian sphere and backfill its BDF history."""

import jax.numpy as jnp

from kernelpack import geometry, manifold, solvers


def velocity(_time, points):
    angular_speed = 1.0 + 6.0 * points[:, 2]
    return angular_speed[:, None] * jnp.column_stack(
        (-points[:, 1], points[:, 0], jnp.zeros(points.shape[0]))
    )


def reference_labels(points, time):
    angle = -(1.0 + 6.0 * points[:, 2]) * time
    return jnp.column_stack(
        (
            jnp.cos(angle) * points[:, 0] - jnp.sin(angle) * points[:, 1],
            jnp.sin(angle) * points[:, 0] + jnp.cos(angle) * points[:, 1],
            points[:, 2],
        )
    )


def main() -> None:
    node_count = 160
    xi = 2
    dt = 0.01
    step_count = 20
    remap_step = 10
    points = geometry.fibonacci_sphere(node_count)
    exact = lambda time, sites: jnp.exp(-time) * (2.0 + reference_labels(sites, time)[:, 0])
    history = solvers.initialize_moving_surface_history(points, exact(0.0, points))
    ell = manifold.surface_polynomial_degree(xi)
    stencil_size = manifold.surface_stencil_size(ell)
    quality_before = jnp.nan
    quality_after = jnp.nan

    for step in range(1, step_count + 1):
        time = step * dt
        new_points = solvers.rk3_material_step((step - 1) * dt, history.points[0], dt, velocity)
        new_points /= jnp.linalg.norm(new_points, axis=1, keepdims=True)
        normals = new_points
        neighbors = manifold.build_surface_stencil_graph(new_points, stencil_size)
        operators, _ = manifold.assemble_tangent_plane_operators(
            new_points, normals, neighbors, xi=xi
        )
        quadrature = jnp.full(node_count, 4.0 * jnp.pi / node_count)
        truth = exact(time, new_points)
        history, info = solvers.moving_surface_adr_step(
            history,
            new_points,
            operators,
            -truth,
            quadrature,
            8.0 * jnp.pi * jnp.exp(-time),
            dt,
            0.0,
            jnp.zeros(3),
            1.0e-9,
            order=min(step, 3),
            hyperviscosity_power=0,
        )

        if step == remap_step:
            nearest = manifold.build_surface_stencil_graph(history.points[0], 2)
            quality_before = manifold.surface_point_quality(history.points[0], nearest)
            target_now = geometry.fibonacci_sphere(node_count)
            target_history = solvers.semi_lagrangian_backfill_points(
                target_now,
                time,
                dt,
                velocity,
                history_levels=2,
            )
            target_history = jnp.concatenate((target_now[None, :, :], target_history), axis=0)
            transferred = []
            for level in range(3):
                target_level = target_history[level]
                cross = manifold.build_cross_surface_stencil_graph(
                    target_level, history.points[level], stencil_size
                )
                weights = manifold.local_tangent_interpolation_weights(
                    history.points[level],
                    target_level,
                    target_level,
                    cross,
                    xi=xi,
                )
                transferred.append(
                    manifold.apply_local_tangent_interpolant(
                        weights, cross, history.concentration[level]
                    )
                )
            history = solvers.MovingSurfaceHistory(
                concentration=jnp.stack(transferred),
                points=target_history,
            )
            nearest = manifold.build_surface_stencil_graph(history.points[0], 2)
            quality_after = manifold.surface_point_quality(history.points[0], nearest)

    truth = exact(step_count * dt, history.points[0])
    error = jnp.linalg.norm(history.concentration[0] - truth) / jnp.linalg.norm(truth)
    print(f"relative l2 error: {float(error):.6e}")
    print(f"quality before remap: {float(quality_before):.6e}")
    print(f"quality after remap: {float(quality_after):.6e}")
    print(f"linear relative residual: {float(info.relative_residual):.6e}")


if __name__ == "__main__":
    main()
