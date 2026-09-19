"""Evolve a sphere under ``X_t = Delta_Gamma X``."""

import jax.numpy as jnp

from kernelpack import geometry, manifold, solvers


def main() -> None:
    node_count = 256
    xi = 2
    radius_zero = 1.4
    dt = 2.0e-4
    step_count = 10
    directions = geometry.fibonacci_sphere(node_count)
    points = radius_zero * directions
    ell = manifold.surface_polynomial_degree(xi)
    neighbors = manifold.build_surface_stencil_graph(
        points,
        manifold.surface_stencil_size(ell),
    )

    for _ in range(step_count):
        normals = points / jnp.linalg.norm(points, axis=1, keepdims=True)
        operators, _ = manifold.assemble_tangent_plane_operators(
            points,
            normals,
            neighbors,
            xi=xi,
        )
        points, info = solvers.mean_curvature_flow_step(
            points,
            normals,
            operators,
            dt,
        )

    final_time = step_count * dt
    exact_radius = jnp.sqrt(radius_zero**2 - 4.0 * final_time)
    radius = jnp.linalg.norm(points, axis=1)
    relative_error = jnp.linalg.norm(radius - exact_radius) / jnp.linalg.norm(
        jnp.full_like(radius, exact_radius)
    )
    print(f"relative radius error: {float(relative_error):.6e}")
    print(f"linear relative residual: {float(info.relative_residual):.6e}")


if __name__ == "__main__":
    main()
