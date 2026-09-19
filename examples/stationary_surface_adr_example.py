"""Solve a manufactured ADR problem on the stationary unit sphere."""

import jax.numpy as jnp

from kernelpack import geometry, manifold, solvers


def main() -> None:
    node_count = 256
    xi = 2
    dt = 0.01
    step_count = 5
    diffusivity = 0.03
    points = geometry.fibonacci_sphere(node_count)
    exact = lambda time: jnp.exp(-time) * (2.0 + points[:, 0])
    ell = manifold.surface_polynomial_degree(xi)
    neighbors = manifold.build_surface_stencil_graph(
        points,
        manifold.surface_stencil_size(ell),
    )
    operators, _ = manifold.assemble_tangent_plane_operators(
        points,
        points,
        neighbors,
        xi=xi,
    )
    history = solvers.initialize_moving_surface_history(points, exact(0.0))
    quadrature = jnp.full((node_count,), 4.0 * jnp.pi / node_count)

    for step in range(1, step_count + 1):
        time = step * dt
        truth = exact(time)
        forcing = -truth + 2.0 * diffusivity * jnp.exp(-time) * points[:, 0]
        history, info = solvers.moving_surface_adr_step(
            history,
            points,
            operators,
            forcing,
            quadrature,
            8.0 * jnp.pi * jnp.exp(-time),
            dt,
            diffusivity,
            jnp.zeros(3),
            1.0e-9,
            order=min(step, 3),
            hyperviscosity_power=0,
        )

    error = jnp.linalg.norm(history.concentration[0] - exact(step_count * dt))
    error /= jnp.linalg.norm(exact(step_count * dt))
    print(f"relative l2 error: {float(error):.6e}")
    print(f"linear relative residual: {float(info.relative_residual):.6e}")


if __name__ == "__main__":
    main()
