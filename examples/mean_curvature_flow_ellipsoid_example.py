"""Smooth an ellipsoid with semi-implicit mean-curvature flow."""

import jax.numpy as jnp

from kernelpack import geometry, manifold, solvers


def main() -> None:
    points = geometry.fibonacci_sphere(256) * jnp.asarray([1.35, 0.85, 0.65])
    xi = 2
    dt = 2.0e-4
    step_count = 5
    ell = manifold.surface_polynomial_degree(xi)
    neighbors = manifold.build_surface_stencil_graph(
        points, manifold.surface_stencil_size(ell)
    )
    normals = manifold.estimate_pca_normals(points, stencil_size=24)
    initial_radius = jnp.linalg.norm(points - jnp.mean(points, axis=0), axis=1)
    for _ in range(step_count):
        operators, _ = manifold.assemble_tangent_plane_operators(
            points, normals, neighbors, xi=xi
        )
        points, info = solvers.mean_curvature_flow_step(
            points, normals, operators, dt
        )
        normals = manifold.estimate_pca_normals(
            points, previous_normals=normals, stencil_size=24
        )
    final_radius = jnp.linalg.norm(points - jnp.mean(points, axis=0), axis=1)
    print(f"initial radius range: {float(jnp.ptp(initial_radius)):.6e}")
    print(f"final radius range: {float(jnp.ptp(final_radius)):.6e}")
    print(f"linear relative residual: {float(info.relative_residual):.6e}")


if __name__ == "__main__":
    main()
