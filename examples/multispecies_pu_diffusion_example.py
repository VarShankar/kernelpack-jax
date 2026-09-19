"""Advance two independently diffusing species with the localized PU solver."""

import jax.numpy as jnp

from kernelpack import solvers

from _common import build_square_domain


def exact(time: float, points: jnp.ndarray) -> jnp.ndarray:
    return jnp.column_stack(
        (
            jnp.exp(-time) * (points[:, 0] ** 2 + points[:, 1] ** 2),
            jnp.exp(-2.0 * time) * (1.0 + points[:, 0]),
        )
    )


def main() -> None:
    domain = build_square_domain()
    points = domain.get_int_bdry_nodes()
    dt = 0.02
    diffusivity = 0.2
    solver = solvers.MultiSpeciesPUDiffusionSolver()
    solver.init(domain, 3, dt, diffusivity)
    solver.set_initial_state(exact(0.0, points))
    forcing = lambda nu, time, x: jnp.column_stack(
        (
            -jnp.exp(-time) * (x[:, 0] ** 2 + x[:, 1] ** 2)
            - 4.0 * nu * jnp.exp(-time),
            -2.0 * jnp.exp(-2.0 * time) * (1.0 + x[:, 0]),
        )
    )
    result = solver.bdf1_step(
        dt,
        forcing,
        lambda time, xb: jnp.zeros(xb.shape[0]),
        lambda time, xb: jnp.ones(xb.shape[0]),
        lambda alpha, beta, normals, time, xb: exact(time, xb),
    )
    truth = exact(dt, points)
    print(f"relative Frobenius error: {float(jnp.linalg.norm(result - truth) / jnp.linalg.norm(truth)):.6e}")


if __name__ == "__main__":
    main()
