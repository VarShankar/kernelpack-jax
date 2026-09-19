"""Advance the same manufactured diffusion problem with FD and PU solvers."""

import jax.numpy as jnp

from kernelpack import solvers

from _common import build_square_domain, exact_solution, unit_dirichlet, zero_neumann


def run_solver(solver) -> jnp.ndarray:
    dt = 0.02
    diffusivity = 0.25
    domain = build_square_domain()
    points = domain.get_int_bdry_nodes()
    forcing = lambda nu, time, x: (
        -jnp.exp(-time) * (x[:, 0] ** 2 + x[:, 1] ** 2)
        - 4.0 * nu * jnp.exp(-time)
    )
    boundary = lambda alpha, beta, normals, time, xb: exact_solution(time, xb)
    solver.init(domain, 3, dt, diffusivity)
    solver.set_initial_state(exact_solution(0.0, points))
    solver.bdf1_step(dt, forcing, zero_neumann, unit_dirichlet, boundary)
    solver.bdf2_step(2.0 * dt, forcing, zero_neumann, unit_dirichlet, boundary)
    return solver.bdf3_step(3.0 * dt, forcing, zero_neumann, unit_dirichlet, boundary)


def main() -> None:
    domain = build_square_domain()
    points = domain.get_int_bdry_nodes()
    final_time = 0.06
    fd_solution = run_solver(
        solvers.DiffusionSolver(
            lap_assembler="fd",
            bc_assembler="fd",
            lap_stencil="wls",
            bc_stencil="wls",
        )
    )
    pu_solution = run_solver(solvers.PUDiffusionSolver())
    truth = exact_solution(final_time, points)
    print(f"FD relative l2 error: {float(jnp.linalg.norm(fd_solution - truth) / jnp.linalg.norm(truth)):.6e}")
    print(f"PU relative l2 error: {float(jnp.linalg.norm(pu_solution - truth) / jnp.linalg.norm(truth)):.6e}")


if __name__ == "__main__":
    main()
