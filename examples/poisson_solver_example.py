"""Solve Poisson's equation on a compact embedded-domain descriptor."""

import jax.numpy as jnp

from kernelpack import solvers

from _common import build_square_domain, exact_solution, unit_dirichlet, zero_neumann


def main() -> None:
    domain = build_square_domain()
    points = domain.get_int_bdry_nodes()
    solver = solvers.PoissonSolver(
        lap_assembler="fd",
        bc_assembler="fd",
        lap_stencil="wls",
        bc_stencil="wls",
    )
    solver.init(domain, 3)
    result = solver.solve(
        lambda x: -4.0 * jnp.ones(x.shape[0]),
        zero_neumann,
        unit_dirichlet,
        lambda alpha, beta, normals, xb: exact_solution(0.0, xb),
    )
    truth = exact_solution(0.0, points)
    error = jnp.linalg.norm(result["u"] - truth) / jnp.linalg.norm(truth)
    print(f"relative l2 error: {float(error):.6e}")


if __name__ == "__main__":
    main()
