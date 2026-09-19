"""Solve linear and nonlinear variable-coefficient Poisson problems."""

import jax.numpy as jnp

from kernelpack import solvers

from _common import build_square_domain, exact_solution, unit_dirichlet, zero_neumann


def main() -> None:
    domain = build_square_domain()
    points = domain.get_int_bdry_nodes()
    exact = lambda x: exact_solution(0.0, x)
    coefficient = lambda x: 1.0 + 0.2 * x[:, 0]

    linear = solvers.VariablePoissonSolver(
        lap_assembler="fd",
        bc_assembler="fd",
        lap_stencil="wls",
        bc_stencil="wls",
        linear_solver="gmres_ilu",
    )
    linear.init(domain, 3)
    linear_result = linear.solve(
        lambda x: -4.0 - 1.2 * x[:, 0],
        coefficient,
        zero_neumann,
        unit_dirichlet,
        lambda alpha, beta, normals, xb: exact(xb),
    )

    nonlinear = solvers.NonlinearVariablePoissonSolver(
        lap_assembler="fd",
        bc_assembler="fd",
        lap_stencil="wls",
        bc_stencil="wls",
    )
    nonlinear.init(domain, 3)
    nonlinear_exact = lambda x: 2.0 + x[:, 0] + x[:, 1]
    initial = jnp.zeros(points.shape[0])
    coefficient_nonlinear = lambda x, u: 1.0 + 0.1 * u
    coefficient_derivative = lambda x, u: 0.1 * jnp.ones(x.shape[0])
    forcing = lambda x, u: -0.2 * jnp.ones(x.shape[0])
    forcing_derivative = lambda x, u: jnp.zeros(x.shape[0])
    nonlinear.prepare_preconditioner(
        forcing_derivative,
        coefficient_nonlinear,
        zero_neumann,
        unit_dirichlet,
        initial_guess=initial,
    )
    nonlinear_result = nonlinear.solve(
        forcing,
        forcing_derivative,
        coefficient_nonlinear,
        coefficient_derivative,
        zero_neumann,
        unit_dirichlet,
        lambda alpha, beta, normals, xb: nonlinear_exact(xb),
        initial_guess=initial,
    )

    print(f"linear relative l2 error: {float(jnp.linalg.norm(linear_result['u'] - exact(points)) / jnp.linalg.norm(exact(points))):.6e}")
    print(f"nonlinear relative l2 error: {float(jnp.linalg.norm(nonlinear_result['u'] - nonlinear_exact(points)) / jnp.linalg.norm(nonlinear_exact(points))):.6e}")
    print(f"nonlinear residual: {float(nonlinear_result['residual_norm']):.6e}")


if __name__ == "__main__":
    main()
