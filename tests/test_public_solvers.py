import jax.numpy as jnp

from kernelpack import solvers
from kernelpack.domain import DomainDescriptor


def build_test_domain():
    coordinates = jnp.linspace(-1.0, 1.0, 7)
    points = jnp.array([[x, y] for x in coordinates for y in coordinates])
    boundary_mask = jnp.any(jnp.isclose(jnp.abs(points), 1.0), axis=1)
    xb = points[boundary_mask]
    xi = points[~boundary_mask]
    normals = jnp.where(jnp.isclose(jnp.abs(xb), 1.0), jnp.sign(xb), 0.0)
    normals /= jnp.linalg.norm(normals, axis=1, keepdims=True)
    xg = xb + 0.25 * normals

    domain = DomainDescriptor()
    domain.set_nodes(xi, xb, xg)
    domain.set_normals(normals)
    domain.set_sep_rad(float(coordinates[1] - coordinates[0]))
    domain.build_structs()
    return domain


def test_poisson_solver_supports_rbf_wls_and_neumann_problems():
    domain = build_test_domain()
    exact = lambda x: x[:, 0] ** 2 + x[:, 1] ** 2
    truth = exact(domain.get_int_bdry_nodes())

    for stencil, tolerance in (("wls", 2.5e-1), ("rbf", 6.0e-1)):
        solver = solvers.PoissonSolver(
            lap_assembler="fd",
            bc_assembler="fd",
            lap_stencil=stencil,
            bc_stencil=stencil,
        )
        solver.init(domain, 3)
        result = solver.solve(
            lambda x: -4.0 * jnp.ones(x.shape[0]),
            lambda xb: jnp.zeros(xb.shape[0]),
            lambda xb: jnp.ones(xb.shape[0]),
            lambda alpha, beta, normals, xb: exact(xb),
        )
        assert jnp.max(jnp.abs(result["u"] - truth)) < tolerance

    neumann = solvers.PoissonSolver(
        lap_assembler="fd",
        bc_assembler="fd",
        lap_stencil="wls",
        bc_stencil="wls",
    )
    neumann.init(domain, 3)
    result = neumann.solve(
        lambda x: jnp.zeros(x.shape[0]),
        lambda xb: jnp.ones(xb.shape[0]),
        lambda xb: jnp.zeros(xb.shape[0]),
        lambda alpha, beta, normals, xb: jnp.zeros(xb.shape[0]),
    )
    assert result["used_nullspace_augmentation"]
    assert jnp.max(jnp.abs(result["u"])) < 1.0e-8


def test_variable_poisson_dense_and_sparse_solvers_agree():
    domain = build_test_domain()
    exact = lambda x: x[:, 0] ** 2 + x[:, 1] ** 2
    coefficient = lambda x: 1.0 + 0.2 * x[:, 0]
    forcing = lambda x: -4.0 - 1.2 * x[:, 0]

    results = []
    for linear_solver in ("dense", "gmres_ilu"):
        solver = solvers.VariablePoissonSolver(
            lap_assembler="fd",
            bc_assembler="fd",
            lap_stencil="wls",
            bc_stencil="wls",
            linear_solver=linear_solver,
        )
        solver.init(domain, 3)
        results.append(
            solver.solve(
                forcing,
                coefficient,
                lambda xb: jnp.zeros(xb.shape[0]),
                lambda xb: jnp.ones(xb.shape[0]),
                lambda alpha, beta, normals, xb: exact(xb),
            )
        )

    truth = exact(domain.get_int_bdry_nodes())
    assert jnp.max(jnp.abs(results[0]["u"] - truth)) < 3.0e-1
    assert jnp.max(jnp.abs(results[1]["u"] - truth)) < 3.0e-1
    assert jnp.max(jnp.abs(results[1]["u"] - results[0]["u"])) < 1.0e-6


def test_nonlinear_variable_poisson_solver_converges():
    domain = build_test_domain()
    exact = lambda x: x[:, 0] ** 2 + x[:, 1] ** 2
    coefficient = lambda x, u: 1.0 + 0.1 * u
    coefficient_u = lambda x, u: 0.1 * jnp.ones(x.shape[0])
    forcing = lambda x, u: -4.0 - 0.8 * u
    forcing_u = lambda x, u: -0.8 * jnp.ones(x.shape[0])

    solver = solvers.NonlinearVariablePoissonSolver(
        lap_assembler="fd",
        bc_assembler="fd",
        lap_stencil="wls",
        bc_stencil="wls",
    )
    solver.init(domain, 3)
    initial = jnp.zeros(domain.get_num_int_bdry_nodes())
    solver.prepare_preconditioner(
        forcing_u,
        coefficient,
        lambda xb: jnp.zeros(xb.shape[0]),
        lambda xb: jnp.ones(xb.shape[0]),
        initial_guess=initial,
    )
    result = solver.solve(
        forcing,
        forcing_u,
        coefficient,
        coefficient_u,
        lambda xb: jnp.zeros(xb.shape[0]),
        lambda xb: jnp.ones(xb.shape[0]),
        lambda alpha, beta, normals, xb: exact(xb),
        initial_guess=initial,
    )
    truth = exact(domain.get_int_bdry_nodes())
    assert jnp.max(jnp.abs(result["u"] - truth)) < 4.0e-1
    assert result["nonlinear_iterations"] > 0
    assert result["residual_norm"] < 1.0e-5


def test_diffusion_solver_reaches_bdf3():
    domain = build_test_domain()
    diffusivity = 0.25
    dt = 0.02
    points = domain.get_int_bdry_nodes()
    exact = lambda time, x: jnp.exp(-time) * (x[:, 0] ** 2 + x[:, 1] ** 2)
    forcing = lambda nu, time, x: (
        -jnp.exp(-time) * (x[:, 0] ** 2 + x[:, 1] ** 2)
        - 4.0 * nu * jnp.exp(-time)
    )
    boundary = lambda alpha, beta, normals, time, xb: exact(time, xb)

    solver = solvers.DiffusionSolver(
        lap_assembler="fd",
        bc_assembler="fd",
        lap_stencil="wls",
        bc_stencil="wls",
    )
    solver.init(domain, 3, dt, diffusivity)
    solver.set_initial_state(exact(0.0, points))
    zero = lambda xb: jnp.zeros(xb.shape[0])
    one = lambda xb: jnp.ones(xb.shape[0])
    u1 = solver.bdf1_step(dt, forcing, zero, one, boundary)
    u2 = solver.bdf2_step(2.0 * dt, forcing, zero, one, boundary)
    u3 = solver.bdf3_step(3.0 * dt, forcing, zero, one, boundary)
    assert jnp.max(jnp.abs(u1 - exact(dt, points))) < 3.0e-1
    assert jnp.max(jnp.abs(u2 - exact(2.0 * dt, points))) < 3.5e-1
    assert jnp.max(jnp.abs(u3 - exact(3.0 * dt, points))) < 4.0e-1


def test_pu_diffusion_and_multispecies_wrappers_advance():
    domain = build_test_domain()
    points = jnp.asarray(domain.get_int_bdry_nodes())

    scalar = solvers.PUDiffusionSolver()
    scalar.init(domain, 4, 0.01, 0.05)
    scalar.set_initial_state(jnp.sin(points[:, 0]))
    scalar_out = scalar.bdf1_step(
        0.01,
        lambda nu, time, x: jnp.zeros(x.shape[0]),
        lambda xb: jnp.zeros(xb.shape[0]),
        lambda xb: jnp.ones(xb.shape[0]),
        lambda alpha, beta, normals, time, xb: jnp.zeros(xb.shape[0]),
    )
    assert scalar_out.shape == (points.shape[0],)

    multi = solvers.MultiSpeciesPUDiffusionSolver()
    multi.init(domain, 4, 0.01, 0.05)
    initial = jnp.column_stack([jnp.sin(points[:, 0]), jnp.cos(points[:, 1])])
    multi.set_initial_state(initial)
    multi_out = multi.bdf1_step(
        0.01,
        lambda nu, time, x: jnp.zeros((x.shape[0], 2)),
        lambda time, xb: jnp.ones(xb.shape[0]),
        lambda time, xb: jnp.zeros(xb.shape[0]),
        lambda alpha, beta, normals, time, xb: jnp.zeros((xb.shape[0], 2)),
    )
    assert multi_out.shape == initial.shape
    assert jnp.all(jnp.isfinite(multi_out))
