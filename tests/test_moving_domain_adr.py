import jax.numpy as jnp

from kernelpack import geometry, nodes, solvers
from kernelpack.domain import DomainDescriptor
from kernelpack.solvers._common import SparseCOOMatrix
from kernelpack.solvers.moving_domain_adr import _paper_preconditioner_diagonal


def _closed_curve(center, radius, seed_count, h, *, build_model=True):
    theta = jnp.arange(seed_count) * (2.0 * jnp.pi / seed_count)
    sites = jnp.asarray(center) + radius * jnp.column_stack([jnp.cos(theta), jnp.sin(theta)])
    surface = geometry.EmbeddedSurface()
    surface.set_data_sites(sites)
    if build_model:
        sample_count = max(8, int(jnp.ceil(2.0 * jnp.pi * radius / h)))
        surface.build_closed_geometric_model_ps(
            2,
            h,
            seed_count,
            sample_count,
            method=1,
            supersample_fac=2,
        )
        surface.build_level_set_from_geometric_model()
    return surface


def _exact_solution(t, points):
    return jnp.exp(-t) * (2.0 + points[:, 0])


def test_moving_adr_schur_diagonal_matches_block_formula():
    dense = jnp.asarray(
        [
            [2.0, 0.0, 1.0, 0.0],
            [0.0, 4.0, 0.0, 2.0],
            [3.0, 0.0, 7.0, 0.0],
            [0.0, 5.0, 0.0, 11.0],
        ]
    )
    rows, cols = jnp.nonzero(dense)
    system = SparseCOOMatrix(
        indices=jnp.column_stack([rows, cols]),
        values=dense[rows, cols],
        shape=dense.shape,
    )
    diagonal = _paper_preconditioner_diagonal(system, n_physical=2)
    assert jnp.allclose(diagonal, jnp.asarray([2.0, 4.0, 5.5, 8.5]))


def test_moving_boundary_sbf_model_preserves_3d_parameter_correspondence():
    center = jnp.array([0.1, -0.2, 0.05])
    seeds = center + geometry.fibonacci_sphere(160)
    model = geometry.MovingBoundarySBFModel(seeds)
    surface = model.evaluate(seeds, 0.18)
    samples = surface.get_uniform_sample_sites()
    normals = surface.get_uniform_nrmls()
    radial = samples - center
    radius_error = jnp.max(jnp.abs(jnp.linalg.norm(radial, axis=1) - 1.0))
    normal_alignment = jnp.min(
        jnp.sum(radial / jnp.linalg.norm(radial, axis=1, keepdims=True) * normals, axis=1)
    )
    assert float(radius_error) < 3e-2
    assert float(normal_alignment) > 0.98

    parameter_sites = model.evaluation_parameter_sites.copy()
    deformed = seeds.at[:, 0].set(1.15 * (seeds[:, 0] - center[0]) + center[0])
    model.evaluate(deformed, 0.18)
    assert bool(jnp.array_equal(parameter_sites, model.evaluation_parameter_sites))


def test_moving_domain_adr_manufactured_solution_and_selective_updates():
    h = 0.16
    dt = 0.01
    nu = 0.03
    reaction = -0.3
    outer = _closed_curve([0.0, 0.0], 1.0, 120, h)
    background = nodes.DomainNodeGenerator().build_domain_descriptor_from_geometry(
        outer,
        h,
        seed=29,
        strip_count=1,
        do_outer_refinement=True,
        outer_fraction_of_h=0.5,
        outer_refinement_zone_size_as_multiple_of_h=2.0,
    )
    hole = _closed_curve([0.35, 0.0], 0.18, 48, h, build_model=False)

    solver = solvers.MovingDomainADRSolver(gmres_tolerance=1e-9, linear_solver="gmres")
    solver.init(background, [hole], 2, dt, nu)
    solver.set_initial_state(_exact_solution)
    discretization_shape = solver.moving_domain.get_discretization_domain().get_all_nodes().shape
    assert discretization_shape[0] > solver.get_current_domain().get_num_total_nodes()

    velocity = lambda t, points: jnp.column_stack([-points[:, 1], points[:, 0]])

    def forcing(nu_value, t, points):
        del nu_value
        exact = _exact_solution(t, points)
        return -exact - jnp.exp(-t) * points[:, 1] - reaction * exact

    numerical = jnp.zeros((0,))
    for step in range(1, 4):
        numerical = solver.step(
            step * dt,
            velocity,
            forcing,
            0.0,
            1.0,
            lambda alpha, beta, normals, t, points: _exact_solution(t, points),
            reaction,
        )
        assert solver.moving_domain.get_discretization_domain().get_all_nodes().shape == discretization_shape
        assert numerical.shape == solver.get_output_nodes().shape[:1]

    truth = _exact_solution(3.0 * dt, solver.get_output_nodes())
    relative_error = jnp.linalg.norm(numerical - truth) / jnp.linalg.norm(truth)
    assert float(relative_error) < 5e-3
    assert solver.completed_steps == 3
    assert solver.last_linear_diagnostics["info"] == 0
    assert not solver.last_linear_diagnostics["used_dense_fallback"]
    assert solver.last_linear_diagnostics["relative_residual"] < 1e-8
    assert solver.get_current_domain().get_num_bdry_nodes() > background.get_num_bdry_nodes()
    assert solver.last_update_diagnostics["laplacian_rows_reused"] > 0
    assert solver.last_update_diagnostics["interpolation_rows_reused"] > 0
    assert (
        solver.last_update_diagnostics["interpolation_rows_recomputed"]
        < solver.moving_domain.get_discretization_domain().get_num_int_bdry_nodes()
    )


def test_moving_domain_adr_runs_one_3d_step():
    coordinates = jnp.linspace(-1.0, 1.0, 5)
    points = jnp.asarray(
        [[x, y, z] for x in coordinates for y in coordinates for z in coordinates],
        dtype=float,
    )
    boundary_mask = jnp.any(jnp.isclose(jnp.abs(points), 1.0), axis=1)
    boundary = points[boundary_mask]
    interior = points[~boundary_mask]
    normals = boundary / jnp.linalg.norm(boundary, axis=1, keepdims=True)
    background = DomainDescriptor()
    background.set_nodes(interior, boundary, boundary + 0.25 * normals)
    background.set_normals(normals)
    background.set_sep_rad(0.5)
    background.build_structs()

    hole = geometry.EmbeddedSurface()
    hole.set_data_sites(jnp.array([0.2, 0.0, 0.0]) + 0.3 * geometry.fibonacci_sphere(60))
    reaction = -0.2
    solver = solvers.MovingDomainADRSolver(gmres_tolerance=1e-8)
    solver.init(background, [hole], xi=2, dt=0.005, diffusivity=0.01)
    solver.set_initial_state(_exact_solution)
    velocity = lambda t, x: jnp.column_stack([-x[:, 1], x[:, 0], jnp.zeros(x.shape[0])])

    def forcing(nu, t, x):
        del nu
        exact = _exact_solution(t, x)
        return -exact - jnp.exp(-t) * x[:, 1] - reaction * exact

    numerical = solver.step(
        0.005,
        velocity,
        forcing,
        0.0,
        1.0,
        lambda alpha, beta, nr, t, x: _exact_solution(t, x),
        reaction,
    )
    truth = _exact_solution(0.005, solver.get_output_nodes())
    relative_error = jnp.linalg.norm(numerical - truth) / jnp.linalg.norm(truth)
    assert float(relative_error) < 2e-2
    assert solver.last_linear_diagnostics["info"] == 0
    assert not solver.last_linear_diagnostics["used_dense_fallback"]
