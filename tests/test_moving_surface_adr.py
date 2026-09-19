import jax
import jax.numpy as jnp
import numpy as np

from kernelpack import geometry, manifold, solvers


def _sphere_operators(node_count=160, xi=2):
    points = geometry.fibonacci_sphere(node_count)
    ell = manifold.surface_polynomial_degree(xi)
    neighbors = manifold.build_surface_stencil_graph(
        points,
        manifold.surface_stencil_size(ell),
        backend="jax",
    )
    operators, cache = manifold.assemble_tangent_plane_operators(
        points,
        points,
        neighbors,
        xi=xi,
    )
    return points, neighbors, operators, cache


def _write_synthetic_ibamr_trajectory(folder, points):
    count = points.shape[0]
    node_ids = np.arange(count)
    np.savetxt(
        folder / "material.csv",
        np.column_stack((node_ids, points)),
        delimiter=",",
        header="node_id,u_x,u_y,u_z",
        comments="",
    )
    np.savetxt(
        folder / "faces.csv",
        np.array([[0, 1, 2]]),
        delimiter=",",
        header="i,j,k",
        comments="",
        fmt="%d",
    )
    np.savetxt(
        folder / "diagnostics.csv",
        np.array([[0, 0.0], [1, 0.01]]),
        delimiter=",",
        header="step,time",
        comments="",
    )
    for step, radius in ((0, 1.0), (1, 1.001)):
        positions = radius * points
        velocities = 0.1 * points
        np.savetxt(
            folder / f"frame_{step:06d}.csv",
            np.column_stack((node_ids, positions, velocities)),
            delimiter=",",
            header="node_id,x,y,z,u,v,w",
            comments="",
        )


def test_surface_degree_and_stencil_rules_match_matlab():
    assert [manifold.surface_polynomial_degree(xi) for xi in (2, 4, 6)] == [3, 5, 7]
    assert [manifold.surface_phs_degree(ell) for ell in (3, 5, 7)] == [5, 5, 7]
    assert [manifold.surface_stencil_size(ell) for ell in (3, 5, 7)] == [21, 43, 73]


def test_tangent_plane_operators_reproduce_sphere_coordinates_and_scale():
    points, neighbors, operators, _ = _sphere_operators()
    numerical_laplacian = manifold.apply_surface_operator(
        operators.laplacian_weights,
        operators.neighbors,
        points,
    )
    laplacian_error = jnp.linalg.norm(numerical_laplacian + 2.0 * points) / jnp.linalg.norm(
        2.0 * points
    )
    assert float(laplacian_error) < 1.5e-2

    numerical_gradient = jax.vmap(
        lambda values: manifold.surface_gradient(operators, values),
        in_axes=1,
        out_axes=1,
    )(points)
    projector = jnp.eye(3)[None, :, :] - points[:, :, None] * points[:, None, :]
    gradient_error = jnp.linalg.norm(numerical_gradient - projector) / jnp.linalg.norm(projector)
    assert float(gradient_error) < 2.5e-2

    scale = 0.3
    scaled_operators, _ = manifold.assemble_tangent_plane_operators(
        scale * points,
        points,
        neighbors,
        xi=2,
    )
    laplacian_scale_error = jnp.linalg.norm(
        scale**2 * scaled_operators.laplacian_weights - operators.laplacian_weights
    ) / jnp.linalg.norm(operators.laplacian_weights)
    gradient_scale_error = jnp.linalg.norm(
        scale * scaled_operators.gradient_weights - operators.gradient_weights
    ) / jnp.linalg.norm(operators.gradient_weights)
    assert float(laplacian_scale_error) < 1.0e-9
    assert float(gradient_scale_error) < 1.0e-9


def test_surface_operator_update_uses_defect_correction_without_refactor():
    points, neighbors, _, cache = _sphere_operators()
    deformed = points.at[:, 0].multiply(1.0001)
    normals = deformed / jnp.linalg.norm(deformed, axis=1, keepdims=True)
    operators, _, info = manifold.update_tangent_plane_operators(
        deformed,
        normals,
        neighbors,
        cache,
        xi=2,
        tolerance=1.0e-8,
        max_defect_iterations=4,
    )
    assert not bool(info.refactored)
    assert float(jnp.max(info.relative_residual)) <= 1.0e-8
    assert int(jnp.max(info.defect_iterations)) <= 4
    assert bool(jnp.all(jnp.isfinite(operators.laplacian_weights)))


def test_surface_hyperviscosity_uses_automatic_q_based_power():
    points, _, operators, _ = _sphere_operators(node_count=96)
    calibration = manifold.calibrate_surface_hyperviscosity(
        operators,
        points,
        points,
        jnp.sqrt(1.0 / points.shape[0]),
        target_order=2,
        arnoldi_iterations=8,
    )
    expected_power = int(jnp.ceil((2 + jnp.max(calibration.growth_exponents)) / 2.0))
    assert calibration.power == expected_power
    assert 2 * calibration.power - float(jnp.max(calibration.growth_exponents)) >= 2
    assert bool(jnp.all(jnp.isfinite(calibration.gamma)))
    assert bool(jnp.all(jnp.isfinite(calibration.tau)))
    assert float(calibration.eta_mean) > 0.0


def test_spherical_sbf_geometry_updates_normals_and_quadrature_on_device():
    controls = geometry.fibonacci_sphere(64)
    evaluation_sites = geometry.fibonacci_sphere(160)
    model = manifold.build_spherical_sbf_model(controls, evaluation_sites)
    surface = jax.jit(manifold.evaluate_spherical_sbf_geometry)(model, controls)
    radius = jnp.linalg.norm(surface.points, axis=1)
    radial_normals = surface.points / radius[:, None]
    assert float(jnp.max(jnp.abs(radius - 1.0))) < 1.0e-4
    assert float(jnp.min(jnp.sum(radial_normals * surface.normals, axis=1))) > 0.999
    assert abs(float(jnp.sum(surface.quadrature_weights)) - 4.0 * jnp.pi) < 2.0e-4

    deformed_controls = controls.at[:, 0].multiply(1.2)
    deformed = jax.jit(manifold.evaluate_spherical_sbf_geometry)(model, deformed_controls)
    assert deformed.points.shape == (evaluation_sites.shape[0], 3)
    assert bool(jnp.all(jnp.isfinite(deformed.normals)))
    assert bool(jnp.all(deformed.quadrature_weights > 0.0))


def test_toroidal_rbf_geometry_matches_an_analytic_torus():
    n_theta = 12
    n_phi = 10
    theta, phi = jnp.meshgrid(
        jnp.linspace(0.0, 2.0 * jnp.pi, n_theta, endpoint=False),
        jnp.linspace(0.0, 2.0 * jnp.pi, n_phi, endpoint=False),
        indexing="ij",
    )
    parameter_sites = jnp.column_stack((theta.ravel(), phi.ravel()))
    major_radius = 2.0
    minor_radius = 0.5
    controls = jnp.column_stack(
        (
            (major_radius + minor_radius * jnp.cos(parameter_sites[:, 0]))
            * jnp.cos(parameter_sites[:, 1]),
            (major_radius + minor_radius * jnp.cos(parameter_sites[:, 0]))
            * jnp.sin(parameter_sites[:, 1]),
            minor_radius * jnp.sin(parameter_sites[:, 0]),
        )
    )
    model = manifold.build_toroidal_rbf_model(
        controls,
        parameter_sites,
        degree=7,
    )
    surface = manifold.evaluate_toroidal_rbf_geometry(model, controls)
    exact_normals = jnp.column_stack(
        (
            jnp.cos(parameter_sites[:, 0]) * jnp.cos(parameter_sites[:, 1]),
            jnp.cos(parameter_sites[:, 0]) * jnp.sin(parameter_sites[:, 1]),
            jnp.sin(parameter_sites[:, 0]),
        )
    )
    exact_area = 4.0 * jnp.pi**2 * major_radius * minor_radius
    relative_interpolation_error = jnp.linalg.norm(surface.points - controls) / jnp.linalg.norm(
        controls
    )
    assert float(relative_interpolation_error) < 1.0e-9
    assert float(jnp.min(jnp.sum(surface.normals * exact_normals, axis=1))) > 0.98
    assert abs(float(jnp.sum(surface.quadrature_weights)) - exact_area) / float(exact_area) < 0.02


def test_sbf_and_local_tangent_surface_transfers_are_accurate():
    points, _, _, _ = _sphere_operators()
    angle = 0.02
    query = points.at[:, 0].set(
        jnp.cos(angle) * points[:, 0] - jnp.sin(angle) * points[:, 1]
    )
    query = query.at[:, 1].set(
        jnp.sin(angle) * points[:, 0] + jnp.cos(angle) * points[:, 1]
    )
    ell = manifold.surface_polynomial_degree(2)
    cross_neighbors = manifold.build_cross_surface_stencil_graph(
        query,
        points,
        manifold.surface_stencil_size(ell),
        backend="jax",
    )
    weights = manifold.local_tangent_interpolation_weights(
        points,
        query,
        query,
        cross_neighbors,
        xi=2,
    )
    target = lambda sites: 2.0 + sites[:, 0] + 0.2 * sites[:, 1] * sites[:, 2]
    local_values = manifold.apply_local_tangent_interpolant(
        weights,
        cross_neighbors,
        target(points),
    )
    local_error = jnp.linalg.norm(local_values - target(query)) / jnp.linalg.norm(target(query))
    assert float(local_error) < 5.0e-5

    control_indices = manifold.farthest_point_subset(points, count=64)
    sbf_values = manifold.spherical_sbf_transfer(
        points,
        target(points),
        query,
        control_indices,
    )
    sbf_error = jnp.linalg.norm(sbf_values - target(query)) / jnp.linalg.norm(target(query))
    assert float(sbf_error) < 1.0e-3


def test_semi_lagrangian_backfill_and_quality_trigger_are_jitted():
    points = geometry.fibonacci_sphere(96)
    dt = 0.01

    def velocity(_time, sites):
        return jnp.column_stack((-sites[:, 1], sites[:, 0], jnp.zeros(sites.shape[0])))

    departures = solvers.semi_lagrangian_backfill_points(
        points,
        0.2,
        dt,
        velocity,
        history_levels=2,
    )

    def rotate(sites, angle):
        return jnp.column_stack(
            (
                jnp.cos(angle) * sites[:, 0] - jnp.sin(angle) * sites[:, 1],
                jnp.sin(angle) * sites[:, 0] + jnp.cos(angle) * sites[:, 1],
                sites[:, 2],
            )
        )

    assert float(jnp.max(jnp.abs(departures[0] - rotate(points, -dt)))) < 1.0e-8
    assert float(jnp.max(jnp.abs(departures[1] - rotate(points, -2.0 * dt)))) < 2.0e-8

    nearest = manifold.build_surface_stencil_graph(points, 2, backend="jax")
    quality = manifold.surface_point_quality(points, nearest)
    will_cross, predicted = manifold.predict_surface_quality_crossing(
        quality,
        quality + 0.1,
        quality + 0.15,
        1.0,
    )
    assert bool(will_cross)
    assert float(predicted) >= float(quality + 0.15)


def test_bdf3_surface_adr_step_is_jitted_and_vmappable():
    points, _, operators, _ = _sphere_operators(node_count=128)
    dt = 0.01
    final_time = 3.0 * dt
    diffusivity = 0.03
    reaction_rate = 0.2
    exact = lambda time: jnp.exp(-time) * (2.0 + points[:, 0])
    history = solvers.MovingSurfaceHistory(
        concentration=jnp.stack((exact(2.0 * dt), exact(dt), exact(0.0))),
        points=jnp.stack((points, points, points)),
    )
    forcing = (
        -(1.0 + reaction_rate) * exact(final_time)
        + 2.0 * diffusivity * jnp.exp(-final_time) * points[:, 0]
    )
    quadrature = jnp.full((points.shape[0],), 4.0 * jnp.pi / points.shape[0])
    target_mass = 8.0 * jnp.pi * jnp.exp(-final_time)

    step = lambda state, source, mass: solvers.moving_surface_adr_step(
        state,
        points,
        operators,
        source,
        quadrature,
        mass,
        dt,
        diffusivity,
        jnp.zeros(3),
        1.0e-10,
        order=3,
        hyperviscosity_power=0,
        reaction_history=reaction_rate * state.concentration,
    )
    updated, info = jax.jit(step)(history, forcing, target_mass)
    relative_error = jnp.linalg.norm(updated.concentration[0] - exact(final_time)) / jnp.linalg.norm(
        exact(final_time)
    )
    assert float(relative_error) < 1.0e-4
    assert int(info.linear_info) == 0
    assert float(info.relative_residual) < 1.0e-9
    assert float(jnp.abs(info.mass_after_projection - target_mass)) < 1.0e-12

    batched_history = solvers.MovingSurfaceHistory(
        concentration=jnp.stack((history.concentration, 0.5 * history.concentration)),
        points=jnp.stack((history.points, history.points)),
    )
    batched_forcing = jnp.stack((forcing, 0.5 * forcing))
    batched_mass = jnp.stack((target_mass, 0.5 * target_mass))
    batched_solution = jax.vmap(lambda state, source, mass: step(state, source, mass)[0])(
        batched_history,
        batched_forcing,
        batched_mass,
    )
    assert batched_solution.concentration.shape == (2, 3, points.shape[0])
    assert jnp.allclose(
        batched_solution.concentration[1, 0],
        0.5 * batched_solution.concentration[0, 0],
        rtol=1.0e-9,
        atol=1.0e-11,
    )


def test_bdf3_full_adr_balance_on_a_breathing_sphere():
    material_sites = geometry.fibonacci_sphere(160)
    dt = 0.01
    time = 3.0 * dt
    diffusivity = 0.03
    radius = lambda t: 1.0 + 0.1 * jnp.sin(t)
    radius_rate = lambda t: 0.1 * jnp.cos(t)
    points_at = lambda t: radius(t) * material_sites
    exact = lambda t: jnp.exp(-t) * (2.0 + material_sites[:, 0])
    points = points_at(time)
    normals = material_sites
    ell = manifold.surface_polynomial_degree(2)
    neighbors = manifold.build_surface_stencil_graph(
        points,
        manifold.surface_stencil_size(ell),
        backend="jax",
    )
    operators, _ = manifold.assemble_tangent_plane_operators(
        points,
        normals,
        neighbors,
        xi=2,
    )
    history = solvers.MovingSurfaceHistory(
        concentration=jnp.stack((exact(2.0 * dt), exact(dt), exact(0.0))),
        points=jnp.stack((points_at(2.0 * dt), points_at(dt), points_at(0.0))),
    )
    exact_now = exact(time)
    forcing = (
        -exact_now
        + 2.0 * radius_rate(time) / radius(time) * exact_now
        + 2.0 * diffusivity * jnp.exp(-time) * material_sites[:, 0] / radius(time) ** 2
    )
    quadrature = jnp.full(
        (material_sites.shape[0],),
        4.0 * jnp.pi * radius(time) ** 2 / material_sites.shape[0],
    )
    target_mass = 8.0 * jnp.pi * radius(time) ** 2 * jnp.exp(-time)
    updated, info = solvers.moving_surface_adr_step(
        history,
        points,
        operators,
        forcing,
        quadrature,
        target_mass,
        dt,
        diffusivity,
        jnp.zeros(3),
        1.0e-10,
        order=3,
        hyperviscosity_power=0,
    )
    relative_error = jnp.linalg.norm(updated.concentration[0] - exact_now) / jnp.linalg.norm(
        exact_now
    )
    assert float(relative_error) < 1.5e-4
    assert int(info.linear_info) == 0
    assert float(info.relative_residual) < 1.0e-9


def test_mean_curvature_flow_step_matches_sphere_radius_law():
    radius_zero = 1.4
    dt = 2.0e-4
    directions, _, unit_operators, _ = _sphere_operators(node_count=128)
    points = radius_zero * directions
    operators, _ = manifold.assemble_tangent_plane_operators(
        points,
        directions,
        unit_operators.neighbors,
        xi=2,
    )
    updated, info = solvers.mean_curvature_flow_step(
        points,
        directions,
        operators,
        dt,
    )
    exact_radius = jnp.sqrt(radius_zero**2 - 4.0 * dt)
    error = jnp.linalg.norm(jnp.linalg.norm(updated, axis=1) - exact_radius)
    error /= jnp.linalg.norm(jnp.full((points.shape[0],), exact_radius))
    assert float(error) < 5.0e-5
    assert int(info.linear_info) == 0
    assert float(info.relative_residual) < 1.0e-9


def test_ibamr_trajectory_reconstructs_geometry_and_velocity(tmp_path):
    points = np.asarray(geometry.fibonacci_sphere(64))
    _write_synthetic_ibamr_trajectory(tmp_path, points)
    trajectory = manifold.IBAMRSurfaceTrajectory(
        tmp_path, xi=2, control_point_count=points.shape[0]
    )
    surface, velocity = trajectory.geometry(1)
    assert surface.points.shape == points.shape
    assert velocity.shape == points.shape
    assert np.max(np.abs(np.linalg.norm(np.asarray(surface.normals), axis=1) - 1.0)) < 1.0e-10
    assert np.all(np.asarray(surface.quadrature_weights) > 0.0)
    assert np.all(np.isfinite(np.asarray(velocity)))


def test_adaptive_hyperviscosity_refresh_stays_on_device():
    points, _, operators, _ = _sphere_operators(node_count=96)
    calibration = manifold.calibrate_surface_hyperviscosity(
        operators,
        points,
        points,
        jnp.sqrt(1.0 / points.shape[0]),
        target_order=2,
        power=2,
        arnoldi_iterations=4,
    )
    predicted, drift, recalibrated = manifold.refresh_surface_hyperviscosity(
        operators,
        points,
        points,
        jnp.sqrt(1.0 / points.shape[0]),
        calibration,
        drift_threshold=jnp.inf,
        arnoldi_iterations=4,
    )
    assert not bool(recalibrated)
    assert jnp.all(jnp.isfinite(predicted.gamma))
    assert jnp.isfinite(drift)

    refreshed, _, recalibrated = manifold.refresh_surface_hyperviscosity(
        operators,
        points,
        points,
        jnp.sqrt(1.0 / points.shape[0]),
        calibration,
        drift_threshold=-1.0,
        arnoldi_iterations=4,
    )
    assert bool(recalibrated)
    assert jnp.all(jnp.isfinite(refreshed.gamma))
