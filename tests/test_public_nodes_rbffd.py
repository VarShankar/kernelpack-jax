import jax
import jax.numpy as jnp
import numpy as np

from kernelpack import domain, geometry, nodes, rbffd


def with_infinite_diagonal(matrix):
    matrix = jnp.asarray(matrix)
    return matrix.at[jnp.diag_indices(matrix.shape[0])].set(jnp.inf)


def test_masked_knn_excludes_inactive_capacity_slots():
    descriptor = domain.DomainDescriptor()
    descriptor.set_nodes(
        jnp.asarray([[0.0, 0.0], [9.9, 0.0], [1.0, 0.0]]),
        jnp.zeros((0, 2)),
    )
    descriptor.set_active_masks(
        jnp.asarray([True, False, True]),
        jnp.zeros((0,), dtype=bool),
        jnp.zeros((0,), dtype=bool),
    )
    descriptor.build_structs()
    indices, _ = descriptor.query_knn(
        "all", jnp.asarray([[10.0, 0.0]]), 2, backend="jax"
    )
    assert set(np.asarray(indices[0]).tolist()) == {0, 2}


def test_seeded_poisson_sampling_is_deterministic_and_separated():
    first, info = nodes.generate_poisson_nodes_in_box(
        0.2, [0.0, 0.0], [1.0, 1.0], seed=19, strip_count=2
    )
    second, _ = nodes.generate_poisson_nodes_in_box(
        0.2, [0.0, 0.0], [1.0, 1.0], seed=19, strip_count=2
    )
    assert jnp.array_equal(first, second)
    assert info["deterministic"]
    distances = with_infinite_diagonal(geometry.distance_matrix(first, first))
    assert distances.min() >= 0.2 * (1.0 - 1.0e-10)


def test_geometry_clipping_keeps_interior_nodes_inside_level_set():
    t = jnp.linspace(0.0, 2.0 * jnp.pi, 40, endpoint=False)
    surface = geometry.EmbeddedSurface()
    surface.set_data_sites(jnp.column_stack([jnp.cos(t), 0.7 * jnp.sin(t)]))
    surface.build_closed_geometric_model_ps(2, 0.1, t.size)
    surface.build_level_set_from_geometric_model()

    generator = nodes.DomainNodeGenerator()
    generator.generate_interior_nodes_from_geometry(
        surface, 0.2, seed=29, strip_count=2
    )
    interior = generator.get_interior_nodes()
    assert interior.shape[0] <= generator.get_raw_poisson_interior_nodes().shape[0]
    assert jnp.all(surface.get_level_set().evaluate(interior) >= 0.2 - 1.0e-10)


def test_curl_free_level_set_gradient_matches_autodiff():
    t = jnp.linspace(0.0, 2.0 * jnp.pi, 36, endpoint=False)
    sites = jnp.column_stack([jnp.cos(t), jnp.sin(t)])
    level_set = geometry.RBFLevelSet()
    level_set.build_level_set_from_cfi(sites, sites)
    points = jnp.array([[0.2, 0.1], [0.5, -0.25], [-0.35, 0.4]])
    analytic = level_set.evaluate_gradient(points)
    differentiated = jax.vmap(
        jax.grad(lambda point: level_set.evaluate(point[None, :])[0])
    )(points)
    assert jnp.allclose(analytic, differentiated, rtol=1.0e-5, atol=1.0e-5)


def test_wls_laplacian_reproduces_quadratics():
    xg, yg = jnp.meshgrid(
        jnp.linspace(-1.0, 1.0, 5),
        jnp.linspace(-1.0, 1.0, 5),
        indexing="ij",
    )
    points = jnp.column_stack([xg.ravel(), yg.ravel()])
    interior = (jnp.abs(points[:, 0]) < 0.999) & (
        jnp.abs(points[:, 1]) < 0.999
    )
    active_rows = jnp.flatnonzero(interior) + 1

    descriptor = domain.DomainDescriptor()
    descriptor.set_nodes(points, jnp.zeros((0, 2)), jnp.zeros((0, 2)))
    descriptor.set_sep_rad(0.5)
    descriptor.build_structs()
    stencil = rbffd.StencilProperties(
        n=9,
        dim=2,
        ell=2,
        spline_degree=3,
        tree_mode="interior_boundary",
        point_set="interior_boundary",
    )
    assembler = rbffd.FDDiffOp(lambda: rbffd.WeightedLeastSquaresStencil())
    assembler.assemble_op(
        descriptor,
        "lap",
        stencil,
        rbffd.OpProperties(record_stencils=True),
        active_rows=active_rows,
    )
    values = points[:, 0] ** 2 + points[:, 1] ** 2
    laplacian = assembler.get_op() @ values
    assert jnp.all(jnp.abs(laplacian[active_rows - 1] - 4.0) < 1.0e-8)
