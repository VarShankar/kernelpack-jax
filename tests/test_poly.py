import jax.numpy as jnp

from kernelpack import poly


def test_poly_basics():
    a, b = poly.jacobi_recurrence(5, 0, 0)
    al, bl = poly.legendre_recurrence(5)
    x = jnp.array([-1, -0.5, 0, 0.5, 1], dtype=float)
    p = poly.poly_eval(a, b, x, 3)
    dp = poly.poly_eval(a, b, x, 3, 1)
    assert jnp.allclose(a, al)
    assert jnp.allclose(b, bl)
    assert jnp.allclose(p[:, 0], 1 / jnp.sqrt(2.0))
    assert jnp.allclose(p[:, 1], jnp.sqrt(3.0 / 2.0) * x)
    assert jnp.allclose(dp[:, 1], jnp.sqrt(3.0 / 2.0))


def test_indices_and_basis():
    td = poly.total_degree_indices(2, 2)
    assert jnp.array_equal(td, jnp.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]]))
    hc = poly.hyperbolic_cross_indices(2, 3)
    assert any(jnp.all(row == jnp.array([0, 0])) for row in hc)
    assert any(jnp.all(row == jnp.array([1, 1])) for row in hc)
    assert not any(jnp.all(row == jnp.array([3, 3])) for row in hc)
    basis = poly.PolynomialBasis.from_total_degree(2, 2)
    basis.fit_normalization_from_points(jnp.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float))
    assert jnp.linalg.norm(basis.center) < 1e-12
    assert abs(basis.scale - 1) < 1e-12


def test_batched_basis_evaluation_matches_pointwise():
    basis = poly.PolynomialBasis.from_total_degree(2, 3)
    x = jnp.array(
        [
            [[-0.5, -0.25], [0.0, 0.1], [0.5, 0.25]],
            [[-0.25, 0.5], [0.25, -0.5], [0.75, 0.0]],
        ],
        dtype=float,
    )
    d = jnp.array([[0, 0], [1, 0], [0, 1]], dtype=int)
    batched = basis.evaluate(x, d, assume_normalized=True)
    pointwise = jnp.stack([basis.evaluate(row, d, assume_normalized=True) for row in x], axis=0)
    assert jnp.allclose(batched, pointwise)
