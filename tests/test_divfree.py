import jax
import jax.numpy as jnp

from kernelpack import divfree


def _rotational_field(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.column_stack((-x[:, 1], x[:, 0]))


def test_divfree_global_interpolant_reproduces_nodal_values_under_jit():
    t = jnp.linspace(0.0, 2.0 * jnp.pi, 18, endpoint=False)
    x = jnp.column_stack((0.85 * jnp.cos(t), 0.65 * jnp.sin(t)))
    u = _rotational_field(x)

    fit = jax.jit(lambda x, u: divfree.fit_divfree_phs(x, u, poly_degree=1, phs_degree=5))
    evaluate = jax.jit(lambda model, xq: divfree.evaluate_divfree_phs(model, xq))
    interp = fit(x, u)
    uhat = evaluate(interp, x)

    assert float(jnp.max(jnp.abs(uhat - u))) < 1e-8


def test_divfree_polynomial_basis_is_jittable():
    x = jnp.array(
        [
            [-0.8, -0.1],
            [-0.2, 0.4],
            [0.3, -0.5],
            [0.9, 0.2],
        ],
        dtype=float,
    )
    basis, center, scale, alpha, transform = jax.jit(lambda pts: divfree.df_poly_basis_from_legendre(pts, poly_degree=1))(x)

    assert basis.shape[0] == 2 * x.shape[0]
    assert basis.shape[1] == transform.shape[1]
    assert transform.shape[0] == alpha.shape[0]
    assert center.shape == (2,)
    assert scale.shape == ()


def test_local_divfree_interpolator_reproduces_nodal_values_under_jit():
    t = jnp.linspace(0.0, 2.0 * jnp.pi, 24, endpoint=False)
    x = jnp.column_stack((jnp.cos(t), 0.75 * jnp.sin(t)))
    u = _rotational_field(x)

    fit = jax.jit(lambda x, u: divfree.fit_local_divfree(x, u, poly_degree=1, phs_degree=5, stencil_size=8))
    evaluate = jax.jit(lambda model, xq: divfree.evaluate_local_divfree(model, xq))
    interp = fit(x, u)
    uhat = evaluate(interp, x)

    assert float(jnp.max(jnp.abs(uhat - u))) < 5e-7


def test_local_divfree_accepts_precomputed_stencils_from_shared_backend():
    t = jnp.linspace(0.0, 2.0 * jnp.pi, 20, endpoint=False)
    x = jnp.column_stack((jnp.cos(t), 0.75 * jnp.sin(t)))
    u = _rotational_field(x)
    center_indices, stencil_indices = divfree.build_local_divfree_stencil_indices(x, 8, backend="cpu")

    fit = jax.jit(
        lambda x, u, centers, stencils: divfree.fit_local_divfree(
            x,
            u,
            poly_degree=1,
            phs_degree=5,
            stencil_size=8,
            center_indices=centers,
            stencil_indices=stencils,
        )
    )
    interp = fit(x, u, center_indices, stencil_indices)
    uhat = divfree.evaluate_local_divfree(interp, x)

    assert float(jnp.max(jnp.abs(uhat - u))) < 5e-7


def test_divfree_evaluation_vmappable():
    t = jnp.linspace(0.0, 2.0 * jnp.pi, 18, endpoint=False)
    x = jnp.column_stack((0.85 * jnp.cos(t), 0.65 * jnp.sin(t)))
    u = _rotational_field(x)
    interp = divfree.fit_divfree_phs(x, u, poly_degree=1, phs_degree=5)
    q = jnp.array([[-0.3, 0.2], [0.1, -0.4], [0.7, 0.0]], dtype=float)

    batched = divfree.evaluate_divfree_phs(interp, q)
    mapped = jax.vmap(lambda point: divfree.evaluate_divfree_phs(interp, point[None, :])[0])(q)

    assert jnp.allclose(batched, mapped, atol=1e-10, rtol=1e-10)


def test_divfree_fit_is_differentiable_through_values():
    t = jnp.linspace(0.0, 2.0 * jnp.pi, 14, endpoint=False)
    x = jnp.column_stack((0.8 * jnp.cos(t), 0.6 * jnp.sin(t)))
    u = _rotational_field(x)
    q = jnp.array([[-0.2, 0.1], [0.3, -0.25]], dtype=float)

    def loss(values):
        model = divfree.fit_divfree_phs(x, values, poly_degree=1, phs_degree=5)
        return jnp.sum(divfree.evaluate_divfree_phs(model, q) ** 2)

    grad = jax.jit(jax.grad(loss))(u)

    assert grad.shape == u.shape
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_local_divfree_interpolator_smoke_3d_under_jit():
    pts = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    u = jnp.column_stack((-pts[:, 1], pts[:, 0], jnp.zeros(pts.shape[0])))
    q = jnp.array([[0.25, 0.25, 0.25], [0.5, 0.1, 0.75]], dtype=float)

    interp = jax.jit(lambda x, u: divfree.fit_local_divfree(x, u, poly_degree=1, phs_degree=5, stencil_size=8))(pts, u)
    uq = jax.jit(lambda model, xq: divfree.evaluate_local_divfree(model, xq))(interp, q)

    assert uq.shape == q.shape
    assert bool(jnp.all(jnp.isfinite(uq)))
