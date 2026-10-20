from __future__ import annotations

import jax
from jax import numpy as jnp


DTYPE = jnp.float64
jax.config.update("jax_enable_x64", True)


def triangle_quad_rule(
    n: int,
    quadrature_fn,
    triangle: jax.Array = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.5, jnp.sqrt(3.0) / 2.0]]),
) -> tuple[jax.Array, jax.Array]:
    def quad_rule_2d(order: int, quadrature):
        quad_nodes, quad_weights = quadrature(order)
        a, b = -1.0, 1.0
        c, d = 0.0, 1.0
        t, w = quad_nodes, quad_weights
        t = (((t - a) * (d - c)) / (b - a)) + c
        det_j = (d - c) / (b - a)
        w = w * det_j
        t = jnp.array(jnp.meshgrid(*[t] * 2))
        t = t.reshape(len(t), -1).T
        w = jnp.outer(w, w).flatten()[:, None]
        return t, w

    quad_rule = quad_rule_2d(n, quadrature_fn)

    def coord_square_to_quadrilateral(x, quadrilateral):
        x1, x2, x3, x4 = quadrilateral
        xi, eta = x
        psi_1 = lambda xi_val, eta_val: (1.0 - xi_val) * (1.0 - eta_val)
        psi_2 = lambda xi_val, eta_val: xi_val * (1.0 - eta_val)
        psi_3 = lambda xi_val, eta_val: xi_val * eta_val
        psi_4 = lambda xi_val, eta_val: (1.0 - xi_val) * eta_val
        return x1 * psi_1(xi, eta) + x2 * psi_2(xi, eta) + x3 * psi_3(xi, eta) + x4 * psi_4(xi, eta)

    def detj_square_to_quadrilateral(x, quadrilateral):
        jacobian = jax.jacfwd(coord_square_to_quadrilateral, argnums=0)(x, quadrilateral)
        return jnp.linalg.det(jacobian)

    def quad_rule_square_to_quadrilateral(rule, quadrilateral):
        t, w = rule
        updated_w = jax.vmap(lambda weight, coord, quad: weight * detj_square_to_quadrilateral(coord, quad), in_axes=[0, 0, None])(w, t, quadrilateral)
        updated_t = jax.vmap(coord_square_to_quadrilateral, in_axes=[0, None])(t, quadrilateral)
        return updated_t, updated_w

    a_pt, b_pt, c_pt = triangle
    centroid = (a_pt + b_pt + c_pt) / 3.0
    d_pt = (a_pt + b_pt) / 2.0
    e_pt = (b_pt + c_pt) / 2.0
    f_pt = (a_pt + c_pt) / 2.0
    quadrilaterals = jnp.array([[a_pt, d_pt, centroid, f_pt], [b_pt, e_pt, centroid, d_pt], [c_pt, f_pt, centroid, e_pt]])

    triangle_quad_t, triangle_quad_w = jax.vmap(quad_rule_square_to_quadrilateral, in_axes=[None, 0])(quad_rule, quadrilaterals)
    return triangle_quad_t.reshape(-1, 2).astype(DTYPE), triangle_quad_w.flatten()[:, None].astype(DTYPE)
