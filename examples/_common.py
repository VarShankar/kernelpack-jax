"""Shared domain and manufactured fields for the public PDE examples."""

import jax.numpy as jnp

from kernelpack.domain import DomainDescriptor


def build_square_domain(grid_size: int = 11) -> DomainDescriptor:
    coordinates = jnp.linspace(-1.0, 1.0, grid_size)
    points = jnp.array([[x, y] for x in coordinates for y in coordinates])
    boundary_mask = jnp.any(jnp.isclose(jnp.abs(points), 1.0), axis=1)
    boundary = points[boundary_mask]
    interior = points[~boundary_mask]
    normals = jnp.where(jnp.isclose(jnp.abs(boundary), 1.0), jnp.sign(boundary), 0.0)
    normals /= jnp.linalg.norm(normals, axis=1, keepdims=True)
    ghosts = boundary + 0.25 * normals

    domain = DomainDescriptor()
    domain.set_nodes(interior, boundary, ghosts)
    domain.set_normals(normals)
    domain.set_sep_rad(float(coordinates[1] - coordinates[0]))
    domain.build_structs()
    return domain


def exact_solution(time: float, points: jnp.ndarray) -> jnp.ndarray:
    return jnp.exp(-time) * (points[:, 0] ** 2 + points[:, 1] ** 2)


def zero_neumann(points: jnp.ndarray) -> jnp.ndarray:
    return jnp.zeros(points.shape[0])


def unit_dirichlet(points: jnp.ndarray) -> jnp.ndarray:
    return jnp.ones(points.shape[0])
