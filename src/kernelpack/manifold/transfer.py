from __future__ import annotations

from functools import partial

from jax import jit, lax, vmap
import jax.numpy as jnp

from kernelpack.poly import PolynomialBasis
from kernelpack.poly.core import _tensor_evaluate_with_recurrence

from .core import _tangent_frame, surface_phs_degree, surface_polynomial_degree


def _basis_data(ell: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    basis = PolynomialBasis.from_total_degree(2, ell, family="legendre")
    return basis.index_set, basis.recurrence_a, basis.recurrence_b


@partial(jit, static_argnames=("phs_degree",))
def _target_centered_local_weights(
    stencil_points: jnp.ndarray,
    target_point: jnp.ndarray,
    target_normal: jnp.ndarray,
    index_set: jnp.ndarray,
    recurrence_a: jnp.ndarray,
    recurrence_b: jnp.ndarray,
    *,
    phs_degree: int,
) -> jnp.ndarray:
    frame = _tangent_frame(target_normal)
    local_points = (stencil_points - target_point) @ frame
    width = jnp.maximum(
        jnp.max(jnp.abs(local_points)),
        jnp.finfo(stencil_points.dtype).eps,
    )
    scaled_points = local_points / width
    pairwise_distance = jnp.linalg.norm(
        scaled_points[:, None, :] - scaled_points[None, :, :],
        axis=2,
    )
    epsilon = jnp.finfo(stencil_points.dtype).eps
    polynomial = _tensor_evaluate_with_recurrence(
        scaled_points,
        index_set,
        jnp.zeros((1, 2), dtype=int),
        recurrence_a,
        recurrence_b,
    )[:, :, 0]
    polynomial_count = index_set.shape[0]
    lhs = jnp.block(
        [
            [(pairwise_distance + epsilon) ** phs_degree, polynomial],
            [
                polynomial.T,
                jnp.zeros(
                    (polynomial_count, polynomial_count),
                    dtype=stencil_points.dtype,
                ),
            ],
        ]
    )
    query_radius = jnp.linalg.norm(scaled_points, axis=1)
    query_polynomial = _tensor_evaluate_with_recurrence(
        jnp.zeros((1, 2), dtype=stencil_points.dtype),
        index_set,
        jnp.zeros((1, 2), dtype=int),
        recurrence_a,
        recurrence_b,
    )[0, :, 0]
    rhs = jnp.concatenate(((query_radius + epsilon) ** phs_degree, query_polynomial))
    return jnp.linalg.solve(lhs, rhs)[: stencil_points.shape[0]]


@partial(jit, static_argnames=("phs_degree",))
def _local_tangent_interpolation_weights(
    source_points: jnp.ndarray,
    target_points: jnp.ndarray,
    target_normals: jnp.ndarray,
    cross_neighbors: jnp.ndarray,
    index_set: jnp.ndarray,
    recurrence_a: jnp.ndarray,
    recurrence_b: jnp.ndarray,
    *,
    phs_degree: int,
) -> jnp.ndarray:
    return vmap(
        lambda stencil, target, normal: _target_centered_local_weights(
            stencil,
            target,
            normal,
            index_set,
            recurrence_a,
            recurrence_b,
            phs_degree=phs_degree,
        )
    )(source_points[cross_neighbors], target_points, target_normals)


def local_tangent_interpolation_weights(
    source_points: jnp.ndarray,
    target_points: jnp.ndarray,
    target_normals: jnp.ndarray,
    cross_neighbors: jnp.ndarray,
    *,
    xi: int,
    theta: int = 2,
) -> jnp.ndarray:
    """Build MATLAB-equivalent target-centered local remap weights."""

    ell = surface_polynomial_degree(xi, theta)
    index_set, recurrence_a, recurrence_b = _basis_data(ell)
    return _local_tangent_interpolation_weights(
        jnp.asarray(source_points, dtype=float),
        jnp.asarray(target_points, dtype=float),
        jnp.asarray(target_normals, dtype=float),
        jnp.asarray(cross_neighbors, dtype=int),
        index_set,
        recurrence_a,
        recurrence_b,
        phs_degree=surface_phs_degree(ell),
    )


@jit
def apply_local_tangent_interpolant(
    weights: jnp.ndarray,
    cross_neighbors: jnp.ndarray,
    source_values: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.einsum("qk,qk...->q...", weights, source_values[cross_neighbors])


@partial(jit, static_argnames=("count",))
def farthest_point_subset(parameter_sites: jnp.ndarray, *, count: int) -> jnp.ndarray:
    """Select a deterministic quasi-uniform SBF control subset."""

    sites = parameter_sites / jnp.maximum(
        jnp.linalg.norm(parameter_sites, axis=1, keepdims=True),
        jnp.finfo(parameter_sites.dtype).eps,
    )
    first = jnp.argmax(jnp.sum((sites - jnp.mean(sites, axis=0)) ** 2, axis=1))
    indices = jnp.zeros((count,), dtype=int).at[0].set(first)
    distance = jnp.sum((sites - sites[first]) ** 2, axis=1)
    distance = distance.at[first].set(-jnp.inf)

    def select(index: int, state: tuple[jnp.ndarray, jnp.ndarray]):
        selected, minimum_distance = state
        next_index = jnp.argmax(minimum_distance)
        selected = selected.at[index].set(next_index)
        candidate_distance = jnp.sum((sites - sites[next_index]) ** 2, axis=1)
        minimum_distance = jnp.minimum(minimum_distance, candidate_distance)
        minimum_distance = minimum_distance.at[next_index].set(-jnp.inf)
        return selected, minimum_distance

    indices, _ = lax.fori_loop(1, count, select, (indices, distance))
    return jnp.sort(indices)


@partial(jit, static_argnames=("degree",))
def spherical_sbf_transfer(
    source_parameter_sites: jnp.ndarray,
    source_values: jnp.ndarray,
    target_parameter_sites: jnp.ndarray,
    control_indices: jnp.ndarray,
    *,
    degree: int = 7,
) -> jnp.ndarray:
    """Transfer a field with the degree-seven spherical PHS used by MATLAB."""

    source = source_parameter_sites / jnp.maximum(
        jnp.linalg.norm(source_parameter_sites, axis=1, keepdims=True),
        jnp.finfo(source_parameter_sites.dtype).eps,
    )
    target = target_parameter_sites / jnp.maximum(
        jnp.linalg.norm(target_parameter_sites, axis=1, keepdims=True),
        jnp.finfo(target_parameter_sites.dtype).eps,
    )
    centers = source[control_indices]
    center_radius = jnp.sqrt(
        jnp.maximum(2.0 - 2.0 * centers @ centers.T, 0.0)
    )
    kernel = (center_radius + jnp.finfo(source.dtype).eps) ** degree
    regularization = 1.0e-12 * jnp.maximum(1.0, jnp.max(jnp.abs(kernel)))
    coefficients = jnp.linalg.solve(
        kernel + regularization * jnp.eye(kernel.shape[0]),
        source_values[control_indices],
    )
    target_radius = jnp.sqrt(
        jnp.maximum(2.0 - 2.0 * target @ centers.T, 0.0)
    )
    return (target_radius + jnp.finfo(source.dtype).eps) ** degree @ coefficients
