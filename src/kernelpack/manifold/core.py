from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
from jax import jit, lax, vmap
from jax import scipy as jsp
import jax.numpy as jnp

from kernelpack.accelerators import warp_available, warp_exact_knn
from kernelpack.poly import PolynomialBasis


class SurfaceOperators(NamedTuple):
    """Matrix-free tangent-plane RBF-FD operators on a closed surface."""

    neighbors: jnp.ndarray
    laplacian_weights: jnp.ndarray
    gradient_weights: jnp.ndarray


class SurfaceOperatorCache(NamedTuple):
    """Reference local factors retained across nearby surface configurations."""

    neighbors: jnp.ndarray
    lhs: jnp.ndarray
    lu: jnp.ndarray
    pivots: jnp.ndarray


class SurfaceUpdateInfo(NamedTuple):
    relative_residual: jnp.ndarray
    defect_iterations: jnp.ndarray
    refactored: jnp.ndarray


def surface_polynomial_degree(xi: int, theta: int = 2) -> int:
    """Return the polynomial degree used by the MATLAB tangent-plane method."""

    if xi < 1:
        raise ValueError("xi must be positive")
    return int(xi + theta - 1)


def surface_phs_degree(ell: int) -> int:
    """Select the lower odd PHS degree, with KernelPack's stability bounds."""

    degree = ell if ell % 2 else ell - 1
    return min(max(int(degree), 5), 11)


def surface_stencil_size(ell: int) -> int:
    """Use twice the two-dimensional polynomial dimension plus one node."""

    polynomial_count = (ell + 1) * (ell + 2) // 2
    return 2 * polynomial_count + 1


@partial(jit, static_argnames=("k",))
def _jax_surface_knn(points: jnp.ndarray, *, k: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    squared_distance = jnp.sum(
        (points[:, None, :] - points[None, :, :]) ** 2,
        axis=2,
    )
    negative_distance, indices = lax.top_k(-squared_distance, k)
    return indices, jnp.sqrt(jnp.maximum(-negative_distance, 0.0))


def build_surface_stencil_graph(
    points: jnp.ndarray,
    stencil_size: int,
    *,
    backend: str = "auto",
) -> jnp.ndarray:
    """Build a fixed surface KNN graph for subsequent JIT-compiled kernels.

    Neighbor selection is discrete setup work. ``auto`` uses the existing
    Warp hash-grid backend when available and otherwise uses a JAX top-k
    implementation. The returned graph is a JAX array and all downstream
    assembly and stepping remain device native.
    """

    points = jnp.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("moving-surface points must have shape (N, 3)")
    if stencil_size < 1 or stencil_size > points.shape[0]:
        raise ValueError("stencil_size must lie between 1 and the surface node count")
    selected = str(backend).lower()
    if selected == "auto":
        selected = "warp" if warp_available() else "jax"
    if selected == "warp":
        indices, _ = warp_exact_knn(points, points, int(stencil_size))
        return jnp.asarray(indices, dtype=int)
    if selected == "jax":
        indices, _ = _jax_surface_knn(points, k=int(stencil_size))
        return indices
    raise ValueError("backend must be 'auto', 'warp', or 'jax'")


@partial(jit, static_argnames=("k",))
def _jax_cross_surface_knn(
    query_points: jnp.ndarray,
    source_points: jnp.ndarray,
    *,
    k: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    squared_distance = jnp.sum(
        (query_points[:, None, :] - source_points[None, :, :]) ** 2,
        axis=2,
    )
    negative_distance, indices = lax.top_k(-squared_distance, k)
    return indices, jnp.sqrt(jnp.maximum(-negative_distance, 0.0))


def build_cross_surface_stencil_graph(
    query_points: jnp.ndarray,
    source_points: jnp.ndarray,
    stencil_size: int,
    *,
    backend: str = "auto",
) -> jnp.ndarray:
    """Find source neighbors for fixed-shape surface transfer kernels."""

    query_points = jnp.asarray(query_points, dtype=float)
    source_points = jnp.asarray(source_points, dtype=float)
    if query_points.ndim != 2 or source_points.ndim != 2:
        raise ValueError("query_points and source_points must be matrices")
    if query_points.shape[1] != 3 or source_points.shape[1] != 3:
        raise ValueError("surface transfer requires three-dimensional points")
    if stencil_size < 1 or stencil_size > source_points.shape[0]:
        raise ValueError("invalid cross-surface stencil size")
    selected = str(backend).lower()
    if selected == "auto":
        selected = "warp" if warp_available() else "jax"
    if selected == "warp":
        indices, _ = warp_exact_knn(query_points, source_points, int(stencil_size))
        return jnp.asarray(indices, dtype=int)
    if selected == "jax":
        indices, _ = _jax_cross_surface_knn(
            query_points,
            source_points,
            k=int(stencil_size),
        )
        return indices
    raise ValueError("backend must be 'auto', 'warp', or 'jax'")


def estimate_pca_normals(
    points: jnp.ndarray,
    *,
    previous_normals: jnp.ndarray | None = None,
    stencil_size: int = 20,
    backend: str = "auto",
) -> jnp.ndarray:
    """Estimate consistently oriented point-cloud normals by local PCA."""

    points = jnp.asarray(points, dtype=float)
    neighbors = build_surface_stencil_graph(points, min(int(stencil_size), points.shape[0]), backend=backend)

    reference = (
        points - jnp.mean(points, axis=0)
        if previous_normals is None
        else jnp.asarray(previous_normals, dtype=points.dtype)
    )
    return _estimate_pca_normals_from_neighbors(points, neighbors, reference)


@jit
def _estimate_pca_normals_from_neighbors(
    points: jnp.ndarray,
    neighbors: jnp.ndarray,
    reference: jnp.ndarray,
) -> jnp.ndarray:
    def local_normal(cloud: jnp.ndarray) -> jnp.ndarray:
        centered = cloud - jnp.mean(cloud, axis=0)
        _, eigenvectors = jnp.linalg.eigh(centered.T @ centered)
        return eigenvectors[:, 0]

    normals = vmap(local_normal)(points[neighbors])
    normals = jnp.where(
        jnp.sum(normals * reference, axis=1, keepdims=True) < 0.0,
        -normals,
        normals,
    )
    return normals / jnp.maximum(
        jnp.linalg.norm(normals, axis=1, keepdims=True),
        jnp.finfo(points.dtype).eps,
    )


@jit
def _normalize(vector: jnp.ndarray) -> jnp.ndarray:
    return vector / jnp.maximum(jnp.linalg.norm(vector), jnp.finfo(vector.dtype).eps)


@jit
def _tangent_frame(normal: jnp.ndarray) -> jnp.ndarray:
    """Reproduce the deterministic Gram-Schmidt frame used by MATLAB."""

    normal = _normalize(normal)
    dominant = jnp.argmax(jnp.abs(normal))
    first_references = jnp.asarray(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=normal.dtype,
    )
    second_references = jnp.asarray(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=normal.dtype,
    )
    first_reference = first_references[dominant]
    second_reference = second_references[dominant]
    tangent_one = _normalize(first_reference - jnp.dot(normal, first_reference) * normal)
    tangent_two = _normalize(
        second_reference
        - jnp.dot(normal, second_reference) * normal
        - jnp.dot(tangent_one, second_reference) * tangent_one
    )
    return jnp.column_stack((tangent_one, tangent_two))


def _basis_data(ell: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    basis = PolynomialBasis.from_total_degree(2, ell, family="legendre")
    return basis.index_set, basis.recurrence_a, basis.recurrence_b


@partial(jit, static_argnames=("phs_degree",))
def _local_tangent_system(
    stencil_points: jnp.ndarray,
    center_normal: jnp.ndarray,
    index_set: jnp.ndarray,
    recurrence_a: jnp.ndarray,
    recurrence_b: jnp.ndarray,
    *,
    phs_degree: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    from kernelpack.poly.core import _tensor_evaluate_with_recurrence

    frame = _tangent_frame(center_normal)
    local_points = (stencil_points - stencil_points[0]) @ frame
    width = jnp.max(jnp.abs(local_points))
    width = jnp.maximum(width, jnp.finfo(stencil_points.dtype).eps)
    scaled_points = local_points / width
    pairwise_distance = jnp.linalg.norm(
        scaled_points[:, None, :] - scaled_points[None, :, :],
        axis=2,
    )
    epsilon = jnp.finfo(stencil_points.dtype).eps

    zero_derivative = jnp.zeros((1, 2), dtype=int)
    polynomial = _tensor_evaluate_with_recurrence(
        scaled_points,
        index_set,
        zero_derivative,
        recurrence_a,
        recurrence_b,
    )[:, :, 0]
    rbf_block = (pairwise_distance + epsilon) ** phs_degree
    polynomial_count = index_set.shape[0]
    lhs = jnp.block(
        [
            [rbf_block, polynomial],
            [polynomial.T, jnp.zeros((polynomial_count, polynomial_count), dtype=stencil_points.dtype)],
        ]
    )

    center_distance = pairwise_distance[:, 0]
    radial_first_over_radius = phs_degree * (center_distance + epsilon) ** (phs_degree - 2)
    radial_second = phs_degree * (phs_degree - 1) * (center_distance + epsilon) ** (phs_degree - 2)
    rbf_laplacian = (radial_second + radial_first_over_radius) / width**2
    rbf_gradient = -scaled_points * radial_first_over_radius[:, None] / width

    derivative_orders = jnp.asarray([[2, 0], [0, 2], [1, 0], [0, 1]], dtype=int)
    polynomial_derivatives = _tensor_evaluate_with_recurrence(
        scaled_points[:1],
        index_set,
        derivative_orders,
        recurrence_a,
        recurrence_b,
    )[0]
    polynomial_laplacian = (
        polynomial_derivatives[:, 0] + polynomial_derivatives[:, 1]
    ) / width**2
    polynomial_gradient = polynomial_derivatives[:, 2:4] / width
    rhs = jnp.concatenate(
        (
            jnp.column_stack((rbf_laplacian, rbf_gradient)),
            jnp.column_stack((polynomial_laplacian, polynomial_gradient)),
        ),
        axis=0,
    )
    return lhs, rhs, frame


@partial(jit, static_argnames=("phs_degree",))
def _surface_systems(
    points: jnp.ndarray,
    normals: jnp.ndarray,
    neighbors: jnp.ndarray,
    index_set: jnp.ndarray,
    recurrence_a: jnp.ndarray,
    recurrence_b: jnp.ndarray,
    *,
    phs_degree: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    return vmap(
        lambda stencil, normal: _local_tangent_system(
            stencil,
            normal,
            index_set,
            recurrence_a,
            recurrence_b,
            phs_degree=phs_degree,
        )
    )(points[neighbors], normals)


@jit
def _operators_from_solution(
    neighbors: jnp.ndarray,
    solution: jnp.ndarray,
    frames: jnp.ndarray,
) -> SurfaceOperators:
    stencil_size = neighbors.shape[1]
    laplacian = solution[:, :stencil_size, 0]
    local_gradient = solution[:, :stencil_size, 1:3]
    ambient_gradient = jnp.einsum("nka,nda->nkd", local_gradient, frames)
    return SurfaceOperators(neighbors, laplacian, ambient_gradient)


@jit
def _factor_and_solve(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    lu, pivots = jsp.linalg.lu_factor(lhs)
    solution = jsp.linalg.lu_solve((lu, pivots), rhs)
    return solution, lu, pivots


def assemble_tangent_plane_operators(
    points: jnp.ndarray,
    normals: jnp.ndarray,
    neighbors: jnp.ndarray,
    *,
    xi: int,
    theta: int = 2,
) -> tuple[SurfaceOperators, SurfaceOperatorCache]:
    """Assemble all tangent-plane rows in one batched device operation."""

    ell = surface_polynomial_degree(xi, theta)
    phs_degree = surface_phs_degree(ell)
    index_set, recurrence_a, recurrence_b = _basis_data(ell)
    lhs, rhs, frames = _surface_systems(
        jnp.asarray(points, dtype=float),
        jnp.asarray(normals, dtype=float),
        jnp.asarray(neighbors, dtype=int),
        index_set,
        recurrence_a,
        recurrence_b,
        phs_degree=phs_degree,
    )
    solution, lu, pivots = _factor_and_solve(lhs, rhs)
    operators = _operators_from_solution(neighbors, solution, frames)
    return operators, SurfaceOperatorCache(neighbors, lhs, lu, pivots)


@partial(jit, static_argnames=("phs_degree", "max_defect_iterations"))
def _update_tangent_plane_operators(
    points: jnp.ndarray,
    normals: jnp.ndarray,
    neighbors: jnp.ndarray,
    cache: SurfaceOperatorCache,
    index_set: jnp.ndarray,
    recurrence_a: jnp.ndarray,
    recurrence_b: jnp.ndarray,
    tolerance: float,
    *,
    phs_degree: int,
    max_defect_iterations: int,
) -> tuple[SurfaceOperators, SurfaceOperatorCache, SurfaceUpdateInfo]:
    lhs, rhs, frames = _surface_systems(
        points,
        normals,
        neighbors,
        index_set,
        recurrence_a,
        recurrence_b,
        phs_degree=phs_degree,
    )
    same_graph = jnp.all(neighbors == cache.neighbors)
    candidate = jsp.linalg.lu_solve((cache.lu, cache.pivots), rhs)
    iterations = jnp.zeros((points.shape[0],), dtype=int)

    def defect_body(_iteration: int, state: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        current, counts = state
        residual = rhs - jnp.einsum("nij,njk->nik", lhs, current)
        denominator = jnp.maximum(
            jnp.linalg.norm(rhs, axis=(1, 2)),
            jnp.finfo(rhs.dtype).eps,
        )
        active = jnp.linalg.norm(residual, axis=(1, 2)) / denominator > tolerance
        correction = jsp.linalg.lu_solve((cache.lu, cache.pivots), residual)
        current = current + jnp.where(active[:, None, None], correction, 0.0)
        return current, counts + active.astype(int)

    candidate, iterations = lax.fori_loop(
        0,
        max_defect_iterations,
        defect_body,
        (candidate, iterations),
    )
    residual = rhs - jnp.einsum("nij,njk->nik", lhs, candidate)
    relative_residual = jnp.linalg.norm(residual, axis=(1, 2)) / jnp.maximum(
        jnp.linalg.norm(rhs, axis=(1, 2)),
        jnp.finfo(rhs.dtype).eps,
    )
    defect_accepted = same_graph & jnp.all(jnp.isfinite(candidate)) & jnp.all(relative_residual <= tolerance)

    def accept_defect(_unused: None):
        return candidate, cache.lhs, cache.lu, cache.pivots, jnp.asarray(False)

    def refactor(_unused: None):
        direct, lu, pivots = _factor_and_solve(lhs, rhs)
        return direct, lhs, lu, pivots, jnp.asarray(True)

    solution, reference_lhs, lu, pivots, refactored = lax.cond(
        defect_accepted,
        accept_defect,
        refactor,
        operand=None,
    )
    operators = _operators_from_solution(neighbors, solution, frames)
    updated_cache = SurfaceOperatorCache(neighbors, reference_lhs, lu, pivots)
    info = SurfaceUpdateInfo(relative_residual, iterations, refactored)
    return operators, updated_cache, info


def update_tangent_plane_operators(
    points: jnp.ndarray,
    normals: jnp.ndarray,
    neighbors: jnp.ndarray,
    cache: SurfaceOperatorCache,
    *,
    xi: int,
    theta: int = 2,
    tolerance: float = 1.0e-10,
    max_defect_iterations: int = 4,
) -> tuple[SurfaceOperators, SurfaceOperatorCache, SurfaceUpdateInfo]:
    """Update surface weights with cached-factor defect correction.

    The common path applies only batched triangular solves and true-system
    residual matvecs. If any row changes neighbors or misses the requested
    residual after ``max_defect_iterations``, all rows are refactored in one
    GPU batch so shapes remain static and the function remains JIT-compatible.
    """

    ell = surface_polynomial_degree(xi, theta)
    phs_degree = surface_phs_degree(ell)
    index_set, recurrence_a, recurrence_b = _basis_data(ell)
    return _update_tangent_plane_operators(
        jnp.asarray(points, dtype=float),
        jnp.asarray(normals, dtype=float),
        jnp.asarray(neighbors, dtype=int),
        cache,
        index_set,
        recurrence_a,
        recurrence_b,
        jnp.asarray(tolerance, dtype=float),
        phs_degree=phs_degree,
        max_defect_iterations=int(max_defect_iterations),
    )


@jit
def apply_surface_operator(
    weights: jnp.ndarray,
    neighbors: jnp.ndarray,
    values: jnp.ndarray,
) -> jnp.ndarray:
    """Apply scalar stencil weights to scalar or vector nodal values."""

    return jnp.einsum("nk,nk...->n...", weights, values[neighbors])


@jit
def surface_gradient(operators: SurfaceOperators, values: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum(
        "nkd,nk...->nd...",
        operators.gradient_weights,
        values[operators.neighbors],
    )


@jit
def surface_divergence(operators: SurfaceOperators, vector_field: jnp.ndarray) -> jnp.ndarray:
    differentiated_components = jnp.einsum(
        "nkd,nkc->ndc",
        operators.gradient_weights,
        vector_field[operators.neighbors],
    )
    return jnp.trace(differentiated_components, axis1=1, axis2=2)


@partial(jit, static_argnames=("power",))
def apply_surface_laplacian_power(
    operators: SurfaceOperators,
    values: jnp.ndarray,
    *,
    power: int,
) -> jnp.ndarray:
    return lax.fori_loop(
        0,
        power,
        lambda _iteration, current: apply_surface_operator(
            operators.laplacian_weights,
            operators.neighbors,
            current,
        ),
        values,
    )


@partial(jit, static_argnames=("power",))
def hyperviscosity_divergence_correction(
    operators: SurfaceOperators,
    velocity: jnp.ndarray,
    gamma: jnp.ndarray,
    *,
    power: int,
) -> jnp.ndarray:
    powered_velocity = apply_surface_laplacian_power(operators, velocity, power=power)
    return powered_velocity @ jnp.asarray(gamma, dtype=velocity.dtype)


@jit
def surface_point_quality(points: jnp.ndarray, two_nearest_neighbors: jnp.ndarray) -> jnp.ndarray:
    """Return the max/min nearest-neighbor ratio used by the remap trigger."""

    distances = jnp.linalg.norm(
        points[two_nearest_neighbors] - points[:, None, :],
        axis=2,
    )
    nearest_nonself = jnp.max(distances, axis=1)
    return jnp.max(nearest_nonself) / jnp.maximum(
        jnp.min(nearest_nonself),
        jnp.finfo(points.dtype).eps,
    )


@jit
def predict_surface_quality_crossing(
    previous_quality: float,
    current_quality: float,
    threshold: float,
    lookahead_steps: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    slope = jnp.maximum(0.0, current_quality - previous_quality)
    predicted = current_quality + jnp.maximum(0.0, lookahead_steps) * slope
    return predicted >= threshold, predicted
