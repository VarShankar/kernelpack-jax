from __future__ import annotations

from functools import partial
from typing import NamedTuple

from jax import jit, vmap
from jax import scipy as jsp
import jax.numpy as jnp


class SphericalSBFModel(NamedTuple):
    """Fixed-parameter global SBF geometry model for sphere-like surfaces."""

    parameter_sites: jnp.ndarray
    evaluation_sites: jnp.ndarray
    base_quadrature_weights: jnp.ndarray
    lu: jnp.ndarray
    pivots: jnp.ndarray


class SurfaceGeometry(NamedTuple):
    points: jnp.ndarray
    normals: jnp.ndarray
    quadrature_weights: jnp.ndarray


class ToroidalRBFModel(NamedTuple):
    """Fixed periodic parameter model for torus-like surfaces."""

    parameter_sites: jnp.ndarray
    evaluation_sites: jnp.ndarray
    base_quadrature_weights: jnp.ndarray
    lu: jnp.ndarray
    pivots: jnp.ndarray
    degree: int


@jit
def _normalize_rows(values: jnp.ndarray) -> jnp.ndarray:
    denominator = jnp.maximum(
        jnp.linalg.norm(values, axis=1, keepdims=True),
        jnp.finfo(values.dtype).eps,
    )
    return values / denominator


@jit
def _sbf_parameter_matrix(parameter_sites: jnp.ndarray) -> jnp.ndarray:
    radius = jnp.sqrt(
        jnp.maximum(2.0 * (1.0 - parameter_sites @ parameter_sites.T), 0.0)
    )
    return radius**8 * jnp.log(radius + jnp.finfo(parameter_sites.dtype).eps)


def build_spherical_sbf_model(
    control_points: jnp.ndarray,
    evaluation_sites: jnp.ndarray | None = None,
    base_quadrature_weights: jnp.ndarray | None = None,
    parameter_sites: jnp.ndarray | None = None,
) -> SphericalSBFModel:
    """Factor the fixed SBF interpolation matrix once.

    ``evaluation_sites`` are points on the unit parameter sphere. Keeping
    them fixed preserves material correspondence as the Cartesian controls
    move and permits a square control fit with a denser evaluation cloud.
    """

    control_points = jnp.asarray(control_points, dtype=float)
    if control_points.ndim != 2 or control_points.shape[1] != 3:
        raise ValueError("control_points must have shape (M, 3)")
    if parameter_sites is None:
        centered = control_points - jnp.mean(control_points, axis=0)
        parameter_sites = _normalize_rows(centered)
    else:
        parameter_sites = _normalize_rows(jnp.asarray(parameter_sites, dtype=float))
        if parameter_sites.shape != control_points.shape:
            raise ValueError("parameter_sites must match control_points")
    if evaluation_sites is None:
        evaluation_sites = parameter_sites
    evaluation_sites = _normalize_rows(jnp.asarray(evaluation_sites, dtype=float))
    if base_quadrature_weights is None:
        base_quadrature_weights = jnp.full(
            (evaluation_sites.shape[0],),
            4.0 * jnp.pi / evaluation_sites.shape[0],
        )
    base_quadrature_weights = jnp.asarray(base_quadrature_weights, dtype=float).reshape(-1)
    if base_quadrature_weights.shape != (evaluation_sites.shape[0],):
        raise ValueError("base_quadrature_weights must match evaluation_sites")
    matrix = _sbf_parameter_matrix(parameter_sites)
    lu, pivots = jsp.linalg.lu_factor(matrix)
    return SphericalSBFModel(
        parameter_sites,
        evaluation_sites,
        base_quadrature_weights,
        lu,
        pivots,
    )


@jit
def _sphere_tangent_frame(point: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    reference = jnp.where(
        jnp.abs(point[2]) > 0.9,
        jnp.asarray([1.0, 0.0, 0.0], dtype=point.dtype),
        jnp.asarray([0.0, 0.0, 1.0], dtype=point.dtype),
    )
    tangent_one = jnp.cross(reference, point)
    tangent_one = tangent_one / jnp.maximum(
        jnp.linalg.norm(tangent_one),
        jnp.finfo(point.dtype).eps,
    )
    tangent_two = jnp.cross(point, tangent_one)
    tangent_two = tangent_two / jnp.maximum(
        jnp.linalg.norm(tangent_two),
        jnp.finfo(point.dtype).eps,
    )
    return tangent_one, tangent_two


@jit
def _sbf_basis_row(
    query: jnp.ndarray,
    parameter_sites: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    difference = query - parameter_sites
    radius = jnp.linalg.norm(difference, axis=1)
    epsilon = jnp.finfo(query.dtype).eps
    basis = radius**8 * jnp.log(radius + epsilon)
    radial_derivative = radius**6 * (8.0 * jnp.log(radius + epsilon) + 1.0)
    tangent_one, tangent_two = _sphere_tangent_frame(query)
    derivative_one = radial_derivative * (difference @ tangent_one)
    derivative_two = radial_derivative * (difference @ tangent_two)
    return basis, derivative_one, derivative_two


@jit
def evaluate_spherical_sbf_geometry(
    model: SphericalSBFModel,
    control_points: jnp.ndarray,
) -> SurfaceGeometry:
    """Evaluate Cartesian markers, analytic normals, and area weights."""

    coefficients = jsp.linalg.lu_solve((model.lu, model.pivots), control_points)
    basis, derivative_one, derivative_two = vmap(
        _sbf_basis_row,
        in_axes=(0, None),
    )(model.evaluation_sites, model.parameter_sites)
    points = basis @ coefficients
    mapped_tangent_one = derivative_one @ coefficients
    mapped_tangent_two = derivative_two @ coefficients
    area_jacobian = jnp.linalg.norm(
        jnp.cross(mapped_tangent_one, mapped_tangent_two),
        axis=1,
    )
    normals = _normalize_rows(jnp.cross(mapped_tangent_one, mapped_tangent_two))
    orientation = jnp.mean(
        jnp.sum((points - jnp.mean(points, axis=0)) * normals, axis=1)
    )
    normals = jnp.where(orientation < 0.0, -normals, normals)
    quadrature_weights = model.base_quadrature_weights * area_jacobian
    return SurfaceGeometry(points, normals, quadrature_weights)


@jit
def evaluate_spherical_sbf_field(
    model: SphericalSBFModel,
    control_values: jnp.ndarray,
    query_sites: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate scalar or vector control data at unit-sphere parameters."""

    coefficients = jsp.linalg.lu_solve((model.lu, model.pivots), control_values)
    basis = vmap(_sbf_basis_row, in_axes=(0, None))(
        _normalize_rows(query_sites),
        model.parameter_sites,
    )[0]
    return basis @ coefficients


@jit
def _periodic_parameter_distance(
    first: jnp.ndarray,
    second: jnp.ndarray,
) -> jnp.ndarray:
    difference = first[:, None, :] - second[None, :, :]
    radius_squared = jnp.sum(2.0 - 2.0 * jnp.cos(difference), axis=2)
    return jnp.sqrt(jnp.maximum(radius_squared, 0.0))


def build_toroidal_rbf_model(
    control_points: jnp.ndarray,
    parameter_sites: jnp.ndarray,
    evaluation_sites: jnp.ndarray | None = None,
    base_quadrature_weights: jnp.ndarray | None = None,
    *,
    degree: int = 7,
) -> ToroidalRBFModel:
    """Factor a periodic two-parameter PHS geometry representation."""

    control_points = jnp.asarray(control_points, dtype=float)
    parameter_sites = jnp.asarray(parameter_sites, dtype=float)
    if control_points.ndim != 2 or control_points.shape[1] != 3:
        raise ValueError("control_points must have shape (M, 3)")
    if parameter_sites.shape != (control_points.shape[0], 2):
        raise ValueError("parameter_sites must have shape (M, 2)")
    if evaluation_sites is None:
        evaluation_sites = parameter_sites
    evaluation_sites = jnp.asarray(evaluation_sites, dtype=float)
    if evaluation_sites.ndim != 2 or evaluation_sites.shape[1] != 2:
        raise ValueError("evaluation_sites must have shape (N, 2)")
    if base_quadrature_weights is None:
        base_quadrature_weights = jnp.full(
            (evaluation_sites.shape[0],),
            (2.0 * jnp.pi) ** 2 / evaluation_sites.shape[0],
        )
    base_quadrature_weights = jnp.asarray(base_quadrature_weights, dtype=float).reshape(-1)
    if base_quadrature_weights.shape != (evaluation_sites.shape[0],):
        raise ValueError("base_quadrature_weights must match evaluation_sites")
    radius = _periodic_parameter_distance(parameter_sites, parameter_sites)
    kernel = (radius + jnp.finfo(radius.dtype).eps) ** degree
    regularization = 1.0e-12 * jnp.maximum(1.0, jnp.max(jnp.abs(kernel)))
    lu, pivots = jsp.linalg.lu_factor(
        kernel + regularization * jnp.eye(kernel.shape[0])
    )
    return ToroidalRBFModel(
        parameter_sites,
        evaluation_sites,
        base_quadrature_weights,
        lu,
        pivots,
        int(degree),
    )


@partial(jit, static_argnames=("degree",))
def _evaluate_toroidal_geometry(
    model: ToroidalRBFModel,
    control_points: jnp.ndarray,
    *,
    degree: int,
) -> SurfaceGeometry:
    coefficients = jsp.linalg.lu_solve((model.lu, model.pivots), control_points)
    difference = model.evaluation_sites[:, None, :] - model.parameter_sites[None, :, :]
    radius = jnp.sqrt(
        jnp.maximum(jnp.sum(2.0 - 2.0 * jnp.cos(difference), axis=2), 0.0)
    )
    epsilon = jnp.finfo(radius.dtype).eps
    basis = (radius + epsilon) ** degree
    radial_factor = degree * (radius + epsilon) ** (degree - 2)
    derivative_theta = radial_factor * jnp.sin(difference[:, :, 0])
    derivative_phi = radial_factor * jnp.sin(difference[:, :, 1])
    points = basis @ coefficients
    tangent_theta = derivative_theta @ coefficients
    tangent_phi = derivative_phi @ coefficients
    cross_product = jnp.cross(tangent_theta, tangent_phi)
    area_jacobian = jnp.linalg.norm(cross_product, axis=1)
    normals = _normalize_rows(cross_product)
    orientation = jnp.mean(
        jnp.sum((points - jnp.mean(points, axis=0)) * normals, axis=1)
    )
    normals = jnp.where(orientation < 0.0, -normals, normals)
    return SurfaceGeometry(
        points,
        normals,
        model.base_quadrature_weights * area_jacobian,
    )


def evaluate_toroidal_rbf_geometry(
    model: ToroidalRBFModel,
    control_points: jnp.ndarray,
) -> SurfaceGeometry:
    return _evaluate_toroidal_geometry(
        model,
        jnp.asarray(control_points, dtype=float),
        degree=int(model.degree),
    )
