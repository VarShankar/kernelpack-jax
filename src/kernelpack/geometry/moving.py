from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

from jax import jit, vmap
from jax.scipy.linalg import lu_factor, lu_solve
import jax.numpy as jnp
import numpy as np

from .core import EmbeddedSurface, fibonacci_sphere


@jit
def _curve_parameter_matrix(theta: jnp.ndarray) -> jnp.ndarray:
    delta = theta[:, None] - theta[None, :]
    radius = jnp.sqrt(jnp.maximum(2.0 * (1.0 - jnp.cos(delta)), 0.0))
    return (radius + jnp.finfo(float).eps) ** 7


@jit
def _surface_parameter_matrix(parameter_sites: jnp.ndarray) -> jnp.ndarray:
    radius = jnp.sqrt(jnp.maximum(2.0 * (1.0 - parameter_sites @ parameter_sites.T), 0.0))
    return radius**8 * jnp.log(radius + jnp.finfo(float).eps)


@jit
def _solve_coefficients(lu: jnp.ndarray, pivots: jnp.ndarray, seed_sites: jnp.ndarray) -> jnp.ndarray:
    return lu_solve((lu, pivots), seed_sites)


def _curve_basis_row(theta: jnp.ndarray, parameter_sites: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    delta = theta - parameter_sites
    radius = jnp.sqrt(jnp.maximum(2.0 * (1.0 - jnp.cos(delta)), 0.0))
    basis = (radius + jnp.finfo(float).eps) ** 7
    derivative = 7.0 * jnp.sin(delta) * (radius + jnp.finfo(float).eps) ** 5
    return basis, derivative


@jit
def _evaluate_curve(
    query: jnp.ndarray,
    parameter_sites: jnp.ndarray,
    coefficients: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    basis, derivative = vmap(_curve_basis_row, in_axes=(0, None))(query, parameter_sites)
    samples = basis @ coefficients
    tangents = derivative @ coefficients
    normals = jnp.column_stack([tangents[:, 1], -tangents[:, 0]])
    normals = normals / jnp.maximum(jnp.linalg.norm(normals, axis=1, keepdims=True), jnp.finfo(float).eps)
    return samples, normals


def _sphere_tangent_frame(point: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    reference = jnp.where(jnp.abs(point[2]) > 0.9, jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 1.0]))
    tangent1 = jnp.cross(reference, point)
    tangent1 = tangent1 / jnp.maximum(jnp.linalg.norm(tangent1), jnp.finfo(float).eps)
    tangent2 = jnp.cross(point, tangent1)
    tangent2 = tangent2 / jnp.maximum(jnp.linalg.norm(tangent2), jnp.finfo(float).eps)
    return tangent1, tangent2


def _surface_basis_row(
    query: jnp.ndarray,
    parameter_sites: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    difference = query - parameter_sites
    radius = jnp.linalg.norm(difference, axis=1)
    basis = radius**8 * jnp.log(radius + jnp.finfo(float).eps)
    radial_derivative = radius**6 * (8.0 * jnp.log(radius + jnp.finfo(float).eps) + 1.0)
    tangent1, tangent2 = _sphere_tangent_frame(query)
    derivative1 = radial_derivative * (difference @ tangent1)
    derivative2 = radial_derivative * (difference @ tangent2)
    return basis, derivative1, derivative2


@jit
def _evaluate_surface(
    query: jnp.ndarray,
    parameter_sites: jnp.ndarray,
    coefficients: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    basis, derivative1, derivative2 = vmap(_surface_basis_row, in_axes=(0, None))(query, parameter_sites)
    samples = basis @ coefficients
    mapped_tangent1 = derivative1 @ coefficients
    mapped_tangent2 = derivative2 @ coefficients
    normals = jnp.cross(mapped_tangent1, mapped_tangent2)
    normals = normals / jnp.maximum(jnp.linalg.norm(normals, axis=1, keepdims=True), jnp.finfo(float).eps)
    orientation = jnp.mean(jnp.sum((samples - jnp.mean(samples, axis=0)) * normals, axis=1))
    normals = jnp.where(orientation < 0.0, -normals, normals)
    return samples, normals


def _oriented_box_sides(points: jnp.ndarray) -> np.ndarray:
    points_np = np.asarray(points, dtype=float)
    covariance = np.cov(points_np, rowvar=False)
    _, vectors = np.linalg.eigh(covariance)
    local = points_np @ vectors
    return np.max(local, axis=0) - np.min(local, axis=0)


def _greedy_thin(points: jnp.ndarray, radius: float) -> np.ndarray:
    """Match the ordered greedy thinning used by the MATLAB moving model."""

    points_np = np.asarray(points, dtype=float)
    if points_np.shape[0] == 0:
        return np.zeros((0,), dtype=int)
    radius = float(radius)
    inverse_radius = 1.0 / radius
    offsets = tuple(product((-1, 0, 1), repeat=points_np.shape[1]))
    cells: dict[tuple[int, ...], list[int]] = {}
    kept: list[int] = []
    radius_sq = radius * radius
    for row, point in enumerate(points_np):
        cell = tuple(np.floor(point * inverse_radius).astype(np.int64))
        conflict = False
        for offset in offsets:
            neighbor = tuple(cell[d] + offset[d] for d in range(points_np.shape[1]))
            for previous in cells.get(neighbor, ()):
                if np.sum((point - points_np[previous]) ** 2) <= radius_sq:
                    conflict = True
                    break
            if conflict:
                break
        if not conflict:
            kept.append(row)
            cells.setdefault(cell, []).append(row)
    return np.asarray(kept, dtype=int)


@dataclass
class MovingBoundarySBFModel:
    """Cached parametric boundary model from the 2021 moving-domain method.

    The model uses r^7 on S^1 and r^8 log(r) on S^2. Parameter sites, LU
    factors, and dense evaluation sites remain fixed while Cartesian marker
    coordinates evolve.
    """

    seed_sites: jnp.ndarray
    dimension: int = field(init=False)
    seed_count: int = field(init=False)
    parameter_sites: jnp.ndarray = field(init=False)
    lu: jnp.ndarray = field(init=False)
    pivots: jnp.ndarray = field(init=False)
    evaluation_parameter_sites: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    evaluation_spacing: float = jnp.nan

    def __post_init__(self) -> None:
        seed_sites = jnp.asarray(self.seed_sites, dtype=float)
        if seed_sites.ndim != 2 or seed_sites.shape[0] == 0 or seed_sites.shape[1] not in (2, 3):
            raise ValueError("moving-boundary seed sites must be a nonempty N-by-2 or N-by-3 array")
        self.seed_sites = seed_sites
        self.dimension = int(seed_sites.shape[1])
        self.seed_count = int(seed_sites.shape[0])
        if self.dimension == 2:
            self.parameter_sites = jnp.linspace(-jnp.pi, jnp.pi, self.seed_count, endpoint=False)
            matrix = _curve_parameter_matrix(self.parameter_sites)
        else:
            centered = seed_sites - jnp.mean(seed_sites, axis=0)
            radii = jnp.linalg.norm(centered, axis=1, keepdims=True)
            if bool(jnp.any(radii <= 100.0 * jnp.finfo(float).eps * jnp.max(radii))):
                raise ValueError("three-dimensional moving-boundary seeds must define nonzero centroid directions")
            self.parameter_sites = centered / radii
            matrix = _surface_parameter_matrix(self.parameter_sites)
        self.lu, self.pivots = lu_factor(matrix)

    def evaluate(self, seed_sites: jnp.ndarray, h: float) -> EmbeddedSurface:
        seed_sites = jnp.asarray(seed_sites, dtype=float)
        if seed_sites.shape != (self.seed_count, self.dimension):
            raise ValueError("moving-boundary seed sites changed shape")
        if not np.isfinite(h) or h <= 0.0:
            raise ValueError("moving-boundary spacing must be positive and finite")
        coefficients = _solve_coefficients(self.lu, self.pivots, seed_sites)
        if self.dimension == 2:
            samples, normals = self._evaluate_curve(seed_sites, coefficients, float(h))
        else:
            samples, normals = self._evaluate_surface(seed_sites, coefficients, float(h))

        surface = EmbeddedSurface()
        surface.set_data_sites(seed_sites)
        surface.set_sample_sites(samples)
        surface.uniform_sample_sites = samples
        surface.nrmls = normals
        surface.uniform_nrmls = normals
        surface.sep_rad = float(h)
        return surface

    def _evaluate_curve(
        self,
        seed_sites: jnp.ndarray,
        coefficients: jnp.ndarray,
        h: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self.evaluation_parameter_sites.size == 0 or self.evaluation_spacing != h:
            sides = _oriented_box_sides(seed_sites)
            target_count = max(8, int(np.floor(2.0 * np.sum(sides) / h)))
            self.evaluation_parameter_sites = jnp.linspace(-jnp.pi, jnp.pi, 2 * target_count, endpoint=False)
            self.evaluation_spacing = h
        samples, normals = _evaluate_curve(self.evaluation_parameter_sites, self.parameter_sites, coefficients)
        keep = _greedy_thin(samples, 0.9 * h)
        return jnp.asarray(np.asarray(samples)[keep]), jnp.asarray(np.asarray(normals)[keep])

    def _evaluate_surface(
        self,
        seed_sites: jnp.ndarray,
        coefficients: jnp.ndarray,
        h: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self.evaluation_parameter_sites.size == 0 or self.evaluation_spacing != h:
            sides = _oriented_box_sides(seed_sites)
            area_estimate = 2.0 * (sides[0] * sides[1] + sides[1] * sides[2] + sides[0] * sides[2])
            target_count = max(12, int(np.floor(area_estimate / h**2)))
            self.evaluation_parameter_sites = fibonacci_sphere(5 * target_count)
            self.evaluation_spacing = h
        samples, normals = _evaluate_surface(self.evaluation_parameter_sites, self.parameter_sites, coefficients)
        keep = _greedy_thin(samples, 0.9 * h)
        return jnp.asarray(np.asarray(samples)[keep]), jnp.asarray(np.asarray(normals)[keep])
