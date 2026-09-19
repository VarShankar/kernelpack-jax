from __future__ import annotations

from functools import partial
from typing import Callable, NamedTuple

from jax import jit, lax
from jax import scipy as jsp
import jax.numpy as jnp

from kernelpack.manifold import (
    SurfaceOperators,
    apply_surface_operator,
    hyperviscosity_divergence_correction,
    surface_divergence,
)


VelocityCallback = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


class MovingSurfaceHistory(NamedTuple):
    """Newest-first BDF histories with fixed device shapes."""

    concentration: jnp.ndarray
    points: jnp.ndarray


class MovingSurfaceStepInfo(NamedTuple):
    linear_info: jnp.ndarray
    relative_residual: jnp.ndarray
    mass_before_projection: jnp.ndarray
    mass_after_projection: jnp.ndarray
    mass_shift: jnp.ndarray


def initialize_moving_surface_history(
    points: jnp.ndarray,
    concentration: jnp.ndarray,
) -> MovingSurfaceHistory:
    """Initialize fixed three-level storage for BDF startup."""

    points = jnp.asarray(points, dtype=float)
    concentration = jnp.asarray(concentration, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if concentration.shape != (points.shape[0],):
        raise ValueError("concentration must contain one scalar per surface node")
    return MovingSurfaceHistory(
        concentration=jnp.broadcast_to(concentration, (3, concentration.size)),
        points=jnp.broadcast_to(points, (3, *points.shape)),
    )


@partial(jit, static_argnames=("velocity",))
def rk3_material_step(
    time: float,
    points: jnp.ndarray,
    dt: float,
    velocity: VelocityCallback,
) -> jnp.ndarray:
    """Advance material labels with the third-order RK rule used by KernelPack."""

    first = velocity(time, points)
    second = velocity(time + 0.5 * dt, points + 0.5 * dt * first)
    third = velocity(time + 0.75 * dt, points + 0.75 * dt * second)
    return points + dt * (2.0 * first + 3.0 * second + 4.0 * third) / 9.0


@partial(jit, static_argnames=("velocity", "history_levels"))
def semi_lagrangian_backfill_points(
    arrival_points: jnp.ndarray,
    arrival_time: float,
    dt: float,
    velocity: VelocityCallback,
    *,
    history_levels: int = 2,
) -> jnp.ndarray:
    """Trace rearranged markers backward to rebuild a BDF history."""

    def backtrace(points: jnp.ndarray, level: jnp.ndarray):
        time = arrival_time - level * dt
        departure = rk3_material_step(time, points, -dt, velocity)
        return departure, departure

    _, departures = lax.scan(
        backtrace,
        arrival_points,
        jnp.arange(history_levels, dtype=arrival_points.dtype),
    )
    return departures


def _bdf_data(order: int) -> tuple[jnp.ndarray, jnp.ndarray, float]:
    if order == 1:
        return jnp.asarray([1.0, 0.0, 0.0]), jnp.asarray([1.0, 0.0, 0.0]), 1.0
    if order == 2:
        return (
            jnp.asarray([4.0 / 3.0, -1.0 / 3.0, 0.0]),
            jnp.asarray([2.0, -1.0, 0.0]),
            2.0 / 3.0,
        )
    if order == 3:
        return (
            jnp.asarray([18.0 / 11.0, -9.0 / 11.0, 2.0 / 11.0]),
            jnp.asarray([3.0, -3.0, 1.0]),
            6.0 / 11.0,
        )
    raise ValueError("BDF order must be 1, 2, or 3")


@partial(jit, static_argnames=("order",))
def bdf_material_velocity(
    new_points: jnp.ndarray,
    point_history: jnp.ndarray,
    dt: float,
    *,
    order: int,
) -> jnp.ndarray:
    if order == 1:
        return (new_points - point_history[0]) / dt
    if order == 2:
        return (3.0 * new_points - 4.0 * point_history[0] + point_history[1]) / (2.0 * dt)
    if order == 3:
        return (
            11.0 * new_points
            - 18.0 * point_history[0]
            + 9.0 * point_history[1]
            - 2.0 * point_history[2]
        ) / (6.0 * dt)
    raise ValueError("BDF order must be 1, 2, or 3")


@jit
def _laplacian_diagonal(operators: SurfaceOperators) -> jnp.ndarray:
    rows = jnp.arange(operators.neighbors.shape[0])[:, None]
    return jnp.sum(
        jnp.where(
            operators.neighbors == rows,
            operators.laplacian_weights,
            0.0,
        ),
        axis=1,
    )


@partial(
    jit,
    static_argnames=(
        "order",
        "hyperviscosity_power",
        "gmres_restart",
        "gmres_max_iterations",
        "project_mass",
    ),
)
def moving_surface_adr_step(
    history: MovingSurfaceHistory,
    new_points: jnp.ndarray,
    operators: SurfaceOperators,
    forcing: jnp.ndarray,
    quadrature_weights: jnp.ndarray,
    target_mass: float,
    dt: float,
    diffusivity: float,
    hyperviscosity_gamma: jnp.ndarray,
    linear_tolerance: float,
    *,
    order: int,
    hyperviscosity_power: int,
    reaction_history: jnp.ndarray | None = None,
    gmres_restart: int = 20,
    gmres_max_iterations: int = 200,
    project_mass: bool = True,
) -> tuple[MovingSurfaceHistory, MovingSurfaceStepInfo]:
    """Take one Lagrangian-Eulerian BDF step on a moving closed surface.

    The material derivative is represented by the newest-first Lagrangian
    history. Diffusion is implicit. Reaction values, the surface-dilation
    term, and its componentwise hyperviscosity correction use the polynomial
    extrapolate, matching the MATLAB method. ``reaction_history`` contains
    reaction values at the same newest-first levels as ``history``; omit it
    for a zero reaction term.
    """

    history_coefficients, extrapolation_coefficients, implicit_scale = _bdf_data(order)
    previous_combination = history_coefficients @ history.concentration
    extrapolated = extrapolation_coefficients @ history.concentration
    if reaction_history is None:
        extrapolated_reaction = jnp.zeros_like(extrapolated)
    else:
        extrapolated_reaction = extrapolation_coefficients @ reaction_history
    material_velocity = bdf_material_velocity(new_points, history.points, dt, order=order)
    dilation = surface_divergence(operators, material_velocity)
    if hyperviscosity_power > 0:
        dilation = dilation + hyperviscosity_divergence_correction(
            operators,
            material_velocity,
            hyperviscosity_gamma,
            power=hyperviscosity_power,
        )
    rhs = (
        previous_combination
        + implicit_scale * dt * (forcing + extrapolated_reaction)
        - implicit_scale * dt * extrapolated * dilation
    )

    def system_matvec(values: jnp.ndarray) -> jnp.ndarray:
        laplacian = apply_surface_operator(
            operators.laplacian_weights,
            operators.neighbors,
            values,
        )
        return values - implicit_scale * dt * diffusivity * laplacian

    diagonal = 1.0 - implicit_scale * dt * diffusivity * _laplacian_diagonal(operators)
    inverse_diagonal = 1.0 / jnp.where(
        jnp.abs(diagonal) > jnp.finfo(diagonal.dtype).eps,
        diagonal,
        1.0,
    )
    solution, linear_info = jsp.sparse.linalg.gmres(
        system_matvec,
        rhs,
        x0=extrapolated,
        tol=linear_tolerance,
        atol=0.0,
        restart=gmres_restart,
        maxiter=gmres_max_iterations,
        M=lambda values: inverse_diagonal * values,
        solve_method="batched",
    )
    residual = system_matvec(solution) - rhs
    relative_residual = jnp.linalg.norm(residual) / jnp.maximum(
        jnp.linalg.norm(rhs),
        jnp.finfo(rhs.dtype).eps,
    )

    mass_before = quadrature_weights @ solution
    if project_mass:
        mass_shift = (target_mass - mass_before) / jnp.sum(quadrature_weights)
        solution = solution + mass_shift
    else:
        mass_shift = jnp.asarray(0.0, dtype=solution.dtype)
    mass_after = quadrature_weights @ solution
    updated_history = MovingSurfaceHistory(
        concentration=jnp.concatenate((solution[None, :], history.concentration[:2]), axis=0),
        points=jnp.concatenate((new_points[None, :, :], history.points[:2]), axis=0),
    )
    return updated_history, MovingSurfaceStepInfo(
        linear_info=linear_info,
        relative_residual=relative_residual,
        mass_before_projection=mass_before,
        mass_after_projection=mass_after,
        mass_shift=mass_shift,
    )
