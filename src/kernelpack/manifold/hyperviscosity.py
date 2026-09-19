from __future__ import annotations

from functools import partial
from typing import NamedTuple

from jax import jit, lax
import jax.numpy as jnp

from .core import SurfaceOperators, apply_surface_laplacian_power, apply_surface_operator


class HyperviscosityCalibration(NamedTuple):
    gamma: jnp.ndarray
    power: int
    tau: jnp.ndarray
    growth_exponents: jnp.ndarray
    eta_mean: jnp.ndarray
    raw_components: jnp.ndarray
    curvature_components: jnp.ndarray


@partial(jit, static_argnames=("iterations",))
def _largest_real_ritz_value(
    weights: jnp.ndarray,
    neighbors: jnp.ndarray,
    *,
    iterations: int,
) -> jnp.ndarray:
    """Estimate the rightmost eigenvalue with a matrix-free Arnoldi sweep."""

    node_count = weights.shape[0]
    phase = (1.0 + jnp.sqrt(5.0)) * jnp.arange(node_count, dtype=weights.dtype)
    initial = jnp.exp(1j * phase)
    initial = initial / jnp.linalg.norm(initial)
    basis = jnp.zeros((node_count, iterations + 1), dtype=initial.dtype).at[:, 0].set(initial)
    hessenberg = jnp.zeros((iterations + 1, iterations), dtype=initial.dtype)
    column_ids = jnp.arange(iterations)

    def arnoldi_step(column: int, state: tuple[jnp.ndarray, jnp.ndarray]):
        current_basis, current_hessenberg = state
        vector = apply_surface_operator(weights, neighbors, current_basis[:, column])
        active = column_ids <= column
        first_projection = current_basis[:, :iterations].conj().T @ vector
        first_projection = jnp.where(active, first_projection, 0.0)
        vector = vector - current_basis[:, :iterations] @ first_projection
        second_projection = current_basis[:, :iterations].conj().T @ vector
        second_projection = jnp.where(active, second_projection, 0.0)
        vector = vector - current_basis[:, :iterations] @ second_projection
        projection = first_projection + second_projection
        norm = jnp.linalg.norm(vector)
        safe_norm = jnp.maximum(norm, jnp.finfo(weights.dtype).eps)
        current_hessenberg = current_hessenberg.at[:iterations, column].set(projection)
        current_hessenberg = current_hessenberg.at[column + 1, column].set(norm)
        current_basis = current_basis.at[:, column + 1].set(vector / safe_norm)
        return current_basis, current_hessenberg

    _, hessenberg = lax.fori_loop(0, iterations, arnoldi_step, (basis, hessenberg))
    ritz_values = jnp.linalg.eigvals(hessenberg[:iterations, :iterations])
    return jnp.max(jnp.real(ritz_values))


@partial(jit, static_argnames=("arnoldi_iterations",))
def estimate_surface_gradient_spectrum(
    operators: SurfaceOperators,
    *,
    arnoldi_iterations: int = 24,
) -> jnp.ndarray:
    return jnp.stack(
        tuple(
            _largest_real_ritz_value(
                operators.gradient_weights[:, :, component],
                operators.neighbors,
                iterations=arnoldi_iterations,
            )
            for component in range(3)
        )
    )


@jit
def estimate_surface_gradient_growth(
    operators: SurfaceOperators,
    points: jnp.ndarray,
    normals: jnp.ndarray,
    h: float,
    tau: jnp.ndarray,
) -> jnp.ndarray:
    wave_number = 2.0 / h
    oscillation = jnp.exp(1j * wave_number * jnp.sum(points, axis=1))
    ambient_gradient = 1j * wave_number * oscillation[:, None] * jnp.ones((1, 3))
    projected_gradient = ambient_gradient - normals * jnp.sum(normals * ambient_gradient, axis=1)[:, None]
    numerical_gradient = jnp.column_stack(
        tuple(
            apply_surface_operator(
                operators.gradient_weights[:, :, component],
                operators.neighbors,
                oscillation,
            )
            for component in range(3)
        )
    )
    error_norm = jnp.linalg.norm(numerical_gradient - projected_gradient, axis=0)
    safe_tau = jnp.maximum(jnp.abs(tau), jnp.finfo(points.dtype).eps)
    return (
        jnp.log(jnp.maximum(error_norm, jnp.finfo(points.dtype).eps))
        - jnp.log(safe_tau)
        - jnp.log(jnp.linalg.norm(oscillation))
    ) / jnp.log(wave_number)


@partial(jit, static_argnames=("power",))
def _fixed_power_calibration(
    operators: SurfaceOperators,
    points: jnp.ndarray,
    h: float,
    tau: jnp.ndarray,
    growth_exponents: jnp.ndarray,
    *,
    power: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    raw_components = (
        tau
        * 2.0 ** (growth_exponents - 2 * power)
        * h ** (2 * power - growth_exponents)
    )
    wave_number = 2.0 / h
    oscillation = jnp.exp(1j * wave_number * jnp.sum(points, axis=1))
    real_eigenvalue = (-1.0) ** power * 3.0**power * wave_number ** (2 * power)
    eta = apply_surface_laplacian_power(operators, oscillation, power=power) / (
        real_eigenvalue * oscillation
    )
    eta_mean = jnp.mean(jnp.abs(jnp.real(eta)))
    prefactor = 3.0 ** (-power) * (-1.0) ** (1 - power)
    gamma = prefactor * raw_components / eta_mean
    return gamma, eta_mean, raw_components


@jit
def surface_curvature_components(
    operators: SurfaceOperators,
    points: jnp.ndarray,
) -> jnp.ndarray:
    mean_curvature_coordinates = apply_surface_operator(
        operators.laplacian_weights,
        operators.neighbors,
        points,
    )
    return jnp.sqrt(jnp.mean(mean_curvature_coordinates**2, axis=0))


@partial(jit, static_argnames=("power", "arnoldi_iterations"))
def _calibrate_fixed_power_on_device(
    operators: SurfaceOperators,
    points: jnp.ndarray,
    normals: jnp.ndarray,
    h: float,
    *,
    power: int,
    arnoldi_iterations: int,
) -> HyperviscosityCalibration:
    tau = estimate_surface_gradient_spectrum(
        operators, arnoldi_iterations=arnoldi_iterations
    )
    growth_exponents = estimate_surface_gradient_growth(
        operators, points, normals, h, tau
    )
    gamma, eta_mean, raw_components = _fixed_power_calibration(
        operators,
        points,
        h,
        tau,
        growth_exponents,
        power=power,
    )
    return HyperviscosityCalibration(
        gamma=gamma,
        power=power,
        tau=tau,
        growth_exponents=growth_exponents,
        eta_mean=eta_mean,
        raw_components=raw_components,
        curvature_components=surface_curvature_components(operators, points),
    )


def calibrate_surface_hyperviscosity(
    operators: SurfaceOperators,
    points: jnp.ndarray,
    normals: jnp.ndarray,
    h: float,
    *,
    target_order: int,
    power: int | None = None,
    minimum_power: int = 1,
    arnoldi_iterations: int = 24,
) -> HyperviscosityCalibration:
    """Evaluate KernelPack's surface hyperviscosity scaling on the device."""

    tau = estimate_surface_gradient_spectrum(
        operators,
        arnoldi_iterations=int(arnoldi_iterations),
    )
    growth_exponents = estimate_surface_gradient_growth(
        operators,
        jnp.asarray(points, dtype=float),
        jnp.asarray(normals, dtype=float),
        jnp.asarray(h, dtype=float),
        tau,
    )
    selected_power = power
    if selected_power is None:
        selected_power = max(
            int(minimum_power),
            int(jnp.ceil((target_order + jnp.max(growth_exponents)) / 2.0)),
        )
    if selected_power < 1:
        raise ValueError("hyperviscosity power must be positive")
    gamma, eta_mean, raw_components = _fixed_power_calibration(
        operators,
        jnp.asarray(points, dtype=float),
        jnp.asarray(h, dtype=float),
        tau,
        growth_exponents,
        power=int(selected_power),
    )
    return HyperviscosityCalibration(
        gamma=gamma,
        power=int(selected_power),
        tau=tau,
        growth_exponents=growth_exponents,
        eta_mean=eta_mean,
        raw_components=raw_components,
        curvature_components=surface_curvature_components(operators, points),
    )


@partial(jit, static_argnames=("power",))
def _predict_geometry_update(
    operators: SurfaceOperators,
    points: jnp.ndarray,
    h: float,
    calibration: HyperviscosityCalibration,
    *,
    power: int,
) -> tuple[HyperviscosityCalibration, jnp.ndarray]:
    curvature = surface_curvature_components(operators, points)
    scale = calibration.curvature_components / jnp.maximum(
        curvature,
        jnp.finfo(points.dtype).eps,
    )
    exponent = 2 * power - calibration.growth_exponents - 1.0
    raw_components = calibration.raw_components * scale**exponent
    wave_number = 2.0 / h
    oscillation = jnp.exp(1j * wave_number * jnp.sum(points, axis=1))
    real_eigenvalue = (-1.0) ** power * 3.0**power * wave_number ** (2 * power)
    eta = apply_surface_laplacian_power(operators, oscillation, power=power) / (
        real_eigenvalue * oscillation
    )
    eta_mean = jnp.mean(jnp.abs(jnp.real(eta)))
    prefactor = 3.0 ** (-power) * (-1.0) ** (1 - power)
    gamma = prefactor * raw_components / eta_mean
    drift = jnp.linalg.norm(gamma - calibration.gamma) / jnp.maximum(
        jnp.linalg.norm(calibration.gamma),
        jnp.finfo(points.dtype).eps,
    )
    return HyperviscosityCalibration(
        gamma=gamma,
        power=power,
        tau=calibration.tau,
        growth_exponents=calibration.growth_exponents,
        eta_mean=eta_mean,
        raw_components=raw_components,
        curvature_components=curvature,
    ), drift


def predict_surface_hyperviscosity(
    operators: SurfaceOperators,
    points: jnp.ndarray,
    h: float,
    calibration: HyperviscosityCalibration,
) -> tuple[HyperviscosityCalibration, jnp.ndarray]:
    """Apply the coordinatewise curvature update between full calibrations."""

    power = int(calibration.power)
    updated, drift = _predict_geometry_update(
        operators,
        jnp.asarray(points, dtype=float),
        jnp.asarray(h, dtype=float),
        calibration,
        power=power,
    )
    return updated._replace(power=power), drift


@partial(jit, static_argnames=("power", "arnoldi_iterations"))
def _refresh_surface_hyperviscosity_on_device(
    operators: SurfaceOperators,
    points: jnp.ndarray,
    normals: jnp.ndarray,
    h: float,
    calibration: HyperviscosityCalibration,
    drift_threshold: float,
    *,
    power: int,
    arnoldi_iterations: int,
) -> tuple[HyperviscosityCalibration, jnp.ndarray, jnp.ndarray]:
    predicted, drift = _predict_geometry_update(
        operators,
        points,
        h,
        calibration,
        power=power,
    )

    def recalibrate(_):
        updated = _calibrate_fixed_power_on_device(
            operators,
            points,
            normals,
            h,
            power=power,
            arnoldi_iterations=arnoldi_iterations,
        )
        return updated, jnp.asarray(True)

    def accept_prediction(_):
        return predicted, jnp.asarray(False)

    updated, recalibrated = lax.cond(
        drift > drift_threshold,
        recalibrate,
        accept_prediction,
        operand=None,
    )
    return updated, drift, recalibrated


def refresh_surface_hyperviscosity(
    operators: SurfaceOperators,
    points: jnp.ndarray,
    normals: jnp.ndarray,
    h: float,
    calibration: HyperviscosityCalibration,
    *,
    drift_threshold: float = 0.15,
    arnoldi_iterations: int = 24,
) -> tuple[HyperviscosityCalibration, jnp.ndarray, jnp.ndarray]:
    """Predict or recalibrate entirely on device without a host drift branch."""

    power = int(calibration.power)
    updated, drift, recalibrated = _refresh_surface_hyperviscosity_on_device(
        operators,
        jnp.asarray(points, dtype=float),
        jnp.asarray(normals, dtype=float),
        jnp.asarray(h, dtype=float),
        calibration,
        jnp.asarray(drift_threshold, dtype=float),
        power=power,
        arnoldi_iterations=int(arnoldi_iterations),
    )
    return updated._replace(power=power), drift, recalibrated
