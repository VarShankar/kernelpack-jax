"""Explicit application of prescribed-trace powers of a sparse Laplacian.

This module applies ``Delta**k`` by repeated sparse Laplacian actions and
ghost closure solves. It never assembles ``L**k`` or changes the implicit
diffusion matrix. Boundary traces of every intermediate field are explicit
inputs, supplied in order as ``c_j = (Delta**j u)|_B``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ._common import SparseCOOMatrix
from ._laplacian_power import (
    BoundaryClosedLaplacian,
    evaluate_boundary_trace_chain,
)


def apply_explicit_laplacian_power(
    blocks: BoundaryClosedLaplacian,
    full_state,
    boundary_traces,
    power: int,
    solve_ghost,
):
    """Apply the exact supplied-trace composition ``Delta**power``.

    The first Laplacian uses ``full_state`` as provided. Later stages use the
    caller's traces ``c_1,...,c_power`` and solve the existing ``L_BG`` block
    for their ghost values. The result has physical-row ordering
    ``[interior; boundary]``. It therefore matches an explicit PDE RHS, whose
    boundary rows are distinct from the state vector's ghost tail.

    ``solve_ghost(rhs)`` must use an eagerly prepared fixed factor when called
    in a production JIT. The function itself performs no setup or host work,
    so it composes with ``jit``, ``vmap``, JVP, and VJP.
    """
    return blocks.apply_from_full_state(
        full_state, boundary_traces, power, solve_ghost
    )


@dataclass
class PreparedExplicitLaplacianPower:
    """Own one reusable cuDSS factor for ``L_BG`` and a fixed Laplacian split."""

    blocks: BoundaryClosedLaplacian
    power: int
    _cache: object
    _factor: object

    @classmethod
    def prepare(
        cls,
        laplacian: SparseCOOMatrix,
        *,
        n_interior: int,
        n_boundary: int,
        power: int,
    ) -> "PreparedExplicitLaplacianPower":
        """Split the Laplacian and factor ``L_BG`` once on the configured GPU.

        The cuDSS import is delayed until this method is called. Call
        ``close()`` after all JIT/autodiff work using this prepared object has
        completed.
        """
        power = int(power)
        if power < 1:
            raise ValueError("power must be at least one")
        blocks = BoundaryClosedLaplacian.from_sparse(
            laplacian,
            n_interior=n_interior,
            n_boundary=n_boundary,
        )
        matrix = SparseCOOMatrix(
            blocks.L_BG.indices,
            blocks.L_BG.values,
            blocks.L_BG.shape,
        )
        from ._cudss import CudssSystemCache

        cache = CudssSystemCache()
        try:
            factor = cache.prepare(matrix)
        except BaseException:
            cache.close()
            raise
        return cls(blocks=blocks, power=power, _cache=cache, _factor=factor)

    def apply_with_traces(self, full_state, boundary_traces):
        """Apply using already evaluated traces, reusing the fixed ``L_BG`` factor."""
        return apply_explicit_laplacian_power(
            self.blocks,
            full_state,
            boundary_traces,
            self.power,
            self._factor.solve,
        )

    def apply_homogeneous(self, interior, power: int = 1):
        """Apply ``A**power`` to an interior vector with homogeneous closure.

        This is the matrix-free operator required by largest-magnitude
        Arnoldi. It shares the same eagerly prepared ``L_BG`` factor as the
        prescribed-trace explicit hyperviscosity path.
        """
        return self.blocks.apply_homogeneous_power(
            interior, power, self._factor.solve
        )

    def apply(self, full_state, time, boundary_coordinates, trace_callback):
        """Evaluate ``c_j(time, x_b)`` and apply the prescribed-trace power."""
        traces = evaluate_boundary_trace_chain(
            trace_callback,
            time,
            boundary_coordinates,
            self.power,
        )
        return self.apply_with_traces(full_state, traces)

    @property
    def phase_counts(self):
        """Expose cuDSS setup/refactor counts for reuse checks; no timings."""
        return self._factor.phase_counts

    def close(self):
        """Release the single prepared ghost-closure factor."""
        self._cache.close()


def imex_bdf3_sbdf3_amplification_roots(z_implicit, z_explicit):
    """Return scalar SBDF3 amplification roots for one split mode.

    For ``u' = lambda_I*u + lambda_E*u``, with diffusion implicit and the
    explicit term extrapolated by the solver's third-order SBDF formula, the
    characteristic polynomial is

    ``(11-6*zI) r^3 - (18+18*zE) r^2``
    ``+ (9+18*zE) r - (2+6*zE)``.

    This small NumPy utility supports offline stability checks after a
    largest-magnitude Arnoldi estimate of the homogeneous boundary-closed
    Laplacian, using ``zI=dt*nu*lambda`` and ``zE=dt*gamma*lambda**k`` for a
    diffusion/hyperviscosity-only split. It does not estimate that spectrum,
    include noncommuting advection/reaction terms, or certify stability of a
    nonnormal matrix.
    """
    zi = complex(z_implicit)
    ze = complex(z_explicit)
    coeffs = np.asarray(
        [
            11.0 - 6.0 * zi,
            -18.0 - 18.0 * ze,
            9.0 + 18.0 * ze,
            -2.0 - 6.0 * ze,
        ],
        dtype=np.complex128,
    )
    if not np.all(np.isfinite(coeffs)) or coeffs[0] == 0:
        raise ValueError("SBDF3 characteristic polynomial is singular or nonfinite")
    return np.roots(coeffs)


def estimate_largest_magnitude_modes(
    apply_homogeneous_laplacian,
    n_interior: int,
    *,
    k: int = 12,
    dimension: int = 80,
    restarts: int = 6,
    tolerance: float = 1e-8,
    seed: int = 1701,
    dtype=jnp.float64,
):
    """Estimate dominant homogeneous-``A`` modes with a device Arnoldi basis.

    The projected eigensystem and Schur restart are small host operations;
    Krylov basis construction and all ``A`` matvecs stay on the JAX device.
    The result is a Ritz sample, not an enclosure of the full spectrum.
    """
    from .spectral import arnoldi_largest_magnitude

    return arnoldi_largest_magnitude(
        apply_homogeneous_laplacian,
        int(n_interior),
        num_modes=int(k),
        dimension=min(int(dimension), int(n_interior)),
        restarts=int(restarts),
        seeds=(int(seed),),
        relative_tolerance=float(tolerance),
        dtype=dtype,
    )


def check_sbdf3_stability_from_laplacian_modes(
    eigenvalues,
    *,
    dt: float,
    nu: float,
    gamma,
    power: int,
    stability_tolerance: float = 1e-10,
):
    """Screen sampled homogeneous-A Ritz modes for the SBDF3 split.

    The implicit mode is ``nu*lambda`` and the explicit hyperviscosity mode is
    ``sum(gamma)*lambda**power``. A result can pass only when every supplied
    mode has all three amplification roots within the unit disk (up to the
    requested tolerance). Since the inputs are Ritz modes, a pass is a sampled
    modal screen, not a stability proof for the full or nonnormal operator.
    """
    dt = float(dt)
    nu = float(nu)
    power = int(power)
    stability_tolerance = float(stability_tolerance)
    gamma_sum = float(np.sum(np.asarray(gamma)))
    values = np.asarray(eigenvalues, dtype=np.complex128).reshape(-1)
    if (not np.isfinite(dt) or dt <= 0 or not np.isfinite(nu)
            or not np.isfinite(gamma_sum) or power < 1
            or not np.isfinite(stability_tolerance) or stability_tolerance < 0):
        raise ValueError("SBDF3 stability inputs must be finite with dt>0 and power>=1")
    modes = []
    for eigenvalue in values:
        z_implicit = dt * nu * eigenvalue
        z_explicit = dt * gamma_sum * eigenvalue**power
        roots = imex_bdf3_sbdf3_amplification_roots(z_implicit, z_explicit)
        maximum = float(np.max(np.abs(roots)))
        modes.append({
            "lambda_real": float(eigenvalue.real),
            "lambda_imag": float(eigenvalue.imag),
            "z_implicit_real": float(z_implicit.real),
            "z_implicit_imag": float(z_implicit.imag),
            "z_explicit_real": float(z_explicit.real),
            "z_explicit_imag": float(z_explicit.imag),
            "max_root_modulus": maximum,
            "roots_within_unit_disk": bool(maximum <= 1.0 + stability_tolerance),
        })
    passed = bool(modes) and all(mode["roots_within_unit_disk"] for mode in modes)
    return {
        "passed_sampled_modes": passed,
        "claim_scope": "sampled Ritz modes only; not a full-spectrum or nonnormal-stability proof",
        "dt": dt,
        "nu": nu,
        "gamma_sum": gamma_sum,
        "power": power,
        "stability_tolerance": stability_tolerance,
        "modes": modes,
    }


def select_sbdf3_timestep_from_laplacian_modes(
    eigenvalues,
    *,
    final_time: float,
    accuracy_steps: int,
    nu: float,
    gamma,
    power: int,
    safety_factor: float = 0.8,
    max_halvings: int = 32,
    generator_tolerance: float = 1e-10,
):
    """Choose dyadic SBDF3 stability substeps without changing accuracy data.

    The requested accuracy step is ``final_time / accuracy_steps``. Stability
    subdivisions double the step count while preserving ``final_time``. A
    sampled positive real part of ``nu*lambda + sum(gamma)*lambda**power``
    fails immediately: reducing ``dt`` cannot remove continuous growth.

    For real negative modes with a stabilizing explicit contribution, the
    exact Jury bound ``dt < 20/(21*b-3*a)`` gives an initial halving estimate,
    where ``a=-nu*lambda`` and ``b=-sum(gamma)*lambda**power``. The final
    choice is then checked against all supplied Ritz modes, including complex
    modes, using the SBDF3 characteristic roots. `generator_tolerance` is a
    relative threshold against ``max(1, |nu*lambda|+|gamma*lambda**k|)``.

    This is a sampled-mode stability screen, not an accuracy estimate or a
    proof for omitted modes/non-normal transient growth. No optimizer or PDE
    rollout is run here.
    """
    final_time = float(final_time)
    accuracy_steps = int(accuracy_steps)
    nu = float(nu)
    power = int(power)
    safety_factor = float(safety_factor)
    max_halvings = int(max_halvings)
    generator_tolerance = float(generator_tolerance)
    gamma_sum = float(np.sum(np.asarray(gamma)))
    modes = np.asarray(eigenvalues, dtype=np.complex128).reshape(-1)
    if (not np.isfinite(final_time) or final_time <= 0.0
            or accuracy_steps <= 0 or not np.isfinite(nu) or nu <= 0.0
            or not np.isfinite(gamma_sum) or power < 1
            or not (0.0 < safety_factor < 1.0)
            or max_halvings < 0 or not np.isfinite(generator_tolerance)
            or generator_tolerance < 0.0):
        raise ValueError("invalid SBDF3 timestep selection inputs")
    accuracy_dt = final_time / accuracy_steps

    if modes.size == 0 or not np.all(np.isfinite(modes)):
        return {
            "passed": False,
            "reason": "no finite Laplacian Ritz modes were supplied",
            "accuracy_steps": accuracy_steps,
            "accuracy_dt": accuracy_dt,
            "actual_steps": accuracy_steps,
            "actual_dt": accuracy_dt,
            "halvings": 0,
            "screen": check_sbdf3_stability_from_laplacian_modes(
                [], dt=accuracy_dt, nu=nu, gamma=[gamma_sum], power=power
            ),
            "generator_check": {"passed": False, "modes": [], "max_real_part": None,
                                "reason": "no finite modes"},
            "explicit_stability_bound_estimated": None,
        }

    generator_records = []
    positive_modes = []
    finite_bounds = []
    for eigenvalue in modes:
        implicit_generator = nu * eigenvalue
        explicit_generator = gamma_sum * eigenvalue**power
        total_generator = implicit_generator + explicit_generator
        scale = max(1.0, abs(implicit_generator) + abs(explicit_generator))
        threshold = generator_tolerance * scale
        is_positive = bool(total_generator.real > threshold)
        record = {
            "lambda_real": float(eigenvalue.real),
            "lambda_imag": float(eigenvalue.imag),
            "generator_real": float(total_generator.real),
            "generator_imag": float(total_generator.imag),
            "positive_tolerance": float(threshold),
            "positive_growth": is_positive,
        }
        generator_records.append(record)
        if is_positive:
            positive_modes.append(record)

        real_tolerance = 1e-12 * max(1.0, abs(eigenvalue.real))
        if abs(eigenvalue.imag) <= real_tolerance and eigenvalue.real < 0.0:
            a = -nu * float(eigenvalue.real)
            b = -float((gamma_sum * eigenvalue**power).real)
            denominator = 21.0 * b - 3.0 * a
            if a > 0.0 and b >= 0.0 and denominator > 0.0:
                finite_bounds.append(20.0 / denominator)

    estimated_bound = min(finite_bounds) if finite_bounds else None
    generator_passed = not positive_modes
    generator_check = {
        "passed": generator_passed,
        "criterion": "Re(nu*lambda + sum(gamma)*lambda**power) <= relative tolerance",
        "relative_tolerance": generator_tolerance,
        "max_real_part": max(record["generator_real"] for record in generator_records),
        "positive_growth_count": len(positive_modes),
        "modes": generator_records,
    }

    if not generator_passed:
        screen = check_sbdf3_stability_from_laplacian_modes(
            modes, dt=accuracy_dt, nu=nu, gamma=[gamma_sum], power=power
        )
        return {
            "passed": False,
            "reason": "sampled continuous generator has positive growth",
            "accuracy_steps": accuracy_steps,
            "accuracy_dt": accuracy_dt,
            "actual_steps": accuracy_steps,
            "actual_dt": accuracy_dt,
            "halvings": 0,
            "screen": screen,
            "generator_check": generator_check,
            "explicit_stability_bound_estimated": estimated_bound,
        }

    halvings = 0
    if estimated_bound is not None:
        target_dt = safety_factor * estimated_bound
        if accuracy_dt > target_dt:
            halvings = int(np.ceil(np.log2(accuracy_dt / target_dt)))
    halvings = min(halvings, max_halvings)

    screen = None
    actual_steps = accuracy_steps * (2**halvings)
    actual_dt = final_time / actual_steps
    while True:
        screen = check_sbdf3_stability_from_laplacian_modes(
            modes, dt=actual_dt, nu=nu, gamma=[gamma_sum], power=power
        )
        if screen["passed_sampled_modes"]:
            return {
                "passed": True,
                "reason": "all supplied Ritz modes pass the SBDF3 root screen",
                "accuracy_steps": accuracy_steps,
                "accuracy_dt": accuracy_dt,
                "actual_steps": actual_steps,
                "actual_dt": actual_dt,
                "halvings": halvings,
                "screen": screen,
                "generator_check": generator_check,
                "explicit_stability_bound_estimated": estimated_bound,
            }
        if halvings >= max_halvings:
            return {
                "passed": False,
                "reason": "SBDF3 root screen still fails at maximum halvings",
                "accuracy_steps": accuracy_steps,
                "accuracy_dt": accuracy_dt,
                "actual_steps": actual_steps,
                "actual_dt": actual_dt,
                "halvings": halvings,
                "screen": screen,
                "generator_check": generator_check,
                "explicit_stability_bound_estimated": estimated_bound,
            }
        halvings += 1
        actual_steps = accuracy_steps * (2**halvings)
        actual_dt = final_time / actual_steps
