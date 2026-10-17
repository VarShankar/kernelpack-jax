"""Sampled SBDF3 stability screen for diffusion plus explicit full-square L^k.

This checker applies the actual three-full-state recurrence. It screens the
diffusion/explicit-hyperviscosity split only; advection, reaction, and
time-varying coefficients are outside its claim.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from ._common import SparseCOOMatrix, build_implicit_system_sparse, sparse_matvec
from ._cudss import CudssSystemCache
from ._explicit_laplacian_power import estimate_largest_magnitude_modes


def check_sparse_power_sbdf3_stability(
    laplacian_physical: SparseCOOMatrix,
    hyper_power_physical: SparseCOOMatrix,
    boundary: SparseCOOMatrix,
    *,
    n_physical: int,
    dt: float,
    nu: float,
    gamma: float,
    num_modes: int = 12,
    dimension: int = 80,
    restarts: int = 6,
    tolerance: float = 1e-8,
    seed: int = 1701,
    stability_tolerance: float = 1e-10,
) -> dict:
    """Estimate dominant modes of the actual homogeneous SBDF3 companion.

    State ordering is ``[interior; boundary; ghost]`` and the companion state
    is ``[u_n, u_{n-1}, u_{n-2}]``. The first block solves

    ``(I - 6 dt nu L_phys / 11) u_{n+1}``
    ``= (18 u_n - 9 u_{n-1} + 2 u_{n-2})_phys / 11``
    ``  + (6 dt gamma / 11) H (3 u_n - 3 u_{n-1} + u_{n-2})``,

    with homogeneous Dirichlet rows, where ``H`` is the supplied physical-row
    slice of the full-square ``L**k``. ``gamma`` is the summed scalar
    coefficient for this isotropic power operator. A prepared cuDSS factor is reused for
    every Arnoldi application. Only the small projected eigensystems are
    transferred to the host by the Arnoldi routine.

    A pass means the requested sampled Ritz modes converged and their
    amplification eigenvalues lie within the unit disk. It is not a
    full-spectrum or nonnormal-transient-growth certificate.
    """
    dt = float(dt)
    nu = float(nu)
    gamma = float(gamma)
    n_physical = int(n_physical)
    n_full = int(laplacian_physical.shape[1])
    if (not np.isfinite(dt) or dt <= 0 or not np.isfinite(nu) or nu <= 0
            or not np.isfinite(gamma) or n_physical < 1):
        raise ValueError("dt, nu, gamma, and n_physical must be finite and valid")
    if laplacian_physical.shape != hyper_power_physical.shape:
        raise ValueError("L and L^k physical-row operators must have matching shapes")
    if laplacian_physical.shape != (n_physical, n_full):
        raise ValueError("physical Laplacian shape must be (n_physical, n_full)")
    if boundary.shape[1] != n_full:
        raise ValueError("Dirichlet boundary operator must have n_full columns")
    if boundary.shape[0] != n_full - n_physical:
        raise ValueError("Dirichlet row count must close the full state system")
    _validate_nodal_dirichlet(boundary, n_physical)
    if num_modes < 1 or dimension < num_modes + 2 or restarts < 1:
        raise ValueError("Arnoldi requires num_modes>=1, dimension>=num_modes+2, restarts>=1")
    if tolerance <= 0 or stability_tolerance < 0:
        raise ValueError("tolerances must be nonnegative, with Arnoldi tolerance positive")

    alpha = 6.0 / 11.0
    implicit = build_implicit_system_sparse(
        laplacian_physical, boundary, n_physical, -alpha * dt * nu
    )
    cache = CudssSystemCache()
    try:
        factor = cache.prepare(implicit)

        def companion_matvec(flat_state):
            states = flat_state.reshape((3, n_full))
            current, previous, older = states[0], states[1], states[2]
            history = (18.0 * current[:n_physical]
                       - 9.0 * previous[:n_physical]
                       + 2.0 * older[:n_physical]) / 11.0
            explicit_history = 3.0 * current - 3.0 * previous + older
            rhs_physical = history + alpha * dt * gamma * sparse_matvec(
                hyper_power_physical.indices,
                hyper_power_physical.values,
                n_physical,
                explicit_history,
            )
            rhs = jnp.concatenate(
                [rhs_physical, jnp.zeros((boundary.shape[0],), dtype=flat_state.dtype)]
            )
            next_state = factor.solve(rhs)
            return jnp.concatenate([next_state, current, previous])

        result = estimate_largest_magnitude_modes(
            companion_matvec,
            3 * n_full,
            k=int(num_modes),
            dimension=int(dimension),
            restarts=int(restarts),
            tolerance=float(tolerance),
            seed=int(seed),
            dtype=laplacian_physical.values.dtype,
        )
        values = np.asarray(result.get("eigenvalues", ()), dtype=np.complex128)
        residuals = np.asarray(result.get("residuals", ()), dtype=np.float64)
        converged = bool(result.get("converged", False))
        finite = bool(np.isfinite(values).all() and np.isfinite(residuals).all())
        modes_complete = values.size >= num_modes
        spectral_radius = float(np.max(np.abs(values))) if values.size else None
        stable = bool(
            converged and finite and modes_complete
            and spectral_radius is not None
            and spectral_radius <= 1.0 + stability_tolerance
        )
        return {
            "passed": stable,
            "converged": converged,
            "finite": finite,
            "requested_modes_complete": bool(modes_complete),
            "reason": (
                "sampled companion modes within unit disk" if stable else
                "Arnoldi modes unresolved" if not converged else
                "requested Ritz modes missing" if not modes_complete else
                "nonfinite Arnoldi modes" if not finite else
                "sampled companion spectral radius exceeds unit disk"
            ),
            "claim_scope": (
                "sampled modes of homogeneous SBDF3 diffusion plus explicit L^k; "
                "excludes advection, reaction, time variation, and nonnormal transient growth"
            ),
            "operator": "three-full-state SBDF3 companion with physical rows of full-square L^k",
            "dt": dt,
            "nu": nu,
            "gamma": gamma,
            "alpha": alpha,
            "n_physical": n_physical,
            "n_full": n_full,
            "companion_dimension": 3 * n_full,
            "requested_modes": int(num_modes),
            "spectral_radius_estimate": spectral_radius,
            "stability_tolerance": float(stability_tolerance),
            "ritz_eigenvalues": values,
            "ritz_residuals": residuals,
            "arnoldi": result,
            "cudss_phase_counts": dict(factor.phase_counts),
        }
    finally:
        cache.close()


def select_sparse_power_sbdf3_timestep(
    laplacian_physical: SparseCOOMatrix,
    hyper_power_physical: SparseCOOMatrix,
    boundary: SparseCOOMatrix,
    *,
    n_physical: int,
    final_time: float,
    accuracy_steps: int,
    nu: float,
    gamma: float,
    max_halvings: int = 16,
    **screen_options,
) -> dict:
    """Choose the first dyadic step whose sampled companion modes pass.

    The baseline accuracy step is ``final_time / accuracy_steps``. Each
    attempt halves that step while preserving ``final_time`` and the supplied
    spatial operators and coefficients. An unresolved/nonfinite Arnoldi
    result stops the search immediately; it is not treated as instability
    evidence that can be repaired by halving.
    """
    final_time = float(final_time)
    accuracy_steps = int(accuracy_steps)
    max_halvings = int(max_halvings)
    if not np.isfinite(final_time) or final_time <= 0 or accuracy_steps < 1:
        raise ValueError("final_time and accuracy_steps must be positive and finite")
    if max_halvings < 0 or max_halvings > 16:
        raise ValueError("max_halvings must lie in [0, 16]")

    accuracy_dt = final_time / accuracy_steps
    attempts = []
    for halving in range(max_halvings + 1):
        actual_steps = accuracy_steps * (2 ** halving)
        dt = final_time / actual_steps
        screen = check_sparse_power_sbdf3_stability(
            laplacian_physical,
            hyper_power_physical,
            boundary,
            n_physical=n_physical,
            dt=dt,
            nu=nu,
            gamma=gamma,
            **screen_options,
        )
        residuals = np.asarray(screen.get("ritz_residuals", ()), dtype=np.float64)
        max_residual = float(np.max(residuals)) if residuals.size else None
        item = {
            "halvings": halving,
            "actual_steps": actual_steps,
            "dt": dt,
            "converged": bool(screen.get("converged", False)),
            "finite": bool(screen.get("finite", False)),
            "spectral_radius_estimate": screen.get("spectral_radius_estimate"),
            "max_ritz_residual": max_residual,
            "passed": bool(screen.get("passed", False)),
            "reason": screen.get("reason", "screen returned no reason"),
        }
        attempts.append(item)
        if item["passed"]:
            return {
                "passed": True,
                "reason": "first dyadic timestep with a passing sampled-mode screen",
                "final_time": final_time,
                "accuracy_steps": accuracy_steps,
                "accuracy_dt": accuracy_dt,
                "actual_steps": actual_steps,
                "actual_dt": dt,
                "halvings": halving,
                "nu": float(nu),
                "gamma": float(gamma),
                "attempts": attempts,
                "selected_screen": screen,
                "claim_scope": screen.get("claim_scope"),
            }
        if not item["converged"] or not item["finite"]:
            return {
                "passed": False,
                "reason": "Arnoldi screen unresolved or nonfinite; timestep search stopped",
                "final_time": final_time,
                "accuracy_steps": accuracy_steps,
                "accuracy_dt": accuracy_dt,
                "actual_steps": actual_steps,
                "actual_dt": dt,
                "halvings": halving,
                "nu": float(nu),
                "gamma": float(gamma),
                "attempts": attempts,
                "selected_screen": None,
                "claim_scope": screen.get("claim_scope"),
            }
    last = attempts[-1]
    return {
        "passed": False,
        "reason": "maximum timestep halvings reached without a passing sampled-mode screen",
        "final_time": final_time,
        "accuracy_steps": accuracy_steps,
        "accuracy_dt": accuracy_dt,
        "actual_steps": last["actual_steps"],
        "actual_dt": last["dt"],
        "halvings": last["halvings"],
        "nu": float(nu),
        "gamma": float(gamma),
        "attempts": attempts,
        "selected_screen": None,
        "claim_scope": "sampled companion modes only; no claim about unsampled spectrum",
    }


def _validate_nodal_dirichlet(boundary: SparseCOOMatrix, n_physical: int) -> None:
    """Require one identity equation fixing each boundary node value."""
    indices = np.asarray(boundary.indices, dtype=np.int64)
    values = np.asarray(boundary.values)
    n_boundary = boundary.shape[0]
    if indices.ndim != 2 or indices.shape[1] != 2 or values.shape != (indices.shape[0],):
        raise ValueError("Dirichlet COO arrays have invalid shapes")
    if not np.isfinite(values).all():
        raise ValueError("Dirichlet matrix contains nonfinite entries")
    # Physical ordering is [interior; boundary], followed by ghosts; the
    # boundary unknown block therefore begins at n_physical - n_boundary.
    expected = n_physical - n_boundary + np.arange(n_boundary, dtype=np.int64)
    diagonal_sum = np.zeros(n_boundary, dtype=np.float64)
    off_diagonal_abs_sum = np.zeros(n_boundary, dtype=np.float64)
    for (row, col), value in zip(indices, values):
        if row < 0 or row >= n_boundary or col < 0 or col >= boundary.shape[1]:
            raise ValueError("Dirichlet COO index lies outside its matrix")
        if col == expected[row]:
            diagonal_sum[row] += float(value)
        else:
            off_diagonal_abs_sum[row] += abs(float(value))
    if (np.max(np.abs(diagonal_sum - 1.0), initial=0.0) > 1e-12
            or np.max(off_diagonal_abs_sum, initial=0.0) > 1e-12):
        raise ValueError("stability screen requires nodal identity Dirichlet rows")
