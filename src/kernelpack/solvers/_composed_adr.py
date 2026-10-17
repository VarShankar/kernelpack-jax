"""Legacy prescribed-trace solves and sparse-power spectral diagnostics.

The legacy fixed sparse auxiliary layout represents the power without materializing
the Schur complement or its powers. Boundary traces are problem data, supplied
explicitly rather than inferred from a Dirichlet value's arbitrary extension.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ._common import (
    SparseCOOMatrix, solve_dense_system, solve_sparse_system_gmres,
    sparse_matrix_to_dense,
)
from ._cudss import (
    CudssSystemCache, _transient_signature, get_transient_cudss_factor,
    solve_sparse_system_cudss,
)
from ._laplacian_power import (
    PreparedAugmentedLaplacianLayout, evaluate_boundary_trace_chain,
)
from .spectral import arnoldi_rightmost


def prepare_power_layout(base):
    """Prepare static structure once after the original Dirichlet rows exist."""
    if base.lap_sparse is None or base.bc_sparse is None:
        raise ValueError("Prepare Laplacian and Dirichlet operators first")
    key = (id(base.lap_sparse), id(base.bc_sparse), int(base.hyperviscosity_power))
    if getattr(base, "power_layout_key", None) == key:
        return base.power_layout
    ni = base.domain.get_num_interior_nodes()
    nb = base.domain.get_num_bdry_nodes()
    ij = np.asarray(base.bc_sparse.indices)
    values = np.asarray(base.bc_sparse.values)
    discrepancy = values - np.where(ij[:, 1] == ni + ij[:, 0], 1.0, 0.0)
    row_error = np.zeros(nb)
    np.add.at(row_error, ij[:, 0], np.abs(discrepancy))
    if row_error.max(initial=0.0) > 1e-10:
        raise ValueError("Laplacian-power ADR currently requires nodal Dirichlet boundaries")
    base.power_layout = PreparedAugmentedLaplacianLayout.from_operators(
        base.lap_sparse, base.bc_sparse, n_interior=ni, n_boundary=nb,
        power=int(base.hyperviscosity_power),
        stage_scale=float(base.domain.get_sep_rad())**2,
    )
    base.power_layout_key = key
    return base.power_layout


def power_system(base, alpha, gamma=None):
    """Pure JAX matrix values on an already prepared sparse layout."""
    layout = base.power_layout
    gamma = base.gamma_components if gamma is None else gamma
    return layout.system(-alpha * base.dt * base.nu,
                         -alpha * base.dt * jnp.sum(gamma))


def prepare_power_cudss(base, orders=(1, 2, 3)):
    """Factor the requested fixed BDF systems, once per coefficient setting."""
    prepare_power_layout(base)
    signature = _transient_signature(base)
    if (signature is not None
            and signature == getattr(base, "cudss_signature", None)
            and all(order in getattr(base, "cudss_factors", {}) for order in orders)):
        return
    if getattr(base, "cudss_cache", None) is not None:
        base.cudss_cache.close()
    if getattr(base, "laplacian_power_solver", "augmented") == "factorized":
        from ._factorized_laplacian_power import PreparedFactorizedLaplacianPower
        factors = {}
        try:
            for order in orders:
                alpha = base._bdf_alpha(order)
                factors[order] = PreparedFactorizedLaplacianPower.prepare(
                    base.lap_sparse,
                    n_interior=base.domain.get_num_interior_nodes(),
                    n_boundary=base.domain.get_num_bdry_nodes(),
                    power=int(base.hyperviscosity_power),
                    ell=-alpha * float(base.dt) * float(base.nu),
                    eta=-alpha * float(base.dt) * float(np.sum(np.asarray(base.gamma_components))),
                    scale=float(base.domain.get_sep_rad())**2)
        except BaseException:
            for factor in factors.values():
                factor.close()
            raise
        class FactorCollection:
            def close(self):
                for factor in factors.values():
                    factor.close()
                factors.clear()
        base.cudss_cache = FactorCollection()
        base.cudss_factors = factors
        base.cudss_signature = signature
        return
    cache = CudssSystemCache()
    try:
        factors = {order: cache.prepare(power_system(base, base._bdf_alpha(order)))
                   for order in orders}
    except BaseException:
        cache.close()
        raise
    base.cudss_cache = cache
    base.cudss_factors = factors
    base.cudss_signature = signature


def _registered_variant_id(base, gamma, order):
    """Check that a frozen factor tag still belongs to this eager solver state."""
    pattern = getattr(base, "cudss_variant_pattern", None)
    variant_ids = getattr(base, "cudss_variant_ids", None)
    signature = getattr(base, "cudss_variant_signature", None)
    if pattern is None or variant_ids is None or signature is None:
        return None
    if _transient_signature(base) != signature:
        return None
    variant_id = variant_ids.get(order)
    if variant_id is None:
        return None
    if gamma is base.gamma_components:
        return variant_id
    if isinstance(gamma, jax.core.Tracer):
        return None
    try:
        return variant_id if np.array_equal(
            np.asarray(gamma), np.asarray(base.gamma_components)
        ) else None
    except (TypeError, ValueError):
        return None


def solve_power_step(base, rhs_physical, rhs_boundary, time, alpha, gamma):
    """Solve the augmented system and return only the original full state."""
    layout = base.power_layout
    callback = getattr(base, "hyperviscosity_boundary_traces", None)
    if callback is None:
        raise ValueError("Laplacian powers require an explicit boundary-trace callback")
    traces = evaluate_boundary_trace_chain(
        callback, time, base.xb, int(base.hyperviscosity_power),
    )
    if (base.linear_solver == "cudss"
            and getattr(base, "laplacian_power_solver", "augmented") == "factorized"):
        order = 1 if alpha == 1.0 else 2 if alpha == 2.0 / 3.0 else 3
        factor = get_transient_cudss_factor(base, order, gamma)
        if factor is not None:
            return factor.solve(rhs_physical, rhs_boundary, traces[1:])
        # Changing/traced coefficients retain the original augmented path.
    scale = -alpha * base.dt * jnp.sum(gamma)
    rhs = layout.rhs(rhs_physical, rhs_boundary, traces, scale)
    if base.linear_solver == "cudss":
        order = 1 if alpha == 1.0 else 2 if alpha == 2.0 / 3.0 else 3
        variant_id = _registered_variant_id(base, gamma, order)
        if variant_id is not None:
            return base.cudss_variant_pattern.solve_variant(
                variant_id, rhs
            )[:base.nf]
        factor = get_transient_cudss_factor(base, order, gamma)
        if factor is not None:
            solution = factor.solve(rhs)
        else:
            solution = solve_sparse_system_cudss(power_system(base, alpha, gamma), rhs)
    elif base.linear_solver in ("dense", "lineax"):
        solution = solve_dense_system(
            sparse_matrix_to_dense(power_system(base, alpha, gamma)), rhs,
        )
    elif base.linear_solver in ("gmres", "gmres_ilu"):
        solution = solve_sparse_system_gmres(
            power_system(base, alpha, gamma), rhs,
            preconditioner=base.fixed_preconditioner,
            tol=base.gmres_tol, atol=base.gmres_atol,
            restart=base.gmres_restart, maxiter=base.gmres_maxiter,
        )
    else:
        raise ValueError(f"Unsupported Laplacian-power solver: {base.linear_solver}")
    return solution[:base.nf]


def power_generator_precheck(base, gamma=None):
    """Check the continuous generator of the selected power formulation.

    The current sparse-power path uses the actual full-square power cropped
    to physical rows. Historical prescribed-trace runs use their closed
    operator polynomial. Neither is inferred from raw Laplacian eigenvalues.
    """
    if getattr(base, "hyperviscosity_method", None) == "laplacian_power":
        return _sparse_power_generator_precheck(base, gamma)
    explicit = getattr(base, "hyperviscosity_treatment", "implicit") == "explicit"
    if explicit:
        base._prepare_explicit_hyperviscosity()
        blocks = base.explicit_power_evaluator.blocks
    else:
        blocks = prepare_power_layout(base).blocks
    gamma = base.gamma_components if gamma is None else gamma
    cache = CudssSystemCache()
    try:
        ghost = blocks.L_BG
        factor = (base.explicit_power_evaluator._factor if explicit else
                  cache.prepare(SparseCOOMatrix(ghost.indices, ghost.values, ghost.shape)))
        @jax.jit
        def matvec(vector):
            lap = blocks.apply_homogeneous_power(vector, 1, factor.solve)
            hyper = blocks.apply_homogeneous_power(
                lap, int(base.hyperviscosity_power) - 1, factor.solve,
            )
            return base.nu * lap + jnp.sum(gamma) * hyper
        dimension = min(80, blocks.n_interior)
        result = arnoldi_rightmost(matvec, blocks.n_interior,
            dimension=dimension, restarts=6, dtype=base.lap_sparse.values.dtype)
        while (result["reason"] == "rightmost Arnoldi search did not converge"
               and dimension < min(320, blocks.n_interior)):
            dimension = min(2 * dimension, 320, blocks.n_interior)
            result = arnoldi_rightmost(matvec, blocks.n_interior,
                dimension=dimension, restarts=6, dtype=base.lap_sparse.values.dtype)
        result.update(operator="nu*A + sum(gamma)*A^k, boundary-closed repeated Laplacian",
                      power=int(base.hyperviscosity_power),
                      final_krylov_dimension_limit=dimension)
        return result
    finally:
        cache.close()


def _sparse_power_generator_precheck(base, gamma=None):
    """Arnoldi on the actual cropped-power generator with Dirichlet rows.

    Form D = nu*L_physical + Gamma*(L_full**k)_physical. Eliminate
    homogeneous Dirichlet ghosts using D's own boundary rows. This is not
    a polynomial of a separately boundary-closed Laplacian.
    """
    from ._laplacian_power import BoundaryClosedLaplacian
    if base.hyper_sparse is None:
        raise ValueError("Assemble the full square sparse power before checking its generator")
    gamma = base.gamma_components if gamma is None else gamma
    operator = SparseCOOMatrix(
        jnp.concatenate([base.lap_sparse.indices, base.hyper_sparse.indices], axis=0),
        jnp.concatenate([base.nu * base.lap_sparse.values,
                         jnp.sum(gamma) * base.hyper_sparse.values]),
        base.lap_sparse.shape,
    )
    blocks = BoundaryClosedLaplacian.from_sparse(
        operator, n_interior=base.domain.get_num_interior_nodes(),
        n_boundary=base.domain.get_num_bdry_nodes(),
    )
    cache = CudssSystemCache()
    try:
        ghost = blocks.L_BG
        factor = cache.prepare(SparseCOOMatrix(ghost.indices, ghost.values, ghost.shape))
        @jax.jit
        def matvec(vector):
            return blocks.apply_homogeneous_power(vector, 1, factor.solve)
        dimension = min(80, blocks.n_interior)
        result = arnoldi_rightmost(matvec, blocks.n_interior,
            dimension=dimension, restarts=6, dtype=operator.values.dtype)
        while (result["reason"] == "rightmost Arnoldi search did not converge"
               and dimension < min(320, blocks.n_interior)):
            dimension = min(2 * dimension, 320, blocks.n_interior)
            result = arnoldi_rightmost(matvec, blocks.n_interior,
                dimension=dimension, restarts=6, dtype=operator.values.dtype)
        result.update(
            operator="Dirichlet-closed nu*L_physical + sum(gamma)*(L_full^k)_physical",
            power=int(base.hyperviscosity_power),
            power_construction="full_square_sparse_power_then_physical_row_crop",
            final_krylov_dimension_limit=dimension,
        )
        return result
    finally:
        cache.close()
