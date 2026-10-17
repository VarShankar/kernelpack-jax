"""Sparse fixed-coefficient factorization of ``I + ell*A + eta*A**k``.

Here ``A = L_II - L_IG L_BG**-1 L_BI`` is the homogeneous boundary-closed
Laplacian on interior nodes. Polynomial factors are represented by sparse
shifted-Laplacian systems over the original interior/ghost blocks; neither
``A`` nor a power of it is assembled. Nonzero and time-dependent Dirichlet
data enter through the affine recurrence in :meth:`solve`.

The polynomial coefficients, root factors, and cuDSS factors are setup-time
constants. Runtime work consists of sparse products and fixed-factor solves,
so input/RHS differentiation remains available through JAX.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ._common import SparseCOOMatrix, sparse_matvec
from ._laplacian_power import BoundaryClosedLaplacian, _block_matvec


def _roots_and_factors(ell: float, eta: float, power: int, scale: float):
    """Factor ``1 + (ell/scale) z + (eta/scale**k) z**k`` over R."""
    if power < 2 or not np.all(np.isfinite([ell, eta, scale])):
        raise ValueError("power must be >=2 and coefficients must be finite")
    if ell == 0.0 or eta == 0.0 or scale <= 0.0:
        raise ValueError("ell, eta, and scale must be nonzero with scale positive")
    coefficients = np.zeros(power + 1, dtype=np.float64)
    coefficients[0] = 1.0
    coefficients[1] += ell / scale
    coefficients[power] += eta / scale**power
    roots = np.polynomial.polynomial.polyroots(coefficients)
    if np.any(np.abs(roots) <= 1e-14 * max(1.0, float(np.max(np.abs(roots))))):
        raise ValueError("a zero polynomial root cannot be represented by a shift factor")

    unused = set(range(power))
    factors = []
    while unused:
        i = min(unused)
        root = roots[i]
        if abs(root.imag) <= 2e-10 * max(1.0, abs(root)):
            factors.append(("real", float(root.real)))
            unused.remove(i)
            continue
        if root.imag < 0.0:
            root = np.conjugate(root)
        partners = [(j, abs(roots[j] - root)) for j in unused if j != i]
        if not partners:
            raise ValueError("complex polynomial root has no conjugate partner")
        j, mismatch = min(partners, key=lambda item: item[1])
        if mismatch > 2e-8 * max(1.0, abs(root)):
            raise ValueError("polynomial roots are not conjugate within tolerance")
        factors.append(("pair", complex(root)))
        unused.remove(i)
        unused.remove(j)

    reconstructed = np.asarray([1.0 + 0.0j])
    for kind, root in factors:
        linear = np.asarray([1.0, -1.0 / root], dtype=np.complex128)
        if kind == "pair":
            linear = np.polynomial.polynomial.polymul(linear, np.conjugate(linear))
        reconstructed = np.polynomial.polynomial.polymul(reconstructed, linear)
    reconstructed_real = reconstructed.real
    errors = np.abs(reconstructed_real - coefficients)
    # Scale coefficient residuals at a representative root radius. Large-root
    # polynomials can have nominally zero intermediate coefficients whose
    # raw absolute residual looks large but is tiny in the polynomial's
    # natural evaluation coordinates.
    reference_radius = float(np.median(np.abs(roots)))
    if not np.isfinite(reference_radius) or reference_radius <= 0.0:
        raise ValueError("polynomial roots do not define a finite reference radius")
    exponents = np.arange(coefficients.size, dtype=np.float64)
    log_weights = exponents * np.log(reference_radius)
    weights = np.exp(log_weights - np.max(log_weights))
    weighted_scale = float(np.sum(np.abs(coefficients) * weights))
    weighted_error = float(np.max(errors * weights))
    if (not np.isfinite(weighted_scale) or weighted_scale <= 0.0
            or not np.isfinite(weighted_error)
            or weighted_error > 3e-9 * weighted_scale):
        raise ValueError("real root factors do not reproduce the normalized polynomial")
    return factors


def _coo_matrix(entries, shape, dtype):
    if entries:
        rows = np.concatenate([item[0] for item in entries]).astype(np.int32, copy=False)
        cols = np.concatenate([item[1] for item in entries]).astype(np.int32, copy=False)
        values = np.concatenate([item[2] for item in entries]).astype(dtype, copy=False)
        indices = np.column_stack((rows, cols)).astype(np.int32, copy=False)
    else:
        indices = np.zeros((0, 2), dtype=np.int32)
        values = np.zeros((0,), dtype=dtype)
    return SparseCOOMatrix(jnp.asarray(indices), jnp.asarray(values), tuple(shape))


def _append_block(entries, block, row_offset, col_offset, multiplier=1.0):
    indices = np.asarray(block.indices, dtype=np.int32)
    values = np.asarray(block.values)
    if indices.size:
        entries.append((
            indices[:, 0] + int(row_offset),
            indices[:, 1] + int(col_offset),
            np.asarray(multiplier, dtype=values.dtype) * values,
        ))


def _append_diagonal(entries, rows, cols, values, dtype):
    rows = np.asarray(rows, dtype=np.int32).reshape(-1)
    cols = np.asarray(cols, dtype=np.int32).reshape(-1)
    values = np.broadcast_to(np.asarray(values, dtype=dtype), rows.shape)
    if rows.size:
        entries.append((rows, cols, values))


def _real_shift_matrix(blocks: BoundaryClosedLaplacian, root: float, scale: float):
    ni, ng = blocks.n_interior, blocks.n_ghost
    dtype = np.asarray(blocks.L_II.values).dtype
    entries = []
    ids = np.arange(ni, dtype=np.int32)
    _append_diagonal(entries, ids, ids, 1.0, dtype)
    _append_block(entries, blocks.L_II, 0, 0, -scale / root)
    _append_block(entries, blocks.L_IG, 0, ni, -scale / root)
    _append_block(entries, blocks.L_BI, ni, 0)
    _append_block(entries, blocks.L_BG, ni, ni)
    return _coo_matrix(entries, (ni + ng, ni + ng), dtype)


def _pair_shift_matrix(blocks: BoundaryClosedLaplacian, root: complex, scale: float):
    """Build the real embedding for one conjugate pair (no dense Schur power)."""
    ni, ng = blocks.n_interior, blocks.n_ghost
    a, b = float(root.real), float(root.imag)
    if b <= 0.0:
        raise ValueError("pair representative must have positive imaginary part")
    radius = abs(root)
    dtype = np.asarray(blocks.L_II.values).dtype
    entries = []
    i = np.arange(ni, dtype=np.int32)
    g = np.arange(ng, dtype=np.int32)
    # Ordering is [x_real, x_imag, z_real, z_imag].
    _append_block(entries, blocks.L_II, 0, 0, scale / radius)
    _append_diagonal(entries, i, i, -a / radius, dtype)
    _append_diagonal(entries, i, ni + i, b / radius, dtype)
    _append_block(entries, blocks.L_IG, 0, 2 * ni, scale / radius)
    _append_diagonal(entries, ni + i, i, -b / radius, dtype)
    _append_block(entries, blocks.L_II, ni, ni, scale / radius)
    _append_diagonal(entries, ni + i, ni + i, -a / radius, dtype)
    _append_block(entries, blocks.L_IG, ni, 2 * ni + ng, scale / radius)
    _append_block(entries, blocks.L_BI, 2 * ni, 0)
    _append_block(entries, blocks.L_BG, 2 * ni, 2 * ni)
    _append_block(entries, blocks.L_BI, 2 * ni + ng, ni)
    _append_block(entries, blocks.L_BG, 2 * ni + ng, 2 * ni + ng)
    size = 2 * (ni + ng)
    return _coo_matrix(entries, (size, size), dtype)


@dataclass
class PreparedFactorizedLaplacianPower:
    """Own reusable cuDSS factors for a fixed ``I+ell*A+eta*A**k`` system."""

    blocks: BoundaryClosedLaplacian
    power: int
    ell: float
    eta: float
    scale: float
    factors: tuple
    _cache: object
    _ghost_factor: object
    _variant_resources: object | None = None
    _variant_ids: tuple[int, ...] | None = None

    @classmethod
    def prepare(
        cls,
        laplacian: SparseCOOMatrix,
        *,
        n_interior: int,
        n_boundary: int,
        power: int,
        ell: float,
        eta: float,
        scale: float,
    ) -> "PreparedFactorizedLaplacianPower":
        """Prepare ghost and polynomial factors once on the configured GPU."""
        power = int(power)
        ell, eta, scale = float(ell), float(eta), float(scale)
        blocks = BoundaryClosedLaplacian.from_sparse(
            laplacian, n_interior=n_interior, n_boundary=n_boundary
        )
        roots = _roots_and_factors(ell, eta, power, scale)
        dtype = np.asarray(laplacian.values).dtype
        ghost_matrix = SparseCOOMatrix(
            blocks.L_BG.indices, blocks.L_BG.values, blocks.L_BG.shape
        )
        from ._cudss import CudssSystemCache

        cache = CudssSystemCache()
        try:
            ghost_factor = cache.prepare(ghost_matrix)
            prepared = []
            for kind, root in roots:
                if kind == "real":
                    matrix = _real_shift_matrix(blocks, root, scale)
                else:
                    matrix = _pair_shift_matrix(blocks, root, scale)
                if np.asarray(matrix.values).dtype != dtype:
                    raise TypeError("shift factor and Laplacian dtypes must match")
                prepared.append((kind, root, cache.prepare(matrix)))
        except BaseException:
            cache.close()
            raise
        return cls(
            blocks=blocks,
            power=power,
            ell=ell,
            eta=eta,
            scale=scale,
            factors=tuple(prepared),
            _cache=cache,
            _ghost_factor=ghost_factor,
        )

    @classmethod
    def prepare_variants(
        cls,
        laplacian: SparseCOOMatrix,
        *,
        n_interior: int,
        n_boundary: int,
        power: int,
        configs: dict,
        scale: float,
    ) -> dict:
        """Prepare fixed coefficient variants over shared sparse patterns.

        ``configs`` maps immutable tags to ``(ell, eta)``. A single ghost
        closure factor and one native variant pattern per polynomial root
        position are shared by all returned solver views. Every variant is
        registered eagerly, and root-factor kinds must have the same sequence
        so the sparse patterns are exactly comparable.
        """
        if not isinstance(configs, dict) or len(configs) < 2:
            raise ValueError("at least two fixed coefficient configurations are required")
        if len(set(configs)) != len(configs):
            raise ValueError("factorized variant tags must be unique")
        scale = float(scale)
        power = int(power)
        roots_by_tag = {tag: _roots_and_factors(float(ell), float(eta), power, scale)
                        for tag, (ell, eta) in configs.items()}
        first_tag = next(iter(configs))
        factor_kinds = tuple(kind for kind, _ in roots_by_tag[first_tag])
        for tag, roots in roots_by_tag.items():
            if tuple(kind for kind, _ in roots) != factor_kinds:
                raise ValueError(f"factorized root kinds differ from shared pattern at tag {tag!r}")

        blocks = BoundaryClosedLaplacian.from_sparse(
            laplacian, n_interior=n_interior, n_boundary=n_boundary
        )
        from ._cudss import CudssSparsePattern, CudssSystemCache

        ghost_cache = CudssSystemCache()
        patterns = []
        try:
            ghost_factor = ghost_cache.prepare(SparseCOOMatrix(
                blocks.L_BG.indices, blocks.L_BG.values, blocks.L_BG.shape
            ))
            variant_ids_by_tag = {tag: [] for tag in configs}
            for index, kind in enumerate(factor_kinds):
                def matrix_for(tag):
                    root = roots_by_tag[tag][index][1]
                    return (_real_shift_matrix(blocks, float(root), scale)
                            if kind == "real" else _pair_shift_matrix(blocks, complex(root), scale))

                first_system = matrix_for(first_tag)
                pattern = CudssSparsePattern(first_system)
                patterns.append(pattern)
                for tag in configs:
                    variant_id = pattern.register_variant(tag, matrix_for(tag))
                    variant_ids_by_tag[tag].append(variant_id)
            resources = _SharedFactorizedVariantResources(
                blocks=blocks, ghost_cache=ghost_cache, ghost_factor=ghost_factor,
                patterns=tuple(patterns), variant_ids_by_tag={
                    tag: tuple(ids) for tag, ids in variant_ids_by_tag.items()
                },
            )
        except BaseException:
            for pattern in patterns:
                pattern.close()
            ghost_cache.close()
            raise

        views = {}
        for tag, (ell, eta) in configs.items():
            root_items = roots_by_tag[tag]
            view_factors = tuple(
                (kind, root, patterns[index], resources.variant_ids_by_tag[tag][index])
                for index, (kind, root) in enumerate(root_items)
            )
            views[tag] = cls(
                blocks=blocks, power=power, ell=float(ell), eta=float(eta),
                scale=scale, factors=view_factors, _cache=resources,
                _ghost_factor=ghost_factor, _variant_resources=resources,
                _variant_ids=resources.variant_ids_by_tag[tag],
            )
        return views

    def _apply_A(self, interior):
        ghost = self._ghost_factor.solve(-_block_matvec(self.blocks.L_BI, interior))
        return (
            _block_matvec(self.blocks.L_II, interior)
            + _block_matvec(self.blocks.L_IG, ghost)
        )

    def _apply_A_power(self, interior, power):
        value = interior
        for _ in range(power):
            value = self._apply_A(value)
        return value

    def _effective_rhs(self, rhs_physical, dirichlet, traces):
        ni, nb = self.blocks.n_interior, self.blocks.n_boundary
        rhs = jnp.asarray(rhs_physical)
        g = jnp.asarray(dirichlet).reshape(-1)
        c = jnp.asarray(traces)
        if rhs.ndim != 1 or int(rhs.shape[0]) != ni + nb:
            raise ValueError("rhs_physical must be [interior; boundary] vector")
        if g.ndim != 1 or int(g.shape[0]) != nb:
            raise ValueError("dirichlet must have length n_boundary")
        if c.ndim != 2 or tuple(c.shape) != (self.power, nb):
            raise ValueError("traces must have shape (power, n_boundary) for c1..ck")
        rhs_i, rhs_b = rhs[:ni], rhs[ni:]
        r0 = (rhs_b - g - self.eta * c[-1]) / self.ell
        b0 = (
            _block_matvec(self.blocks.L_IB, g)
            + _block_matvec(
                self.blocks.L_IG,
                self._ghost_factor.solve(r0 - _block_matvec(self.blocks.L_BB, g)),
            )
        )
        affine = b0
        for stage in range(1, self.power):
            cj, cnext = c[stage - 1], c[stage]
            # The ghost closure for the full stage also has -L_BI q_j. That
            # contribution is precisely the homogeneous A action below, so
            # b_j contains only prescribed traces.
            bj = (
                _block_matvec(self.blocks.L_IB, cj)
                + _block_matvec(
                    self.blocks.L_IG,
                    self._ghost_factor.solve(
                        cnext - _block_matvec(self.blocks.L_BB, cj)
                    ),
                )
            )
            affine = self._apply_A(affine) + bj
        return rhs_i - self.ell * b0 - self.eta * affine, r0, g

    def solve(self, rhs_physical, dirichlet, traces):
        """Solve and return full ``[interior; boundary; ghost]`` state.

        `rhs_physical` is ordered `[rhs_I; rhs_B]`; `dirichlet` supplies the
        current boundary value ``g=c_0``; `traces[j-1]` supplies
        ``c_j=(Delta**j u)|_B`` for ``j=1,...,k``. The original boundary-row
        Laplacian ``r0`` is separately inferred from the boundary equation,
        and is not assumed equal to ``c_1``.
        """
        reduced_rhs, r0, g = self._effective_rhs(rhs_physical, dirichlet, traces)
        value = reduced_rhs
        for item in self.factors:
            kind, root, factor = item[:3]
            variant_id = item[3] if len(item) > 3 else None
            if kind == "real":
                rhs_aug = jnp.concatenate(
                    [value, jnp.zeros((self.blocks.n_ghost,), dtype=value.dtype)]
                )
                solved = (factor.solve_variant(variant_id, rhs_aug) if variant_id is not None
                          else factor.solve(rhs_aug))
                value = solved[: self.blocks.n_interior]
            else:
                ni, ng = self.blocks.n_interior, self.blocks.n_ghost
                rhs_aug = jnp.concatenate([
                    value,
                    jnp.zeros((ni + 2 * ng,), dtype=value.dtype),
                ])
                solved = (factor.solve_variant(variant_id, rhs_aug) if variant_id is not None
                          else factor.solve(rhs_aug))
                _, b = float(root.real), float(root.imag)
                value = (abs(root) / b) * solved[ni : 2 * ni]
        ghost_rhs = (
            r0
            - _block_matvec(self.blocks.L_BI, value)
            - _block_matvec(self.blocks.L_BB, g)
        )
        ghost = self._ghost_factor.solve(ghost_rhs)
        return jnp.concatenate([value, g, ghost])

    @property
    def phase_counts(self):
        """Per-factor cuDSS setup/refactor counts for reuse verification."""
        return {
            "ghost": self._ghost_factor.phase_counts,
            **{f"factor_{index}_{item[0]}": item[2].phase_counts
               for index, item in enumerate(self.factors)},
        }

    def close(self):
        """Release all fixed cuDSS factors."""
        self._cache.close()


class _SharedFactorizedVariantResources:
    """Idempotent owner shared by all coefficient-variant solver views."""

    def __init__(self, *, blocks, ghost_cache, ghost_factor, patterns, variant_ids_by_tag):
        self.blocks = blocks
        self.ghost_cache = ghost_cache
        self.ghost_factor = ghost_factor
        self.patterns = patterns
        self.variant_ids_by_tag = variant_ids_by_tag
        self._closed = False

    def close(self):
        if self._closed:
            return
        self._closed = True
        for pattern in self.patterns:
            pattern.close()
        self.ghost_cache.close()
