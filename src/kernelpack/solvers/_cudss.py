"""Optional CUDA direct solves for KernelPack COO systems.

Changing matrix values reuse prepared symbolic contexts and numerical storage,
with implicit matrix/RHS derivatives. Fixed transient systems reuse their
numerical factors too. Unprepared generic COO calls remain a fallback.

All setup work that scales with N or nnz (COO -> CSR, duplicate summation,
transpose maps, cache keys) runs on the JAX device. Only scalars (entry
counts, validity flags, fingerprint words) are read back on the host.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_batching import custom_vmap


# ----------------------------------------------------------------------------- device COO -> CSR
@jax.jit
def _sort_coo(indices):
    """Lexicographic (row, col) order of COO entries and the first-of-run flags."""
    rows = indices[:, 0].astype(jnp.int32)
    cols = indices[:, 1].astype(jnp.int32)
    order = jnp.lexsort((cols, rows)).astype(jnp.int32)   # last key is primary: row, then column
    r, c = rows[order], cols[order]
    new = jnp.concatenate((jnp.ones((1,), bool), (r[1:] != r[:-1]) | (c[1:] != c[:-1])))
    return order, r, c, new


@partial(jax.jit, static_argnums=(4, 5))
def _coalesce_sorted(order, r, c, new, size, n_rows):
    """Unique sorted entries, the entry -> unique-entry map, CSR row pointers and the transpose maps."""
    run = (jnp.cumsum(new.astype(jnp.int32)) - 1).astype(jnp.int32)       # unique index of each sorted entry
    groups = jnp.zeros_like(run).at[order].set(run)                       # ... of each original entry
    first = jnp.nonzero(new, size=size)[0]
    urows, ucols = r[first], c[first]
    rows_ptr = jnp.concatenate((jnp.zeros((1,), jnp.int32),
                                jnp.cumsum(jnp.bincount(urows, length=n_rows)).astype(jnp.int32)))
    t_order = jnp.lexsort((urows, ucols)).astype(jnp.int32)               # column, then row
    t_ptr = jnp.concatenate((jnp.zeros((1,), jnp.int32),
                             jnp.cumsum(jnp.bincount(ucols, length=n_rows)).astype(jnp.int32)))
    return groups, urows, ucols, rows_ptr, t_order, t_ptr, urows[t_order]


@partial(jax.jit, static_argnums=(2,))
def _coalesce_values(groups, values, size):
    return jax.ops.segment_sum(values, groups, num_segments=size)


def _device_coo_structure(indices, n_rows):
    """Sorted unique COO structure of ``indices`` (k x 2), computed on the device.

    Returns the unique entry count, the map from input entries to unique
    entries, CSR row pointers and columns, the unique rows, and the CSR
    transpose maps; the same arrays as np.unique(axis=0) + bincount + lexsort.
    """
    indices = jnp.asarray(indices)
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError("COO indices must have shape (nnz, 2)")
    if indices.shape[0] == 0:
        raise ValueError("COO matrix has no entries")
    order, r, c, new = _sort_coo(indices)
    size = int(jnp.sum(new))                                               # one scalar to the host
    groups, urows, ucols, rows_ptr, t_order, t_ptr, t_cols = _coalesce_sorted(order, r, c, new, size, int(n_rows))
    return dict(size=size, groups=groups, rows=urows, cols=ucols, indptr=rows_ptr,
                transpose_order=t_order, transpose_indptr=t_ptr, transpose_cols=t_cols)


# ----------------------------------------------------------------------------- device fingerprints
def _fmix32(h):
    h = h ^ (h >> 16)
    h = h * jnp.uint32(0x85EBCA6B)
    h = h ^ (h >> 13)
    h = h * jnp.uint32(0xC2B2AE35)
    return h ^ (h >> 16)


def _as_words(a):
    if a.dtype == jnp.bool_:
        a = a.astype(jnp.uint8)
    if a.dtype.itemsize in (4, 8):
        return jax.lax.bitcast_convert_type(a, jnp.uint32).reshape(-1)
    return a.astype(jnp.uint32).reshape(-1)


_FINGERPRINT_SEEDS = (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344)


@jax.jit
def _fingerprint_words(*arrays):
    """Four 32-bit order-sensitive hashes of the exact bit patterns of the arrays.

    Each word is mixed with its position and array number, then summed modulo
    2^32; integer sums are exact, so the result does not depend on the
    reduction order on the device.
    """
    out = []
    for seed in _FINGERPRINT_SEEDS:
        total = jnp.uint32(0)
        for k, a in enumerate(arrays):
            w = _as_words(a)
            pos = jnp.arange(w.shape[0], dtype=jnp.uint32)
            key = _fmix32(pos * jnp.uint32(0x9E3779B1) + jnp.uint32((seed + 0x632BE5AB * (k + 1)) & 0xFFFFFFFF))
            total = total + jnp.sum(_fmix32(w ^ key) * jnp.uint32(2 * k + 1), dtype=jnp.uint32)
        out.append(total)
    return jnp.stack(out)


def device_fingerprint(*arrays):
    """128-bit hex fingerprint of the exact contents, shapes and dtypes of device arrays.

    Replaces host ``tobytes()``/SHA-256 keys: the arrays never leave the
    device; only four 32-bit words are read back.
    """
    arrays = tuple(jnp.asarray(a) for a in arrays)
    words = np.asarray(_fingerprint_words(*arrays)) if arrays else np.zeros(4, np.uint32)
    layout = "|".join(f"{a.dtype.str}{a.shape}" for a in arrays)
    return "".join(f"{int(v):08x}" for v in words) + ":" + layout


class CudssSparsePattern:
    """Prepare CSR maps and A/A.T symbolic contexts once, eagerly (all on the device)."""
    def __init__(self, system):
        from kernelpack.sparse import CudssFactor
        n = system.shape[0]
        ij = jnp.asarray(system.indices).astype(jnp.int32)
        structure = _device_coo_structure(ij, n)
        self.shape = system.shape
        self.input_size = int(ij.shape[0])
        self.original_indices = ij
        self.size = structure["size"]
        self.groups = structure["groups"]
        self.rows = structure["indptr"]
        self.cols = structure["cols"]
        self.coo_rows = structure["rows"]
        self.transpose_order = structure["transpose_order"]
        self.transpose_rows = structure["transpose_indptr"]
        self.transpose_cols = structure["transpose_cols"]
        self._variant_ids = {}
        self._variant_values = {}
        self._variant_keys = {}
        reference=jnp.zeros(self.size,dtype=system.values.dtype).at[self.groups].add(system.values)
        reference=jax.device_put(reference,jax.devices("gpu")[0])
        self._factor=CudssFactor(self.rows,self.cols,reference,self.shape,
                                 matrix_type="general",mutable_values=True)

    def register_variant(self, key, system):
        """Bind an immutable matrix-value vector to a unique native variant tag.

        The COO structure must exactly match this pattern. Values are
        coalesced once on the device and stored there; solve calls never
        accept caller-provided values, preventing a tag/value mismatch.
        """
        if system.shape != self.shape:
            raise ValueError("cuDSS variant shape does not match its sparse pattern")
        indices = jnp.asarray(system.indices)
        if indices.shape != self.original_indices.shape or not bool(
            jnp.all(indices.astype(jnp.int32) == self.original_indices)
        ):
            raise ValueError("cuDSS variant COO indices do not match its sparse pattern")
        try:
            existing = self._variant_ids.get(key)
        except TypeError as exc:
            raise TypeError("cuDSS variant keys must be hashable") from exc
        values = jnp.asarray(system.values)
        if values.shape != (self.input_size,) or values.dtype not in (jnp.float32, jnp.float64):
            raise ValueError("cuDSS variant values must match the pattern and use float32/float64")
        if values.dtype != self._factor.dtype:
            raise ValueError("cuDSS variant dtype must match the prepared factor")
        if not bool(jnp.all(jnp.isfinite(values))):
            raise ValueError("cuDSS variant values must be finite")
        coalesced = jax.device_put(_coalesce_values(self.groups, values, self.size), self._factor.device)
        if existing is not None:
            if not bool(jnp.array_equal(self._variant_values[existing], coalesced)):
                raise ValueError("a registered cuDSS variant key cannot be rebound to different values")
            return existing
        variant_id = len(self._variant_ids)
        self._variant_ids[key] = variant_id
        self._variant_keys[variant_id] = key
        self._variant_values[variant_id] = jax.lax.stop_gradient(coalesced)
        return variant_id

    def solve_variant(self, variant_id, rhs):
        """Solve using a registered, frozen matrix variant and its exact transpose."""
        from kernelpack.sparse.cudss import _check_solution

        if variant_id not in self._variant_values:
            raise KeyError(f"unknown cuDSS matrix variant id: {variant_id}")
        b = jnp.asarray(rhs)
        if b.ndim not in (1, 2) or b.shape[0] != self.shape[0]:
            raise ValueError("cuDSS variant RHS must be a matching vector or matrix")
        if b.dtype != self._factor.dtype:
            raise ValueError("cuDSS variant RHS dtype must match the prepared factor")
        values = self._variant_values[variant_id]
        rows, cols = self.coo_rows, self.cols

        def matvec(x, transpose=False):
            rr, cc = (cols, rows) if transpose else (rows, cols)
            weights = values if x.ndim == 1 else values[:, None]
            return jnp.zeros_like(x).at[rr].add(weights * x[cc])

        def direct(target, transpose):
            vector = target.ndim == 1
            columns = target[:, None] if vector else target
            data = values[self.transpose_order] if transpose else values

            @custom_vmap
            def native(matrix_values, rhs_columns):
                return self._factor._variant_solve(
                    matrix_values, rhs_columns, variant_id, transpose=transpose
                )

            @native.def_vmap
            def rule(axis_size, in_batched, matrix_values, rhs_columns):
                if in_batched[0]:
                    raise ValueError(
                        "registered cuDSS variants are frozen; matrix values cannot be batched"
                    )
                if in_batched[1]:
                    packed = jnp.moveaxis(rhs_columns, 1, 0)
                    solved = native(
                        matrix_values, packed.reshape(self.shape[0], -1)
                    ).reshape(packed.shape)
                    return jnp.moveaxis(solved, 0, 1), True
                return native(matrix_values, rhs_columns), False

            result = native(data, columns)
            result = result[:, 0] if vector else result
            return _check_solution(
                result, target, lambda x: matvec(x, transpose),
                anorm=jnp.linalg.norm(values),
            )

        return jax.lax.custom_linear_solve(
            matvec, b,
            solve=lambda _operator, target: direct(target, False),
            transpose_solve=lambda _operator, target: direct(target, True),
            symmetric=False,
        )

    @property
    def phase_counts(self):
        return self._factor.phase_counts

    def close(self):
        self._factor.close()


def solve_sparse_system_cudss(system, rhs, pattern=None):
    from kernelpack.sparse.cudss import _load, _check_solution

    lib = _load()
    b = jnp.asarray(rhs)
    if system.shape[0] != system.shape[1] or b.shape[0] != system.shape[0]:
        raise ValueError("cuDSS requires a square system and matching RHS")
    if b.ndim not in (1, 2) or b.dtype not in (jnp.float32, jnp.float64):
        raise ValueError("cuDSS requires a float32/float64 vector or matrix RHS")
    name = ("kp_cudss_refactor_" if pattern is not None else "kp_cudss_coo_") + ("f64" if b.dtype == jnp.float64 else "f32")
    if not hasattr(lib, name):
        raise RuntimeError("Rebuild cuDSS FFI for changing-matrix solves: python -m kernelpack.sparse.build_cudss")
    indices = jnp.asarray(system.indices, dtype=jnp.int32)
    values = jnp.asarray(system.values, dtype=b.dtype)
    if pattern is not None:
        if system.shape != pattern.shape or values.size != pattern.input_size:
            raise ValueError("cuDSS prepared sparse pattern does not match this system")
        pattern_matches = jnp.all(indices == pattern.original_indices)
        csr_values = jnp.zeros(pattern.size, dtype=values.dtype).at[pattern.groups].add(values)
        # Use the reduced operator for residuals and implicit derivatives too,
        # rather than repeating gather/scatter work for duplicate COO entries.
        indices = jnp.column_stack((pattern.coo_rows, pattern.cols))
        values = csr_values

    def matvec(x, transpose=False):
        rows, cols = (indices[:, 1], indices[:, 0]) if transpose else (indices[:, 0], indices[:, 1])
        weights = values if x.ndim == 1 else values[:, None]
        return jnp.zeros_like(x).at[rows].add(weights * x[cols])

    def direct(target, transpose):
        vector = target.ndim == 1
        columns = target[:, None] if vector else target
        if pattern is None:
            call = jax.ffi.ffi_call(
                name, jax.ShapeDtypeStruct(columns.shape, columns.dtype),
                has_side_effect=True, vmap_method="sequential",
                input_layouts=[(0, 1), (0,), (1, 0)], output_layouts=(1, 0),
            )
            result = call(indices, values, columns, transpose=transpose)
        else:
            data = csr_values[pattern.transpose_order] if transpose else csr_values
            @custom_vmap
            def native(matrix_values, rhs_columns):
                return pattern._factor._refactor_solve(matrix_values,rhs_columns,transpose=transpose)

            @native.def_vmap
            def rule(axis_size, in_batched, matrix_values, rhs_columns):
                if in_batched[0]:
                    if not in_batched[1]:
                        rhs_columns = jnp.broadcast_to(rhs_columns, (axis_size,) + rhs_columns.shape)
                    return jax.lax.map(lambda pair: native(*pair), (matrix_values, rhs_columns)), True
                if in_batched[1]:
                    packed = jnp.moveaxis(rhs_columns, 1, 0)
                    solved = native(matrix_values, packed.reshape(system.shape[0], -1)).reshape(packed.shape)
                    return jnp.moveaxis(solved, 0, 1), True
                return native(matrix_values, rhs_columns), False
            result = native(data, columns)
            result = jnp.where(pattern_matches, result, jnp.full_like(result, jnp.nan))
        result = result[:, 0] if vector else result
        return _check_solution(result, target, lambda x: matvec(x, transpose),
                               anorm=jnp.linalg.norm(values))

    return jax.lax.custom_linear_solve(
        matvec, b, solve=lambda _op, target: direct(target, False),
        transpose_solve=lambda _op, target: direct(target, True), symmetric=False,
    )


class CudssSystemCache:
    """Reusable fixed matrices, keyed by their exact (coalesced CSR) contents.

    Preparation is eager. A traced matrix uses the changing-matrix backend;
    never reuse a numerical factor for new or differentiated matrix values.
    The COO -> CSR conversion (sort by (row, col), duplicate sums, row
    pointers) runs on the device. The cache key is a device fingerprint of
    the CSR arrays; a key hit is confirmed by an exact device comparison
    with the stored factor's CSR arrays, so a fingerprint collision can never
    return a factor of a different matrix.
    """
    def __init__(self):
        self.factors = {}

    @staticmethod
    def _csr(system):
        n = int(system.shape[0])
        structure = _device_coo_structure(system.indices, n)
        data = _coalesce_values(structure["groups"], jnp.asarray(system.values), structure["size"])
        return structure["indptr"], structure["cols"], data

    def prepare(self, system):
        from kernelpack.sparse import CudssFactor
        indptr, cols, data = self._csr(system)
        key = (tuple(system.shape), device_fingerprint(indptr, cols, data))
        for factor in self.factors.get(key, ()):
            if (factor._cols.shape == cols.shape and bool(
                    jnp.array_equal(factor._indptr, indptr) & jnp.array_equal(factor._cols, cols)
                    & jnp.array_equal(factor._values, data))):
                return factor
        factor = CudssFactor(indptr, cols, data, tuple(system.shape), matrix_type="general", rhs_block_size=8)
        self.factors.setdefault(key, []).append(factor)
        return factor

    def close(self):
        for factors in self.factors.values():
            for factor in factors:
                factor.close()
        self.factors.clear()


class _Ref:
    """Identity key that keeps its object alive.

    A bare ``id(obj)`` can be reused by a new object once ``obj`` is garbage
    collected, which would let a stale factor match a different matrix or
    gamma. Holding the reference inside the stored signature prevents reuse,
    and equality is object identity.
    """

    __slots__ = ("obj",)

    def __init__(self, obj):
        self.obj = obj

    def __eq__(self, other):
        return isinstance(other, _Ref) and other.obj is self.obj

    def __hash__(self):
        return id(self.obj)

    def __repr__(self):
        return f"_Ref({type(self.obj).__name__}@{id(self.obj):#x})"


def _transient_signature(base):
    scalars = (base.dt, base.nu)
    if any(isinstance(v, jax.core.Tracer) for v in scalars):
        return None
    gamma = getattr(base, "gamma_components", None)
    explicit_hyper = getattr(base, "hyperviscosity_treatment", "implicit") == "explicit"
    if explicit_hyper:
        gamma = None
    if isinstance(gamma, jax.core.Tracer):
        return None
    return (float(base.dt), float(base.nu), _Ref(base.lap_sparse),
            _Ref(base.bc_sparse), None if explicit_hyper else _Ref(getattr(base, "hyper_sparse", None)),
            None if gamma is None else _Ref(gamma),
            getattr(base, "hyperviscosity_method", None),
            None if explicit_hyper else getattr(base, "hyperviscosity_power", None),
            getattr(base, "hyperviscosity_treatment", "implicit"),
            getattr(base, "laplacian_power_solver", "augmented"),
            (_Ref(getattr(base, "power_layout", None))
             if getattr(base, "hyperviscosity_method", None) == "prescribed_trace_power" and not explicit_hyper else None))


def _same_values(a, b):
    # gamma has one entry per dimension; compare on the host, which also works while tracing
    # (jnp ops on constants are staged into the trace and could not be converted to bool).
    return np.array_equal(np.asarray(a), np.asarray(b))


def prepare_transient_cudss(base, *, orders=(1, 2, 3)):
    """Prepare BDF1/2/3 matrices after spatial and boundary assembly."""
    if (getattr(base, "hyperviscosity_method", None) == "prescribed_trace_power"
            and getattr(base, "hyperviscosity_treatment", "implicit") == "implicit"):
        from ._composed_adr import prepare_power_cudss
        return prepare_power_cudss(base, orders=orders)
    from ._common import build_implicit_system_sparse
    if base.bc_sparse is None or base.lap_sparse is None:
        raise ValueError("Prepare sparse boundary operators before cuDSS factors")
    if (_transient_signature(base) == getattr(base, "cudss_signature", None)
            and getattr(base, "cudss_cache", None) is not None
            and all(order in getattr(base, "cudss_factors", {}) for order in orders)):
        return
    if getattr(base, "cudss_cache", None) is not None:
        base.cudss_cache.close()
    cache = CudssSystemCache()
    factors = {}
    try:
        operator = (base._implicit_operator_sparse(base.nu, base.gamma_components)
                    if hasattr(base, "hyper_sparse") else base.lap_sparse)
        for order, alpha in ((1, 1.), (2, 2./3.), (3, 6./11.)):
            if order not in orders:
                continue
            scale = -alpha * base.dt * (1. if hasattr(base, "hyper_sparse") else base.nu)
            system = build_implicit_system_sparse(operator, base.bc_sparse, base.n, scale)
            factors[order] = cache.prepare(system)
    except BaseException:
        cache.close()
        raise
    base.cudss_cache = cache
    base.cudss_factors = factors
    base.cudss_signature = _transient_signature(base)


def close_transient_cudss(base):
    if getattr(base, "cudss_cache", None) is not None:
        base.cudss_cache.close()
    base.cudss_cache = None
    base.cudss_factors = {}
    base.cudss_signature = None


def get_transient_cudss_factor(base, order, gamma=None):
    signature = _transient_signature(base)
    if (getattr(base, "hyperviscosity_treatment", "implicit") != "explicit"
            and gamma is not None and gamma is not base.gamma_components):
        if isinstance(gamma, jax.core.Tracer) or not _same_values(gamma, base.gamma_components):
            return None
    prepared = getattr(base, "cudss_signature", None)
    if signature is not None and signature == prepared:
        return getattr(base, "cudss_factors", {}).get(order)
    return None


def solve_transient_cudss(base, system, rhs, order, gamma=None):
    factor = get_transient_cudss_factor(base, order, gamma)
    if factor is not None:
        return factor.solve(rhs)
    signature = _transient_signature(base)
    if (getattr(base, "hyperviscosity_treatment", "implicit") != "explicit"
            and gamma is not None and gamma is not base.gamma_components):
        if isinstance(gamma, jax.core.Tracer) or not _same_values(gamma, base.gamma_components):
            return solve_sparse_system_cudss(system, rhs)
    # Eager stateful use can prepare lazily. Traced/changing matrices use
    # the fully differentiable changing-matrix path rather than stale factors.
    if signature is not None and not isinstance(system.values, jax.core.Tracer):
        prepare_transient_cudss(base)
        return base.cudss_factors[order].solve(rhs)
    return solve_sparse_system_cudss(system, rhs)
