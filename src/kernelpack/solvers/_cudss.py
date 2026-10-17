"""Optional CUDA direct solves for KernelPack COO systems.

Changing matrix values reuse prepared symbolic contexts and numerical storage,
with implicit matrix/RHS derivatives. Fixed transient systems reuse their
numerical factors too. Unprepared generic COO calls remain a fallback.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_batching import custom_vmap


class CudssSparsePattern:
    """Prepare CSR maps and A/A.T symbolic contexts once, eagerly."""
    def __init__(self, system):
        from kernelpack.sparse import CudssFactor
        ij = np.asarray(system.indices, dtype=np.int32)
        unique, groups = np.unique(ij, axis=0, return_inverse=True)
        n = system.shape[0]
        self.shape = system.shape
        self.input_size = len(ij)
        self.original_indices = jnp.asarray(ij)
        self.size = len(unique)
        self.groups = jnp.asarray(groups, dtype=jnp.int32)
        self.rows = jnp.asarray(np.r_[0, np.cumsum(np.bincount(unique[:, 0], minlength=n))], dtype=jnp.int32)
        self.cols = jnp.asarray(unique[:, 1], dtype=jnp.int32)
        self.coo_rows = jnp.asarray(unique[:, 0], dtype=jnp.int32)
        order = np.lexsort((unique[:, 0], unique[:, 1]))
        self.transpose_order = jnp.asarray(order, dtype=jnp.int32)
        self.transpose_rows = jnp.asarray(np.r_[0, np.cumsum(np.bincount(unique[:, 1], minlength=n))], dtype=jnp.int32)
        self.transpose_cols = jnp.asarray(unique[order, 0], dtype=jnp.int32)
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
        coalesced once on the host and stored on device; solve calls never
        accept caller-provided values, preventing a tag/value mismatch.
        """
        if system.shape != self.shape:
            raise ValueError("cuDSS variant shape does not match its sparse pattern")
        indices = np.asarray(system.indices, dtype=np.int32)
        if indices.shape != np.asarray(self.original_indices).shape or not np.array_equal(
            indices, np.asarray(self.original_indices)
        ):
            raise ValueError("cuDSS variant COO indices do not match its sparse pattern")
        try:
            existing = self._variant_ids.get(key)
        except TypeError as exc:
            raise TypeError("cuDSS variant keys must be hashable") from exc
        values = np.asarray(system.values)
        if values.shape != (self.input_size,) or values.dtype not in (np.float32, np.float64):
            raise ValueError("cuDSS variant values must match the pattern and use float32/float64")
        if values.dtype != np.dtype(self._factor.dtype):
            raise ValueError("cuDSS variant dtype must match the prepared factor")
        if not np.all(np.isfinite(values)):
            raise ValueError("cuDSS variant values must be finite")
        coalesced = np.zeros((self.size,), dtype=values.dtype)
        np.add.at(coalesced, np.asarray(self.groups), values)
        if existing is not None:
            if not np.array_equal(np.asarray(self._variant_values[existing]), coalesced):
                raise ValueError("a registered cuDSS variant key cannot be rebound to different values")
            return existing
        variant_id = len(self._variant_ids)
        self._variant_ids[key] = variant_id
        self._variant_keys[variant_id] = key
        self._variant_values[variant_id] = jax.device_put(
            jax.lax.stop_gradient(jnp.asarray(coalesced)), self._factor.device
        )
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
    """Reusable fixed matrices, keyed by their exact COO arrays.

    Preparation is eager. A traced matrix uses the changing-matrix backend;
    never reuse a numerical factor for new or differentiated matrix values.
    """
    def __init__(self):
        self.factors = {}

    def prepare(self, system):
        from scipy.sparse import coo_matrix
        from kernelpack.sparse import CudssFactor
        ij = np.asarray(system.indices)
        values = np.asarray(system.values)
        key = (system.shape, ij.tobytes(), values.dtype.str, values.tobytes())
        if key not in self.factors:
            matrix = coo_matrix((values, (ij[:, 0], ij[:, 1])), shape=system.shape).tocsr()
            matrix.sum_duplicates()
            matrix.sort_indices()
            self.factors[key] = CudssFactor(
                matrix.indptr, matrix.indices, jnp.asarray(matrix.data),
                matrix.shape, matrix_type="general", rhs_block_size=8,
            )
        return self.factors[key]

    def close(self):
        for factor in self.factors.values():
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
        if isinstance(gamma, jax.core.Tracer) or not np.array_equal(np.asarray(gamma), np.asarray(base.gamma_components)):
            return None
    prepared = getattr(base, "cudss_signature", None)
    if signature is not None and signature == prepared:
        return base.cudss_factors.get(order)
    return None


def solve_transient_cudss(base, system, rhs, order, gamma=None):
    factor = get_transient_cudss_factor(base, order, gamma)
    if factor is not None:
        return factor.solve(rhs)
    signature = _transient_signature(base)
    if (getattr(base, "hyperviscosity_treatment", "implicit") != "explicit"
            and gamma is not None and gamma is not base.gamma_components):
        if isinstance(gamma, jax.core.Tracer) or not np.array_equal(np.asarray(gamma), np.asarray(base.gamma_components)):
            return solve_sparse_system_cudss(system, rhs)
    # Eager stateful use can prepare lazily. Traced/changing matrices use
    # the fully differentiable changing-matrix path rather than stale factors.
    if signature is not None and not isinstance(system.values, jax.core.Tracer):
        prepare_transient_cudss(base)
        return base.cudss_factors[order].solve(rhs)
    return solve_sparse_system_cudss(system, rhs)
