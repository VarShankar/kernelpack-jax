"""Reusable cuDSS factors with typed JAX FFI solves.

Factor construction is eager. The bound ``solve`` method supports ``jit`` and
batched ``vmap``, RHS forward/reverse-mode autodifferentiation, and transpose solves. Matrix
gradients, serialization, and distributed/sharded solves are not supported.
"""

from __future__ import annotations

import ctypes
from numbers import Integral
import os
from pathlib import Path
import threading
import weakref
import time
import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_batching import custom_vmap
from .products import PaddedCSRAction, _padded_csr_action_from_validated


class CudssUnavailableError(RuntimeError):
    pass


def _check_solution(solution, rhs, matvec, anorm=None):
    """Reject inaccurate primal/transpose solves, including compiled calls.

    Uses a normwise backward-error test per column,
        ||A x - b|| <= tol * (||A|| ||x|| + ||b||),
    which a backward-stable direct solve satisfies regardless of cond(A).
    The older test ||A x - b|| <= tol ||b|| + abs_floor could reject valid
    solves of ill-conditioned systems (e.g. with strong hyperviscosity) and
    accept poor solves whose RHS norm sat near the absolute floor. ``anorm``
    should be an upper bound on ||A||_2 (the Frobenius norm of the values is
    used by all callers); without it the test falls back to ||b|| only.
    """
    residual = matvec(solution) - rhs
    tol = 5e-5 if rhs.dtype == jnp.float32 else 1e-8
    norms = jnp.linalg.norm(residual, axis=0)
    scale = jnp.linalg.norm(rhs, axis=0)
    if anorm is not None:
        scale = scale + anorm * jnp.linalg.norm(solution, axis=0)
    limits = tol * scale + jnp.finfo(rhs.dtype).tiny
    valid = jnp.all(jnp.isfinite(solution)) & jnp.all(jnp.isfinite(norms)) & jnp.all(norms <= limits)

    # Stay entirely on the device. Failed solves cannot masquerade as finite
    # states/gradients; guarded inverse residuals can reject the trial.
    return jnp.where(valid, solution, jnp.full_like(solution, jnp.nan))


_library = None
_lock = threading.Lock()


def _load():
    global _library
    with _lock:
        if _library is not None:
            return _library
        path = Path(
            os.environ.get(
                "KERNELPACK_CUDSS_LIBRARY", Path(__file__).with_name("_cudss_ffi.so")
            )
        )
        try:
            lib = ctypes.CDLL(str(path))
        except OSError as exc:
            raise CudssUnavailableError(
                "cuDSS FFI is optional and has not been built or its CUDA libraries are unavailable. "
                "Run python -m kernelpack.sparse.build_cudss; see the README section 'GPU sparse direct solves'."
            ) from exc
        lib.kp_cudss_create.restype = ctypes.c_int64
        lib.kp_cudss_create.argtypes = [
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_char_p,
        ]
        if hasattr(lib,"kp_cudss_create_with_reordering"):
            lib.kp_cudss_create_with_reordering.argtypes=[*lib.kp_cudss_create.argtypes,ctypes.c_int]
            lib.kp_cudss_create_with_reordering.restype=ctypes.c_int64
        lib.kp_cudss_destroy.argtypes = [ctypes.c_int64]
        lib.kp_cudss_destroy.restype = None
        lib.kp_cudss_error.restype = ctypes.c_char_p
        lib.kp_cudss_solve_count.argtypes = [ctypes.c_int64]
        lib.kp_cudss_solve_count.restype = ctypes.c_int64
        if hasattr(lib, "kp_cudss_check_status"):
            lib.kp_cudss_check_status.argtypes = [ctypes.c_int64]
            lib.kp_cudss_check_status.restype = ctypes.c_int
        if hasattr(lib, "kp_cudss_phase_counts"):
            lib.kp_cudss_phase_counts.argtypes = [ctypes.c_int64, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64)]
            lib.kp_cudss_phase_counts.restype = ctypes.c_int
        if hasattr(lib,"kp_cudss_resource_counts"):
            lib.kp_cudss_resource_counts.argtypes=[ctypes.c_int64,ctypes.POINTER(ctypes.c_int64),ctypes.POINTER(ctypes.c_int64)]
            lib.kp_cudss_resource_counts.restype=ctypes.c_int
        if hasattr(lib,"kp_cudss_factor_stats"):
            lib.kp_cudss_factor_stats.argtypes=[ctypes.c_int64,
                ctypes.POINTER(ctypes.c_int64),ctypes.POINTER(ctypes.c_int64)]
            lib.kp_cudss_factor_stats.restype=ctypes.c_int
        if hasattr(lib,"kp_cudss_refactor_values"):
            lib.kp_cudss_refactor_values.argtypes=[ctypes.c_int64,ctypes.c_void_p]
            lib.kp_cudss_refactor_values.restype=ctypes.c_int
        if hasattr(lib,"kp_cudss_replace_context"):
            lib.kp_cudss_replace_context.argtypes=[ctypes.c_int64,ctypes.c_int64]
            lib.kp_cudss_replace_context.restype=ctypes.c_int
        if hasattr(lib,"kp_cudss_rebuild_context"):
            lib.kp_cudss_rebuild_context.argtypes=[ctypes.c_int64,*lib.kp_cudss_create.argtypes]
            lib.kp_cudss_rebuild_context.restype=ctypes.c_int
        if hasattr(lib,"kp_cudss_rebuild_context_with_reordering"):
            lib.kp_cudss_rebuild_context_with_reordering.argtypes=[ctypes.c_int64,*lib.kp_cudss_create.argtypes,ctypes.c_int]
            lib.kp_cudss_rebuild_context_with_reordering.restype=ctypes.c_int
        if hasattr(lib,"kp_cudss_graph_mode"):
            lib.kp_cudss_graph_mode.argtypes=[ctypes.c_int64,ctypes.c_int]
            lib.kp_cudss_graph_mode.restype=ctypes.c_int64
        for dtype in ("f32", "f64"):
            triangular_name = "kp_cudss_triangular_" + dtype
            if hasattr(lib, triangular_name):
                jax.ffi.register_ffi_target(triangular_name, jax.ffi.pycapsule(getattr(lib, triangular_name)), platform="CUDA")
            name = "kp_cudss_solve_" + dtype
            jax.ffi.register_ffi_target(
                name, jax.ffi.pycapsule(getattr(lib, name)), platform="CUDA"
            )
            coo_name = "kp_cudss_coo_" + dtype
            if hasattr(lib, coo_name):
                jax.ffi.register_ffi_target(
                    coo_name, jax.ffi.pycapsule(getattr(lib, coo_name)), platform="CUDA"
                )
            csr_name = "kp_cudss_csr_" + dtype
            if hasattr(lib, csr_name):
                jax.ffi.register_ffi_target(csr_name, jax.ffi.pycapsule(getattr(lib, csr_name)), platform="CUDA")
            refactor_name = "kp_cudss_refactor_" + dtype
            if hasattr(lib, refactor_name):
                jax.ffi.register_ffi_target(refactor_name, jax.ffi.pycapsule(getattr(lib, refactor_name)), platform="CUDA")
            variant_name = "kp_cudss_variant_" + dtype
            if hasattr(lib, variant_name):
                jax.ffi.register_ffi_target(variant_name, jax.ffi.pycapsule(getattr(lib, variant_name)), platform="CUDA")
        if hasattr(lib,"kp_cudss_permutation"):
            jax.ffi.register_ffi_target("kp_cudss_permutation",
                jax.ffi.pycapsule(lib.kp_cudss_permutation),platform="CUDA")
        _library = lib
        return lib


def cudss_available() -> bool:
    """Check that the optional extension loads and a CUDA JAX device exists."""
    try:
        _load()
        return any(d.platform == "gpu" for d in jax.devices())
    except (CudssUnavailableError, RuntimeError):
        return False


@jax.jit
def _validate_gpu_csr(rp,ci,row_ids):
    n=len(rp)-1;nnz=len(ci)
    return ((rp[0]==0)&(rp[-1]==nnz)&jnp.all(jnp.diff(rp)>=0)
        &jnp.all((ci>=0)&(ci<n))&jnp.all((row_ids>=0)&(row_ids<n))
        &jnp.all(row_ids[1:]>=row_ids[:-1])
        &jnp.all(jnp.bincount(row_ids,length=n)==jnp.diff(rp))
        &jnp.all((row_ids[1:]!=row_ids[:-1])|(ci[1:]>ci[:-1])))


@jax.jit
def _validate_padded_gpu_csr(rp,ci,row_ids,values):
    """Validate exact CSR prefix plus harmless row-sorted zero COO padding."""
    n=len(rp)-1
    valid=jnp.arange(len(ci),dtype=jnp.int32)<rp[-1]
    pairs=jnp.arange(max(0,len(ci)-1),dtype=jnp.int32)+1<rp[-1]
    counts=jnp.bincount(row_ids,weights=valid.astype(jnp.int32),length=n)
    return ((rp[0]==0)&(rp[-1]>0)&(rp[-1]<=len(ci))&jnp.all(jnp.diff(rp)>=0)
        &jnp.all((ci>=0)&(ci<n))&jnp.all((row_ids>=0)&(row_ids<n))
        &jnp.all(row_ids[1:]>=row_ids[:-1])&jnp.all(counts==jnp.diff(rp))
        &jnp.all(jnp.where(pairs,(row_ids[1:]!=row_ids[:-1])|(ci[1:]>ci[:-1]),True))
        &jnp.all(jnp.where(valid,True,(values==0)&(row_ids==n-1))))


@jax.jit
def _transpose_gpu_csr(rows, columns, values, indptr):
    """Sorted CSR transpose without downloading values or index arrays."""
    n=len(indptr)-1
    order=jnp.lexsort((rows,columns))
    tr=jnp.concatenate([jnp.zeros(1,dtype=jnp.int32),
        jnp.cumsum(jnp.bincount(columns,length=n),dtype=jnp.int32)])
    return tr,rows[order].astype(jnp.int32),values[order]


_REORDERING_SELECTORS={"default":0,"none":1,"amd":2}


def _create_context(lib,reordering,*args):
    if reordering=="default":
        return lib.kp_cudss_create(*args)
    if not hasattr(lib,"kp_cudss_create_with_reordering"):
        raise CudssUnavailableError("Rebuild cuDSS FFI for explicit reordering selection")
    return lib.kp_cudss_create_with_reordering(*args,_REORDERING_SELECTORS[reordering])


class CudssFactor:
    """Own a fixed, square CSR matrix and its cuDSS factor on one CUDA device.

    Use as a context manager or call ``close`` after all dependent computations
    have completed. Executables referencing a closed factor fail explicitly.
    ``matrix_type='spd'`` asserts real symmetry and positive definiteness; use
    ``'general'`` for LU. No silent fallback or regularization is introduced.
    ``reordering='default'`` preserves cuDSS's vendor default. Explicit
    ``'none'`` and ``'amd'`` select natural and AMD ordering. The choice is fixed
    for this factor, its LU transpose, pattern updates and failed-update retries.
    CSR must have int32-compatible, sorted, unique column indices in each row.
    ``mutable_values=True`` reserves the contexts for the changing-matrix PDE
    backend; matrix values must then be supplied on every native solve.
    ``padded_capacity=True`` is optional for eager mutable-SPD GP workspaces:
    CSR pointers count only the true active prefix; extra JAX entries must be
    zero and row-sorted. They never enter the native factor graph. Original
    matrix residuals use a prepared rectangular CSR extension, placing each
    zero tail entry at a distinct valid dummy-row coordinate. Construction
    retains one additional int32 column buffer and small extended row pointers.
    """

    def __init__(
        self,
        indptr,
        indices,
        data,
        shape,
        *,
        matrix_type="spd",
        rhs_block_size=8,
        threading_lib=None,
        mutable_values=False,
        row_indices=None,
        profile_setup=False,
        padded_capacity=False,
        padded_action_scratch_bytes=32*1024**2,
        reordering="default",
    ):
        if not isinstance(reordering,str) or reordering not in _REORDERING_SELECTORS:
            raise ValueError("reordering must be 'default', 'none', or 'amd'")
        self._reordering=reordering
        setup_start=time.perf_counter()
        self.setup_timings={}
        if matrix_type not in ("spd", "general"):
            raise ValueError("matrix_type must be 'spd' or 'general'")
        if len(shape) != 2 or shape[0] != shape[1] or shape[0] < 1:
            raise ValueError("cuDSS requires a nonempty square matrix")
        if int(rhs_block_size) != rhs_block_size or rhs_block_size < 1:
            raise ValueError("rhs_block_size must be a positive integer")
        self.rhs_block_size = int(rhs_block_size)
        self.padded_capacity=bool(padded_capacity)
        if not isinstance(padded_action_scratch_bytes,Integral) or padded_action_scratch_bytes<64:
            raise ValueError("padded_action_scratch_bytes must hold a 64-byte index chunk")
        self.padded_action_scratch_bytes=int(padded_action_scratch_bytes)
        if self.padded_capacity and (matrix_type!="spd" or not mutable_values or row_indices is None):
            raise ValueError("Padded capacity requires mutable SPD with explicit row_indices")
        n = int(shape[0])
        device=data.device if isinstance(data,jax.Array) else next(
            (d for d in jax.devices() if d.platform=="gpu"),None)
        if device is None or device.platform!="gpu":
            raise CudssUnavailableError("cuDSS requires a CUDA JAX device")
        rp, ci = jnp.asarray(indptr,device=device), jnp.asarray(indices,device=device)
        if rp.dtype not in (jnp.int32,jnp.int64) or ci.dtype not in (jnp.int32,jnp.int64):
            raise ValueError("CSR indices must be integers")
        if rp.shape != (n+1,) or ci.ndim != 1:
            raise ValueError("Invalid CSR dimensions")
        nnz=len(ci)
        row_ids=(jnp.asarray(row_indices,device=device) if row_indices is not None else
            jnp.repeat(jnp.arange(n,dtype=jnp.int32),jnp.diff(rp),total_repeat_length=nnz))
        if row_ids.shape != (nnz,) or row_ids.dtype not in (jnp.int32,jnp.int64):
            raise ValueError("row_indices must match the integer CSR entry rows")
        capacity=nnz
        if n >= 2**31 or capacity >= 2**31:
            raise ValueError("This cuDSS backend supports int32 CSR sizes")
        if self.padded_capacity and n+(capacity+n-1)//n>=2**31:
            raise ValueError("Extended padded CSR row count must fit int32")
        av = jnp.asarray(data,device=device)
        if av.shape != (nnz,) or av.dtype not in (jnp.float32, jnp.float64):
            raise ValueError(
                "CSR values must be a float32 or float64 vector matching nnz"
            )
        if not bool(jnp.all(jnp.isfinite(av))):
            raise ValueError("CSR values must be finite")
        valid_csr=(_validate_padded_gpu_csr(rp,ci,row_ids,av) if self.padded_capacity
            else _validate_gpu_csr(rp,ci,row_ids))
        if not bool(valid_csr):
            raise ValueError("Invalid CSR prefix, unsorted/duplicate active indices, or nonzero/unsorted padding")
        nnz=int(rp[-1]) if self.padded_capacity else capacity
        device = av.device
        if device.platform != "gpu":
            raise CudssUnavailableError("cuDSS requires a CUDA JAX device")
        self.dtype = av.dtype
        self.shape = (n, n)
        self.actual_nnz=nnz
        self.capacity=capacity
        self.matrix_type = matrix_type
        self.mutable_values = bool(mutable_values)
        self._cuda_graph_enabled = False
        self.device = device
        rows = jax.device_put(rp.astype(np.int32), device)
        cols = jax.device_put(ci.astype(np.int32), device)
        for a in (rows, cols, av):
            a.block_until_ready()
        lib = _load()
        if not hasattr(lib, "kp_cudss_check_status"):
            raise CudssUnavailableError("Rebuild cuDSS FFI for asynchronous variable-width solves")
        if self.mutable_values and not hasattr(lib, "kp_cudss_refactor_f64"):
            raise CudssUnavailableError("Rebuild cuDSS FFI for reusable symbolic analysis")
        if threading_lib is None:
            threading_lib = os.environ.get("CUDSS_THREADING_LIB", "")
        self._threading_lib = os.fsencode(threading_lib)
        if profile_setup:
            self.setup_timings["validation_and_native_load"]=time.perf_counter()-setup_start
        factor_start=time.perf_counter()
        factor_id = _create_context(lib,self.reordering,
            device.local_hardware_id,
            n,
            nnz,
            rows.unsafe_buffer_pointer(),
            cols.unsafe_buffer_pointer(),
            av.unsafe_buffer_pointer(),
            int(av.dtype == jnp.float64),
            int(matrix_type == "spd"),
            self.rhs_block_size,
            self._threading_lib,
        )
        if factor_id < 0:
            raise RuntimeError(
                "cuDSS factorization failed: " + lib.kp_cudss_error().decode()
            )
        if profile_setup:
            self.setup_timings["native_analysis_and_factorization"]=time.perf_counter()-factor_start
        self._id = factor_id
        self._native_context_valid = True
        self._finalizer = weakref.finalize(self, lib.kp_cudss_destroy, factor_id)
        self._transpose_id = factor_id
        self._transpose_finalizer = None
        # Keep the frozen sparse operator for JAX's implicit derivative rule.
        self._rows = jax.device_put(row_ids.astype(jnp.int32),device)
        self._cols = cols
        self._indptr=rows
        self._values = jax.lax.stop_gradient(av)
        self._matrix_norm = jnp.linalg.norm(self._values)
        if self.padded_capacity:
            self._csr_operator=_padded_csr_action_from_validated(
                self._values,self._cols,self._indptr,scratch_bytes=self.padded_action_scratch_bytes)
        else:
            from jax.experimental.sparse import BCSR
            self._csr_operator = BCSR((self._values, cols, rows), shape=self.shape)
        self._csr_transpose = self._csr_operator
        self._exact_action_cache = {}
        self._training_update_counts = {"pattern_rebuilds": 0, "numeric_refactors": 0}
        self._last_update_failed = False
        if matrix_type == "general":
            # cuDSS 0.8 supports only non-transposed solves. Prepare A.T once,
            # using sparse CSR conversion rather than a dense approximate inverse.
            try:
                tr,tc,tv = _transpose_gpu_csr(self._rows,cols,self._values,rows)
                self._csr_transpose = BCSR((jax.lax.stop_gradient(tv), tc, tr), shape=self.shape)
                for a in (tr, tc, tv):
                    a.block_until_ready()
                transpose_id = _create_context(lib,self.reordering,
                    device.local_hardware_id, n, nnz,
                    tr.unsafe_buffer_pointer(), tc.unsafe_buffer_pointer(),
                    tv.unsafe_buffer_pointer(), int(av.dtype == jnp.float64),
                    0, self.rhs_block_size, os.fsencode(threading_lib),
                )
                if transpose_id < 0:
                    raise RuntimeError(
                        "cuDSS transpose factorization failed: "
                        + lib.kp_cudss_error().decode()
                    )
                self._transpose_id = transpose_id
                self._transpose_finalizer = weakref.finalize(
                    self, lib.kp_cudss_destroy, transpose_id
                )
            except BaseException:
                self._finalizer()
                raise

    @property
    def reordering(self):
        """Requested immutable cuDSS ordering for all contexts of this factor."""
        return self._reordering

    @classmethod
    def from_dense(cls, matrix, **kwargs):
        """Eager GPU CSR conversion; its cost is part of model setup."""
        from jax.experimental.sparse import BCSR

        matrix = jnp.asarray(matrix)
        if matrix.device.platform!="gpu":
            raise CudssUnavailableError("cuDSS requires a CUDA JAX device")
        if matrix.ndim != 2:
            raise ValueError("matrix must be two-dimensional")
        nse = int(jnp.count_nonzero(matrix))
        csr = BCSR.fromdense(matrix, nse=nse)
        return cls(csr.indptr, csr.indices, csr.data, matrix.shape, **kwargs)

    def enable_cuda_graph(self, enabled=True):
        """Replay fixed-matrix full solves with an internal CUDA graph.

        Lagrange evaluation and residual checks are unchanged. First use of
        each RHS width captures a graph; subsequent calls include two GPU
        buffer copies. This does not capture the entire JAX predictor.
        Eager mutable-SPD workspaces invalidate captured graphs on every
        numeric update and preserve graph mode after an exact-pattern rebuild.
        """
        self._require_native_context()
        if self.mutable_values and self.matrix_type!="spd":
            raise ValueError("Mutable graph replay is supported for eager SPD workspaces only")
        lib=_load()
        if not hasattr(lib,"kp_cudss_graph_mode"):
            raise CudssUnavailableError("Rebuild cuDSS FFI to enable CUDA graph replay")
        for factor_id in set((self._id,self._transpose_id)):
            if lib.kp_cudss_graph_mode(factor_id,int(enabled))<0:
                raise RuntimeError("cuDSS factor is closed")
        self._cuda_graph_enabled=bool(enabled)

    @property
    def graph_launch_count(self):
        self._require_native_context()
        lib=_load()
        if not hasattr(lib,"kp_cudss_graph_mode"):
            raise CudssUnavailableError("Rebuild cuDSS FFI for CUDA graph counters")
        counts=[int(lib.kp_cudss_graph_mode(i,-1)) for i in set((self._id,self._transpose_id))]
        if any(count<0 for count in counts):
            raise RuntimeError("cuDSS factor is closed")
        return sum(counts)

    def _require_native_context(self):
        if not self._finalizer.alive:
            raise RuntimeError("cuDSS factor is closed")
        if not self._native_context_valid:
            raise RuntimeError("cuDSS workspace is invalid after a failed rebuild; call update_matrix to recover")

    def _solve(self, rhs, *, transpose=False, triangular=None):
        self._require_native_context()
        b = jnp.asarray(rhs)
        if b.dtype != self.dtype or b.ndim not in (1, 2) or b.shape[0] != self.shape[0]:
            raise ValueError("RHS dtype and leading dimension must match the factor")
        vector = b.ndim == 1
        b = b[:, None] if vector else b
        if b.shape[1] < 1:
            raise ValueError("RHS must have at least one column")
        blocks = []
        for start in range(0, b.shape[1], self.rhs_block_size):
            part = b[:, start : start + self.rhs_block_size]
            call = jax.ffi.ffi_call(
                ("kp_cudss_solve_" if triangular is None else "kp_cudss_triangular_") + ("f64" if self.dtype == jnp.float64 else "f32"),
                jax.ShapeDtypeStruct(part.shape, part.dtype),
                has_side_effect=True, vmap_method="sequential",
                input_layouts=[(1, 0)], output_layouts=(1, 0),
            )
            factor_id = self._transpose_id if transpose else self._id
            attrs = dict(factor_id=np.int64(factor_id))
            if triangular is not None:
                attrs["backward"] = np.int64(triangular)
            blocks.append(call(part, **attrs))
        x = jnp.concatenate(blocks, axis=1)
        return x[:, 0] if vector else x

    def resource_counts(self):
        """Native descriptor/event bundles created and reused (diagnostics)."""
        lib=_load();created=ctypes.c_int64();reused=ctypes.c_int64()
        if lib.kp_cudss_resource_counts(self._id,ctypes.byref(created),ctypes.byref(reused)) != 0:
            raise RuntimeError("cuDSS factor is closed")
        return dict(created=created.value,reused=reused.value)

    def storage_statistics(self):
        """Native factor fill and cuDSS workspace estimates, in bytes.

        Estimates describe the analysis-time RHS capacity and exclude the
        wrapper's owned CSR, graph RHS/solution buffers and JAX arrays. They
        are not process-wide peak GPU memory measurements.
        """
        lib=_load()
        if not hasattr(lib,"kp_cudss_factor_stats"):
            raise CudssUnavailableError("Rebuild cuDSS FFI for factor storage statistics")
        fill=ctypes.c_int64()
        memory=(ctypes.c_int64*16)()
        if lib.kp_cudss_factor_stats(self._id,ctypes.byref(fill),memory)!=0:
            raise RuntimeError("cuDSS factor statistics failed: "+lib.kp_cudss_error().decode())
        return dict(reordering=self.reordering,factor_nnz=fill.value,permanent_device_bytes=memory[0],
            peak_device_bytes=memory[1],permanent_host_bytes=memory[2],peak_host_bytes=memory[3],
            actual_nnz=self.actual_nnz,buffer_capacity=self.capacity,
            owned_csr_bytes=(self.shape[0]+1+self.actual_nnz)*4+self.actual_nnz*self.dtype.itemsize,
            padded_jax_entry_bytes=self.capacity*(8+self.dtype.itemsize),
            padded_action_index_bytes=(self._csr_operator.indices.nbytes+self._csr_operator.indptr.nbytes
                                       if self.padded_capacity else 0),
            padded_action_extra_rows=(self._csr_operator.extended_shape[0]-self.shape[0]
                                      if self.padded_capacity else 0),
            padded_action_build_scratch_bytes=(min(self.capacity,self.padded_action_scratch_bytes//64)*64
                                               if self.padded_capacity else 0),
            graph_dense_buffer_bytes=2*self.shape[0]*self.rhs_block_size*self.dtype.itemsize)

    def triangular_solve(self, rhs, *, backward=False):
        """SPD whitening L^{-1} Pi b, or adjoint Pi.T L^{-T} b.

        The reordered Cholesky factor stays native. These factor actions are
        intended for eager QR setup and JIT inference; no AD rule is supplied.
        """
        if self.matrix_type != "spd" or self.mutable_values:
            raise ValueError("Triangular actions require an immutable SPD factor")
        lib = _load()
        if not hasattr(lib, "kp_cudss_triangular_f64"):
            raise CudssUnavailableError("Rebuild cuDSS FFI for triangular actions")
        return self._solve(rhs, triangular=bool(backward))

    def refactor_values(self, values):
        """Eager numeric-only update of a mutable SPD workspace.

        Call after all dependent JAX results have completed. The CSR pattern
        is unchanged, so symbolic analysis and the native handle are reused.
        No matrix data are copied to the CPU. This method is intentionally
        outside JIT/AD; training supplies its sparse implicit derivatives.
        """
        self._require_native_context()
        if not self.mutable_values or self.matrix_type != "spd":
            raise ValueError("Eager refactor requires a mutable SPD factor")
        lib=_load()
        if not hasattr(lib,"kp_cudss_refactor_values"):
            raise CudssUnavailableError("Rebuild cuDSS FFI for eager SPD refactoring")
        values=jnp.asarray(values,device=self.device)
        if values.shape!=self._values.shape or values.dtype!=self.dtype:
            raise ValueError("Refactor values must match the current CSR size and dtype")
        if not bool(jnp.all(jnp.isfinite(values))):
            raise ValueError("Refactor values must be finite")
        if self.padded_capacity and not bool(_validate_padded_gpu_csr(self._indptr,self._cols,self._rows,values)):
            raise ValueError("Padded refactor must preserve a zero, row-sorted tail")
        values.block_until_ready()
        if lib.kp_cudss_refactor_values(self._id,values.unsafe_buffer_pointer())!=0:
            self._last_update_failed=True
            raise RuntimeError("cuDSS refactor failed: "+lib.kp_cudss_error().decode())
        self._last_update_failed=False
        self._values=jax.lax.stop_gradient(values)
        self._matrix_norm=jnp.linalg.norm(self._values)
        if self.padded_capacity:
            self._csr_operator=self._csr_operator.with_data(self._values)
        else:
            from jax.experimental.sparse import BCSR
            self._csr_operator=BCSR((self._values,self._cols,self._indptr),shape=self.shape)
        self._csr_transpose=self._csr_operator
        self._training_update_counts["numeric_refactors"]+=1

    def update_matrix(self, indptr, indices, values, *, row_indices=None):
        """Update an exact CSR training matrix while keeping its handle stable.

        Complete pattern equality is checked on GPU. An unchanged pattern
        reuses symbolic analysis; a changed pattern gets a new exact sparse
        factor, with no inflated support superset. The old factor is released
        before rebuilding, so two complete factors never overlap. A failed
        rebuild invalidates the workspace until a subsequent update succeeds.
        Cached action executables
        keep the same handle and receive CSR arrays dynamically. New array
        shapes may require a new compilation. Drain dependent results before
        calling; previously prepared predictions are not valid after updates.
        """
        if not self.mutable_values or self.matrix_type!="spd":
            raise ValueError("Matrix updates require a mutable SPD workspace")
        if not self._finalizer.alive:
            raise RuntimeError("cuDSS factor is closed")
        lib=_load()
        if self.reordering=="default":
            if not hasattr(lib,"kp_cudss_rebuild_context"):
                raise CudssUnavailableError("Rebuild cuDSS FFI for bounded-memory exact-pattern updates")
            rebuild=lib.kp_cudss_rebuild_context
        else:
            if not hasattr(lib,"kp_cudss_rebuild_context_with_reordering"):
                raise CudssUnavailableError("Rebuild cuDSS FFI for explicitly ordered exact-pattern updates")
            rebuild=lib.kp_cudss_rebuild_context_with_reordering
        pointers=jnp.asarray(indptr,device=self.device)
        columns=jnp.asarray(indices,device=self.device)
        values=jnp.asarray(values,device=self.device)
        if pointers.dtype not in (jnp.int32,jnp.int64) or columns.dtype not in (jnp.int32,jnp.int64):
            raise ValueError("Updated CSR pointers and indices must be integers")
        if values.dtype!=self.dtype:
            raise ValueError("Updated values must preserve workspace dtype")
        if pointers.shape!=self._indptr.shape:
            raise ValueError("Updated matrix must preserve workspace dimensions")
        if columns.ndim!=1 or values.shape!=columns.shape or len(columns)>=2**31:
            raise ValueError("Updated CSR entries must match and fit int32 indexing")
        n=self.shape[0]
        if self.padded_capacity and n+(len(columns)+n-1)//n>=2**31:
            raise ValueError("Extended padded CSR row count must fit int32")
        if self.padded_capacity and row_indices is None:
            raise ValueError("Padded matrix update requires explicit row_indices")
        row_indices=(jnp.asarray(row_indices,device=self.device) if row_indices is not None else
            jnp.repeat(jnp.arange(self.shape[0],dtype=jnp.int32),jnp.diff(pointers),
                       total_repeat_length=len(columns)))
        if row_indices.shape!=columns.shape or row_indices.dtype not in (jnp.int32,jnp.int64):
            raise ValueError("Updated row indices must match the integer CSR entry rows")
        valid=(_validate_padded_gpu_csr(pointers,columns,row_indices,values) if self.padded_capacity else
               _validate_gpu_csr(pointers,columns,row_indices))
        if not bool(valid & jnp.all(jnp.isfinite(values))):
            raise ValueError("Invalid updated CSR prefix, indices, values or padding")
        active=jnp.arange(len(columns),dtype=jnp.int32)<pointers[-1]
        same=(columns.shape==self._cols.shape and
            bool(jnp.all(pointers==self._indptr)&jnp.all(jnp.where(active,columns==self._cols,True))))
        if same and self._native_context_valid:
            self.refactor_values(values)
            return False
        if self._native_context_valid and not self._last_update_failed:
            self.check_status()
        pointers=pointers.astype(jnp.int32)
        columns=columns.astype(jnp.int32)
        row_indices=row_indices.astype(jnp.int32)
        for array in (pointers,columns,row_indices,values):
            array.block_until_ready()
        # Public factor actions must reject this update even if preparing the
        # new device action raises (for example under memory pressure). Drop
        # old mapped buffers before allocating the new support coordinates.
        # Native CSR storage owns its own copies until the bounded rebuild.
        self._native_context_valid=False
        self._last_update_failed=True
        self._csr_operator=None
        self._csr_transpose=None
        self._rows,self._cols,self._values=(row_indices,columns,jax.lax.stop_gradient(values))
        self._indptr=pointers
        self.actual_nnz=int(pointers[-1]) if self.padded_capacity else len(columns)
        self.capacity=len(columns)
        self._matrix_norm=jnp.linalg.norm(self._values)
        if self.padded_capacity:
            self._csr_operator=_padded_csr_action_from_validated(
                self._values,self._cols,self._indptr,scratch_bytes=self.padded_action_scratch_bytes)
        else:
            from jax.experimental.sparse import BCSR
            self._csr_operator=BCSR((self._values,self._cols,self._indptr),shape=self.shape)
        self._csr_transpose=self._csr_operator
        ordering_args=() if self.reordering=="default" else (_REORDERING_SELECTORS[self.reordering],)
        if rebuild(self._id,self.device.local_hardware_id,self.shape[0],
                self.actual_nnz,pointers.unsafe_buffer_pointer(),columns.unsafe_buffer_pointer(),
                values.unsafe_buffer_pointer(),int(self.dtype==jnp.float64),1,
                self.rhs_block_size,self._threading_lib,*ordering_args)!=0:
            raise RuntimeError("cuDSS factorization failed: "+lib.kp_cudss_error().decode())
        self._native_context_valid=True
        if self._cuda_graph_enabled:
            self.enable_cuda_graph()
        self._training_update_counts["pattern_rebuilds"]+=1
        self._last_update_failed=False
        return True

    @property
    def training_update_counts(self):
        """Counts accumulated across exact-pattern workspace replacements."""
        return dict(self._training_update_counts)

    def _checked_current_solve(self, values, rows, columns, indptr, rhs, operator=None):
        if self.padded_capacity:
            if operator is None:
                raise ValueError("Padded current solve requires its dynamic prepared CSR action")
            matvec=operator.native_matmul
        else:
            from jax.experimental.sparse import CSR
            matrix=CSR((values,columns,indptr),shape=self.shape)
            matvec=lambda x:matrix@x
        result=self._batched_solve(rhs,False)
        return _check_solution(result,rhs,matvec,anorm=jnp.linalg.norm(values))

    def solve_current(self, rhs):
        """Checked GPU solve of an eagerly updated SPD workspace, without AD.

        Current CSR arrays are dynamic JIT inputs so repeated updates cannot
        reuse a stale residual matrix. Use ``solve`` for immutable inference.
        """
        self._require_native_context()
        if not self.mutable_values or self.matrix_type!="spd":
            raise ValueError("Current-factor solve requires a mutable SPD workspace")
        key=("current_solve",)
        if key not in self._exact_action_cache:
            self._exact_action_cache[key]=jax.jit(self._checked_current_solve)
        operator=self._csr_operator if self.padded_capacity else None
        return self._exact_action_cache[key](self._values,self._rows,self._cols,self._indptr,rhs,operator)

    def whiten_current(self, rhs, *, backward=False):
        """GPU whitening or its adjoint for an eagerly updated SPD workspace."""
        self._require_native_context()
        if not self.mutable_values or self.matrix_type!="spd":
            raise ValueError("Current whitening requires a mutable SPD workspace")
        key=("current_whiten",bool(backward))
        if key not in self._exact_action_cache:
            self._exact_action_cache[key]=jax.jit(lambda b:self._solve(b,triangular=bool(backward)))
        return self._exact_action_cache[key](rhs)

    def logdet(self, *, block_size=None):
        """Exact SPD log determinant using bounded GPU factor actions.

        cuDSS 0.8 does not expose the SPD Cholesky diagonal for the reordering
        algorithms needed by ``triangular_solve``. The diagonal of L^-1 is
        therefore extracted from identity-column blocks in factor coordinates:
        log(det(A)) = -2 sum(log(diag(L^-1))). No dense matrix or inverse is
        formed. This requires N triangular RHS columns; its cost must be
        counted as training work rather than advertised as constant time.
        """
        self._require_native_context()
        if self.matrix_type != "spd":
            raise ValueError("Log determinant requires an SPD factor")
        width = self.rhs_block_size if block_size is None else block_size
        if int(width) != width or width < 1:
            raise ValueError("block_size must be a positive integer")
        width = min(int(width), self.rhs_block_size, self.shape[0])
        if not hasattr(_load(), "kp_cudss_permutation"):
            raise CudssUnavailableError("Rebuild cuDSS FFI for exact SPD log determinants")
        n = self.shape[0]
        key=("logdet",width)
        if key in self._exact_action_cache:
            return self._exact_action_cache[key]()

        @jax.jit
        def compute():
            permutation = self.reordering_permutation()
            row = jnp.arange(n, dtype=jnp.int32)[:, None]
            local = jnp.arange(width, dtype=jnp.int32)

            def block(total, start):
                column = start + local
                owner = permutation[jnp.minimum(column,n-1)]
                rhs = ((row == owner[None, :]) & (column[None, :] < n)).astype(self.dtype)
                inverse_lower = self._solve(rhs,triangular=False)
                diagonal = inverse_lower[jnp.minimum(column, n-1), local]
                valid = column < n
                good = jnp.all(jnp.where(valid, jnp.isfinite(diagonal) & (diagonal > 0), True))
                contribution = jnp.sum(jnp.where(valid, jnp.log(jnp.where(valid, diagonal, 1.)), 0.))
                return jnp.where(good, total - 2*contribution, jnp.nan), None

            starts = jnp.arange((n + width - 1)//width, dtype=jnp.int32)*width
            return jax.lax.scan(block, jnp.asarray(0., dtype=self.dtype), starts)[0]

        self._exact_action_cache[key]=compute
        return compute()

    def reordering_permutation(self):
        """Return cuDSS's row reordering as an int32 GPU vector."""
        self._require_native_context()
        if self.matrix_type != "spd":
            raise ValueError("Permutation action requires an SPD factor")
        if not hasattr(_load(),"kp_cudss_permutation"):
            raise CudssUnavailableError("Rebuild cuDSS FFI for device reordering access")
        return jax.ffi.ffi_call("kp_cudss_permutation",
            jax.ShapeDtypeStruct((self.shape[0],),jnp.int32),has_side_effect=True,
            output_layouts=(0,),vmap_method="sequential")(factor_id=np.int64(self._id))

    def inverse_on_support(self, *, block_size=None):
        """Exact inverse entries on this factor's CSR pattern, entirely on GPU.

        Solves stream N identity RHS columns through the existing checked
        sparse factor and retain only ``A^-1[rows, cols]``. Storage is O(nnz +
        N*block_size), not a dense N-by-N inverse. In particular, for symmetric
        dA with this pattern, ``sum(result * dA_values)`` equals
        ``trace(A^-1 dA)``. No stochastic trace approximation is introduced.
        """
        self._require_native_context()
        if self.matrix_type != "spd":
            raise ValueError("Selected inverse requires an SPD factor")
        width = self.rhs_block_size if block_size is None else block_size
        if int(width) != width or width < 1:
            raise ValueError("block_size must be a positive integer")
        width = min(int(width), self.rhs_block_size, self.shape[0])
        n = self.shape[0]
        key=("inverse_on_support",width)
        if key in self._exact_action_cache:
            operator=self._csr_operator if self.padded_capacity else None
            return self._exact_action_cache[key](self._rows,self._cols,self._indptr,self._values,operator)

        @jax.jit
        def compute(rows,columns,indptr,values,operator):
            row = jnp.arange(n, dtype=jnp.int32)[:, None]
            local = jnp.arange(width, dtype=jnp.int32)

            def block(selected, start):
                column = start + local
                rhs = (row == column[None, :]).astype(self.dtype)
                solved = self._checked_current_solve(values,rows,columns,indptr,rhs,operator)
                owns = (columns >= start) & (columns < start + width) & (jnp.arange(len(values))<indptr[-1])
                entries = solved[rows, jnp.clip(columns-start, 0, width-1)]
                return jnp.where(owns, entries, selected), None

            starts = jnp.arange((n + width - 1)//width, dtype=jnp.int32)*width
            return jax.lax.scan(block, jnp.zeros_like(values), starts)[0]

        self._exact_action_cache[key]=compute
        operator=self._csr_operator if self.padded_capacity else None
        return compute(self._rows,self._cols,self._indptr,self._values,operator)

    def inverse_on_support_and_logdet(self, *, block_size=None):
        """Return exact CSR inverse entries and the SPD log determinant.

        Each original-coordinate identity block passes through the forward
        factor action once. Its factor-coordinate diagonal gives the log
        determinant; the backward action then gives the inverse columns.
        The composed solve retains the same full original-CSR residual check
        as ``inverse_on_support``. A failed block invalidates both outputs.
        Only CSR inverse entries and bounded RHS blocks are stored, requiring
        O(nnz + N*block_size) space and 2N triangular RHS columns in total.

        Mutable workspaces fetch their current reordering on every execution
        and pass their current CSR buffers dynamically. No inverse entries,
        diagonal values, or numerical permutation are cached across updates.
        """
        self._require_native_context()
        if self.matrix_type != "spd":
            raise ValueError("Selected inverse and log determinant require an SPD factor")
        width = self.rhs_block_size if block_size is None else block_size
        if int(width) != width or width < 1:
            raise ValueError("block_size must be a positive integer")
        width = min(int(width), self.rhs_block_size, self.shape[0])
        if not hasattr(_load(), "kp_cudss_permutation"):
            raise CudssUnavailableError("Rebuild cuDSS FFI for exact SPD log determinants")
        n = self.shape[0]
        key=("inverse_on_support_and_logdet",width)
        if key in self._exact_action_cache:
            operator=self._csr_operator if self.padded_capacity else None
            return self._exact_action_cache[key](self._rows,self._cols,self._indptr,self._values,operator)

        @jax.jit
        def compute(rows,columns,indptr,values,operator):
            if self.padded_capacity:
                matvec=operator.native_matmul
            else:
                from jax.experimental.sparse import CSR
                matrix=CSR((values,columns,indptr),shape=self.shape)
                matvec=lambda x:matrix@x
            matrix_norm = jnp.linalg.norm(values)
            # permutation[k] is the original index owned by factor row k.
            # Original identity column j therefore has its inverse-L diagonal
            # at factor row inverse_permutation[j]. Compute this on the GPU
            # after reading the current native factor's ordering.
            inverse_permutation = jnp.argsort(self.reordering_permutation())
            row = jnp.arange(n, dtype=jnp.int32)[:, None]
            local = jnp.arange(width, dtype=jnp.int32)
            active = jnp.arange(len(values)) < indptr[-1]

            def block(carry, start):
                selected, total, all_good = carry
                column = start + local
                valid = column < n
                rhs = ((row == column[None, :]) & valid[None, :]).astype(self.dtype)
                inverse_lower = self._solve(rhs,triangular=False)
                factor_row = inverse_permutation[jnp.minimum(column,n-1)]
                diagonal = inverse_lower[factor_row,local]
                solved = self._solve(inverse_lower,triangular=True)
                solved = _check_solution(solved,rhs,matvec,anorm=matrix_norm)
                good = (jnp.all(jnp.isfinite(inverse_lower))
                    & jnp.all(jnp.isfinite(solved))
                    & jnp.all(jnp.where(valid,jnp.isfinite(diagonal)&(diagonal>0),True)))
                contribution = jnp.sum(jnp.log(jnp.where(valid,diagonal,1.)))
                owns = (columns >= start) & (columns < start + width) & active
                entries = solved[rows,jnp.clip(columns-start,0,width-1)]
                return (jnp.where(owns,entries,selected),total-2*contribution,all_good&good), None

            starts = jnp.arange((n + width - 1)//width,dtype=jnp.int32)*width
            initial = (jnp.zeros_like(values),jnp.asarray(0.,dtype=self.dtype),jnp.asarray(True))
            selected,total,good = jax.lax.scan(block,initial,starts)[0]
            # Preserve the zero padded tail even when an actual entry fails.
            selected = jnp.where(active,jnp.where(good,selected,jnp.nan),0.)
            return selected,jnp.where(good,total,jnp.nan)

        self._exact_action_cache[key]=compute
        operator=self._csr_operator if self.padded_capacity else None
        return compute(self._rows,self._cols,self._indptr,self._values,operator)

    def _batched_solve(self, rhs, transpose):
        @custom_vmap
        def run(b):
            return self._solve(b, transpose=transpose)

        @run.def_vmap
        def rule(axis_size, in_batched, b):
            if not in_batched[0]:
                return run(b), False
            packed = jnp.moveaxis(b, 1, 0)
            solved = run(packed.reshape(self.shape[0], -1)).reshape(packed.shape)
            return jnp.moveaxis(solved, 0, 1), True
        return run(rhs)

    def matmul(self,rhs,*,transpose=False):
        """Native CSR product; pack mapped RHS axes before the cuSPARSE call.

        The JVP uses BCSR's batchable linear rule, retaining RHS derivatives
        without requiring a CSR primitive batching rule or global JAX flags.
        """
        from jax.experimental.sparse import CSR
        matrix=self._csr_transpose if transpose else self._csr_operator
        native=(matrix.native_matmul if self.padded_capacity else
                lambda b:CSR((matrix.data,matrix.indices,matrix.indptr),shape=self.shape)@b)
        @custom_vmap
        def run(b):
            return native(b)
        @run.def_vmap
        def batch(axis_size,in_batched,b):
            if not in_batched[0]:
                return run(b),False
            packed=jnp.moveaxis(b,1,0)
            result=run(packed.reshape(self.shape[0],-1)).reshape(packed.shape)
            return jnp.moveaxis(result,0,1),True
        @jax.custom_jvp
        def product(b):
            return run(b)
        @product.defjvp
        def product_jvp(primals,tangents):
            b,=primals;db,=tangents
            return run(b),matrix@db
        return product(rhs)

    def original_matrix_operator(self):
        """Read-only GPU-array snapshot of this factor's original matrix action.

        The pytree has no factor/native references and is passed explicitly to
        compiled residual checks. Its native_matmul uses exact original-prefix
        values; only validated zero capacity entries occupy separate dummy rows.
        Updating the factor does not mutate an earlier snapshot. Values-only
        refactors share the prepared indices. No numerical array is downloaded.
        """
        self._require_native_context()
        if self.padded_capacity:
            return self._csr_operator
        return PaddedCSRAction(self._values,self._cols,self._indptr,self.shape[0])

    def device_state(self):
        """Return immutable GPU CSR data for explicit compiled-call inputs.

        Pass this tuple to ``solve_with_state`` as a dynamic JIT argument.
        Doing so keeps closed-over device arrays out of JAX's constant
        handling, including when device-to-host transfers are forbidden.
        The tuple belongs to this factor; it does not update its native data.
        """
        if not self._finalizer.alive:
            raise RuntimeError("cuDSS factor is closed")
        if self.mutable_values:
            raise ValueError("Immutable device state is unavailable for mutable factors")
        return (self._rows,self._cols,self._indptr,self._values,self._matrix_norm)

    def solve_with_state(self, rhs, state, *, transpose=False):
        """Checked immutable solve with every sparse array supplied dynamically.

        ``state`` is the unchanged tuple returned by this factor's
        ``device_state``. RHS derivatives, mapped RHS packing, primal and
        adjoint residual checks have the same rules as ``solve``. Matrix
        derivatives remain unsupported. General LU prepares separate native
        factors for A and A.T; both residual operators use this explicit data.
        """
        if not self._finalizer.alive:
            raise RuntimeError("cuDSS factor is closed")
        if self.mutable_values:
            raise ValueError("Mutable cuDSS factors require matrix values on every solve")
        if not isinstance(state,(tuple,list)) or len(state)!=5:
            raise ValueError("Expected this factor's five-array immutable device state")
        rows,columns,indptr,values,anorm=map(jax.lax.stop_gradient,state)
        if (rows.shape!=self._rows.shape or columns.shape!=self._cols.shape
                or indptr.shape!=(self.shape[0]+1,) or values.shape!=self._values.shape
                or values.dtype!=self.dtype or anorm.shape!=()):
            raise ValueError("Device state must match this immutable factor")
        b = jnp.asarray(rhs)
        if b.dtype != self.dtype or b.ndim not in (1, 2) or b.shape[0] != self.shape[0]:
            raise ValueError("RHS dtype and leading dimension must match the factor")
        if b.ndim == 2 and b.shape[1] < 1:
            raise ValueError("RHS must have at least one column")
        from jax.experimental.sparse import BCSR,CSR
        matrix=BCSR((values,columns,indptr),shape=self.shape)
        if self.matrix_type=="spd":
            matrix_t=matrix
        else:
            tr,tc,tv=_transpose_gpu_csr(rows,columns,values,indptr)
            matrix_t=BCSR((tv,tc,tr),shape=self.shape)

        def apply(x, transposed):
            return (matrix_t if transposed else matrix)@x

        def checked_product(rhs, transposed):
            sparse=matrix_t if transposed else matrix
            @custom_vmap
            def run(value,data,indices,pointers):
                return CSR((data,indices,pointers),shape=self.shape)@value
            @run.def_vmap
            def batch(axis_size,in_batched,value,data,indices,pointers):
                if any(in_batched[1:]):
                    raise ValueError("Only RHS batching is supported for a fixed cuDSS factor")
                if not in_batched[0]:
                    return run(value,data,indices,pointers),False
                packed=jnp.moveaxis(value,1,0)
                result=run(packed.reshape(self.shape[0],-1),data,indices,pointers).reshape(packed.shape)
                return jnp.moveaxis(result,0,1),True
            @jax.custom_jvp
            def product(value,data,indices,pointers):
                return run(value,data,indices,pointers)
            @product.defjvp
            def product_jvp(primals,tangents):
                value,data,indices,pointers=primals
                tangent=tangents[0]
                linear=BCSR((data,indices,pointers),shape=self.shape)
                return run(value,data,indices,pointers),linear@tangent
            return product(rhs,sparse.data,sparse.indices,sparse.indptr)

        def direct(target, transposed):
            solution = self._batched_solve(target, transposed)
            return _check_solution(solution,target,lambda x:checked_product(x,transposed),anorm=anorm)

        return jax.lax.custom_linear_solve(
            lambda x: apply(x, transpose), b,
            solve=lambda _operator, target: direct(target, transpose),
            transpose_solve=lambda _operator, target: direct(target, not transpose),
            symmetric=self.matrix_type == "spd",
        )

    def solve(self, rhs, *, transpose=False):
        """Solve A x = rhs (or A.T x = rhs), differentiating the RHS only.

        The matrix is fixed setup-time data. For guarded compiled inference,
        pass ``device_state()`` explicitly to ``solve_with_state`` so JAX need
        not process closed-over device arrays as compile-time constants.
        """
        return self.solve_with_state(rhs,self.device_state(),transpose=transpose)

    def solve_transpose(self, rhs):
        """Solve A.T x = rhs using the prepared transpose factor."""
        return self.solve(rhs, transpose=True)

    def _refactor_solve(self, values, rhs, *, transpose=False):
        """Native numerical refactor/solve; caller provides the implicit AD rule."""
        if not self._finalizer.alive:
            raise RuntimeError("cuDSS factor is closed")
        if not self.mutable_values:
            raise ValueError("Refactorization requires mutable_values=True")
        if self.padded_capacity:
            raise ValueError("Padded workspaces require eager update_matrix and solve_current")
        values, rhs = jnp.asarray(values), jnp.asarray(rhs)
        if values.shape != self._values.shape or values.dtype != self.dtype:
            raise ValueError("Refactor values must match the prepared CSR values")
        if rhs.ndim != 2 or rhs.shape[0] != self.shape[0] or rhs.dtype != self.dtype:
            raise ValueError("Refactor RHS must be a matrix matching factor dimensions/dtype")
        factor_id = self._transpose_id if transpose else self._id
        blocks=[]
        for start in range(0,rhs.shape[1],self.rhs_block_size):
            part=rhs[:,start:start+self.rhs_block_size]
            call=jax.ffi.ffi_call(
                "kp_cudss_refactor_"+("f64" if self.dtype==jnp.float64 else "f32"),
                jax.ShapeDtypeStruct(part.shape,part.dtype),has_side_effect=True,
                vmap_method="sequential",input_layouts=[(0,),(1,0)],output_layouts=(1,0))
            blocks.append(call(values,part,factor_id=np.int64(factor_id)))
        return jnp.concatenate(blocks,axis=1)

    def _variant_solve(self, values, rhs, variant_id, *, transpose=False):
        """Solve an immutable registered matrix variant, reusing its current LU.

        Internal API: the owning pattern binds each key permanently to frozen
        CSR values. Changing matrices must use ``_refactor_solve`` instead.
        Primal and transpose contexts track their active keys independently.
        """
        if not self._finalizer.alive:
            raise RuntimeError("cuDSS factor is closed")
        if not self.mutable_values:
            raise ValueError("Matrix variants require mutable_values=True")
        if self.padded_capacity:
            raise ValueError("Padded workspaces require eager update_matrix and solve_current")
        if not isinstance(variant_id, (int, np.integer)) or variant_id < 0:
            raise ValueError("Frozen variant keys must be nonnegative integers")
        values, rhs = jnp.asarray(values), jnp.asarray(rhs)
        if values.shape != self._values.shape or values.dtype != self.dtype:
            raise ValueError("Variant values must match the prepared CSR values")
        if rhs.ndim != 2 or rhs.shape[0] != self.shape[0] or rhs.dtype != self.dtype or rhs.shape[1] < 1:
            raise ValueError("Variant RHS must be a nonempty matrix matching the factor")
        name = "kp_cudss_variant_" + ("f64" if self.dtype == jnp.float64 else "f32")
        if not hasattr(_load(), name):
            raise CudssUnavailableError("Rebuild cuDSS FFI for immutable matrix variants")
        factor_id = self._transpose_id if transpose else self._id
        blocks = []
        for start in range(0, rhs.shape[1], self.rhs_block_size):
            part = rhs[:, start:start + self.rhs_block_size]
            call = jax.ffi.ffi_call(
                name, jax.ShapeDtypeStruct(part.shape, part.dtype),
                has_side_effect=True, vmap_method="sequential",
                input_layouts=[(0,), (1, 0)], output_layouts=(1, 0))
            blocks.append(call(values, part, factor_id=np.int64(factor_id),
                               variant_id=np.int64(variant_id)))
        return jnp.concatenate(blocks, axis=1)

    @property
    def phase_counts(self):
        """Host counters for prepared analyses and enqueued numerical refactors."""
        lib=_load()
        totals={"analysis":0,"refactor":0}
        for factor_id in set((self._id,self._transpose_id)):
            analysis,refactors=ctypes.c_int64(),ctypes.c_int64()
            if lib.kp_cudss_phase_counts(factor_id,ctypes.byref(analysis),ctypes.byref(refactors)) != 0:
                raise RuntimeError("cuDSS factor is closed")
            totals["analysis"]+=analysis.value
            totals["refactor"]+=refactors.value
        return totals

    @property
    def solve_count(self):
        lib = _load()
        count = int(lib.kp_cudss_solve_count(self._id))
        if self._transpose_id != self._id and count >= 0:
            count += int(lib.kp_cudss_solve_count(self._transpose_id))
        return count

    def close(self):
        try:
            # A failed eager SPD update has already raised its numerical
            # error. Native destruction still drains the stream; reporting
            # the same INFO again here would mask optimizer rejection or a
            # preceding exception. Immutable/unreported errors keep checks.
            if self._finalizer.alive and not (self.mutable_values and self._last_update_failed):
                self.check_status()
        finally:
            try:
                if self._transpose_finalizer is not None:
                    self._transpose_finalizer()
                self._finalizer()
            finally:
                # Cached jitted bound methods close over this factor. Break
                # that cycle after native work drains so its GPU CSR buffers
                # can be released when the caller drops the closed factor.
                self._exact_action_cache.clear()
                self._native_context_valid=False

    def check_status(self):
        """Drain dispatched work and check native status at a block boundary.

        Wait for dependent JAX results first; this cannot drain FFI handlers
        JAX has not dispatched yet. Steady solves do not wait on the CPU.
        """
        lib = _load()
        for factor_id in set((self._id, self._transpose_id)):
            if lib.kp_cudss_check_status(factor_id) != 0:
                raise RuntimeError("cuDSS status check failed: " + lib.kp_cudss_error().decode())

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
