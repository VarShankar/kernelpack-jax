"""Fixed sparse powers of a full square nodal Laplacian.

The power is formed on the complete ``[interior; boundary; ghost]`` node set.
Only after the final sparse product are ghost equation rows removed, leaving
the physical-row operator expected by the ADR system builder.  This is a
setup-time operation: the geometry/operator is fixed and is not differentiated.
No intermediate is densified and no tolerance-based dropping is performed.
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import numpy as np

from ._common import SparseCOOMatrix


SparsePowerBackend = Literal["cupy", "cpu"]


class SparsePowerBackendUnavailable(RuntimeError):
    """Raised when an explicitly requested sparse-product backend is absent."""


class SparsePowerComputationError(RuntimeError):
    """Raised with the product stage that failed during sparse setup."""


def _validate_input(matrix: SparseCOOMatrix, power: int, n_physical: int):
    if isinstance(power, bool) or int(power) != power or int(power) < 1:
        raise ValueError("power must be a positive integer")
    power = int(power)
    if isinstance(n_physical, bool) or int(n_physical) != n_physical:
        raise ValueError("n_physical must be an integer")
    n_physical = int(n_physical)
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("full Laplacian input must be square")
    n_full = int(matrix.shape[0])
    if not 0 < n_physical <= n_full:
        raise ValueError("n_physical must lie in [1, n_full]")

    indices = np.asarray(matrix.indices)
    values = np.asarray(matrix.values)
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError("COO indices must have shape (nnz, 2)")
    if values.ndim != 1 or values.size != indices.shape[0]:
        raise ValueError("COO values must be a vector matching the indices")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("COO indices must be integers")
    if values.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("sparse powers require float32 or float64 values")
    if indices.size and (
        np.any(indices < 0)
        or np.any(indices[:, 0] >= n_full)
        or np.any(indices[:, 1] >= n_full)
    ):
        raise ValueError("COO index lies outside the full square matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("COO values must be finite")
    return power, n_physical, indices.astype(np.int32, copy=False), values


def power_sparse_laplacian_rows(
    full_laplacian: SparseCOOMatrix,
    power: int,
    n_physical: int,
    *,
    backend: SparsePowerBackend = "cupy",
    block_size: int = 128,
) -> SparseCOOMatrix:
    """Return the physical rows of an exact sparse power of a square Laplacian.

    Args:
        full_laplacian: Square COO matrix over every interior, boundary, and
            ghost node. Its node ordering must match the caller's ADR state.
        power: Positive integer ``k``; computes ``L**k`` by repeated sparse
            matrix products without forming a dense matrix.
        n_physical: Number of leading physical rows to retain after the final
            product. No row is cropped from an intermediate power.
        backend: ``"cupy"`` uses CuPy/CUDA sparse CSR multiplication. The
            explicit ``"cpu"`` option uses SciPy CSR. There is no automatic
            fallback: selecting a backend is part of the numerical setup.
        block_size: Number of full-square output rows multiplied per CuPy
            block. The GPU path always computes all rows and columns; it only
            reduces the temporary row extent of each sparse product.

    Returns:
        A ``SparseCOOMatrix`` of shape ``(n_physical, n_full)``. Duplicate
        entries are coalesced. Exact zeros may remain stored in the CuPy
        result because its zero-compression kernel corrupted signed values
        in a reproduced environment; stored zeros do not change the matrix.
        No magnitude threshold is applied.

    This operation is intended for immutable setup data. The returned matrix
    values are constants for subsequent JAX differentiation through state and
    parameter-dependent right-hand sides.
    """
    power, n_physical, indices, values = _validate_input(
        full_laplacian, power, n_physical
    )
    if isinstance(block_size, bool) or int(block_size) != block_size or int(block_size) < 1:
        raise ValueError("block_size must be a positive integer")
    block_size = int(block_size)
    backend = str(backend).lower()
    if backend not in ("cupy", "cpu"):
        raise ValueError("backend must be 'cupy' or 'cpu'")

    if backend == "cpu":
        return _power_cpu(indices, values, full_laplacian.shape, power, n_physical)
    return _power_cupy(
        indices, values, full_laplacian.shape, power, n_physical, block_size
    )


def _power_cpu(indices, values, shape, power, n_physical):
    from scipy.sparse import coo_matrix

    current = coo_matrix(
        (values, (indices[:, 0], indices[:, 1])), shape=shape
    ).tocsr()
    current.sum_duplicates()
    # Keep explicit zeros.  CuPy's CSR zero-compression path has produced
    # incorrect results for signed float64 data on supported CUDA builds.
    current.sort_indices()
    if not np.all(np.isfinite(current.data)):
        raise SparsePowerComputationError("coalesced input Laplacian contains nonfinite values")
    base = current.copy()
    for exponent in range(2, power + 1):
        try:
            product = current @ base
            current = product.tocsr()
            current.sum_duplicates()
            current.sort_indices()
        except Exception as exc:
            raise SparsePowerComputationError(
                f"SciPy sparse product failed while forming power {exponent}"
            ) from exc
    cropped = current[:n_physical, :].tocoo()
    out_indices = np.column_stack((cropped.row, cropped.col)).astype(np.int32, copy=False)
    out_values = np.asarray(cropped.data, dtype=values.dtype)
    if not np.all(np.isfinite(out_values)):
        raise SparsePowerComputationError("sparse power contains nonfinite values")
    return SparseCOOMatrix(
        indices=jnp.asarray(out_indices, dtype=jnp.int32),
        values=jnp.asarray(out_values),
        shape=(n_physical, int(shape[1])),
    )


def _power_cupy(indices, values, shape, power, n_physical, block_size):
    try:
        import cupy as cp
        from cupyx.scipy.sparse import coo_matrix
    except (ImportError, ModuleNotFoundError) as exc:
        raise SparsePowerBackendUnavailable(
            "CuPy sparse is unavailable; request backend='cpu' explicitly if a CPU setup is intended"
        ) from exc

    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise SparsePowerBackendUnavailable("CuPy found no CUDA device")
    except SparsePowerBackendUnavailable:
        raise
    except Exception as exc:
        raise SparsePowerBackendUnavailable("CuPy could not initialize its CUDA backend") from exc

    current = coo_matrix(
        (
            cp.asarray(values),
            (cp.asarray(indices[:, 0]), cp.asarray(indices[:, 1])),
        ),
        shape=shape,
    ).tocsr()
    current.sum_duplicates()
    # Do not call cupyx eliminate_zeros here: its CSR compression kernel can
    # corrupt signed float64 values on some CUDA/CuPy combinations. Stored
    # exact zeros are valid CSR entries and do not change the operator.
    current.sort_indices()
    if not bool(cp.isfinite(current.data).all().item()):
        raise SparsePowerComputationError("coalesced input Laplacian contains nonfinite values")
    base = current
    n_full = int(shape[0])
    host_rows = []
    host_cols = []
    host_values = []
    try:
        for start in range(0, n_full, block_size):
            stop = min(start + block_size, n_full)
            row_block = base[start:stop, :].tocsr()
            for exponent in range(2, power + 1):
                try:
                    product = row_block @ base
                    row_block = product.tocsr()
                    row_block.sum_duplicates()
                    row_block.sort_indices()
                    if not bool(cp.isfinite(row_block.data).all().item()):
                        raise SparsePowerComputationError(
                            f"CuPy sparse power {exponent} contains nonfinite values"
                        )
                    del product
                except SparsePowerComputationError:
                    raise
                except Exception as exc:
                    raise SparsePowerComputationError(
                        f"CuPy sparse product failed while forming power {exponent} "
                        f"for rows [{start}, {stop}); full nnz={int(base.nnz)}, "
                        f"block nnz={int(row_block.nnz)}"
                    ) from exc
            block_coo = row_block.tocoo()
            # Each block has a disjoint set of global rows. Copying each final
            # block to host releases its GPU product arrays for reuse. No
            # multiplication or truncation occurs on the host.
            host_rows.append(cp.asnumpy(block_coo.row) + start)
            host_cols.append(cp.asnumpy(block_coo.col))
            host_values.append(cp.asnumpy(block_coo.data).astype(values.dtype, copy=False))
            del block_coo, row_block
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()

        # Concatenate every completed row block into a full-square COO power.
        # Only after this complete power exists are ghost equation rows cropped.
        full_rows = np.concatenate(host_rows) if host_rows else np.zeros((0,), dtype=np.int32)
        full_cols = np.concatenate(host_cols) if host_cols else np.zeros((0,), dtype=np.int32)
        full_values = np.concatenate(host_values) if host_values else np.zeros((0,), dtype=values.dtype)
        physical_mask = full_rows < n_physical
        out_rows = full_rows[physical_mask]
        out_cols = full_cols[physical_mask]
        out_values = full_values[physical_mask]
        out_indices = np.column_stack((out_rows, out_cols)).astype(np.int32, copy=False)
        # Boolean indexing above owns the selected physical values. Drop all
        # concatenated full-power host storage before JAX allocates its upload.
        del (
            host_rows, host_cols, host_values, full_rows, full_cols, full_values,
            physical_mask, out_rows, out_cols,
        )
    except SparsePowerComputationError:
        raise
    finally:
        del base, current
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
    if not np.all(np.isfinite(out_values)):
        raise SparsePowerComputationError("sparse power contains nonfinite values")
    return SparseCOOMatrix(
        indices=jnp.asarray(out_indices, dtype=jnp.int32),
        values=jnp.asarray(out_values),
        shape=(n_physical, int(shape[1])),
    )
