"""Bounded GPU sampled dense products on an exact CSR support.

The result contains only CSR entries of ``left @ right.T``. A dense row tile
is temporary workspace, never a retained full Gram matrix. Numerical arrays
remain explicit dynamic arguments throughout JIT, vmap and differentiation.
"""
from functools import partial
from dataclasses import dataclass
from numbers import Integral

import jax
import jax.numpy as jnp


def _metadata(left,right,indptr,rows,columns,scratch_bytes,algorithm):
    arrays=(left,right,indptr,rows,columns)
    device=None
    for value in arrays:
        if isinstance(value,jax.core.Tracer):
            continue
        if not isinstance(value,jax.Array) or value.device.platform!="gpu":
            raise ValueError("CSR edge inner products require GPU arrays")
        if device is not None and value.device!=device:
            raise ValueError("CSR edge operands must share one GPU device")
        device=value.device
    if (left.ndim!=2 or right.ndim!=2 or not len(left) or not len(right)
            or not left.shape[1] or left.shape[1]!=right.shape[1]
            or left.dtype not in (jnp.float32,jnp.float64) or right.dtype!=left.dtype):
        raise ValueError("Dense operands must be matching nonempty floating feature matrices")
    if (indptr.shape!=(len(left)+1,) or rows.ndim!=1 or columns.shape!=rows.shape
            or not len(rows) or any(value.dtype not in (jnp.int32,jnp.int64)
                                    for value in (indptr,rows,columns))):
        raise ValueError("CSR pointers and equally sized row/column arrays must be integers")
    if max(len(left),len(right),len(rows))>=2**31:
        raise ValueError("CSR edge products require int32-compatible sizes")
    if not isinstance(scratch_bytes,Integral) or scratch_bytes<1:
        raise ValueError("scratch_bytes must be a positive integer")
    if algorithm not in ("auto","gather","tiled_gemm"):
        raise ValueError("algorithm must be auto, gather, or tiled_gemm")
    if algorithm=="auto":
        algorithm=("gather" if left.shape[1]<64 or 4*len(rows)<len(left)*len(right)
                   else "tiled_gemm")
    scalar=left.dtype.itemsize
    index=max(value.dtype.itemsize for value in (indptr,rows,columns))
    # The gather path accounts for two gathered feature rows, their product,
    # gathered values, masks, and eight live integer/index vectors per edge.
    gather_per_edge=max((3*left.shape[1]+2)*scalar+8*index+8,16*index+32)
    chunk=min(len(rows),scratch_bytes//gather_per_edge)
    if chunk<1:
        raise ValueError("Scratch budget cannot hold one gathered edge")
    # GEMM workspace ledger per row: N columns times (tile value + sampled
    # value + ten integer/index vectors + masks), plus one L feature row.
    # Required output and vendor GEMM workspace are accounted separately.
    tile_per_row=len(right)*(2*scalar+10*index+8)+left.shape[1]*scalar
    tile_rows=min(len(left),128,(2**31-1)//len(right),scratch_bytes//tile_per_row)
    if algorithm=="tiled_gemm" and tile_rows<1:
        raise ValueError("Scratch budget cannot hold one dense product row tile")
    return algorithm,int(chunk),max(1,int(tile_rows))


@partial(jax.jit,static_argnames=("nrows","ncols","chunk"))
def _csr_structure_valid(indptr,rows,columns,*,nrows,ncols,chunk):
    capacity=len(rows)
    actual=indptr[-1]
    valid=((indptr[0]==0)&(actual>0)&(actual<=capacity)
        &jnp.all(jnp.diff(indptr)>=0)&jnp.all((indptr>=0)&(indptr<=capacity)))
    def step(valid,start):
        offset=jnp.arange(chunk,dtype=jnp.int32)
        remaining=capacity-start
        inside=offset<remaining
        position=start+jnp.minimum(offset,remaining)
        safe=jnp.minimum(position,capacity-1)
        row,column=rows[safe],columns[safe]
        bounded=(row>=0)&(row<nrows)&(column>=0)&(column<ncols)
        safe_row=jnp.clip(row,0,nrows-1)
        prefix=position<actual
        owns=(position>=indptr[safe_row])&(position<indptr[safe_row+1])
        previous=jnp.maximum(safe-1,0)
        increasing=(position==0)|(rows[previous]!=row)|(column>columns[previous])
        admissible=bounded&jnp.where(prefix,owns&increasing,row==nrows-1)
        return valid&jnp.all(jnp.where(inside,admissible,True)),None
    return jax.lax.scan(step,valid,jnp.arange(0,capacity,chunk,dtype=jnp.int32))[0]


def _gather_products(left,right,indptr,rows,columns,*,chunk):
    capacity=len(rows)
    actual=indptr[-1]
    output=jnp.zeros(capacity,dtype=left.dtype)
    def step(output,start):
        offset=jnp.arange(chunk,dtype=jnp.int32)
        remaining=capacity-start
        inside=offset<remaining
        position=start+jnp.minimum(offset,remaining)
        safe=jnp.minimum(position,capacity-1)
        row=jnp.clip(rows[safe],0,len(left)-1)
        column=jnp.clip(columns[safe],0,len(right)-1)
        value=jnp.sum(left[row]*right[column],axis=1)
        destination=jnp.where(inside&(position<actual),position,capacity)
        return output.at[destination].set(value,mode="drop"),None
    return jax.lax.scan(step,output,jnp.arange(0,capacity,chunk,dtype=jnp.int32))[0]


def _tiled_products(left,right,indptr,rows,columns,*,tile_rows):
    nrows,ncols=len(left),len(right)
    capacity=len(rows)
    actual=indptr[-1]
    offsets=jnp.arange(tile_rows*ncols,dtype=jnp.int32)
    output=jnp.zeros(capacity,dtype=left.dtype)
    def step(output,start):
        row_offsets=jnp.arange(tile_rows,dtype=jnp.int32)
        block=left[start+jnp.minimum(row_offsets,nrows-start-1)]
        product=jnp.matmul(block,right.T,precision=jax.lax.Precision.HIGHEST)
        first=jnp.clip(indptr[start],0,capacity)
        end=start+jnp.minimum(tile_rows,nrows-start)
        last=jnp.clip(indptr[end],first,capacity)
        position=first+jnp.minimum(offsets,capacity-first)
        safe=jnp.clip(position,0,capacity-1)
        local_row=jnp.clip(rows[safe]-start,0,tile_rows-1)
        column=jnp.clip(columns[safe],0,ncols-1)
        value=product[local_row,column]
        valid=(offsets<last-first)&(position<actual)&(position<capacity)
        destination=jnp.where(valid,position,capacity)
        return output.at[destination].set(value,mode="drop"),None
    return jax.lax.scan(step,output,jnp.arange(0,nrows,tile_rows,dtype=jnp.int32))[0]


@partial(jax.jit,static_argnames=("algorithm","chunk","tile_rows","validate"),backend="gpu")
def _edge_products_impl(left,right,indptr,rows,columns,*,algorithm,chunk,tile_rows,validate):
    if algorithm=="tiled_gemm":
        result=_tiled_products(left,right,indptr,rows,columns,tile_rows=tile_rows)
    else:
        result=_gather_products(left,right,indptr,rows,columns,chunk=chunk)
    if validate:
        valid=_csr_structure_valid(indptr,rows,columns,nrows=len(left),ncols=len(right),chunk=chunk)
        result=jnp.where(valid,result,jnp.nan)
    return result


def csr_edge_inner_products(left,right,indptr,rows,columns,*,scratch_bytes=32*1024**2,
                            algorithm="auto",validate=True):
    """Return CSR-supported entries of ``left @ right.T`` entirely on GPU.

    ``left`` and ``right`` are (rows,features) matrices of the same floating
    dtype. CSR entries must be unique and strictly column-sorted within each
    row; ``indptr`` determines the active prefix. Capacity padding has valid
    column indices and row index ``len(left)-1``, matching cuDSS's padded CSR
    contract. The returned padded tail is exactly zero.

    ``auto`` uses bounded edge gathers for fewer than 64 features or capacity
    density below 0.25; otherwise it uses row-tiled GEMM and exact CSR gathers.
    Explicit ``gather`` and ``tiled_gemm`` select numerical comparison paths.
    They perform the same mathematical inner products with different floating
    summation orders. No dense full Gram matrix, covariance or inverse is kept.

    Metadata errors raise. With ``validate=True`` numerical CSR contract
    violations return NaNs on GPU, including under JIT. Callers may disable
    this duplicate check only after validating that exact support elsewhere
    (for example in CudssFactor). No numerical flag/count is read on the host.
    The scratch budget includes product tiles, feature gathers and conservative
    index/mask buffers. Required output storage and vendor GEMM workspace are
    separate. All inputs must be resident on one GPU; matrix/RHS derivatives,
    JIT and mapped dense operands use ordinary JAX rules.
    """
    if not isinstance(validate,bool):
        raise ValueError("validate must be a static Python boolean")
    algorithm,chunk,tile_rows=_metadata(left,right,indptr,rows,columns,scratch_bytes,algorithm)
    return _edge_products_impl(left,right,indptr,rows,columns,algorithm=algorithm,
        chunk=chunk,tile_rows=tile_rows,validate=validate)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class PaddedCSRAction:
    """Logical square action backed by valid, unique rectangular CSR entries.

    ``data``, ``indices`` and ``indptr`` describe an extended CSR matrix with
    ``logical_size`` columns. Its first logical_size rows are the original
    matrix. Every capacity-tail entry is zero and occupies a distinct valid
    column in a dummy row. No native factor or original CSR is referenced.

    ``native_matmul`` uses the stock CSR cuSPARSE primitive on GPU; callers
    pack mapped RHS axes before that primitive, as CudssFactor already does.
    ``@`` uses BCSR's existing batchable differentiation rules. In the installed
    JAX configuration those derivative actions use generic GPU operations,
    without changing a global sparse-lowering flag.
    """

    data: object
    indices: object
    indptr: object
    logical_size: int
    transposed: bool = False

    def tree_flatten(self):
        return (self.data, self.indices, self.indptr), (self.logical_size, self.transposed)

    @classmethod
    def tree_unflatten(cls, auxiliary, children):
        logical_size, transposed = auxiliary
        return cls(*children, logical_size, transposed=transposed)

    @property
    def shape(self):
        return (self.logical_size, self.logical_size)

    @property
    def extended_shape(self):
        return (len(self.indptr)-1, self.logical_size)

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        return type(self)(self.data, self.indices, self.indptr, self.logical_size,
                          transposed=not self.transposed)

    def with_data(self, data):
        """Reuse prepared support coordinates after an exact-pattern refactor."""
        if data.shape != self.data.shape or data.dtype != self.data.dtype:
            raise ValueError("Prepared CSR action values must preserve size and dtype")
        return type(self)(data, self.indices, self.indptr, self.logical_size,
                          transposed=self.transposed)

    def _rhs(self, rhs):
        rhs = jnp.asarray(rhs)
        if rhs.ndim not in (1, 2) or rhs.shape[0] != self.logical_size or rhs.dtype != self.dtype:
            raise ValueError("Prepared CSR action requires a matching vector or matrix RHS")
        if self.transposed:
            padding = [(0, self.extended_shape[0]-self.logical_size)] + [(0, 0)]*(rhs.ndim-1)
            rhs = jnp.pad(rhs, padding)
        return rhs

    def native_matmul(self, rhs):
        """Apply the original matrix using native CSR, padding/cropping only RHS rows."""
        from jax.experimental.sparse import CSR, csr_matmat, csr_matvec
        rhs = self._rhs(rhs)
        matrix = CSR((self.data, self.indices, self.indptr), shape=self.extended_shape)
        result = (csr_matvec(matrix, rhs, transpose=self.transposed) if rhs.ndim == 1
                  else csr_matmat(matrix, rhs, transpose=self.transposed))
        return result if self.transposed else result[:self.logical_size]

    def __matmul__(self, rhs):
        """Batchable linear action for the existing BCSR JVP/transpose rules."""
        from jax.experimental.sparse import BCSR, bcsr_dot_general
        rhs = self._rhs(rhs)
        matrix = BCSR((self.data, self.indices, self.indptr), shape=self.extended_shape,
                      indices_sorted=True, unique_indices=True)
        contraction = 0 if self.transposed else 1
        result = bcsr_dot_general(matrix, rhs,
            dimension_numbers=(((contraction,), (0,)), ((), ())))
        return result if self.transposed else result[:self.logical_size]


@partial(jax.jit, static_argnames=("logical_size", "chunk"), backend="gpu")
def _prepare_padded_csr_indices(columns, indptr, *, logical_size, chunk):
    """Remap only capacity padding, with bounded temporary integer vectors."""
    capacity = len(columns)
    actual = indptr[-1]
    mapped = jnp.empty_like(columns)
    local = jnp.arange(chunk, dtype=jnp.int32)

    def block(mapped, start):
        remaining = capacity-start
        inside = local < remaining
        position = start+jnp.minimum(local, remaining-1)
        column = jnp.where(position < actual, columns[position],
                           (position-actual) % logical_size)
        destination = jnp.where(inside, position, capacity)
        return mapped.at[destination].set(column, mode="drop"), None

    starts = jnp.arange((capacity+chunk-1)//chunk, dtype=jnp.int32)*chunk
    mapped = jax.lax.scan(block, mapped, starts)[0]
    extra_rows = (capacity+logical_size-1)//logical_size
    # Zero-based dummy-row starts are strictly below capacity. Every sum
    # below remains <=capacity, including for capacities near int32 max.
    row_start = jnp.arange(extra_rows, dtype=jnp.int32)*logical_size
    remaining = capacity-actual
    offset = jnp.minimum(row_start, remaining)
    endpoint = actual+offset+jnp.minimum(logical_size, remaining-offset)
    extended_indptr = jnp.concatenate((indptr, endpoint))
    return mapped, extended_indptr


def _padded_csr_action_from_validated(data, columns, indptr, *, scratch_bytes=32*1024**2):
    """Prepare an already-validated exact CSR prefix and zero capacity tail.

    CudssFactor performs its unchanged complete GPU CSR/value validation
    before calling this helper. Required mapped columns occupy 4*capacity
    bytes; extended row pointers are small. The scratch limit covers a
    conservative 64 bytes per live chunk entry, excluding required output.
    Neither a capacity-length iota nor mapped row-index buffer is formed.
    """
    if not isinstance(scratch_bytes, Integral) or scratch_bytes < 64:
        raise ValueError("padded_action_scratch_bytes must hold one 64-byte index chunk")
    logical_size, capacity = len(indptr)-1, len(columns)
    if (logical_size < 1 or capacity < 1 or capacity >= 2**31
            or logical_size+(capacity+logical_size-1)//logical_size >= 2**31):
        raise ValueError("Extended CSR action dimensions must fit int32")
    chunk = min(capacity, int(scratch_bytes)//64)
    mapped, extended = _prepare_padded_csr_indices(columns, indptr,
        logical_size=logical_size, chunk=chunk)
    return PaddedCSRAction(data, mapped, extended, logical_size)
