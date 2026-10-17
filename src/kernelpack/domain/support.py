"""Shared GPU radius topology with bounded pairwise scratch storage."""
import jax
import jax.numpy as jnp
from kernelpack.kernels import squared_distances, safe_sqrt


@jax.jit
def support_counts(z, radius):
    n = len(z)
    if not 0 < n < 2**31:
        raise ValueError("GPU CSR support requires 0 < n < 2**31")
    block = min(64, n)
    def step(_, start):
        ids = start + jnp.arange(block, dtype=jnp.int32)
        mask = (safe_sqrt(squared_distances(z[jnp.minimum(ids,n-1)],z,boundary_squared=radius**2)) < radius) & (ids[:,None] < n)
        return None, jnp.sum(mask, axis=1, dtype=jnp.int32)
    counts = jax.lax.scan(step,None,jnp.arange(0,n,block,dtype=jnp.int32))[1].ravel()[:n]
    prefix=jnp.cumsum(counts,dtype=jnp.int32)
    # Each row count is <2**31. The first signed-prefix overflow is
    # therefore negative, even if a later prefix wraps positive again.
    prefix=prefix.at[-1].set(jnp.where(jnp.all(prefix>=0),prefix[-1],-1))
    return jnp.concatenate([jnp.zeros(1,dtype=jnp.int32),prefix])


@jax.jit
def compact_support(z, radius, indptr, rows, cols):
    n = len(z)
    block = min(64,n)
    capacity = len(cols)
    def step(carry,start):
        ri = start+jnp.arange(block,dtype=jnp.int32)
        mask = (safe_sqrt(squared_distances(z[jnp.minimum(ri,n-1)],z,boundary_squared=radius**2)) < radius) & (ri[:,None] < n)
        slots = indptr[start]+jnp.cumsum(mask.ravel(),dtype=jnp.int32)-1
        slots = jnp.where(mask.ravel(),slots,capacity)
        out_rows,out_cols = carry
        out_rows = out_rows.at[slots].set(jnp.repeat(ri,n),mode="drop")
        out_cols = out_cols.at[slots].set(jnp.tile(jnp.arange(n,dtype=jnp.int32),block),mode="drop")
        return (out_rows,out_cols),None
    return jax.lax.scan(step,(rows,cols),jnp.arange(0,n,block,dtype=jnp.int32))[0]
