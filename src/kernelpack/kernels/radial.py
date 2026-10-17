"""Device-only radial kernels with stable distances and coincidence limits."""
import math
import jax
import jax.numpy as jnp
from jax.custom_batching import custom_vmap
from .matern import matern_squared


def squared_norm(delta):
    # Parallel reduction, rather than one device loop iteration per coordinate.
    # Cartesian products below use this identical direct-difference reduction.
    return jnp.sum(delta*delta,axis=-1)


def _direct_squared_distances(x, y):
    # Tile both point axes to bound feature scratch to 32 MiB (one pair
    # minimum). Row blocking alone is insufficient for high-dimensional data.
    # The reduction is parallel across features, including 7000-coordinate
    # function inputs; norm-dot cancellation is avoided at coincident inputs.
    if x.shape[-1]!=y.shape[-1]:
        raise ValueError("Distance features must match")
    if not len(x) or not len(y):
        return jnp.zeros((len(x),len(y)),dtype=x.dtype)
    pairs=max(1,(32*1024**2)//max(1,2*x.shape[1]*x.dtype.itemsize))
    width=min(len(x),64,max(1,math.isqrt(pairs)),
        max(1,(16*1024**2)//(len(y)*x.dtype.itemsize)))
    columns=min(len(y),max(1,pairs//width))
    def step(_,start):
        block=x[jnp.minimum(start+jnp.arange(width),len(x)-1)]
        def inner(_,column):
            part=y[jnp.minimum(column+jnp.arange(columns),len(y)-1)]
            return None,squared_norm(block[:,None,:]-part[None,:,:])
        distances=jax.lax.scan(inner,None,jnp.arange(0,len(y),columns))[1]
        return None,jnp.transpose(distances,(1,0,2)).reshape(width,-1)[:,:len(y)]
    return jax.lax.scan(step,None,jnp.arange(0,len(x),width))[1].reshape(-1,len(y))[:len(x)]


# Fixed GP geometry caches can opt into a direct reduction independently of
# the faster screened query path. Both variants stay on the caller's device.
direct_squared_distances=jax.jit(_direct_squared_distances)


@jax.jit
def _squared_distances_impl(x, y, boundary_squared=None):
    """Euclidean squared distances with bounded GPU scratch.

    Spatial coordinates use direct differences. Wide float64 function inputs
    use GEMM, with conservative cancellation screening and direct repairs.
    ``boundary_squared`` additionally repairs uncertain support comparisons.
    The result, a second result-sized self-symmetrization buffer and vendor GEMM
    workspace are outside the 64 MiB scratch estimate. No repair count or
    numerical array is transferred to the host. Feature rows are gathered in
    bounded blocks; the complete feature inputs are never padded or copied.
    """
    if x.shape[-1]!=y.shape[-1]:
        raise ValueError("Distance features must match")
    if (not len(x) or not len(y) or x.shape[-1]<256
            or x.dtype!=jnp.float64 or y.dtype!=jnp.float64):
        return _direct_squared_distances(x,y)
    # Ledger: 16 MiB screening/compaction, 16 MiB feature rows, 28 MiB direct
    # repair gathers and 4 MiB padding of the required distance output.
    dense_bytes=16*1024**2
    if len(y)*(4*x.dtype.itemsize+12)>dense_bytes:
        return _direct_squared_distances(x,y)
    width=min(len(x),128,max(1,dense_bytes//(len(y)*(4*x.dtype.itemsize+12))),
        max(1,(16*1024**2)//(x.shape[1]*x.dtype.itemsize)))
    pair_chunk=max(1,(28*1024**2)//(4*x.shape[1]*x.dtype.itemsize))
    capacity=min(width*len(y),4096)
    pair_chunk=min(pair_chunk,capacity)
    ynorm=squared_norm(y)
    eps=jnp.finfo(x.dtype).eps
    accumulation=(x.shape[1]+4)*eps
    gamma=accumulation/(1-accumulation)
    # Includes norm/dot accumulation and the direct reference's rounding.
    # Acceptance certifies relative squared-distance error <= 1e-9 under
    # ordinary float64 arithmetic; coincidence is always repaired directly.
    relative_target=1e-9
    entries=width*len(y)
    def step(_,start):
        block=x[jnp.minimum(start+jnp.arange(width),len(x)-1)]
        norm_sum=squared_norm(block)[:,None]+ynorm[None,:]
        value=norm_sum-2*jnp.matmul(block,y.T,precision=jax.lax.Precision.HIGHEST)
        bound=(32*gamma)*norm_sum
        ambiguous=((~jnp.isfinite(value))|(~jnp.isfinite(bound))
            |(norm_sum<jnp.finfo(x.dtype).tiny/gamma)
            |(value<=bound*(1+relative_target)/relative_target))
        if boundary_squared is not None:
            boundary=jnp.asarray(boundary_squared,dtype=x.dtype)
            boundary_ambiguous=jnp.abs(value-boundary)<=bound+8*eps*jnp.abs(boundary)
            ambiguous=ambiguous|boundary_ambiguous
        count=jnp.sum(ambiguous,dtype=jnp.int32)
        def repair(_):
            ids=jnp.nonzero(ambiguous.ravel(),size=capacity,fill_value=entries)[0]
            ids=jnp.pad(ids,(0,pair_chunk),constant_values=entries)
            def body(state):
                start,result=state
                selected=jax.lax.dynamic_slice(ids,(start,),(pair_chunk,))
                safe=jnp.minimum(selected,entries-1)
                direct=squared_norm(block[safe//len(y)]-y[safe%len(y)])
                selected=jnp.where(jnp.arange(pair_chunk)+start<count,selected,entries)
                result=result.at[selected].set(direct,mode="drop")
                return start+pair_chunk,result
            _,flat=jax.lax.while_loop(lambda state:state[0]<count,body,(jnp.int32(0),value.ravel()))
            return flat.reshape(width,len(y))
        use_pair_repair=(count<=capacity)&(accumulation<.25)
        if boundary_squared is not None:
            # A paired reduction can round one ULP differently from the
            # matrix reduction. Preserve strict support decisions by using
            # the complete bounded direct block at an uncertain boundary.
            use_pair_repair=use_pair_repair&~jnp.any(boundary_ambiguous)
        result=jax.lax.cond(use_pair_repair,repair,
            lambda _:_direct_squared_distances(block,y),operand=None)
        return None,result
    result=jax.lax.scan(step,None,jnp.arange(0,len(x),width))[1].reshape(-1,len(y))[:len(x)]
    # Self-distance calls can cross a JIT boundary as two distinct tracers.
    # Keep the full geometry matrix exactly symmetric even in that case.
    if x.shape==y.shape:
        result=jax.lax.cond(jnp.all(x==y),
            lambda a:((a+a.T)*.5).at[jnp.diag_indices(len(x))].set(0.),lambda a:a,result)
    return result


_batched_squared_distances=custom_vmap(_squared_distances_impl)


@_batched_squared_distances.def_vmap
def _distance_batch(axis_size,in_batched,x,y,boundary_squared):
    bx,by,bb=in_batched
    if bx and not by and not bb:
        shape=x.shape
        result=_batched_squared_distances(x.reshape(-1,shape[-1]),y,boundary_squared)
        return result.reshape(shape[:-1]+(len(y),)),True
    if by and not bx and not bb:
        shape=y.shape
        result=_batched_squared_distances(x,y.reshape(-1,shape[-1]),boundary_squared)
        return result.reshape((len(x),)+shape[:-1]).transpose(1,0,2),True
    if not bx and not by and not bb:
        return _batched_squared_distances(x,y,boundary_squared),False
    # Independent input sets or support thresholds cannot share cross pairs.
    # Device-map them so vmap does not turn the repair conditional into a
    # select that executes the expensive direct fallback for every batch.
    def one(index):
        values=tuple(value[index] if batched else value
            for value,batched in zip((x,y,boundary_squared),in_batched))
        return _batched_squared_distances(*values)
    return jax.lax.map(one,jnp.arange(axis_size)),True


_compiled_squared_distances=jax.jit(_batched_squared_distances)


@jax.jit
def _squared_distance_tangent(x,y,dx,dy):
    # Use the exact Euclidean derivative, independently of cancellation masks
    # and dynamic repair queues. Static bounded scans retain reverse-mode AD.
    if not len(x) or not len(y):
        return jnp.zeros((len(x),len(y)),dtype=x.dtype)
    pairs=max(1,(32*1024**2)//max(1,4*x.shape[1]*x.dtype.itemsize))
    width=min(len(x),128,max(1,math.isqrt(pairs)),
        max(1,(16*1024**2)//(len(y)*x.dtype.itemsize)))
    columns=min(len(y),max(1,pairs//width))
    @jax.checkpoint
    def step(_,start):
        ids=jnp.minimum(start+jnp.arange(width),len(x)-1)
        block,dblock=x[ids],dx[ids]
        @jax.checkpoint
        def inner(_,column):
            ids=jnp.minimum(column+jnp.arange(columns),len(y)-1)
            part,dpart=y[ids],dy[ids]
            delta=block[:,None,:]-part[None,:,:]
            ddelta=dblock[:,None,:]-dpart[None,:,:]
            return None,2*jnp.sum(delta*ddelta,axis=-1)
        values=jax.lax.scan(inner,None,jnp.arange(0,len(y),columns))[1]
        return None,jnp.transpose(values,(1,0,2)).reshape(width,-1)[:,:len(y)]
    return jax.lax.scan(step,None,jnp.arange(0,len(x),width))[1].reshape(-1,len(y))[:len(x)]


@jax.custom_jvp
def squared_distances(x,y,boundary_squared=None):
    """Bounded GPU Euclidean distances; see the implementation for screening."""
    return _compiled_squared_distances(x,y,boundary_squared)


@squared_distances.defjvp
def _squared_distances_jvp(primals,tangents):
    x,y,boundary=primals
    dx,dy,_=tangents
    return squared_distances(x,y,boundary),_squared_distance_tangent(x,y,dx,dy)


@jax.custom_jvp
def safe_sqrt(s):
    return jnp.sqrt(jnp.maximum(s, 0.))


@safe_sqrt.defjvp
def _sqrt_jvp(primals, tangents):
    s, = primals
    ds, = tangents
    value = safe_sqrt(s)
    denominator = jnp.where(s > 0, 2 * value, 1.)
    return value, jnp.where(s > 0, ds / denominator, 0.)


@jax.custom_jvp
def _wendland_slope(s):
    r = safe_sqrt(jnp.clip(s, 0., 1.))
    return jnp.where(s < 1., -28./3 * (1-r)**5 * (1+5*r), 0.)


@_wendland_slope.defjvp
def _wendland_slope_jvp(primals, tangents):
    s, = primals
    ds, = tangents
    r = safe_sqrt(jnp.clip(s, 0., 1.))
    return _wendland_slope(s), jnp.where(s < 1., 140 * (1-r)**4, 0.) * ds


@jax.custom_jvp
def wendland_c4_squared(s):
    """Normalized C4 Wendland, argument = squared scaled distance."""
    r = safe_sqrt(jnp.clip(s, 0., 1.))
    return jnp.where(s < 1., (1-r)**6 * (1+6*r+35*r*r/3), 0.)


@wendland_c4_squared.defjvp
def _wendland_jvp(primals, tangents):
    s, = primals
    ds, = tangents
    return wendland_c4_squared(s), _wendland_slope(s) * ds


def matern52_squared(s):
    """Reuse the shared collocation Matérn derivative machinery."""
    return matern_squared(s, p=2, shape=1.)
