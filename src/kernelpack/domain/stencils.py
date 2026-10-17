"""Shared JAX stencil selection, including rectangular periodic domains.

Indices always refer to original nodes. Coordinates are nearest periodic
images about each query; consumers must use those coordinates for local fits.
"""
from functools import partial
import jax
import jax.numpy as jnp
from kernelpack.kernels import squared_distances, squared_norm, safe_sqrt


@partial(jax.jit,static_argnames=("k",))
def exact_knn(query_points,points,point_active,*,k):
    """Existing shared top-k search with direct, bounded-storage distances."""
    distances=jnp.where(point_active[None,:],squared_distances(query_points,points),jnp.inf)
    negative,indices=jax.lax.top_k(-distances,k)
    return indices,safe_sqrt(jnp.maximum(-negative,0.))


@jax.jit
def nearest_periodic_images(points,queries,lengths,periodic):
    """points: (...,nodes,dim), queries: (...,dim); ties away from zero."""
    delta=points-queries[...,None,:]
    length=jnp.where(periodic,lengths,1.)
    ratio=delta/length
    shift=jnp.copysign(jnp.floor(jnp.abs(ratio)+.5),ratio)
    delta=delta-jnp.where(periodic,length*shift,0.)
    return queries[...,None,:]+delta


@partial(jax.jit,static_argnames=("k",))
def _select_stencil_batch(points,queries,*,k,periodic=None,lengths=None,point_active=None,
                    center_indices=None,use_periodic=None):
    """Return (original indices, local image coordinates, valid-row mask).

    Stable index ordering breaks equal-distance ties. Periodic selection
    promotes the explicitly assigned original node and greedily removes coincident
    image locations using the existing FastOL tolerance. No host filtering.
    Inactive capacity rows are excluded. Insufficient distinct nodes set
    valid=False; callers must reject those rows before assembling systems.
    center_indices are zero-based; -1 denotes a non-nodal query. If omitted,
    exact unique nodal matches are inferred; ambiguous duplicate matches fail
    in periodic mode. use_periodic is independent of the coordinate mask:
    enabled mode still deduplicates/promotes when the entire mask is false.
    """
    if points.ndim!=2 or queries.ndim!=2 or points.shape[1]!=queries.shape[1]:
        raise ValueError("points and queries must be matching spatial matrices")
    if not 0<k<=len(points):
        raise ValueError("stencil size must be between 1 and the node count")
    active=jnp.ones(len(points),dtype=bool) if point_active is None else jnp.asarray(point_active,dtype=bool)
    if active.shape!=(len(points),):
        raise ValueError("active mask must have one entry per node")
    if center_indices is not None:
        centers=jnp.asarray(center_indices)
        if centers.shape!=(len(queries),) or not jnp.issubdtype(centers.dtype,jnp.integer):
            raise ValueError("center_indices must be an integer vector, one per query")
        safe_centers=jnp.clip(centers,0,len(points)-1)
        assigned_valid=(centers==-1)|((centers>=0)&(centers<len(points))
            &active[safe_centers]&jnp.all(points[safe_centers]==queries,axis=1))
    else:
        centers=jnp.full(len(queries),-1,dtype=jnp.int32)
        assigned_valid=jnp.ones(len(queries),dtype=bool)
    def ordinary(_):
        ids,dist=exact_knn(queries,points,active,k=k)
        return ids,points[ids],jnp.all(jnp.isfinite(dist),axis=1)&assigned_valid
    if use_periodic is None:
        use_periodic=periodic is not None
    if periodic is not None and lengths is None:
        raise ValueError("periodic stencil selection requires domain lengths")
    periodic=jnp.zeros(points.shape[1],dtype=bool) if periodic is None else jnp.asarray(periodic,dtype=bool)
    lengths=jnp.ones(points.shape[1],dtype=points.dtype) if lengths is None else jnp.asarray(lengths,dtype=points.dtype)
    if periodic.shape!=(points.shape[1],) or lengths.shape!=(points.shape[1],):
        raise ValueError("periodic mask and lengths must match spatial dimension")
    def periodic_selection(_):
        def one(query,assigned,assignment_valid):
            images=nearest_periodic_images(points,query,lengths,periodic)
            delta=images-query
            scale=jnp.maximum(1.,jnp.maximum(jnp.max(jnp.where(active[:,None],jnp.abs(points),0.)),
                jnp.max(jnp.where(active[:,None],jnp.abs(images),0.))))
            tol=32*(jnp.nextafter(scale,jnp.asarray(jnp.inf,dtype=points.dtype))-scale)
            if center_indices is None:
                own=jnp.all(points==query,axis=1)&active
                matches=jnp.sum(own)
                own_id=jnp.argmax(own)
                promote=matches==1
                assignment_valid=matches<=1
            else:
                own_id=assigned
                promote=assigned>=0
            scores=jnp.where(active,safe_sqrt(squared_norm(delta)),jnp.inf)
            scores=jnp.where(promote&(jnp.arange(len(points))==own_id),-jnp.inf,scores)
            order=jnp.argsort(scores,stable=True).astype(jnp.int32)
            initial=(jnp.int32(0),jnp.int32(0),jnp.zeros(k,dtype=jnp.int32),
                     jnp.zeros((k,points.shape[1]),dtype=points.dtype))
            def condition(state):
                position,count,_,_=state
                return (position<len(points))&(count<k)
            def step(state):
                position,count,ids,chosen=state
                idx=order[position]
                duplicate=jnp.any((jnp.arange(k)<count)&(safe_sqrt(squared_norm(chosen-images[idx]))<=tol))
                accept=active[idx]&~duplicate
                ids=ids.at[count].set(jnp.where(accept,idx,ids[count]))
                chosen=chosen.at[count].set(jnp.where(accept,images[idx],chosen[count]))
                return position+1,count+accept.astype(jnp.int32),ids,chosen
            _,count,ids,chosen=jax.lax.while_loop(condition,step,initial)
            return ids,chosen,assignment_valid&(count==k)&jnp.all(jnp.isfinite(chosen))
        return jax.vmap(one)(queries,centers,assigned_valid)
    return jax.lax.cond(jnp.asarray(use_periodic,dtype=bool),periodic_selection,ordinary,None)


@partial(jax.jit,static_argnames=("k","scratch_bytes"))
def select_stencils(points,queries,*,k,periodic=None,lengths=None,point_active=None,
                    center_indices=None,use_periodic=None,scratch_bytes=64*1024**2):
    """Exact selection in fixed GPU batches, without changing candidate order.

    The scratch allowance sizes query batches conservatively for image arrays,
    distances, sort indices and duplicate filtering. It excludes returned arrays
    and vendor sort workspace; at least one query must fit. See the batch helper
    for center identity and periodic-mode semantics.
    """
    if scratch_bytes < 1:
        raise ValueError("scratch_bytes must be positive")
    if points.ndim!=2 or queries.ndim!=2 or points.shape[1]!=queries.shape[1]:
        raise ValueError("points and queries must be matching spatial matrices")
    if not 0<k<=len(points):
        raise ValueError("stencil size must be between 1 and the node count")
    center_indices=None if center_indices is None else jnp.asarray(center_indices)
    if center_indices is not None and (center_indices.shape!=(len(queries),)
            or not jnp.issubdtype(center_indices.dtype,jnp.integer)):
        raise ValueError("center_indices must be an integer vector, one per query")
    if not len(queries):
        return (jnp.zeros((0,k),dtype=jnp.int32),
            jnp.zeros((0,k,points.shape[1]),dtype=points.dtype),jnp.zeros(0,dtype=bool))
    per_query=max(1,len(points)*((6*points.shape[1]+16)*points.dtype.itemsize+16)
        +k*(points.shape[1]*points.dtype.itemsize+8))
    batch=min(len(queries),max(1,scratch_bytes//per_query))
    kwargs=dict(k=k,periodic=periodic,lengths=lengths,point_active=point_active,use_periodic=use_periodic)
    if batch==len(queries):
        return _select_stencil_batch(points,queries,center_indices=center_indices,**kwargs)
    padding=(-len(queries))%batch
    padded=jnp.pad(queries,((0,padding),(0,0)),mode="edge").reshape(-1,batch,points.shape[1])
    centers=(jnp.full((len(padded),batch),-1,dtype=jnp.int32) if center_indices is None
        else jnp.pad(center_indices,(0,padding),mode="edge").reshape(-1,batch))
    def step(_,data):
        q,ids=data
        return None,_select_stencil_batch(points,q,
            center_indices=None if center_indices is None else ids,**kwargs)
    _,(ids,images,valid)=jax.lax.scan(step,None,(padded,centers))
    return (ids.reshape(-1,k)[:len(queries)],
        images.reshape(-1,k,points.shape[1])[:len(queries)],valid.ravel()[:len(queries)])
