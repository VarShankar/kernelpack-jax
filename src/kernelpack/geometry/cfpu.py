"""Ambient CFPU supports inherited from parametric geometry; no inverse maps.

Positive values are inside, matching kernelpack's existing level-set contract.
Local potentials use curl-free cubic PHS with constant vector polynomials and
the interpolatory r-plus-constant residual correction of Drake/Fuselier/Wright.
Away from the cover, two geometry quadratures check the boundary-flux sign.
"""
from dataclasses import dataclass, field
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, lax


@dataclass(frozen=True)
class CFPUOptions:
    support_inflation: float = 1.25
    max_support_growth: float = 2.
    max_cover_refinements: int = 3
    min_sites: int = 8
    max_sites: int = 512
    quadrature_order: int = 24
    winding_tolerance: float = .10
    interpolation_tolerance: float = 1e-8
    feature_grade_factor: float = 1.
    feature_bandwidth: float = 2.
    # 'band' (default): outside the cover, use the sign of the connected component
    # of the uncovered region, inherited from f_loc on the adjacent covered cells;
    # if a component borders both signs or no covered cell, fall back to 'winding'.
    # 'winding': outside the cover, blend to a boundary-flux winding number.
    farfield: str = 'band'


class CFPUCoverError(ValueError):
    def __init__(self,message,parent_patches):
        super().__init__(message); self.parent_patches=parent_patches


@jax.custom_jvp
def _norm(x):
    return jnp.sqrt(jnp.sum(x*x,axis=-1))


@_norm.defjvp
def _norm_jvp(primals,tangents):
    x,=primals; dx,=tangents
    r=_norm(x)
    return r,jnp.where(r>0,jnp.sum(x*dx,axis=-1)/jnp.where(r>0,r,1.),0.)


@jit
def _local_solve(sites,normals,mask):
    """Padded batched saddle solves; passive rows are algebraically isolated."""
    count,capacity,dim=sites.shape
    delta=sites[:,:,None,:]-sites[:,None,:,:]
    radius=_norm(delta); safe=jnp.where(radius>0,radius,1.)
    hessian=3*(radius[:,:,:,None,None]*jnp.eye(dim)+
        delta[:,:,:, :,None]*delta[:,:,:,None,:]/safe[:,:,:,None,None])
    gram=-hessian*mask[:,:,None,None,None]*mask[:,None,:,None,None]
    gram=gram.transpose(0,1,3,2,4).reshape(count,capacity*dim,capacity*dim)
    active=jnp.repeat(mask,dim,axis=1)
    regularization=1e-12*jnp.maximum(1.,jnp.max(jnp.abs(gram),axis=(1,2)))
    gram+=jnp.eye(capacity*dim)[None,:,:]*(regularization[:,None]*active+1-active)[:,None,:]
    poly=(jnp.broadcast_to(jnp.eye(dim),(count,capacity,dim,dim))*mask[:,:,None,None]).reshape(count,capacity*dim,dim)
    saddle=jnp.concatenate((jnp.concatenate((gram,poly),axis=2),
        jnp.concatenate((poly.transpose(0,2,1),jnp.zeros((count,dim,dim))),axis=2)),axis=1)
    rhs=jnp.concatenate(((normals*mask[:,:,None]).reshape(count,capacity*dim),jnp.zeros((count,dim))),axis=1)
    coefficients=jnp.linalg.solve(saddle,rhs[:,:,None])[:,:,0]
    c=coefficients[:,:capacity*dim].reshape(count,capacity,dim)
    b=coefficients[:,capacity*dim:]
    potential=-3*jnp.sum(radius*jnp.einsum('pijd,pjd->pij',delta,c),axis=2)+jnp.einsum('pnd,pd->pn',sites,b)
    kernel=radius*mask[:,:,None]*mask[:,None,:]+jnp.eye(capacity)[None,:,:]*(1-mask)[:,None,:]
    column=mask[:,:,None]
    residual_matrix=jnp.concatenate((jnp.concatenate((kernel,column),axis=2),
        jnp.concatenate((column.transpose(0,2,1),jnp.zeros((count,1,1))),axis=2)),axis=1)
    residual_rhs=jnp.concatenate((potential*mask,jnp.zeros((count,1))),axis=1)
    residual=jnp.linalg.solve(residual_matrix,residual_rhs[:,:,None])[:,:,0]
    return c,b,residual[:,:capacity],residual[:,capacity]


@partial(jit,static_argnums=(10,))
def _local_value(x,centers,radii,sites,c,b,residual,constant,threshold,scale,max_active):
    q=(x-centers)/radii[:,None]
    r=_norm(q); t=jnp.maximum(1-r,0.)
    psi=t**4*(4*r+1)
    selected,indices=lax.top_k(psi,max_active)
    delta=q[indices,None,:]-sites[indices]
    distance=_norm(delta)
    raw=-3*jnp.sum(distance*jnp.einsum('pnd,pnd->pn',delta,c[indices]),axis=1)+jnp.einsum('pd,pd->p',q[indices],b[indices])
    correction=jnp.sum(distance*residual[indices],axis=1)+constant[indices]
    denominator=jnp.sum(selected)
    value=jnp.sum(selected*(-raw+correction)*radii[indices])/jnp.where(denominator>0,denominator,1.)
    z=jnp.clip(denominator/threshold,0.,1.)
    blend=z**3*(10-15*z+6*z*z)
    return value,blend,denominator


@jit
def _winding(x,points,area_normals):
    delta=points-x
    radius=_norm(delta)
    integral=jnp.sum(jnp.sum(delta*area_normals,axis=1)/jnp.maximum(radius,1e-150)**points.shape[1])
    return integral/(2*jnp.pi if points.shape[1]==2 else 4*jnp.pi)


@partial(jit,static_argnums=(15,))
def _point(x,centers,radii,sites,c,b,residual,constant,threshold,scale,
           coarse_points,coarse_normals,fine_points,fine_normals,tolerance,max_active):
    local,blend,_=_local_value(x,centers,radii,sites,c,b,residual,constant,threshold,scale,max_active)
    def far(_):
        coarse=_winding(x,coarse_points,coarse_normals)
        fine=_winding(x,fine_points,fine_normals)
        reliable=_winding_sign_reliable(coarse,fine,tolerance)
        return jnp.where(reliable,scale*(fine-.5),jnp.nan)
    # Boundary/interior-cover queries do not evaluate either winding integral.
    return lax.cond(blend>=1.,lambda _:local,lambda _:
        blend*local+(1-blend)*far(None),operand=None)


@jit
def _winding_sign_reliable(coarse,fine,tolerance):
    """Test classification margin, rather than absolute winding accuracy.

    Coarse/fine discrepancy is an empirical uncertainty estimate, not a
    rigorous quadrature error bound. Retain sign agreement and proximity to
    an integer; require the fine estimate's distance from the 1/2 decision
    threshold to exceed that discrepancy plus the tolerance margin.
    """
    finite=jnp.isfinite(coarse)&jnp.isfinite(fine)
    same=(coarse>.5)==(fine>.5)
    margin=jnp.abs(fine-.5)>jnp.abs(coarse-fine)+tolerance
    integer=jnp.abs(fine-jnp.round(fine))<=tolerance
    return finite & same & margin & integer


@partial(jit,static_argnums=(15,))
def _evaluate(points,*args):
    if len(points)==0: return jnp.empty((0,),dtype=points.dtype)
    width=16; padding=(-len(points))%width
    padded=jnp.concatenate((points,jnp.repeat(points[-1:],padding,axis=0)))
    def block(query):
        local,blend,_=jax.vmap(lambda p:_local_value(p,*args[:9],args[-1]))(query)
        # vmap of a per-point cond can evaluate both branches. One scalar cond
        # per block genuinely skips winding quadrature for all-local blocks.
        return lax.cond(jnp.all(blend>=1.),lambda _:local,
            lambda _:jax.vmap(lambda p:_point(p,*args))(query),operand=None)
    return lax.map(block,padded.reshape(-1,width,points.shape[1])).reshape(-1)[:len(points)]


@partial(jit,static_argnums=(15,))
def _gradient(points,*args):
    if len(points)==0: return jnp.empty_like(points)
    width=8; padding=(-len(points))%width
    padded=jnp.concatenate((points,jnp.repeat(points[-1:],padding,axis=0)))
    local_gradient=jax.grad(lambda p:_local_value(p,*args[:9],args[-1])[0])
    full_gradient=jax.grad(lambda p:_point(p,*args))
    def block(query):
        _,blend,_=jax.vmap(lambda p:_local_value(p,*args[:9],args[-1]))(query)
        return lax.cond(jnp.all(blend>=1.),lambda _:jax.vmap(local_gradient)(query),
            lambda _:jax.vmap(full_gradient)(query),operand=None)
    return lax.map(block,padded.reshape(-1,width,points.shape[1])).reshape(-1,points.shape[1])[:len(points)]


@partial(jit,static_argnums=(3,))
def _overlap_counts(points,centers,radii,chunk):
    """Number of patches with psi>0 (r<1) at each point; padded points/patches allowed."""
    def block(x):
        q=_norm((x[:,None,:]-centers[None,:,:])/radii[None,:,None])
        return jnp.sum(q<1,axis=1)
    return lax.map(block,points.reshape(-1,chunk,points.shape[1])).reshape(-1)


def _local_batch(n,width,capacity,dim,budget=1<<26):
    """Points per vmapped block so the (block, width, capacity, dim) temporaries fit."""
    per=max(1,width*capacity*dim)
    batch=1<<int(np.floor(np.log2(max(1,budget//per))))
    return int(min(n,max(1,batch)))


@partial(jit,static_argnums=(1,11))
def _grouped_local(points,width,centers,radii,sites,c,b,residual,constant,threshold,scale,batch):
    return lax.map(lambda p:_local_value(p,centers,radii,sites,c,b,residual,constant,threshold,scale,width),
                   points,batch_size=batch)


@partial(jit,static_argnums=(1,))
def _grouped_point(points,width,*args):
    def one(p): return _point(p,*args,width)
    return lax.map(one,points,batch_size=64)


@partial(jit,static_argnums=(1,11))
def _grouped_local_gradient(points,width,centers,radii,sites,c,b,residual,constant,threshold,scale,batch):
    g=jax.grad(lambda p:_local_value(p,centers,radii,sites,c,b,residual,constant,threshold,scale,width)[0])
    return lax.map(g,points,batch_size=batch)


@partial(jit,static_argnums=(1,))
def _grouped_point_gradient(points,width,*args):
    g=jax.grad(lambda p:_point(p,*args,width))
    return lax.map(g,points,batch_size=32)


def _width(count):
    """Power-of-two top-k width per point (>= its overlap count, >= 1)."""
    count=np.maximum(np.asarray(count,np.int64),1)
    return (1<<np.ceil(np.log2(count)).astype(np.int64)).astype(np.int64)


def _grouped(model,points,gradient=False):
    """Varun's grouping: bin points by overlap count, one fixed-shape kernel per bin.

    Within a bin of width w the top-w PU weights contain every nonzero weight,
    so the result equals the global-``max_active`` evaluation (the dropped
    weights are exact zeros). Points outside full cover (blend<1) are the only
    ones that evaluate the boundary-flux winding integrals.
    """
    from kernelpack.accelerators.jax_spatial import bucket,pad_rows
    pts=np.asarray(points,float)
    n,dim=pts.shape
    if n==0: return jnp.empty((0,dim) if gradient else (0,),dtype=float)
    centers,radii=model.centers,model.radii
    n_pad=bucket(n); chunk=min(n_pad,max(64,(1<<22)//max(1,int(centers.shape[0]))))
    chunk=1<<int(np.floor(np.log2(chunk)))
    counts=np.asarray(_overlap_counts(jnp.asarray(pad_rows(pts,n_pad,0.)),centers,radii,chunk))[:n]
    widths=_width(counts)
    core=(centers,radii,model.sites,model.coefficients,model.polynomial,model.residual,model.constants,
          model.threshold,model.scale)
    full=core+(model.coarse_points,model.coarse_normals,model.fine_points,model.fine_normals,model.tolerance)
    out=np.empty((n,dim) if gradient else n)
    for w in np.unique(widths):
        ids=np.flatnonzero(widths==w); w=int(min(w,int(centers.shape[0])))
        sub=jnp.asarray(pad_rows(pts[ids],bucket(len(ids)),0.))
        batch=_local_batch(sub.shape[0],w,int(model.sites.shape[1]),dim)
        local,blend,weight=map(np.asarray,_grouped_local(sub,w,*core,batch))
        local,blend,weight=local[:len(ids)],blend[:len(ids)],weight[:len(ids)]
        if gradient:
            out[ids]=np.asarray(_grouped_local_gradient(sub,w,*core,max(1,batch//4)))[:len(ids)]
        else:
            out[ids]=local
        if model.band is not None:
            # Inside the cover phi = f_loc; outside, the component sign times D.
            outside=weight<=0.
            if outside.any():
                if gradient: out[ids[outside]]=0.
                else: out[ids[outside]]=model.scale*_band_sign(model.band,pts[ids[outside]])
            continue
        far=np.flatnonzero(blend<1.)
        if len(far):
            fsub=jnp.asarray(pad_rows(pts[ids[far]],bucket(len(far)),0.))
            if gradient: out[ids[far]]=np.asarray(_grouped_point_gradient(fsub,w,*full))[:len(far)]
            else: out[ids[far]]=np.asarray(_grouped_point(fsub,w,*full))[:len(far)]
    return jnp.asarray(out)


def _local_only(model,points):
    """f_loc and the total PU weight at points, grouped by overlap count."""
    from kernelpack.accelerators.jax_spatial import bucket,pad_rows
    pts=np.asarray(points,float); n,dim=pts.shape
    centers,radii=model.centers,model.radii
    n_pad=bucket(n); chunk=min(n_pad,max(64,(1<<22)//max(1,int(centers.shape[0]))))
    chunk=1<<int(np.floor(np.log2(chunk)))
    counts=np.asarray(_overlap_counts(jnp.asarray(pad_rows(pts,n_pad,0.)),centers,radii,chunk))[:n]
    widths=_width(counts)
    core=(centers,radii,model.sites,model.coefficients,model.polynomial,model.residual,model.constants,
          model.threshold,model.scale)
    local=np.empty(n); weight=np.empty(n)
    for w in np.unique(widths):
        ids=np.flatnonzero(widths==w); w=int(min(w,int(centers.shape[0])))
        sub=jnp.asarray(pad_rows(pts[ids],bucket(len(ids)),0.))
        batch=_local_batch(sub.shape[0],w,int(model.sites.shape[1]),dim)
        value,_,total=map(np.asarray,_grouped_local(sub,w,*core,batch))
        local[ids]=value[:len(ids)]; weight[ids]=total[:len(ids)]
    return local,weight


@jax.jit
def _label_components(free):
    """Face-connected components of the True cells of a d-dimensional grid (min-label propagation)."""
    big=jnp.iinfo(jnp.int32).max
    labels=jnp.where(free,jnp.arange(free.size,dtype=jnp.int32).reshape(free.shape),big)
    def neighbours_min(l):
        out=l
        for axis in range(l.ndim):
            for shift in (1,-1):
                rolled=jnp.roll(l,shift,axis=axis)
                edge=jnp.arange(l.shape[axis])==(0 if shift==1 else l.shape[axis]-1)
                shape=[1]*l.ndim; shape[axis]=l.shape[axis]
                rolled=jnp.where(edge.reshape(shape),big,rolled)
                out=jnp.minimum(out,rolled)
        return jnp.where(free,out,big)
    def cond(state): return state[1]
    def body(state):
        l,_=state; new=neighbours_min(l)
        return new,jnp.any(new!=l)
    labels,_=lax.while_loop(cond,body,(labels,jnp.bool_(True)))
    return labels


@jax.jit
def _component_signs(labels,free,cell_sign):
    """For each free cell, the sign shared by the covered face-neighbours of its component.

    Returns (sign per cell, conflict flag, orphan flag)."""
    big=jnp.iinfo(jnp.int32).max
    n=labels.size
    pos=jnp.zeros_like(free); neg=jnp.zeros_like(free)
    for axis in range(labels.ndim):
        for shift in (1,-1):
            nb_sign=jnp.roll(cell_sign,shift,axis=axis); nb_free=jnp.roll(free,shift,axis=axis)
            edge=jnp.arange(labels.shape[axis])==(0 if shift==1 else labels.shape[axis]-1)
            shape=[1]*labels.ndim; shape[axis]=labels.shape[axis]
            valid=free & ~nb_free & ~edge.reshape(shape)
            pos=pos|(valid&(nb_sign>0)); neg=neg|(valid&(nb_sign<0))
    flat=jnp.where(free,labels,0).reshape(-1)
    any_pos=jax.ops.segment_max(pos.reshape(-1).astype(jnp.int32),flat,num_segments=n)
    any_neg=jax.ops.segment_max(neg.reshape(-1).astype(jnp.int32),flat,num_segments=n)
    comp_pos=any_pos[flat].reshape(labels.shape)>0; comp_neg=any_neg[flat].reshape(labels.shape)>0
    sign=jnp.where(free,jnp.where(comp_pos,1,-1),cell_sign)
    conflict=jnp.any(free&comp_pos&comp_neg); orphan=jnp.any(free&~comp_pos&~comp_neg)
    return sign,conflict,orphan


def _build_band(model,points):
    """Sign of phi on the uncovered region, by connected components of a background grid."""
    pts=np.asarray(points,float)
    lo=pts.min(axis=0); hi=pts.max(axis=0); pad=0.05*(hi-lo)+1e-12
    lo=lo-pad; hi=hi+pad
    cell=0.5*float(np.min(np.asarray(model.radii)))
    shape=tuple(int(s) for s in np.maximum(1,np.ceil((hi-lo)/cell)).astype(int))
    grids=np.meshgrid(*[lo[a]+(np.arange(shape[a])+0.5)*cell for a in range(len(shape))],indexing='ij')
    centres=np.stack([g.reshape(-1) for g in grids],axis=1)
    local,weight=_local_only(model,centres)
    covered=(weight>0).reshape(shape)
    cell_sign=np.where(covered,np.sign(local).reshape(shape),0).astype(np.int32)
    free=jnp.asarray(~covered)
    labels=_label_components(free)
    sign,conflict,orphan=_component_signs(labels,free,jnp.asarray(cell_sign))
    if bool(conflict): raise ValueError('CFPU cover does not separate interior from exterior: a component borders both signs')
    if bool(orphan): raise ValueError('CFPU band sign: an uncovered component has no covered neighbour')
    components=int(np.unique(np.asarray(labels)[np.asarray(free)]).size) if bool(jnp.any(free)) else 0
    return dict(lo=lo,cell=cell,shape=shape,sign=np.asarray(sign,dtype=np.int8).reshape(-1),components=components)


def _band_traced(model,points,gradient=False):
    """Band-mode phi (or its gradient) for traced points: global-width local evaluation plus a device sign lookup."""
    n,dim=points.shape
    width=int(model.centers.shape[0])
    core=(model.centers,model.radii,model.sites,model.coefficients,model.polynomial,model.residual,model.constants,
          model.threshold,model.scale)
    batch=_local_batch(n,width,int(model.sites.shape[1]),dim)
    local,_,weight=_grouped_local(points,width,*core,batch)
    covered=weight>0.
    if gradient:
        g=_grouped_local_gradient(points,width,*core,max(1,batch//4))
        return jnp.where(covered[:,None],g,0.)
    band=model.band
    shape=jnp.asarray(band['shape'])
    idx=jnp.floor((points-jnp.asarray(band['lo']))/band['cell']).astype(jnp.int64)
    inside=jnp.all((idx>=0)&(idx<shape),axis=1)
    idx=jnp.clip(idx,0,shape-1)
    flat=jnp.ravel_multi_index(tuple(idx.T),tuple(band['shape']),mode='clip')
    sign=jnp.where(inside,jnp.asarray(band['sign'])[flat],-1).astype(float)
    return jnp.where(covered,local,model.scale*sign)


def _band_sign(band,points):
    pts=np.asarray(points,float)
    idx=np.floor((pts-band['lo'])/band['cell']).astype(np.int64)
    shape=np.asarray(band['shape'])
    inside=np.all((idx>=0)&(idx<shape),axis=1)
    idx=np.clip(idx,0,shape-1)
    flat=np.ravel_multi_index(idx.T,band['shape'])
    return np.where(inside,band['sign'][flat],-1).astype(float)


@dataclass
class CFPUModel:
    centers: object
    radii: object
    sites: object
    coefficients: object
    polynomial: object
    residual: object
    constants: object
    threshold: float
    scale: float
    coarse_points: object
    coarse_normals: object
    fine_points: object
    fine_normals: object
    tolerance: float
    max_active: int
    diagnostics: dict=field(default_factory=dict)
    # Band sign grid (farfield='band'): origin, cell size, shape, per-cell sign.
    band: dict=None

    def args(self):
        return (self.centers,self.radii,self.sites,self.coefficients,self.polynomial,
            self.residual,self.constants,self.threshold,self.scale,self.coarse_points,
            self.coarse_normals,self.fine_points,self.fine_normals,self.tolerance,self.max_active)

    def evaluate(self,points):
        # Under tracing (e.g. jax.jit(level.evaluate)) host binning is impossible;
        # the global top-k width (patch count) gives identical values.
        if isinstance(points,jax.core.Tracer):
            if self.band is not None: return _band_traced(self,points)
            return _evaluate(points,*self.args())
        return _grouped(self,points)

    def gradient(self,points):
        if isinstance(points,jax.core.Tracer):
            if self.band is not None: return _band_traced(self,points,gradient=True)
            return _gradient(points,*self.args())
        return _grouped(self,points,gradient=True)

    def checked_evaluate(self,points):
        result=self.evaluate(points)
        if not np.isfinite(np.asarray(result)).all():
            raise ValueError('CFPU far-field sign is unresolved: refine geometry quadrature/cover before classification')
        return result


def _piece_quadrature(segment,order):
    """Value-only parameter evaluation and quadrature; no surface mesh.

    Returns the mapped quadrature points, their area-weighted normals, the parameters, and one boolean
    footprint mask per parametric support (rectangle or spherical cap) over the parameters."""
    from kernelpack.nodes.hierarchical import points_in_polygon_bucketed
    model=segment.geom_model
    if model.get('type')=='spherical-pu':
        # Closed piece over S^{d-1}: product rule on the sphere, metric of the map along an orthonormal
        # tangent basis, so area = w |J t_1 x J t_2| (d = 3) or w |J t_1| (d = 2).
        from .spherical_pu import sphere_quadrature
        pu=model['model']; u,qw=sphere_quadrature(pu.dim,order)
        xyz=np.asarray(pu.evaluate(u)); jac,normal=map(np.asarray,pu.frame(u))
        normal=normal*model.get('normal_sign',1.)
        metric=np.linalg.norm(jac[:,:,0],axis=1) if pu.dim==2 else np.linalg.norm(np.cross(jac[:,:,0],jac[:,:,1]),axis=1)
        footprints=[pu.cap_mask(u,j) for j in range(int(pu.centers.shape[0]))]
        return xyz,normal*(qw*metric)[:,None],u,footprints
    if model.get('type')=='parametric-pu':
        outlines=model.get('outlines')
        if outlines is None: raise ValueError('CFPU requires the physical parameter footprint')
        pu=model['model']; lower=np.asarray(outlines[0]).min(axis=0); upper=np.asarray(outlines[0]).max(axis=0)
        parent_centers=np.asarray(pu.centers)*np.asarray(pu.scale)+np.asarray(pu.offset)
        parent_widths=np.asarray(pu.widths)*np.asarray(pu.scale)
        mapper=pu.evaluate; frame=pu.frame
        normal_sign=model.get('normal_sign',1.)
    elif model.get('type')=='affine-face':
        outlines=model['outlines']; lower=np.asarray(outlines[0]).min(axis=0); upper=np.asarray(outlines[0]).max(axis=0)
        parent_centers=np.asarray(model['cover_centers']); parent_widths=np.asarray(model['cover_widths'])
        origin=jnp.asarray(model['origin']); basis=jnp.asarray(model['basis']); normal=jnp.asarray(model['normal'])
        mapper=lambda uv:origin+jnp.asarray(uv)@basis.T
        frame=lambda uv:(jnp.broadcast_to(basis,(len(uv),*basis.shape)),jnp.broadcast_to(normal,(len(uv),len(normal))))
        normal_sign=1.
    else:
        raise ValueError('CFPU requires shared PU pieces (parametric or spherical) or explicit affine-face geometry; legacy global patches are unsupported')
    nodes,weights=np.polynomial.legendre.leggauss(order)
    axes=[(a+b)/2+(b-a)*nodes/2 for a,b in zip(lower,upper)]
    weight_axes=[(b-a)*weights/2 for a,b in zip(lower,upper)]
    uv=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,len(lower))
    qw=np.prod(np.stack(np.meshgrid(*weight_axes,indexing='ij'),axis=-1),axis=-1).ravel()
    if len(lower)==2:
        keep=points_in_polygon_bucketed(uv,np.asarray(outlines[0]))
        for hole in outlines[1:]: keep &= ~points_in_polygon_bucketed(uv,np.asarray(hole))
        uv=uv[keep]; qw=qw[keep]
    xyz=np.asarray(mapper(uv)); jac,normal=frame(uv); jac,normal=np.asarray(jac),np.asarray(normal)*normal_sign
    metric=np.linalg.norm(jac[:,:,0],axis=1) if len(lower)==1 else np.linalg.norm(np.cross(jac[:,:,0],jac[:,:,1]),axis=1)
    footprints=[np.all(np.abs(uv-center)<width,axis=1) for center,width in zip(parent_centers,parent_widths)]
    return xyz,normal*(qw*metric)[:,None],uv,footprints


@partial(jit,static_argnums=(3,))
def _ball_counts(points,centers,radii,k):
    """Members per ball (|x-c|<r) and the k-th smallest distance to each center."""
    def one(center,radius):
        d=_norm(points-center)
        return jnp.sum(d<radius),-lax.top_k(-d,k)[0][-1]
    return lax.map(lambda a:one(*a),(centers,radii),batch_size=8)


@jit
def _ball_coverage(points,centers,radii):
    """Sum of Wendland weights per point; radius-0 (padding) balls contribute nothing."""
    def one(x):
        r=_norm(x-centers)/jnp.where(radii>0,radii,1.)
        t=jnp.where(radii>0,jnp.maximum(1-r,0.),0.)
        return jnp.sum(t**4*(4*r+1))
    return lax.map(one,points,batch_size=256)


@partial(jit,static_argnums=(4,))
def _ball_members(points,normals,centers,radii,capacity):
    """Ascending member lists per ball, as local (scaled) sites with their normals."""
    def one(center,radius):
        member=_norm(points-center)<radius
        order=jnp.argsort(~member,stable=True)[:capacity]
        ok=member[order]
        return (jnp.where(ok[:,None],(points[order]-center)/radius,0.),
                jnp.where(ok[:,None],normals[order],0.),ok.astype(points.dtype))
    return lax.map(lambda a:one(*a),(centers,radii),batch_size=8)


def build_cfpu(segments,points,normals,*,options=CFPUOptions(),cover=None):
    """Map one existing geometry support to each ambient ball, then fit CFPU."""
    from scipy.spatial import cKDTree
    points=np.asarray(points,float); normals=np.asarray(normals,float)
    if points.ndim!=2 or points.shape[1] not in (2,3) or normals.shape!=points.shape or not np.isfinite(points).all() or not np.isfinite(normals).all():
        raise ValueError('finite oriented curve/surface samples are required')
    if options.farfield not in ('winding','band'):
        raise ValueError("CFPU farfield must be 'winding' or 'band'")
    if options.support_inflation<=1 or options.min_sites<2 or options.max_sites<options.min_sites or options.quadrature_order<8 or not 0<options.winding_tolerance<.5:
        raise ValueError('invalid CFPU options')
    normals=normals/np.linalg.norm(normals,axis=1)[:,None]
    if not np.isfinite(normals).all(): raise ValueError('CFPU normals must be nonzero')
    scale=float(np.linalg.norm(np.ptp(points,axis=0)))
    spacing=float(np.median(cKDTree(points).query(points,k=2)[0][:,1]))
    coarse=[]; fine=[]; centers=[]; radii=[]; parents=[]
    for index,segment in enumerate(segments):
        coarse.append(_piece_quadrature(segment,options.quadrature_order)[:2])
        xyz,area,uv,footprints=_piece_quadrature(segment,2*options.quadrature_order)
        fine.append((xyz,area))
        for patch,inside in enumerate(footprints):
            footprint=xyz[inside]
            if not len(footprint): continue
            # The mapped sampled footprint sets the envelope; bounded inflation
            # gives ambient thickness without solving for parameter coordinates.
            world_center=footprint.mean(axis=0)
            radius=options.support_inflation*max(float(np.linalg.norm(footprint-world_center,axis=1).max()),2*spacing)
            centers.append(world_center); radii.append(radius); parents.append((index,patch))
    centers=np.asarray(centers); radii=np.asarray(radii)
    if cover is not None:
        # Controlled density comparisons reuse the complete accepted ambient
        # cover, including any minimum-site growth, rather than changing it.
        if parents!=[tuple(p) for p in cover.diagnostics['parent_patches']]:
            raise ValueError('frozen CFPU cover has different parent patches')
        centers=np.asarray(cover.centers).copy(); radii=np.asarray(cover.radii).copy()
    if not len(centers): raise ValueError('empty mapped CFPU cover')
    coarse_points,coarse_normals=map(lambda items:jnp.asarray(np.vstack(items)),zip(*coarse))
    fine_points,fine_normals=map(lambda items:jnp.asarray(np.vstack(items)),zip(*fine))
    flux=np.asarray(fine_normals).sum(axis=0)
    area=float(np.linalg.norm(np.asarray(fine_normals),axis=1).sum())
    if np.linalg.norm(flux)>options.winding_tolerance*area:
        raise ValueError('CFPU far-field extension requires a consistently oriented closed boundary')
    from kernelpack.accelerators.jax_spatial import bucket,pad_rows
    if len(points)<options.min_sites: raise ValueError('CFPU needs more oriented geometry samples')
    n_pad=bucket(len(points),64); p_pad=bucket(len(centers),8)
    far=np.where(np.arange(n_pad)[:,None]<len(points),0.,1e300)
    pts_dev=jnp.asarray(pad_rows(points,n_pad,0.)+far)
    cent_dev=jnp.asarray(pad_rows(centers,p_pad,0.))
    counts,kth=map(np.asarray,_ball_counts(pts_dev,cent_dev,jnp.asarray(pad_rows(radii,p_pad,0.)),options.min_sites))
    counts,kth=counts[:len(centers)],kth[:len(centers)]
    initial_radii=radii.copy()
    short=np.flatnonzero(counts<options.min_sites)
    if len(short):
        if cover is not None: raise ValueError('frozen CFPU cover has insufficient fit sites')
        required=1.01*kth[short]
        if np.any(required>options.max_support_growth*radii[short]):
            raise ValueError('CFPU minimum-site growth exceeds its bound; refine the shared geometry/data')
        radii[short]=np.maximum(radii[short],required)
        counts=np.asarray(_ball_counts(pts_dev,cent_dev,jnp.asarray(pad_rows(radii,p_pad,0.)),options.min_sites)[0])[:len(centers)]
    if np.any((counts<options.min_sites)|(counts>options.max_sites)):
        flagged=[parents[i] for i in np.flatnonzero(counts>options.max_sites)]
        raise CFPUCoverError('CFPU supports need shared geometry refinement or more data: '+str(counts.tolist()),flagged)
    rad_dev=jnp.asarray(pad_rows(radii,p_pad,1.))
    coverage=np.asarray(_ball_coverage(pts_dev,cent_dev,jnp.where(jnp.arange(p_pad)<len(centers),rad_dev,0.)))[:len(points)]
    if float(coverage.min())<=1e-12: raise ValueError('mapped CFPU cover misses boundary samples; refine/inflate the shared geometry cover')
    capacity=bucket(int(counts.max()),1)
    nrm_dev=jnp.asarray(pad_rows(normals,n_pad,0.))
    sites,values,mask=_ball_members(pts_dev,nrm_dev,cent_dev,rad_dev,capacity)
    sites,values,mask=(np.asarray(sites)[:len(centers)],np.asarray(values)[:len(centers)],np.asarray(mask)[:len(centers)])
    # Group by capacity: a small planar fit must not inherit the cap's largest
    # dense saddle size. One batched solve per power-of-two bucket, chunked only
    # by a device-memory budget.
    capacities=np.array([bucket(int(k),1) for k in counts])
    dim=points.shape[1]
    c=np.zeros_like(sites); b=np.zeros((len(centers),dim))
    residual=np.zeros((len(centers),capacity)); constant=np.zeros(len(centers))
    for size in np.unique(capacities):
        group=np.flatnonzero(capacities==size)
        per=12*((int(size)*dim+dim)**2+int(size)**2*dim*dim)*8
        batch=bucket(max(1,min(len(group),(3<<30)//per)),1)
        for start in range(0,len(group),batch):
            g=group[start:start+batch]; m=len(g)
            pad_mask=pad_rows(mask[g,:size],batch,0.)
            sc,sb,sr,sk=_local_solve(jnp.asarray(pad_rows(sites[g,:size],batch,0.)),
                jnp.asarray(pad_rows(values[g,:size],batch,0.)),jnp.asarray(pad_mask))
            c[g,:size]=np.asarray(sc)[:m]; b[g]=np.asarray(sb)[:m]
            residual[g,:size]=np.asarray(sr)[:m]; constant[g]=np.asarray(sk)[:m]
    if not all(np.all(np.isfinite(item)) for item in (c,b,residual,constant)):
        raise ValueError('nonfinite CFPU local solve')
    # Evaluation groups points by their own overlap count; the global bound is
    # kept only for legacy callers of the per-point kernels (patch count is safe).
    max_active=len(centers)
    c,b,residual,constant=map(jnp.asarray,(c,b,residual,constant))
    model=CFPUModel(jnp.asarray(centers),jnp.asarray(radii),jnp.asarray(sites),c,b,residual,constant,
        float(coverage.min())*.5 if cover is None else cover.threshold,
        scale if cover is None else cover.scale,coarse_points,coarse_normals,fine_points,fine_normals,options.winding_tolerance,max_active)
    band_fallback=None
    if options.farfield=='band':
        try: model.band=_build_band(model,points)
        except ValueError as error:
            # The PU cover does not determine the far-field sign; use the winding number.
            band_fallback=str(error); model.band=None
            print(f'[cfpu] band far-field sign failed ({error}); falling back to the winding number',flush=True)
    nodal=np.asarray(model.evaluate(points))
    if not np.isfinite(nodal).all() or np.abs(nodal).max()>options.interpolation_tolerance*scale:
        raise ValueError('CFPU failed boundary interpolation validation')
    model.diagnostics=dict(patch_count=len(centers),parent_patches=parents,support_counts=counts.tolist(),
        padded_capacity=capacity,max_active=max_active,min_boundary_weight=float(coverage.min()),
        solve_capacities=capacities.tolist(),overlap_bound='per-point overlap counts (grouped evaluation)',
        max_boundary_residual=float(np.abs(nodal).max()),net_flux_relative=float(np.linalg.norm(flux)/area),
        quadrature_orders=[options.quadrature_order,2*options.quadrature_order],options=vars(options),
        farfield='checked boundary flux quadrature; no global CFI fallback',inverse_map_solves=False,
        support_growth=(radii/initial_radii).tolist(),
        frozen_cover=cover is not None,
        local_kernel='curl-free r^3 plus constant vectors; r plus constant residual')
    if band_fallback is not None:
        model.diagnostics.update(farfield='winding number (band sign failed)',band_fallback_reason=band_fallback)
    if model.band is not None:
        model.diagnostics.update(farfield='band sign by connected components',band_cell=model.band['cell'],
            band_shape=list(model.band['shape']),band_components=model.band['components'])
    return model
