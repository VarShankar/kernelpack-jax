"""Meshfree partition-of-unity models of parametric coordinate maps.

Local cubic PHS fits have affine augmentation. Rectangles describe overlapping
supports in parameter space; no surface elements or connectivity are used.
The caller supplies a valid parametrization and its physical trimming curves.
"""
from dataclasses import dataclass, field
from functools import partial
from itertools import product
import math
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, lax


@dataclass(frozen=True)
class PUOptions:
    initial_grid: int = 2
    overlap: float = 1.5
    relative_position_tolerance: float = 1e-3
    normal_tolerance: float = .05
    min_jacobian_ratio: float = .05
    min_sites: int = 12
    max_refinements: int = 3
    max_patches: int = 128


class PUFitError(ValueError):
    def __init__(self,message,history):
        super().__init__(message)
        self.history=history


def rectangular_cover(lower,upper,grid=2):
    """Common initial parameter cover for PU maps and exact affine pieces."""
    lower,upper=jnp.asarray(lower),jnp.asarray(upper)
    axes=[jnp.linspace(a,b,grid+1) for a,b in zip(lower,upper)]
    return [(jnp.array([axes[k][i] for k,i in enumerate(index)]),
             jnp.array([axes[k][i+1] for k,i in enumerate(index)]))
            for index in product(range(grid),repeat=len(lower))]


def refine_cover_cells(cells,indices):
    """Dyadically refine requested supports in the common parameter hierarchy."""
    result=[]
    for index,(lower,upper) in enumerate(cells):
        if index not in indices: result.append((lower,upper)); continue
        middle=(jnp.asarray(lower)+jnp.asarray(upper))/2
        for bits in product((0,1),repeat=len(lower)):
            bits=jnp.asarray(bits,bool)
            result.append((jnp.where(bits,middle,lower),jnp.where(bits,upper,middle)))
    return result


@dataclass
class ParametricSurfaceData:
    """One smooth curve or face, with an explicit parameter footprint.

    outlines are ordered closed loops in parameter space. The first bounds
    the face; subsequent loops are holes. Validation data are separate geometry
    samples, never measurements from a PDE solve. Normals are optional.
    """
    positions: object
    parameters: object
    outlines: object
    validation_parameters: object = None
    validation_positions: object = None
    validation_normals: object = None
    options: PUOptions = field(default_factory=PUOptions)
    parameterization: object = None

    def __len__(self): return len(self.positions)

    @classmethod
    def from_geometry(cls,positions,boundary_loops,*,normals=None,
            validation_positions=None,validation_normals=None,options=PUOptions()):
        # Independently assembled face pieces can repeat the same physical site
        # a few floating-point ulps apart. Coalesce those sites before fitting;
        # distinct physical points that share a radial direction still fail the
        # projection guards below. This tolerance is numerical, not geometric.
        from scipy.spatial import cKDTree
        positions=np.asarray(positions,float)
        if positions.ndim!=2 or positions.shape[1] not in (2,3) or not len(positions) or not np.isfinite(positions).all():
            raise ValueError('finite 2D curve or 3D face data are required')
        tolerance=64*np.finfo(float).eps*max(float(np.abs(positions).max()),float(np.ptp(positions,axis=0).max()),np.finfo(float).tiny)
        pairs=cKDTree(positions).query_pairs(tolerance,output_type='ndarray')
        if len(pairs):
            parent=np.arange(len(positions))
            def root(i):
                while parent[i]!=i:
                    parent[i]=parent[parent[i]]; i=parent[i]
                return i
            for a,b in pairs:
                a,b=root(a),root(b)
                parent[max(a,b)]=min(a,b)
            representatives=np.array([root(i) for i in range(len(positions))])
            keep=np.unique(representatives)
            if normals is not None:
                normals=np.asarray(normals,float)
                if normals.shape!=positions.shape or not np.isfinite(normals).all() or np.any(np.linalg.norm(normals,axis=1)==0):
                    raise ValueError('invalid face normals')
                unit=normals/np.linalg.norm(normals,axis=1)[:,None]
                if np.max(np.linalg.norm(unit-unit[representatives],axis=1))>1e-6:
                    raise ValueError('coincident sites have inconsistent smooth-face normals')
                normals=normals[keep]
            positions=positions[keep]
        coordinates=(parameterize_curve_piece(positions,boundary_loops) if positions.shape[1]==2
                     else parameterize_surface_piece(positions,boundary_loops,normals=normals))
        parameters=coordinates.project(positions)
        outlines=[coordinates.project(loop) for loop in boundary_loops]
        checks=None if validation_positions is None else coordinates.project(validation_positions)
        result=cls(positions,parameters,outlines,checks,validation_positions,validation_normals,options)
        result.parameterization=coordinates
        return result


@jit
def _solve_local_fits(sites, values, active):
    # Padded rows have zero kernel/polynomial entries and a unit diagonal. They
    # cannot change a real fit. One batched device solve for equal-size buckets.
    mask=active.astype(sites.dtype)
    delta=sites[:,:,None,:]-sites[:,None,:,:]
    kernel=jnp.sum(delta*delta,axis=-1)**1.5
    kernel=kernel*mask[:,:,None]*mask[:,None,:]
    poly=jnp.concatenate((jnp.ones((*sites.shape[:2],1)),sites),axis=-1)*mask[:,:,None]
    count=sites.shape[1]; terms=sites.shape[2]+1
    reg=1e-12*jnp.maximum(1.,jnp.max(jnp.abs(kernel),axis=(1,2)))
    diagonal=reg[:,None]*mask+(1-mask)
    kernel=kernel+jnp.eye(count)[None,:,:]*diagonal[:,:,None]
    a=jnp.concatenate((jnp.concatenate((kernel,poly),axis=2),
        jnp.concatenate((jnp.swapaxes(poly,1,2),jnp.zeros((len(sites),terms,terms))),axis=2)),axis=1)
    origin=jnp.sum(values*mask[:,:,None],axis=1)/jnp.sum(mask,axis=1)[:,None]
    rhs=jnp.concatenate(((values-origin[:,None,:])*mask[:,:,None],
                         jnp.zeros((len(sites),terms,values.shape[2]))),axis=1)
    coefficients=jnp.linalg.solve(a,rhs)
    return coefficients[:,:count],coefficients[:,count:],origin


def _point(parameter,sites,weights,poly,origins,centers,widths,offset,scale,max_active):
    z=(parameter-offset)/scale
    relative=jnp.abs((z-centers)/widths)
    # Tensor-product C2 Wendland weights. These are PU weights, not the PHS.
    q=jnp.maximum(1-relative,0.)
    psi=jnp.prod(q**4*(4*relative+1),axis=1)
    selected,indices=lax.top_k(psi,max_active)
    local=(z-centers[indices])/widths[indices]
    delta=local[:,None,:]-sites[indices]
    kernel=jnp.sum(delta*delta,axis=-1)**1.5
    values=jnp.einsum('pn,pnm->pm',kernel,weights[indices])
    values+=jnp.einsum('pt,ptm->pm',jnp.concatenate((jnp.ones((max_active,1)),local),axis=1),poly[indices])
    values+=origins[indices]
    denominator=jnp.sum(selected)
    return jnp.sum(selected[:,None]*values,axis=0)/denominator


@partial(jit,static_argnums=(9,))
def _evaluate(parameters,sites,weights,poly,origins,centers,widths,offset,scale,max_active):
    def evaluate(p): return _point(p,sites,weights,poly,origins,centers,widths,offset,scale,max_active)
    return lax.map(evaluate,parameters,batch_size=64)


@partial(jit,static_argnums=(9,))
def _frame(parameters,sites,weights,poly,origins,centers,widths,offset,scale,max_active):
    def evaluate(p): return _point(p,sites,weights,poly,origins,centers,widths,offset,scale,max_active)
    derivative=jax.jacfwd(evaluate)
    jacobian=lax.map(derivative,parameters,batch_size=16)
    if parameters.shape[1]==2:
        normal=jnp.cross(jacobian[:,:,0],jacobian[:,:,1])
    else:
        tangent=jacobian[:,:,0]
        normal=jnp.stack((tangent[:,1],-tangent[:,0]),axis=1)
    length=jnp.linalg.norm(normal,axis=1)
    return jacobian,normal/length[:,None]


@partial(jit,static_argnums=(5,))
def _rect_overlap_counts(parameters,centers,widths,offset,scale,chunk):
    """Number of rectangle supports with psi>0 at each parameter point."""
    def block(p):
        z=(p-offset)/scale
        return jnp.sum(jnp.all(jnp.abs(z[:,None,:]-centers[None,:,:])<widths[None,:,:],axis=2),axis=1)
    return lax.map(block,parameters.reshape(-1,chunk,parameters.shape[1])).reshape(-1)


def _point_batch(n,width,capacity,budget=1<<25):
    per=max(1,width*capacity*4)
    return int(min(n,max(1,1<<int(np.floor(np.log2(max(1,budget//per)))))))


@partial(jit,static_argnums=(1,10))
def _grouped_evaluate(parameters,width,sites,weights,poly,origins,centers,widths,offset,scale,batch):
    return lax.map(lambda p:_point(p,sites,weights,poly,origins,centers,widths,offset,scale,width),parameters,batch_size=batch)


@partial(jit,static_argnums=(1,10))
def _grouped_frame(parameters,width,sites,weights,poly,origins,centers,widths,offset,scale,batch):
    derivative=jax.jacfwd(lambda p:_point(p,sites,weights,poly,origins,centers,widths,offset,scale,width))
    jacobian=lax.map(derivative,parameters,batch_size=batch)
    if parameters.shape[1]==2:
        normal=jnp.cross(jacobian[:,:,0],jacobian[:,:,1])
    else:
        tangent=jacobian[:,:,0]
        normal=jnp.stack((tangent[:,1],-tangent[:,0]),axis=1)
    return jacobian,normal/jnp.linalg.norm(normal,axis=1)[:,None]


@jit
def _width_order(counts):
    """Power-of-two width exponent per point, a stable sort by it, and the exponent histogram."""
    e=32-lax.clz(jnp.maximum(counts,1).astype(jnp.int32)-1)
    return jnp.argsort(e,stable=True),jnp.zeros(33,jnp.int32).at[e].add(1)


@partial(jit,static_argnums=(3,))
def _group_ids(order,start,count,size):
    pos=jnp.arange(size)
    return jnp.where(pos<count,order[jnp.minimum(start+pos,order.shape[0]-1)],order.shape[0])


def device_groups(counts):
    """Yield (width, count, ids) per power-of-two overlap width, ascending.

    ids hold that group's point indices in ascending order, padded to bucket(count)
    with n (out of range). Only the 33-entry width histogram reaches the host."""
    from kernelpack.accelerators.jax_spatial import bucket
    order,hist=_width_order(counts); start=0
    for e,count in enumerate(np.asarray(hist).tolist()):
        if not count: continue
        yield 1<<e,count,_group_ids(order,start,count,bucket(count))
        start+=count


@jit
def take_rows(x,ids):
    """x[ids] with zero rows for out-of-range ids (the padding of device_groups)."""
    return jnp.take(x,ids,axis=0,mode='fill',fill_value=0)


@jit
def put_rows(out,ids,values):
    return out.at[ids].set(values[:len(ids)].astype(out.dtype),mode='drop')


@partial(jit,static_argnums=(1,))
def pad_device(x,n,value=0.):
    return jnp.concatenate((x,jnp.full((n-x.shape[0],)+x.shape[1:],value,x.dtype)))


def _grouped_parametric(model,parameters,frame=False):
    """Bin parameter points by overlap count; one fixed-shape kernel per bin, on device.

    The top-w weights of a bin contain every nonzero weight, so results equal
    the global ``max_active`` evaluation (dropped weights are exact zeros).
    """
    from kernelpack.accelerators.jax_spatial import bucket
    params=jnp.asarray(parameters,dtype=float)
    n=len(params); dim_out=int(model.origins.shape[-1])
    if n==0:
        return (jnp.zeros((0,dim_out,params.shape[1])),jnp.zeros((0,dim_out))) if frame else jnp.zeros((0,dim_out))
    count=int(model.centers.shape[0])
    n_pad=bucket(n); chunk=min(n_pad,max(64,(1<<22)//max(1,count)))
    chunk=1<<int(np.floor(np.log2(chunk)))
    counts=_rect_overlap_counts(pad_device(params,n_pad),model.centers,model.widths,model.offset,model.scale,chunk)[:n]
    args=(model.sites,model.weights,model.poly_coeffs,model.origins,model.centers,model.widths,model.offset,model.scale)
    if frame: jac=jnp.zeros((n,dim_out,params.shape[1])); nr=jnp.zeros((n,dim_out))
    else: out=jnp.zeros((n,dim_out))
    for w,_,ids in device_groups(counts):
        w=int(min(w,count)); sub=take_rows(params,ids)
        batch=_point_batch(sub.shape[0],w,int(model.sites.shape[1]))
        if frame:
            j,v=_grouped_frame(sub,w,*args,max(1,batch//4)); jac=put_rows(jac,ids,j); nr=put_rows(nr,ids,v)
        else:
            out=put_rows(out,ids,_grouped_evaluate(sub,w,*args,batch))
    return (jac,nr) if frame else out


def _max_overlap(centers,widths):
    # Rectangle membership changes only at support boundaries. Midpoints of
    # consecutive boundaries cover every open membership region exactly.
    axes=[]
    for axis in range(centers.shape[1]):
        events=np.unique(np.r_[centers[:,axis]-widths[:,axis],centers[:,axis]+widths[:,axis]])
        axes.append((events[:-1]+events[1:])/2)
    maximum=1
    for coordinate in product(*axes):
        maximum=max(maximum,int(np.sum(np.all(np.abs(np.asarray(coordinate)-centers)<widths,axis=1))))
    return maximum


@dataclass
class ParametricPUModel:
    sites: object
    weights: object
    poly_coeffs: object
    origins: object
    centers: object
    widths: object
    offset: object
    scale: object
    max_active: int
    cells: list
    diagnostics: dict = field(default_factory=dict)

    def _args(self):
        return (self.sites,self.weights,self.poly_coeffs,self.origins,self.centers,
                self.widths,self.offset,self.scale,self.max_active)

    def evaluate(self,parameters):
        # Traced callers use the global top-k width (patch count): identical values.
        if isinstance(parameters,jax.core.Tracer): return _evaluate(parameters,*self._args())
        return _grouped_parametric(self,parameters)

    def frame(self,parameters):
        if isinstance(parameters,jax.core.Tracer): return _frame(parameters,*self._args())
        return _grouped_parametric(self,parameters,frame=True)


@partial(jit,static_argnums=(3,))
def _rect_members(normalized,centers,widths,capacity):
    """Per-support member indices (ascending, -1 padded), counts and local coordinates."""
    member=jnp.all(jnp.abs(normalized[None,:,:]-centers[:,None,:])<widths[:,None,:],axis=2)  # (P,N)
    counts=jnp.sum(member,axis=1)
    order=jnp.argsort(~member,axis=1,stable=True)[:,:capacity]
    valid=jnp.take_along_axis(member,order,axis=1)
    local=(normalized[order]-centers[:,None,:])/widths[:,None,:]
    return jnp.where(valid,order,-1),counts,jnp.where(valid[:,:,None],local,0.),valid


@jit
def _rect_member_counts(normalized,centers,widths):
    return jnp.sum(jnp.all(jnp.abs(normalized[None,:,:]-centers[:,None,:])<widths[:,None,:],axis=2),axis=1)


@jit
def _affine_ranks(local,valid):
    p=jnp.concatenate((jnp.ones(local.shape[:2]+(1,)),local),axis=2)*valid[:,:,None]
    return jnp.sum(jnp.linalg.svd(p,compute_uv=False)>1e-10,axis=1)


@jit
def _usable_data(local,valid,ids,positions,usable):
    sites=local[usable]; active=valid[usable]; members=ids[usable]
    return sites,active,jnp.where(active[:,:,None],positions[jnp.maximum(members,0)],0.)


@partial(jit,static_argnums=(5,))
def _bucket_data(sites,values,active,g,m,size):
    """Rows g (padded with an out-of-range index) of one size bucket; padded supports get one zero site."""
    rows=jnp.arange(g.shape[0])
    return (take_rows(sites,g)[:,:size],take_rows(values,g)[:,:size],
        take_rows(active,g)[:,:size]|(rows[:,None]>=m)&(jnp.arange(size)[None,:]==0))


@jit
def _put_fits(weights,poly,origins,g,w,pc,o):
    size=w.shape[1]
    return weights.at[g,:size].set(w,mode='drop'),poly.at[g].set(pc,mode='drop'),origins.at[g].set(o,mode='drop')


def _build(parameters,positions,cells,offset,scale,options):
    """Device memberships, ranks and local solves; only cover topology is host metadata."""
    from kernelpack.accelerators.jax_spatial import bucket
    parameters=jnp.asarray(parameters,dtype=float); positions=jnp.asarray(positions,dtype=float)
    normalized=(parameters-offset)/scale
    centers=jnp.stack([(jnp.asarray(lo)+jnp.asarray(hi))/2 for lo,hi in cells])
    widths=jnp.stack([(jnp.asarray(hi)-jnp.asarray(lo))*options.overlap/2 for lo,hi in cells])
    p_pad=bucket(len(cells),8); n_pad=bucket(len(normalized),64)
    norm_dev=pad_device(normalized,n_pad).at[len(normalized):].set(1e300)
    cent_dev=pad_device(centers,p_pad); wid_dev=pad_device(widths,p_pad)
    counts=_rect_member_counts(norm_dev,cent_dev,wid_dev)[:len(cells)]
    nonempty=counts>0
    if not bool(jnp.any(nonempty)): raise ValueError('PU cover contains no geometry data')
    capacity=bucket(int(jnp.max(counts)),1)
    ids,_,local,valid=_rect_members(norm_dev,cent_dev,wid_dev,capacity)
    ranks=_affine_ranks(local,valid)[:len(cells)]
    bad=nonempty&((counts<options.min_sites)|(ranks<normalized.shape[1]+1))
    if bool(jnp.any(bad)):
        cell=int(jnp.argmax(bad)); raise ValueError(f'PU support needs more geometry data: patch {cell}, sites {int(counts[cell])}')
    # One scalar per support controls the Python cover list; fit arrays never leave the device.
    selected=[i for i in range(len(cells)) if bool(nonempty[i])]
    usable=jnp.asarray(selected,dtype=jnp.int32)
    centers=centers[usable]; widths=widths[usable]; cells=[cells[i] for i in selected]
    counts=counts[usable]
    sites,active,values=_usable_data(local,valid,ids,positions,usable)
    weights=jnp.zeros(sites.shape[:2]+(positions.shape[1],)); poly=jnp.zeros((len(cells),sites.shape[2]+1,positions.shape[1]))
    origins=jnp.zeros((len(cells),positions.shape[1]))
    for size,count,group in device_groups(jnp.maximum(counts,8)):
        per=(size+sites.shape[2]+1)**2*8*4
        batch=bucket(max(1,min(count,(1<<31)//per)),1)
        for start in range(0,count,batch):
            amount=min(batch,count-start)
            gi=pad_device(group[start:start+amount],batch,len(cells))
            w,pc,o=_solve_local_fits(*_bucket_data(sites,values,active,gi,amount,size))
            weights,poly,origins=_put_fits(weights,poly,origins,gi,w,pc,o)
    if not bool(jnp.all(jnp.isfinite(weights))): raise ValueError('nonfinite PU geometry fit')
    model=ParametricPUModel(sites,weights,poly,origins,centers,widths,jnp.asarray(offset),jnp.asarray(scale),len(cells),cells)
    model.diagnostics=dict(patch_count=len(cells),support_counts=[int(c) for c in counts],
        padded_capacity=capacity,max_active=model.max_active,kernel='r^3 + Pi_1',
        evaluation='grouped by overlap count; max_active is the patch count (always valid)')
    return model


def fit_parametric_pu(parameters,positions,*,validation_parameters=None,
        validation_positions=None,validation_normals=None,parameter_bounds=None,options=PUOptions(),initial_cells=None):
    """Fit/refine a coordinate-map PU until physical-space indicators pass.

    If no separate validation samples are supplied, hold out one fifth of sites
    while selecting the cover, then refit with all sites. That stopping indicator
    estimates accuracy; it does not certify unseen geometry. Separate samples
    are preferred when the input geometry can supply them.
    """
    parameters=jnp.asarray(parameters,float); positions=jnp.asarray(positions,float)
    if (parameters.ndim!=2 or parameters.shape[1] not in (1,2)
            or positions.shape!=(len(parameters),parameters.shape[1]+1)
            or not jnp.isfinite(parameters).all() or not jnp.isfinite(positions).all()):
        raise ValueError('finite parameters and matching physical coordinate data are required')
    if (bool(jnp.any(jnp.all(jnp.diff(parameters[jnp.lexsort(tuple(parameters[:,i] for i in reversed(range(parameters.shape[1]))))],axis=0)==0,axis=1)))
            or options.initial_grid<1 or not 1<options.overlap<2
            or options.relative_position_tolerance<=0 or options.normal_tolerance<=0
            or options.min_sites<parameters.shape[1]+2 or options.max_refinements<0
            or not 0<options.min_jacobian_ratio<1
            or not all(math.isfinite(value) for value in (
                options.overlap, options.relative_position_tolerance,
                options.normal_tolerance, options.min_jacobian_ratio))):
        raise ValueError('invalid PU data or options')
    if len(parameters)<options.min_sites:
        raise ValueError('PU support needs more geometry data')
    lower,upper=(parameters.min(axis=0),parameters.max(axis=0)) if parameter_bounds is None else map(jnp.asarray,parameter_bounds)
    if (lower.shape!=(parameters.shape[1],) or upper.shape!=lower.shape
            or not jnp.isfinite(lower).all() or not jnp.isfinite(upper).all()):
        raise ValueError('invalid parameter bounds')
    offset=(upper+lower)/2
    scale=(upper-lower)/2
    if jnp.any(scale<=0): raise ValueError('parameter data do not span a parameter domain')
    length=float(2*jnp.linalg.norm(positions-positions.mean(axis=0),axis=1).max())
    if not length>0:
        raise ValueError('geometry data must have nonzero physical extent')
    tolerance=options.relative_position_tolerance*length
    if validation_parameters is None:
        if validation_positions is not None or validation_normals is not None:
            raise ValueError('validation parameters are required')
        check=jnp.arange(2,len(parameters),5)
        train=jnp.asarray([i for i in range(len(parameters)) if i<2 or (i-2)%5],dtype=jnp.int32)
        vp,vx=parameters[check],positions[check]
        fit_parameters,fit_positions=parameters[train],positions[train]
        validation_kind='heldout_data_sites'
    else:
        vp=jnp.asarray(validation_parameters,float); vx=jnp.asarray(validation_positions,float)
        fit_parameters,fit_positions=parameters,positions
        validation_kind='separate_geometry_samples'
    if (vp.ndim!=2 or vp.shape[1]!=parameters.shape[1] or vx.shape!=(len(vp),positions.shape[1])
            or not len(vp) or not jnp.isfinite(vp).all() or not jnp.isfinite(vx).all()):
        raise ValueError('finite matching geometry validation arrays are required')
    if validation_parameters is not None:
        from kernelpack.domain.stencils import select_stencils
        _,nearest_points,valid=select_stencils(parameters,vp,k=1)
        if not bool(jnp.all(valid)): raise ValueError('invalid validation parameter geometry')
        nearest=jnp.linalg.norm(nearest_points[:,0]-vp,axis=1)
        if jnp.all(nearest<=1e-12*jnp.max(scale)):
            raise ValueError('validation must include samples outside the interpolation sites')
    vn=None if validation_normals is None else jnp.asarray(validation_normals,float)
    if vn is not None:
        if vn.shape!=vx.shape or not jnp.isfinite(vn).all() or jnp.any(jnp.linalg.norm(vn,axis=1)==0):
            raise ValueError('finite nonzero validation normals must match positions')
        vn=vn/jnp.linalg.norm(vn,axis=1)[:,None]
    cells=rectangular_cover(-jnp.ones(parameters.shape[1]),jnp.ones(parameters.shape[1]),options.initial_grid) if initial_cells is None else initial_cells
    history=[]; best_score=float("inf")
    for refinement in range(options.max_refinements+1):
        if len(cells)>options.max_patches: raise ValueError('PU refinement exceeded max_patches')
        try: model=_build(fit_parameters,fit_positions,cells,offset,scale,options)
        except ValueError as error:
            raise PUFitError(str(error),history) from error
        prediction=jnp.asarray(model.evaluate(vp)); jac,nr=model.frame(vp)
        jac,nr=jnp.asarray(jac),jnp.asarray(nr)
        error=jnp.linalg.norm(prediction-vx,axis=1)
        singular=jnp.linalg.svd(jnp.nan_to_num(jac),compute_uv=False)
        regular=jnp.isfinite(jac).all(axis=(1,2)) & jnp.isfinite(nr).all(axis=1)
        regular &= singular[:,-1]>options.min_jacobian_ratio*jnp.maximum(singular[:,0],1e-300)
        normal_error=jnp.zeros(len(vp)) if vn is None else jnp.linalg.norm(nr-vn,axis=1)
        failed=(error>tolerance)|(normal_error>options.normal_tolerance)|(~regular)|(~jnp.isfinite(error))
        ratio=float(jnp.min(singular[:,-1]/jnp.maximum(singular[:,0],1e-300)))
        score=max(float(error.max())/tolerance,float(normal_error.max())/options.normal_tolerance,
                  options.min_jacobian_ratio/max(ratio,1e-300))
        if not jnp.isfinite(error).all() or not jnp.isfinite(normal_error).all(): score=float("inf")
        history.append(dict(refinement=refinement,patch_count=len(model.cells),
            max_position_error=float(error.max()),max_normal_error=None if vn is None else float(normal_error.max()),
            min_jacobian_ratio=ratio,indicator_score=None if not jnp.isfinite(score) else score,
            failed_samples=int(failed.sum())))
        if not failed.any():
            if validation_parameters is None: model=_build(parameters,positions,model.cells,offset,scale,options)
            model.diagnostics.update(validation=validation_kind,absolute_position_tolerance=float(tolerance),
                options=vars(options),history=history,validated_samples=len(vp),rbf_degree=3,polynomial_basis_terms=parameters.shape[1]+1)
            return model
        if refinement>0 and score>=best_score:
            raise PUFitError('PU geometric model did not meet physical indicators: refinement did not improve the worst indicator; more geometry data or a better parameter domain is required',history)
        best_score=score
        if refinement==options.max_refinements:
            raise PUFitError(f'PU geometric model did not meet physical indicators: {history[-1]}; tolerance={tolerance}',history)
        z=(vp-offset)/scale
        # Refine supports touching failures, including overlap contributions.
        # Selection depends on geometry indicators, never a PDE convergence rate.
        refined=[]
        for lo,hi in model.cells:
            center=(lo+hi)/2; width=(hi-lo)*options.overlap/2
            if jnp.any(failed & jnp.all(jnp.abs(z-center)<width,axis=1)):
                mid=(lo+hi)/2
                for bits in product((0,1),repeat=parameters.shape[1]):
                    bits=jnp.asarray(bits,bool)
                    refined.append((jnp.where(bits,mid,lo),jnp.where(bits,hi,mid)))
            else: refined.append((lo,hi))
        cells=refined


def stereographic_parameters(positions,origin,*,scales=(1.,1.,1.),pole_axis=0):
    """Generic radial hemisphere coordinates; regular through a vertical rim.

    The origin, positive coordinate scales, and hemisphere axis describe input
    geometry. The face must be a single-valued radial surface about that origin.
    This routine supplies coordinates; it does not fit or sample that surface.
    """
    points=np.asarray(positions,float); origin=np.asarray(origin,float); scales=np.asarray(scales,float)
    if points.ndim!=2 or points.shape[1]!=3 or origin.shape!=(3,) or scales.shape!=(3,) or np.any(scales<=0):
        raise ValueError('invalid radial coordinate geometry')
    local=(points-origin)/scales
    length=np.linalg.norm(local,axis=1)
    if not np.isfinite(local).all() or np.any(length<=0): raise ValueError('invalid radial geometry sites')
    unit=local/length[:,None]
    if pole_axis not in (0,1,2) or np.any(unit[:,pole_axis]<-1e-10):
        raise ValueError('geometry must lie in the specified radial hemisphere')
    free=[i for i in range(3) if i!=pole_axis]
    return unit[:,free]/(1+unit[:,pole_axis,None])


@dataclass
class SurfaceParameterization:
    method: str
    origin: object
    basis: object
    scales: object

    def project(self,positions):
        local=(np.asarray(positions)-self.origin)@self.basis
        if self.method=='plane': return local[:,:2]
        return stereographic_parameters(local,np.zeros(3),scales=self.scales)


def parameterize_curve_piece(positions,boundary_loops,*,min_projection_ratio=.25):
    """Use a regular PCA projection for an ordered open curve, or refuse it.

    No arc-length approximation is mixed into the coordinate map. Curves that
    fold in this projection need caller-supplied regular parameters.
    """
    points=np.asarray(positions,float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<2 or not np.isfinite(points).all():
        raise ValueError('finite ordered 2D curve sites are required')
    loops=[np.asarray(loop,float) for loop in boundary_loops]
    if len(loops)!=1 or loops[0].shape!=(2,2):
        raise ValueError('an open curve needs its two physical endpoints')
    origin=points.mean(axis=0)
    _,_,vectors=np.linalg.svd(points-origin,full_matrices=False)
    basis=vectors[0,:,None]
    if float((points[-1]-points[0])@basis[:,0])<0: basis=-basis
    projected=(points-origin)@basis
    if np.any(np.diff(projected[:,0])<=0) or float(_sampled_projection_ratio(jnp.asarray(points),jnp.asarray(projected)))<min_projection_ratio:
        raise ValueError('curve projection folds or is poorly separated; supply a regular parameter domain')
    return SurfaceParameterization('plane',origin,basis,np.ones(1))


@jit
def _sampled_projection_ratio(points,parameters):
    ids=jnp.arange(len(points))
    def minimum(i):
        physical=jnp.sum((points-points[i])**2,axis=1)
        parametric=jnp.sum((parameters-parameters[i])**2,axis=1)
        return jnp.min(jnp.where(ids==i,jnp.inf,parametric/jnp.maximum(physical,1e-300)))
    return jnp.sqrt(jnp.min(lax.map(minimum,ids,batch_size=64)))


def parameterize_surface_piece(positions,boundary_loops,*,normals=None,min_cosine=.25):
    """One shared policy for every open smooth face: plane, then radial chart.

    Checks use surface geometry and ordered boundaries, never an example name.
    A radial chart is attempted only for a face lying in one hemisphere about
    its boundary centroid with nongrazing radial normals. Unsupported input
    requires a supplied regular parameter domain rather than a silent fallback.
    """
    from .core import project_to_best_fit_plane
    points=np.asarray(positions,float)
    loops=[np.asarray(loop,float) for loop in boundary_loops]
    if (not loops or points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all()
            or any(loop.ndim!=2 or loop.shape[1]!=3 or len(loop)<3 or not np.isfinite(loop).all() for loop in loops)):
        raise ValueError('finite face data and ordered physical boundary loops are required')
    if normals is None:
        from scipy.spatial import cKDTree
        _,ids=cKDTree(points).query(points,k=min(16,len(points)))
        nearby=jnp.asarray(points[ids]); shifted=nearby-nearby.mean(axis=1,keepdims=True)
        covariance=jnp.einsum('nki,nkj->nij',shifted,shifted)
        _,vectors=jnp.linalg.eigh(covariance)
        normals=np.asarray(vectors[:,:,0])
        oriented=False
    else:
        normals=np.asarray(normals,float)
        oriented=True
    if normals.shape!=points.shape or not np.isfinite(normals).all() or np.any(np.linalg.norm(normals,axis=1)==0):
        raise ValueError('invalid face normals')
    normals=normals/np.linalg.norm(normals,axis=1)[:,None]
    _,origin,basis=project_to_best_fit_plane(jnp.asarray(points))
    basis=np.array(basis,copy=True); origin=np.asarray(origin)
    plane_normal=np.cross(basis[:,0],basis[:,1])
    cosines=normals@plane_normal
    if (cosines.mean()<0 if oriented else (points.mean(axis=0)-np.vstack(loops).mean(axis=0))@plane_normal<0):
        basis[:,1]*=-1; cosines*=-1
    measure=cosines if oriented else np.abs(cosines)
    if float(measure.min())>=min_cosine:
        projected=(points-origin)@basis
        if float(_sampled_projection_ratio(jnp.asarray(points),jnp.asarray(projected)))>=min_cosine:
            return SurfaceParameterization('plane',origin,basis,np.ones(2))
    boundary=np.vstack(loops); origin=boundary.mean(axis=0)
    _,_,tangent=project_to_best_fit_plane(jnp.asarray(boundary))
    tangent=np.array(tangent,copy=True)
    pole=np.cross(tangent[:,0],tangent[:,1])
    if pole@(points.mean(axis=0)-origin)<0:
        tangent[:,1]*=-1; pole*=-1
    frame=np.column_stack((pole,tangent))
    local=(points-origin)@frame
    scales=np.max(np.abs(local),axis=0)
    if np.any(scales<=1e-12*max(float(scales.max()),1)):
        raise ValueError('surface needs a supplied regular parameter domain')
    radial=local/scales
    length=np.linalg.norm(radial,axis=1)
    if np.any(length<=1e-12) or np.any(local[:,0]<-1e-10*scales[0]):
        raise ValueError('surface needs a supplied regular parameter domain')
    # Transform normals contravariantly under the coordinate scaling.
    transformed=(normals@frame)*scales
    transformed/=np.linalg.norm(transformed,axis=1)[:,None]
    cosine=np.sum(transformed*radial/length[:,None],axis=1)
    orientation=1 if cosine.mean()>=0 else -1
    if (np.min(orientation*cosine) if oriented else np.abs(cosine).min())<min_cosine:
        raise ValueError('surface needs a supplied regular parameter domain')
    if oriented and orientation<0: frame[:,2]*=-1
    chart=SurfaceParameterization('radial_hemisphere',origin,frame,scales)
    projected=chart.project(points)
    if len(np.unique(np.round(projected,12),axis=0))!=len(projected):
        raise ValueError('radial projection collides; a regular parameter domain is required')
    if float(_sampled_projection_ratio(jnp.asarray(radial),jnp.asarray(projected)))<.05:
        raise ValueError('radial projection is poorly separated; a regular parameter domain is required')
    return chart
