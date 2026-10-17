"""Geometry-independent enrichment of CFPU input sites near known sharp features."""
import numpy as np
import jax.numpy as jnp
from jax import jit, lax


@jit
def _off_outline(points, outline):
    end=jnp.roll(outline,-1,axis=0); edge=end-outline
    length2=jnp.sum(edge*edge,axis=1)
    tolerance=1e-12*jnp.maximum(1.,jnp.linalg.norm(jnp.ptp(outline,axis=0)))
    def interior(p):
        t=jnp.clip(jnp.sum((p-outline)*edge,axis=1)/jnp.maximum(length2,1e-300),0.,1.)
        distance2=jnp.sum((p-outline-t[:,None]*edge)**2,axis=1)
        return jnp.min(distance2)>tolerance*tolerance
    return lax.map(interior,points,batch_size=128)


@jit
def _off_outline_masked(points, start, end, edge_valid, tolerance):
    edge=end-start
    length2=jnp.sum(edge*edge,axis=1)
    def interior(p):
        t=jnp.clip(jnp.sum((p-start)*edge,axis=1)/jnp.maximum(length2,1e-300),0.,1.)
        distance2=jnp.where(edge_valid,jnp.sum((p-start-t[:,None]*edge)**2,axis=1),jnp.inf)
        return jnp.min(distance2)>tolerance*tolerance
    return lax.map(interior,points,batch_size=128)


def _off_outline_bucketed(points, outline):
    """``_off_outline`` with power-of-two padded points and edges; device mask."""
    from kernelpack.accelerators.jax_spatial import bucket, pad_rows, pad_rows_device
    pts=jnp.asarray(points,float); loop=np.asarray(outline,float)
    if not pts.shape[0]: return jnp.zeros(0,bool)
    m=len(loop); m_pad=bucket(m,8)
    start=pad_rows(loop,m_pad,0.); end=pad_rows(np.roll(loop,-1,axis=0),m_pad,0.)
    valid=np.arange(m_pad)<m
    tolerance=1e-12*max(1.,float(np.linalg.norm(np.ptp(loop,axis=0))))
    out=_off_outline_masked(pad_rows_device(pts,bucket(pts.shape[0],256),0.),jnp.asarray(start),jnp.asarray(end),
        jnp.asarray(valid),tolerance)
    return out[:pts.shape[0]]


def feature_graded_cfpu_sites(surface, *, factor=2., bandwidth=2., seed=11, backend='auto'):
    """Preserve every original fit site and retain finer samples in a sharp band.

    Extra samples lie in the interior of their incident smooth piece and carry
    that piece's normal. Existing edge/corner interpolation sites and their
    original representative normals are retained unchanged for a density-only
    comparison. This routine does not alter PDE boundary or interior nodes.
    """
    from kernelpack.nodes.hierarchical import sample_parametric_curve, sample_parametric_face, sample_planar_face
    from .core import _assembled_boundary_keep_mask
    if not np.isfinite(factor) or factor<1 or not np.isfinite(bandwidth) or bandwidth<=0:
        raise ValueError('finite feature factor >=1 and positive bandwidth required')
    from kernelpack.accelerators.jax_spatial import compact_rows, nearest_within
    points=jnp.asarray(surface.xb_uniform,float); normals=jnp.asarray(surface.nrmls_uniform,float)
    # The features of the assembled boundary: supplied, or derived from the trimming loops.
    derived=getattr(surface,'derived_features',())
    features=(jnp.asarray(surface.feature_sites,float) if len(surface.feature_sites) else
        jnp.asarray(derived[0],float) if len(derived) and len(derived[0]) else None)
    diagnostic=dict(factor=float(factor),bandwidth=float(bandwidth),baseline_count=int(points.shape[0]),
        added_count=0,per_piece_added={},new_normals='incident smooth-piece normals',
        original_feature_constraints='preserved unchanged')
    if factor==1 or features is None: return points,normals,diagnostic
    h=float(surface.sep_rad)
    if h<=0: raise ValueError('feature enrichment requires an assembled physical separation radius')
    # Only distances to the feature set are used: duplicates and order do not matter.
    feature_points=features
    extra=[]; extra_normals=[]; owners=[]
    for index,segment in enumerate(surface.segments):
        model=segment.geom_model; kind=model.get('type')
        if kind=='parametric-pu':
            pu=model['model']; loops=model['outlines']
            if segment.surf_dim==1:
                uv=jnp.asarray(sample_parametric_curve(pu,loops,h/factor,supersample_fac=2))
                interval=np.asarray(loops[0])[:,0]
                uv=compact_rows((uv[:,0]>interval[0]+1e-12)&(uv[:,0]<interval[1]-1e-12),uv)
            else:
                uv=sample_parametric_face(pu,loops,h/factor,seed=seed+index,
                    backend=backend,supersample_fac=2)
            xyz=pu.evaluate(uv)
        elif kind=='spherical-pu':
            # A closed piece has no trimming loops; only the band test below applies.
            from .spherical_pu import sample_sphere_piece
            pu=model['model']; loops=()
            uv=sample_sphere_piece(pu,h/factor,seed=seed+index,supersample_fac=2)
            xyz=pu.evaluate(uv)
        elif kind=='affine-face':
            loops=model['outlines']
            uv=sample_planar_face(loops[0],h/factor,seed=seed+index,
                smooth=False,backend=backend,supersample_fac=2)
            xyz=jnp.asarray(model['origin'])+uv@jnp.asarray(model['basis']).T
        else: raise ValueError('feature enrichment requires shared PU or affine pieces')
        # Device nearest-feature distance (exact within the band), outline tests
        # and final thinning.
        uv=jnp.asarray(uv,float); xyz=jnp.asarray(xyz,float)
        keep=nearest_within(xyz,feature_points,bandwidth*h)[0]<bandwidth*h
        if kind!='spherical-pu' and uv.shape[1]==2:
            for loop in loops: keep&=_off_outline_bucketed(uv,np.asarray(loop))
        uv,xyz=compact_rows(keep,uv,xyz)
        if not uv.shape[0]: continue
        if kind in ('parametric-pu','spherical-pu'):
            _,nr=pu.frame(uv); nr*=model.get('normal_sign',1.)
        else: nr=jnp.broadcast_to(jnp.asarray(model['normal']),xyz.shape)
        extra.append(xyz); extra_normals.append(jnp.asarray(nr,float))
        owners.append(np.full(uv.shape[0],index))
    if not extra: return points,normals,diagnostic
    candidates=jnp.vstack([points,*extra]); values=jnp.vstack([normals,*extra_normals])
    n_base=int(points.shape[0])
    priority=jnp.r_[jnp.ones(n_base),jnp.zeros(candidates.shape[0]-n_base)]
    keep=_assembled_boundary_keep_mask(candidates,priority,h/factor)
    if not bool(jnp.all(keep[:n_base])): raise ValueError('feature enrichment changed baseline fit sites')
    retained=np.asarray(jnp.bincount(jnp.asarray(np.concatenate(owners)),weights=keep[n_base:].astype(int),
        length=len(surface.segments)))
    diagnostic.update(added_count=int(retained.sum()),
        per_piece_added={int(i):int(c) for i,c in enumerate(retained) if c},
        fit_spacing=h/factor,feature_band=bandwidth*h,candidate_count=int(candidates.shape[0]))
    points,normals=compact_rows(keep,candidates,values)
    return points,normals,diagnostic
