"""Geometry-independent enrichment of CFPU input sites near known sharp features."""
import numpy as np
import jax.numpy as jnp
from jax import jit, lax
from scipy.spatial import cKDTree


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
    """``_off_outline`` with power-of-two padded points and edges."""
    from kernelpack.accelerators.jax_spatial import bucket, pad_rows
    pts=np.asarray(points,float); loop=np.asarray(outline,float)
    if not len(pts): return np.zeros(0,bool)
    m=len(loop); m_pad=bucket(m,8)
    start=pad_rows(loop,m_pad,0.); end=pad_rows(np.roll(loop,-1,axis=0),m_pad,0.)
    valid=np.arange(m_pad)<m
    tolerance=1e-12*max(1.,float(np.linalg.norm(np.ptp(loop,axis=0))))
    out=_off_outline_masked(jnp.asarray(pad_rows(pts,bucket(len(pts),256),0.)),jnp.asarray(start),jnp.asarray(end),
        jnp.asarray(valid),tolerance)
    return np.array(out)[:len(pts)]


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
    points=np.asarray(surface.xb_uniform); normals=np.asarray(surface.nrmls_uniform)
    # The features of the assembled boundary: supplied, or derived from the trimming loops.
    derived=getattr(surface,'derived_features',())
    features=[np.asarray(surface.feature_sites)] if len(surface.feature_sites) else (
        [np.asarray(derived[0])] if len(derived) and len(derived[0]) else [])
    diagnostic=dict(factor=float(factor),bandwidth=float(bandwidth),baseline_count=len(points),
        added_count=0,per_piece_added={},new_normals='incident smooth-piece normals',
        original_feature_constraints='preserved unchanged')
    if factor==1 or not features: return points,normals,diagnostic
    h=float(surface.sep_rad)
    if h<=0: raise ValueError('feature enrichment requires an assembled physical separation radius')
    from kernelpack.accelerators.jax_spatial import nearest_within
    feature_points=np.unique(np.vstack(features),axis=0)
    extra=[]; extra_normals=[]; owners=[]
    for index,segment in enumerate(surface.segments):
        model=segment.geom_model; kind=model.get('type')
        if kind=='parametric-pu':
            pu=model['model']; loops=model['outlines']
            if segment.surf_dim==1:
                uv=sample_parametric_curve(pu,loops,h/factor,supersample_fac=2)
                interval=np.asarray(loops[0])[:,0]
                uv=uv[(uv[:,0]>interval[0]+1e-12)&(uv[:,0]<interval[1]-1e-12)]
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
        uv=np.asarray(uv); xyz=np.asarray(xyz)
        keep=nearest_within(xyz,feature_points,bandwidth*h)[0]<bandwidth*h
        uv=uv[keep]; xyz=xyz[keep]
        if kind!='spherical-pu' and uv.shape[1]==2:
            off=np.ones(len(uv),dtype=bool)
            for loop in loops: off &= _off_outline_bucketed(uv,np.asarray(loop))
            uv=uv[off]; xyz=xyz[off]
        if not len(uv): continue
        if kind in ('parametric-pu','spherical-pu'):
            _,nr=pu.frame(uv); nr*=model.get('normal_sign',1.)
        else: nr=jnp.broadcast_to(jnp.asarray(model['normal']),xyz.shape)
        extra.append(np.asarray(xyz)); extra_normals.append(np.asarray(nr))
        owners.extend([index]*len(uv))
    if not extra: return points,normals,diagnostic
    candidates=np.vstack([points,*extra]); values=np.vstack([normals,*extra_normals])
    priority=jnp.r_[jnp.ones(len(points)),jnp.zeros(len(candidates)-len(points))]
    keep=np.asarray(_assembled_boundary_keep_mask(jnp.asarray(candidates),priority,h/factor))
    if not keep[:len(points)].all(): raise ValueError('feature enrichment changed baseline fit sites')
    retained=np.array(owners)[keep[len(points):]]
    diagnostic.update(added_count=int(keep[len(points):].sum()),
        per_piece_added={int(i):int((retained==i).sum()) for i in np.unique(retained)},
        fit_spacing=h/factor,feature_band=bandwidth*h,candidate_count=len(candidates))
    return candidates[keep],values[keep],diagnostic
