"""Footprint-preserving sampling of planar faces for the 3D hierarchy."""
import numpy as np
import math
import jax.numpy as jnp
from jax import jit, lax

from kernelpack.accelerators import WarpBridsonOptions, warp_available, warp_bridson_sample
from kernelpack.geometry import EmbeddedSurface
from kernelpack.geometry.core import _poisson_disk_sample_box_2d


@jit
def points_in_polygon(points, outline):
    """Even-odd test for an ordered simple closed outline; includes its boundary.

    Vectorize bounded batches of candidates against all outline edges. This
    avoids a device launch per edge and bounds temporary storage independently
    of the total number of face candidates.
    """
    a = outline
    b = jnp.roll(outline, -1, axis=0)
    edge = b-a
    dy = edge[:, 1]
    safe_dy = jnp.where(dy != 0, dy, 1.)
    cross_tol = 1e-12*jnp.maximum(1., jnp.linalg.norm(edge, axis=1))

    def contains(point):
        rel = point-a
        cross = edge[:,0]*rel[:,1]-edge[:,1]*rel[:,0]
        on = jnp.abs(cross) <= cross_tol
        on &= jnp.sum(rel*edge, axis=1) >= -1e-12
        on &= jnp.sum((point-b)*edge, axis=1) <= 1e-12
        crosses = (a[:,1] > point[1]) != (b[:,1] > point[1])
        crosses &= point[0] < a[:,0]+(point[1]-a[:,1])*edge[:,0]/safe_dy
        return (jnp.sum(crosses) % 2 != 0) | jnp.any(on)
    return lax.map(contains, points, batch_size=256)


@jit
def _points_in_polygon_masked(points, outline, edge_valid):
    a = outline
    b = jnp.roll(outline, -1, axis=0)
    edge = b-a
    dy = edge[:, 1]
    safe_dy = jnp.where(dy != 0, dy, 1.)
    cross_tol = 1e-12*jnp.maximum(1., jnp.linalg.norm(edge, axis=1))

    def contains(point):
        rel = point-a
        cross = edge[:,0]*rel[:,1]-edge[:,1]*rel[:,0]
        on = jnp.abs(cross) <= cross_tol
        on &= jnp.sum(rel*edge, axis=1) >= -1e-12
        on &= jnp.sum((point-b)*edge, axis=1) <= 1e-12
        crosses = (a[:,1] > point[1]) != (b[:,1] > point[1])
        crosses &= point[0] < a[:,0]+(point[1]-a[:,1])*edge[:,0]/safe_dy
        return (jnp.sum(crosses & edge_valid) % 2 != 0) | jnp.any(on & edge_valid)
    return lax.map(contains, points, batch_size=256)


def points_in_polygon_device(points, outline):
    """``points_in_polygon`` with power-of-two padded shapes (one compile per bucket); device mask.

    The outline is closed by its own last edge; padding edges are masked out.
    """
    from kernelpack.accelerators.jax_spatial import bucket, pad_rows_device
    pts = jnp.asarray(points, float)
    loop = jnp.asarray(outline, float)
    if pts.shape[0] == 0:
        return jnp.zeros(0, bool)
    m = len(loop)
    m_pad = bucket(m + 1, 8)
    # Closing edge (last -> first) is explicit; the padded tail repeats the first
    # vertex and its zero-length edges are masked.
    closed = jnp.vstack([loop, loop[:1]])
    padded = pad_rows_device(closed,m_pad,0.).at[m+1:].set(loop[0])
    edge_valid = jnp.arange(m_pad)<m
    n = pts.shape[0]
    inside = _points_in_polygon_masked(pad_rows_device(pts, bucket(n, 256), 0.0), jnp.asarray(padded), jnp.asarray(edge_valid))
    return inside[:n]


def points_in_polygon_bucketed(points, outline):
    """Host-mask wrapper of ``points_in_polygon_device``."""
    return np.array(points_in_polygon_device(points, outline))


def sample_planar_face(outline, radius, *, seed=0, smooth=True,
                       supersample_fac=2, backend="auto"):
    """Supersample a PDS box and clip to the actual face; assembly thins in 3D.

    A nonsmooth outline is an ordered closed polyline (do not repeat the last
    vertex). Smooth outlines are evaluated with the periodic curve model.
    Input-data density controls geometry accuracy, never the output spacing.
    """
    curve = jnp.asarray(outline, dtype=float)
    if curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 3:
        raise ValueError("a planar face needs an ordered 2D closed outline")
    if radius <= 0 or supersample_fac < 1 or not bool(jnp.isfinite(curve).all()):
        raise ValueError("finite geometry, positive spacing and supersampling are required")
    if float(jnp.linalg.norm(curve[0] - curve[-1])) < 1e-12:
        curve = curve[:-1]
    if len(curve) < 3 or bool(jnp.any(jnp.linalg.norm(jnp.roll(curve,-1,axis=0)-curve,axis=1)<1e-14)):
        raise ValueError("outline has too few vertices or consecutive duplicates")
    candidate_radius = float(radius) / math.sqrt(float(supersample_fac))
    if smooth:
        perimeter = float(jnp.linalg.norm(jnp.roll(curve,-1,axis=0)-curve,axis=1).sum())
        surface = EmbeddedSurface()
        surface.set_data_sites(jnp.asarray(curve))
        surface.build_closed_geometric_model_ps(2, candidate_radius, len(curve),
            ne=max(len(curve), int(np.ceil(4 * perimeter / candidate_radius))), method=2)
        curve = jnp.asarray(surface.get_sample_sites())
    # Oversample each piece of the supplied outline without smoothing its joins.
    lengths = jnp.linalg.norm(jnp.roll(curve,-1,axis=0)-curve,axis=1)
    boundary = jnp.vstack([a+jnp.arange(max(1,int(jnp.ceil(length/candidate_radius))))[:,None]
        /max(1,int(jnp.ceil(length/candidate_radius)))*(b-a)
        for a,b,length in zip(curve,jnp.roll(curve,-1,axis=0),lengths)])
    mins, maxs = curve.min(axis=0), curve.max(axis=0)
    if bool(jnp.any(maxs <= mins)):
        raise ValueError("face outline must span a 2D area")
    if backend=="auto" and curve.device.platform=="gpu":
        backend="gpu"
    if backend == "gpu":
        # Dart throwing as a greedy independent set on device, restricted to the
        # face; the outline test is the acceptance predicate.
        from kernelpack.accelerators.jax_spatial import poisson_disk_sample
        inside = poisson_disk_sample(mins, maxs, lambda p: jnp.full(p.shape[0], candidate_radius), candidate_radius,
                                     seed=int(seed), accept=lambda p: points_in_polygon_device(p, curve),
                                     require_gpu=curve.device.platform=="gpu")
        return jnp.vstack([jnp.asarray(boundary), inside])
    if backend == "warp" or (backend == "auto" and warp_available()):
        opts = WarpBridsonOptions(candidate_radius, candidate_radius, 30, int(seed),
            "fixed_radius", 1., 0., np.zeros((0, 2)))
        candidates = warp_bridson_sample(mins, maxs, opts)
    elif backend in ("auto", "cpu", "python", "numpy"):
        candidates = _poisson_disk_sample_box_2d(mins, maxs, candidate_radius, seed=seed, attempts=30)
    else:
        raise ValueError("planar face backend must be auto, warp, python or numpy")
    candidates = jnp.asarray(candidates)
    keep = points_in_polygon(candidates, jnp.asarray(curve))
    return jnp.vstack([jnp.asarray(boundary), candidates[keep]])


def sample_parametric_curve(model,outlines,radius,*,supersample_fac=2):
    """Supersample an open parameter interval using its physical tangent metric.

    Endpoints remain first/last; feature-priority physical thinning is done by
    piecewise assembly, as for faces. No topology or example names are inferred.
    """
    interval=jnp.asarray(outlines[0],float)
    if len(outlines)!=1 or interval.shape!=(2,1) or radius<=0 or supersample_fac<1:
        raise ValueError('an open PU curve needs one increasing parameter interval')
    lo,hi=interval[:,0]
    if hi<=lo: raise ValueError('curve parameter interval must increase')
    jac,_=model.frame(jnp.linspace(lo,hi,257)[:,None])
    largest=float(jnp.max(jnp.linalg.norm(jac[:,:,0],axis=1)))
    if not math.isfinite(largest) or largest<=0: raise ValueError('invalid PU curve metric')
    count=max(2,int(jnp.ceil((hi-lo)*1.1*largest*supersample_fac/radius))+1)
    return jnp.linspace(lo,hi,count)[:,None]


def sample_parametric_face(model,outlines,radius,*,seed=0,backend="auto",supersample_fac=2):
    """Metric-aware PDS in a trimmed parameter domain, then 3D assembly.

    PU patches are computational supports; ordered loops define the physical
    face. Holes and concavities are never replaced by a convex hull.
    """
    loops=[jnp.asarray(loop,float) for loop in outlines]
    if not loops: raise ValueError("a physical face needs a parameter boundary")
    if radius<=0 or supersample_fac<1: raise ValueError("invalid face spacing")
    mins,maxs=loops[0].min(axis=0),loops[0].max(axis=0)
    u,v=jnp.meshgrid(jnp.linspace(mins[0],maxs[0],25),jnp.linspace(mins[1],maxs[1],25))
    probe=jnp.vstack((jnp.column_stack((u.ravel(),v.ravel())),jnp.vstack(loops)))
    keep=points_in_polygon_device(probe,loops[0])
    for hole in loops[1:]: keep &= ~points_in_polygon_device(probe,hole)
    from kernelpack.accelerators.jax_spatial import compact_rows
    jac,_=model.frame(compact_rows(keep,probe))
    largest=float(jnp.max(jnp.linalg.svd(jac,compute_uv=False)[:,0]))
    if not math.isfinite(largest) or largest<=0: raise ValueError("invalid PU parameter metric")
    parameter_radius=radius/(1.1*largest)
    candidates=sample_planar_face(loops[0],parameter_radius,seed=seed,smooth=False,
        supersample_fac=supersample_fac,backend=backend)
    for hole in loops[1:]:
        from kernelpack.accelerators.jax_spatial import compact_rows
        candidates=compact_rows(~points_in_polygon_device(candidates,hole),candidates)
        lengths=jnp.linalg.norm(jnp.roll(hole,-1,axis=0)-hole,axis=1)
        edges=[]; step=parameter_radius/math.sqrt(supersample_fac)
        for a,b,length in zip(hole,jnp.roll(hole,-1,axis=0),lengths):
            count=max(1,int(jnp.ceil(length/step)))
            edges.append(a+jnp.arange(count)[:,None]/count*(b-a))
        candidates=jnp.vstack((candidates,jnp.vstack(edges)))
    return candidates
