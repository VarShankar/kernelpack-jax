"""Smooth closed curves and surfaces as one piece over the sphere, with a partition of unity over caps.

A smooth closed curve (d = 2) or surface (d = 3) is given by samples y_k with parameters u_k on the unit
sphere S^{d-1}: the directions (y_k - c)/|y_k - c| about a center c (the spherical parametrization of
Shankar, Wright, Kirby and Fogelson 2018, valid for star-shaped boundaries), or any other parameters the
caller supplies (for example latitude and longitude, see latlong_to_directions). The coordinate map
x : S^{d-1} -> R^d is the partition of unity

    x(u) = sum_j psi_j(u) s_j(u) / sum_j psi_j(u),

over spherical caps K_j = {u : u . c_j > cos alpha_j}. The weight psi_j is the C2 Wendland function of the
chordal distance |u - c_j| / (2 sin(alpha_j / 2)). The local fit s_j is a cubic PHS interpolant with linear
augmentation in the scaled gnomonic coordinates

    z_j(u) = E_j u / (tan(alpha_j) u . c_j),

where the rows of E_j are an orthonormal basis of the tangent space at c_j; z_j maps K_j onto the unit ball.
This is the global SBF interpolant of the 2018 paper localized to caps, so the fits have the same size and
cost as those of the parametric partition of unity on rectangles (parametric_pu.py). There is no trimming
loop and no seam; the piece has no features.

Cover. Each cap has a covering radius beta_j and support radius alpha_j = overlap * beta_j. The initial caps
are the four quadrant arcs (d = 2) or the twelve vertices of the icosahedron (d = 3), whose covering radii
cover the sphere. Refinement replaces a cap by caps of half its covering radius: the two halves of an arc,
or the center and six points at angular distance (sqrt(3)/2) beta_j for a surface cap, which cover a planar
disk exactly; the overlap absorbs the curvature of the sphere. Validation and refinement follow
fit_parametric_pu: held-out (or separate) samples, a relative position tolerance, optional normals, and a
lower bound on the ratio of the singular values of the Jacobian.

Frames. Derivatives are taken along an orthonormal tangent basis (t_1, ..., t_{d-1}) at u with
(t_1, t_2, u) right handed (d = 3) or t_1 = u rotated by +pi/2 (d = 2). With this orientation the normal
J t_1 x J t_2 (d = 3), or J t_1 rotated by -pi/2 (d = 2), points away from c for an orientation-preserving
map; for directions about c this is out of the enclosed region.
"""
from dataclasses import dataclass, field
from functools import partial

import math
import jax
import jax.numpy as jnp
from jax import jit, lax

from .parametric_pu import PUFitError, _solve_local_fits


@dataclass(frozen=True)
class SphericalPUOptions:
    overlap: float = 1.5
    relative_position_tolerance: float = 1e-3
    normal_tolerance: float = .05
    min_jacobian_ratio: float = .05
    min_sites: int = 12
    max_refinements: int = 4
    max_caps: int = 4096


# ---------------------------------------------------------------- sphere helpers
def latlong_to_directions(latlong):
    """Unit vectors from (latitude, longitude) in radians, (n, 2) -> (n, 3); angles (n,) or (n, 1) -> (n, 2)."""
    a = jnp.asarray(latlong, float)
    if a.ndim == 1 or a.shape[1] == 1:
        t = a.reshape(-1)
        return jnp.column_stack([jnp.cos(t), jnp.sin(t)])
    lat, lon = a[:, 0], a[:, 1]
    return jnp.column_stack([jnp.cos(lat) * jnp.cos(lon), jnp.cos(lat) * jnp.sin(lon), jnp.sin(lat)])


def directions_about(positions, center):
    """Unit directions (y - c)/|y - c| of the samples about the center."""
    s = jnp.asarray(positions, float) - jnp.asarray(center, float)
    r = jnp.linalg.norm(s, axis=1)
    if jnp.any(r == 0):
        raise ValueError("a sample coincides with the center")
    return s / r[:, None]


def tangent_basis(c):
    """Orthonormal tangent basis E (rows) at unit vectors c: (m, d-1, d), with (e_1, e_2, c) right handed."""
    c = jnp.asarray(c, float)
    if c.shape[1] == 2:
        return jnp.stack([-c[:, 1], c[:, 0]], axis=1)[:, None, :]
    a = jnp.eye(3)[jnp.argmin(jnp.abs(c), axis=1)]
    t1 = a - jnp.sum(a * c, axis=1, keepdims=True) * c
    t1 = t1 / jnp.linalg.norm(t1, axis=1, keepdims=True)
    t2 = jnp.cross(c, t1)
    return jnp.stack([t1, t2], axis=1)


def _tangent_basis_point(u):
    """Traceable tangent basis at one unit vector u, as in tangent_basis."""
    if u.shape[0] == 2:
        return jnp.stack([-u[1], u[0]])[None, :]
    a = jnp.eye(3)[jnp.argmin(jnp.abs(u))]
    t1 = a - jnp.dot(a, u) * u
    t1 = t1 / jnp.linalg.norm(t1)
    return jnp.stack([t1, jnp.cross(u, t1)])


def initial_caps(dim):
    """Centers and covering radii of the initial cover of S^{d-1}."""
    if dim == 2:
        centers = jnp.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
        return centers, jnp.full(4, jnp.pi / 4)
    g = (1 + jnp.sqrt(5)) / 2
    v = []
    for a in (-1., 1.):
        for b in (-g, g):
            v += [(0., a, b), (a, b, 0.), (b, 0., a)]
    v = jnp.asarray(v)
    v = v / jnp.linalg.norm(v, axis=1, keepdims=True)
    # covering radius: angle from a vertex to the centroid of an incident face (all faces are equal)
    i = 0
    nbr = jnp.argsort(v @ v[i])[::-1][1:6]
    j = nbr[0]
    k = nbr[1:][jnp.argmax((v[nbr[1:]] @ v[j]) > .4)]
    centroid = (v[i] + v[j] + v[k]) / jnp.linalg.norm(v[i] + v[j] + v[k])
    beta = jnp.arccos(jnp.clip(centroid @ v[i], -1, 1))
    return v, jnp.full(len(v), beta * (1 + 1e-6))


def split_caps(centers, betas, indices):
    """Replace the caps in indices by caps of half the covering radius (see the module docstring)."""
    centers, betas = jnp.asarray(centers, float), jnp.asarray(betas, float)
    out_c, out_b = [], []
    for i, (c, b) in enumerate(zip(centers, betas)):
        if i not in indices:
            out_c.append(c[None]); out_b.append(b[None]); continue
        if len(c) == 2:
            t = tangent_basis(c[None])[0, 0]
            kids = jnp.array([jnp.cos(b / 2) * c + s * jnp.sin(b / 2) * t for s in (-1., 1.)])
        else:
            e1, e2 = tangent_basis(c[None])[0]
            d = jnp.sqrt(3) / 2 * b
            ring = [jnp.cos(d) * c + jnp.sin(d) * (jnp.cos(th) * e1 + jnp.sin(th) * e2)
                    for th in jnp.arange(6) * jnp.pi / 3]
            kids = jnp.vstack([c[None], jnp.stack(ring)])
        out_c.append(kids / jnp.linalg.norm(kids, axis=1, keepdims=True)); out_b.append(jnp.full(len(kids), b / 2))
    return jnp.vstack(out_c), jnp.concatenate(out_b)


@partial(jit, static_argnames=("order",))
def gauss_legendre(order):
    """Gauss--Legendre rule by the symmetric Jacobi eigensystem on the device."""
    index = jnp.arange(1, order, dtype=jnp.float64)
    off = index / jnp.sqrt(4*index*index-1)
    jacobi = jnp.diag(off, 1) + jnp.diag(off, -1)
    nodes, vectors = jnp.linalg.eigh(jacobi)
    return nodes, 2*vectors[0]**2


def sphere_quadrature(dim, order):
    """Product quadrature on S^{d-1}: Gauss-Legendre in u_3 times the trapezoidal rule in longitude (d = 3),
    or the trapezoidal rule with 4 * order points (d = 2). Returns unit vectors and weights (sum = |S^{d-1}|)."""
    if dim not in (2,3) or order<1:
        raise ValueError("positive quadrature order and dimension2 or3 required")
    if dim == 2:
        n = 4 * order
        t = 2 * jnp.pi * jnp.arange(n) / n
        return jnp.column_stack([jnp.cos(t), jnp.sin(t)]), jnp.full(n, 2 * jnp.pi / n)
    if dim != 3 or order < 1:
        raise ValueError("positive quadrature order and dimension2 or3 required")
    z, wz = gauss_legendre(order)
    n = 2 * order
    lon = 2 * jnp.pi * (jnp.arange(n) + .5) / n
    zz, ll = jnp.meshgrid(z, lon, indexing="ij")
    rho = jnp.sqrt(1 - zz ** 2)
    u = jnp.stack([rho * jnp.cos(ll), rho * jnp.sin(ll), zz], axis=-1).reshape(-1, 3)
    w = (wz[:, None] * jnp.full(n, 2 * jnp.pi / n)[None, :]).ravel()
    return u, w


# ---------------------------------------------------------------- device kernels
@partial(jit, static_argnums=(3,))
def _cap_members(directions, centers, chords, capacity):
    """Per-cap member indices (ascending, -1 padded) and counts: members satisfy |u - c_j| < chord_j."""
    def one(c, r):
        member = jnp.sum((directions - c) ** 2, axis=1) < r * r
        order = jnp.argsort(~member, stable=True)[:capacity]
        ok = member[order]
        return jnp.where(ok, order, -1), jnp.sum(member)
    return lax.map(lambda a: one(*a), (centers, chords), batch_size=8)


@jit
def _cap_counts(directions, centers, chords):
    return lax.map(lambda a: jnp.sum(jnp.sum((directions - a[0]) ** 2, axis=1) < a[1] * a[1]),
                   (centers, chords), batch_size=8)


def _gnomonic(u, c, E, tan_alpha):
    """Scaled gnomonic coordinates of u (..., d) about caps (c, E, tan alpha); safe outside the hemisphere."""
    cos = jnp.sum(u * c, axis=-1)
    safe = jnp.where(cos > 1e-6, cos, 1.)
    return jnp.einsum("...kd,...d->...k", E, u) / (safe * tan_alpha)[..., None]


def _point(u, sites, weights, poly, origins, centers, bases, tans, chords, width):
    s = jnp.sum((centers - u) ** 2, axis=1) / (chords * chords)
    r = jnp.sqrt(jnp.maximum(s, 1e-300))
    q = jnp.maximum(1 - r, 0.)
    psi = q ** 4 * (4 * r + 1)
    selected, idx = lax.top_k(psi, width)
    z = _gnomonic(u[None, :], centers[idx], bases[idx], tans[idx])          # (w, d-1)
    delta = z[:, None, :] - sites[idx]
    kernel = jnp.sum(delta * delta, axis=-1) ** 1.5
    values = jnp.einsum("pn,pnm->pm", kernel, weights[idx])
    values += jnp.einsum("pt,ptm->pm", jnp.concatenate((jnp.ones((width, 1)), z), axis=1), poly[idx])
    values += origins[idx]
    return jnp.sum(selected[:, None] * values, axis=0) / jnp.sum(selected)


def _frame_point(u, *args):
    T = _tangent_basis_point(u)
    def along(t):
        v = u + t @ T
        return _point(v / jnp.linalg.norm(v), *args)
    J = jax.jacfwd(along)(jnp.zeros(T.shape[0]))                            # (d, d-1)
    if u.shape[0] == 3:
        n = jnp.cross(J[:, 0], J[:, 1])
    else:
        n = jnp.stack([J[1, 0], -J[0, 0]])
    return J, n / jnp.linalg.norm(n)


@partial(jit, static_argnums=(1, 10))
def _grouped_evaluate(u, width, sites, weights, poly, origins, centers, bases, tans, chords, batch):
    return lax.map(lambda p: _point(p, sites, weights, poly, origins, centers, bases, tans, chords, width), u,
                   batch_size=batch)


@partial(jit, static_argnums=(1, 10))
def _grouped_frame(u, width, sites, weights, poly, origins, centers, bases, tans, chords, batch):
    return lax.map(lambda p: _frame_point(p, sites, weights, poly, origins, centers, bases, tans, chords, width), u,
                   batch_size=batch)


@partial(jit, static_argnums=(3,))
def _overlap_counts(u, centers, chords, chunk):
    def block(p):
        d2 = jnp.sum((p[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        return jnp.sum(d2 < chords[None, :] ** 2, axis=1)
    return lax.map(block, u.reshape(-1, chunk, u.shape[1])).reshape(-1)


# ---------------------------------------------------------------- model
@dataclass
class SphericalPUModel:
    sites: object        # (m, n, d-1) gnomonic coordinates of the fit sites, zero padded
    weights: object      # (m, n, d) PHS coefficients
    poly_coeffs: object  # (m, d, d) coefficients of 1, z_1, ..., z_{d-1}
    origins: object      # (m, d) mean of the fitted values
    centers: object      # (m, d) cap centers c_j
    bases: object        # (m, d-1, d) tangent bases E_j
    tans: object         # (m,) tan(alpha_j)
    chords: object       # (m,) 2 sin(alpha_j / 2)
    betas: jnp.ndarray    # (m,) covering radii
    diagnostics: dict = field(default_factory=dict)

    @property
    def dim(self):
        return int(self.centers.shape[1])

    def _args(self):
        return (self.sites, self.weights, self.poly_coeffs, self.origins, self.centers, self.bases, self.tans, self.chords)

    def _grouped(self, u, frame=False):
        from kernelpack.accelerators.jax_spatial import bucket
        from .parametric_pu import device_groups, pad_device, take_rows, put_rows
        u = jnp.asarray(u, dtype=float)
        u = u / jnp.linalg.norm(u, axis=1, keepdims=True)
        n, d = u.shape
        m = int(self.centers.shape[0])
        if n == 0:
            return (jnp.zeros((0,d,d-1)),jnp.zeros((0,d))) if frame else jnp.zeros((0,d))
        n_pad = bucket(n)
        chunk = 1 << (max(1,min(n_pad,max(64,(1<<22)//max(1,m)))).bit_length()-1)
        # A valid unit vector pads every group; out-of-range output rows are dropped.
        pad = pad_device(u,n_pad).at[n:,0].set(1.)
        counts = _overlap_counts(pad,self.centers,self.chords,chunk)[:n]
        if bool(jnp.any(counts==0)):
            raise ValueError("spherical PU cover misses parameter points")
        if frame:
            jac=jnp.zeros((n,d,d-1)); nr=jnp.zeros((n,d))
        else:
            out=jnp.zeros((n,d))
        for width,count,ids in device_groups(counts):
            width=min(width,m)
            sub=take_rows(u,ids).at[count:,0].set(1.)
            per=max(1,width*int(self.sites.shape[1])*4)
            batch=min(len(sub),1 << (max(1,(1<<25)//per).bit_length()-1))
            if frame:
                j,v=_grouped_frame(sub,width,*self._args(),max(1,batch//4))
                jac=put_rows(jac,ids,j); nr=put_rows(nr,ids,v)
            else:
                out=put_rows(out,ids,_grouped_evaluate(sub,width,*self._args(),batch))
        return (jac,nr) if frame else out

    def evaluate(self, directions):
        """Positions x(u) at unit vectors u (n, d)."""
        if isinstance(directions, jax.core.Tracer):
            m = int(self.centers.shape[0])
            return lax.map(lambda p: _point(p / jnp.linalg.norm(p), *self._args(), m), directions)
        return self._grouped(directions)

    def frame(self, directions):
        """Jacobians J (n, d, d-1) along the tangent bases at u, and unit normals (n, d)."""
        if isinstance(directions, jax.core.Tracer):
            m = int(self.centers.shape[0])
            return lax.map(lambda p: _frame_point(p / jnp.linalg.norm(p), *self._args(), m), directions)
        return self._grouped(directions, frame=True)

    def cap_mask(self, directions, cap):
        """Membership of unit vectors in the support of one cap."""
        u = jnp.asarray(directions, float)
        c = jnp.asarray(self.centers)[cap]
        return jnp.sum((u - c) ** 2, axis=1) < float(jnp.asarray(self.chords)[cap]) ** 2


def _build(directions, positions, centers, betas, options):
    """Device memberships and local solves, grouped by support capacity."""
    from kernelpack.accelerators.jax_spatial import bucket
    from .parametric_pu import pad_device, device_groups, _bucket_data, _put_fits
    directions=jnp.asarray(directions,dtype=float); positions=jnp.asarray(positions,dtype=float)
    centers=jnp.asarray(centers,dtype=float); betas=jnp.asarray(betas,dtype=float)
    d=directions.shape[1]; alphas=options.overlap*betas
    if bool(jnp.any(alphas>=jnp.pi/2)):
        raise ValueError("spherical PU supports must lie in open hemispheres")
    chords=2*jnp.sin(alphas/2)
    m_pad,n_pad=bucket(len(centers),8),bucket(len(directions),64)
    u_dev=pad_device(directions,n_pad).at[len(directions):].set(1e300)
    c_dev=pad_device(centers,m_pad); r_dev=pad_device(chords,m_pad)
    counts=_cap_counts(u_dev,c_dev,r_dev)[:len(centers)]
    if bool(jnp.any(counts<options.min_sites)):
        index=int(jnp.argmin(counts))
        raise ValueError(f"spherical PU cap needs more geometry data: cap {index}, sites {int(counts[index])}")
    capacity=bucket(int(jnp.max(counts)),1)
    ids,_=_cap_members(u_dev,c_dev,r_dev,capacity); ids=ids[:len(centers)]
    active=ids>=0; bases=tangent_basis(centers); tans=jnp.tan(alphas)
    member_u=directions[jnp.maximum(ids,0)]
    cosine=jnp.sum(member_u*centers[:,None,:],axis=2)
    sites=jnp.einsum("mkd,mnd->mnk",bases,member_u)/jnp.where(active,cosine,1.)[...,None]/tans[:,None,None]
    sites=jnp.where(active[...,None],sites,0.)
    matrix=jnp.concatenate([jnp.ones(active.shape+(1,)),sites],axis=2)*active[...,None]
    rank=jnp.linalg.matrix_rank(matrix)
    if bool(jnp.any(rank<d)):
        raise ValueError("spherical PU cap data do not span the cap")
    values=jnp.where(active[...,None],positions[jnp.maximum(ids,0)],0.)
    weights=jnp.zeros(sites.shape[:2]+(d,)); poly=jnp.zeros((len(centers),d,d)); origins=jnp.zeros((len(centers),d))
    for size,count,group in device_groups(jnp.maximum(counts,8)):
        per=(size+d)**2*8*4
        batch=bucket(max(1,min(count,(1<<31)//per)),1)
        for start in range(0,count,batch):
            amount=min(batch,count-start)
            g=pad_device(group[start:start+amount],batch,len(centers))
            w,pc,o=_solve_local_fits(*_bucket_data(sites,values,active,g,amount,size))
            weights,poly,origins=_put_fits(weights,poly,origins,g,w,pc,o)
    if not bool(jnp.all(jnp.isfinite(weights))):
        raise ValueError("nonfinite spherical PU fit")
    model=SphericalPUModel(sites,weights,poly,origins,centers,bases,tans,chords,betas)
    model.diagnostics=dict(cap_count=len(centers),support_counts=[int(v) for v in counts],padded_capacity=capacity,
        kernel="r^3 + Pi_1 in scaled gnomonic coordinates",weights="Wendland C2 of chordal distance")
    return model


def fit_spherical_pu(directions, positions, *, validation_directions=None, validation_positions=None,
                     validation_normals=None, options=SphericalPUOptions(), initial_cover=None):
    """Fit and refine the cap cover until the physical indicators pass (as fit_parametric_pu).

    directions: (n, d) unit vectors (parameters); positions: (n, d) samples. Without separate validation
    samples, every fifth sample is held out while the cover is selected and the accepted cover is refitted
    with all samples. initial_cover: optional (centers, covering radii).
    """
    u = jnp.asarray(directions, float); x = jnp.asarray(positions, float)
    if (u.ndim != 2 or u.shape[1] not in (2, 3) or x.shape != u.shape
            or not jnp.isfinite(u).all() or not jnp.isfinite(x).all()):
        raise ValueError("finite unit-vector parameters and matching positions are required")
    if not 1 < options.overlap < 2 or options.min_sites < u.shape[1] + 1 or options.max_refinements < 0:
        raise ValueError("invalid spherical PU options")
    u = u / jnp.linalg.norm(u, axis=1, keepdims=True)
    length = float(2 * jnp.linalg.norm(x - x.mean(axis=0), axis=1).max())
    if not length>0:
        raise ValueError("geometry data must have nonzero physical extent")
    tolerance = options.relative_position_tolerance * length
    if validation_directions is None:
        check = jnp.arange(2, len(u), 5)
        train = jnp.asarray([i for i in range(len(u)) if i<2 or (i-2)%5],dtype=jnp.int32)
        vu, vx = u[check], x[check]
        fu, fx = u[train], x[train]
        kind = "heldout_data_sites"
    else:
        vu = jnp.asarray(validation_directions, float); vu = vu / jnp.linalg.norm(vu, axis=1, keepdims=True)
        vx = jnp.asarray(validation_positions, float)
        fu, fx = u, x
        kind = "separate_geometry_samples"
    vn = None if validation_normals is None else jnp.asarray(validation_normals, float)
    if (vu.ndim!=2 or vu.shape[1]!=u.shape[1] or vx.shape!=vu.shape or not len(vu)
            or not bool(jnp.all(jnp.isfinite(vu))) or not bool(jnp.all(jnp.isfinite(vx)))):
        raise ValueError("finite matching geometry validation arrays are required")
    if vn is not None:
        if vn.shape!=vx.shape or not bool(jnp.all(jnp.isfinite(vn))) or bool(jnp.any(jnp.linalg.norm(vn,axis=1)==0)):
            raise ValueError("finite nonzero validation normals must match positions")
        vn = vn / jnp.linalg.norm(vn, axis=1, keepdims=True)
    centers, betas = initial_caps(u.shape[1]) if initial_cover is None else map(jnp.asarray, initial_cover)
    history = []; best = math.inf
    for refinement in range(options.max_refinements + 1):
        if len(centers) > options.max_caps:
            raise PUFitError("spherical PU refinement exceeded max_caps", history)
        try:
            model = _build(fu, fx, centers, betas, options)
        except ValueError as error:
            raise PUFitError(str(error), history) from error
        pred = jnp.asarray(model.evaluate(vu)); jac, nr = map(jnp.asarray, model.frame(vu))
        error = jnp.linalg.norm(pred - vx, axis=1)
        sv = jnp.linalg.svd(jnp.nan_to_num(jac), compute_uv=False)
        regular = jnp.isfinite(jac).all(axis=(1, 2)) & jnp.isfinite(nr).all(axis=1)
        if u.shape[1] == 3:
            regular &= sv[:, -1] > options.min_jacobian_ratio * jnp.maximum(sv[:, 0], 1e-300)
        else:
            regular &= sv[:, 0] > 0
        nerr = jnp.zeros(len(vu)) if vn is None else jnp.linalg.norm(nr - vn, axis=1)
        failed = (error > tolerance) | (nerr > options.normal_tolerance) | ~regular | ~jnp.isfinite(error)
        ratio = float(jnp.min(sv[:, -1] / jnp.maximum(sv[:, 0], 1e-300)))
        score = max(float(error.max()) / tolerance, float(nerr.max()) / options.normal_tolerance,
                    options.min_jacobian_ratio / max(ratio, 1e-300) if u.shape[1] == 3 else 0.)
        if not math.isfinite(score):
            score = math.inf
        history.append(dict(refinement=refinement, cap_count=len(centers), max_position_error=float(error.max()),
                            max_normal_error=None if vn is None else float(nerr.max()), min_jacobian_ratio=ratio,
                            indicator_score=None if not math.isfinite(score) else score, failed_samples=int(failed.sum())))
        if not failed.any():
            if validation_directions is None:
                model = _build(u, x, centers, betas, options)
            model.diagnostics.update(validation=kind, absolute_position_tolerance=float(tolerance),
                                     options=vars(options), history=history, validated_samples=len(vu))
            return model
        if refinement > 0 and score >= best:
            raise PUFitError("spherical PU did not meet the physical indicators: refinement did not improve the "
                             "worst indicator; more geometry data or a better parameterization is required", history)
        best = score
        if refinement == options.max_refinements:
            raise PUFitError(f"spherical PU did not meet the physical indicators: {history[-1]}", history)
        from kernelpack.accelerators.jax_spatial import compact_rows
        bad = compact_rows(failed,vu)
        chords = 2 * jnp.sin(options.overlap * betas / 2)
        touched = {j for j in range(len(centers)) if jnp.any(jnp.sum((bad - centers[j]) ** 2, axis=1) < chords[j] ** 2)}
        centers, betas = split_caps(centers, betas, touched)


# ---------------------------------------------------------------- data and sampling
@dataclass
class ClosedSurfaceData:
    """A smooth closed curve or surface as one piece with parameters on the unit sphere.

    directions are the parameters u_k (unit vectors). from_samples computes them about a center, or converts
    latitude/longitude. Normals are optional; when given they are used to check star-shapedness about the
    center (only for directions about a center) and as validation normals.
    """
    positions: object
    directions: object
    center: object = None
    validation_directions: object = None
    validation_positions: object = None
    validation_normals: object = None
    options: SphericalPUOptions = field(default_factory=SphericalPUOptions)

    def __len__(self):
        return len(self.positions)

    @classmethod
    def from_samples(cls, positions, *, center=None, latlong=None, normals=None, options=SphericalPUOptions()):
        x = jnp.asarray(positions, float)
        if x.ndim != 2 or x.shape[1] not in (2, 3) or not jnp.isfinite(x).all():
            raise ValueError("finite 2D curve or 3D surface samples are required")
        if latlong is not None:
            u = latlong_to_directions(latlong)
            if u.shape != x.shape:
                raise ValueError("parameters do not match the samples")
            c = None
        else:
            c = .5 * (x.min(axis=0) + x.max(axis=0)) if center is None else jnp.asarray(center, float)
            u = directions_about(x, c)
            if normals is not None:
                nr = jnp.asarray(normals, float)
                if jnp.any(jnp.sum(nr * u, axis=1) <= 0):
                    raise ValueError("the boundary is not star-shaped about the center (or the normals point inward)")
        return cls(x, u, c, options=options)


def sample_sphere_piece(model, radius, *, seed=0, supersample_fac=2):
    """Dart-throwing candidates on S^{d-1} whose images are spaced at most about radius / sqrt(supersample_fac);
    piecewise assembly thins them in physical space, as for faces (hierarchical.sample_parametric_face)."""
    from kernelpack.accelerators.jax_spatial import poisson_disk_sample
    from .core import fibonacci_sphere
    d = model.dim
    if radius <= 0 or supersample_fac < 1:
        raise ValueError("invalid spacing")
    probe = jnp.asarray(fibonacci_sphere(2000)) if d == 3 else latlong_to_directions(jnp.linspace(0, 2 * jnp.pi, 720, endpoint=False))
    jac, _ = model.frame(probe)
    largest = float(jnp.max(jnp.linalg.svd(jnp.asarray(jac), compute_uv=False)[:, 0]))
    if not jnp.isfinite(largest) or largest <= 0:
        raise ValueError("invalid spherical PU metric")
    chord = float(radius) / (1.1 * largest * jnp.sqrt(float(supersample_fac)))
    if d == 2:
        n = max(8, int(jnp.ceil(2 * jnp.pi / chord)))
        phase=jax.random.uniform(jax.random.key(int(seed)),(),dtype=jnp.float64)
        t = 2 * jnp.pi * (jnp.arange(n) + phase) / n
        return jnp.asarray(latlong_to_directions(t))

    def draw(key, n):
        g = jax.random.normal(key,(n,d),dtype=jnp.float64)
        return g / jnp.linalg.norm(g, axis=1, keepdims=True)
    n_draw = int(6 * 4 * jnp.pi / chord ** 2)
    pts = poisson_disk_sample(-jnp.ones(d), jnp.ones(d), lambda p: jnp.full(len(p), chord), chord,
                              seed=int(seed), draw=draw, n_draw=n_draw,
                              require_gpu=model.sites.device.platform=="gpu")
    return jnp.asarray(pts)
