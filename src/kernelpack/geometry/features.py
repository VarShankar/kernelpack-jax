"""Feature sites, normals, and priorities derived from the pieces of a piecewise smooth boundary.

The trimming loops of the pieces, mapped to physical space, are the places where pieces meet.
For spacing h we sample every loop at its vertices and at arc-length spacing at most h/4 along
each loop edge. A sample f belongs to the pieces L(f) whose mapped loops pass within a small
tolerance of f, and receives
    normal    nu(f) = sum_{l in L(f)} n_l(f) / |sum_{l in L(f)} n_l(f)|,
    priority  p(f)  = min(|L(f)| - 1, d - 1),
the codimension of the feature for generic arrangements: 1 on a crease where two surface pieces
meet (or at a corner of a curve), 2 at a vertex where three or more surface pieces meet.
Samples on a single piece (|L| = 1) are not features, and neither are samples on a smooth seam, where the
normals of all incident pieces agree to within SMOOTH_SEAM_DEGREES (pieces that split a smooth boundary into
charts, as for a closed curve or surface).
"""
import numpy as np
import jax
import jax.numpy as jnp

from kernelpack.accelerators.jax_spatial import compact_rows, nearest_within_sparse, row_norm

SITE_SPACING = .25       # sites per loop edge at spacing <= SITE_SPACING * h
DENSE_SPACING = 1 / 64   # incidence test against loops resolved at this spacing * h
INCIDENCE_TOLERANCE = .02  # a site lies on a loop within this distance * h
SMOOTH_SEAM_DEGREES = 2.   # incident normals within this angle: a smooth seam, not a feature


def _loop_samples(segment, loop, spacing):
    """Parameters on a closed parameter-space loop: every vertex and points at physical spacing <= spacing.

    Device array; the per-edge counts (static shapes) come from the mapped loop vertices."""
    uv_h = np.asarray(loop, float)
    uv = jnp.asarray(uv_h)
    xyz = jnp.asarray(segment.evaluate_parametric(uv))
    nxt = np.roll(np.arange(len(uv_h)), -1)
    lengths = np.asarray(row_norm(xyz[nxt] - xyz))
    counts = np.maximum(1, np.ceil(lengths / spacing).astype(int))
    edge = np.repeat(np.arange(len(uv_h)), counts)
    t = np.concatenate([np.arange(c) / c for c in counts])[:, None]
    a = uv[edge]
    # a + t (b - a) as separate device ops (no multiply-add contraction), as numpy evaluates it.
    return a + _product(jnp.asarray(t), uv[nxt[edge]] - a)


@jax.jit
def _product(t, d):
    return jax.lax.optimization_barrier(t * d)


def _piece_loop_points(segment, spacing):
    """Physical points and unit normals of a piece's mapped trimming loops (device)."""
    model = segment.geom_model
    loops = model.get("outlines")
    if loops is None or not len(loops):
        return None, None
    if segment.surf_dim == 1:
        uv = jnp.asarray(np.asarray(loops[0], float).reshape(-1, 1))   # the two endpoints of a curve
    else:
        uv = jnp.concatenate([_loop_samples(segment, loop, spacing) for loop in loops])
    x = jnp.asarray(segment.evaluate_parametric(uv), float)
    n = jnp.asarray(segment.evaluate_parametric_frame(uv)[-1], float)
    return x, n / row_norm(n)[:, None]


@jax.jit
def _dot_rows(a, b):
    """Row dot products summed in coordinate order without contraction (numpy's np.sum(a * b, axis=1))."""
    p = jax.lax.optimization_barrier(a * b)
    s = p[:, 0]
    for k in range(1, p.shape[1]):
        s = s + p[:, k]
    return s


def derive_features(segments, h, smooth_seam_degrees=SMOOTH_SEAM_DEGREES):
    """Feature sites, unit normals, and integer priorities for spacing h (see the module docstring); device arrays."""
    h = float(h)
    sites, dense = [], []
    for segment in segments:
        sites.append(_piece_loop_points(segment, SITE_SPACING * h)[0])
        dense.append(_piece_loop_points(segment, DENSE_SPACING * h))
    sites = [s for s in sites if s is not None and s.size]
    if not sites:
        return jnp.zeros((0, 0)), jnp.zeros((0, 0)), jnp.zeros(0, int)
    x = jnp.concatenate(sites)
    d = x.shape[1]
    tolerance = INCIDENCE_TOLERANCE * h
    count = jnp.zeros(x.shape[0], int)
    normal = jnp.zeros_like(x)
    first = jnp.zeros_like(x)                 # normal of the first incident piece
    min_dot = jnp.ones(x.shape[0])            # smallest cosine between an incident normal and the first one
    for points, normals in dense:
        if points is None or not points.size:
            continue
        # Nearest loop point, exact within the tolerance (beyond it only "not incident" matters).
        dist, idx = nearest_within_sparse(x, points, tolerance)
        on = dist <= tolerance
        nb = normals[jnp.where(on, idx, 0)]
        fresh = on & (count == 0)
        first = jnp.where(fresh[:, None], nb, first)
        later = on & ~fresh
        min_dot = jnp.where(later, jnp.minimum(min_dot, _dot_rows(nb, first)), min_dot)
        count = count + on
        normal = jnp.where(on[:, None], normal + nb, normal)
    smooth = min_dot >= np.cos(np.radians(smooth_seam_degrees))
    keep = (count >= 2) & ~smooth
    x, normal, count = compact_rows(keep, x, normal, count)
    norm = row_norm(normal)
    if bool(jnp.any(norm < 1e-8)):
        raise ValueError("feature normals cancel; incident pieces have opposite normals")
    priority = jnp.minimum(count - 1, d - 1).astype(int)
    return x, normal / norm[:, None], priority
