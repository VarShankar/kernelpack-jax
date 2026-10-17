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
import jax.numpy as jnp
from scipy.spatial import cKDTree

SITE_SPACING = .25       # sites per loop edge at spacing <= SITE_SPACING * h
DENSE_SPACING = 1 / 64   # incidence test against loops resolved at this spacing * h
INCIDENCE_TOLERANCE = .02  # a site lies on a loop within this distance * h
SMOOTH_SEAM_DEGREES = 2.   # incident normals within this angle: a smooth seam, not a feature


def _loop_samples(segment, loop, spacing):
    """Parameters on a closed parameter-space loop: every vertex and points at physical spacing <= spacing."""
    uv = np.asarray(loop, float)
    xyz = np.asarray(segment.evaluate_parametric(jnp.asarray(uv)))
    nxt = np.roll(np.arange(len(uv)), -1)
    lengths = np.linalg.norm(xyz[nxt] - xyz, axis=1)
    counts = np.maximum(1, np.ceil(lengths / spacing).astype(int))
    pieces = [uv[i] + (np.arange(c)[:, None] / c) * (uv[nxt[i]] - uv[i]) for i, c in enumerate(counts)]
    return np.vstack(pieces)


def _piece_loop_points(segment, spacing):
    """Physical points, parameters and normals of a piece's mapped trimming loops."""
    model = segment.geom_model
    loops = model.get("outlines")
    if loops is None or not len(loops):
        return np.zeros((0, 0)), np.zeros((0, 0))
    if segment.surf_dim == 1:
        uv = np.asarray(loops[0], float).reshape(-1, 1)          # the two endpoints of a curve
    else:
        uv = np.vstack([_loop_samples(segment, loop, spacing) for loop in loops])
    uvj = jnp.asarray(uv)
    x = np.asarray(segment.evaluate_parametric(uvj))
    n = np.asarray(segment.evaluate_parametric_frame(uvj)[-1], float)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    return x, n


def derive_features(segments, h, smooth_seam_degrees=SMOOTH_SEAM_DEGREES):
    """Feature sites, unit normals, and integer priorities for spacing h (see the module docstring)."""
    h = float(h)
    sites, dense = [], []
    for segment in segments:
        sites.append(_piece_loop_points(segment, SITE_SPACING * h)[0])
        dense.append(_piece_loop_points(segment, DENSE_SPACING * h))
    sites = [s for s in sites if s.size]
    if not sites:
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros(0, int)
    x = np.vstack(sites)
    d = x.shape[1]
    tolerance = INCIDENCE_TOLERANCE * h
    count = np.zeros(len(x), int)
    normal = np.zeros_like(x)
    first = np.zeros_like(x)                 # normal of the first incident piece
    min_dot = np.ones(len(x))                # smallest cosine between an incident normal and the first one
    for points, normals in dense:
        if not points.size:
            continue
        dist, idx = cKDTree(points).query(x)
        on = dist <= tolerance
        fresh = on & (count == 0)
        first[fresh] = normals[idx[fresh]]
        later = on & ~fresh
        min_dot[later] = np.minimum(min_dot[later], np.sum(normals[idx[later]] * first[later], axis=1))
        count += on
        normal[on] += normals[idx[on]]
    smooth = min_dot >= np.cos(np.radians(smooth_seam_degrees))
    keep = (count >= 2) & ~smooth
    x, normal, count = x[keep], normal[keep], count[keep]
    norm = np.linalg.norm(normal, axis=1)
    if np.any(norm < 1e-8):
        raise ValueError("feature normals cancel; incident pieces have opposite normals")
    priority = np.minimum(count - 1, d - 1).astype(int)
    return x, normal / norm[:, None], priority
