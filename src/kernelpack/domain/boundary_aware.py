"""Boundary-aware stencil selection by ray tracing against the implicit layer.

Let phi be the implicit layer (positive in the domain), L a Lipschitz bound of phi near the boundary and
delta >= 0 a tolerance. A segment [a, b] is *visible* if phi >= -delta along it. Visibility is decided by

* a certificate: phi(a) + phi(b) > L |b - a| implies visibility (Lipschitz cones from both ends), and
* sphere tracing along the remaining segments: from z, advance by max(phi(z) / L, eps); the segment is
  hidden as soon as phi(z) < -delta, and visible once the certificate holds from z or the end is reached.

Off the cover of the implicit layer, phi is the constant +-scale of the band extension; there it is replaced
by +-cap with cap = 0.2 min_j R_j, a lower bound of the distance to the boundary for uncovered points
(every boundary point lies in some ball B_j within R_j / 1.25 of its center).

A ghost node is judged through its boundary node. For a center c the distance to a candidate x is
|x - c| if x is visible from c, otherwise the shortest path c -> x_j -> x through one visible candidate x_j
near the boundary that sees x (infinite if there is none); a hidden ghost adds |ghost - boundary node|.
The stencil is the k candidates of smallest distance. If the k Euclidean nearest are all visible they are
the stencil, since no distance is smaller than the Euclidean one.

All arrays stay on the device. The host reads only scalars: the number of segments still being traced
(once per sphere-tracing step, to size the next level-set evaluation), the number of centers whose k
Euclidean nearest are not all visible, and the number of two-hop paths per block. Arrays whose length is
one of these counts are padded to powers of two (few compiled shapes); padded rows repeat a valid point and
are discarded. This relies on phi at a point not depending on the other points of the evaluation, which
holds for the grouped CFPU evaluation (it pads internally as well).
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

__all__ = ["StencilLevelSet", "boundary_aware_knn"]


class StencilLevelSet:
    """phi for stencil selection: evaluate(points) of a CFPU model with the band far field capped."""

    def __init__(self, model, h, *, lipschitz=1.5, tolerance_fraction=0.05, min_step_fraction=0.1):
        self.model = model
        self.h = float(h)
        self.lipschitz = float(lipschitz)
        self.delta = tolerance_fraction * self.h
        self.eps = min_step_fraction * self.h
        self.cap = 0.2 * float(jnp.min(jnp.asarray(model.radii)))
        self.scale = float(model.scale)
        self.evaluations = 0

    def __call__(self, points, n_valid=None):
        """phi at the rows of `points` (a device array); n_valid counts the evaluations of padded inputs."""
        points = jnp.asarray(points, float)
        if points.shape[0] == 0:
            return jnp.zeros(0)
        self.evaluations += int(points.shape[0] if n_valid is None else n_valid)
        v = jnp.asarray(self.model.evaluate(points), float)
        return _cap_far_field(v, self.scale, self.cap)


@jax.jit
def _cap_far_field(v, scale, cap):
    far = jnp.abs(jnp.abs(v) - scale) <= 1e-12 * scale
    return jnp.where(far, jnp.sign(v) * cap, v)


def _bucket(n, minimum=64):
    """Smallest power of two >= max(n, minimum)."""
    return 1 << (max(int(n), int(minimum), 1) - 1).bit_length()


def _norm(v):
    """Euclidean norm over the last axis, summed left to right (the order of numpy's norm for d <= 3)."""
    acc = v[..., 0] * v[..., 0]
    for i in range(1, v.shape[-1]):
        acc = acc + v[..., i] * v[..., i]
    return jnp.sqrt(acc)


@jax.jit
def _trace_init(a, b, phi_a, phi_b, valid, L, delta):
    length = _norm(b - a)
    visible = valid & (phi_a + phi_b > L * length)                     # certificate
    decided = visible | (phi_a < -delta) | (phi_b < -delta)            # an end outside: hidden
    return length, visible, valid & ~decided


@jax.jit
def _trace_advance(s, phi_z, length, visible, active, L, eps):
    """One step of every active segment; segments that reach their end are visible."""
    s = jnp.where(active, s + jnp.maximum(phi_z / L, eps), s)
    done = active & (s >= length)
    active = active & ~done
    return s, visible | done, active, jnp.count_nonzero(active)


@partial(jax.jit, static_argnames=("size",))
def _trace_points(a, b, s, length, active, *, size):
    """Current points of the active segments in increasing row order, padded to `size` rows
    (index n, coordinates of the first active point)."""
    n = a.shape[0]
    idx = jnp.nonzero(active, size=size, fill_value=n)[0]
    safe = jnp.where(idx < n, idx, idx[0])
    u = (s[safe] / length[safe])[:, None]
    return idx, a[safe] + u * (b[safe] - a[safe])


@jax.jit
def _trace_update(idx, v, phi_z, phi_b, s, length, visible, active, L, delta):
    safe = jnp.minimum(idx, phi_z.shape[0] - 1)                         # padded rows are dropped below
    hidden = v < -delta
    cert = v + phi_b[safe] > L * (length[safe] - s[safe])
    phi_z = phi_z.at[idx].set(v, mode="drop")
    visible = visible.at[idx].set(visible[safe] | (cert & ~hidden), mode="drop")
    active = active.at[idx].set(~(hidden | cert), mode="drop")
    return phi_z, visible, active


def _trace(phi: StencilLevelSet, a, b, phi_a, phi_b, valid, max_iter=400):
    """Visibility of the segments [a_i, b_i] (rows) with valid_i, given phi at both ends."""
    L, delta, eps = phi.lipschitz, phi.delta, phi.eps
    length, visible, active = _trace_init(a, b, phi_a, phi_b, valid, L, delta)
    s = jnp.zeros(a.shape[0], a.dtype); phi_z = phi_a
    for _ in range(max_iter):
        s, visible, active, count = _trace_advance(s, phi_z, length, visible, active, L, eps)
        count = int(count)
        if count == 0:
            break
        idx, z = _trace_points(a, b, s, length, active, size=_bucket(count))
        phi_z, visible, active = _trace_update(idx, phi(z, n_valid=count), phi_z, phi_b, s, length, visible,
                                               active, L, delta)
    return visible & ~active                                           # undecided after max_iter: hidden


@partial(jax.jit, static_argnames=("k",))
def _segments(points, phi_nodes, cand_idx, tgt, row_valid, *, k):
    """Segments center -> target of the first k candidates, row-major over (center, candidate)."""
    centers = cand_idx[:, 0]
    a = jnp.repeat(points[centers], k, axis=0); t = tgt[:, :k].reshape(-1)
    return a, points[t], jnp.repeat(phi_nodes[centers], k), phi_nodes[t], jnp.repeat(row_valid, k)


@partial(jax.jit, static_argnames=("size",))
def _select_rows(mask, *, size):
    """Indices of the true rows (padded with 0) and the validity of the padded list."""
    idx = jnp.nonzero(mask, size=size, fill_value=0)[0]
    return idx, jnp.arange(size) < jnp.count_nonzero(mask)


@partial(jax.jit, static_argnames=("m",))
def _bends(dcj, xj, xt, near, hid, *, m):
    """For each hidden k, the m shortest paths c -> x_j -> x_k over bends j near the boundary.

    Returns the (h, m, k) path lengths (inf where there is no such bend), the bend indices j and the
    number of finite paths."""
    via = dcj[:, :, None] + _norm(xj[:, :, None, :] - xt[:, None, :, :])       # (h, j, k)
    via = jnp.where(near[:, :, None] & hid[:, None, :], via, jnp.inf)
    neg, part = jax.lax.top_k(-jnp.swapaxes(via, 1, 2), m)                     # (h, k, m)
    vals = jnp.swapaxes(-neg, 1, 2)
    return vals, jnp.swapaxes(part, 1, 2), jnp.count_nonzero(jnp.isfinite(vals))


@partial(jax.jit, static_argnames=("size",))
def _bend_segments(points, phi_nodes, ci, th, vals, part, r0, *, size):
    """Segments x_j -> x_k of the finite paths of one block of hard centers (padded to `size`)."""
    finite = jnp.isfinite(vals)
    rr, mm, kk = jnp.nonzero(finite, size=size, fill_value=0)
    valid = jnp.arange(size) < jnp.count_nonzero(finite)
    pr = rr + r0; pj = part[rr, mm, kk]
    j = ci[pr, pj]; t = th[pr, kk]
    return pr, kk, vals[rr, mm, kk], valid, points[j], points[t], phi_nodes[j], phi_nodes[t]


def boundary_aware_knn(points, owner, phi_nodes, cand_idx, cand_dist, k, phi: StencilLevelSet, *,
                       n_bend=8, bend_band=2.0, stats=None):
    """Stencils of size k from Euclidean candidates (cand_idx, cand_dist: (M, K), column 0 = the center).

    points: tree points; owner[i]: index of the point whose visibility represents point i (itself for
    physical nodes, its boundary node for ghosts); phi_nodes: phi at the points. Device arrays in and out.
    """
    points = jnp.asarray(points, float); owner = jnp.asarray(owner)
    phi_nodes = jnp.asarray(phi_nodes, float)
    cand_idx = jnp.asarray(cand_idx); cand_dist = jnp.asarray(cand_dist, float)
    M, K = cand_idx.shape
    out_i = cand_idx[:, :k]; out_d = cand_dist[:, :k]
    if M == 0:
        return out_i, out_d
    Mb = _bucket(M, 16)                                      # rows padded with copies of row 0
    row_valid = jnp.arange(Mb) < M
    rows_p = jnp.where(row_valid, jnp.arange(Mb), 0)
    cand_p, dist_p = cand_idx[rows_p], cand_dist[rows_p]
    tgt = owner[cand_p]                                      # (Mb, K) visibility targets

    # 1. the k Euclidean nearest
    vis_k = _trace(phi, *_segments(points, phi_nodes, cand_p, tgt, row_valid, k=k)).reshape(Mb, k)
    hard_mask = row_valid & ~vis_k.all(axis=1)
    H = int(jnp.count_nonzero(hard_mask))
    if stats is not None:
        stats.update(centers=M, hard=H)
    if H == 0:
        return out_i, out_d
    hard, hvalid = _select_rows(hard_mask, size=_bucket(H, 16))
    Hb = int(hard.shape[0])

    # 2. all K candidates of the hard centers
    ci = cand_p[hard]; th = tgt[hard]                         # (Hb, K)
    vis = _trace(phi, *_segments(points, phi_nodes, ci, th, hvalid, k=K)).reshape(Hb, K)
    dist = jnp.where(vis, dist_p[hard], jnp.inf)

    # 3. two-hop distances of hidden candidates through visible candidates near the boundary
    #    (blocks of hard centers: for each hidden k, the n_bend shortest paths c -> x_j -> x_k)
    near = vis & (phi_nodes[ci] < bend_band * phi.h)
    hid = ~vis
    xj_all = points[ci]                                        # (Hb, K, d) candidates (bends)
    xt_all = points[th]                                        # (Hb, K, d) visibility targets
    dcj = _norm(xj_all - points[ci[:, 0]][:, None, :])         # (Hb, K)
    m = min(n_bend, K)
    chunk = min(Hb, 1 << (max(1, int(2e7 // (K * K))).bit_length() - 1))   # power of two: divides Hb
    best = jnp.full((Hb, K), jnp.inf)
    for r0 in range(0, Hb, chunk):
        sl = slice(r0, r0 + chunk)
        vals, part, count = _bends(dcj[sl], xj_all[sl], xt_all[sl], near[sl], hid[sl], m=m)
        count = int(count)
        if count == 0:
            continue
        pr, pk, pd, valid, a, b, phi_a, phi_b = _bend_segments(points, phi_nodes, ci, th, vals, part, r0,
                                                               size=_bucket(count))
        ok = _trace(phi, a, b, phi_a, phi_b, valid)
        best = best.at[pr, pk].min(jnp.where(ok, pd, jnp.inf))
    ghost = owner[ci] != ci
    extra = jnp.where(ghost, _norm(points[ci] - points[owner[ci]]), 0.)
    dist = jnp.where(vis, dist, best + extra)

    # Candidates not reachable with at most one bend (e.g. across a wall in a slot narrower than the stencil)
    # rank after all reachable ones, in Euclidean order.
    unreachable = ~jnp.isfinite(dist)
    rank = jnp.where(unreachable, 1e30 + dist_p[hard], dist)
    order = jnp.argsort(rank, axis=1, stable=True)[:, :k]
    sel_d = jnp.take_along_axis(jnp.where(unreachable, dist_p[hard], dist), order, axis=1)
    sel_i = jnp.take_along_axis(ci, order, axis=1)
    if stats is not None:
        fallback = hvalid & jnp.take_along_axis(unreachable, order, axis=1).any(axis=1)
        changed = hvalid & (jnp.sort(sel_i, 1) != jnp.sort(ci[:, :k], 1)).any(1)
        stats.update(fallback_rows=int(jnp.count_nonzero(fallback)), changed=int(jnp.count_nonzero(changed)))
    target = jnp.where(hvalid, hard, M)                        # padded rows are dropped
    return out_i.at[target].set(sel_i, mode="drop"), out_d.at[target].set(sel_d, mode="drop")
