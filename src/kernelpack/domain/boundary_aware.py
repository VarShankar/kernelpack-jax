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
"""
from __future__ import annotations

import numpy as np

__all__ = ["StencilLevelSet", "boundary_aware_knn"]


class StencilLevelSet:
    """phi for stencil selection: evaluate(points) of a CFPU model with the band far field capped."""

    def __init__(self, model, h, *, lipschitz=1.5, tolerance_fraction=0.05, min_step_fraction=0.1):
        self.model = model
        self.h = float(h)
        self.lipschitz = float(lipschitz)
        self.delta = tolerance_fraction * self.h
        self.eps = min_step_fraction * self.h
        self.cap = 0.2 * float(np.min(np.asarray(model.radii)))
        self.scale = float(model.scale)
        self.evaluations = 0

    def __call__(self, points):
        points = np.asarray(points, float)
        if len(points) == 0:
            return np.zeros(0)
        self.evaluations += len(points)
        v = np.asarray(self.model.evaluate(points), float)
        far = np.abs(np.abs(v) - self.scale) <= 1e-12 * self.scale
        return np.where(far, np.sign(v) * self.cap, v)


def _trace(phi: StencilLevelSet, a, b, phi_a, phi_b, max_iter=400):
    """Visibility of the segments [a_i, b_i] (rows), given phi at both ends."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    length = np.linalg.norm(b - a, axis=1)
    L = phi.lipschitz
    visible = np.zeros(len(a), bool)
    decided = phi_a + phi_b > L * length                     # certificate
    visible[decided] = True
    decided |= (phi_a < -phi.delta) | (phi_b < -phi.delta)  # an end outside: hidden
    s = np.zeros(len(a)); phi_z = phi_a.copy()
    active = np.flatnonzero(~decided)
    for _ in range(max_iter):
        if active.size == 0:
            break
        step = np.maximum(phi_z[active] / L, phi.eps)
        s[active] += step
        done = s[active] >= length[active]
        visible[active[done]] = True
        active = active[~done]
        if active.size == 0:
            break
        u = (s[active] / length[active])[:, None]
        z = a[active] + u * (b[active] - a[active])
        v = phi(z)
        phi_z[active] = v
        hidden = v < -phi.delta
        cert = v + phi_b[active] > L * (length[active] - s[active])
        visible[active[cert & ~hidden]] = True
        active = active[~(hidden | cert)]
    visible[active] = False                                   # undecided after max_iter: treat as hidden
    return visible


def boundary_aware_knn(points, owner, phi_nodes, cand_idx, cand_dist, k, phi: StencilLevelSet, *,
                       n_bend=8, bend_band=2.0, stats=None):
    """Stencils of size k from Euclidean candidates (cand_idx, cand_dist: (M, K), column 0 = the center).

    points: tree points; owner[i]: index of the point whose visibility represents point i (itself for
    physical nodes, its boundary node for ghosts); phi_nodes: phi at the points.
    """
    points = np.asarray(points, float)
    cand_idx = np.asarray(cand_idx); cand_dist = np.asarray(cand_dist, float)
    M, K = cand_idx.shape
    centers = cand_idx[:, 0]
    tgt = owner[cand_idx]                                    # (M, K) visibility targets
    out_i = cand_idx[:, :k].copy(); out_d = cand_dist[:, :k].copy()

    # 1. the k Euclidean nearest
    rows = np.repeat(np.arange(M), k); cols = np.tile(np.arange(k), M)
    a = points[centers[rows]]; b = points[tgt[rows, cols]]
    vis_k = _trace(phi, a, b, phi_nodes[centers[rows]], phi_nodes[tgt[rows, cols]]).reshape(M, k)
    hard = np.flatnonzero(~vis_k.all(axis=1))
    if stats is not None:
        stats.update(centers=M, hard=int(hard.size))
    if hard.size == 0:
        return out_i, out_d

    # 2. all K candidates of the hard centers
    H = hard.size
    rows = np.repeat(hard, K); cols = np.tile(np.arange(K), H)
    a = points[centers[rows]]; b = points[tgt[rows, cols]]
    vis = _trace(phi, a, b, phi_nodes[centers[rows]], phi_nodes[tgt[rows, cols]]).reshape(H, K)
    dist = np.where(vis, cand_dist[hard], np.inf)

    # 3. two-hop distances of hidden candidates through visible candidates near the boundary
    #    (vectorized over chunks of hard centers: for each hidden k, the n_bend shortest paths c -> x_j -> x_k)
    near = vis & (phi_nodes[cand_idx[hard]] < bend_band * phi.h)
    best = np.full((H, K), np.inf)
    xc = points[centers[hard]]                                 # (H, d)
    xj_all = points[cand_idx[hard]]                            # (H, K, d) candidates (bends)
    xt_all = points[tgt[hard]]                                 # (H, K, d) visibility targets
    dcj = np.linalg.norm(xj_all - xc[:, None, :], axis=2)      # (H, K)
    chunk = max(1, int(2e7 // (K * K)))
    pr, pj, pk, pd = [], [], [], []
    for r0 in range(0, H, chunk):
        sl = slice(r0, r0 + chunk)
        hid = ~vis[sl]
        if not hid.any() or not near[sl].any():
            continue
        via = dcj[sl][:, :, None] + np.linalg.norm(xj_all[sl][:, :, None, :] - xt_all[sl][:, None, :, :], axis=3)  # (h, j, k)
        via = np.where(near[sl][:, :, None] & hid[:, None, :], via, np.inf)
        m = min(n_bend, K)
        part = np.argpartition(via, m - 1, axis=1)[:, :m, :]                 # (h, m, k) best bends per hidden k
        vals = np.take_along_axis(via, part, axis=1)
        rr, mm, kk = np.nonzero(np.isfinite(vals))
        pr.append(rr + r0); pj.append(part[rr, mm, kk]); pk.append(kk); pd.append(vals[rr, mm, kk])
    if pr:
        pr, pj, pk, pd = (np.concatenate(v) for v in (pr, pj, pk, pd))
        a = points[cand_idx[hard[pr], pj]]; b = points[tgt[hard[pr], pk]]
        ok = _trace(phi, a, b, phi_nodes[cand_idx[hard[pr], pj]], phi_nodes[tgt[hard[pr], pk]])
        np.minimum.at(best, (pr[ok], pk[ok]), pd[ok])
        ghost = owner[cand_idx[hard]] != cand_idx[hard]
        extra = np.where(ghost, np.linalg.norm(points[cand_idx[hard]] - points[owner[cand_idx[hard]]], axis=-1), 0.)
        dist = np.where(vis, dist, best + extra)

    # Candidates not reachable with at most one bend (e.g. across a wall in a slot narrower than the stencil)
    # rank after all reachable ones, in Euclidean order.
    unreachable = ~np.isfinite(dist)
    rank = np.where(unreachable, 1e30 + cand_dist[hard], dist)
    order = np.argsort(rank, axis=1, kind="stable")[:, :k]
    sel_d = np.take_along_axis(np.where(unreachable, cand_dist[hard], dist), order, axis=1)
    if stats is not None:
        stats.update(fallback_rows=int(np.take_along_axis(unreachable, order, axis=1).any(axis=1).sum()))
    out_i[hard] = np.take_along_axis(cand_idx[hard], order, axis=1)
    out_d[hard] = sel_d
    if stats is not None:
        stats.update(changed=int((np.sort(out_i[hard], 1) != np.sort(cand_idx[hard, :k], 1)).any(1).sum()))
    return out_i, out_d
