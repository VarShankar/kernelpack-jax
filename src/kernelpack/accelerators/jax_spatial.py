"""Fixed-shape GPU spatial primitives in JAX.

Every jitted kernel sees power-of-two padded shapes, so it compiles once per
size bucket rather than once per call. Data stay on the device; the host sees
only scalars that fix static shapes (counts, bounding boxes, bucket sizes) and
the seeded host RNG draws, which are uploaded once.

Primitives
----------
* cell tables: points binned into a uniform grid, ``table[cell, slot]``;
* radius neighbour lists under the symmetric rule ``|x - y| < s * max(r_x, r_y)``;
* nearest distance to a point set, exact within a search radius;
* ``greedy_independent_set``: the lexicographically first maximal independent
  set of the conflict graph in priority order, computed in parallel rounds.
  A point joins once every higher-priority neighbour is decided and none
  joined, which reproduces the sequential greedy result exactly;
* ``poisson_disk_sample``: dart throwing as a greedy independent set over
  random candidates (random priorities), with fixed seed points, a variable
  radius field, an acceptance predicate, and refill passes.
"""
from __future__ import annotations

from functools import partial
import math
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, lax

_UNDECIDED, _IN, _OUT = 0, 1, 2


def enable_compilation_cache(path: str | None = None) -> str:
    """Persistent XLA cache for the fixed-shape kernels (use a local, fast filesystem)."""
    import os
    path = os.path.expanduser(path or os.environ.get("KERNELPACK_JAX_CACHE", "~/.cache/kernelpack-jax-xla"))
    os.makedirs(path, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", path)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    return path


def bucket(n: int, minimum: int = 1024) -> int:
    """Smallest power of two >= max(n, minimum)."""
    n = max(int(n), int(minimum), 1)
    return 1 << int(math.ceil(math.log2(n)))


def pad_rows(x, n: int, value=0) -> np.ndarray:
    """Pad a host array along axis 0 to ``n`` rows."""
    x = np.asarray(x)
    if x.shape[0] > n:
        raise ValueError("cannot pad to fewer rows")
    if x.shape[0] == n:
        return x
    pad = np.full((n - x.shape[0],) + x.shape[1:], value, dtype=x.dtype)
    return np.concatenate([x, pad])


def pad_rows_device(x, n: int, value=0):
    """Pad a device array along axis 0 to ``n`` rows."""
    x = jnp.asarray(x)
    if x.shape[0] > n:
        raise ValueError("cannot pad to fewer rows")
    if x.shape[0] == n:
        return x
    return jnp.concatenate([x, jnp.full((n - x.shape[0],) + x.shape[1:], value, dtype=x.dtype)])


def compact_rows(mask, *arrays):
    """Rows of each array where ``mask`` holds, in order (one scalar sync for the count)."""
    mask = jnp.asarray(mask, bool)
    idx = jnp.nonzero(mask, size=int(jnp.sum(mask)))[0]
    out = tuple(jnp.asarray(a)[idx] for a in arrays)
    return out[0] if len(out) == 1 else out


@jit
def ordered_sq_norm(d):
    """Row sums of squares ((d_0^2 + d_1^2) + d_2^2) + ..., in coordinate order and without
    multiply-add contraction: bit-identical to numpy's ``np.sum(d**2, axis=1)`` for d <= 8 columns."""
    sq = lax.optimization_barrier(d * d)
    s = sq[..., 0]
    for k in range(1, d.shape[-1]):
        s = s + sq[..., k]
    return s


def row_norm(x):
    """``np.linalg.norm(x, axis=1)`` on the device, bit for bit."""
    return jnp.sqrt(ordered_sq_norm(jnp.asarray(x)))


@jit
def _masked_box(points, valid):
    big = jnp.asarray(jnp.inf, points.dtype)
    return (jnp.min(jnp.where(valid[:, None], points, big), axis=0),
            jnp.max(jnp.where(valid[:, None], points, -big), axis=0), jnp.any(valid))


def masked_box(points, valid=None):
    """Host (lo, hi, any_valid) of the valid rows: one small transfer."""
    points = jnp.asarray(points)
    valid = jnp.ones(points.shape[0], bool) if valid is None else jnp.asarray(valid, bool)
    lo, hi, anyv = jax.device_get(_masked_box(points, valid))
    return np.asarray(lo, float), np.asarray(hi, float), bool(anyv)


def _offsets(dim: int) -> np.ndarray:
    grids = np.meshgrid(*([np.arange(-1, 2)] * dim), indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=1).astype(np.int32)


class CellGrid:
    """Uniform grid over a box with one-ring neighbour lookup (cell >= search radius)."""

    def __init__(self, lo, hi, cell: float):
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        if cell <= 0 or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
            raise ValueError("grid needs a finite box and positive cell size")
        self.dim = int(lo.size)
        self.cell = float(cell)
        self.lo = lo - 1e-9 * max(1.0, float(np.max(np.abs(hi - lo))))
        self.dims = np.maximum(1, np.ceil((hi - self.lo) / self.cell).astype(np.int64) + 1)
        n_cells = int(np.prod(self.dims))
        if n_cells > 2**31 - 1:
            raise ValueError("cell grid too large; increase the cell size")
        self.n_cells = n_cells
        self.n_cells_pad = bucket(n_cells, 64)
        self.offsets = _offsets(self.dim)

    def device_args(self):
        return (jnp.asarray(self.lo), jnp.asarray(self.cell), jnp.asarray(self.dims.astype(np.int32)))


def _cell_coords(points, lo, cell, dims):
    c = jnp.floor((points - lo) / cell).astype(jnp.int32)
    return jnp.clip(c, 0, dims - 1)


def _flatten(c, dims):
    flat = c[..., -1]
    for axis in range(c.shape[-1] - 2, -1, -1):
        flat = flat * dims[axis] + c[..., axis]
    return flat


@partial(jit, static_argnums=(5,))
def _cell_counts(points, valid, lo, cell, dims, n_cells_pad):
    flat = _flatten(_cell_coords(points, lo, cell, dims), dims)
    flat = jnp.where(valid, flat, n_cells_pad)
    return jax.ops.segment_sum(valid.astype(jnp.int32), flat, num_segments=n_cells_pad + 1)[:n_cells_pad]


@partial(jit, static_argnums=(5, 6))
def _cell_table(points, valid, lo, cell, dims, n_cells_pad, capacity):
    n = points.shape[0]
    flat = _flatten(_cell_coords(points, lo, cell, dims), dims)
    flat = jnp.where(valid, flat, n_cells_pad)
    order = jnp.argsort(flat, stable=True)
    sorted_flat = flat[order]
    counts = jax.ops.segment_sum(jnp.ones(n, jnp.int32), flat, num_segments=n_cells_pad + 1)
    starts = jnp.cumsum(counts) - counts
    slot = jnp.arange(n, dtype=jnp.int32) - starts[sorted_flat]
    row = jnp.where((sorted_flat < n_cells_pad) & (slot < capacity), sorted_flat, n_cells_pad)
    table = jnp.full((n_cells_pad + 1, capacity), -1, jnp.int32)
    table = table.at[row, jnp.clip(slot, 0, capacity - 1)].set(order.astype(jnp.int32), mode="drop")
    return table[:n_cells_pad]


def build_cell_table(grid: CellGrid, points, valid):
    """Device cell table for padded ``points`` (host or device); capacity is bucketed."""
    args = grid.device_args()
    pts = jnp.asarray(points, float)
    val = jnp.asarray(valid, bool)
    counts = _cell_counts(pts, val, *args, grid.n_cells_pad)
    capacity = bucket(int(jnp.max(counts)), 16)
    return _cell_table(pts, val, *args, grid.n_cells_pad, capacity)


def _ring_candidates(query, table, offsets, lo, cell, dims):
    c = _cell_coords(query, lo, cell, dims)
    neighbour = c[None, :] + offsets
    inside = jnp.all((neighbour >= 0) & (neighbour < dims), axis=1)
    flat = _flatten(jnp.clip(neighbour, 0, dims - 1), dims)
    rows = table[flat]
    rows = jnp.where(inside[:, None], rows, -1)
    return rows.reshape(-1)


@partial(jit, static_argnums=(10,))
def _conflict_counts(queries, query_radii, query_ids, points, radii, valid, table, offsets, lo_cell_dims, scale, chunk):
    lo, cell, dims = lo_cell_dims

    def one(q, rq, qi):
        ids = _ring_candidates(q, table, offsets, lo, cell, dims)
        ok = ids >= 0
        safe = jnp.where(ok, ids, 0)
        d2 = jnp.sum((points[safe] - q) ** 2, axis=1)
        thr = scale * jnp.maximum(rq, radii[safe])
        hit = ok & valid[safe] & (safe != qi) & (d2 < thr * thr)
        return jnp.sum(hit)

    blocks = queries.reshape(-1, chunk, queries.shape[1])
    rb = query_radii.reshape(-1, chunk)
    ib = query_ids.reshape(-1, chunk)
    return lax.map(lambda a: jax.vmap(one)(*a), (blocks, rb, ib)).reshape(-1)


@partial(jit, static_argnums=(10, 11))
def _conflict_lists(queries, query_radii, query_ids, points, radii, valid, table, offsets, lo_cell_dims, scale, chunk, width):
    lo, cell, dims = lo_cell_dims

    def one(q, rq, qi):
        ids = _ring_candidates(q, table, offsets, lo, cell, dims)
        ok = ids >= 0
        safe = jnp.where(ok, ids, 0)
        d2 = jnp.sum((points[safe] - q) ** 2, axis=1)
        thr = scale * jnp.maximum(rq, radii[safe])
        hit = ok & valid[safe] & (safe != qi) & (d2 < thr * thr)
        order = jnp.argsort(~hit, stable=True)[:width]
        return jnp.where(hit[order], ids[order], -1)

    blocks = queries.reshape(-1, chunk, queries.shape[1])
    rb = query_radii.reshape(-1, chunk)
    ib = query_ids.reshape(-1, chunk)
    return lax.map(lambda a: jax.vmap(one)(*a), (blocks, rb, ib)).reshape(-1, width)


def _chunk_for(n_pad: int, ring: int, capacity_hint: int = 32) -> int:
    # A function of the padded size only (bounded temporaries for capacity <= 256),
    # so the chunk never adds a compile variant of its own.
    del ring, capacity_hint
    return int(min(n_pad, 2048))


def conflict_lists(points, radii, valid, *, scale: float = 1.0):
    """Padded neighbour lists of the symmetric conflict graph.

    ``points``/``radii``/``valid`` (host or device) must already be padded to a power of two.
    Two valid points conflict iff ``|x - y| < scale * max(r_x, r_y)``.
    Returns a device int32 array ``(n_pad, width)`` with -1 padding.
    """
    pts = jnp.asarray(points, float)
    rad = jnp.asarray(radii, float)
    val = jnp.asarray(valid, bool)
    n_pad = pts.shape[0]
    lo, hi, any_valid = masked_box(pts, val)
    rmax = float(jnp.max(jnp.where(val, rad, -jnp.inf))) * float(scale) if any_valid else 1.0
    if not any_valid:
        lo, hi = np.zeros(pts.shape[1]), np.ones(pts.shape[1])
    grid = CellGrid(lo, hi, max(rmax, 1e-300))
    table = build_cell_table(grid, jnp.where(val[:, None], pts, jnp.asarray(lo)), val)
    offsets = jnp.asarray(grid.offsets)
    lcd = grid.device_args()
    ids = jnp.arange(n_pad, dtype=jnp.int32)
    chunk = _chunk_for(n_pad, len(grid.offsets), table.shape[1])
    counts = _conflict_counts(pts, rad, ids, pts, rad, val, table, offsets, lcd, float(scale), chunk)
    width = bucket(int(jnp.max(jnp.where(val, counts, 0))), 32)
    return _conflict_lists(pts, rad, ids, pts, rad, val, table, offsets, lcd, float(scale), chunk, width)


@jit
def _greedy(neighbors, rank, state0):
    pad = neighbors < 0
    nb = jnp.where(pad, 0, neighbors)
    higher = (~pad) & (rank[nb] < rank[:, None])

    def cond(carry):
        state, _ = carry
        return jnp.any(state == _UNDECIDED)

    def body(carry):
        state, rounds = carry
        ns = state[nb]
        any_in = jnp.any((ns == _IN) & ~pad, axis=1)
        blocked = jnp.any((ns == _UNDECIDED) & higher, axis=1)
        undecided = state == _UNDECIDED
        state = jnp.where(undecided & any_in, _OUT, state)
        state = jnp.where(undecided & ~any_in & ~blocked, _IN, state)
        return state, rounds + 1

    return lax.while_loop(cond, body, (state0, jnp.int32(0)))


def greedy_independent_set(neighbors, rank, *, fixed=None, excluded=None):
    """Lexicographically first maximal independent set in ``rank`` order.

    ``rank``: unique integers, lower = higher priority. ``fixed`` points are
    in the set from the start (their mutual conflicts are ignored);
    ``excluded`` points are out from the start. Returns (in_set, rounds).
    """
    rank = jnp.asarray(rank, jnp.int32)
    state = jnp.zeros(rank.shape[0], jnp.int32)
    if excluded is not None:
        state = jnp.where(jnp.asarray(excluded), _OUT, state)
    if fixed is not None:
        state = jnp.where(jnp.asarray(fixed), _IN, state)
    state, rounds = _greedy(jnp.asarray(neighbors), rank, state)
    return state == _IN, int(rounds)


@partial(jit, static_argnums=(7, 8))
def _nearest_within(queries, points, valid, table, offsets, lo_cell_dims, radius, chunk, ordered):
    lo, cell, dims = lo_cell_dims

    def one(q):
        ids = _ring_candidates(q, table, offsets, lo, cell, dims)
        ok = ids >= 0
        safe = jnp.where(ok, ids, 0)
        diff = points[safe] - q
        d2 = jnp.where(ok & valid[safe], ordered_sq_norm(diff) if ordered else jnp.sum(diff ** 2, axis=1), jnp.inf)
        if ordered:     # ties: lowest point index
            j = jnp.argmin(jnp.where(d2 == jnp.min(d2), safe, jnp.iinfo(jnp.int32).max))
        else:
            j = jnp.argmin(d2)
        d = jnp.sqrt(d2[j])
        return jnp.where(d <= radius, d, jnp.inf), jnp.where(d <= radius, safe[j], -1)

    blocks = queries.reshape(-1, chunk, queries.shape[1])
    d, i = lax.map(lambda b: jax.vmap(one)(b), blocks)
    return d.reshape(-1), i.reshape(-1)


def nearest_within(queries, points, radius: float, *, ordered: bool = False):
    """Distance and index of the nearest point, exact when <= ``radius`` (else inf, -1).

    Host or device inputs; device outputs. ``ordered``: distances are computed as
    ``sqrt(sum_k (x_k - y_k)^2)`` in coordinate order without contraction (bit-identical to
    numpy/scipy) and ties go to the lowest point index.
    """
    q = jnp.asarray(queries, float)
    p = jnp.asarray(points, float)
    nq, npnt = q.shape[0], p.shape[0]
    if nq == 0:
        return jnp.zeros(0), jnp.zeros(0, jnp.int32)
    if npnt == 0:
        return jnp.full(nq, jnp.inf), jnp.full(nq, -1, jnp.int32)
    qp = pad_rows_device(q, bucket(nq), 0.0)
    pp = pad_rows_device(p, bucket(npnt), 0.0)
    valid = jnp.arange(pp.shape[0]) < npnt
    qlo, qhi, _ = masked_box(qp, jnp.arange(qp.shape[0]) < nq)
    plo, phi, _ = masked_box(pp, valid)
    grid = CellGrid(np.minimum(qlo, plo), np.maximum(qhi, phi), max(float(radius), 1e-300))
    table = build_cell_table(grid, jnp.where(valid[:, None], pp, p[0]), valid)
    chunk = _chunk_for(qp.shape[0], len(grid.offsets), table.shape[1])
    d, i = _nearest_within(qp, pp, valid, table, jnp.asarray(grid.offsets), grid.device_args(), float(radius),
                           chunk, bool(ordered))
    return d[:nq], i[:nq]


@partial(jit, static_argnums=(9, 10))
def _sorted_nearest(queries, points, skeys, order, offsets, lo, cell, dims, radius, cap, chunk):
    n = skeys.shape[0]
    big = jnp.iinfo(jnp.int32).max

    def one(q):
        c = jnp.clip(jnp.floor((q - lo) / cell).astype(skeys.dtype), 0, dims - 1)
        nb = c[None, :] + offsets
        inside = jnp.all((nb >= 0) & (nb < dims), axis=1)
        keys = _flatten(jnp.clip(nb, 0, dims - 1), dims)
        j = jnp.searchsorted(skeys, keys, side="left")[:, None] + jnp.arange(cap)
        jj = jnp.clip(j, 0, n - 1)
        hit = (inside[:, None] & (j < n) & (skeys[jj] == keys[:, None])).reshape(-1)
        ids = order[jj].reshape(-1)
        d2 = jnp.where(hit, ordered_sq_norm(points[ids] - q), jnp.inf)
        k = jnp.argmin(jnp.where(d2 == jnp.min(d2), ids, big))      # ties: lowest point index
        d = jnp.sqrt(d2[k])
        return jnp.where(d <= radius, d, jnp.inf), jnp.where(d <= radius, ids[k], -1)

    d, i = lax.map(lambda b: jax.vmap(one)(b), queries.reshape(-1, chunk, queries.shape[1]))
    return d.reshape(-1), i.reshape(-1)


@jit
def _sorted_keys(points, valid, lo, cell, dims):
    c = jnp.clip(jnp.floor((points - lo) / cell).astype(dims.dtype), 0, dims - 1)
    keys = jnp.where(valid, _flatten(c, dims), jnp.iinfo(dims.dtype).max)
    order = jnp.argsort(keys, stable=True)
    skeys = keys[order]
    first = jnp.searchsorted(skeys, skeys, side="left")
    run = jnp.where(skeys < jnp.iinfo(dims.dtype).max, jnp.arange(skeys.shape[0]) - first + 1, 0)
    return skeys, order.astype(jnp.int32), jnp.max(run)


def nearest_within_sparse(queries, points, radius: float):
    """``nearest_within(..., ordered=True)`` for small radii: cells of width ``radius`` addressed by
    sorted cell keys (memory O(N), not O(number of cells)). Device outputs; ties go to the lowest index."""
    q = jnp.asarray(queries, float)
    p = jnp.asarray(points, float)
    nq, npnt = q.shape[0], p.shape[0]
    if nq == 0:
        return jnp.zeros(0), jnp.zeros(0, jnp.int32)
    if npnt == 0:
        return jnp.full(nq, jnp.inf), jnp.full(nq, -1, jnp.int32)
    if not jax.config.jax_enable_x64:
        raise ValueError("nearest_within_sparse needs jax_enable_x64 (int64 cell keys)")
    qp = pad_rows_device(q, bucket(nq), 0.0)
    pp = pad_rows_device(p, bucket(npnt), 0.0)
    valid = jnp.arange(pp.shape[0]) < npnt
    qlo, qhi, _ = masked_box(qp, jnp.arange(qp.shape[0]) < nq)
    plo, phi, _ = masked_box(pp, valid)
    lo, hi = np.minimum(qlo, plo), np.maximum(qhi, phi)
    cell = max(float(radius), 1e-300)
    lo = lo - 1e-9 * max(1.0, float(np.max(np.abs(hi - lo))))
    dims = np.maximum(1, np.ceil((hi - lo) / cell).astype(np.int64) + 1)
    if float(np.prod(dims.astype(float))) > 2.0**62:
        raise ValueError("cell keys overflow; increase the radius")
    dims_d = jnp.asarray(dims, jnp.int64)
    lo_d, cell_d = jnp.asarray(lo), jnp.asarray(cell)
    skeys, order, run = _sorted_keys(pp, valid, lo_d, cell_d, dims_d)
    cap = bucket(int(run), 4)
    return tuple(a[:nq] for a in _sorted_nearest(qp, pp, skeys, order, jnp.asarray(_offsets(q.shape[1]), jnp.int64),
                                                 lo_d, cell_d, dims_d, float(radius), cap, _chunk_for(qp.shape[0], 0)))


@partial(jit, static_argnums=(8, 9))
def _tile_knn(queries, points, valid, qtable, qcells, ptable, offsets, dims, k, batch):
    """Cell-blocked kNN: all queries of one cell share the points of the surrounding cells.

    qtable (T, Q): query indices of each tile (-1 padded); qcells (T, d): the tile's cell.
    The candidates of a tile are sorted by index, so ties go to the lower index (as dense top-k).
    """
    big = jnp.iinfo(jnp.int32).max

    def tile(qids, cell):
        nb = cell[None, :] + offsets
        inside = jnp.all((nb >= 0) & (nb < dims), axis=1)
        cand = jnp.where(inside[:, None], ptable[_flatten(jnp.clip(nb, 0, dims - 1), dims)], -1).reshape(-1)
        cand = jnp.sort(jnp.where(cand >= 0, cand, big))
        ok = cand < big
        safe = jnp.where(ok, cand, 0)
        ok = ok & valid[safe]
        q = queries[jnp.maximum(qids, 0)]
        from kernelpack.kernels import squared_distances
        d2 = squared_distances(q, points[safe])     # the dense search's formula, so rounding and ties agree
        d2 = jnp.where(ok[None, :], d2, jnp.inf)
        neg, pos = lax.top_k(-d2, k)
        return safe[pos], -neg

    T = qtable.shape[0]
    qb = qtable.reshape(T // batch, batch, -1)
    cb = qcells.reshape(T // batch, batch, -1)
    return lax.map(lambda a: jax.vmap(tile)(*a), (qb, cb))


def grid_knn(queries: np.ndarray, points: np.ndarray, k: int, valid: np.ndarray | None = None,
             memory_budget: float = 2e9):
    """Exact k nearest neighbours by cell-blocked brute force (O(N k) work, bounded memory).

    The points are binned in a uniform grid of width w. All queries in one cell form a tile whose
    candidates are the points of the 3^d surrounding cells; the tile's distances are one dense block
    and its k nearest are found by a top-k. Every point within w of a query is a candidate, so the
    result is exact for every query whose k-th distance is at most w; the few others are searched
    densely. Ties are broken by the lower point index, as in a dense top-k over all points.
    ``memory_budget`` (bytes) bounds the distance blocks held at once.
    Returns (indices (nq, k) into ``points``, distances (nq, k)).
    """
    q = np.atleast_2d(np.asarray(queries, float))
    p = np.atleast_2d(np.asarray(points, float))
    valid = np.ones(len(p), bool) if valid is None else np.asarray(valid, bool)
    nq, npnt, dim = q.shape[0], p.shape[0], p.shape[1]
    n_valid = int(valid.sum())
    if not 0 < k <= n_valid:
        raise ValueError("k must be between 1 and the number of valid points")
    lo = np.minimum(q.min(axis=0), p[valid].min(axis=0))
    hi = np.maximum(q.max(axis=0), p[valid].max(axis=0))
    extent = np.maximum(hi - lo, 1e-12 * max(1.0, float(np.max(hi - lo))))
    width = 1.2 * (float(np.prod(extent)) * k / n_valid) ** (1.0 / dim)
    while True:
        grid = CellGrid(lo, hi, width)
        pp = pad_rows(p, bucket(npnt), 0.0)
        vp = pad_rows(valid, pp.shape[0], False)
        ptable = build_cell_table(grid, np.where(vp[:, None], pp, p[valid][0]), vp)
        if len(grid.offsets) * int(ptable.shape[1]) >= k:
            break
        width *= 1.5
    # Tiles: the queries of each occupied cell.
    cc = np.clip(np.floor((q - grid.lo) / grid.cell).astype(np.int64), 0, grid.dims - 1)
    flat = np.zeros(nq, np.int64)
    for axis in range(dim - 1, -1, -1):
        flat = flat * grid.dims[axis] + cc[:, axis]
    order = np.argsort(flat, kind="stable")
    cells, start, count = np.unique(flat[order], return_index=True, return_counts=True)
    cap = bucket(int(count.max()), 8)
    n_tiles = len(cells)
    rank = np.arange(nq) - np.repeat(start, count)
    qtable = np.full((n_tiles, cap), -1, np.int64)
    qtable[np.repeat(np.arange(n_tiles), count), rank] = order
    qcells = cc[order[start]]
    n_cand = len(grid.offsets) * int(ptable.shape[1])
    per_tile = 4.0 * cap * n_cand * 8
    batch = int(max(1, min(bucket(n_tiles, 1), 1 << int(np.floor(np.log2(max(1.0, memory_budget / per_tile)))))))
    t_pad = -(-n_tiles // batch) * batch
    qtable = pad_rows(qtable, t_pad, -1)
    qcells = pad_rows(qcells, t_pad, 0)
    qp = pad_rows(q, bucket(nq), 0.0)
    i, d2 = _tile_knn(jnp.asarray(qp), jnp.asarray(pp), jnp.asarray(vp), jnp.asarray(qtable.astype(np.int32)),
                      jnp.asarray(qcells.astype(np.int32)), ptable, jnp.asarray(grid.offsets),
                      jnp.asarray(grid.dims.astype(np.int32)), int(k), batch)
    i = np.asarray(i).reshape(t_pad * cap, k)
    d2 = np.asarray(d2).reshape(t_pad * cap, k)
    rows = qtable.reshape(-1)
    keep = rows >= 0
    out_i = np.empty((nq, k), np.int64)
    out_d = np.empty((nq, k))
    out_i[rows[keep]] = i[keep]
    out_d[rows[keep]] = np.sqrt(d2[keep])
    # Queries whose k-th neighbour lies beyond the guaranteed radius: dense search.
    bad = np.flatnonzero(out_d[:, -1] > grid.cell)
    if len(bad):
        from kernelpack.domain.stencils import exact_knn
        vj = jnp.asarray(valid)
        pj = jnp.asarray(p)
        for s in range(0, len(bad), 1024):
            rows_b = bad[s:s + 1024]
            bi, bd = exact_knn(jnp.asarray(q[rows_b]), pj, vj, k=int(k))
            out_i[rows_b], out_d[rows_b] = np.asarray(bi), np.asarray(bd)
    return out_i, out_d


@partial(jit, static_argnums=(1, 2))
def _uniform(key, n, dim, lo, hi):
    return lo + jax.random.uniform(key, (n, dim), dtype=jnp.float64) * (hi - lo)


@jit
def _compact_first(cand, ok):
    """Accepted rows first (input order); the remaining rows repeat the first accepted row,
    so the padded array has the same bounding box as the accepted rows."""
    idx = jnp.argsort(~ok, stable=True)
    n = jnp.sum(ok)
    rows = jnp.arange(cand.shape[0])
    return cand[jnp.where(rows < n, idx, idx[0])]


@partial(jit, static_argnums=(7,))
def _pass_layout(prior, prior_r, n_prior, cand, cand_r, n_cand, perm, n_pad):
    """[prior rows, candidate rows, zero padding] and the dart-throwing ranks: prior rows keep their
    index, candidate j gets n_prior + perm[j], padding keeps its index."""
    rows = jnp.arange(n_pad)
    ip = jnp.clip(rows, 0, prior.shape[0] - 1)
    ic = jnp.clip(rows - n_prior, 0, cand.shape[0] - 1)
    in_p = rows < n_prior
    in_c = (rows >= n_prior) & (rows < n_prior + n_cand)
    pts = jnp.where(in_p[:, None], prior[ip], jnp.where(in_c[:, None], cand[ic], 0.0))
    rad = jnp.where(in_p, prior_r[ip], jnp.where(in_c, cand_r[ic], 0.0))
    rank = jnp.where(in_c, n_prior + perm[ic], rows)
    return pts, rad, in_p | in_c, in_p, rank


@jit
def _pass_accept(pts, rad, in_set, rank, n_prior, n):
    """Prior rows followed by the selected candidates in rank order (zero padded), and their count."""
    rows = jnp.arange(pts.shape[0])
    take = (rows >= n_prior) & (rows < n) & in_set
    order = jnp.argsort(jnp.where(take, rank, jnp.iinfo(rank.dtype).max), stable=True)
    n_take = jnp.sum(take)
    src = jnp.where(rows < n_prior, rows, order[jnp.clip(rows - n_prior, 0, pts.shape[0] - 1)])
    keep = rows < n_prior + n_take
    return jnp.where(keep[:, None], pts[src], 0.0), jnp.where(keep, rad[src], 0.0), n_take


def poisson_disk_sample(
    lo,
    hi,
    radius_of: Callable,
    min_radius: float,
    *,
    seed: int = 0,
    accept: Callable | None = None,
    fixed_points=None,
    fixed_radii=None,
    oversample: float = 6.0,
    passes: int = 4,
    max_candidates: int = 1 << 23,
    draw: Callable[[np.random.Generator, int], np.ndarray] | None = None,
    n_draw: int | None = None,
):
    """Variable-radius Poisson-disk sampling as greedy independent sets.

    Each pass draws uniform candidates at ``oversample`` per ``min_radius**d``,
    keeps those passing ``accept``, and selects the greedy independent set in a
    random priority order (dart throwing) against all points accepted so far
    and the ``fixed_points``. Conflict rule: ``|x - y| < max(r_x, r_y)``.
    Refill passes stop when a pass adds nothing. Returns the accepted points (device).
    ``draw(rng, n)`` replaces the uniform box candidates (for example points on
    a sphere, with ``lo, hi`` a box containing them); ``n_draw`` then sets the
    number of candidates per pass.

    The random draws (candidates, priorities) come from a seeded host RNG and are
    uploaded once per pass; everything else stays on the device. ``accept`` and
    ``radius_of`` receive a device array of ``n_draw`` rows (for ``radius_of``: the
    accepted candidates first, the rest repeating the first one) and may return host
    or device arrays.
    """
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    dim = lo.size
    volume = float(np.prod(np.maximum(hi - lo, 1e-300)))
    if n_draw is None:
        n_draw = oversample * volume / float(min_radius) ** dim
    n_draw = int(min(max_candidates, max(256, n_draw)))
    n_draw = bucket(n_draw, 256)
    prior = (jnp.zeros((0, dim)) if fixed_points is None else jnp.asarray(fixed_points, float).reshape(-1, dim))
    n_fixed = int(prior.shape[0])
    prior_r = (jnp.full(n_fixed, float(min_radius)) if fixed_radii is None
               else jnp.broadcast_to(jnp.asarray(fixed_radii, float), (n_fixed,)))
    if n_fixed == 0:
        prior, prior_r = jnp.zeros((1, dim)), jnp.zeros(1)
    n_prior = n_fixed
    # Host RNG: O(N) draws/permutations cost milliseconds and add no compile variants.
    rng = np.random.default_rng(int(seed))
    for pass_index in range(int(passes)):
        cand = lo + rng.random((n_draw, dim)) * (hi - lo) if draw is None else np.asarray(draw(rng, n_draw), float)
        cand = jnp.asarray(cand)
        ok = jnp.ones(n_draw, bool) if accept is None else jnp.asarray(accept(cand), bool)
        n_cand = int(jnp.sum(ok))
        if not n_cand:
            break
        cand = _compact_first(cand, ok)
        cand_r = jnp.asarray(radius_of(cand), float)
        n = n_prior + n_cand
        n_pad = bucket(n)
        perm = np.zeros(n_draw, np.int64)
        perm[:n_cand] = rng.permutation(n_cand)
        pts, rad, valid, is_fixed, rank = _pass_layout(prior, prior_r, n_prior, cand, cand_r, n_cand,
                                                       jnp.asarray(perm), n_pad)
        neighbors = conflict_lists(pts, rad, valid)
        in_set, _ = greedy_independent_set(neighbors, rank, fixed=is_fixed, excluded=~valid)
        prior, prior_r, n_take = _pass_accept(pts, rad, in_set, rank, n_prior, n)
        n_take = int(n_take)
        if not n_take:
            break
        n_prior += n_take
    return prior[n_fixed:n_prior] if n_prior > n_fixed else jnp.zeros((0, dim))
