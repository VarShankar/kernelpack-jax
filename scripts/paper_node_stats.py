"""Statistics of the node-generation algorithms on the paper clouds (no solves).

1. Far field: the implicit layer with the band extension vs the generalized winding number (the same CFPU fits,
   only the sign away from the cover differs): sign disagreements and evaluation time at random points in the
   enlarged bounding box of the boundary.
2. Grouped evaluation: work of the grouped sum, sum_e 2^e |G_e|, against N max m for evaluating f_loc at the
   interior nodes, where m(x) is the number of balls containing x.
3. Parallel greedy independent sets: random candidates with radius r in the unit cube; rounds of the parallel
   algorithm and agreement with the sequential greedy algorithm.

Usage (WSL venv):  PYTHONPATH=src:scripts python scripts/paper_node_stats.py CLOUD_DIR OUT.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import jax

from kernelpack.accelerators.jax_spatial import conflict_lists, greedy_independent_set, bucket, pad_rows
import stencil_level_set

cloud_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
result = dict(farfield=[], grouping=[], greedy=[])
rng = np.random.default_rng(0)


def timed(f, *a):
    t0 = time.perf_counter()
    y = np.asarray(f(*a))
    return y, time.perf_counter() - t0


for geometry in ("h_channel", "treasure_chest"):
    for h in (0.16, 0.08, 0.04):
        z = np.load(cloud_dir / f"{geometry}_h{h:.2f}_s11.npz")
        xb, xi = z["bdry_nodes"], z["int_nodes"]
        model = stencil_level_set.implicit_layer(geometry, h)
        lo, hi = xb.min(0), xb.max(0)
        pad = 0.05 * (hi - lo)
        q = rng.uniform(lo - pad, hi + pad, size=(200_000, xb.shape[1]))
        band = model.band
        model.evaluate(q[:1000]); fb, tb = timed(model.evaluate, q)
        model.band = None
        model.evaluate(q[:1000]); fw, tw = timed(model.evaluate, q)
        model.band = band
        result["farfield"].append(dict(geometry=geometry, h=h, points=len(q), disagree=int((np.sign(fb) != np.sign(fw)).sum()),
                                       band_s=tb, winding_s=tw, balls=int(len(np.asarray(model.centers))),
                                       components=band["components"] if band is not None else None))
        c, r = np.asarray(model.centers), np.asarray(model.radii)
        m = np.zeros(len(xi), int)
        for k in range(0, len(c), 256):
            d = np.linalg.norm(xi[:, None, :] - c[None, k:k + 256, :], axis=2)
            m += (d < r[None, k:k + 256]).sum(1)
        mm = np.maximum(m, 1)
        grouped = int((2 ** np.ceil(np.log2(mm))).sum())
        result["grouping"].append(dict(geometry=geometry, h=h, points=len(xi), max_m=int(m.max()), mean_m=float(m.mean()),
                                       grouped_work=grouped, full_work=int(len(xi) * m.max()), exact_work=int(m.sum())))
        print(json.dumps(result["farfield"][-1]), json.dumps(result["grouping"][-1]), flush=True)

for d in (2, 3):
    for n in (10**3, 10**4, 10**5, 10**6):
        x = rng.uniform(0, 1, size=(n, d))
        r = np.full(n, (0.6 / n) ** (1 / d))           # about 0.6 conflicts per unit volume ball per point
        npad = bucket(n)
        xp, rp = pad_rows(x, npad), pad_rows(r, npad, 1.0)
        valid = np.arange(npad) < n
        nb = conflict_lists(xp, rp, valid)
        rank = np.concatenate([rng.permutation(n), np.arange(n, npad)]).astype(np.int32)
        t0 = time.perf_counter()
        ins, rounds = greedy_independent_set(nb, rank, excluded=~valid)
        tp = time.perf_counter() - t0
        ins = np.asarray(ins)[:n]
        nbh = np.asarray(nb)[:n]
        seq = np.zeros(n, bool)
        for i in np.argsort(rank[:n]):
            j = nbh[i][nbh[i] >= 0]
            if not seq[j[j < n]].any():
                seq[i] = True
        result["greedy"].append(dict(dim=d, candidates=n, accepted=int(ins.sum()), rounds=rounds,
                                     identical=bool((ins == seq).all()), parallel_s=tp,
                                     max_degree=int((nbh >= 0).sum(1).max())))
        print(json.dumps(result["greedy"][-1]), flush=True)

out.write_text(json.dumps(result, indent=1))
