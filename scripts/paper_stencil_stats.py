"""Stencil selection statistics on the paper clouds (no solves).

For every cloud and order p, the stencils of the physical nodes are the n = 2 binom(p+1+d, d) + 1 nearest of all
nodes. We count the centers whose n Euclidean nearest nodes include a node hidden behind the boundary
(a segment that leaves the domain, Section "Stencil selection"), and the centers whose boundary-aware stencil
differs from the Euclidean one.

Usage (WSL venv):
  PYTHONPATH=src:scripts python scripts/paper_stencil_stats.py CLOUD_DIR OUT.jsonl [GEOMETRIES [SPACINGS]]
"""
import json
import sys
import time
from math import comb
from pathlib import Path

import numpy as np

import darcy_forward_convergence_study as darcy
import stencil_level_set

cloud_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
geometries = sys.argv[3].split(",") if len(sys.argv) > 3 else ("h_channel", "treasure_chest")
spacings = [float(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else (0.16, 0.12, 0.08, 0.06, 0.04)
with out.open("w") as f:
    for geometry in geometries:
        for h in spacings:
            path = cloud_dir / f"{geometry}_h{h:.2f}_s11.npz"
            domain, _ = darcy.load_domain_file(path)
            stencil_level_set.attach(domain, geometry, h)
            x = np.asarray(domain.get_int_bdry_nodes())
            d = x.shape[1]
            for p in (2, 4, 6):
                n = 2 * comb(p + 1 + d, d) + 1
                domain.stencil_stats = {}
                t0 = time.perf_counter()
                domain.query_knn("all", x, n)
                s = dict(domain.stencil_stats)
                rec = dict(geometry=geometry, h=h, order=p, n=n, centers=len(x),
                           hidden=s.get("hard", 0), changed=s.get("changed", 0),
                           fallback_rows=s.get("fallback_rows", 0), seconds=time.perf_counter() - t0)
                rec["hidden_fraction"] = rec["hidden"] / rec["centers"]
                rec["changed_fraction"] = rec["changed"] / rec["centers"]
                print(json.dumps(rec), flush=True)
                f.write(json.dumps(rec) + "\n")
                f.flush()
