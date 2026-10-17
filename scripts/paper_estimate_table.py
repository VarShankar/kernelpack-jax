"""Table of the forward error-estimate quantities (paper, forward results).

Reads Outputs_and_results/paper_forward_v4_20261011/estimate_check.jsonl
(scripts/paper_forward_estimate_check.py) and writes
KernelPackJaxPaper/gen_forward_estimate.tex: per geometry and order, the
least-squares slope of the RMS truncation error ||tau||/sqrt(N) against h over
h <= 0.08 (largest over the families), the range of the effective stability
constant kappa_h = ||E||/||tau|| over h <= 0.08 and all families, and the
largest scaled stencil sum lambda_L.
"""
import collections
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Outputs_and_results/paper_forward_v4_20261011/estimate_check.jsonl"
OUT = ROOT / "KernelPackJaxPaper/gen_forward_estimate.tex"
def sci(x):
    e = int(math.floor(math.log10(x)))
    return f"${x / 10 ** e:.1f}\\times10^{{{e}}}$"


NAMES = {"h_channel": "H-channel", "treasure_chest": "treasure chest"}

rows = [json.loads(l) for l in SRC.read_text().splitlines()]
groups = collections.defaultdict(list)
for r in rows:
    groups[(r["geometry"], r["order"])].append(r)

lines = [r"\begin{table}", r"\centering",
         r"\caption{Quantities of the elliptic error estimate \eqref{eq:elliptic_bound} for the Darcy problem, over $h\leq0.08$: "
         r"the least-squares slope of the truncation error $\norm{\tau}_2/\sqrt N$ against $h$ (smallest and largest over the families), "
         r"the range of the effective stability constant $\hat S_h$ of \eqref{eq:effective_stability}, and the largest scaled stencil sum $W_L$ of \eqref{eq:stencil_sums}.}",
         r"\label{tab:forward_estimate}", r"\begin{tabular}{llcccc}", r"\toprule",
         r"Geometry & $p$ & slope of $\norm{\tau}$ & $\min\hat S_h$ & $\max\hat S_h$ & $\max W_L$\\", r"\midrule"]
for geometry in ("h_channel", "treasure_chest"):
    for order in (2, 4, 6):
        rr = [r for r in groups[(geometry, order)] if r["h"] <= 0.08 + 1e-12]
        if not rr:
            continue
        slopes = []
        for fam in sorted({r["family"] for r in rr}):
            pts = sorted((r["h"], r["tau_phys_l2"] / math.sqrt(r["physical_nodes"])) for r in rr if r["family"] == fam)
            hs, ts = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
            slopes.append(np.polyfit(hs, ts, 1)[0])
        kap = [r["kappa"] for r in rr]
        lam = max(r["lam_L"] for r in rr)
        name = NAMES[geometry] if order == 2 else ""
        lines.append(f"{name} & {order} & {min(slopes):.1f}--{max(slopes):.1f} & {sci(min(kap))} & {sci(max(kap))} & {lam:.0f}\\\\")
    if geometry == "h_channel":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
text = "\n".join(lines) + "\n"
OUT.write_text(text, encoding="utf-8")
print(text)
