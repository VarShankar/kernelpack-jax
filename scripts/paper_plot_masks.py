"""Exact domain masks on the display grids of the field plots (velocity, obstacle solution).

The fields are computed on the grids of scripts/paper_velocity_fields.py; for display, scripts/matlab
interpolates them to a grid REFINE times finer with the PHS + polynomial basis of the paper. This script
evaluates the exact inside tests on that finer grid, so that the plotted domain edge does not follow whole
cells of the computation grid. Writes Outputs_and_results/paper_figures/plot_masks.json:
{geometry: {"x": [...], "y": [...], "rows": ["0110...", ...]}}.

Usage (CPU, a few seconds): PYTHONPATH=src:scripts python scripts/paper_plot_masks.py
"""
import json
from pathlib import Path

import numpy as np

from paper_velocity_fields import GRIDS, grid
from cfpu_classification_study import inside

REFINE = 4
OUT = Path(__file__).resolve().parents[1] / "Outputs_and_results/paper_figures/plot_masks.json"


def main():
    out = {}
    for geometry in GRIDS:
        d = GRIDS[geometry][0]
        xs, ys, _, _ = grid(geometry)
        xf = np.linspace(xs[0], xs[-1], REFINE * (len(xs) - 1) + 1)
        yf = np.linspace(ys[0], ys[-1], REFINE * (len(ys) - 1) + 1)
        X, Y = np.meshgrid(xf, yf)
        pts = np.column_stack([X.ravel(), Y.ravel()]) if d == 2 else np.column_stack([X.ravel(), 0 * X.ravel(), Y.ravel()])
        mask = np.asarray(inside(pts, geometry)).reshape(len(yf), len(xf))
        out[geometry] = dict(x=xf.round(6).tolist(), y=yf.round(6).tolist(),
                             rows=["".join("1" if v else "0" for v in row) for row in mask])
        print(geometry, mask.shape, int(mask.sum()), flush=True)
    OUT.write_text(json.dumps(out))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
