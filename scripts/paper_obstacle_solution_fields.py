"""Solution fields of the problem with an internal obstacle (Section 6.5), on the section y = 0.

Evaluates, on the plotting grid of scripts/paper_velocity_fields.py, the manufactured solution u*(T), the
true reaction rate R(u*, x; theta*) and the recovered one R(u*, x; theta_h) of one reaction law, with
theta_h read from the inverse results (no PDE solve). Writes
Outputs_and_results/paper_figures/obstacle_solution.json, drawn by scripts/matlab/paper_plots.m
(obstacle_solution).

Usage (CPU is enough): JAX_PLATFORMS=cpu PYTHONPATH=src:scripts python scripts/paper_obstacle_solution_fields.py
"""
import json
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import advection_diffusion_forward_convergence_study as ad
from paper_inverse import rd_parameterization
from paper_velocity_fields import grid

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Outputs_and_results/paper_inverse_obstacle_20261015/adr/results.jsonl"
OUT = ROOT / "Outputs_and_results/paper_figures/obstacle_solution.json"
FAMILY, H, ORDER, T = "sin_product", 0.06, 4, 0.5


def main():
    rows = [json.loads(l) for l in open(RESULTS)]
    rows = [r for r in rows if r.get("status") == "ok" and r["family"] == FAMILY
            and abs(r["h"] - H) < 1e-9 and r["order"] == ORDER]
    rec = rows[-1]                                   # latest record wins
    xs, ys, pts, ok = grid("obstacle_chest")
    x = jnp.asarray(pts[ok])
    theta_true, reaction_of = rd_parameterization(FAMILY, 3)
    u = ad.exact_state(T, x)
    r_true = reaction_of(jnp.asarray(rec["truth_theta"]), u, x)
    r_fit = reaction_of(jnp.asarray(rec["fit_theta"]), u, x)

    def full(v):
        f = np.full(len(pts), np.nan)
        f[ok] = np.asarray(v)
        return f.reshape(len(ys), len(xs)).tolist()

    out = dict(x=xs.tolist(), y=ys.tolist(), family=FAMILY, h=H, order=ORDER,
               parameter_rel_error=rec["parameter_rel_error"], field_rel_error=rec["field_rel_error"],
               u_exact=full(u), reaction_true=full(r_true), reaction_fit=full(r_fit),
               reaction_error=full(jnp.abs(r_fit - r_true)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, allow_nan=True).replace("NaN", "null"))
    print("wrote", OUT, "| max |R_h - R*| =", float(jnp.max(jnp.abs(r_fit - r_true))),
          "| field rel error", rec["field_rel_error"])


if __name__ == "__main__":
    main()
