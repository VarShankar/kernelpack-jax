"""Velocity fields of the paper's advection problems on plotting grids (figure fig:velocity).

The fields are those of the runs: reference_velocity.forward_velocity (advection-diffusion, and the
advection-diffusion-reaction problem with the obstacle, scaled by 1.18) and the true transport velocity
sum_i theta*_i w_i of both basis sets (paper_inverse.py). In 3D we sample the section y = 0; the forward
field curl(Psi e_y) lies in that plane, the transport field has an out-of-plane component, which enters the
speed. Writes Outputs_and_results/paper_figures/velocity_fields.npz (read by scripts/paper_figures.py).

Usage (WSL, CPU is enough): JAX_PLATFORMS=cpu PYTHONPATH=src:scripts .venv-wsl/bin/python scripts/paper_velocity_fields.py
"""
from pathlib import Path

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import reference_velocity as rv
from cfpu_classification_study import inside
from advection_velocity_inverse_convergence_study import theta_reference

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Outputs_and_results/paper_figures/velocity_fields.npz"
SPEED = 1.18
GRIDS = {  # geometry: (dimension, x range, second-axis range, spacing)
    "h_channel": (2, (0.0, 2.0), (-0.06, 1.06), 0.01),
    "treasure_chest": (3, (0.0, 1.8), (-0.35, 0.35), 0.01),
    "obstacle_chest": (3, (0.0, 1.8), (-0.35, 0.35), 0.01),
}


def grid(geometry):
    d, (x0, x1), (y0, y1), s = GRIDS[geometry]
    xs, ys = np.arange(x0, x1 + s / 2, s), np.arange(y0, y1 + s / 2, s)
    X, Y = np.meshgrid(xs, ys)
    pts = np.column_stack([X.ravel(), Y.ravel()]) if d == 2 else np.column_stack([X.ravel(), 0 * X.ravel(), Y.ravel()])
    return xs, ys, pts, inside(pts, geometry)


def main():
    out = {}
    for geometry in GRIDS:
        xs, ys, pts, ok = grid(geometry)
        fields = {"forward": SPEED * np.asarray(rv.forward_velocity(geometry, pts[ok]))}
        if geometry != "obstacle_chest":
            for family in ("smooth", "c1"):
                theta = rv.transport_theta(geometry, family, theta_reference(pts.shape[1]), SPEED)
                fields[f"transport_{family}"] = np.asarray(
                    np.einsum("i,ind->nd", np.asarray(theta), np.asarray(rv.transport_basis(geometry, family, pts[ok]))))
        for name, v in fields.items():
            full = np.full((len(pts), pts.shape[1]), np.nan)
            full[ok] = v
            key = f"{geometry}__{name}"
            out[key + "__x"], out[key + "__y"] = xs, ys
            out[key + "__u"] = full[:, 0].reshape(len(ys), len(xs))
            out[key + "__v"] = full[:, -1].reshape(len(ys), len(xs))      # 2D: v_y; 3D section y=0: v_z
            out[key + "__speed"] = np.linalg.norm(full, axis=1).reshape(len(ys), len(xs))
            print(geometry, name, "max speed on grid", float(np.nanmax(out[key + "__speed"])), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **out)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
