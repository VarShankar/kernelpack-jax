"""Shaded 3D rendering of the treasure chest with an internal obstacle (results, obstacle example).

The outer chest is translucent; the obstacle (a rotated, scaled copy of the chest, see
obstacle_chest_geometry.py) is opaque, with its boundary nodes that face the viewer. Writes
KernelPackJaxPaper/Images/obstacle_nodes.pdf. Same view and style as paper_chest_nodes_figure.py.

Usage (WSL): JAX_PLATFORMS=cpu PYTHONPATH=src:scripts .venv-wsl/bin/python scripts/paper_obstacle_nodes_figure.py [h]
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import paper_chest_nodes_figure as chest
from obstacle_chest_geometry import to_local, to_world

ROOT = Path(__file__).resolve().parents[1]
CLOUDS = ROOT / "Outputs_and_results/paper_clouds_obstacle_20261015"
OUT = ROOT / "KernelPackJaxPaper/Images"


def quads(patches):
    out = []
    for X, Y, Z in patches:
        P = np.stack(np.broadcast_arrays(X, Y, Z), axis=-1)
        q = np.stack([P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]], axis=2)
        out.append(q.reshape(-1, 4, 3))
    return np.concatenate(out)


def collection(q, rgb, alpha):
    nrm = np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1])
    ok = np.linalg.norm(nrm, axis=1) > 1e-14
    q, nrm = q[ok], nrm[ok] / np.linalg.norm(nrm[ok], axis=1, keepdims=True)
    light = np.array([-0.45, -0.55, 1.0]); light /= np.linalg.norm(light)
    shade = 0.28 + 0.72 * np.abs(nrm @ light)
    fc = np.column_stack([rgb[0] * shade, rgb[1] * shade, rgb[2] * shade, np.full(len(q), alpha)])
    coll = Poly3DCollection(q, facecolors=fc, edgecolors=fc if alpha >= 1 else "none",
                            linewidths=0.3 if alpha >= 1 else 0, zsort="average")
    coll.set_rasterized(True)
    return coll


def main(h=0.04):
    z = np.load(CLOUDS / f"obstacle_chest_h{h:.2f}_s11.npz")
    xb, nb = z["bdry_nodes"], z["normals"]
    q = to_local(xb)
    on_obstacle = (q[:, 0] > -0.02) & (q[:, 0] < 1.82) & (np.abs(q[:, 1]) < 0.62) & (np.abs(q[:, 2]) < 0.37)
    d = chest.view_dir()
    fig, ax = chest.axes()
    outer = collection(quads(chest.surface_patches()), (0.80, 0.80, 0.83), 0.18)
    outer.set_zorder(1)
    ax.add_collection3d(outer)
    inner_q = to_world(quads(chest.surface_patches(40)).reshape(-1, 3)).reshape(-1, 4, 3)
    inner = collection(inner_q, (0.80, 0.80, 0.83), 1.0)
    inner.set_zorder(2)
    ax.add_collection3d(inner)
    # obstacle normals point into the obstacle; its visible side has -n_b . d > 0
    front = on_obstacle & (-(nb @ d) > 0.05)
    p = xb[front]
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=7, c=[(0.85, 0.37, 0.05)], depthshade=False, linewidths=0,
               zorder=5, rasterized=True)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "obstacle_nodes.pdf", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("boundary", len(xb), "on obstacle", int(on_obstacle.sum()), "drawn", int(front.sum()))


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.04)
