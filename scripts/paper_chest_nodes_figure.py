"""Shaded 3D renderings of a treasure-chest node set for the paper (Figure 2, bottom row).

Left: interior nodes in front of the shaded surface. Right: boundary nodes on the shaded surface
(only nodes whose normal faces the viewer are drawn). Writes
KernelPackJaxPaper/Images/chest_nodes_{interior,boundary}.pdf.

Usage (WSL): .venv-wsl/bin/python scripts/paper_chest_nodes_figure.py [h]
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[1]
CLOUDS = ROOT / "Outputs_and_results/paper_clouds_v6_20261014"
OUT = ROOT / "KernelPackJaxPaper/Images"

L, A, B, CAP = 1.2, 0.6, 0.35, 0.6          # box (0,L)x(-A,A)x(-B,B), cap length CAP
ELEV, AZIM = 28.0, -58.0

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmb10"], "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True, "font.size": 15, "axes.labelsize": 17,
    "axes.unicode_minus": False,
})


def surface_patches(n=60):
    """(X, Y, Z) grids covering the boundary of the treasure chest."""
    s = np.linspace(0, 1, n)
    pats = []
    # five box faces (the face x=L is replaced by the cap and the rim pieces)
    u, v = np.meshgrid(np.linspace(0, L, n), np.linspace(-A, A, n))
    pats += [(u, v, -B + 0 * u), (u, v, B + 0 * u)]
    u, v = np.meshgrid(np.linspace(0, L, n), np.linspace(-B, B, n))
    pats += [(u, -A + 0 * u, v), (u, A + 0 * u, v)]
    u, v = np.meshgrid(np.linspace(-A, A, n), np.linspace(-B, B, n))
    pats += [(0 * u, u, v)]
    # superelliptic polar coordinates: |y/A|^4 + |z/B|^4 = rho^4
    t = np.linspace(0, 2 * np.pi, 4 * n)
    c, sn = np.cos(t), np.sin(t)
    ey = A * np.sign(c) * np.sqrt(np.abs(c))
    ez = B * np.sign(sn) * np.sqrt(np.abs(sn))
    rho = s[:, None]
    pats.append((L + CAP * np.sqrt(np.clip(1 - rho**4, 0, None)), rho * ey, rho * ez))   # cap
    with np.errstate(divide="ignore"):
        rmax = np.minimum(1 / np.sqrt(np.abs(c)), 1 / np.sqrt(np.abs(sn)))
    r = 1 + s[:, None] * (rmax - 1)
    pats.append((L + 0 * r * ey, r * ey, r * ez))                                         # rim pieces
    return pats


def view_dir():
    e, a = np.radians(ELEV), np.radians(AZIM)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def axes():
    fig = plt.figure(figsize=(4.4, 3.5))
    ax = fig.add_subplot(projection="3d", computed_zorder=False)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_box_aspect((L + CAP + 0.1, 2 * A, 2 * B), zoom=0.95)
    ax.set_xlim(-0.05, L + CAP + 0.05)
    ax.set_ylim(-A, A)
    ax.set_zlim(-B, B)
    lab = lambda v: [rf"$\mathbf{{{x:g}}}$".replace("-", "{-}") for x in v]
    for setter, v in ((ax.set_xticks, [0, 0.6, 1.2, 1.8]), (ax.set_yticks, [0, 0.6]), (ax.set_zticks, [0])):
        setter(v, lab(v))
    ax.set_xlabel(r"$\mathbf{x_1}$", labelpad=12)
    ax.set_ylabel(r"$\mathbf{x_2}$", labelpad=8)
    # mplot3d drops the z label at this view, so place it by hand next to the z ticks
    ax.text2D(0.99, 0.66, r"$\mathbf{x_3}$", transform=ax.transAxes, fontsize=17)
    ax.tick_params(axis="y", pad=1)
    ax.tick_params(axis="z", pad=4)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_alpha(0.0)
        a._axinfo["grid"]["color"] = (0.85, 0.85, 0.85, 1)
    return fig, ax


def draw_surface(ax, alpha):
    """All boundary patches as one depth-sorted polygon collection, Lambert-shaded."""
    quads = []
    for X, Y, Z in surface_patches():
        P = np.stack(np.broadcast_arrays(X, Y, Z), axis=-1)
        q = np.stack([P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]], axis=2)
        quads.append(q.reshape(-1, 4, 3))
    q = np.concatenate(quads)
    nrm = np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1])
    ok = np.linalg.norm(nrm, axis=1) > 1e-14
    q, nrm = q[ok], nrm[ok] / np.linalg.norm(nrm[ok], axis=1, keepdims=True)
    light = np.array([-0.45, -0.55, 1.0]); light /= np.linalg.norm(light)
    shade = 0.28 + 0.72 * np.abs(nrm @ light)
    fc = np.column_stack([0.80 * shade, 0.80 * shade, 0.83 * shade, np.full(len(q), alpha)])
    coll = Poly3DCollection(q, facecolors=fc, edgecolors=fc if alpha >= 1 else "none",
                            linewidths=0.4 if alpha >= 1 else 0, zsort="average")
    coll.set_zorder(1)
    coll.set_rasterized(True)
    ax.add_collection3d(coll)


def main(h=0.08):
    z = np.load(CLOUDS / f"treasure_chest_h{h:.2f}_s11.npz")
    xi, xb, nb = z["int_nodes"], z["bdry_nodes"], z["normals"]
    OUT.mkdir(parents=True, exist_ok=True)
    d = view_dir()

    # interior nodes over a translucent surface; nearer nodes drawn later and darker
    fig, ax = axes()
    draw_surface(ax, 0.35)
    order = np.argsort(xi @ d)
    p = xi[order]
    depth = (p @ d - (p @ d).min()) / np.ptp(p @ d)
    col = plt.cm.Blues(0.55 + 0.4 * depth)
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=2.2, c=col, depthshade=False, linewidths=0, zorder=5, rasterized=True)
    fig.savefig(OUT / "chest_nodes_interior.pdf", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    # boundary nodes facing the viewer on an opaque surface
    fig, ax = axes()
    draw_surface(ax, 1.0)
    front = nb @ d > 0.05
    p = xb[front]
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=9, c=[(0.85, 0.37, 0.05)], depthshade=False, linewidths=0,
               zorder=5, rasterized=True)
    fig.savefig(OUT / "chest_nodes_boundary.pdf", dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("interior", len(xi), "boundary", len(xb), "front", int(front.sum()))


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.08)
