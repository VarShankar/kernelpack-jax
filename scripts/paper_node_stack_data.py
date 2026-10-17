"""Data for the node-generation stack figure (treasure chest).

Builds the chest geometry as scripts/generate_paper_clouds.py does
(fit_resolution 41, CFPU max_sites 512) and saves
  - the cap's parametric PU cover (cells in normalized parameters, overlap),
    its trimming loop in normalized parameters, and the cap map on a grid,
  - the CFPU balls at h_cover (cover only: the local fits are not needed for
    the figure and are replaced by zeros, which leaves the cover, including its
    refinement, unchanged), and the background cells of the slice x_3 = 0 with
    their coverage; inside/outside of the free cells is taken from the exact
    chest, which the band sign reproduces exactly (CLAUDE.md, band test),
  - the v4 cloud (seed 11) at h_nodes.
Output: Outputs_and_results/paper_figures/node_stack_chest.npz
"""
import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import kernelpack.geometry.cfpu as cfpu
from kernelpack.geometry import CFPUOptions
from cfpu_refinement_study import initial_geometry, resample
from treasure_chest_geometry import L, D, A, B, P

ROOT = Path(__file__).resolve().parents[1]


def cover_only():
    def zero_solve(sites, normals, mask):
        count, capacity, dim = sites.shape
        return (jnp.zeros_like(sites), jnp.zeros((count, dim)), jnp.zeros((count, capacity)), jnp.zeros(count))
    cfpu._local_solve = zero_solve


def chest_inside(x):
    box = (x[:, 0] > 0) & (x[:, 0] <= L) & (np.abs(x[:, 1]) < A) & (np.abs(x[:, 2]) < B)
    cap = (x[:, 0] > L) & (((x[:, 0] - L) / D) ** 2 + np.abs(x[:, 1] / A) ** P + np.abs(x[:, 2] / B) ** P < 1)
    return box | cap


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h-nodes', type=float, default=.08)
    p.add_argument('--h-cover', type=float, default=.04)
    p.add_argument('--out', type=Path, default=ROOT / 'Outputs_and_results/paper_figures/node_stack_chest.npz')
    a = p.parse_args()
    cover_only()
    base, _, _ = initial_geometry('treasure_chest', .16, fit_resolution=41)
    out = {}
    surface = resample(base, 'treasure_chest', a.h_cover, seed=11, fit_resolution=41)
    surface.build_level_set(options=CFPUOptions(feature_grade_factor=2, max_sites=512, farfield='winding'))
    cap = [s for s in surface.segments if s.geom_model['type'] == 'parametric-pu']
    assert len(cap) == 1
    gm = cap[0].geom_model
    model = gm['model']
    offset, scale = np.asarray(model.offset), np.asarray(model.scale)
    cells = np.array([np.r_[lo, hi] for lo, hi in model.cells])
    out['cap_cells'] = cells
    out['cap_overlap'] = float(np.median(np.asarray(model.widths) / ((cells[:, 2:] - cells[:, :2]) / 2)))
    out['cap_outline'] = (np.asarray(gm['outlines'][0]) - offset) / scale
    g = np.linspace(-1, 1, 41)
    U, V = np.meshgrid(g, g, indexing='ij')
    zg = np.column_stack([U.ravel(), V.ravel()])
    out['cap_grid_z'] = zg
    out['cap_grid_x'] = np.asarray(model.evaluate(zg * scale + offset))

    cf = surface.level_set.cfpu_model
    centers, radii = np.asarray(cf.centers), np.asarray(cf.radii)
    out['ball_centers'], out['ball_radii'] = centers, radii
    cell = .5 * float(radii.min())
    lo = np.array([-.1, -A - .1]); hi = np.array([L + D + .1, A + .1])
    n = np.ceil((hi - lo) / cell).astype(int)
    X, Y = np.meshgrid(lo[0] + (np.arange(n[0]) + .5) * cell, lo[1] + (np.arange(n[1]) + .5) * cell, indexing='ij')
    pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    covered = np.zeros(len(pts), bool)
    for s in range(0, len(centers), 512):
        covered |= np.any(np.linalg.norm(pts[:, None] - centers[None, s:s + 512], axis=2) < radii[None, s:s + 512], axis=1)
    out['slice_cells'] = pts[:, :2]
    out['slice_free'] = ~covered
    out['slice_sign'] = np.where(chest_inside(pts), 1, -1)
    out['slice_cell'] = cell
    out['slice_z'] = 0.
    print('balls', len(radii), 'cap cells', len(cells), 'slice cells', len(pts),
          'free inside', int(((~covered) & chest_inside(pts)).sum()), flush=True)

    cloud = np.load(ROOT / f'Outputs_and_results/paper_clouds_v6_20261014/treasure_chest_h{a.h_nodes:.2f}_s11.npz')
    for key in ('int_nodes', 'bdry_nodes', 'ghost_nodes', 'normals'):
        out[key] = cloud[key]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    print(a.out, {k: np.shape(v) for k, v in out.items()}, flush=True)


if __name__ == '__main__':
    main()
