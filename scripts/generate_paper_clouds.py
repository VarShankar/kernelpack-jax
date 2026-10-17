"""Generate and validate the paper's node clouds (curved H-channel, treasure chest).

One coordinate fit per geometry; each (h, seed) gets its own boundary and
interior draw. Every cloud is checked against the exact geometry:
interior nodes inside, ghosts outside, boundary nodes on the surface,
normals against the analytic outward normal, and minimum node separation.

Geometry settings (see CLAUDE.md, cloud audit 2026-10-10): the chest cap is fit
with fit_resolution=41 (was 21). CFPU cover failures refit the shared cap cover
on the same data; with 21 the cap error grew from 8e-5 (h=.16) to 6e-4 (h=.04),
with 41 it stays near 1e-4. CFPU max_sites stays at 512: 2048 avoids the refits
but thins the near-boundary node layer (~7% fewer nodes within 1.5h, 3x fewer
within .75h) and made ghost-closed Laplacians unstable on several chest clouds.

Interior clipping uses clearance_mode='seeds' (kernelpack nodes/core.py): boundary
and ghost nodes act as fixed Poisson-disk seeds (interior nodes >= .6h from all of
them) and the level set only rejects points with phi < .2h. The old phi >= .6h clip
treats the CFPU level set as a distance; near convex edges it is much smaller than
the distance, which left boundary nodes with no interior neighbour within 2-3h and
made the ghost-closed Laplacian unstable on 4/54 chest/H-channel (h<=.08) pairs
(1/54 with 'seeds', next to an H-channel reentrant slot corner).
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import time

import jax
import numpy as np
from scipy.spatial import cKDTree

# Fixed-shape GPU kernels compile once per size bucket; keep them across runs.
from kernelpack.accelerators.jax_spatial import enable_compilation_cache
enable_compilation_cache()

from kernelpack.geometry import CFPUOptions
from kernelpack.nodes import DomainNodeGenerator
from cfpu_classification_study import inside
from cfpu_refinement_study import initial_geometry, resample
from treasure_chest_geometry import cap_normal, L, D, A, B, P

H_AMPLITUDE = .06
# Straight pieces of the H-channel boundary: endpoints and outward normal.
H_SEGMENTS = [((0, 0), (0, .4), (-1, 0)), ((2, 0), (2, .4), (1, 0)), ((0, .6), (0, 1), (-1, 0)),
              ((2, .6), (2, 1), (1, 0)), ((.9, .4), (.9, .6), (-1, 0)), ((1.1, .4), (1.1, .6), (1, 0)),
              ((0, .4), (.9, .4), (0, 1)), ((0, .6), (.9, .6), (0, -1)),
              ((1.1, .4), (2, .4), (0, 1)), ((1.1, .6), (2, .6), (0, -1))]


def h_channel_boundary_check(xb, normals):
    """Distance to the exact boundary and normal error away from corners."""
    best = np.full(len(xb), np.inf)
    exact_normal = np.full(xb.shape, np.nan)
    xs = np.linspace(0, 2, 400001)
    for base, sign in ((0., -1.), (1., 1.)):
        curve = np.column_stack([xs, base + sign * H_AMPLITUDE * np.sin(np.pi * xs / 2)])
        dist, idx = cKDTree(curve).query(xb)
        slope = sign * H_AMPLITUDE * np.pi / 2 * np.cos(np.pi * xs[idx] / 2)
        n = np.column_stack([-slope, np.ones_like(slope)]) * sign
        n /= np.linalg.norm(n, axis=1)[:, None]
        closer = dist < best
        best[closer] = dist[closer]; exact_normal[closer] = n[closer]
    for a, b, n in H_SEGMENTS:
        a, b = np.asarray(a, float), np.asarray(b, float)
        u = np.clip((xb - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
        dist = np.linalg.norm(xb - a - u[:, None] * (b - a), axis=1)
        closer = dist < best
        best[closer] = dist[closer]
        # Corner nodes (segment endpoints) have no unique normal.
        exact_normal[closer] = np.where(((u > 0) & (u < 1))[closer, None], np.asarray(n, float), np.nan)
    ok = np.isfinite(exact_normal).all(axis=1)
    angle = np.degrees(np.arccos(np.clip(np.sum(normals[ok] * exact_normal[ok], axis=1), -1, 1)))
    return best, angle


def chest_boundary_check(xb, normals, tol=1e-9):
    cap = xb[:, 0] > L + tol
    planes = np.column_stack([np.abs(xb[:, 0]), np.abs(np.abs(xb[:, 1]) - A),
                              np.abs(np.abs(xb[:, 2]) - B), np.abs(xb[:, 0] - L)])
    dist = planes.min(axis=1)
    x = xb[cap]
    grad = np.column_stack([2 * (x[:, 0] - L) / D**2, P * np.sign(x[:, 1]) * np.abs(x[:, 1])**(P - 1) / A**P,
                            P * np.sign(x[:, 2]) * np.abs(x[:, 2])**(P - 1) / B**P])
    level = ((x[:, 0] - L) / D)**2 + np.abs(x[:, 1] / A)**P + np.abs(x[:, 2] / B)**P
    dist[cap] = np.abs(level - 1) / np.linalg.norm(grad, axis=1)
    angle = [np.degrees(np.arccos(np.clip(np.sum(normals[cap] * cap_normal(x), axis=1), -1, 1)))]
    # Planar face nodes lying on exactly one face plane (not edges/corners/shoulder rim;
    # rim nodes carry the bisector of the cap and shoulder normals by design).
    on = planes < tol
    rim = np.abs(np.abs(xb[:, 1] / A)**P + np.abs(xb[:, 2] / B)**P - 1) < 1e-6
    single = (on.sum(axis=1) == 1) & ~(rim & on[:, 3])
    face_normals = [(-1, 0, 0), None, None, (1, 0, 0)]
    for k in range(4):
        sel = ~cap & on[:, k] & single
        if not sel.any():
            continue
        if face_normals[k] is None:
            n = np.zeros((sel.sum(), 3)); n[:, k] = np.sign(xb[sel, k])
        else:
            n = np.broadcast_to(np.asarray(face_normals[k], float), (sel.sum(), 3))
        angle.append(np.degrees(np.arccos(np.clip(np.sum(normals[sel] * n, axis=1), -1, 1))))
    return dist, np.concatenate(angle)


def obstacle_check(xb, normals):
    """Exact distance to the outer chest or to the obstacle, and normal errors (chest_boundary_check, applied to
    the obstacle in its local frame and scaled back). Points are assigned to the nearer of the two surfaces."""
    import obstacle_chest_geometry as og
    d_out, _ = chest_boundary_check(xb, normals)
    q = og.to_local(xb)
    n_local = -(normals @ og.R)                         # obstacle normals point into it; outward in the local chest
    d_in, _ = chest_boundary_check(q, n_local)
    d_in = og.SCALE * d_in
    outer = d_out <= d_in
    _, angle_out = chest_boundary_check(xb[outer], normals[outer])
    _, angle_in = chest_boundary_check(q[~outer], n_local[~outer])
    return np.where(outer, d_out, d_in), np.concatenate([angle_out, angle_in])


def validate(geometry, h, xi, xb, xg, normals):
    check = dict(h_channel=h_channel_boundary_check, obstacle_chest=obstacle_check).get(geometry, chest_boundary_check)
    dist, angle = check(xb, normals)
    # At reentrant corners a ghost can lie on an adjacent boundary edge; only
    # ghosts strictly inside the domain are errors.
    ghost_to_boundary, _ = check(xg, np.tile(np.eye(xg.shape[1])[:1], (len(xg), 1)))
    xall = np.vstack([xi, xb, xg])
    sep = cKDTree(xall).query(xall, 2)[0][:, 1]
    return dict(n_interior=len(xi), n_boundary=len(xb),
                interior_outside=int(np.sum(~inside(xi, geometry))),
                ghosts_inside=int(np.sum(inside(xg, geometry) & (ghost_to_boundary > 1e-9))),
                ghosts_on_boundary=int(np.sum(ghost_to_boundary <= 1e-9)),
                boundary_max_distance=float(dist.max()), normal_max_error_deg=float(angle.max()),
                normal_median_error_deg=float(np.median(angle)), min_separation_over_h=float(sep.min() / h),
                ghost_offset_over_h=[float(v / h) for v in np.percentile(np.linalg.norm(xg - xb, axis=1), [0, 100])])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometries', nargs='+', default=['h_channel', 'treasure_chest'])
    p.add_argument('--h', nargs='+', type=float, default=[.16, .12, .08, .06, .04])
    p.add_argument('--seeds', nargs='+', type=int, default=[11, 12, 13])
    p.add_argument('--base-h', type=float, default=.16,
                   help='h at which the coordinate fits are built (fixed so subsets of --h reproduce the full run)')
    p.add_argument('--fit-resolution', type=int, default=41)
    p.add_argument('--cfpu-max-sites', type=int, default=512)
    p.add_argument('--boundary-fraction', type=float, default=1.0,
                   help='Boundary sampling radius as a fraction of h. Sampling at .6h did not remove the '
                        'near-boundary gaps and overflows the CFPU cover at h=.04.')
    p.add_argument('--clearance-mode', choices=('seeds', 'level_set'), default='seeds',
                   help="'seeds': interior nodes keep >= .6h from boundary and ghost nodes and phi >= .2h. "
                        "'level_set': the old phi >= .6h clip, which leaves 2-3h holes at convex edges.")
    p.add_argument('--sampler', choices=('gpu', 'bridson'), default='gpu',
                   help="'gpu': interior Poisson-disk sampling as greedy independent sets on device.")
    p.add_argument('--no-boundary-refinement', action='store_true',
                   help='Uniform interior spacing h everywhere (no .6h layer within 1.5h of the boundary).')
    p.add_argument('--farfield', choices=('winding', 'band'), default='band',
                   help="Sign of the implicit layer outside the CFPU cover: winding number, or connected-component band sign.")
    p.add_argument('--output', type=Path, default=Path('Outputs_and_results/paper_clouds_20261010'))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    files = ['scripts/generate_paper_clouds.py', 'scripts/cfpu_refinement_study.py', 'scripts/treasure_chest_geometry.py',
             'src/kernelpack/geometry/core.py', 'src/kernelpack/geometry/cfpu.py', 'src/kernelpack/geometry/parametric_pu.py',
             'src/kernelpack/geometry/feature_sampling.py', 'src/kernelpack/nodes/core.py', 'src/kernelpack/nodes/hierarchical.py',
             'src/kernelpack/accelerators/jax_spatial.py']
    provenance = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                      dirty=bool(subprocess.check_output(['git', 'status', '--porcelain', '--', *files], text=True).strip()),
                      sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
                      options=dict(fit_resolution=a.fit_resolution, cfpu_max_sites=a.cfpu_max_sites,
                                   boundary_fraction=a.boundary_fraction, clearance_mode=a.clearance_mode,
                                   farfield=a.farfield, boundary_refinement=not a.no_boundary_refinement,
                                   surface_guard_fraction=.2, sampler=a.sampler, feature_grade_factor=2, outer_fraction_of_h=.6,
                                   outer_refinement_zone_size_as_multiple_of_h=1.5))
    (a.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    report_path = a.output / 'validation.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for geometry in a.geometries:
        start = time.perf_counter()
        base, _, _ = initial_geometry(geometry, a.base_h, fit_resolution=a.fit_resolution)
        print(f'{geometry}: coordinate fits {time.perf_counter() - start:.1f}s', flush=True)
        for h in a.h:
            for seed in a.seeds:
                name = f'{geometry}_h{h:.2f}_s{seed}'
                if (a.output / f'{name}.npz').exists() and name in report:
                    continue
                timing = {}
                start = time.perf_counter()
                surface = resample(base, geometry, a.boundary_fraction * h, seed=seed, fit_resolution=a.fit_resolution)
                timing['boundary_s'] = time.perf_counter() - start
                start = time.perf_counter()
                surface.build_level_set(options=CFPUOptions(feature_grade_factor=2, max_sites=a.cfpu_max_sites,
                                                                farfield=a.farfield))
                timing['cfpu_s'] = time.perf_counter() - start
                refinements = len(surface.level_set.cfpu_model.diagnostics.get('shared_refinement_history', []))
                start = time.perf_counter()
                domain = DomainNodeGenerator().build_domain_descriptor_from_geometry(
                    surface, h, backend='warp', seed=seed, do_outer_refinement=not a.no_boundary_refinement,
                    outer_fraction_of_h=.6, outer_refinement_zone_size_as_multiple_of_h=1.5,
                    clearance_mode=a.clearance_mode, surface_guard_fraction=.2, sampler=a.sampler)
                timing['interior_s'] = time.perf_counter() - start
                xi, xb, xg, nr = (np.asarray(v) for v in (domain.get_interior_nodes(), domain.get_bdry_nodes(),
                                                           domain.get_ghost_nodes(), domain.get_nrmls()))
                result = validate(geometry, h, xi, xb, xg, nr)
                result['cfpu_cover_refinements'] = refinements
                result['timing'] = {k: round(v, 2) for k, v in timing.items()}
                np.savez(a.output / f'{name}.npz', int_nodes=xi, bdry_nodes=xb, ghost_nodes=xg, normals=nr,
                         sep_rad=h, dim=xi.shape[1], geometry=geometry, h=h, seed=seed, label=name)
                report[name] = result
                report_path.write_text(json.dumps(report, indent=2) + '\n')
                print(name, json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
