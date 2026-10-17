"""Convergence of the node generator's boundary representation in the density of the INPUT data.

The node spacing h is not a geometry refinement parameter: the geometry seen by the node
generator is fixed by the sampled input data. This script refines the input data and
measures both geometry layers against the exact boundary on independent test points.

Inputs given as samples (everything else is exact):
  * H-channel: all 12 pieces are given as n equispaced-in-parameter samples
    (paper: n = 65, validation 2n+1 = 131) and fit by the parametric PU
    (scripts/cfpu_refinement_study.initial_geometry). The 10 straight pieces are reproduced
    exactly by the r^3 + P_1 local fits, so only the two sine walls y = -/+0.06 sin(pi x/2)
    (+1) carry parametric error.
  * Treasure chest: the 9 planar faces are exact affine pieces given by their outlines;
    the cap ((x-L)/D)^2 + |y/A|^4 + |z/B|^4 = 1, x >= L, is given as points + exact normals on
    an (s, t) grid controlled by fit_resolution (paper: 41; treasure_chest_geometry.descriptors),
    fit by one parametric PU.
  * Implicit layer phi (CFPU): fit to boundary sites drawn from the parametric layer at
    spacing h (feature graded by 2), CFPUOptions(feature_grade_factor=2, max_sites=512,
    farfield='band') exactly as scripts/generate_paper_clouds.py and stencil_level_set.py
    (base parametric fit built at h=.16, then resample(base, geometry, h, seed=11)).
    Its own input density is therefore the site spacing h.

Studies
  A  parametric layer vs input sample spacing delta (fit_resolution / samples per piece).
  B1 implicit layer vs CFPU site spacing h at the paper input resolution (41 / 65).
  B2 implicit layer vs input sample spacing delta at fixed h = .04 (finest paper h).

Error definitions (all lengths in domain units)
  Test points: random exact boundary points, fixed seed, independent of all fit data.
  Parametric (A, and the cap in B after any cover refits): for exact x on a curved piece,
    u = parameterization.project(x), y = F(u) (fitted map), m = fitted unit normal at u.
    dist   = distance of y to the exact piece surface (extended smoothly): wall = Newton
             closest point on the sine graph; cap = gradient projection onto g = 0
             (4 steps p <- p - g grad g/|grad g|^2; distance = |p - y|, relative error O(dist*kappa)).
    angle  = arccos |m . n_exact(foot)| in degrees; 'flips' counts m . n_exact < 0
             relative to the majority sign of the piece.
    Reported max and RMS over all test points and over points >= e_par = .02 from the piece's
    edges (wall: distance to its endpoints; cap: distance to the rim curve).
  Implicit (B): d_phi(x) = |phi(x)| / |grad phi(x)| on exact boundary points of the whole
    boundary (first-order distance from x to the zero level). Max and RMS over all points and
    over points >= e_phi = .05 from every feature curve (corners/edges/rim, where phi rounds
    the crease), over points >= 2h from every feature curve (the rounding zone scales with the
    CFPU site spacing h), and binned by feature distance in units of h. Sign test: x +- eps n_exact, eps in {.02, .005}; truth = exact inside test;
    error rate = fraction with (phi > 0) != inside.
  Spacing delta: wall = max distance between consecutive samples; cap = median nearest-
    neighbour distance of the (unique) cap data sites (max NN distance also recorded).
  Slopes: least-squares fit of log(error) vs log(delta) (or log h) over all successful levels,
    plus adjacent-pair orders.

Usage (WSL, GPU):
  PYTHONPATH=src:scripts JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv-wsl/bin/python scripts/paper_geometry_convergence.py
Writes Outputs_and_results/paper_geometry_convergence_20261014/results.json.
"""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from kernelpack.accelerators.jax_spatial import enable_compilation_cache
enable_compilation_cache()

from kernelpack.geometry import CFPUOptions, ParametricSurfaceData, PiecewiseSmoothEmbeddedSurface
from cfpu_classification_study import inside
from cfpu_refinement_study import initial_geometry, resample
from parametric_geometry_convergence_study import sample_h_channel_segment, h_channel_exact_segments
from run_paper_geometry_accuracy import h_channel_normals
from treasure_chest_geometry import cap_normal, L, D, A, B, P

AMP = .06
E_PAR, E_PHI, EPS = .02, .05, (.02, .005)
RNG_SEED = 20261014


# ---------------------------------------------------------------- geometry builders
def build_h_channel(n):
    """Paper H-channel parametric layer with n samples per piece (initial_geometry uses n=65)."""
    descriptions = []
    for index in range(12):
        data = sample_h_channel_segment(index, n)
        check = sample_h_channel_segment(index, 2*n+1)
        descriptions.append(ParametricSurfaceData.from_geometry(data, [data[[0, -1]]],
            validation_positions=check, validation_normals=h_channel_normals(index, check)))
    surface = PiecewiseSmoothEmbeddedSurface()
    surface.generate_piecewise_smooth_surface_by_segment(descriptions,
        [i in (1, 2, 4, 6, 8, 10) for i in range(12)], .16)
    return surface, descriptions


def build(geometry, level):
    if geometry == 'h_channel':
        surface, descriptions = build_h_channel(level)
        return surface, descriptions, [str(i) for i in range(12)]
    return initial_geometry(geometry, .16, fit_resolution=level)


def spacing(geometry, level, descriptions, labels):
    if geometry == 'h_channel':
        x = sample_h_channel_segment(0, level)
        return dict(delta=float(np.linalg.norm(np.diff(x, axis=0), axis=1).max()), n_samples_per_piece=level)
    cap = np.unique(np.round(np.asarray(descriptions[labels.index('cap')].positions), 13), axis=0)
    nn = cKDTree(cap).query(cap, 2)[0][:, 1]
    return dict(delta=float(np.median(nn)), delta_max_nn=float(nn.max()), n_cap_sites=len(cap))


# ---------------------------------------------------------------- exact test points
def wall_points(index, m, rng):
    t = rng.uniform(0, 1, m)
    base, sign = (0., -1.) if index == 0 else (1., 1.)
    return np.column_stack([2*t, base + sign*AMP*np.sin(np.pi*t)])


def cap_points(m, rng):
    ang = rng.uniform(0, 2*np.pi, m)
    c, s_ = np.cos(ang), np.sin(ang)
    rim = np.column_stack([A*np.sign(c)*np.abs(c)**(2/P), B*np.sign(s_)*np.abs(s_)**(2/P)])
    s = np.where(np.arange(m) % 2 == 0, rng.uniform(0, 1, m), (1-rng.uniform(0, 1, m)**2)**(1/P))
    return np.column_stack([L + D*np.sqrt(np.maximum(0, 1-s**P)), s[:, None]*rim])


def rim_cloud(m=40000):
    ang = np.linspace(0, 2*np.pi, m, endpoint=False)
    c, s_ = np.cos(ang), np.sin(ang)
    return np.column_stack([np.full(m, L), A*np.sign(c)*np.abs(c)**(2/P), B*np.sign(s_)*np.abs(s_)**(2/P)])


def chest_feature_cloud(step=1e-4):
    def seg(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        t = np.linspace(0, 1, int(np.ceil(np.linalg.norm(b-a)/step))+1)
        return a + t[:, None]*(b-a)
    parts = [rim_cloud()]
    for x in (0., L):
        for sy in (-1, 1):
            parts.append(seg((x, sy*A, -B), (x, sy*A, B)))
        for sz in (-1, 1):
            parts.append(seg((x, -A, sz*B), (x, A, sz*B)))
    for sy in (-1, 1):
        for sz in (-1, 1):
            parts.append(seg((0, sy*A, sz*B), (L, sy*A, sz*B)))
    return cKDTree(np.vstack(parts))


def boundary_points(geometry, rng, m_curved, m_flat):
    """All exact boundary test points: points, exact normals, curved-piece mask, feature distance."""
    if geometry == 'h_channel':
        pts, nrm = [], []
        for index in range(12):
            if index < 2:
                x = wall_points(index, m_curved, rng)
            else:
                a, b = h_channel_exact_segments()[index]
                t = rng.uniform(0, 1, m_flat)
                x = a + t[:, None]*(b-a)
            pts.append(x); nrm.append(h_channel_normals(index, x))
        x = np.vstack(pts)
        corners = np.unique(h_channel_exact_segments().reshape(-1, 2), axis=0)
        fdist = cKDTree(corners).query(x)[0]
        curved = np.r_[np.ones(2*m_curved, bool), np.zeros(10*m_flat, bool)]
        return x, np.vstack(nrm), curved, fdist
    cap = cap_points(m_curved, rng)
    pts, nrm = [cap], [cap_normal(cap)]
    yz = rng.uniform([-A, -B], [A, B], (m_flat, 2))
    pts.append(np.column_stack([np.zeros(m_flat), yz])); nrm.append(np.tile([-1., 0, 0], (m_flat, 1)))
    for sign in (-1, 1):
        xz = rng.uniform([0, -B], [L, B], (m_flat, 2))
        pts.append(np.column_stack([xz[:, 0], np.full(m_flat, sign*A), xz[:, 1]])); nrm.append(np.tile([0., sign, 0], (m_flat, 1)))
        xy = rng.uniform([0, -A], [L, A], (m_flat, 2))
        pts.append(np.column_stack([xy, np.full(m_flat, sign*B)])); nrm.append(np.tile([0., 0, sign], (m_flat, 1)))
    yz = rng.uniform([-A, -B], [A, B], (8*m_flat, 2))
    yz = yz[(np.abs(yz[:, 0])/A)**P + (np.abs(yz[:, 1])/B)**P > 1][:m_flat]
    pts.append(np.column_stack([np.full(len(yz), L), yz])); nrm.append(np.tile([1., 0, 0], (len(yz), 1)))
    x = np.vstack(pts)
    curved = np.zeros(len(x), bool); curved[:len(cap)] = True
    return x, np.vstack(nrm), curved, chest_feature_cloud().query(x)[0]


# ---------------------------------------------------------------- exact projections
def wall_foot(index, y):
    base, a = (0., -AMP) if index == 0 else (1., AMP)
    k = np.pi/2
    x = y[:, 0].copy()
    for _ in range(8):
        c, c1, c2 = base + a*np.sin(k*x), a*k*np.cos(k*x), -a*k*k*np.sin(k*x)
        f1 = (x - y[:, 0]) + (c - y[:, 1])*c1
        f2 = 1 + c1*c1 + (c - y[:, 1])*c2
        x = x - f1/f2
    foot = np.column_stack([x, base + a*np.sin(k*x)])
    return foot, h_channel_normals(index, foot)


def cap_g(p):
    g = ((p[:, 0]-L)/D)**2 + (np.abs(p[:, 1])/A)**P + (np.abs(p[:, 2])/B)**P - 1
    grad = np.column_stack([2*(p[:, 0]-L)/D**2, P*np.sign(p[:, 1])*np.abs(p[:, 1])**(P-1)/A**P,
                            P*np.sign(p[:, 2])*np.abs(p[:, 2])**(P-1)/B**P])
    return g, grad


def cap_foot(y):
    p = y.copy()
    for _ in range(4):
        g, grad = cap_g(p)
        p = p - (g/np.sum(grad*grad, axis=1))[:, None]*grad
    return p, cap_normal(p)


def stats(v, mask=None):
    v = np.asarray(v) if mask is None else np.asarray(v)[mask]
    if not len(v): return dict(max=None, rms=None, n=0)
    return dict(max=float(np.max(v)), rms=float(np.sqrt(np.mean(v**2))), n=int(len(v)))


def parametric_errors(geometry, surface, descriptions, labels, tests):
    """Distance/normal errors of the fitted parametric layer on the curved pieces."""
    out = {}
    for name, (index, x, edge) in tests.items():
        seg = surface.segments[index]
        u = descriptions[index].parameterization.project(x)
        y = np.asarray(seg.evaluate_parametric(jnp.asarray(u)))
        m = np.asarray(seg.evaluate_parametric_frame(jnp.asarray(u))[-1])
        foot, n = wall_foot(index, y) if geometry == 'h_channel' else cap_foot(y)
        dist = np.linalg.norm(y - foot, axis=1)
        dot = np.sum(m*n, axis=1)
        major = np.sign(np.median(dot))
        angle = np.degrees(np.arccos(np.clip(np.abs(dot), 0, 1)))
        keep = edge >= E_PAR
        out[name] = dict(dist_all=stats(dist), dist_interior=stats(dist, keep),
                         angle_deg_all=stats(angle), angle_deg_interior=stats(angle, keep),
                         coordinate_error_max=float(np.linalg.norm(y - x, axis=1).max()),
                         normal_flips=int(np.sum(np.sign(dot) != major)))
    return out


def curved_tests(geometry, rng, m):
    if geometry == 'h_channel':
        tests = {}
        for index in (0, 1):
            x = wall_points(index, m, rng)
            ends = sample_h_channel_segment(index, 2)
            tests[f'wall_{index}'] = (index, x, cKDTree(ends).query(x)[0])
        return tests
    x = cap_points(m, rng)
    return {'cap': (None, x, cKDTree(rim_cloud()).query(x)[0])}


def phi_errors(geometry, model, x, nrm, curved, fdist, h):
    xs = jnp.asarray(x)
    phi = np.asarray(model.evaluate(xs)); grad = np.asarray(model.gradient(xs))
    gn = np.linalg.norm(grad, axis=1)
    d = np.abs(phi)/gn
    keep = fdist >= E_PHI
    far = fdist >= 2*h
    bands = {}
    for lo, hi in ((0, .5), (.5, 1), (1, 2), (2, 4), (4, np.inf)):
        k = (fdist >= lo*h) & (fdist < hi*h)
        if k.any(): bands[f'{lo}-{hi}h'] = stats(d, k)
    center = np.array([[1., .2]]) if geometry == 'h_channel' else np.array([[.3, 0., 0.]])
    inside_positive = bool(np.asarray(model.evaluate(jnp.asarray(center)))[0] > 0)
    sign = {}
    for eps in EPS:
        q = np.vstack([x + eps*nrm, x - eps*nrm])
        val = np.asarray(model.evaluate(jnp.asarray(q)))
        truth = inside(q, geometry)
        pred = (val > 0) if inside_positive else (val < 0)
        bad = (pred != truth) | ~np.isfinite(val)
        sign[f'eps_{eps:g}'] = dict(error_rate=float(bad.mean()), errors=int(bad.sum()), n=int(len(q)),
                                   errors_beyond_e_phi=int(bad[np.r_[keep, keep]].sum()),
                                   errors_beyond_2h=int(bad[np.r_[far, far]].sum()))
    return dict(nonfinite=int(np.sum(~np.isfinite(phi))), dphi_all=stats(d), dphi_smooth=stats(d, keep),
                dphi_curved_smooth=stats(d, keep & curved), dphi_flat_smooth=stats(d, keep & ~curved),
                dphi_far2h=stats(d, far), dphi_curved_far2h=stats(d, far & curved), dphi_by_feature_distance=bands,
                grad_norm_range=[float(gn.min()), float(gn.max())], inside_positive=inside_positive, sign=sign)


def fit_phi(base, geometry, h, seed=11):
    surface = resample(base, geometry, h, seed=seed)
    surface.build_level_set(options=CFPUOptions(feature_grade_factor=2, max_sites=512, farfield='band'))
    model = surface.level_set.cfpu_model
    xd = np.asarray(surface.level_set.xd)
    nn = cKDTree(xd).query(xd, 2)[0][:, 1]
    info = dict(h=h, n_fit_sites=len(xd), site_spacing_median=float(np.median(nn)),
                cover_refinements=len(model.diagnostics.get('shared_refinement_history', [])),
                patches=model.diagnostics.get('patch_count'),
                max_boundary_residual=model.diagnostics.get('max_boundary_residual'),
                band_fallback_reason=model.diagnostics.get('band_fallback_reason'))
    return surface, model, info


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(y) & (y > 0)
    if ok.sum() < 2: return None
    lx, ly = np.log(x[ok]), np.log(y[ok])
    return dict(ls=float(np.polyfit(lx, ly, 1)[0]), adjacent=[float(v) for v in np.diff(ly)/np.diff(lx)])


def get(row, path):
    for k in path.split('/'):
        if row is None or k not in row: return np.nan
        row = row[k]
    return np.nan if row is None else row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--geometries', nargs='+', default=['h_channel', 'treasure_chest'])
    p.add_argument('--hch-samples', type=int, nargs='+', default=[21, 25, 33, 41, 49, 65, 97, 129])
    p.add_argument('--chest-fit', type=int, nargs='+', default=[11, 16, 21, 31, 41, 61, 81])
    p.add_argument('--phi-h', type=float, nargs='+', default=[.16, .12, .08, .06, .04, .03])
    p.add_argument('--phi-fixed-h', type=float, default=.04)
    p.add_argument('--output', type=Path, default=Path('Outputs_and_results/paper_geometry_convergence_20261014'))
    a = p.parse_args()
    assert jax.default_backend() == 'gpu'
    a.output.mkdir(parents=True, exist_ok=True)
    files = ['scripts/paper_geometry_convergence.py', 'scripts/cfpu_refinement_study.py', 'scripts/treasure_chest_geometry.py',
             'src/kernelpack/geometry/core.py', 'src/kernelpack/geometry/cfpu.py', 'src/kernelpack/geometry/parametric_pu.py',
             'src/kernelpack/geometry/feature_sampling.py']
    result = dict(provenance=dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                                  dirty=bool(subprocess.check_output(['git', 'status', '--porcelain', '--', *files], text=True).strip()),
                                  sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
                                  devices=[str(d) for d in jax.devices()], test_seed=RNG_SEED),
                  settings=dict(e_par=E_PAR, e_phi=E_PHI, sign_eps=list(EPS), paper_level=dict(h_channel=65, treasure_chest=41),
                                cfpu=dict(feature_grade_factor=2, max_sites=512, farfield='band'), base_h=.16, seed=11),
                  definitions=__doc__)
    out_path = a.output / 'results.json'
    t_all = time.perf_counter()
    for geometry in a.geometries:
        levels = a.hch_samples if geometry == 'h_channel' else a.chest_fit
        paper = 65 if geometry == 'h_channel' else 41
        rng = np.random.default_rng(RNG_SEED)
        tests = curved_tests(geometry, rng, 20000)
        bx, bn, bcurved, bfd = boundary_points(geometry, rng, 20000, 4000)
        G = result.setdefault(geometry, dict(A=[], B1=[], B2=[], failures=[]))
        G['n_test_points'] = dict(parametric=sum(len(v[1]) for v in tests.values()), implicit=len(bx))
        cap_index = None
        bases = {}
        # ---- A (and B2 at fixed h)
        for level in levels:
            t0 = time.perf_counter()
            try:
                base, desc, labels = build(geometry, level)
            except Exception as err:
                G['failures'].append(dict(study='A', level=level, error=repr(err)[:400])); print(geometry, level, 'FAIL', repr(err)[:200], flush=True)
                continue
            if geometry != 'h_channel':
                cap_index = labels.index('cap')
                tests = {'cap': (cap_index, tests['cap'][1], tests['cap'][2])}
            row = dict(level=level, **spacing(geometry, level, desc, labels),
                       pu_patches={k: base.segments[v[0]].geom_model['model'].diagnostics.get('patch_count') for k, v in tests.items()},
                       **parametric_errors(geometry, base, desc, labels, tests))
            row['time_s'] = round(time.perf_counter() - t0, 1)
            G['A'].append(row)
            if level == paper: bases['paper'] = (base, desc, labels)
            print(geometry, 'A', json.dumps({k: row[k] for k in ('level', 'delta', 'time_s')}),
                  {k: (row[k]['dist_all']['max'], row[k]['angle_deg_all']['max']) for k in tests}, flush=True)
            t0 = time.perf_counter()
            try:
                surface, model, info = fit_phi(base, geometry, a.phi_fixed_h)
                info.update(level=level, delta=row['delta'], **phi_errors(geometry, model, bx, bn, bcurved, bfd, info['h']))
                info['parametric_after_refits'] = parametric_errors(geometry, surface, desc, labels, tests)
                info['time_s'] = round(time.perf_counter() - t0, 1)
                G['B2'].append(info)
                print(geometry, 'B2', level, info['dphi_all']['max'], info['dphi_smooth']['max'], info['cover_refinements'], flush=True)
            except Exception as err:
                G['failures'].append(dict(study='B2', level=level, error=repr(err)[:400])); print(geometry, 'B2', level, 'FAIL', repr(err)[:200], flush=True)
            jax.clear_caches()
            out_path.write_text(json.dumps(result, indent=1) + '\n')
        # ---- B1: phi vs site spacing h at the paper input resolution
        base, desc, labels = bases['paper']
        for h in a.phi_h:
            t0 = time.perf_counter()
            try:
                surface, model, info = fit_phi(base, geometry, h)
                info.update(**phi_errors(geometry, model, bx, bn, bcurved, bfd, info['h']))
                info['parametric_after_refits'] = parametric_errors(geometry, surface, desc, labels, tests)
                info['time_s'] = round(time.perf_counter() - t0, 1)
                G['B1'].append(info)
                print(geometry, 'B1', h, info['dphi_all']['max'], info['dphi_smooth']['max'], info['cover_refinements'], flush=True)
            except Exception as err:
                G['failures'].append(dict(study='B1', h=h, error=repr(err)[:400])); print(geometry, 'B1', h, 'FAIL', repr(err)[:200], flush=True)
            jax.clear_caches()
            out_path.write_text(json.dumps(result, indent=1) + '\n')
        # ---- slopes
        S = G['slopes'] = {}
        keys = list(tests)
        for k in keys:
            for metric in ('dist_all/max', 'dist_all/rms', 'dist_interior/max', 'dist_interior/rms',
                           'angle_deg_all/max', 'angle_deg_all/rms', 'angle_deg_interior/max', 'angle_deg_interior/rms'):
                S[f'A/{k}/{metric}'] = slope([r['delta'] for r in G['A']], [get(r, f'{k}/{metric}') for r in G['A']])
        for study, xkey in (('B1', 'h'), ('B2', 'delta')):
            for metric in ('dphi_all/max', 'dphi_all/rms', 'dphi_smooth/max', 'dphi_smooth/rms',
                           'dphi_curved_smooth/max', 'dphi_flat_smooth/max', 'dphi_far2h/max', 'dphi_far2h/rms',
                           'dphi_curved_far2h/max', 'dphi_curved_far2h/rms'):
                S[f'{study}/{metric}'] = slope([r[xkey] for r in G[study]], [get(r, metric) for r in G[study]])
        out_path.write_text(json.dumps(result, indent=1) + '\n')
    result['runtime_s'] = round(time.perf_counter() - t_all, 1)
    out_path.write_text(json.dumps(result, indent=1) + '\n')
    print('done', result['runtime_s'], flush=True)


if __name__ == '__main__':
    main()
