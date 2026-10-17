"""GPU CFPU refinement against actual assembled normal-fit sites, no timings."""
import argparse
import copy
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from kernelpack.geometry import ParametricSurfaceData, PiecewiseSmoothEmbeddedSurface
from kernelpack.nodes import DomainNodeGenerator
from kernelpack.nodes.hierarchical import sample_parametric_curve, sample_parametric_face, sample_planar_face
from parametric_geometry_convergence_study import sample_h_channel_segment, domed_square_patch_points
from run_paper_geometry_accuracy import h_channel_normals
from treasure_chest_geometry import descriptors, cap_normal, L, D, A, B, P


def initial_geometry(geometry, h, fit_resolution=21, seed=11):
    if geometry == 'h_channel':
        descriptions = []
        for index in range(12):
            data = sample_h_channel_segment(index, 65)
            check = sample_h_channel_segment(index, 131)
            descriptions.append(ParametricSurfaceData.from_geometry(data, [data[[0, -1]]],
                validation_positions=check, validation_normals=h_channel_normals(index, check)))
        surface = PiecewiseSmoothEmbeddedSurface()
        surface.generate_piecewise_smooth_surface_by_segment(descriptions,
            [i in (1, 2, 4, 6, 8, 10) for i in range(12)], h)
        return surface, descriptions, [str(i) for i in range(12)]
    if geometry == 'obstacle_chest':
        from obstacle_chest_geometry import descriptors_with_obstacle
        description, labels = descriptors_with_obstacle(h, fit_resolution=fit_resolution)
    else:
        description, labels = descriptors(h, fit_resolution=fit_resolution)
    surface = DomainNodeGenerator().build_piecewise_smooth_boundary_hierarchical(
        **description, radius=h, seed=seed, backend='gpu', method=1, supersample_fac=2)
    return surface, description['bdry_desc'], labels


def resample(base, geometry, h, seed=11, fit_resolution=21):
    """Reuse coordinate fits; use the exact production supersampling/assembly kernels."""
    surface = copy.deepcopy(base)
    for index, segment in enumerate(surface.segments):
        model = segment.geom_model
        if model['type'] == 'parametric-pu':
            sampler = sample_parametric_curve if segment.surf_dim == 1 else sample_parametric_face
            kwargs = {} if segment.surf_dim == 1 else dict(seed=seed+index, backend='gpu')
            uv = sampler(model['model'], model['outlines'], h, supersample_fac=2, **kwargs)
            points = model['model'].evaluate(uv)
            _, normal = model['model'].frame(uv)
            normal *= model.get('normal_sign', 1.)
            model['fit_context']['radius'] = h
            segment.sep_rad = h
        else:
            uv = sample_planar_face(model['outlines'][0], h, seed=seed+index,
                smooth=False, supersample_fac=2, backend='gpu')
            points = jnp.asarray(model['origin']) + uv @ jnp.asarray(model['basis']).T
            normal = jnp.broadcast_to(jnp.asarray(model['normal']), points.shape)
        segment.set_sample_sites(points)
        segment.nrmls = normal
        segment.uniform_sample_sites = points
        segment.uniform_nrmls = normal
    surface.seed = int(seed)
    surface._assemble_boundary(h, **base.assembly_options)
    return surface


def reference_cloud(geometry, descriptions, labels):
    """Fixed independent analytic queries, never used as fit or validation sites."""
    rng = np.random.default_rng(90071)
    points, normals, parameters, pieces = [], [], [], []
    for index, label in enumerate(labels):
        if geometry == 'h_channel':
            # Interpolate random parameter values in a dense analytic curve;
            # use the actual analytic sine, rather than a piecewise-linear reference.
            t = rng.uniform(0, 1, 128)
            endpoints = sample_h_channel_segment(index, 2)
            x = endpoints[0] + t[:, None]*(endpoints[1]-endpoints[0])
            if index < 2:
                x[:, 1] = (0 if index == 0 else 1) + (-1 if index == 0 else 1)*.06*np.sin(np.pi*t)
            n = h_channel_normals(index, x)
            if index in (1, 2, 4, 6, 8, 10): n = -n
            uv = descriptions[index].parameterization.project(x)
        elif label == 'cap':
            x = domed_square_patch_points('dome', 512)
            n = cap_normal(x)
            uv = descriptions[index].parameterization.project(x)
        else:
            count = 128
            if label == 'back':
                yz = rng.uniform([-A, -B], [A, B], (count, 2))
                x = np.column_stack((np.zeros(count), yz)); n = np.tile([-1., 0, 0], (count, 1))
            elif label.startswith('side_y'):
                sign = int(label.rsplit('_', 1)[1])
                xz = rng.uniform([0, -B], [L, B], (count, 2))
                x = np.column_stack((xz[:, 0], np.full(count, sign*A), xz[:, 1]))
                n = np.tile([0., sign, 0], (count, 1))
            elif label.startswith('side_z'):
                sign = int(label.rsplit('_', 1)[1])
                xy = rng.uniform([0, -A], [L, A], (count, 2))
                x = np.column_stack((xy, np.full(count, sign*B)))
                n = np.tile([0., 0, sign], (count, 1))
            else:
                sy, sz = map(int, label.split('_')[1:])
                candidates = rng.uniform([0, 0], [A, B], (4096, 2))
                yz = candidates[(candidates[:, 0]/A)**P+(candidates[:, 1]/B)**P > 1][:count]
                assert len(yz) == count
                x = np.column_stack((np.full(count, L), yz*np.array([sy, sz])))
                n = np.tile([1., 0, 0], (count, 1))
            uv = None
        points.append(x); normals.append(n); parameters.append(uv)
        pieces.extend([index]*len(x))
    return np.vstack(points), np.vstack(normals), parameters, np.array(pieces)


def mapped_reference(surface, points, parameters, pieces):
    fitted, normals = [], []
    for index, segment in enumerate(surface.segments):
        if parameters[index] is None:
            fitted.append(points[pieces == index])
            normals.append(np.tile(segment.geom_model['normal'], (sum(pieces == index), 1)))
        else:
            fitted.append(np.asarray(segment.evaluate_parametric(parameters[index])))
            normals.append(np.asarray(segment.evaluate_parametric_frame(parameters[index])[-1]))
    return np.vstack(fitted), np.vstack(normals)


def normal_roots(model, points, normals, h):
    """Signed normal-line zero-surface displacement, with explicit success mask.

    This is a normal displacement error, not the global nearest-point distance.
    Select the nearest sign-changing interval on a fixed 33-point normal line
    in [-h,h], then bisect. Missing brackets/nonfinite residuals remain failures.
    """
    @jax.jit
    def solve(x, n):
        grid = jnp.linspace(-h, h, 33)
        query = x[:, None, :] + grid[None, :, None]*n[:, None, :]
        values = model.evaluate(query.reshape(-1, x.shape[1])).reshape(len(x), len(grid))
        valid = jnp.isfinite(values[:, :-1]) & jnp.isfinite(values[:, 1:])
        cross = valid & (values[:, :-1]*values[:, 1:] <= 0)
        score = jnp.where(cross, jnp.abs((grid[:-1]+grid[1:])/2)[None, :], jnp.inf)
        chosen = jnp.argmin(score, axis=1)
        bracketed = jnp.any(cross, axis=1)
        lo, hi = grid[chosen], grid[chosen+1]
        vlo = jnp.take_along_axis(values, chosen[:, None], axis=1)[:, 0]
        def step(_, state):
            lo, hi, vlo = state
            middle = (lo+hi)/2
            vmid = model.evaluate(x+middle[:, None]*n)
            left = vlo*vmid <= 0
            return jnp.where(left, lo, middle), jnp.where(left, middle, hi), jnp.where(left, vlo, vmid)
        lo, hi, _ = jax.lax.fori_loop(0, 44, step, (lo, hi, vlo))
        offset = (lo+hi)/2
        residual = model.evaluate(x+offset[:, None]*n)
        good = bracketed & jnp.isfinite(residual) & (jnp.abs(residual) <= 1e-10*model.scale)
        return offset, residual, good
    return tuple(np.asarray(v) for v in solve(jnp.asarray(points), jnp.asarray(normals)))


def metrics(offset, good):
    return dict(rms_normal_displacement=float(np.sqrt(np.mean(offset**2))) if good.all() else None,
        max_normal_displacement=float(np.max(np.abs(offset))) if good.all() else None,
        root_failures=int((~good).sum()), query_count=len(offset))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--geometry', choices=['h_channel', 'treasure_chest'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--h', type=float, nargs='+', default=[.16, .12, .08, .06, .04])
    args = parser.parse_args()
    assert jax.default_backend() == 'gpu'
    args.output.mkdir(parents=True, exist_ok=False)
    sources = ['src/kernelpack/geometry/cfpu.py', 'src/kernelpack/geometry/core.py',
        'src/kernelpack/geometry/parametric_pu.py', 'src/kernelpack/nodes/core.py',
        'src/kernelpack/nodes/hierarchical.py', 'scripts/treasure_chest_geometry.py',
        'scripts/cfpu_refinement_study.py', 'scripts/parametric_geometry_convergence_study.py']
    provenance = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        hashes={f: hashlib.sha256(Path(f).read_text().encode()).hexdigest() for f in sources},
        geometry=args.geometry, levels=args.h, backend=jax.default_backend(),
        devices=[str(d) for d in jax.devices()], timings=False,
        coordinate_input_resolution=65 if args.geometry == 'h_channel' else 21,
        seed=11, reference_seed=90071, coordinate_inputs_fixed=True,
        primary_metric='RMS normal-line displacement to the analytic boundary; fixed independent queries',
        axis='N_fit' if args.geometry == 'h_channel' else 'sqrt(N_fit)',
        count_definition='unique assembled boundary sites actually supplied to CFPU, including features; not sum of overlapping local counts')
    (args.output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    (args.output/'runner_at_execution.py').write_text(Path(__file__).read_text())
    print('building fixed coordinate model', flush=True)
    base, descriptions, labels = initial_geometry(args.geometry, args.h[0])
    exact, normal, parameters, pieces = reference_cloud(args.geometry, descriptions, labels)
    rows = []
    for h in args.h:
        print('sampling and fitting CFPU h=', h, flush=True)
        try:
            surface = resample(base, args.geometry, h)
            surface.build_level_set()
            model = surface.level_set.cfpu_model
            fit = np.asarray(surface.level_set.xd)
            near = cKDTree(fit).query(exact)[0]
            fitted, fitted_normal = mapped_reference(surface, exact, parameters, pieces)
            offset, residual, good = normal_roots(model, exact, normal, h)
            mapped_offset, mapped_residual, mapped_good = normal_roots(model, fitted, fitted_normal, h)
            nfit = len(fit); dimension = exact.shape[1]
            row = dict(h=h, n_fit=nfit, axis_value=float(nfit**(1/(dimension-1))),
                coordinate_input_sites=sum(len(s.data_sites) for s in base.segments if s.geom_model['type']=='parametric-pu'),
                supports=model.diagnostics['patch_count'],
                min_reference_distance_to_fit=float(near.min()), **metrics(offset, good),
                mapped_rms_normal_displacement=metrics(mapped_offset, mapped_good)['rms_normal_displacement'],
                mapped_root_failures=int((~mapped_good).sum()),
                coordinate_rms_position_error=float(np.sqrt(np.mean(np.sum((fitted-exact)**2, axis=1)))),
                max_root_residual=float(np.nanmax(np.abs(residual))),
                max_fit_residual=model.diagnostics['max_boundary_residual'],
                pieces={label: metrics(offset[pieces==i], good[pieces==i]) for i, label in enumerate(labels)},
                model=model.diagnostics)
            np.savez(args.output/f'level_h{h:g}.npz', exact=exact, normal=normal, pieces=pieces,
                mapped=fitted, fit_sites=fit, fit_normals=np.asarray(surface.level_set.nrd),
                displacement=offset, residual=residual, good=good,
                mapped_displacement=mapped_offset, mapped_residual=mapped_residual, mapped_good=mapped_good)
            rows.append(row)
            (args.output/'results.json').write_text(json.dumps(rows, indent=2)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k not in ('pieces','model')}), flush=True)
        except Exception as error:
            (args.output/'failure.json').write_text(json.dumps(dict(h=h, error=repr(error), completed=len(rows)), indent=2)+'\n')
            raise
        # Different boundary capacities trigger new compiled programs. Release
        # prior shapes so a refinement sweep does not accumulate GPU workspaces.
        jax.clear_caches()
    fields = [k for k in rows[0] if k not in ('pieces', 'model')]
    with (args.output/'accuracy.csv').open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=fields); writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in rows)
    x = np.array([r['axis_value'] for r in rows])
    orders = {}
    for metric in ('rms_normal_displacement', 'max_normal_displacement', 'mapped_rms_normal_displacement'):
        if all(r[metric] is not None and r[metric]>0 for r in rows):
            error = np.array([r[metric] for r in rows])
            orders[metric] = dict(adjacent_orders=(-np.diff(np.log(error))/np.diff(np.log(x))).tolist(),
                fitted_order=float(-np.polyfit(np.log(x), np.log(error), 1)[0]))
    (args.output/'orders.json').write_text(json.dumps(orders, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6,4))
    for metric, label in [('rms_normal_displacement','Exact boundary: RMS'),
                          ('max_normal_displacement','Exact boundary: max'),
                          ('mapped_rms_normal_displacement','Coordinate model: RMS')]:
        y = [r[metric] if r[metric] is not None else np.nan for r in rows]
        ax.loglog(x,y,'o-',label=label)
    ax.set_xlabel(r'$N_{\mathrm{fit}}$' if exact.shape[1]==2 else r'$\sqrt{N_{\mathrm{fit}}}$')
    ax.set_ylabel('Normal-line displacement'); ax.set_title(args.geometry.replace('_',' '))
    ax.grid(True,which='both',alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(args.output/'convergence.png',dpi=180); fig.savefig(args.output/'convergence.pdf')
    print(json.dumps(orders), flush=True)


if __name__ == '__main__': main()
