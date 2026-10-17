"""Paired GPU density experiment with identical CFPU ambient covers."""
import argparse
import csv
import hashlib
import json
import subprocess
from functools import partial
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from kernelpack.geometry.cfpu import CFPUOptions, build_cfpu, _evaluate
from kernelpack.geometry.feature_sampling import feature_graded_cfpu_sites
from cfpu_refinement_study import initial_geometry, resample, reference_cloud, mapped_reference, metrics
from parametric_geometry_convergence_study import sample_h_channel_segment
from treasure_chest_geometry import descriptors


@partial(jax.jit,static_argnums=(17,))
def _roots(x,n,h,*args):
    """Dynamic model arrays avoid embedding fit coefficients in compiled programs."""
    grid=jnp.linspace(-h,h,33)
    query=x[:,None,:]+grid[None,:,None]*n[:,None,:]
    values=_evaluate(query.reshape(-1,x.shape[1]),*args).reshape(len(x),len(grid))
    cross=jnp.isfinite(values[:,:-1])&jnp.isfinite(values[:,1:])&(values[:,:-1]*values[:,1:]<=0)
    score=jnp.where(cross,jnp.abs((grid[:-1]+grid[1:])/2)[None,:],jnp.inf)
    chosen=jnp.argmin(score,axis=1); bracketed=jnp.any(cross,axis=1)
    lo,hi=grid[chosen],grid[chosen+1]
    vlo=jnp.take_along_axis(values,chosen[:,None],axis=1)[:,0]
    def step(_,state):
        lo,hi,vlo=state; middle=(lo+hi)/2
        vmid=_evaluate(x+middle[:,None]*n,*args); left=vlo*vmid<=0
        return jnp.where(left,lo,middle),jnp.where(left,middle,hi),jnp.where(left,vlo,vmid)
    lo,hi,_=jax.lax.fori_loop(0,44,step,(lo,hi,vlo))
    offset=(lo+hi)/2; residual=_evaluate(x+offset[:,None]*n,*args)
    good=bracketed&jnp.isfinite(residual)&(jnp.abs(residual)<=1e-10*args[8])
    return offset,residual,good,values[:,16]


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--geometry',choices=['h_channel','treasure_chest'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--h',type=float,nargs='+',default=[.16,.12,.08,.06,.04])
    args=parser.parse_args(); assert jax.default_backend()=='gpu'
    args.output.mkdir(parents=True,exist_ok=False)
    files=['src/kernelpack/geometry/cfpu.py','src/kernelpack/geometry/core.py',
        'src/kernelpack/geometry/feature_sampling.py','src/kernelpack/geometry/parametric_pu.py',
        'src/kernelpack/nodes/core.py','src/kernelpack/nodes/hierarchical.py',
        'scripts/cfpu_refinement_study.py','scripts/cfpu_feature_grading_study.py',
        'scripts/treasure_chest_geometry.py','scripts/parametric_geometry_convergence_study.py']
    options=CFPUOptions(max_sites=4096,max_cover_refinements=0)
    provenance=dict(geometry=args.geometry,levels=args.h,backend=jax.default_backend(),
        devices=[str(d) for d in jax.devices()],timings=False,
        commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        hashes={f:hashlib.sha256(Path(f).read_text().encode()).hexdigest() for f in files},
        options=vars(options),grade_factor=2,feature_bandwidth=2,
        cover_control='graded fit reuses uniform centers, radii, PU taper threshold and far-field scale exactly',
        capacity_control='same 4096-site guard for both; no adaptive cover refinement in this controlled comparison',
        coordinate_input_sites='fixed at 65 per H-channel piece / cap resolution 21',
        abscissa='actual N_fit in 2D, sqrt(actual N_fit) in 3D',
        original_feature_constraints='retained unchanged; additions use one-sided smooth-piece normals')
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (args.output/'runner_at_execution.py').write_text(Path(__file__).read_text())
    print('building fixed coordinate model',flush=True)
    base,descriptions,labels=initial_geometry(args.geometry,args.h[0])
    exact,normal,parameters,pieces=reference_cloud(args.geometry,descriptions,labels)
    if args.geometry=='h_channel':
        features=np.unique(np.vstack([sample_h_channel_segment(i,2) for i in range(12)]),axis=0)
    else: features=descriptors(.01,fit_resolution=21)[0]['feature_sites']
    distance=cKDTree(features).query(exact)[0]
    away=distance>.05*np.linalg.norm(np.ptp(exact,axis=0))
    rows=[]
    for h in args.h:
        try:
            print('resampling h=',h,flush=True)
            surface=resample(base,args.geometry,h)
            surface.build_level_set(options=options)
            baseline=surface.level_set.cfpu_model
            points,normals,sampling=feature_graded_cfpu_sites(surface,factor=2,bandwidth=2,backend='warp')
            print('fitting graded cloud',sampling,flush=True)
            graded=build_cfpu(surface.segments,points,normals,options=options,cover=baseline)
            assert np.array_equal(graded.centers,baseline.centers)
            assert np.array_equal(graded.radii,baseline.radii)
            assert graded.threshold==baseline.threshold and graded.scale==baseline.scale
            fitted,_=mapped_reference(surface,exact,parameters,pieces)
            for name,model,fit,nr in [('uniform',baseline,np.asarray(surface.xb_uniform),np.asarray(surface.nrmls_uniform)),
                                      ('graded',graded,points,normals)]:
                offset,residual,good,potential=map(np.asarray,_roots(jnp.asarray(exact),jnp.asarray(normal),h,*model.args()))
                assert np.isfinite(potential).all(),'nonfinite exact-boundary potential'
                assert len(np.unique(fit,axis=0))==len(fit),'duplicate fit sites'
                nfit=len(fit)
                row=dict(h=h,mode=name,n_fit=nfit,axis_value=float(nfit**(1/(exact.shape[1]-1))),
                    supports=model.diagnostics['patch_count'],**metrics(offset,good),
                    away_query_count=int(away.sum()),away_root_failures=int((~good[away]).sum()),
                    away_rms=metrics(offset[away],good[away])['rms_normal_displacement'],
                    near_query_count=int((~away).sum()),near_root_failures=int((~good[~away]).sum()),
                    near_rms=metrics(offset[~away],good[~away])['rms_normal_displacement'],
                    rms_potential=float(np.sqrt(np.mean(potential**2))),
                    near_rms_potential=float(np.sqrt(np.mean(potential[~away]**2))),
                    max_root_residual=float(np.nanmax(np.abs(residual))),
                    max_fit_residual=model.diagnostics['max_boundary_residual'],
                    coordinate_rms_position_error=float(np.sqrt(np.mean(np.sum((fitted-exact)**2,axis=1)))),
                    min_reference_distance_to_fit=float(cKDTree(fit).query(exact)[0].min()),
                    sampling=sampling if name=='graded' else dict(added_count=0),model=model.diagnostics)
                np.savez(args.output/f'{name}_h{h:g}.npz',exact=exact,normal=normal,pieces=pieces,away=away,
                    feature_distance=distance,fit_sites=fit,fit_normals=nr,
                    displacement=offset,residual=residual,good=good,potential=potential,
                    centers=np.asarray(model.centers),radii=np.asarray(model.radii))
                rows.append(row)
                (args.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
                print(json.dumps({k:v for k,v in row.items() if k not in ('sampling','model')}),flush=True)
            # Keep paired models alive only within one level. Release shape-
            # dependent compiled workspaces before the next density increment.
            del baseline,graded
            jax.clear_caches()
        except Exception as error:
            (args.output/'failure.json').write_text(json.dumps(dict(h=h,error=repr(error),completed=len(rows)),indent=2)+'\n')
            raise
    fields=[k for k in rows[0] if k not in ('sampling','model')]
    with (args.output/'accuracy.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=fields); writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in rows)


if __name__=='__main__': main()
