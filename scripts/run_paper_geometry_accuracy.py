"""GPU parametric geometry accuracy studies, with independent fixed check points.

No timing measurements. Original paper outputs are read-only comparison data.
"""
import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kernelpack.geometry import EmbeddedSurface
from parametric_geometry_convergence_study import (
    sample_h_channel_segment, h_channel_exact_segments, relative_errors,
    domed_square_patch_points, _normalized_arc_length_to_curve_parameter,
    H_CHANNEL_CURVE_AMPLITUDE,
)


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def normals_error(fitted, exact):
    dots=np.sum(fitted*exact,axis=1)
    aligned=fitted*np.where(dots>=0,1.,-1.)[:,None]
    difference=aligned-exact
    return dict(normal_rms=float(np.sqrt(np.mean(np.sum(difference**2,axis=1)))),
                normal_max=float(np.linalg.norm(difference,axis=1).max()))


def fit(data, dim, count):
    surface=EmbeddedSurface(); surface.set_data_sites(jnp.asarray(data))
    # Surface accuracy only needs the fitted map; avoid drawing thousands of
    # unused output candidates before evaluating the independent check points.
    surface.build_geometric_model_ps(dim,.08,len(data),ne=count if dim==2 else 1,
                                    method=2,supersample_fac=1)
    assert surface.geom_model['degree']==3
    return surface


def h_channel(count, check_count):
    from kernelpack.geometry import ParametricSurfaceData
    predictions=[]; references=[]; diagnostics=[]
    for index in range(len(h_channel_exact_segments())):
        data=sample_h_channel_segment(index,count)
        exact=sample_h_channel_segment(index,check_count)
        checks=sample_h_channel_segment(index,257)
        description=ParametricSurfaceData.from_geometry(data,[data[[0,-1]]],
            validation_positions=checks,validation_normals=h_channel_normals(index,checks))
        surface=EmbeddedSurface(); surface.set_data_sites(description.positions)
        surface.build_parametric_pu_model(description.parameters,outlines=description.outlines,
            validation_parameters=description.validation_parameters,validation_positions=checks,
            validation_normals=description.validation_normals,options=description.options)
        uv=description.parameterization.project(exact)
        fitted=np.asarray(surface.evaluate_parametric(uv))
        _,normal=surface.evaluate_parametric_frame(uv)
        nr=h_channel_normals(index,exact)
        diagnostics.append(dict(piece=index,data_sites=len(data),check_points=len(exact),
            pu_patches=surface.geom_model['model'].diagnostics['patch_count'],
            **relative_errors(fitted,exact),**normals_error(np.asarray(normal),nr)))
        predictions.append(fitted); references.append(exact)
    return dict(nd_build_points=12*count,nb_eval_points=12*check_count,
                **relative_errors(np.vstack(predictions),np.vstack(references))), diagnostics


def h_channel_normals(index,points):
    if index<2:
        amplitude=(-1 if index==0 else 1)*H_CHANNEL_CURVE_AMPLITUDE
        tangent=np.column_stack((np.full(len(points),2.),amplitude*np.pi*np.cos(np.pi*points[:,0]/2)))
    else:
        a,b=h_channel_exact_segments()[index]
        tangent=np.broadcast_to(b-a,(len(points),2))
    nr=np.column_stack((tangent[:,1],-tangent[:,0]))
    return nr/np.linalg.norm(nr,axis=1)[:,None]


def treasure_chest(resolution, check_count, legacy_charts=False):
    if not legacy_charts:
        return treasure_chest_pu(resolution,check_count)
    from treasure_chest_geometry import descriptors, cap_normal
    description,labels=descriptors(.08,fit_resolution=resolution,use_pu=False)
    predictions=[]; references=[]; diagnostics=[]
    # Same independent exact check cloud as the paper's coordinate-error study.
    # All four cap charts evaluate disjoint subsets of this fixed cap cloud.
    cap=domed_square_patch_points('dome',check_count)
    signs=np.where(cap[:,1:3]>=0,1,-1)
    cap_used=0
    for data,label in zip(description['bdry_desc'],labels):
        if not label.startswith('cap'):
            continue
        sy,sz=map(int,label.split('_')[1:])
        exact=cap[(signs[:,0]==sy)&(signs[:,1]==sz)]
        cap_used+=len(exact)
        surface=fit(data,3,1)
        model=surface.geom_model
        uv=(jnp.asarray(exact)-model['origin'])@model['basis']
        fitted=np.asarray(surface.evaluate_parametric(uv))
        _,_,nr=surface._eval_open_surface_frame(uv)
        diagnostics.append(dict(piece=label,data_sites=len(data),check_points=len(exact),
            **relative_errors(fitted,exact),**normals_error(np.asarray(nr),cap_normal(exact))))
        predictions.append(fitted); references.append(exact)
    assert cap_used==len(cap), 'Each independent cap point must be assigned once'
    # Planar maps are exact affine lifts. Their footprint/feature correctness is
    # checked separately by the hierarchical geometry tests, not by zero errors.
    for piece in ('west','y_min','y_max','z_min','z_max','shoulder'):
        exact=domed_square_patch_points(piece,check_count)
        predictions.append(exact); references.append(exact)
    metrics=relative_errors(np.vstack(predictions),np.vstack(references))
    cap_metrics=relative_errors(np.vstack(predictions[:4]),np.vstack(references[:4]))
    return dict(nd_build_points=sum(len(p) for p in description['bdry_desc']),
        nd_cap_points=sum(d['data_sites'] for d in diagnostics),
        nb_eval_points=7*check_count,cap_rel_l2=cap_metrics['rel_l2'],
        cap_rel_linf=cap_metrics['rel_linf'],**metrics), diagnostics


def treasure_chest_pu(resolution,check_count):
    from treasure_chest_geometry import descriptors,cap_normal
    from kernelpack.geometry.parametric_pu import fit_parametric_pu
    description,_=descriptors(.08,fit_resolution=resolution)
    data=description['bdry_desc'][-1]
    model=fit_parametric_pu(data.parameters,data.positions,
        validation_parameters=data.validation_parameters,validation_positions=data.validation_positions,
        validation_normals=data.validation_normals,options=data.options,
        parameter_bounds=(np.asarray(data.outlines[0]).min(axis=0),np.asarray(data.outlines[0]).max(axis=0)))
    cap=domed_square_patch_points('dome',check_count)
    uv=data.parameterization.project(cap)
    fitted=np.asarray(model.evaluate(uv)); _,nr=model.frame(uv)
    diagnostics=[dict(piece='cap',data_sites=len(data),check_points=len(cap),
        pu_patches=model.diagnostics['patch_count'],**relative_errors(fitted,cap),
        **normals_error(np.asarray(nr),cap_normal(cap)))]
    predictions=[fitted]; references=[cap]
    for piece in ('west','y_min','y_max','z_min','z_max','shoulder'):
        exact=domed_square_patch_points(piece,check_count)
        predictions.append(exact); references.append(exact)
    metrics=relative_errors(np.vstack(predictions),np.vstack(references))
    cap_metrics=relative_errors(fitted,cap)
    return dict(nd_build_points=sum(len(p) for p in description['bdry_desc']),
        nd_cap_points=len(data),nb_eval_points=7*check_count,pu_patches=model.diagnostics['patch_count'],
        cap_rel_l2=cap_metrics['rel_l2'],cap_rel_linf=cap_metrics['rel_linf'],**metrics),diagnostics


def slopes(rows,dim):
    x=np.array([r['nd_build_points'] for r in rows],float)**(1/dim)
    result={}
    fields=['rel_l2','component_rel_l2_max','rel_linf']
    if 'cap_rel_l2' in rows[0]: fields.extend(['cap_rel_l2','cap_rel_linf'])
    for field in fields:
        errors=np.array([r[field] for r in rows])
        result[field]=dict(fitted_decay_order=float(-np.polyfit(np.log(x),np.log(errors),1)[0]),
            adjacent_decay_orders=(-np.diff(np.log(errors))/np.diff(np.log(x))).tolist())
    return result


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--geometry',choices=['h_channel','treasure_chest'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--legacy-charts',action='store_true',help='Explicitly reproduce the archived four-chart cap model')
    p.add_argument('--resolutions',type=int,nargs='+',help='Geometry data resolutions; shared model options stay fixed')
    args=p.parse_args()
    assert jax.default_backend()=='gpu','Run the accuracy study on GPU'
    args.output.mkdir(parents=True,exist_ok=False)
    files=['src/kernelpack/geometry/core.py','src/kernelpack/geometry/parametric_pu.py','scripts/parametric_geometry_convergence_study.py',
           'scripts/run_paper_geometry_accuracy.py']
    if args.geometry=='treasure_chest': files.extend(['scripts/treasure_chest_geometry.py'])
    metadata=dict(geometry=args.geometry,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_hashes={f:hashlib.sha256(Path(f).read_text().encode()).hexdigest() for f in files},
        backend=jax.default_backend(),devices=[str(d) for d in jax.devices()],kernel='r^3 + Pi_1',
        timings=False,check_points='fixed independent analytic geometry points; fitting and evaluation use the same parameter projection',
        normal_errors='orientation aligned, per smooth patch; joins have incident normals',
        geometric_model='legacy four charts' if args.legacy_charts else 'shared parametric PU')
    (args.output/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    counts=args.resolutions or ([8,16,32,64,128,256] if args.geometry=='h_channel' else [7,11,15,21,27,35])
    if len(counts)<3 or any(b<=a for a,b in zip(counts,counts[1:])):
        raise ValueError('Provide at least three strictly increasing geometry data resolutions')
    metadata['input_resolutions']=counts
    metadata['planar_face_metrics']='Treasure chest planar coordinates are exact affine lifts; curved cap errors are also reported separately'
    (args.output/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    rows=[]; pieces=[]; failures=[]
    for count in counts:
        print(f'{args.geometry}: input resolution {count}',flush=True)
        try:
            metrics,diagnostics=h_channel(count,400) if args.geometry=='h_channel' else treasure_chest(count,4000,args.legacy_charts)
        except ValueError as error:
            if not str(error).startswith(('PU support needs more geometry data','PU geometric model did not meet physical indicators')): raise
            failure=dict(input_resolution=count,reason=str(error),history=getattr(error,'history',[]))
            failures.append(failure)
            (args.output/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
            print(json.dumps(failure),flush=True)
            continue
        row=dict(geometry=args.geometry,input_resolution=count,**metrics)
        assert all(np.isfinite(value) for value in metrics.values())
        assert all(np.isfinite(v) for d in diagnostics for v in d.values()
                   if isinstance(v,(int,float,np.number)))
        rows.append(row); pieces.extend(dict(input_resolution=count,**d) for d in diagnostics)
        write_csv(args.output/'accuracy.csv',rows); write_csv(args.output/'pieces.csv',pieces)
        print(json.dumps(row),flush=True)
    dim=2 if args.geometry=='treasure_chest' else 1
    if len(rows)<3:
        raise RuntimeError('Fewer than three geometry resolutions passed; accuracy.csv and failures.json are diagnostics, not a completed convergence study')
    measured=slopes(rows,dim)
    (args.output/'slopes.json').write_text(json.dumps(measured,indent=2)+'\n')
    x=np.array([r['nd_build_points'] for r in rows],float)**(1/dim)
    fig,ax=plt.subplots(figsize=(6.5,4.5),layout='constrained')
    for field,label,marker in [('rel_l2','Relative $\\ell_2$ error','o'),
        ('component_rel_l2_max','Max component relative $\\ell_2$ error','s')]:
        ax.loglog(x,[r[field] for r in rows],marker+'-',label=label)
    ax.set_xlabel('$N_d$' if dim==1 else '$\\sqrt{N_d}$')
    ax.set_ylabel('Parametric interpolation error')
    ax.set_title('Curved H-channel' if dim==1 else 'Treasure chest: parametric PU geometry')
    ax.grid(which='both',alpha=.25); ax.legend()
    fig.savefig(args.output/'parametric_geometry_accuracy.png',dpi=200)
    plt.close(fig)
    print(json.dumps(measured),flush=True)


if __name__=='__main__': main()
