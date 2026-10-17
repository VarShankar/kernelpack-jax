"""GPU production CFPU localization/classification study; no timings or PDEs.

Distances to an independent analytic boundary point cover are upper bounds.
A conservative cover fill bound supplies lower bounds; ambiguous distance-band
assignments are reported, not silently assigned. The point cover is no mesh.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from scipy.spatial import cKDTree
from kernelpack.geometry import CFPUOptions
from cfpu_refinement_study import initial_geometry, resample, reference_cloud, mapped_reference, metrics
from cfpu_feature_grading_study import _roots
from parametric_geometry_convergence_study import sample_h_channel_segment
from treasure_chest_geometry import A, B, L, D, P


def inside(points, geometry):
    if geometry == 'obstacle_chest':
        from obstacle_chest_geometry import inside as obstacle_inside
        return obstacle_inside(points)
    if geometry == 'h_channel':
        x,y=points.T; low=-.06*np.sin(np.pi*x/2)
        return (x>=0)&(x<=2)&(y>=low)&(y<=1-low)&((y<=.4)|(y>=.6)|((x>=.9)&(x<=1.1)))
    x,y,z=points.T; rho=(abs(y)/A)**P+(abs(z)/B)**P
    return (x>=0)&(abs(y)<=A)&(abs(z)<=B)&((x<=L)|((x<=L+D)&(((x-L)/D)**2+rho<=1)))


def boundary_cover(geometry):
    if geometry=='h_channel':
        n=8192
        cloud=np.vstack([sample_h_channel_segment(i,n+1) for i in range(12)])
        # Maximum speed of the two analytic sine curves; straight pieces shorter.
        return cloud, np.sqrt(4+(.06*np.pi)**2)/(2*n)
    # Cap: s=(1-cos(theta))/2 removes endpoint graph singularities.
    # |dX/dtheta| <= sqrt((D*P/2)^2+(sqrt(A^2+B^2)/2)^2).
    # Each of the two rim charts has speed <= sqrt(A^2+B^2).
    nr,nt=1024,256; split=2**(-1/P); radius=np.hypot(A,B)
    theta=np.linspace(0,np.pi,nr+1); s=(1-np.cos(theta))/2
    t=np.linspace(0,split,nt+1)
    rim=np.vstack([np.column_stack([A*(1-t**P)**(1/P),B*t]),
                   np.column_stack([A*t,B*(1-t**P)**(1/P)])])
    parts=[]
    for sy in (-1,1):
        for sz in (-1,1):
            yz=s[:,None,None]*rim[None,:,:]*[sy,sz]
            xx=np.broadcast_to((L+D*np.sqrt(np.maximum(0,1-s**P)))[:,None],yz.shape[:2])
            parts.append(np.concatenate([xx[...,None],yz],axis=2).reshape(-1,3))
    # Exact planar rectangles plus shoulder interiors and their analytic rim.
    step=.002
    def grid(lo,hi): return np.linspace(lo,hi,int(np.ceil((hi-lo)/step))+1)
    y,z=np.meshgrid(grid(-A,A),grid(-B,B),indexing='ij')
    yz=np.column_stack([y.ravel(),z.ravel()])
    parts.append(np.column_stack([np.zeros(len(yz)),yz]))
    shoulder=yz[(abs(yz[:,0])/A)**P+(abs(yz[:,1])/B)**P>=1]
    parts.append(np.column_stack([np.full(len(shoulder),L),shoulder]))
    for axis,extent,other in [(1,A,B),(2,B,A)]:
        xx,oo=np.meshgrid(grid(0,L),grid(-other,other),indexing='ij')
        for sign in (-1,1):
            q=np.empty((xx.size,3)); q[:,0]=xx.ravel(); q[:,axis]=sign*extent
            q[:,3-axis]=oo.ravel(); parts.append(q)
    cap_bound=np.hypot(D*P/2,radius/2)*np.pi/(2*nr)+radius*split/(2*nt)
    # Shoulder grid nearest point may lie in the hole. Move to the analytic
    # rim instead: <= grid diagonal + rim-cover bound by triangle inequality.
    shoulder_bound=np.sqrt(2)*step+radius*split/(2*nt)
    return np.vstack(parts),max(cap_bound,shoulder_bound)


def sharp_rays(surface, geometry, descriptions):
    points=[]; normals=[]; pieces=[]
    if geometry=='h_channel':
        for i,seg in enumerate(surface.segments):
            x=sample_h_channel_segment(i,2)
            uv=descriptions[i].parameterization.project(x)
            _,n=seg.geom_model['model'].frame(uv)
            n=np.asarray(n)*seg.geom_model.get('normal_sign',1.)
            points.extend(x); normals.extend(n); pieces.extend([i]*2)
    else:
        # Independent sites on straight sharp edges, each incident face normal.
        rng=np.random.default_rng(77123)
        for varying in range(3):
            fixed=[a for a in range(3) if a!=varying]
            ranges=[(0,L),(-A,A),(-B,B)]
            for a in ranges[fixed[0]]:
                for b in ranges[fixed[1]]:
                    x=np.zeros((16,3)); x[:,fixed[0]]=a; x[:,fixed[1]]=b
                    x[:,varying]=rng.uniform(*ranges[varying],16)
                    for axis,value in zip(fixed,[a,b]):
                        n=np.zeros_like(x); n[:,axis]= -1 if value==ranges[axis][0] else 1
                        points.extend(x); normals.extend(n); pieces.extend([axis]*16)
        # Cap/shoulder rim is a sharp join except at its four axis junctions.
        t=rng.uniform(.02,2**(-1/P)-.02,32)
        rim=np.vstack([np.column_stack([A*(1-t**P)**(1/P),B*t]),
                       np.column_stack([A*t,B*(1-t**P)**(1/P)])])
        seg=surface.segments[-1]
        for sy in (-1,1):
            for sz in (-1,1):
                x=np.column_stack([np.full(len(rim),L),rim*[sy,sz]])
                uv=descriptions[-1].parameterization.project(x)
                _,n=seg.geom_model['model'].frame(uv)
                n=np.asarray(n)*seg.geom_model.get('normal_sign',1.)
                points.extend(x); normals.extend(n); pieces.extend([9]*len(x))
                n=np.tile([1.,0,0],(len(x),1))
                points.extend(x); normals.extend(n); pieces.extend([5]*len(x))
    return np.asarray(points),np.asarray(normals),np.asarray(pieces)


def classification(values, truth, lower, upper, edges):
    finite=np.isfinite(values); wrong=finite&((values>0)!=truth)
    rows=[]; assigned=np.zeros(len(values),bool)
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(lower>=lo)&(upper<hi); assigned|=mask
        count=int(mask.sum())
        rows.append(dict(lower=lo,upper=hi,count=count,inside_count=int(truth[mask].sum()),
            wrong=int(wrong[mask].sum()),unresolved=int((~finite[mask]).sum()),
            wrong_fraction=float(wrong[mask].sum()/count) if count else None))
    return dict(probe_count=len(values),wrong=int(wrong.sum()),unresolved=int((~finite).sum()),
        wrong_inside=int((wrong&truth).sum()),wrong_outside=int((wrong&~truth).sum()),
        distance_band_ambiguous=int((~assigned).sum()),bands=rows,
        sampled_error_layer_lower=float(lower[wrong].max()) if wrong.any() else 0.,
        sampled_error_layer_upper=float(upper[wrong].max()) if wrong.any() else 0.,
        unresolved_distance_upper=float(upper[~finite].max()) if (~finite).any() else 0.)


def feature_roots(model, points, normals, h):
    offset,residual,good,center=map(np.asarray,_roots(jnp.asarray(points),jnp.asarray(normals),h,*model.args()))
    # A sharp feature may be a tangent zero on this incident normal line.
    # Check the sampled origin before accepting a farther sign crossing.
    at_origin=np.isfinite(center)&(np.abs(center)<=1e-12*model.scale)
    return np.where(at_origin,0.,offset),np.where(at_origin,center,residual),good|at_origin,center


def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument('--geometry',required=True,choices=['h_channel','treasure_chest'])
    p.add_argument('--output',required=True,type=Path); p.add_argument('--h',nargs='+',type=float,default=[.16,.12,.08,.06,.04])
    args=p.parse_args(); assert jax.default_backend()=='gpu'; args.output.mkdir(parents=True,exist_ok=False)
    options=CFPUOptions(feature_grade_factor=2)
    sources=['scripts/cfpu_classification_study.py','scripts/cfpu_refinement_study.py',
        'scripts/cfpu_feature_grading_study.py','src/kernelpack/geometry/cfpu.py',
        'src/kernelpack/geometry/feature_sampling.py','src/kernelpack/geometry/core.py',
        'scripts/treasure_chest_geometry.py','scripts/parametric_geometry_convergence_study.py']
    provenance=dict(geometry=args.geometry,backend=jax.default_backend(),devices=list(map(str,jax.devices())),
        options=vars(options),levels=args.h,timings=False,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        hashes={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources},
        distance_definition='Euclidean nearest analytic boundary distance bounded by point-cover fill bound',
        distance_work='independent host KDTree diagnostic; CFPU fits/evaluations and normal roots on GPU',
        classification_width='maximum wrong-probe distance interval, a sampled statistic, not a guarantee',
        abscissa='N_fit in 2D, sqrt(N_fit) in 3D')
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (args.output/'runner_at_execution.py').write_text(Path(__file__).read_text())
    print('building independent boundary distance cover',flush=True)
    cover,epsilon=boundary_cover(args.geometry); tree=cKDTree(cover)
    print('distance cover',len(cover),'fill bound',epsilon,flush=True)
    base,descriptions,labels=initial_geometry(args.geometry,args.h[0])
    exact,analytic_normal,parameters,pieces=reference_cloud(args.geometry,descriptions,labels)
    rng=np.random.default_rng(883019)
    bounds=([- .3,-.3],[2.3,1.3]) if args.geometry=='h_channel' else ([-.3,-.8,-.55],[2.1,.8,.55])
    volume=rng.uniform(*bounds,size=(12000,exact.shape[1]))
    # Fixed offsets and reference cloud for the entire refinement study.
    offsets=np.r_[-np.geomspace(.0002,.2,10)[::-1],np.geomspace(.0002,.2,10)]
    probes=np.vstack([volume,(exact[:,None,:]+offsets[None,:,None]*analytic_normal[:,None,:]).reshape(-1,exact.shape[1])])
    truth=inside(probes,args.geometry); upper=tree.query(probes)[0]; lower=np.maximum(0,upper-epsilon)
    near=np.linalg.norm(exact[:,None,:]-np.unique(np.vstack([sample_h_channel_segment(i,2) for i in range(12)]),axis=0)[None,:,:],axis=-1).min(axis=1) if args.geometry=='h_channel' else cKDTree(np.asarray(base.feature_sites)).query(exact)[0]
    away=near>.05*np.linalg.norm(np.ptp(exact,axis=0))
    edges=[0,.005,.01,.02,.04,.08,.16,.32,1.,10.]
    np.savez(args.output/'independent_probes.npz',points=probes,inside=truth,distance_lower=lower,distance_upper=upper,reference=exact,pieces=pieces,away=away)
    provenance.update(distance_cover_sites=len(cover),distance_fill_bound=epsilon,band_edges=edges)
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    del tree,cover
    rows=[]
    for h in args.h:
        print('production graded CFPU h=',h,flush=True)
        surface=resample(base,args.geometry,h); surface.build_level_set(options=options)
        model=surface.level_set.cfpu_model
        _,normal=mapped_reference(surface,exact,parameters,pieces)
        offset,residual,good,_=map(np.asarray,_roots(jnp.asarray(exact),jnp.asarray(normal),h,*model.args()))
        sx,sn,sp=sharp_rays(surface,args.geometry,descriptions)
        so,sr,sg,sc=feature_roots(model,sx,sn,h)
        values=np.asarray(model.evaluate(jnp.asarray(probes)))
        fit=np.asarray(surface.level_set.xd)
        row=dict(h=h,n_fit=len(fit),axis_value=float(len(fit)**(1/(exact.shape[1]-1))),supports=model.diagnostics['patch_count'],
            localization=metrics(offset,good),away=metrics(offset[away],good[away]),near=metrics(offset[~away],good[~away]),
            sharp_incident_rays=metrics(so,sg),classification=classification(values,truth,lower,upper,edges),
            minimum_reference_fit_distance=float(cKDTree(fit).query(exact)[0].min()))
        np.savez(args.output/f'level_h{h:g}.npz',values=values,displacement=offset,residual=residual,good=good,
            geometry_normals=normal,sharp_points=sx,sharp_normals=sn,sharp_pieces=sp,sharp_displacement=so,sharp_good=sg,
            sharp_residual=sr,sharp_center_values=sc,fit_sites=fit)
        np.savez(args.output/f'model_h{h:g}.npz',**{f'arg{i}':np.asarray(v) for i,v in enumerate(model.args())})
        rows.append(row); (args.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(row),flush=True); del model,surface; jax.clear_caches()


if __name__=='__main__': main()
