"""Accuracy-only GPU forward campaign on the completed shared geometry pipeline."""
import argparse
import hashlib
import json
import os
import subprocess
import traceback
import shutil
from pathlib import Path
import gc
import time
import jax
from kernelpack.accelerators.jax_spatial import enable_compilation_cache
enable_compilation_cache()
import jax.numpy as jnp
import numpy as np
from kernelpack.geometry import CFPUOptions
from kernelpack.nodes import DomainNodeGenerator
from kernelpack.solvers._common import SparseCOOMatrix,sparse_matvec
from kernelpack.solvers.spectral import (
    dirichlet_implicit_generator_precheck,
    dirichlet_laplacian_precheck,
    neumann_operator_precheck,
)
from kernelpack.solvers.variable_poisson import _build_variable_pde_operator_sparse
from kernelpack.solvers._sparse_power_stability import check_sparse_power_sbdf3_stability
from kernelpack.solvers.reaction_diffusion_stateless import StatelessReactionDiffusionSolver
from kernelpack.solvers.eulerian_advection_diffusion_reaction import StatelessEulerianAdvectionDiffusionReactionSolver
from cfpu_refinement_study import initial_geometry,resample
from cfpu_classification_study import inside
import darcy_forward_convergence_study as darcy
from darcy_inverse_convergence_study import manufactured_state_1 as darcy_manufactured_state
import reaction_diffusion_forward_convergence_study as rd
import advection_diffusion_forward_convergence_study as ad
import reference_velocity

jax.config.update("jax_enable_x64",True)


def timestep(h,order,T=.5,factor=.25,cfl_factor=None):
    # BDF3/third-order explicit extrapolation: dt^3=O(h^order), with dt=O(h)
    # also retained for transport. Round to reach exactly the same final time.
    # With cfl_factor, the two restrictions get separate constants:
    # dt = min(cfl_factor*h, factor*h^(order/3)) (CFL cap, accuracy).
    target=factor*min(h,h**(order/3)) if cfl_factor is None else min(cfl_factor*h,factor*h**(order/3))
    steps=max(5,int(np.ceil(T/target)))
    return T/steps,steps


def normal_derivative(fn,x,normals):
    """n . grad fn at the points x for a batched scalar function fn(x)."""
    g=jax.vmap(jax.grad(lambda z:fn(z[None,:])[0]))(jnp.asarray(x))
    return jnp.sum(g*jnp.asarray(normals),axis=1)


def error(u,exact):
    assert bool(jnp.all(jnp.isfinite(u))),"nonfinite direct-solve state"
    return float(jnp.linalg.norm(u-exact)/jnp.linalg.norm(exact))


def load_domain(path,surface=None):
    domain,_=darcy.load_domain_file(path)
    if surface is not None:domain.set_outer_level_set(surface.get_level_set())
    return domain


def save_domain(domain,path,geometry,h,seed):
    np.savez(path,int_nodes=np.asarray(domain.get_interior_nodes()),
        bdry_nodes=np.asarray(domain.get_bdry_nodes()),ghost_nodes=np.asarray(domain.get_ghost_nodes()),
        normals=np.asarray(domain.get_nrmls()),sep_rad=h,dim=domain.s_dim if hasattr(domain,"s_dim") else domain.get_all_nodes().shape[1],
        geometry=geometry,h=h,seed=seed,label=f"{geometry}_h{h:g}_s{seed}_shared_pu_cfpu")


def rd_final(base,family,theta,steps):
    wrapper=StatelessReactionDiffusionSolver.from_solver(base)
    forcing=rd.make_forcing(family,base.nu,theta)
    reaction=rd.build_reaction_term(family,theta)
    bc=lambda _a,_b,nr,t,x:normal_derivative(lambda z:rd.exact_state(t,z),x,nr)      # Neumann: du/dn = g
    u0,u1,u2=[rd.exact_state(k*base.dt,base.x) for k in range(3)]
    def run():
        def step(k,carry):
            a,b,c=carry
            new=wrapper.bdf3_state(a,b,c,(k+1)*base.dt,reaction=reaction,bc=bc,forcing=forcing)
            return b,c,new
        return jax.lax.fori_loop(2,steps,step,(u0,u1,u2))[-1]
    return jax.jit(run)()


def ad_final(base,gamma,velocity,steps):
    wrapper=StatelessEulerianAdvectionDiffusionReactionSolver.from_solver(base,prepare_orders=(3,))
    forcing=ad.manufactured_forcing(base.nu,velocity,0.)
    reaction=lambda t,u,x:jnp.zeros_like(u)
    bc=lambda _a,_b,_n,t,x:ad.exact_state(t,x)
    u0,u1,u2=[ad.exact_state(k*base.dt,base.xall) for k in range(3)]
    def run():
        def step(k,carry):
            a,b,c=carry; dt=base.dt
            new=wrapper.bdf3_full_state(a,b,c,(k-2)*dt,(k-1)*dt,k*dt,(k+1)*dt,
                velocity=velocity,reaction=reaction,bc=bc,forcing=forcing,gamma_components=gamma)
            return b,c,new
        return jax.lax.fori_loop(2,steps,step,(u0,u1,u2))[-1][:base.n]
    return jax.jit(run)()


def phase_counts(base):
    return {str(k):v.phase_counts for k,v in getattr(base,"cudss_factors",{}).items()}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--geometries",nargs="+",default=["h_channel","treasure_chest"])
    p.add_argument("--h",nargs="+",type=float,default=[.16,.12,.08,.06,.04])
    p.add_argument("--orders",nargs="+",type=int,default=[2,4,6])
    p.add_argument("--problems",nargs="+",default=["darcy","rd","ad"])
    p.add_argument("--dt-factor",type=float,default=.25)
    p.add_argument("--ad-dt-factor",type=float,default=None,help="AD timestep factor (default: --dt-factor).")
    p.add_argument("--ad-velocity-amplitude",type=float,default=.12,
        help="Maximum speed of the AD velocity field; also the speed scale in the hyperviscosity gamma.")
    p.add_argument("--seed",type=int,default=11,help="Cloud seed; with --domain-dir loads {geometry}_h{h:.2f}_s{seed}.npz.")
    p.add_argument("--fit-resolution",type=int,default=21,help="Treasure-chest cap fit resolution (generate_paper_clouds uses 41).")
    p.add_argument("--cfpu-max-sites",type=int,default=512)
    p.add_argument("--ad-treatment",choices=("explicit","implicit"),default="implicit",
        help="Time placement for cropped sparse H; explicit runs require the sampled three-history SBDF3 amplification screen.")
    p.add_argument("--ad-hyperviscosity-method",choices=("direct_rbf","laplacian_power"),default="direct_rbf",
        help="AD hyperviscosity construction: direct RBF-FD Delta^k, or full-square sparse Laplacian power.")
    p.add_argument("--ad-hyperviscosity-rows",choices=("physical","interior","far"),default="physical",
        help="For direct RBF-FD, interior retains the physical PDE on boundary rows that determine ghosts.")
    p.add_argument("--ad-velocity-field",choices=("cloud","reference"),default="cloud",
        help="cloud: velocity from each cloud's own implicit layer, normalized on that cloud (changes with h); "
             "reference: one fixed field per geometry (reference_velocity.py), the same at every h.")
    p.add_argument("--ad-hyperviscosity-power",default="auto",
        help="Laplacian power k of the AD hyperviscosity: 'auto' (k=ceil((order+max q)/2), q measured per cloud), "
             "'order' (k=order/2+1, independent of the cloud), or a positive integer.")
    p.add_argument("--ad-dt-cfl-factor",type=float,default=None,help="With this, AD dt = min(c_cfl h, c h^(p/3)) with c = --ad-dt-factor.")
    p.add_argument("--boundary-aware-stencils",action="store_true",help="Select stencils by ray tracing against the implicit layer (kernelpack.domain.boundary_aware).")
    p.add_argument("--domain-dir",type=Path,default=None,help="Reuse verified clouds; never regenerate them merely for changing the solver.")
    p.add_argument("--validated-redraws",type=Path,default=None,
        help="Use first spectrum-passing PDS redraw per cloud/order from a uniform redraw audit.")
    p.add_argument("--resume",action="store_true")
    a=p.parse_args(); assert jax.default_backend()=="gpu"
    if a.ad_hyperviscosity_method=="laplacian_power" and a.ad_hyperviscosity_rows=="interior":
        p.error("interior hyperviscosity rows currently require --ad-hyperviscosity-method direct_rbf")
    ad_row_mask={"interior":"boundary","far":"near_boundary"}.get(a.ad_hyperviscosity_rows,"none")  # far: no hyperviscosity within 2h of the boundary
    ad_boundary_protocol=(
        ("direct_rbf_interior_rows_physical_PDE_for_ghosts" if a.ad_hyperviscosity_rows=="interior"
         else "direct_rbf_physical_rows_I_plus_B")
        if a.ad_hyperviscosity_method == "direct_rbf"
        else "full_square_sparse_power_then_physical_row_crop"
    )
    redraws={}
    if a.validated_redraws:
        for trial in json.loads((a.validated_redraws/"results.json").read_text()):
            if trial["spectral"]["passed"]:
                redraws.setdefault((trial["geometry"],trial["h"],trial["order"]),trial)
    a.output.mkdir(parents=True,exist_ok=a.resume); (a.output/"domains").mkdir(exist_ok=True)
    paths=["src/kernelpack/geometry/cfpu.py","src/kernelpack/geometry/parametric_pu.py",
        "src/kernelpack/nodes/core.py","src/kernelpack/nodes/hierarchical.py",
        "src/kernelpack/solvers/variable_poisson.py","src/kernelpack/solvers/reaction_diffusion.py",
        "src/kernelpack/solvers/reaction_diffusion_stateless.py",
        "src/kernelpack/solvers/eulerian_advection_diffusion_reaction.py",
        "src/kernelpack/solvers/_cudss.py","src/kernelpack/sparse/cudss.py",
        "src/kernelpack/solvers/_laplacian_power.py","src/kernelpack/solvers/_composed_adr.py",
        "src/kernelpack/solvers/_sparse_laplacian_power.py",
        "src/kernelpack/solvers/_sparse_power_stability.py",
        "src/kernelpack/sparse/_native/cudss_ffi.cc","src/kernelpack/sparse/_cudss_ffi.so",
        "scripts/run_cudss_paper_forward.py","scripts/darcy_forward_convergence_study.py",
        "scripts/reaction_diffusion_forward_convergence_study.py",
        "scripts/advection_diffusion_forward_convergence_study.py",
        "scripts/reaction_diffusion_inverse_convergence_study.py",
        "scripts/darcy_inverse_convergence_study.py","inverse problems/darcy_model_selection.py",
        "scripts/cfpu_refinement_study.py","scripts/treasure_chest_geometry.py"]
    paths+=["src/kernelpack/rbffd/core.py","src/kernelpack/solvers/_pu.py","src/kernelpack/solvers/spectral.py",
        "src/kernelpack/solvers/_common.py","src/kernelpack/poly/core.py","src/kernelpack/domain/core.py",
        "src/kernelpack/geometry/core.py"]
    hashes={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in paths}
    runtime_library=Path(os.environ.get(
        "KERNELPACK_CUDSS_LIBRARY", "src/kernelpack/sparse/_cudss_ffi.so"
    )).resolve()
    if not runtime_library.is_file():
        raise FileNotFoundError(f"configured cuDSS runtime library is missing: {runtime_library}")
    provenance=dict(commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        hashes=hashes,backend=jax.default_backend(),devices=list(map(str,jax.devices())),
        cudss_runtime_library=dict(path=str(runtime_library),
            sha256=hashlib.sha256(runtime_library.read_bytes()).hexdigest()),
        options=dict(geometries=a.geometries,h=a.h,orders=a.orders,problems=a.problems,dt_factor=a.dt_factor,
            ad_hyperviscosity_method=a.ad_hyperviscosity_method,ad_treatment=a.ad_treatment,
            ad_hyperviscosity_rows=a.ad_hyperviscosity_rows,
            domain_dir=str(a.domain_dir) if a.domain_dir else None,
            validated_redraws=str(a.validated_redraws) if a.validated_redraws else None),
        linear_solver="cudss",stencil="rbf",timings="wall-clock seconds per case: shared_setup_s/shared_precheck_s (Laplacian gate), setup_s, precheck_s, solve_s (Darcy, first family includes jit compile), rollout_s (RD/AD)",
        forcing="original continuous manufactured forcing; no hyperviscosity compensation",
        construction=("direct_rbf_delta_power" if a.ad_hyperviscosity_method == "direct_rbf"
                     else "full_square_sparse_laplacian_power"),
        boundary=(("physical PDE boundary rows determine ghosts; direct RBF-FD hyperviscosity only on interior rows"
                   if a.ad_hyperviscosity_rows=="interior"
                   else "direct RBF-FD Delta^k assembled on physical PDE rows I+B; ghost columns retained")
                  if a.ad_hyperviscosity_method == "direct_rbf"
                  else "full-square sparse Laplacian power at all nodes, then crop physical rows"),
        hyperviscosity_method=a.ad_hyperviscosity_method,
        hyperviscosity_engine=("direct_rbf_sparse" if a.ad_hyperviscosity_method == "direct_rbf" else "sparse"),
        hyperviscosity_treatment=a.ad_treatment,
        startup="analytic values at t=0,dt,2dt; subsequent states computed with BDF3",
        timestep="dt_target=dt_factor*min(h,h**(spatial_order/3)); T=.5 reached exactly",
        geometry="common parametric PU / CFPU, feature_grade_factor=2, Warp Poisson disk sampling",seed=a.seed,fit_resolution=a.fit_resolution,cfpu_max_sites=a.cfpu_max_sites,ad_velocity_amplitude=a.ad_velocity_amplitude,ad_velocity_field=a.ad_velocity_field,ad_dt_factor=a.ad_dt_factor or a.dt_factor,
        gamma=("existing Euclidean paper formula, auto k=ceil((order+max(q))/2); no fixed-gamma fallback"
               if a.ad_hyperviscosity_power=="auto" else
               f"existing Euclidean paper formula with measured q; k={'order/2+1' if a.ad_hyperviscosity_power=='order' else a.ad_hyperviscosity_power}; no fixed-gamma fallback"),
        ad_hyperviscosity_power=a.ad_hyperviscosity_power,
        temporal_check="none; dt=dt_factor*min(h,h**(order/3)) (CFL and BDF3 error O(h^order))",
        accuracy_axis="physical node count**(-1/d)",operator_chunk_size=128,
        hyperviscosity_boundary_protocol=ad_boundary_protocol,
        spectral_precheck=("mandatory Arnoldi on the boundary-closed combined spatial operator nu*L + gamma*H for direct_rbf; positive or unconverged checks do not pass"
                           if a.ad_hyperviscosity_method == "direct_rbf"
                           else "mandatory Arnoldi on boundary-closed homogeneous-Dirichlet Laplacian and composed power generator; positive or unconverged checks do not pass"),
        cpqr="shared polynomial conditioning fallback, retaining full requested moment equations")
    if a.validated_redraws:
        provenance["redraw_selection"]=dict(
            rule="first passing seed in the bounded uniform audit; actual solver arrays rechecked before every solve",
            audit=json.loads((a.validated_redraws/"provenance.json").read_text()),
            audit_results_hash=hashlib.sha256((a.validated_redraws/"results.json").read_bytes()).hexdigest(),
            selected=[dict(geometry=g,h=h,order=p,seed=t["seed"],
                domain_hash=hashlib.sha256((a.validated_redraws/t["domain_file"]).read_bytes()).hexdigest())
                for (g,h,p),t in redraws.items()])
    provenance_path=a.output/"provenance.json"
    if a.resume:
        old=json.loads(provenance_path.read_text())
        runner="scripts/run_cudss_paper_forward.py"
        assert {k:v for k,v in old["hashes"].items() if k!=runner}=={k:v for k,v in hashes.items() if k!=runner}
        assert old["options"]==provenance["options"]
        if old["hashes"][runner]!=hashes[runner]:
            old.setdefault("harness_revisions",[]).append(dict(previous_hash=old["hashes"][runner],new_hash=hashes[runner],
                reason="correct COO residual-check API and isolate errors by family; numerical library sources unchanged"))
            old["hashes"]=hashes
            provenance_path.write_text(json.dumps(old,indent=2)+"\n")
            (a.output/"runner_after_harness_fix.py").write_text(Path(__file__).read_text())
    else:
        provenance_path.write_text(json.dumps(provenance,indent=2)+"\n")
        (a.output/"runner_at_execution.py").write_text(Path(__file__).read_text())
    rows=json.loads((a.output/"results.json").read_text()) if a.resume and (a.output/"results.json").exists() else []
    spectral_cache={};spectral_checks=[]
    def precheck(solver,domain,geometry,h,order,problem,operator=None):
        if problem in ("shared_laplacian","rd","darcy"):
            # Neumann rows (Darcy, RD): reduced operator on mean-zero physical values. For Darcy, operator = -A(theta).
            result=neumann_operator_precheck(solver.lap_sparse if operator is None else operator,solver.bc_sparse,domain,cache=spectral_cache)
        else:
            result=dirichlet_laplacian_precheck(solver.lap_sparse,solver.bc_sparse,domain,cache=spectral_cache)
        explicit_stability=None
        if problem == "ad" and result["passed"] and solver.hyperviscosity_method == "direct_rbf":
            raw=result
            result=dirichlet_implicit_generator_precheck(
                solver.lap_sparse,solver.hyper_sparse,solver.bc_sparse,domain,
                nu=float(solver.nu),gamma=float(np.sum(np.asarray(solver.gamma_components))),
                cache=spectral_cache,
            )
            result["laplacian_precheck"]=raw
        elif problem == "ad" and result["passed"] and solver.hyperviscosity_method == "laplacian_power":
            from kernelpack.solvers._composed_adr import power_generator_precheck
            raw=result
            result=power_generator_precheck(solver)
            result["laplacian_precheck"]=raw
            result["fingerprint"]=raw["fingerprint"]
        if (problem == "ad" and solver.hyperviscosity_treatment == "explicit"
                and result.get("passed", False)):
            explicit_stability=check_sparse_power_sbdf3_stability(
                solver.lap_sparse, solver.hyper_sparse, solver.bc_sparse,
                n_physical=int(domain.get_num_int_bdry_nodes()), dt=float(solver.dt),
                nu=float(solver.nu), gamma=float(np.sum(np.asarray(solver.gamma_components))),
            )
            result["explicit_sbdf3_stability"]={
                "passed": bool(explicit_stability.get("passed",False)),
                "converged": bool(explicit_stability.get("converged",False)),
                "finite": bool(explicit_stability.get("finite",False)),
                "reason": str(explicit_stability.get("reason","missing reason")),
                "claim_scope": str(explicit_stability.get("claim_scope","")),
                "operator": str(explicit_stability.get("operator","")),
                "dt": float(explicit_stability["dt"]),
                "nu": float(explicit_stability["nu"]),
                "gamma": float(explicit_stability["gamma"]),
                "requested_modes": int(explicit_stability["requested_modes"]),
                "spectral_radius_estimate": explicit_stability.get("spectral_radius_estimate"),
                "stability_tolerance": float(explicit_stability["stability_tolerance"]),
                "ritz_eigenvalues_real": np.asarray(explicit_stability.get("ritz_eigenvalues",[])).real.tolist(),
                "ritz_eigenvalues_imag": np.asarray(explicit_stability.get("ritz_eigenvalues",[])).imag.tolist(),
                "ritz_residuals": np.asarray(explicit_stability.get("ritz_residuals",[]),dtype=float).tolist(),
            }
            if not result["explicit_sbdf3_stability"]["passed"]:
                result["passed"]=False
                result["reason"]="explicit SBDF3 hyperviscosity amplification screen failed: "+result["explicit_sbdf3_stability"]["reason"]
        spectral_checks.append(dict(geometry=geometry,h=h,order=order,problem=problem,**result))
        (a.output/"spectral_prechecks.json").write_text(json.dumps(spectral_checks,indent=2)+"\n")
        if not result["passed"]:
            name=a.output/f"rejected_operator_{result['fingerprint'][:16]}.npz"
            if not name.exists():
                payload=dict(lap_indices=np.asarray(solver.lap_sparse.indices),lap_values=np.asarray(solver.lap_sparse.values),
                    lap_shape=solver.lap_sparse.shape,bc_indices=np.asarray(solver.bc_sparse.indices),bc_values=np.asarray(solver.bc_sparse.values),
                    bc_shape=solver.bc_sparse.shape,interior_count=domain.get_num_interior_nodes(),
                    boundary_count=domain.get_num_bdry_nodes(),all_nodes=np.asarray(domain.get_all_nodes()))
                if problem == "ad" and solver.hyperviscosity_method == "direct_rbf":
                    combined=solver._implicit_operator_sparse(solver.nu,solver.gamma_components)
                    payload.update(operator_indices=np.asarray(combined.indices),operator_values=np.asarray(combined.values),
                        operator_shape=combined.shape,hyper_indices=np.asarray(solver.hyper_sparse.indices),
                        hyper_values=np.asarray(solver.hyper_sparse.values),hyper_shape=solver.hyper_sparse.shape,
                        nu=float(solver.nu),gamma=float(np.sum(np.asarray(solver.gamma_components))))
                np.savez_compressed(name,**payload)
        print("ARNOLDI",geometry,h,order,problem,json.dumps(result),flush=True)
        return result
    def recorded(geometry,h,order,problem,family):
        return any(all(r.get(k)==v for k,v in dict(geometry=geometry,h=h,order=order,problem=problem,family=family).items()) for r in rows)
    def record(row):
        rows.append(row); (a.output/"results.json").write_text(json.dumps(rows,indent=2)+"\n")
        print(json.dumps(row),flush=True)
    for geometry in a.geometries:
        print("GEOMETRY",geometry,"building accepted shared model",flush=True)
        # Coordinate fits at h=.16, as in generate_paper_clouds.py (--base-h).
        base,_,_=initial_geometry(geometry,.16,fit_resolution=a.fit_resolution)
        for h in a.h:
            print("DOMAIN",geometry,h,flush=True)
            surface=resample(base,geometry,h,seed=a.seed,fit_resolution=a.fit_resolution)
            surface.build_level_set(options=CFPUOptions(feature_grade_factor=2,max_sites=a.cfpu_max_sites))
            domain_path=a.output/"domains"/f"{geometry}_h{h:g}.npz"
            source_domain=a.domain_dir/f"{geometry}_h{h:.2f}_s{a.seed}.npz" if a.domain_dir else None
            if source_domain is not None and source_domain.exists() and not domain_path.exists():
                shutil.copyfile(source_domain,domain_path)
            if a.resume and domain_path.exists():domain=load_domain(domain_path,surface)
            elif source_domain is not None and source_domain.exists():domain=load_domain(domain_path,surface)
            else:
                domain=DomainNodeGenerator().build_domain_descriptor_from_geometry(surface,h,backend="warp",seed=a.seed,
                    do_outer_refinement=True,outer_fraction_of_h=.6,outer_refinement_zone_size_as_multiple_of_h=1.5)
                assert inside(np.asarray(domain.get_interior_nodes()),geometry).all(),"generated node outside analytic domain"
                save_domain(domain,domain_path,geometry,h,a.seed)
                (a.output/"domains"/f"{geometry}_h{h:g}_geometry.json").write_text(json.dumps(surface.level_set.cfpu_model.diagnostics,indent=2)+"\n")
            x=domain.get_int_bdry_nodes(); xa=domain.get_all_nodes(); xb=domain.get_bdry_nodes()
            n=len(x); dim=x.shape[1]
            print("DOMAIN READY",geometry,h,"physical",n,"total",len(xa),flush=True)
            if a.boundary_aware_stencils:
                domain.set_stencil_level_set(surface.level_set.cfpu_model,h)
            default_domain=domain
            for order in a.orders:
                domain=default_domain
                selected_path=domain_path
                trial=redraws.get((geometry,h,order))
                if trial is not None:
                    selected_path=a.output/"domains"/f"{geometry}_h{h:g}_p{order}.npz"
                    source=a.validated_redraws/trial["domain_file"]
                    if not selected_path.exists():shutil.copyfile(source,selected_path)
                    assert hashlib.sha256(selected_path.read_bytes()).digest()==hashlib.sha256(source.read_bytes()).digest()
                    domain=load_domain(selected_path,surface)
                with np.load(selected_path) as saved: domain_seed=int(saved["seed"]) if "seed" in saved else 11
                x=domain.get_int_bdry_nodes();xa=domain.get_all_nodes();xb=domain.get_bdry_nodes()
                n=len(x);dim=x.shape[1]
                common=dict(geometry=geometry,h=h,order=order,physical_nodes=n,total_nodes=len(xa),
                    axis_value=float(n**(-1/dim)),domain_seed=domain_seed,domain_file=str(selected_path))
                dt,steps=timestep(h,order,factor=a.dt_factor);nu=rd.solver_settings(dim)[2]
                gate=None
                try:
                    t0=time.perf_counter()
                    gate=rd.build_solver_template(domain,order=order,dt=dt,nu=nu,stencil="rbf",linear_solver="cudss",sparse_assembly_chunk_size=128)
                    gate.prepare_boundary_operator(0.,lambda z:jnp.ones(len(z)),lambda z:jnp.zeros(len(z)))   # Neumann (RD)
                    t1=time.perf_counter()
                    spectral=precheck(gate,domain,geometry,h,order,"shared_laplacian")
                    common.update(shared_setup_s=t1-t0,shared_precheck_s=time.perf_counter()-t1)
                    if not spectral["passed"]:
                        for problem,families in [("darcy",darcy.FAMILIES),("rd",rd.FAMILIES),("ad",["prescribed"])]:
                            if problem not in a.problems:continue
                            for family in families:
                                if not recorded(geometry,h,order,problem,family):
                                    record(dict(**common,problem=problem,family=family,status="spectral_rejected",
                                        reason=spectral["reason"],spectral_fingerprint=spectral["fingerprint"]))
                        gate.close_cudss();del gate;jax.clear_caches();gc.collect();continue
                except Exception as e:
                    if gate is not None:gate.close_cudss()
                    for problem,families in [("darcy",darcy.FAMILIES),("rd",rd.FAMILIES),("ad",["prescribed"])]:
                        if problem not in a.problems:continue
                        for family in families:
                            if not recorded(geometry,h,order,problem,family):
                                record(dict(**common,problem=problem,family=family,status="spectral_setup_failed",error=repr(e)))
                    jax.clear_caches();gc.collect();continue
                if "darcy" in a.problems:
                    solver=None
                    try:
                        print("ASSEMBLE",geometry,h,order,"Darcy",flush=True)
                        t0=time.perf_counter()
                        solver=darcy.build_solver(domain,order,"rbf","cudss",128)
                        solver.prepare_boundary_operator(jnp.ones(len(xb)),jnp.zeros(len(xb)))   # pure Neumann
                        t1=time.perf_counter()
                        timing=dict(setup_s=t1-t0)
                        grads=tuple(solver.grad_sparse) if isinstance(solver.grad_sparse,(list,tuple)) else (solver.grad_sparse,)
                        normals_b=domain.get_nrmls()
                        frozen=solver.freeze()
                        def solve(k,f,g):
                            out=frozen.solve_from_fields(k,f,g)
                            matrix=out["system_matrix_sparse"]
                            state=out["full_state"] if out["lagrange_multiplier"] is None else jnp.concatenate((out["full_state"],jnp.atleast_1d(out["lagrange_multiplier"])))
                            residual=jnp.linalg.norm(sparse_matvec(matrix.indices,matrix.values,matrix.shape[0],state)-out["rhs"])/jnp.linalg.norm(out["rhs"])
                            return out["u"],residual
                        compiled=jax.jit(solve)
                        for family in darcy.FAMILIES:
                            if recorded(geometry,h,order,"darcy",family):continue
                            theta=darcy.sample_theta(family,dim); case=darcy.build_cases(family,theta,x,xb)[0]
                            coeff=darcy.coefficient_field(family,theta,xa)
                            t2=time.perf_counter()
                            A=_build_variable_pde_operator_sparse(solver.lap_sparse,grads,coeff,n)
                            check=precheck(solver,domain,geometry,h,order,"darcy",operator=SparseCOOMatrix(A.indices,-A.values,A.shape))
                            precheck_s=time.perf_counter()-t2
                            if not check["passed"]:
                                record(dict(**common,**timing,precheck_s=precheck_s,problem="darcy",family=family,status="spectral_rejected",
                                    reason=check["reason"],spectral_fingerprint=check["fingerprint"])); continue
                            # Flux condition K du/dn = g with g = K* du*/dn: the rows are du/dn = g/K at the boundary nodes.
                            flux_over_k=normal_derivative(darcy_manufactured_state,xb,normals_b)
                            t0=time.perf_counter()
                            u,residual=compiled(coeff,case.forcing,flux_over_k)
                            u=u+(jnp.mean(case.target)-jnp.mean(u))     # fix the constant: mean over the physical nodes
                            err=error(u,case.target); residual=float(residual); solve_s=time.perf_counter()-t0
                            assert residual<1e-8,"direct algebraic residual too large"
                            record(dict(**common,**timing,precheck_s=precheck_s,solve_s=solve_s,problem="darcy",family=family,status="ok",relative_l2=err,
                                darcy_operator_rightmost_real=check["checks"][-1]["rightmost_real"] if check.get("checks") else None,
                                algebraic_relative_residual=float(residual),cudss_phases=solver.cudss_pattern.phase_counts))
                            np.savez(a.output/f"darcy_{geometry}_h{h:g}_p{order}_{family}.npz",state=np.asarray(u),exact=np.asarray(case.target))
                    except Exception as e:
                        traceback.print_exc()
                        for family in darcy.FAMILIES:
                            if not recorded(geometry,h,order,"darcy",family):record(dict(**common,problem="darcy",family=family,status="failed",error=repr(e)))
                    finally:
                        if solver is not None:solver.close_cudss()
                    jax.clear_caches();gc.collect()
                if "rd" in a.problems:
                    solver=None
                    try:
                        dt,steps=timestep(h,order,factor=a.dt_factor); nu=rd.solver_settings(dim)[2]
                        print("ASSEMBLE",geometry,h,order,"RD",dt,steps,flush=True)
                        solver=gate
                        assert precheck(solver,domain,geometry,h,order,"rd")["passed"],"actual RD Laplacian failed spectral precheck"

                        for family in rd.FAMILIES:
                            if recorded(geometry,h,order,"rd",family):continue
                            try:
                                t0=time.perf_counter()
                                u=rd_final(solver,family,rd.sample_theta(family,dim),steps)
                                err=error(u,rd.exact_state(.5,x)); rollout_s=time.perf_counter()-t0
                                np.savez(a.output/f"rd_{geometry}_h{h:g}_p{order}_{family}.npz",state=np.asarray(u),exact=np.asarray(rd.exact_state(.5,x)))
                                record(dict(**common,rollout_s=rollout_s,problem="rd",family=family,status="ok",dt=dt,steps=steps,relative_l2=err,cudss_phases=phase_counts(solver)))
                            except Exception as e:
                                traceback.print_exc(); record(dict(**common,problem="rd",family=family,status="failed",dt=dt,steps=steps,error=repr(e)))
                    except Exception as e:
                        traceback.print_exc()
                        for family in rd.FAMILIES:
                            if not recorded(geometry,h,order,"rd",family):record(dict(**common,problem="rd",family=family,status="failed",error=repr(e)))
                    finally:
                        if solver is not None:solver.close_cudss()
                    jax.clear_caches();gc.collect()
                if "ad" in a.problems and not recorded(geometry,h,order,"ad","prescribed"):
                    solver=None
                    try:
                        dt,steps=timestep(h,order,factor=a.ad_dt_factor or a.dt_factor,cfl_factor=a.ad_dt_cfl_factor)
                        print("ASSEMBLE",geometry,h,order,"AD",dt,steps,flush=True)
                        t0=time.perf_counter()
                        solver,gamma,gamma_status=ad.build_solver_template(domain,order=order,dt=dt,nu=.05,stencil="rbf",linear_solver="cudss",
                            velocity_amplitude=a.ad_velocity_amplitude,gamma_mode="estimate",fixed_gamma=-1e-4,gamma_scale=1.,gamma_h_power=0.,
                            hyperviscosity_power=(str(order//2+1) if a.ad_hyperviscosity_power=="order" else a.ad_hyperviscosity_power),
                            boundary_order=None,hyperviscosity_row_mask=ad_row_mask,
                            hyperviscosity_method=a.ad_hyperviscosity_method,
                            hyperviscosity_treatment=a.ad_treatment,laplacian_power_solver="sparse",
                            hyperviscosity_mask_distance_factor=2.,sparse_assembly_chunk_size=128,gamma_arpack_ncv=None,
                            gamma_arpack_maxiter=None,setup_device="default",gmres_tol=1e-8,gmres_atol=1e-10,
                            gmres_restart=40,gmres_maxiter=None,ilu_application="host_callback",
                            linear_solve_diagnostics=False,linear_solve_diagnostics_threshold=1e-6,force_boundary_paired_ghost_stencil=False)
                        t1=time.perf_counter()
                        assert precheck(solver,domain,geometry,h,order,"ad")["passed"],"actual AD generator failed spectral precheck"
                        timing=dict(setup_s=t1-t0,precheck_s=time.perf_counter()-t1)
                        # The established field has a separable sin(pi*t) time factor.
                        # Its fixed-site spatial values are evaluated once, not per step.
                        spatial=(a.ad_velocity_amplitude*reference_velocity.forward_velocity(geometry,x) if a.ad_velocity_field=="reference"
                                 else ad.incompressible_boundary_vanishing_velocity(domain,a.ad_velocity_amplitude)(.5,x))
                        assert bool(jnp.isfinite(spatial).all())
                        def velocity(t,z):
                            assert z.shape==x.shape
                            return jnp.sin(jnp.pi*t)*spatial
                        t0=time.perf_counter()
                        u=ad_final(solver,gamma,velocity,steps); err=error(u,ad.exact_state(.5,x)); timing["rollout_s"]=time.perf_counter()-t0
                        np.savez(a.output/f"ad_{geometry}_h{h:g}_p{order}.npz",state=np.asarray(u),exact=np.asarray(ad.exact_state(.5,x)))
                        row=dict(**common,**timing,problem="ad",family="prescribed",status="ok",dt=dt,steps=steps,relative_l2=err,
                            gamma=np.asarray(gamma).tolist(),gamma_status=gamma_status,hyperviscosity_power=solver.hyperviscosity_power,
                            hyperviscosity_method=solver.hyperviscosity_method,
                            hyperviscosity_rows=a.ad_hyperviscosity_rows,
                            laplacian_power_solver=solver.laplacian_power_solver,
                            hyperviscosity_treatment=solver.hyperviscosity_treatment,
                            hyperviscosity_boundary_protocol=ad_boundary_protocol,
                            cudss_phases=phase_counts(solver))
                        record(row)
                    except Exception as e:
                        traceback.print_exc();record(dict(**common,problem="ad",family="prescribed",status="failed",
                            hyperviscosity_method=a.ad_hyperviscosity_method,laplacian_power_solver="sparse",
                            hyperviscosity_rows=a.ad_hyperviscosity_rows,
                            hyperviscosity_treatment=a.ad_treatment,
                            hyperviscosity_boundary_protocol=ad_boundary_protocol,error=repr(e)))
                    finally:
                        if solver is not None:solver.close_cudss()
                    jax.clear_caches();gc.collect()
            del domain,default_domain,surface; jax.clear_caches();gc.collect()
    (a.output/"completed.json").write_text(json.dumps(dict(records=len(rows),failed=sum(r["status"]!="ok" for r in rows)),indent=2)+"\n")


if __name__=="__main__":main()
