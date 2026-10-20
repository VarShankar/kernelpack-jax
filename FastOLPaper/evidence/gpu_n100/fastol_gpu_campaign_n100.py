"""One immutable, GPU-only FastOL paper experiment per fresh process.

File decoding, split identity metadata and artifact serialization are host work.
Field transforms, normalization, training, validation and prediction use CUDA.
This file does not run when imported. See docs/fastol_gpu_campaign.md.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import inspect
import json
import logging
import math
import os
from pathlib import Path
import statistics
import subprocess
import time


CASES = {
    "taylor_green": ("taylor_green", "data_exact_matt.mat", "taylor_green_fekete.mat", "init_velocity", 500, True),
    "taylor_green_time": ("taylor_green", "data_time.mat", "taylor_green_fekete.mat", "init_velocity", 500, True),
    "species_transport": ("species_transport", "data.mat", "species_transport_fekete.mat", "init_velocity", 7000, False),
    "forced_turb": ("forced_turb", "data.mat", "forced_turb_fekete.mat", "forcing", 7000, True),
    "merge_vortices_easier": ("merge_vortices_easier", "data.mat", "merge_vortices_easier_fekete.mat", "init_vorticity", 500, True),
    "lid_cavity_flow": ("lid_cavity_flow", "data.mat", "lid_cavity_flow_fekete.mat", "init_velocity", 1000, False),
    "flow_cylinder_laminar": ("flow_cylinder", "data_laminar.mat", "flow_cylinder_laminar_fekete.mat", "init_velocity", 1000, False),
    "flow_cylinder_shedding": ("flow_cylinder", "data_shedding.mat", "flow_cylinder_shedding_fekete.mat", "init_velocity", 1000, False),
    "backward_facing_step": ("backward_facing_step", "data.mat", "backward_facing_step_fekete.mat", "init_velocity", 1000, False),
    "buoyancy_cavity_flow": ("buoyancy_cavity_flow", "data.mat", "buoyancy_cavity_flow_fekete.mat", "init_temperature", 5000, False),
}
TRAINING_SIZES = (100, 500, 1000, 5000, 10000)
COVARIANCE_FAMILY = "wendland_c4_3d_times_matern52_residual_gate"
MODEL_SPECS = (("fixed_affine", False, 15), ("trainable_affine", True, 15),
               ("fixed_no_poly", False, 0), ("trainable_no_poly", True, 0))


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf8")).hexdigest()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new_json(path, value):
    # Exclusive creation prevents accidentally overwriting a frozen selection.
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if hasattr(value, "shape"):
        # Explicit artifact export only; this copy is never used in numerics.
        import jax
        return json_value(jax.device_get(value).tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def protocol(root, case, n):
    """Resolve fixed parameters from the latest corrected spatial campaign."""
    initial = root / "validation_results_v5" / case
    campaign = ("spacetime_polynomial_20261005" if case == "taylor_green_time" else
                "multiscale_matern_stable_20261005" if case in ("forced_turb", "species_transport") else
                "referee_periodic_20261004" if case in ("taylor_green", "merge_vortices_easier") else
                "validation_results_v5")
    final = root / campaign / case
    settings_path = final / f"selection_N_{n}.csv"
    if not settings_path.exists():
        # Periodic replay froze the original supports; it stored MAT selections.
        settings_path = initial / f"selection_N_{n}.csv"
    settings = rows(settings_path)
    if len(settings) != 1:
        raise ValueError(f"Expected one frozen selection: {settings_path}")
    selected = settings[0]
    test_csv = final / "test" / "exhaustive_results_long.csv"
    local = ("local_dfmatern_multiscale_poly" if case in ("forced_turb", "species_transport") else "local_dfphs_poly")
    controls = [r for r in rows(test_csv) if int(r["N"]) == n and r["spatial_family"] == local
                and r["lagrange_method"] == "1" and r["assembly_method"] in ("full", "baseline")]
    prescribed = {}
    # Existing baseline parameter metadata are allowed here; baseline errors
    # are intentionally excluded until AFTER the new selection is frozen.
    for family in ("sparse_c4_wendland", "dense_c4_matern"):
        matched = [r for r in controls if r["operator_family"] == family]
        if len(matched) != 1:
            raise ValueError(f"Expected one {family} control in {test_csv} at N={n}")
        r = matched[0]
        prescribed[family] = {key: r[key] for key in (
            "operator_ep", "operator_nugget", "num_spatial_points", "input_spatial_points",
            "input_representation", "local_order", "spatial_family", "code_identifier", "data_identifier")}
    if prescribed["sparse_c4_wendland"]["operator_ep"] != selected["operator_ep"]:
        raise ValueError("Replay support does not match its frozen selection")
    return {"case": case, "N": n, "campaign": campaign, "baseline_csv": str(test_csv),
            "baseline_csv_sha256": sha256(test_csv), "selection_csv": str(settings_path),
            "selection_csv_sha256": sha256(settings_path), "split_file": str(final / "example_split.mat"),
            "prescribed": prescribed, "spatial_family": local}


def decode_field(path, name, example_ids, device):
    """Compatibility alias for the shared GPU file adapter."""
    from kernelpack.operator_learning.datasets import read_fastol_field_gpu
    return read_fastol_field_gpu(path,name,example_ids,device=device)


def physical_indices(path, field, case, logical_ids):
    """Compatibility alias for shared archived example identities."""
    from kernelpack.operator_learning.datasets import fastol_physical_indices
    return fastol_physical_indices(path,field,case,logical_ids)


def load_fields(data_root, case, logical_ids, settings, device):
    from kernelpack.operator_learning.datasets import load_fastol_paper_dataset
    data,actual,permutation=load_fastol_paper_dataset(data_root,case,logical_ids,
        output_points=int(settings['num_spatial_points']),input_points=int(settings['input_spatial_points']),
        representation=settings['input_representation'],device=device)
    return data.inputs,data.points,data.velocity,data.source_files,actual,permutation


def spatial_options(case, selected):
    options = {"order": int(float(selected["local_order"])), "use_periodic": CASES[case][-1]}
    if case == "forced_turb":
        options.update(periodic_dims=[True] * 3, domain_bounds=[[0.] * 3, [2 * math.pi] * 3])
    if case in ("forced_turb", "species_transport"):
        options.update(spatial_kernel="matern_multiscale", matern_base_length_factor=16.,
                       matern_num_modes=5, matern_gamma=8 / 3)
    if case == "taylor_green_time":
        options.update(time_points=[.7, .8, .9, 1.], temporal_interpolation="cubic_polynomial")
    return options


def memory_snapshot(device):
    """JAX allocator figures and process memory are distinct measurements."""
    report = {"jax_allocator": device.memory_stats(), "process_gpu_bytes": None,
              "note": "Process sampling may be unavailable under WSL; allocator statistics are not total live model memory."}
    try:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            pid, used = [part.strip() for part in line.split(",")]
            if int(pid) == os.getpid() and used.isdigit():
                report["process_gpu_bytes"] = int(used) * 1024**2
    except (OSError, ValueError, subprocess.CalledProcessError):
        pass
    return report


def counted_model_arrays(model, predictor):
    """Count owned visible GPU buffers, explicitly excluding opaque allocations."""
    import jax
    seen_objects, seen_buffers = set(), set()
    arrays = []
    def visit(value, label):
        if id(value) in seen_objects:
            return
        seen_objects.add(id(value))
        if isinstance(value, jax.Array):
            key = (str(value.device), value.unsafe_buffer_pointer())
            if key not in seen_buffers:
                seen_buffers.add(key)
                arrays.append({"name": label, "bytes": value.nbytes, "shape": list(value.shape), "dtype": str(value.dtype)})
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{label}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{label}[{index}]")
        elif value.__class__.__module__.startswith("kernelpack") and hasattr(value, "__dict__"):
            for key, child in value.__dict__.items():
                visit(child, f"{label}.{key}")
    visit(model, "model")
    visit(predictor.buffers,"prepared_predictor")
    return {"visible_model_array_bytes": sum(item["bytes"] for item in arrays),
            "prepared_training_field_bytes": predictor.precomputed_bytes,
            "arrays": arrays,
            "exclusions": "Native cuDSS storage is reported separately; XLA executables, temporary buffers and allocator overhead are excluded. This is not total GPU storage."}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--N", type=int, choices=TRAINING_SIZES, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/e/fluid_datasets"))
    parser.add_argument("--baseline-root", type=Path, default=Path("/mnt/e/kernelpack-matlab-validation/experiments/fastol"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-steps", type=int, default=60)
    parser.add_argument("--validation-steps", type=int, default=35)
    parser.add_argument("--repetitions", type=int, default=11)
    parser.add_argument("--spatial-batch-size", type=int, default=32)
    parser.add_argument("--spatial-setup-mib", type=int, default=1024)
    parser.add_argument("--policy", choices=("accuracy",), default="accuracy")
    parser.add_argument("--cudss-reordering", choices=("amd",), default="amd")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(name)s %(levelname)s %(message)s')
    if args.repetitions < 3:
        parser.error("At least three synchronized timing repetitions are required")
    if args.spatial_setup_mib<1:
        parser.error("Spatial setup memory budget must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_PLATFORMS", "cuda")
    import jax
    jax.config.update("jax_enable_x64", True)
    if hasattr(jax.config,"jax_use_simplified_jaxpr_constants"):
        jax.config.update("jax_use_simplified_jaxpr_constants",True)
    import jax.numpy as jnp
    from kernelpack.operator_learning import FastOL, KernelTrainingConfig, load_frozen_split
    from kernelpack.operator_learning.trainable import train_compact_kernel, calibrate_compact_kernel, freeze_compact_operator
    from kernelpack.operator_learning.local_dfphs import LocalDFPHSDecoder
    devices = jax.devices("gpu")
    if len(devices) != 1:
        raise RuntimeError("Select exactly one CUDA device with CUDA_VISIBLE_DEVICES")
    device = devices[0]
    def progress(stage,**extra):
        print(json.dumps(dict(timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            stage=stage,dataset=args.case,N=args.N,**extra)),flush=True)
    progress('loading_training_validation')
    # N=100 is not in the MATLAB protocol; its N-independent settings (spatial reconstruction, input
    # sampling, frozen split) are taken from N=500. Our method selects its own radius and regularization.
    record = protocol(args.baseline_root, args.case, max(args.N, 500)); record["N"] = args.N
    fixed = record["prescribed"]["sparse_c4_wendland"]
    options = spatial_options(args.case, fixed)
    unsupported = set(options) - set(inspect.signature(LocalDFPHSDecoder).parameters)
    if unsupported:
        raise RuntimeError(f"Missing exact paper spatial protocol: {sorted(unsupported)}; no substitute is permitted")
    split = load_frozen_split(record["split_file"])
    training_ids = split["train"][:args.N].tolist()
    validation_ids = split["validation"].tolist()
    test_ids = split["test"].tolist()
    if len(training_ids) != args.N or len(validation_ids) != 100 or len(test_ids) != 100:
        raise ValueError("Paper protocol requires N training, 100 validation, 100 test examples")
    ids = training_ids + validation_ids
    inputs, points, fields, sources, actual_ids, permutation = load_fields(args.data_root, args.case, ids, fixed, device)
    x, xv, y, yv = inputs[:args.N], inputs[args.N:], fields[:args.N], fields[args.N:]
    fields.block_until_ready()
    del inputs, fields
    progress('training_validation_loaded',input_shape=list(x.shape),output_shape=list(y.shape))
    record.update(source_files={str(path): sha256(path) for path in sources}, split_sha256=sha256(record["split_file"]),
                  training_ids=training_ids, validation_ids=validation_ids, test_ids=test_ids,
                  physical_training_validation_ids=actual_ids, spatial_options=options,
                  input_shape=list(x.shape), output_shape=list(y.shape), dtype=str(x.dtype),
                  gpu=str(device), jax_version=jax.__version__, cuda_graphs=True,
                  test_targets_loaded=False, policy=args.policy, cudss_reordering=args.cudss_reordering)
    record.update(schema_version="matched_dimension_safe_v1", covariance_family=COVARIANCE_FAMILY,
        model_names=[name for name, _, _ in MODEL_SPECS], operator_evaluation="lagrange",
        initial_support_density=.25, initial_nugget=1e-6, reference_density=1.,
        density_fractions=[.10,.25,.50,.75,.90], validation_start_policy="common_budget",
        timed_native_query_batch_size=32, timed_query_count=100,
        comparison="fixed versus training-learned covariance shape, each with and without an identical affine mean",
        fixed_shape="zero log_parallel_scales and perpendicular_ratio=1; validation selects radius and nugget",
        shared_transform="Training-only normalization/PCA/gates and identical explicit Taylor-Green response transformation in all four models")
    record["comparison_contract"]={"covariance_family":COVARIANCE_FAMILY,
        "initial_support_density_target":.25,"initial_nugget":1e-6,"reference_density":1.,
        "models":[name for name,_,_ in MODEL_SPECS],"same_polynomial_within_pairs":True,
        "operator_evaluation":"lagrange","native_query_batch_size":32,
        "validation_fields":100,"test_fields":100}
    record["memory_budgets"]={"spatial_setup_bytes":args.spatial_setup_mib*1024**2,
        "training_geometry_cache_bytes":1024**3,"training_assembly_bytes":32*1024**2}
    if permutation is not None:
        write_new_json(args.output / "forced_loader_permutation.json", permutation)
    repo = Path(__file__).resolve().parents[1]
    tracked = {Path(__file__).resolve(),repo / "scripts/run_fastol_gpu_campaign.py",
               *(repo / "src/kernelpack").rglob("*.py"),
               repo / "src/kernelpack/sparse/_native/cudss_ffi.cc",
               repo / "src/kernelpack/sparse/_cudss_ffi.so"}
    record["source_sha256"] = {str(path.relative_to(repo)): sha256(path) for path in sorted(tracked)}
    native_path=Path(os.environ.get('KERNELPACK_CUDSS_LIBRARY',repo/'src/kernelpack/sparse/_cudss_ffi.so')).resolve()
    record['native_library']={'path':str(native_path),'sha256':sha256(native_path)}
    write_new_json(args.output / "protocol.json", record)
    snapshots = {"data_loaded": memory_snapshot(device)}

    progress('shared_spatial_preparation_started')
    spatial_start=time.perf_counter()
    spatial_reconstructor=FastOL.prepare_spatial_reconstructor(points,y,
        spatial_batch_size=args.spatial_batch_size,
        spatial_setup_bytes=args.spatial_setup_mib*1024**2,**options)
    jax.block_until_ready(spatial_reconstructor.buffers)
    spatial_preparation_seconds=time.perf_counter()-spatial_start
    progress('shared_spatial_preparation_complete',seconds=spatial_preparation_seconds,
        buffer_bytes=sum(spatial_reconstructor.buffers_sizes))
    snapshots['spatial_prepared']=memory_snapshot(device)

    @jax.jit
    def field_error(prediction, truth):
        axes = (1, 3) if truth.ndim == 4 else (1, 2)
        numerator = jnp.linalg.norm(prediction - truth, axis=axes)
        denominator = jnp.linalg.norm(truth, axis=axes)
        return jnp.mean(numerator / jnp.where(denominator == 0, 1., denominator))

    # All four selections are frozen before the first held-out target load.
    # The old ambient-distance Wendland is archived metadata only.
    import numpy as np
    shared_geometry = None
    selected_fits = {}
    selected_records = {}
    selected_validation_fields = {}
    for name, optimize, mean_dimension in MODEL_SPECS:
        config = KernelTrainingConfig(mean_dimension=mean_dimension,
            training_steps=args.training_steps, validation_steps=args.validation_steps,
            cudss_reordering=args.cudss_reordering, validation_start_policy="common_budget",
            response_scaling="taylor_green_ray" if args.case in ("taylor_green","taylor_green_time") else "none")
        progress('model_selection_started',model=name,config=asdict(config),optimize_geometry=optimize)
        start=time.perf_counter()
        initial_radius=None if shared_geometry is None else float(jnp.exp(shared_geometry.initial[0]))
        fit=train_compact_kernel(x,y,training_ids=jnp.asarray(training_ids,device=device),
            radius=initial_radius,nugget=1e-6,config=config,optimize=optimize,geometry=shared_geometry)
        fit=replace(fit,supplied_modes=None)
        if shared_geometry is None:
            shared_geometry=fit
            common={key:jax.device_get(getattr(fit,key)) for key in
                ("center","basis","parallel_basis","mean_scale","delta","response_scale")}
            common.update(normalizer_mean=jax.device_get(fit.normalizer.mean),
                normalizer_std=jax.device_get(fit.normalizer.std_safe),normalizer_zero=jax.device_get(fit.normalizer.zero))
            if fit.ray_direction is not None:
                common.update({key:jax.device_get(getattr(fit,key)) for key in
                    ("ray_direction","ray_zero","ray_median")})
            np.savez(args.output/"shared_geometry.npz",**common)
        selection=calibrate_compact_kernel(fit,xv,yv,
            validation_ids=jnp.asarray(validation_ids,device=device),reference_density=1.,
            reference_error=None,policy=args.policy,reconstruct=spatial_reconstructor,backend="cudss")
        elapsed=time.perf_counter()-start
        op=selection.operator
        op.P.block_until_ready()
        selected_weights=op.lagrange_map(xv)
        selected_fields=(selected_weights@y.reshape(args.N,-1)).reshape((len(xv),)+y.shape[1:])
        selected_validation_fields[name]=spatial_reconstructor(selected_fields).block_until_ready()
        del selected_weights,selected_fields
        geometry={key:jax.device_get(getattr(op,key)) for key in
            ("center","basis","parallel_basis","scale","delta","log_parallel_scales")}
        geometry.update(normalizer_mean=jax.device_get(op.normalizer.mean),
            normalizer_std=jax.device_get(op.normalizer.std_safe),normalizer_zero=jax.device_get(op.normalizer.zero))
        if config.response_scaling!="none":
            geometry.update({key:jax.device_get(getattr(op,key)) for key in
                ("response_scale","ray_direction","ray_zero","ray_median")})
        geometry_file=args.output/f"selected_geometry_{name}.npz"
        np.savez(geometry_file,**geometry)
        # Export bytes only for provenance; no host array feeds any numerics.
        mean_hash=hashlib.sha256(jax.device_get(op.P).tobytes(order="C")).hexdigest()
        selected_records[name]=dict(config=asdict(config),parameters=json_value(selection.parameters),
            mean_dimension=mean_dimension,polynomial_columns=op.P.shape[1],mean_basis_sha256=mean_hash,
            covariance_family=COVARIANCE_FAMILY,
            selection_kind="trained_geometry_validation" if optimize else "fixed_geometry_validation",
            training_steps_executed=args.training_steps if optimize else 0,
            radius=op.radius,nugget=op.nugget,perpendicular_ratio=op.perpendicular_ratio,
            log_parallel_scales=json_value(op.log_parallel_scales),
            validation_error=selection.validation_error,training_kernel_density=selection.density,
            meets_validation_reference=selection.meets_reference,
            candidates=json_value(selection.candidates),training_histories=json_value(fit.histories),
            training_validation_seconds=elapsed,training_timings=json_value(fit.timings),
            validation_component_timings=json_value(selection.timings),
            output_energy_factor=json_value(fit.output_factor_metadata),
            geometry_file=geometry_file.name,geometry_sha256=sha256(geometry_file))
        selected_fits[name]=(replace(fit,perpendicular_distances=None,supplied_modes=None),selection.parameters)
        selection.close()
        del selection,op,fit
        progress('model_selection_complete',model=name,validation_error=selected_records[name]['validation_error'],
            density=selected_records[name]['training_kernel_density'],seconds=elapsed)
    if selected_records['fixed_affine']['mean_basis_sha256']!=selected_records['trainable_affine']['mean_basis_sha256']:
        raise RuntimeError("Fixed and trainable affine models did not use exactly the same polynomial basis")
    # No fitting-only N-by-N distance cache remains during held-out timings.
    shared_geometry=None
    frozen=dict(schema_version="matched_dimension_safe_v1",dataset=args.case,N=args.N,policy=args.policy,
        protocol_sha256=sha256(args.output/"protocol.json"),test_targets_loaded=False,
        shared_geometry_sha256=sha256(args.output/"shared_geometry.npz"),models=selected_records)
    write_new_json(args.output/"selection.json",frozen)
    frozen_hash=sha256(args.output/"selection.json")
    progress('all_four_selections_frozen',selection_sha256=frozen_hash)
    snapshots['selection_complete']=memory_snapshot(device)
    # FIRST held-out target delivery, after all four immutable selections.
    q,held_points,truth,_,held_actual,_=load_fields(args.data_root,args.case,test_ids,fixed,device)
    if not bool(jnp.all(held_points==points)):
        raise ValueError("Held-out geometry differs from frozen reconstruction")
    if sha256(args.output/"selection.json")!=frozen_hash:
        raise RuntimeError("Frozen selection was modified")
    results={}

    def evaluate(name, fitted):
        t = time.perf_counter()
        predictor = fitted.prepare_predictor(operator_evaluation="lagrange",
            spatial_reconstructor=spatial_reconstructor)
        jax.block_until_ready(predictor.buffers)
        preparation = time.perf_counter() - t
        # Prime the PUBLIC batching route, then time exactly its native32 scan.
        batch=32
        t=time.perf_counter()
        primed=fitted.predict(q,query_batch_size=batch).block_until_ready()
        cached=fitted._prepared_prediction
        batch_predictor=cached[1]
        runner=cached[4][batch]
        padded_q=jnp.pad(q,((0,(-len(q))%batch),(0,0)),mode='edge').reshape(-1,batch,q.shape[-1])
        @jax.jit
        def timed_run(queries,buffers):
            values=runner(queries,buffers)
            return values.reshape((-1,)+values.shape[2:])[:len(q)]
        with jax.transfer_guard_device_to_host("disallow"):
            prediction=timed_run(padded_q,batch_predictor.buffers).block_until_ready()
        first=time.perf_counter()-t
        if not bool(jnp.all(jnp.isfinite(prediction))):
            raise RuntimeError("Nonfinite checked prediction")
        factor=fitted.operator_model.sparse_factor
        launches_before=factor.graph_launch_count
        solves_before=factor.solve_count
        durations=[]
        for _ in range(args.repetitions):
            t=time.perf_counter()
            with jax.transfer_guard("disallow"):
                prediction=timed_run(padded_q,batch_predictor.buffers).block_until_ready()
            durations.append(time.perf_counter()-t)
        replay_count = None if factor is None else factor.graph_launch_count - launches_before
        native_solves=factor.solve_count-solves_before
        expected_solves=args.repetitions*len(padded_q)
        if replay_count<expected_solves or native_solves!=expected_solves:
            raise RuntimeError("Every timed native32 batch must use one sparse Lagrange solve and CUDA graph replay")
        # Capture measured storage before extra verification creates additional
        # query-width graphs or predictor caches. Checks never alter selection.
        native_storage=None if factor is None else factor.storage_statistics()
        visible_storage=counted_model_arrays(fitted,batch_predictor)
        measured_memory=memory_snapshot(device)
        if fitted._shared_spatial_reconstructor is not spatial_reconstructor:
            raise RuntimeError("The model did not retain the single shared spatial map")
        output_shape=fitted.velocity.shape[1:]
        def validation_reference(queries,state,fields,spatial_buffers):
            weights=fitted.lagrange_map(queries,state=state)
            combined=(weights@fields.reshape(len(fields),-1)).reshape((len(queries),)+output_shape)
            return spatial_reconstructor.compiled(combined,spatial_buffers)
        reference_function=jax.jit(validation_reference)
        state=fitted.device_state()
        with jax.transfer_guard_device_to_host('disallow'):
            reference=reference_function(xv,state,fitted.velocity,spatial_reconstructor.buffers).block_until_ready()
            compiled_validation=predictor(xv).block_until_ready()
        validation_gap=jnp.linalg.norm(compiled_validation-reference)/jnp.maximum(jnp.linalg.norm(reference),1.)
        selected_reference=selected_validation_fields.pop(name)
        frozen_handoff_gap=jnp.linalg.norm(compiled_validation-selected_reference)/jnp.maximum(jnp.linalg.norm(selected_reference),1.)
        reversed_queries=q[::-1]
        with jax.transfer_guard('disallow'):
            reversed_prediction=predictor(reversed_queries).block_until_ready()
        reverse_gap=jnp.linalg.norm(reversed_prediction-prediction[::-1])/jnp.maximum(jnp.linalg.norm(prediction),1.)
        # Exercise the ordinary public native query-batch route on ALL100
        # held-out examples, then guard its compiled scan with explicit state.
        batch=32
        batched=fitted.predict(q,query_batch_size=batch).block_until_ready()
        cached=fitted._prepared_prediction
        batch_predictor=cached[1]
        runner=cached[4][batch]
        padded_q=jnp.pad(q,((0,(-len(q))%batch),(0,0)),mode='edge').reshape(-1,batch,q.shape[-1])
        with jax.transfer_guard('disallow'):
            guarded_batch=runner(padded_q,batch_predictor.buffers).block_until_ready()
        guarded_batch=guarded_batch.reshape((-1,)+guarded_batch.shape[2:])[:len(q)]
        with jax.transfer_guard('disallow'):
            unbatched=predictor(q).block_until_ready()
        batch_gap=jnp.linalg.norm(batched-unbatched)/jnp.maximum(jnp.linalg.norm(unbatched),1.)
        guarded_batch_gap=jnp.maximum(
            jnp.linalg.norm(guarded_batch-batched)/jnp.maximum(jnp.linalg.norm(batched),1.),
            jnp.linalg.norm(prediction-batched)/jnp.maximum(jnp.linalg.norm(batched),1.))
        gaps=jnp.stack([validation_gap,frozen_handoff_gap,reverse_gap,batch_gap,guarded_batch_gap])
        if not bool(jnp.all(jnp.isfinite(gaps))&jnp.all(gaps<=1e-8)):
            raise RuntimeError("Physical-field, reversed-query or native-batching accuracy check failed")
        checks=dict(validation_field_relative_gap=float(validation_gap),
            frozen_selection_field_relative_gap=float(frozen_handoff_gap),
            compiled_validation_error=float(field_error(compiled_validation,yv)),
            reversed_query_relative_gap=float(reverse_gap),native_query_batch_size=batch,
            native_query_batch_relative_gap=float(batch_gap),guarded_batch_relative_gap=float(guarded_batch_gap),
            native_query_batch_transfer_guard_passed=True,shared_spatial_map_reused=True,
            timed_native_query_batch_size=32,timed_query_count=100,
            timed_native_solve_count=native_solves,expected_timed_native_solves=expected_solves,passed=True)
        # Explicit artifact export after all timings; no exported arrays feed
        # back into fitting, validation, prediction or numerical comparisons.
        axes=(1,3) if truth.ndim==4 else (1,2)
        denominator=jnp.linalg.norm(truth,axis=axes)
        errors=jnp.linalg.norm(prediction-truth,axis=axes)/jnp.where(denominator==0,1.,denominator)
        prediction_file=args.output/f"{name}_predictions.npz"
        np.savez(prediction_file,prediction=jax.device_get(prediction),relative_error=jax.device_get(errors))
        return {"test_relative_field_error": float(field_error(prediction, truth)),
                "prediction_file":prediction_file.name,"prediction_sha256":sha256(prediction_file),
                "preparation_seconds": preparation, "first_prediction_compile_capture_seconds": first,
                "operator_setup_seconds":getattr(fitted,'operator_setup_seconds',None),
                "shared_spatial_preparation_seconds":spatial_preparation_seconds,
                "warm_prediction_seconds": durations, "warm_prediction_median_seconds": statistics.median(durations),
                "spatial_mode": predictor.spatial_mode, "precomputed_bytes": predictor.precomputed_bytes,
                "graph_replays_during_timing": replay_count,
                "gpu_transfer_guard_passed": True,
                "timed_native_query_batch_size":32,"timed_query_count":100,
                "native_factor_storage": native_storage,
                "counted_storage": visible_storage,
                "accuracy_checks":checks,
                "memory": measured_memory}

    for name,_,_ in MODEL_SPECS:
        fit,theta=selected_fits.pop(name)
        start=time.perf_counter()
        operator=freeze_compact_operator(fit,theta)
        fitted=FastOL(x,points,y,epsilon=1.,operator_model=operator,backend="cudss",
            spatial_batch_size=args.spatial_batch_size,spatial_setup_bytes=args.spatial_setup_mib*1024**2,
            spatial_reconstructor=spatial_reconstructor,**options)
        jax.block_until_ready(fitted.device_state())
        fitted.operator_setup_seconds=time.perf_counter()-start
        try:
            results[name]=evaluate(name,fitted)
            results[name].update({key:selected_records[name][key] for key in
                ("mean_dimension","polynomial_columns","mean_basis_sha256","covariance_family","geometry_sha256")})
            results[name]['selected_model_sha256']=canonical_digest(selected_records[name])
            results[name]['training_validation_seconds']=selected_records[name]['training_validation_seconds']
            progress('model_test_complete',model=name,error=results[name]['test_relative_field_error'])
        finally:
            fitted.close()
        del fitted,operator,fit
    archive = [r for r in rows(record["baseline_csv"]) if int(r["N"]) == args.N
               and r["spatial_family"] == record["spatial_family"] and r["lagrange_method"] == "1"
               and r["assembly_method"] in ("full", "baseline")]
    output = {"schema_version":"matched_dimension_safe_v1", "dataset": args.case, "N": args.N, "selection_sha256": frozen_hash,
              "test_ids": test_ids, "physical_test_ids": held_actual, "models": results,
              "archived_baseline_rows": archive, "memory_snapshots": snapshots,
              "shared_spatial_preparation_seconds":spatial_preparation_seconds,
              "shared_spatial_preparation_count":1,
              "storage_note": "Kernel density is not total storage. Native factor nnz and analysis-time permanent/peak estimates are separate from JAX buffers and observed process memory; estimates are not measured whole-process peaks.",
              "test_targets_used_for_selection": False,
              "shared_geometry_sha256":frozen["shared_geometry_sha256"],
              "timing_note":"Four models share one spatial setup. Native batches of32 process100 fields with edge-padding. Fresh process per dataset/size, but JIT compilation can be reused between model variants; setup is not an independently cold process per model."}
    write_new_json(args.output / "results.json", json_value(output))
    print(json.dumps(json_value(output)), flush=True)


if __name__ == "__main__":
    main()
