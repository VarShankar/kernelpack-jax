"""Supervise fresh-process GPU paper cells with a persistent progress journal.

No JAX/NumPy imports occur here. All scientific numerics belong to the GPU-only
single-cell driver. Without --launch this prints a plan and starts no jobs.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import time
import zipfile


CASES=("taylor_green","forced_turb","merge_vortices_easier","taylor_green_time",
    "species_transport","flow_cylinder_laminar","flow_cylinder_shedding",
    "backward_facing_step","buoyancy_cavity_flow","lid_cavity_flow")
SIZES=(10000,5000,1000,500)
MODELS=("fixed_affine","trainable_affine","fixed_no_poly","trainable_no_poly")
SCHEMA="matched_dimension_safe_v1"
COVARIANCE="wendland_c4_3d_times_matern52_residual_gate"


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def local_path(name):
    path=Path(name)
    if sys.platform=="win32" and str(name).startswith("/mnt/"):
        parts=str(name).split("/")
        return Path(parts[2].upper()+":/"+"/".join(parts[3:]))
    return path


def npz_array_header(path,name):
    """Read shape/dtype metadata only; never load or evaluate a saved array."""
    with zipfile.ZipFile(path) as archive, archive.open(name+".npy") as stream:
        if stream.read(6)!=b"\x93NUMPY":
            raise RuntimeError("Invalid exported NPY magic: "+str(path))
        version=stream.read(2)
        if version==b"\x01\x00":
            length=struct.unpack("<H",stream.read(2))[0]
        elif version in (b"\x02\x00",b"\x03\x00"):
            length=struct.unpack("<I",stream.read(4))[0]
        else:
            raise RuntimeError("Unsupported exported NPY version: "+str(path))
        if length>1024**2:
            raise RuntimeError("Unreasonably large exported NPY header")
        return ast.literal_eval(stream.read(length).decode("utf-8" if version==b"\x03\x00" else "latin1"))


def write_atomic(path,value):
    temporary=path.with_name(path.name+f".{os.getpid()}.tmp")
    with temporary.open("w",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,path)


def source_manifest(repo):
    files={repo/"scripts/fastol_gpu_campaign.py",repo/"scripts/run_fastol_gpu_campaign.py",
        repo/"src/kernelpack/sparse/_native/cudss_ffi.cc",
        repo/"src/kernelpack/sparse/_cudss_ffi.so"}
    files.update((repo/"src/kernelpack").rglob("*.py"))
    return {path.relative_to(repo).as_posix():digest(path) for path in sorted(files)}


def freeze_sources(repo,root,expected):
    """Retain the executed source, including pre-existing local changes."""
    snapshot=root/"source_snapshot"
    for name,checksum in expected.items():
        target=snapshot/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(repo/name,target)
        if digest(target)!=checksum:
            raise RuntimeError("Source changed while freezing the campaign: "+name)
    if source_manifest(repo)!=expected:
        raise RuntimeError("Source changed while freezing the campaign")


def verify_snapshot(snapshot,expected):
    actual=source_manifest(snapshot)
    if actual!=expected:
        changed=sorted(name for name in set(actual)|set(expected) if actual.get(name)!=expected.get(name))
        raise RuntimeError("Frozen execution source/native snapshot changed: "+", ".join(changed))


def gpu_inventory(index):
    command=["nvidia-smi","-i",str(index),"--query-compute-apps=pid,process_name","--format=csv,noheader,nounits"]
    try:
        result=subprocess.run(command,capture_output=True,text=True,timeout=10,check=False)
    except (OSError,subprocess.TimeoutExpired) as error:
        return {"available":False,"reason":str(error),"processes":[]}
    if result.returncode:
        return {"available":False,"reason":result.stderr.strip() or result.stdout.strip(),"processes":[]}
    processes=[]
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields=line.split(",",1)
        if len(fields)!=2 or not fields[0].strip().isdigit():
            return {"available":False,"reason":"Unusable GPU PID inventory: "+line,"processes":[]}
        processes.append({"pid":int(fields[0]),"name":fields[1].strip()})
    return {"available":True,"reason":None,"processes":processes}


def runtime_manifest(args,repo,execution_repo):
    result=subprocess.run(["git","rev-parse","HEAD"],cwd=repo,capture_output=True,text=True,check=True)
    runtime={"created":now(),"queue_pid":os.getpid(),"python":str(Path(args.python).resolve()),
        "git_commit":result.stdout.strip(),"platform":sys.platform,"python_version":sys.version,"gpu_index":args.gpu_index,
        "source_sha256":source_manifest(repo),"execution_root":str(execution_repo),
        "execution_source":"verified_snapshot","environment":{
            "CUDA_VISIBLE_DEVICES":str(args.gpu_index),"JAX_PLATFORMS":"cuda",
            "JAX_ENABLE_X64":"1",
            "JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS":"true",
            "XLA_PYTHON_CLIENT_PREALLOCATE":"false","PYTHONUNBUFFERED":"1",
            "PYTHONPATH":str(execution_repo/"src"),
            "KERNELPACK_CUDSS_LIBRARY":str(execution_repo/"src/kernelpack/sparse/_cudss_ffi.so")},
        "exclusive_ack":None}
    runtime["queue_environment_packages"]={}
    for package in ("jax","jaxlib","nvidia-cudss","nvidia-cudss-cu13","nvidia-cuda-runtime",
            "nvidia-cuda-runtime-cu13","nvidia-cublas","nvidia-cusolver"):
        try:
            runtime["queue_environment_packages"][package]=metadata.version(package)
        except metadata.PackageNotFoundError:
            runtime["queue_environment_packages"][package]=None
    try:
        gpu=subprocess.run(["nvidia-smi","-i",str(args.gpu_index),
            "--query-gpu=name,uuid,driver_version,memory.total","--format=csv,noheader"],
            capture_output=True,text=True,timeout=10,check=False)
        runtime["gpu_metadata"]={"returncode":gpu.returncode,"stdout":gpu.stdout.strip(),"stderr":gpu.stderr.strip()}
    except (OSError,subprocess.TimeoutExpired) as error:
        runtime["gpu_metadata"]={"error":str(error)}
    if args.exclusive_ack:
        ack=json.loads(args.exclusive_ack.read_text(encoding="utf-8"))
        if ack.get("gpu_exclusive") is not True or not ack.get("reason"):
            raise ValueError("Exclusive acknowledgement requires gpu_exclusive=true and a reason")
        runtime["exclusive_ack"]={"path":str(args.exclusive_ack.resolve()),"sha256":digest(args.exclusive_ack),"record":ack}
    return runtime


class Journal:
    def __init__(self,root):
        self.root=root
        self.state={"queue_pid":os.getpid(),"status":"starting","updated":now()}

    def event(self,event,**fields):
        record={"timestamp":now(),"event":event,**fields}
        with (self.root/"journal.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(record,allow_nan=False)+"\n")
            stream.flush()
            os.fsync(stream.fileno())
        message=f"{record['timestamp']} {event} "+" ".join(f"{key}={value}" for key,value in fields.items() if key!="gpu_inventory")
        with (self.root/"progress.log").open("a",encoding="utf-8") as stream:
            stream.write(message+"\n")
            stream.flush()
        print(message,flush=True)

    def update(self,**fields):
        self.state.update(fields)
        self.state["updated"]=now()
        write_atomic(self.root/"status.json",self.state)


def validate_complete(folder,repetitions,cell,plan,source_sha256):
    """Accept only the four-model, fully checked, immutable matched experiment.

    This is stdlib-only. No model evaluation, array decode, GPU initialization,
    or result modification happens during acceptance or one-shot reporting.
    """
    if (plan.get("schema_version")!=SCHEMA or plan.get("models")!=list(MODELS)
            or any(plan.get(key)!=value for key,value in (("training_steps",60),("validation_steps",35),
                ("repetitions",11),("spatial_batch_size",32),("cudss_reordering","amd")))
            or repetitions!=11 or plan.get("adopt_cells")):
        raise RuntimeError("This comparison requires the fresh full60/35/11/native32 AMD protocol")
    results=json.loads((folder/"results.json").read_text(encoding="utf-8"))
    selected=json.loads((folder/"selection.json").read_text(encoding="utf-8"))
    protocol=json.loads((folder/"protocol.json").read_text(encoding="utf-8"))
    if selected.get("schema_version")!=SCHEMA or results.get("schema_version")!=SCHEMA:
        raise RuntimeError("Legacy or unknown comparison schema; no old cell may be adopted")
    if set(results.get("models",{}))!=set(MODELS) or set(selected.get("models",{}))!=set(MODELS):
        raise RuntimeError("Exactly the four matched dimension-safe models are required")
    if results.get("dataset")!=cell["case"] or results.get("N")!=cell["N"]:
        raise RuntimeError("Completed result does not match its planned dataset and training size")
    if protocol.get("case")!=cell["case"] or protocol.get("N")!=cell["N"]:
        raise RuntimeError("Completed protocol does not match its planned dataset and training size")
    if selected.get("dataset")!=cell["case"] or selected.get("N")!=cell["N"] or selected.get("policy")!=plan["policy"]:
        raise RuntimeError("Completed selection does not match the planned cell/policy")
    if protocol.get('cudss_reordering')!=plan['cudss_reordering']:
        raise RuntimeError('Completed factor ordering differs from the frozen campaign plan')
    for key,value in (("schema_version",SCHEMA),("covariance_family",COVARIANCE),("model_names",list(MODELS)),
            ("operator_evaluation","lagrange"),("initial_support_density",.25),("initial_nugget",1e-6),
            ("reference_density",1.),("validation_start_policy","common_budget"),
            ("timed_native_query_batch_size",32),("timed_query_count",100),("test_targets_loaded",False)):
        if protocol.get(key)!=value:
            raise RuntimeError("Incorrect or unrecorded matched experiment contract: "+key)
    if protocol.get("dtype")!="float64" or "cuda" not in protocol.get("gpu","").lower():
        raise RuntimeError("The experiment did not certify CUDA float64 execution")
    if protocol.get("cuda_graphs") is not True:
        raise RuntimeError("CUDA graphs were not enabled")
    if protocol.get("source_sha256")!=source_sha256:
        raise RuntimeError("Completed cell source/native fingerprint differs from the complete frozen manifest")
    if protocol.get("native_library",{}).get("sha256")!=source_sha256.get("src/kernelpack/sparse/_cudss_ffi.so"):
        raise RuntimeError("The recorded loaded native library differs from the frozen binary")
    if results.get("selection_sha256")!=digest(folder/"selection.json"):
        raise RuntimeError("Frozen selection hash does not match results")
    if selected.get("protocol_sha256")!=digest(folder/"protocol.json"):
        raise RuntimeError("Frozen selection protocol hash does not match")
    if selected.get("shared_geometry_sha256")!=digest(folder/"shared_geometry.npz"):
        raise RuntimeError("Frozen common training geometry archive is missing or changed")
    if results.get("shared_geometry_sha256")!=selected["shared_geometry_sha256"]:
        raise RuntimeError("Result does not refer to the frozen common geometry")
    if selected.get("test_targets_loaded") is not False or results.get("test_targets_used_for_selection") is not False:
        raise RuntimeError("Train/validation/test separation was not certified")
    groups=[]
    for key,count in (("training_ids",cell["N"]),("validation_ids",100),("test_ids",100)):
        ids=protocol.get(key,[])
        if len(ids)!=count or len(set(ids))!=count or any(type(value) is not int or value<0 for value in ids):
            raise RuntimeError("Invalid complete ordered input split: "+key)
        groups.append(set(ids))
    if any(groups[i]&groups[j] for i in range(3) for j in range(i)):
        raise RuntimeError("Training/validation/test logical IDs overlap")
    physical_test=results.get("physical_test_ids",[])
    if (results.get("test_ids")!=protocol["test_ids"] or len(physical_test)!=100
            or len(set(physical_test))!=100):
        raise RuntimeError("Completed prediction test IDs do not match the complete frozen split")
    split=local_path(protocol.get("split_file",""))
    if not split.is_file() or digest(split)!=protocol.get("split_sha256"):
        raise RuntimeError("The frozen split file SHA256 is absent or has changed")
    inputs=protocol.get("source_files",{})
    if not inputs:
        raise RuntimeError("Input file SHA256 fingerprints are absent")
    for name,checksum in inputs.items():
        path=local_path(name)
        if not path.is_file() or digest(path)!=checksum:
            raise RuntimeError("An immutable input file is absent or changed: "+name)
    for key in ("selection_csv","baseline_csv"):
        path=local_path(protocol.get(key,""))
        if not path.is_file() or digest(path)!=protocol.get(key+"_sha256"):
            raise RuntimeError("Archived reference metadata is absent or changed: "+key)
    if (results.get('shared_spatial_preparation_count')!=1
            or not math.isfinite(results.get('shared_spatial_preparation_seconds',math.nan))
            or results['shared_spatial_preparation_seconds']<=0):
        raise RuntimeError('The single shared spatial preparation was not certified')
    metrics={}
    for name in MODELS:
        model=results["models"][name]
        choice=selected["models"][name]
        dimension=0 if name.endswith("no_poly") else 15
        fixed=name.startswith("fixed_")
        if choice.get("mean_dimension")!=dimension or model.get("mean_dimension")!=dimension:
            raise RuntimeError("Incorrect polynomial configuration for "+name)
        columns=choice.get("polynomial_columns")
        if (type(columns) is not int or (dimension==0 and columns!=0)
                or (dimension==15 and not 2<=columns<=16)):
            raise RuntimeError("Incorrect realized polynomial columns for "+name)
        for key in ("mean_dimension","polynomial_columns","mean_basis_sha256","covariance_family","geometry_sha256"):
            if model.get(key)!=choice.get(key):
                raise RuntimeError("Result differs from frozen model selection: "+name+"/"+key)
        if choice.get("covariance_family")!=COVARIANCE:
            raise RuntimeError("Incorrect or dimension-unsafe covariance for "+name)
        if model.get("selected_model_sha256")!=canonical_digest(choice):
            raise RuntimeError("The per-model immutable selection fingerprint differs: "+name)
        if choice.get("geometry_sha256")!=digest(folder/("selected_geometry_"+name+".npz")):
            raise RuntimeError("Frozen model geometry archive is missing or changed: "+name)
        if (not isinstance(choice.get("mean_basis_sha256"),str)
                or len(choice["mean_basis_sha256"])!=64):
            raise RuntimeError("The polynomial basis fingerprint is absent: "+name)
        config=choice.get("config",{})
        for key in ("training_steps","validation_steps"):
            if config.get(key)!=plan[key]:
                raise RuntimeError("Completed optimizer settings differ from the campaign plan: "+name+"/"+key)
        if config.get("cudss_reordering")!=plan["cudss_reordering"]:
            raise RuntimeError("Selected factor ordering differs from the frozen plan: "+name)
        expected_scaling="taylor_green_ray" if cell["case"] in ("taylor_green","taylor_green_time") else "none"
        if config.get("response_scaling")!=expected_scaling:
            raise RuntimeError("The explicit response transformation is unrecorded or inconsistent: "+name)
        if config.get("mean_dimension")!=dimension:
            raise RuntimeError("Optimizer mean differs from the claimed polynomial setting: "+name)
        if choice.get("selection_kind")!=("fixed_geometry_validation" if fixed else "trained_geometry_validation"):
            raise RuntimeError("Incorrect fixed/trained selection procedure: "+name)
        if choice.get("training_steps_executed")!=(0 if fixed else plan["training_steps"]):
            raise RuntimeError("Actual training step count differs: "+name)
        histories=choice.get("training_histories",[])
        penalties=config.get("penalties",[])
        if fixed:
            if histories or choice.get("perpendicular_ratio")!=1 or any(choice.get("log_parallel_scales",[1])):
                raise RuntimeError("The fixed control trained covariance geometry: "+name)
        elif len(histories)!=len(penalties) or penalties!=[0.,1.] or any(len(history)!=plan["training_steps"] for history in histories):
            raise RuntimeError("Full GP optimization trajectory evidence is absent: "+name)
        polished=[row for row in choice.get("candidates",[]) if row.get("origin")=="validation_polish"]
        if (len(polished)<(4 if fixed else 8)
                or any(len(row.get("history") or [])!=plan["validation_steps"] for row in polished)):
            raise RuntimeError("Complete validation optimization trajectory evidence is absent: "+name)
        if config.get("validation_start_policy")!="common_budget":
            raise RuntimeError("Fixed/trainable validation starts were not matched: "+name)
        if model.get('native_factor_storage',{}).get('reordering')!=plan['cudss_reordering']:
            raise RuntimeError('Observed factor ordering differs from the frozen campaign plan: '+name)
        prediction_name=model.get("prediction_file")
        if not isinstance(prediction_name,str) or not prediction_name:
            raise RuntimeError("Missing exported prediction path for "+name)
        prediction_path=(folder/prediction_name).resolve()
        if not prediction_path.is_relative_to(folder.resolve()) or not prediction_path.is_file():
            raise RuntimeError("Exported prediction is missing or outside its cell: "+name)
        if model.get("prediction_sha256")!=digest(prediction_path):
            raise RuntimeError("Exported prediction hash differs: "+name)
        for array in ("prediction","relative_error"):
            header=npz_array_header(prediction_path,array)
            if header.get("shape",(0,))[0]!=100 or header.get("descr") not in ("<f8","=f8",">f8"):
                raise RuntimeError("Exported complete100 prediction/error is not float64: "+name+"/"+array)
        if not math.isfinite(model["test_relative_field_error"]) or model["test_relative_field_error"]<0:
            raise RuntimeError("Nonfinite or negative test metric")
        error=choice.get("validation_error",math.nan)
        if not math.isfinite(error) or error<0:
            raise RuntimeError("Nonfinite or negative frozen validation metric: "+name)
        nnz=model.get("native_factor_storage",{}).get("actual_nnz")
        if type(nnz) is not int or not cell["N"]<=nnz<=cell["N"]**2:
            raise RuntimeError("Actual native kernel nnz is absent or invalid: "+name)
        density=nnz/cell["N"]**2
        if not math.isclose(choice.get("training_kernel_density",math.nan),density,rel_tol=0.,abs_tol=1e-15):
            raise RuntimeError("Selected support differs from actual native kernel nnz: "+name)
        timings=model["warm_prediction_seconds"]
        if len(timings)!=repetitions or not all(math.isfinite(value) and value>0 for value in timings):
            raise RuntimeError("Missing synchronized warm timing measurements")
        if model.get("warm_prediction_median_seconds")!=statistics.median(timings):
            raise RuntimeError("Warm median differs from the synchronized timing samples: "+name)
        if (model.get("gpu_transfer_guard_passed") is not True
                or model.get("graph_replays_during_timing",0)<4*repetitions
                or model.get("timed_native_query_batch_size")!=32 or model.get("timed_query_count")!=100):
            raise RuntimeError("GPU transfer/graph replay checks are not certified")
        checks=model.get('accuracy_checks',{})
        if (checks.get('passed') is not True or checks.get('shared_spatial_map_reused') is not True
                or checks.get('native_query_batch_transfer_guard_passed') is not True
                or checks.get('native_query_batch_size')!=32
                or checks.get('timed_native_query_batch_size')!=32
                or checks.get('timed_query_count')!=100):
            raise RuntimeError('Full-field, native query-batch and shared-map checks were not certified')
        for gap_name in ('validation_field_relative_gap','frozen_selection_field_relative_gap','reversed_query_relative_gap',
                     'native_query_batch_relative_gap','guarded_batch_relative_gap'):
            gap=checks.get(gap_name,math.nan)
            if not math.isfinite(gap) or not 0<=gap<=1e-8:
                raise RuntimeError('A full-field or native query-batch accuracy check failed: '+gap_name)
        compiled_error=checks.get("compiled_validation_error",math.nan)
        if not math.isfinite(compiled_error) or abs(compiled_error-error)>1e-8*(1+error):
            raise RuntimeError("Checked complete validation fields differ from the frozen selection: "+name)
        if model.get("shared_spatial_preparation_seconds")!=results["shared_spatial_preparation_seconds"]:
            raise RuntimeError("Model does not record the single common spatial preparation: "+name)
        for key in ("operator_setup_seconds","preparation_seconds","first_prediction_compile_capture_seconds"):
            value=model.get(key,math.nan)
            if not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                raise RuntimeError("Positive separately measured cold/preparation timing is missing: "+name+"/"+key)
        for group,key in (("native_factor_storage","factor_nnz"),("native_factor_storage","permanent_device_bytes"),
                ("counted_storage","visible_model_array_bytes")):
            value=model.get(group,{}).get(key)
            if type(value) is not int or value<=0:
                raise RuntimeError("Measured native/visible storage accounting is absent: "+name+"/"+key)
        metrics[name]={"validation_error":error,"test_error":model["test_relative_field_error"],
            "density":density,"warm_100query_median_seconds":statistics.median(timings),
            "native_factor_nnz":model["native_factor_storage"].get("factor_nnz"),
            "native_permanent_device_bytes":model["native_factor_storage"].get("permanent_device_bytes"),
            "visible_model_array_bytes":model.get("counted_storage",{}).get("visible_model_array_bytes"),
            "operator_setup_seconds":model.get("operator_setup_seconds"),
            "first_prediction_compile_capture_seconds":model.get("first_prediction_compile_capture_seconds"),
            "training_validation_seconds":choice.get("training_validation_seconds"),
            "prediction_sha256":model["prediction_sha256"]}
    for suffix in ("no_poly","affine"):
        left,right=(selected["models"][kind+"_"+suffix] for kind in ("fixed","trainable"))
        if left["mean_basis_sha256"]!=right["mean_basis_sha256"] or left["polynomial_columns"]!=right["polynomial_columns"]:
            raise RuntimeError("The fixed and trained model do not use the exact same polynomial basis: "+suffix)
    return {"models":metrics,"results_sha256":digest(folder/"results.json"),
        "selection_sha256":results["selection_sha256"],"shared_geometry_sha256":selected["shared_geometry_sha256"]}


def collect_report(root,output):
    """Produce one finite scalar report, never a watcher or a numerical job."""
    manifest=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    plan=manifest["plan"]
    expected=manifest["runtime"]["source_sha256"]
    verify_snapshot(root/"source_snapshot",expected)
    queue=json.loads((root/"status.json").read_text(encoding="utf-8")) if (root/"status.json").is_file() else {}
    cells=[]
    comparisons=[]
    for cell in plan["cells"]:
        folder=root/"cells"/cell["key"]
        row={**cell,"status":"pending"}
        if (folder/"results.json").is_file():
            try:
                metrics=validate_complete(folder,plan["repetitions"],cell,plan,expected)
                row.update(status="complete",**metrics)
                models=metrics["models"]
                for label,reference,candidate in (
                        ("learning_no_poly","fixed_no_poly","trainable_no_poly"),
                        ("learning_affine","fixed_affine","trainable_affine"),
                        ("polynomial_fixed","fixed_no_poly","fixed_affine"),
                        ("polynomial_trainable","trainable_no_poly","trainable_affine")):
                    comparison={"cell":cell["key"],"comparison":label,"reference":reference,"candidate":candidate}
                    for metric in ("test_error","density","warm_100query_median_seconds",
                            "native_factor_nnz","native_permanent_device_bytes","visible_model_array_bytes"):
                        a,b=models[reference].get(metric),models[candidate].get(metric)
                        comparison[metric+"_candidate_over_reference"]=b/a if a is not None and b is not None and a>0 else None
                    comparisons.append(comparison)
            except (OSError,ValueError,TypeError,KeyError,RuntimeError,zipfile.BadZipFile) as error:
                row.update(status="invalid",reason=type(error).__name__+": "+str(error))
        elif folder.exists():
            current=(queue.get("current_cell") or {}).get("key")==cell["key"]
            row.update(status="running" if current and queue.get("status")=="running" else "partial")
        cells.append(row)
    output.mkdir(parents=True,exist_ok=True)
    counts={state:sum(row["status"]==state for row in cells) for state in ("complete","invalid","running","partial","pending")}
    report={"created":now(),"schema_version":SCHEMA,"campaign":str(root),"queue_status":queue.get("status"),
        "source_manifest_verified":True,"counts":counts,"cells":cells,"matched_comparisons":comparisons,
        "timing_note":"Warm times synchronize all100 physical queries using four native32 batches (last edge padded); compile/capture, operator setup, shared spatial preparation and training/validation are separate costs. Archived CPU first-batch times are not matched warm hardware speedups.",
        "storage_note":"Kernel density, native factor fill/analysis storage estimates and visible JAX arrays are separate measurements; none is a measured whole-process peak.",
        "monitor_started":False}
    write_atomic(output/"summary.json",report)
    table=[]
    for row in cells:
        if row["status"]=="complete":
            table.extend({"case":row["case"],"N":row["N"],"model":name,**row["models"][name]} for name in MODELS)
    with (output/"models.csv").open("w",encoding="utf-8",newline="") as stream:
        fields=["case","N","model"]+sorted(set(table[0])-{"case","N","model"}) if table else ["case","N","model"]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(table)
    with (output/"comparisons.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(comparisons[0]) if comparisons else ["cell","comparison","reference","candidate"])
        writer.writeheader();writer.writerows(comparisons)
    lines=["# Matched dimension-safe GPU comparison","",f"{counts['complete']}/{len(cells)} complete and verified; queue {queue.get('status','unknown')}.","",
        "Fixed and trainable kernels use the same dimension-safe covariance family, exact training geometry, polynomial basis within each pair, validation starts, spatial reconstruction and operator Lagrange formulation.","",
        "| Dataset | N | Model | Test error | Density | Warm100 queries (s) | Factor nnz | Visible arrays (bytes) |",
        "|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in table:
        lines.append(f"| {row['case']} | {row['N']} | {row['model']} | {row['test_error']:.6g} | {100*row['density']:.4g}% | {row['warm_100query_median_seconds']:.6g} | {row['native_factor_nnz']} | {row['visible_model_array_bytes']} |")
    lines.extend(["",report["timing_note"],"",report["storage_note"],""])
    for row in cells:
        if row["status"]=="invalid":
            lines.extend(["Invalid cell "+row["key"]+": "+row["reason"],""])
    (output/"report.md").write_text("\n".join(lines),encoding="utf-8")
    return {"counts":counts,"summary":str(output/"summary.json"),"report":str(output/"report.md"),"monitor_started":False}


def terminate(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL)
        process.wait(timeout=10)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--python",default=sys.executable)
    parser.add_argument("--launch",action="store_true")
    parser.add_argument("--report",action="store_true",help="Write one stdlib-only verified scalar report and exit; never starts a monitor or a GPU job")
    parser.add_argument("--report-output",type=Path)
    parser.add_argument("--detach",action="store_true",help="Run the supervisor persistently with its own launcher log")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--cases",nargs="+",choices=CASES,default=CASES)
    parser.add_argument("--sizes",nargs="+",type=int,choices=SIZES,default=SIZES)
    parser.add_argument("--data-root",type=Path,default=Path("/mnt/e/fluid_datasets"))
    parser.add_argument("--baseline-root",type=Path,default=Path("/mnt/e/kernelpack-matlab-validation/experiments/fastol"))
    parser.add_argument("--training-steps",type=int,default=60)
    parser.add_argument("--validation-steps",type=int,default=35)
    parser.add_argument("--repetitions",type=int,default=11)
    parser.add_argument("--spatial-batch-size",type=int,default=32)
    parser.add_argument("--policy",choices=("accuracy","compression"),default="accuracy")
    parser.add_argument("--cudss-reordering",choices=("amd",),default="amd")
    parser.add_argument("--gpu-index",type=int,default=0)
    parser.add_argument("--poll-seconds",type=float,default=15.)
    parser.add_argument("--exclusive-ack",type=Path)
    args=parser.parse_args()
    if args.report:
        if args.launch or args.detach or args.resume:
            parser.error("--report is a finite read-only collection mode, separate from launching or resuming")
        root=args.output_root.resolve()
        output=args.report_output.resolve() if args.report_output else root.with_name(root.name+"_report")
        if output==root or output.is_relative_to(root):
            parser.error("A scalar report must stay outside the frozen campaign directory")
        print(json.dumps(collect_report(root,output),indent=2))
        return
    if len(set(args.cases))!=len(args.cases) or len(set(args.sizes))!=len(args.sizes):
        parser.error("cases and sizes must not contain duplicates")
    repo=Path(__file__).resolve().parents[1]
    cells=[{"case":case,"N":n,"key":f"{case}_N{n}"} for n in args.sizes for case in args.cases]
    plan={"schema_version":SCHEMA,"models":list(MODELS),"covariance_family":COVARIANCE,
        "cells":cells,"policy":args.policy,"training_steps":args.training_steps,
        "validation_steps":args.validation_steps,"repetitions":args.repetitions,
        "cudss_reordering":args.cudss_reordering,
        "spatial_batch_size":args.spatial_batch_size,"data_root":str(args.data_root),"baseline_root":str(args.baseline_root),
        "adopt_cells":{}}
    if not args.launch:
        print(json.dumps({"launched":False,"plan":plan},indent=2))
        return
    if sys.platform!="linux":
        parser.error("Launch the GPU queue in Linux/WSL with the CUDA Python environment")
    if (args.training_steps,args.validation_steps,args.repetitions,args.spatial_batch_size)!=(60,35,11,32):
        parser.error("The matched campaign requires full60 training/35 validation/11 synchronized timings/native32 batches")
    if (not math.isfinite(args.poll_seconds) or args.poll_seconds<1 or args.repetitions<3
            or min(args.training_steps,args.validation_steps,args.spatial_batch_size)<1 or args.gpu_index<0):
        parser.error("steps/batch size must be positive, poll-seconds >=1, repetitions >=3, GPU index >=0")
    if args.detach:
        launcher=args.output_root.resolve().with_name(args.output_root.name+".launcher.log")
        launcher.parent.mkdir(parents=True,exist_ok=True)
        command=[args.python,str(Path(__file__).resolve()),*[value for value in sys.argv[1:] if value!="--detach"]]
        with launcher.open("a",encoding="utf-8") as stream:
            process=subprocess.Popen(command,cwd=repo,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({"dispatched":True,"queue_pid":process.pid,"launcher_log":str(launcher),
            "status_file":str(args.output_root.resolve()/"status.json"),"setup_confirmed":False}),flush=True)
        return
    root=args.output_root.resolve()
    root.mkdir(parents=True,exist_ok=args.resume)
    execution_repo=root/"source_snapshot"
    runtime=runtime_manifest(args,repo,execution_repo)
    if args.resume:
        previous=json.loads((root/"manifest.json").read_text(encoding="utf-8"))
        if previous["plan"]!=plan or previous["runtime"]["source_sha256"]!=runtime["source_sha256"]:
            raise RuntimeError("Resume requires identical protocol and source/native fingerprints; preserve this revision and start a new one after fixes")
    else:
        write_atomic(root/"manifest.json",{"plan":plan,"runtime":runtime})
        freeze_sources(repo,root,runtime["source_sha256"])
    verify_snapshot(execution_repo,runtime["source_sha256"])
    (root/"logs").mkdir(exist_ok=True)
    (root/"cells").mkdir(exist_ok=True)
    journal=Journal(root)
    lock_path=repo/"Outputs_and_results/.fastol_gpu_campaign.lock"
    lock_path.parent.mkdir(exist_ok=True)
    try:
        descriptor=os.open(lock_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError:
        message=f"GPU queue lock already exists: {lock_path}; inspect its PID before resolving a stale lock"
        journal.event("queue_stopped",reason=message)
        journal.update(status="stopped",reason=message)
        raise RuntimeError(message)
    with os.fdopen(descriptor,"w",encoding="utf-8") as stream:
        json.dump({"pid":os.getpid(),"root":str(root),"created":now()},stream)
    child=None
    completed=[]
    def stop_requested(signum,_frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop_requested)
    try:
        journal.event("queue_started",pid=os.getpid(),cell_count=len(cells),root=str(root))
        for cell_index,cell in enumerate(cells):
            verify_snapshot(execution_repo,runtime["source_sha256"])
            folder=root/"cells"/cell["key"]
            if folder.exists():
                if not args.resume:
                    raise RuntimeError(f"Output already exists: {folder}")
                metrics=validate_complete(folder,args.repetitions,cell,plan,runtime["source_sha256"])
                completed.append(cell["key"])
                journal.event("completed_cell_verified",cell=cell["key"],metrics=metrics,
                    artifact_directory=str(folder),adopted=False,exclusivity="original queue journal")
                continue
            inventory=gpu_inventory(args.gpu_index)
            if inventory["available"] and inventory["processes"]:
                raise RuntimeError("Another compute process is using the GPU: "+json.dumps(inventory))
            if not inventory["available"] and runtime["exclusive_ack"] is None:
                raise RuntimeError("GPU PID inventory is unavailable; require a recorded --exclusive-ack after the parent confirms exclusivity: "+json.dumps(inventory))
            command=[args.python,"-u",str(execution_repo/"scripts/fastol_gpu_campaign.py"),"--case",cell["case"],"--N",str(cell["N"]),
                "--output",str(folder),"--data-root",str(args.data_root),"--baseline-root",str(args.baseline_root),
                "--training-steps",str(args.training_steps),"--validation-steps",str(args.validation_steps),
                "--repetitions",str(args.repetitions),"--spatial-batch-size",str(args.spatial_batch_size),"--policy",args.policy,
                "--cudss-reordering",args.cudss_reordering]
            environment=os.environ.copy()
            environment.update(runtime["environment"])
            log_path=root/"logs"/(cell["key"]+".log")
            start=time.monotonic()
            with log_path.open("x",encoding="utf-8") as log:
                log.write(f"Started {now()}\n{shlex.join(command)}\n")
                log.flush()
                child=subprocess.Popen(command,cwd=execution_repo,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                journal.event("cell_started",cell=cell["key"],position=cell_index+1,total=len(cells),child_pid=child.pid,log=str(log_path),gpu_inventory=inventory)
                while child.poll() is None:
                    inventory=gpu_inventory(args.gpu_index)
                    outsiders=[item for item in inventory["processes"] if item["pid"]!=child.pid]
                    if inventory["available"] and outsiders:
                        terminate(child)
                        raise RuntimeError("GPU timing exclusivity was lost; cell preserved for review: "+json.dumps(outsiders))
                    if not inventory["available"] and runtime["exclusive_ack"] is None:
                        terminate(child)
                        raise RuntimeError("GPU inventory became unavailable during the cell")
                    journal.update(status="running",current_cell=cell,position=cell_index+1,total=len(cells),
                        completed=completed,child_pid=child.pid,elapsed_seconds=time.monotonic()-start,
                        log=str(log_path),gpu_inventory=inventory,
                        exclusivity="PID inventory" if inventory["available"] else "recorded parent acknowledgement; PID inventory unavailable")
                    journal.event("cell_running",cell=cell["key"],child_pid=child.pid,elapsed_seconds=round(time.monotonic()-start,1),gpu_inventory=inventory)
                    time.sleep(args.poll_seconds)
                code=child.returncode
            child=None
            if code:
                raise RuntimeError(f"Cell {cell['key']} exited {code}; inspect {log_path}. Queue stopped without changing the protocol")
            verify_snapshot(execution_repo,runtime["source_sha256"])
            metrics=validate_complete(folder,args.repetitions,cell,plan,runtime["source_sha256"])
            completed.append(cell["key"])
            journal.event("cell_completed",cell=cell["key"],elapsed_seconds=round(time.monotonic()-start,1),metrics=metrics)
            journal.update(status="between_cells",completed=completed,child_pid=None)
        journal.event("queue_completed",completed=len(completed))
        journal.update(status="complete",completed=completed,current_cell=None,child_pid=None)
    except BaseException as error:
        if child is not None:
            terminate(child)
        reason=type(error).__name__+": "+str(error)
        journal.event("queue_stopped",reason=reason,completed=completed)
        journal.update(status="stopped",reason=reason,completed=completed,child_pid=None)
        raise
    finally:
        lock_path.unlink()


if __name__=="__main__":
    main()
