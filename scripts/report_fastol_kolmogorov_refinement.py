"""Authenticate and reduce local refinement plus a matched spatial comparison.

All mathematical reductions use CUDA float64; NumPy is NPZ I/O through the
shared analyzer. CSV/JSON formatting and opaque file hashes are host I/O.
No reconstruction is regenerated, no interval is changed, and physical errors
are reported without an accuracy acceptance threshold.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback

os.environ.setdefault("JAX_PLATFORMS", "cuda")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import analyze_fastol_actual_cloud_refinement as analysis

OLD_H, NEW_H, ORDERS = (.26,.19,.14,.101), (.075,.05), (2,4,6)
PROTOCOL_SHA = "9de9a798fd196cb972c6d91733710acd8dbb16d1f5ea6313307e6cddc83e025f"


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def child(base, name):
    path = (base/name).resolve()
    if not path.is_relative_to(base):
        raise RuntimeError("Artifact path escapes producer directory: "+str(name))
    return path


def pin(files, label, path, expected=None):
    digest = analysis.sha(path)
    if expected is not None and digest != expected:
        raise RuntimeError("Artifact hash mismatch: "+label)
    files[label] = path
    return digest


def load_case(base, name, protocol_sha, completion, files, label):
    result_path = child(base,name+"/result.json")
    pin(files,label+"/result",result_path,completion["result_sha256"][name+"/result.json"])
    row = json.loads(result_path.read_text())
    if (row["case"] != name or row["protocol_sha256"] != protocol_sha
        or row["reference_npz_sha256"] != analysis.REFERENCE_SHA
        or not row["numerical_checks"]["numerical_checks_passed"]
        or not row["numerical_checks"]["full_requested_space_retained"]
        or not row["covariance_bound_check_passed"]):
        raise RuntimeError("Producer identity or numerical checks disagree: "+name)
    for suffix, path, key in (
        ("input",child(base,row["input_artifact"]),"input_artifact_sha256"),
        ("maps",child(base,name+"/spatial_diagnostics.npz"),"spatial_diagnostics_npz_sha256"),
        ("statistics",child(base,name+"/statistics.npz"),"statistics_npz_sha256")):
        pin(files,label+"/"+suffix,path,row[key])
    return row


def verify_extension(folder, study, files):
    completion_path, protocol_path = folder/"completion.json", folder/"protocol.json"
    pin(files,"extension/completion",completion_path)
    pin(files,"extension/protocol",protocol_path)
    completion, protocol = (json.loads(p.read_text()) for p in (completion_path,protocol_path))
    if (not completion["completed"] or completion["statistical_cells"] != 6
        or completion["reference_npz_sha256"] != analysis.REFERENCE_SHA
        or completion["protocol_sha256"] != canonical_sha(protocol)
        or completion["hashes_before"] != completion["hashes_after"]):
        raise RuntimeError("Extension completion/protocol is not authentic and complete")
    if (protocol["spacings"] != list(NEW_H) or protocol["orders"] != list(ORDERS)
        or protocol["original_protocol_sha256"] != PROTOCOL_SHA
        or protocol["matern_base_length_factor"] != 16. or protocol["matern_num_modes"] != 5
        or protocol["matern_gamma"] != 8/3 or protocol["allow_pseudoinverse"]
        or not protocol["compute_gradients"]):
        raise RuntimeError("Extension changed the declared production method")
    for key, expected in completion["hashes_before"].items():
        if key.startswith("current/"):
            path = child(folder/"source",key[8:])
        elif key.startswith("original_source/"):
            path = child(study/"study_source",key[16:])
        elif key == "reference":
            path = study/"increment_statistics/reference.npz"
        elif key == "original_protocol":
            path = study/"increment_statistics/protocol.json"
        else:
            raise RuntimeError("Unknown extension manifest key: "+key)
        pin(files,"extension/pin/"+key,path,expected)
    return completion, protocol


def module_source(repo, name):
    if name == "diagnose_fastol_two_stencils":
        return repo/"scripts/diagnose_fastol_two_stencils.py"
    if name != "kernelpack" and not name.startswith("kernelpack."):
        raise RuntimeError("Unknown producer module pin: "+name)
    stem = repo/"src"/name.replace(".","/")
    candidates = [stem.with_suffix(".py"),stem/"__init__.py"]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise RuntimeError("Cannot resolve producer module source: "+name)
    return found[0]


def verify_global(folder, study, repo, files):
    path = folder/"result.json"
    pin(files,"global/result",path)
    report = json.loads(path.read_text())
    if (not report["complete"] or report["precision"] != "float64"
        or not report["source_and_original_inputs_unchanged"]
        or report["hashes_before"] != report["hashes_after"]
        or report["orders"] != list(ORDERS) or report["all_original_pairs"] != 256
        or report["matern_mode_count"] != 5 or report["matern_gamma"] != 8/3
        or report["global_covariance_cholesky_count"] != 1
        or report["global_covariance_symmetry"] > 2e-12
        or report["global_cholesky_relative_residual"] > 1e-10
        or len(report["global_checks"]) != 3 or len(report["comparisons"]) != 3):
        raise RuntimeError("Matched global/local comparison is incomplete or inconsistent")
    for key, expected in report["hashes_before"].items():
        if key.startswith("original_source/"):
            source = child(study/"study_source",key[16:])
        elif key.startswith("current_source/"):
            source = module_source(repo,key[15:])
        elif key == "diagnostic_source":
            source = folder/"compare_fastol_global_local_matern.py"
        elif key in ("reference","inputs","original_protocol","original_source_manifest"):
            source = {"reference":study/"increment_statistics/reference.npz",
                "inputs":study/"increment_statistics/irregular3d_h0.101_inputs.npz",
                "original_protocol":study/"increment_statistics/protocol.json",
                "original_source_manifest":study/"study_source/source_manifest.json"}[key]
        elif key.startswith(("result_","map_","statistics_")):
            kind, number = key.split("_",1)
            filename = {"result":"result.json","map":"spatial_diagnostics.npz","statistics":"statistics.npz"}[kind]
            source = study/"increment_statistics"/f"irregular3d_h0.101_p{number}_fixed_physical_band"/filename
        else:
            raise RuntimeError("Unknown global producer manifest key: "+key)
        pin(files,"global/pin/"+key,source,expected)
    pin(files,"global/arrays",folder/"comparison_arrays.npz",report["arrays_sha256"])
    for order, checks in zip(ORDERS,report["global_checks"]):
        if (checks["polynomial_dimension"] != analysis.RANKS[order]
            or checks["whitened_rank_ratio"] <= 0
            or checks["full_raw_polynomial_reproduction"] > 1e-8
            or checks["transformed_polynomial_reproduction"] > 1e-8
            or checks["original_unanchored_backward_error"] > 1e-10
            or checks["anchored_saddle_backward_error"] > 1e-10
            or checks["nodal_identity_relative_error"] > 1e-9
            or checks["nodal_identity_maximum_error"] > 1e-8
            or checks["normalized_divergence"] > 1e-9):
            raise RuntimeError("Matched global numerical checks are not accepted")
    routes = report["independent_route_checks"]
    if (routes["node_feature_vs_saved_field_gap"]>1e-9
        or routes["half_angle_reference_identity_gap"]>1e-9
        or any(value>1e-9 for value in routes["feature_vs_nodal_prediction_gaps"])):
        raise RuntimeError("Matched global independent route checks failed")
    return report


@jax.jit
def reduce_matched(truth, prediction, mse, radius, interval):
    true = truth.reshape(32,8)
    pred, error = prediction.reshape(6,32,8), mse.reshape(6,32,8)
    mean = jnp.mean(pred,axis=2)
    worst = jnp.max(jnp.abs(pred/true[None]-1),axis=2)
    normalized = jnp.sum(error,axis=2)/jnp.sum(true,axis=1)[None]
    slopes = jnp.diff(jnp.log(mean),axis=1)/jnp.diff(jnp.log(radius))[None]
    return dict(mean_s2=mean,worst_relative_s2_error=worst,normalized_increment_mse=normalized,
        separation_log_slopes=slopes,
        maximum_amplitude_bound_violation=jnp.max(jnp.maximum(
            jnp.abs(jnp.sqrt(pred/true[None])-1)-jnp.sqrt(error/true[None]),0.)),
        original_interval_worst_relative_s2_error=jnp.max(jnp.where(interval[None],worst,-jnp.inf),axis=1),
        original_interval_worst_normalized_mse=jnp.max(jnp.where(interval[None,:,None],error/true[None],-jnp.inf),axis=(1,2)))


@jax.jit
def log_log_fit(x, y):
    """Equal-weight OLS with intercept along the complete final axis.

    Nonpositive data make a fit undefined; they are never discarded to select
    a different fitting window. The caller exports null in that event.
    """
    valid = jnp.all(x>0)&jnp.all(jnp.isfinite(x))&jnp.all((y>0)&jnp.isfinite(y),axis=-1)
    log_x = jnp.log(jnp.where(x>0,x,1.))
    log_y = jnp.log(jnp.where(y>0,y,1.))
    centered_x = log_x-jnp.mean(log_x)
    centered_y = log_y-jnp.mean(log_y,axis=-1,keepdims=True)
    return dict(exponent=jnp.sum(centered_y*centered_x,axis=-1)/jnp.sum(centered_x**2),defined=valid)


@jax.jit
def extension_checks(gradient, saved_divergence, fields, indices, weights, nodal_prediction,
                     field_jacobian, py, pz, energies, reference_pair_s2):
    divergence = jnp.einsum("qakca->qkc",gradient)
    normalized = jnp.sqrt(jnp.sum(divergence**2,axis=(1,2)))/jnp.maximum(
        jnp.sqrt(jnp.sum(gradient**2,axis=(1,2,3,4))),jnp.finfo(jnp.float64).tiny)
    reconstructed = jnp.einsum("qakc,qkdc->qad",weights,fields[indices],
                                precision=jax.lax.Precision.HIGHEST)
    relative = lambda a,b:jnp.linalg.norm(a-b)/jnp.maximum(jnp.linalg.norm(b),1e-300)
    return dict(maximum_normalized_weight_divergence=jnp.max(normalized),
        saved_weight_divergence_gap=jnp.max(jnp.abs(normalized-saved_divergence)),
        saved_nodal_prediction_gap=relative(reconstructed,nodal_prediction),
        feature_nodal_prediction_gap=relative(jnp.transpose(nodal_prediction,(0,2,1)),jnp.concatenate((py,pz))),
        scalar_covariance_identity_gap=jnp.max(jnp.abs(energies[:,1]-reference_pair_s2)),
        normalized_field_divergence=jnp.linalg.norm(jnp.einsum("qaad->qd",field_jacobian))
            /jnp.maximum(jnp.linalg.norm(field_jacobian),1.))


def check_extension_case(base,row,ref):
    maps = analysis.archive(base/row["case"]/"spatial_diagnostics.npz",("weights","weight_gradients",
        "stencil_indices","gradient_finite","derivative_defined","query_node_min_distance","divergence_weight_normalized"))
    saved = analysis.archive(base/row["case"]/"statistics.npz",("nodal_route_prediction","nodal_route_jacobian",
        "manufactured_prediction_y","manufactured_prediction_z","energies"))
    fields = analysis.archive(base/row["input_artifact"],("nodal_values",))["nodal_values"]
    if maps["weight_gradients"].shape != (512,3,analysis.WIDTHS[row["value_order"]],3,3):
        raise RuntimeError("Extension full weight-Jacobian dimensions changed")
    if any(value.dtype != jnp.float64 for value in (maps["weights"],maps["weight_gradients"],fields,
            saved["nodal_route_prediction"],saved["nodal_route_jacobian"],saved["energies"])):
        raise RuntimeError("Extension derivative/field precision changed")
    if saved["nodal_route_prediction"].shape!=(512,3,8) or saved["nodal_route_jacobian"].shape!=(512,3,3,8):
        raise RuntimeError("Extension must retain every endpoint and all eight original draws")
    analysis.require(jnp.all(maps["gradient_finite"])&jnp.all(maps["derivative_defined"])
        &jnp.all(maps["query_node_min_distance"]>1e-12)
        &jnp.all(jnp.isfinite(maps["weight_gradients"]))
        &jnp.all(jnp.isfinite(saved["nodal_route_jacobian"])),"Extension derivatives not finite/defined away from nodes")
    phase = (ref["query_z"]-ref["query_y"])@ref["wavevectors"].T
    pair_s2 = 8*(jnp.sin(.5*phase)**2@ref["weights"])
    checks = extension_checks(maps["weight_gradients"],maps["divergence_weight_normalized"],fields,
        maps["stencil_indices"],maps["weights"],saved["nodal_route_prediction"],saved["nodal_route_jacobian"],
        saved["manufactured_prediction_y"],saved["manufactured_prediction_z"],saved["energies"],pair_s2)
    for key,tolerance in (("maximum_normalized_weight_divergence",1e-9),("saved_weight_divergence_gap",1e-12),
        ("saved_nodal_prediction_gap",1e-9),("feature_nodal_prediction_gap",1e-9),
        ("scalar_covariance_identity_gap",1e-11),("normalized_field_divergence",1e-9)):
        analysis.require(jnp.isfinite(checks[key])&(checks[key]<=tolerance),"Independent extension check failed: "+key)
    if (row["manufactured_normalized_divergence"]>1e-9
        or row["manufactured_feature_vs_nodal_route_relative_gap"]>1e-9
        or row["scalar_prior_s2_identity_max_gap"]>1e-11):
        raise RuntimeError("Producer extension derivative/route checks failed")
    return analysis.export(checks)


def csv_table(path, rows):
    with path.open("x",newline="",encoding="utf-8") as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root",type=Path,required=True)
    parser.add_argument("--extension",type=Path,required=True)
    parser.add_argument("--global-comparison",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    study, extension, global_folder, output = (p.resolve() for p in
        (args.study_root,args.extension,args.global_comparison,args.output))
    if any(output.is_relative_to(base) for base in (study/"increment_statistics",study/"study_source",extension,global_folder)):
        raise ValueError("Report output must be outside immutable producer/input directories")
    output.mkdir(parents=True,exist_ok=False)
    (output/"cases").mkdir()
    files, before = {}, {}
    try:
        repo, old = Path(__file__).resolve().parents[1], study/"increment_statistics"
        pin(files,"collector_source",Path(__file__).resolve())
        pin(files,"analyzer_source",Path(analysis.__file__).resolve())
        pin(files,"original/manifest",study/"study_source/source_manifest.json",analysis.MANIFEST_SHA)
        manifest = json.loads(files["original/manifest"].read_text())
        for key, expected in manifest["source_sha256"].items():
            pin(files,"original/source/"+key,child(study/"study_source",key),expected)
        pin(files,"original/reference",old/"reference.npz",analysis.REFERENCE_SHA)
        pin(files,"original/protocol",old/"protocol.json")
        pin(files,"original/completion",old/"completion.json")
        original_protocol, original_completion = (json.loads(files["original/"+key].read_text())
                                                  for key in ("protocol","completion"))
        if (canonical_sha(original_protocol) != PROTOCOL_SHA or not original_completion["completed"]
            or original_completion["protocol_sha256"] != PROTOCOL_SHA
            or original_completion["reference_npz_sha256"] != analysis.REFERENCE_SHA):
            raise RuntimeError("Original study completion/protocol changed")
        extension_completion, extension_protocol = verify_extension(extension,study,files)
        global_report = verify_global(global_folder,study,repo,files)
        native, controls = [], []
        for base, spacings, completion, psha in ((old,OLD_H,original_completion,PROTOCOL_SHA),
            (extension,NEW_H,extension_completion,canonical_sha(extension_protocol))):
            for h in spacings:
                for order in ORDERS:
                    name = f"irregular3d_h{h:g}_p{order}_ram_local_tau16"
                    row = load_case(base,name,psha,completion,files,"native/"+name)
                    if (row["spacing"] != h or row["value_order"] != order or row["policy"] != "ram_local_tau16"
                        or row["map_metadata"]["matern_base_length_factor"] != 16.
                        or row["map_metadata"]["matern_num_modes"] != 5
                        or row["map_metadata"]["matern_gamma"] != 8/3):
                        raise RuntimeError("Native convergence cell changes production method")
                    native.append((base,row))
        for order in ORDERS:
            name = f"irregular3d_h0.101_p{order}_fixed_physical_band"
            row = load_case(old,name,PROTOCOL_SHA,original_completion,files,"matched_control/"+name)
            if (row["policy"]!="fixed_physical_band" or row["value_order"]!=order
                or row["polynomial_degree"]!=order-1
                or row["fixed_physical_matern_base_length"]!=global_report["physical_matern_base_length"]):
                raise RuntimeError("Matched control changes physical kernel or polynomial space")
            controls.append(row)
        before = {key:analysis.sha(path) for key,path in files.items()}
        analysis.write(output/"hashes_before.json",before)
        for module in (Path(__file__).resolve(),Path(analysis.__file__).resolve()):
            shutil.copyfile(module,output/module.name)
        devices = jax.devices()
        if not devices or any(d.platform != "gpu" or "nvidia" not in d.device_kind.lower() for d in devices):
            raise RuntimeError("Collector requires NVIDIA CUDA; no CPU numerical fallback")
        jax.config.update("jax_default_device",devices[0])
        ref = analysis.archive(old/"reference.npz",("radius","query_y","query_z","certified_interval",
            "reference_directional_s2","wavevectors","weights"))
        analysis.require((ref["radius"][-1]==.8)&jnp.asarray(ref["query_y"].shape==(256,3)),"Original query grid changed")
        pair_mean = jnp.mean(ref["reference_directional_s2"],axis=1)
        reference = analysis.export(dict(radius=ref["radius"],certified_interval=ref["certified_interval"],
            pair_mean_s2=pair_mean,two_thirds_guide=pair_mean[-1]*(ref["radius"]/ref["radius"][-1])**(2/3)))
        local, native_bins, local_table = [], [], []
        for base,row in native:
            summary, bins = analysis.analyze(base,row,ref)
            extra = check_extension_case(base,row,ref) if base==extension else None
            analysis.write(output/"cases"/(row["case"]+".json"),dict(summary=summary,per_bin=bins,independent_extension_checks=extra))
            last = summary["fixed_radius_0p8"]
            outcome = {key:value for key,value in row.items() if key.startswith("resolved_interval_")}
            all_radius_outcome = {key:value for key,value in row.items() if key.startswith("maximum_relative_")}
            item = dict(order=row["value_order"],h=row["spacing"],nodes=row["nodes"],case=row["case"],
                mean_s2=[b["reconstructed_s2"] for b in bins],
                worst_relative_s2_error_r08=last["worst_direction_relative_s2_error"],
                normalized_increment_mse_r08=last["normalized_increment_mse"],
                worst_relative_s2_error=[b["worst_direction_relative_s2_error"] for b in bins],
                normalized_increment_mse=[b["normalized_increment_mse"] for b in bins],
                separation_log_slopes=[b["reconstructed_forward_log_slope"] for b in bins[:-1]],
                separation_fit_last4=summary["fixed_last_four_radius_bins"],stability=summary["stability"],
                original_full_window_outcome=outcome,numerical_checks=row["numerical_checks"],
                original_all_radius_aggregate_outcome=all_radius_outcome,
                independent_extension_checks=extra)
            local.append(item)
            native_bins.extend(bins)
            local_table.append(dict(order=item["order"],h=item["h"],nodes=item["nodes"],
                worst_relative_s2_error_r08=item["worst_relative_s2_error_r08"],
                normalized_increment_mse_r08=item["normalized_increment_mse_r08"],
                maximum_lebesgue=item["stability"]["maximum_block_lebesgue_constant"],
                maximum_query_radius=item["stability"]["maximum_query_radius"],
                last4_separation_slope=item["separation_fit_last4"]["fitted_reconstructed_log_slope"],
                original_window_worst_relative_s2_error=outcome["resolved_interval_max_relative_s2_error"],
                original_window_sufficient_bound_met=outcome["resolved_interval_bound_met"]))
            print(json.dumps(dict(stage="native_case_collected",case=row["case"])),flush=True)
        control_bins = {}
        for row in controls:
            summary, bins = analysis.analyze(old,row,ref)
            control_bins[row["value_order"]] = bins
            analysis.write(output/"cases"/(row["case"]+".json"),dict(summary=summary,per_bin=bins))
        arrays = analysis.archive(global_folder/"comparison_arrays.npz",("nodes","queries","radius","certified_interval",
            "reference_s2","increment_variance","increment_mse"))
        analysis.require(jnp.array_equal(arrays["queries"],jnp.concatenate((ref["query_y"],ref["query_z"])))
            &jnp.array_equal(arrays["radius"],ref["radius"])
            &jnp.array_equal(arrays["certified_interval"],ref["certified_interval"]),"Matched physical query grid changed")
        for key, shape in (("reference_s2",(256,)),("increment_variance",(6,256)),("increment_mse",(6,256))):
            if arrays[key].shape != shape or arrays[key].dtype != jnp.float64:
                raise RuntimeError("Matched statistics dimensions/precision changed")
            analysis.require(jnp.all(jnp.isfinite(arrays[key]))&jnp.all(arrays[key]>=0),"Invalid matched covariance energy")
        analysis.require(jnp.all(arrays["reference_s2"]>0),"Zero reference increment variance")
        original_nodes = analysis.archive(old/controls[0]["input_artifact"],("nodes",))["nodes"]
        analysis.require(jnp.array_equal(original_nodes,arrays["nodes"]),"Matched global cloud differs from original cloud")
        reduced = reduce_matched(arrays["reference_s2"],arrays["increment_variance"],arrays["increment_mse"],
            ref["radius"],ref["certified_interval"])
        analysis.require(reduced["maximum_amplitude_bound_violation"]<=1e-10,"Matched exact increment-amplitude bound failed")
        analysis.require(jnp.all(jnp.abs(jnp.mean(arrays["reference_s2"].reshape(32,8),axis=1)-pair_mean)<=1e-10),
                         "Matched reference differs from original direction-mean structure function")
        last4_models = analysis.export(log_log_fit(ref["radius"][-4:],reduced["mean_s2"][:,-4:]))
        last4_reference = analysis.export(log_log_fit(ref["radius"][-4:],pair_mean[-4:]))
        reference_exponent = last4_reference["exponent"] if last4_reference["defined"] else None
        matched, matched_bins, matched_table = [], [], []
        for i,order in enumerate(ORDERS):
            source = global_report["comparisons"][i]
            if source["order"] != order:
                raise RuntimeError("Matched model order changed")
            for label,index in (("global",i),("local",i+3)):
                analysis.require(jnp.isclose(reduced["original_interval_worst_relative_s2_error"][index],
                    source[label]["maximum_original_interval_relative_s2_error"],rtol=1e-9,atol=1e-12),
                    "Matched original-window S2 reduction disagrees with producer")
                analysis.require(jnp.isclose(reduced["original_interval_worst_normalized_mse"][index],
                    source[label]["maximum_original_interval_relative_increment_mse"],rtol=1e-9,atol=1e-12),
                    "Matched original-window increment-MSE reduction disagrees with producer")
            saved_control = jnp.asarray([b["reconstructed_s2"] for b in control_bins[order]],dtype=jnp.float64)
            analysis.require(jnp.all(jnp.isclose(saved_control,reduced["mean_s2"][i+3],rtol=1e-9,atol=1e-12)),
                             "Matched local curve differs from authenticated saved control")
            item = analysis.export(dict(order=order,h=.101,nodes=1770,
                local_mean_s2=reduced["mean_s2"][i+3],global_mean_s2=reduced["mean_s2"][i],
                local_worst_relative_s2_error=reduced["worst_relative_s2_error"][i+3],
                global_worst_relative_s2_error=reduced["worst_relative_s2_error"][i],
                local_normalized_increment_mse=reduced["normalized_increment_mse"][i+3],
                global_normalized_increment_mse=reduced["normalized_increment_mse"][i],
                local_separation_log_slopes=reduced["separation_log_slopes"][i+3],
                global_separation_log_slopes=reduced["separation_log_slopes"][i],
                separation_fit_last4=dict(radius_lower=ref["radius"][-4],radius_upper=ref["radius"][-1],count=4,
                    fitted_reference_log_slope=reference_exponent,
                    fitted_local_log_slope=last4_models["exponent"][i+3] if last4_models["defined"][i+3] else None,
                    fitted_global_log_slope=last4_models["exponent"][i] if last4_models["defined"][i] else None,
                    qualification="S2 versus separation at the same fixed last four original radius bins; not convergence versus h."),
                original_full_window_outcome=source,global_numerical_checks=global_report["global_checks"][i],
                local_numerical_checks=controls[i]["numerical_checks"]))
            matched.append(item)
            matched_table.append(dict(order=order,h=.101,nodes=1770,
                local_worst_relative_s2_error_r08=item["local_worst_relative_s2_error"][-1],
                global_worst_relative_s2_error_r08=item["global_worst_relative_s2_error"][-1],
                local_normalized_increment_mse_r08=item["local_normalized_increment_mse"][-1],
                global_normalized_increment_mse_r08=item["global_normalized_increment_mse"][-1],
                reference_last4_separation_slope=item["separation_fit_last4"]["fitted_reference_log_slope"],
                local_last4_separation_slope=item["separation_fit_last4"]["fitted_local_log_slope"],
                global_last4_separation_slope=item["separation_fit_last4"]["fitted_global_log_slope"],
                local_original_window_error=source["local"]["maximum_original_interval_relative_s2_error"],
                global_original_window_error=source["global"]["maximum_original_interval_relative_s2_error"]))
            for b in range(32):
                matched_bins.append(dict(order=order,radius_bin=b,radius=reference["radius"][b],
                    in_original_reference_certified_interval=reference["certified_interval"][b],
                    local_mean_s2=item["local_mean_s2"][b],global_mean_s2=item["global_mean_s2"][b],
                    local_worst_relative_s2_error=item["local_worst_relative_s2_error"][b],
                    global_worst_relative_s2_error=item["global_worst_relative_s2_error"][b],
                    local_normalized_increment_mse=item["local_normalized_increment_mse"][b],
                    global_normalized_increment_mse=item["global_normalized_increment_mse"][b],
                    local_separation_log_slope=item["local_separation_log_slopes"][b] if b<31 else None,
                    global_separation_log_slope=item["global_separation_log_slopes"][b] if b<31 else None))
        finest = [row for row in local if row["order"]==6 and row["h"]==.05]
        if len(local)!=18 or len(matched)!=3 or len(finest)!=1:
            raise RuntimeError("Incomplete final report dimensions")
        h = jnp.asarray(OLD_H+NEW_H,dtype=jnp.float64)
        # Every predeclared h level enters each fit, in the same order. These
        # descriptive exponents do not assert an asymptotic convergence rate.
        errors = jnp.asarray([[next(row["worst_relative_s2_error_r08"] for row in local
            if row["order"]==order and row["h"]==spacing) for spacing in OLD_H+NEW_H] for order in ORDERS],dtype=jnp.float64)
        mses = jnp.asarray([[next(row["normalized_increment_mse_r08"] for row in local
            if row["order"]==order and row["h"]==spacing) for spacing in OLD_H+NEW_H] for order in ORDERS],dtype=jnp.float64)
        error_fit, mse_fit = analysis.export((log_log_fit(h,errors),log_log_fit(h,mses)))
        convergence_fits = [dict(order=order,h_range=[.05,.26],count=6,h_levels=list(OLD_H+NEW_H),
            separation=.8,s2_error_exponent=error_fit["exponent"][i] if error_fit["defined"][i] else None,
            increment_mse_exponent=mse_fit["exponent"][i] if mse_fit["defined"][i] else None,
            qualification="Effective finite-range slopes: equal-weight log-log least squares with intercept over all six fixed nominal h levels; no asymptotic-order claim.")
            for i,order in enumerate(ORDERS)]
        guides = analysis.export(dict(h=h,two_thirds_error=jnp.asarray(
            finest[0]["worst_relative_s2_error_r08"],dtype=jnp.float64)*(h/.05)**(2/3)))
        after = {key:analysis.sha(path) for key,path in files.items()}
        if before != after:
            raise RuntimeError("Source or authenticated artifact changed during collection")
        csv_table(output/"local_summary.csv",local_table)
        csv_table(output/"local_per_bin.csv",native_bins)
        csv_table(output/"matched_summary.csv",matched_table)
        csv_table(output/"matched_per_bin.csv",matched_bins)
        csv_table(output/"local_convergence_fits.csv",[dict(order=row["order"],h_min=.05,h_max=.26,count=6,
            separation=.8,s2_error_exponent=row["s2_error_exponent"],increment_mse_exponent=row["increment_mse_exponent"])
            for row in convergence_fits])
        analysis.write(output/"report.json",dict(complete=True,device=str(devices[0]),dtype="float64",
            jax_version=jax.__version__,schema="fastol_kolmogorov_refinement_report_v1",
            reference=reference,local=local,guides=guides,matched=matched,convergence_fits=convergence_fits,
            reference_npz_sha256=analysis.REFERENCE_SHA,
            native_kernel=dict(modes=5,gamma=8/3,base_length_factor=16.,length_policy="stencil-relative"),
            matched_kernel=dict(modes=5,gamma=8/3,length_policy="one fixed physical band",
                base_length=global_report["physical_matern_base_length"],
                inverse_lengths=global_report["physical_inverse_lengths"],
                relative_mixture_weights=global_report["physical_relative_mixture_weights"]),
            native_case_count=18,matched_order_count=3,hashes_before=before,hashes_after=after,
            csv_sha256={name:analysis.sha(output/name) for name in
                ("local_summary.csv","local_per_bin.csv","matched_summary.csv","matched_per_bin.csv","local_convergence_fits.csv")},
            qualifications=dict(native="Production five-mode stencil-relative tau16, all six actual global clouds; requested orders identify polynomial degrees, not measured rough-field convergence orders.",
                matched="Separate matched spatial locality comparison: finest original cloud, same fixed physical five-mode kernel and polynomial space. These are not tau16 convergence cells.",
                windows="All original 32 radii and certified window retained; fixed r=0.8 and last-four separation fits remain explicitly identified. No window is selected from reconstruction accuracy.",
                guides="r^(2/3) guide anchored to reference at r=0.8; h^(2/3) guide anchored to finest native order-six worst-direction error. Neither is a fitted asymptotic-order claim.",
                numerical="CUDA float64 reductions only. All producer accuracy/identity checks are preserved; no physical-error gate, new solve, timing or scalability claim.")))
        print(json.dumps(dict(stage="report_complete",native_cases=18,matched_orders=3)),flush=True)
    except BaseException as error:
        analysis.write(output/"failure.json",dict(error_type=type(error).__name__,error=str(error),
            traceback=traceback.format_exc(),hashes_before=before,
            hashes_after={key:analysis.sha(path) for key,path in files.items() if path.exists()}))
        raise


if __name__ == "__main__":
    main()
