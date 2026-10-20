"""Verify saved spatial-study evidence and prepare figures using CUDA float64.

This program never imports the numerical study or recomputes a local solve.
NumPy is restricted to loading saved arrays; scientific reductions, rates,
errors and reference curves are evaluated on CUDA. Run only after the study
has released the GPU. MATLAB subsequently renders the exported plot arrays.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ["JAX_PLATFORMS"] = "cuda"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp


NODAL_ORACLE_FLAG = "independent_nodal_requested_polynomial_reproduction"
REVIEWED_ORACLE_RECHECK_SHA256 = "18b06d40fdecb15c9e113329bc0ae84a15566bbe3df367ac5304308e3d62ba96"
REVIEWED_NODAL_RESIDUAL_SOURCES = {
    "scripts/fastol_spatial_study_core.py":"6b24d072d321a6064c4a7eb80b3b1fc826eb30b0a3920344b8ba8b32a20db284",
    "scripts/fastol_spatial_properties.py":"3d9a426c4069ac655b908ef8a35b290c89d5272e47fd3a239368e80f2e6a7e80",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def exported(value):
    """Explicit artifact serialization, including JSON-safe nonfinite values."""
    if isinstance(value, jax.Array):
        return exported(jax.device_get(value).tolist())
    if isinstance(value, dict):
        return {str(k): exported(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [exported(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def scalar(value):
    return jax.device_get(value).item()


def same(a, b, *, rtol=2e-11, atol=2e-14):
    a, b = jnp.asarray(a, dtype=jnp.float64), jnp.asarray(b, dtype=jnp.float64)
    return bool(scalar(jnp.all(jnp.isfinite(a) & jnp.isfinite(b)
        & (jnp.abs(a-b) <= atol + rtol*jnp.abs(b)))))


def verify_sources(root, pins):
    failures = []
    for name, expected in pins.items():
        relative = Path(name.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            failures.append(f"Unscoped source pin: {name}")
            continue
        path = root/relative
        if not path.is_file() or digest(path) != expected:
            failures.append(f"Source hash mismatch: {relative.as_posix()}")
    manifest_path = root/"source_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest is None:
        failures.append("Frozen source manifest is missing")
    else:
        for name, expected in manifest["source_sha256"].items():
            relative = Path(name.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts:
                failures.append(f"Unscoped manifest source pin: {name}")
            elif not (root/relative).is_file() or digest(root/relative) != expected:
                failures.append(f"Frozen manifest mismatch: {name}")
        for name, expected in pins.items():
            if manifest["source_sha256"].get(name) != expected:
                failures.append(f"Protocol/source-manifest disagreement: {name}")
    return dict(passed=not failures, failures=failures,
        checked_protocol_sources=len(pins), manifest=manifest,
        source_manifest_sha256=digest(manifest_path) if manifest else None)


def load_npz(path, device, names=None):
    import numpy as np  # Saved-array I/O only; no NumPy scientific operations.
    arrays, hashes = {}, {}
    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files if names is None else names:
            value = archive[name]
            if value.dtype.kind not in "biuf":
                raise ValueError(f"Unsupported saved-array type: {name}/{value.dtype}")
            if value.dtype.kind == "f" and value.dtype.name != "float64":
                raise ValueError(f"Non-float64 saved numerical array: {name}/{value.dtype}")
            hashes[name] = hashlib.sha256(value.tobytes(order="C")).hexdigest()
            arrays[name] = jax.device_put(value, device)
    return arrays, hashes


def rms_errors(arrays, count):
    prediction, truth = arrays["prediction"][:count], arrays["truth"][:count]
    error = prediction-truth
    point_error = jnp.linalg.norm(error, axis=1)
    truth_energy = jnp.sum(truth*truth)
    jac = arrays["prediction_jacobian"][:count]
    exact_jac = arrays["truth_jacobian"][:count]
    mask = arrays["query_diagnostic_derivative_defined"][:count]
    n = jnp.maximum(jnp.sum(mask), 1)
    divergence = jnp.where(mask, jnp.trace(jac, axis1=1, axis2=2), 0.)
    gradient_energy = jnp.sum(jnp.where(mask[:, None, None], jac*jac, 0.))/n
    tiny = jnp.finfo(truth.dtype).tiny
    return dict(relative_rms=jnp.sqrt(jnp.sum(error*error)/jnp.maximum(truth_energy, tiny)),
        absolute_rms=jnp.sqrt(jnp.mean(point_error**2)), sampled_maximum=jnp.max(point_error),
        gradient_relative_rms=jnp.sqrt(jnp.sum(jnp.where(mask[:, None, None], (jac-exact_jac)**2, 0.))
            / jnp.maximum(jnp.sum(jnp.where(mask[:, None, None], exact_jac**2, 0.)), tiny)),
        divergence_absolute_rms=jnp.sqrt(jnp.sum(divergence**2)/n),
        divergence_sampled_maximum=jnp.max(jnp.abs(divergence)),
        divergence_normalized_rms=jnp.sqrt(jnp.sum(divergence**2)/n)/jnp.maximum(1., jnp.sqrt(gradient_energy)),
        finite=jnp.all(jnp.isfinite(prediction))
            & jnp.all(jnp.where(mask[:, None, None], jnp.isfinite(jac), True)))


def periodic_metrics(arrays, groups):
    result = {}
    for kind in ("face", "edge", "corner", "wrapped"):
        value, jacobian = [], []
        for group in groups:
            if group["kind"] != kind or group["start"] == group["reference_start"]:
                continue
            a = slice(group["reference_start"], group["reference_stop"])
            b = slice(group["start"], group["stop"])
            value.append(jnp.linalg.norm(arrays["prediction"][b]-arrays["prediction"][a], axis=1))
            jacobian.append(jnp.linalg.norm(arrays["prediction_jacobian"][b]
                - arrays["prediction_jacobian"][a], axis=(1, 2)))
        if value:
            result[kind] = dict(value_maximum=jnp.max(jnp.concatenate(value)),
                jacobian_maximum=jnp.max(jnp.concatenate(jacobian)))
    return result


def expected_cells():
    return {f"{suite}_d{dim}_p{order}_{kernel}_level{level}"
        for suite in ("irregular", "periodic") for dim in (2, 3)
        for order in (2, 4, 6) for kernel in ("phs", "matern")
        for level in (range(4) if suite == "irregular" else range(1))}


def scattered_geometry_evidence(arrays, cloud):
    """Independent reductions of the canonical-fit/exact-boundary evidence."""
    check = arrays["geometry_validation_exact_positions"]
    predicted = arrays["geometry_validation_fitted_positions"]
    normals = arrays["geometry_validation_exact_normals"]
    fitted_normals = arrays["geometry_validation_fitted_normals"]
    boundary = arrays["geometry_exact_projected_boundary"]
    fitted_boundary = arrays["geometry_fitted_boundary"]
    x, y = boundary[:, 0], boundary[:, 1]
    if boundary.shape[1] == 2:
        value = 1-x*x-1.4*y*y-.15*x**4-.10*x*x*y*y-.20*y**4+.10*x**3
    else:
        z = boundary[:, 2]
        value = (1-x*x-1.3*y*y-1.7*z*z-.10*(x**4+y**4+z**4)
            -.08*x*x*y*y-.06*y*y*z*z+.10*x*y*z)
    return dict(
        maximum_geometry_validation_position_error=jnp.max(jnp.linalg.norm(predicted-check, axis=1)),
        maximum_geometry_validation_normal_error=jnp.max(jnp.linalg.norm(fitted_normals-normals, axis=1)),
        maximum_boundary_projection_distance=jnp.max(jnp.linalg.norm(boundary-fitted_boundary, axis=1)),
        maximum_exact_boundary_residual=jnp.max(jnp.abs(value)),
        maximum_cfpu_exact_boundary_residual=jnp.max(jnp.abs(arrays["geometry_cfpu_boundary_values"])),
        minimum_cfpu_node_value=jnp.min(arrays["geometry_cfpu_node_values"]),
        exact_boundary_oracle_passed=jnp.max(jnp.abs(value)) <= 1e-12,
        exact_boundary_in_visibility_layer=jnp.all(
            arrays["geometry_cfpu_boundary_values"] >= -cloud["boundary_visibility_tolerance"]),
        exact_nodes_in_visibility_layer=jnp.all(
            arrays["geometry_cfpu_node_values"] >= -cloud["boundary_visibility_tolerance"]),
        geometry_finite=jnp.all(jnp.isfinite(predicted)) & jnp.all(jnp.isfinite(fitted_normals)))


def verify_cell(path, protocol, protocol_hash, device):
    record = json.loads(path.read_text())
    failures = []
    if record["protocol_sha256"] != protocol_hash:
        failures.append("Protocol hash mismatch")
    artifact = path.parent/"arrays.npz"
    if not artifact.exists() or digest(artifact) != record["arrays_sha256"]:
        raise ValueError("Saved NPZ hash mismatch")
    arrays, hashes = load_npz(artifact, device)
    for key, field in (("nodes", "nodes_sha256"), ("queries", "queries_sha256"),
                       ("sampled_h_queries", "sampled_h_queries_sha256")):
        if hashes[key] != record[field]:
            failures.append(f"Input hash mismatch: {key}")
    if record["dimension"] != arrays["nodes"].shape[1] or record["points"] != len(arrays["nodes"]):
        failures.append("Geometry shape disagrees with record")
    count = (protocol[f"interior_queries_{record['dimension']}d"]
             if record["suite"] == "irregular" else len(arrays["queries"]))
    metrics = rms_errors(arrays, count)
    for key, value in metrics.items():
        if key == "finite":
            if not bool(scalar(value)):
                failures.append("Nonfinite independently checked predictions")
        elif not same(value, record["errors"][key]):
            failures.append(f"Saved error disagrees with independent reduction: {key}")
    h = jnp.sqrt(jnp.max(arrays["sampled_h_nearest_squared"]))
    if not same(h, record["geometry"]["sampled_fill_distance"]):
        failures.append("Saved sampled fill distance disagrees with saved distance array")
    nodal_error = jnp.linalg.norm(arrays["nodal_prediction"]-arrays["nodal_values"])
    nodal_error /= jnp.maximum(jnp.linalg.norm(arrays["nodal_values"]), jnp.finfo(jnp.float64).tiny)
    if not same(nodal_error, record["nodal_field_relative_error"]):
        failures.append("Saved nodal interpolation error disagrees with independent reduction")
    tolerance = protocol["tolerance"]
    geometry_evidence = None
    if protocol.get("schema") == "fastol_spatial_properties_scattered_v2":
        if protocol["batch_size"] != 32 or protocol["warm_repetitions"] != 11:
            failures.append("The full scattered study must use native map batches 32 and 11 warm repetitions")
        if len(record["timings"]["warm_value_evaluation_seconds"]) != 11:
            failures.append("Missing synchronized warm timing repetitions")
        fallback_rows=arrays["unreachable_euclidean_stencil_fallback_rows"]
        if not same(fallback_rows,record["unreachable_euclidean_stencil_fallback_rows"]):
            failures.append("Unreachable Euclidean stencil fallback count disagrees")
        if not bool(scalar(fallback_rows==0)):
            failures.append("Boundary-aware stencils used an unreachable Euclidean fill")
        expected_normalization = "sqrt(sum_q(||f(q)||^2 + mean_i ||f(x_qi)||^2)) per generator"
        for field in ("independent_polynomial", "independent_nodal_polynomial"):
            if record[field].get("normalization") != expected_normalization:
                failures.append("The independent polynomial oracle omitted stencil-input normalization")
        if record["suite"] == "irregular":
            geometry_evidence = scattered_geometry_evidence(arrays, record["cloud"])
            for field, value in geometry_evidence.items():
                if field.endswith("passed") or field.endswith("layer") or field.endswith("finite"):
                    if not bool(scalar(value)):
                        failures.append("Independent scattered geometry screen failed: " + field)
                elif not same(value, record["cloud"][field]):
                    failures.append("Scattered geometry reduction disagrees: " + field)
        elif record["cloud"]["policy"] != "kernelpack_GPU_Poisson_MIS_on_2pi_torus":
            failures.append("The periodic cloud did not use the shared minimum-image Poisson sampler")
    independent_checks = dict(finite_predictions=metrics["finite"],
        nodal_field_interpolation=nodal_error <= tolerance,
        local_divergence=metrics["divergence_normalized_rms"] <= tolerance,
        independent_requested_polynomial=jnp.max(arrays["independent_polynomial_generator_relative_rms"]) <= tolerance,
        independent_nodal_requested_polynomial=jnp.max(arrays["independent_nodal_polynomial_generator_relative_rms"]) <= tolerance)
    for prefix in ("nodal", "query"):
        get = lambda key: arrays[prefix+"_diagnostic_"+key]
        independent_checks[prefix+"_full_rank"] = jnp.all(get("retained_rank") == record["polynomial_dimension"])
        independent_checks[prefix+"_finite"] = jnp.all(get("finite"))
        independent_checks[prefix+"_no_fallback"] = ~jnp.any(get("fallback_activated"))
        for key in ("solver_relative_residual", "solver_backward_error", "interpolation_relative_residual",
                    "retained_polynomial_relative_error", "requested_polynomial_relative_error"):
            independent_checks[prefix+"_"+key] = jnp.all(get(key) <= tolerance)
        independent_checks[prefix+"_condition_screen"] = jnp.all(
            get("equilibrated_condition_times_epsilon") <= protocol["condition_epsilon_limit"])
    derivative_mask = arrays["query_diagnostic_derivative_defined"]
    independent_checks["defined_derivatives_exist"] = jnp.any(derivative_mask)
    independent_checks["gradient_finite"] = jnp.all(jnp.where(derivative_mask,
        arrays["query_diagnostic_gradient_finite"], True))
    independent_checks["divergence_weights"] = jnp.all(jnp.where(derivative_mask,
        arrays["query_diagnostic_divergence_weight_normalized"] <= tolerance, True))
    periodic = periodic_metrics(arrays, record["periodic_probe_groups"] or [])
    periodic_reference_scales=dict(value_maximum=jnp.maximum(1.,jnp.max(jnp.linalg.norm(arrays["truth"],axis=1))),
        jacobian_maximum=jnp.maximum(1.,jnp.max(jnp.linalg.norm(arrays["truth_jacobian"],axis=(1,2)))))
    for kind, values in periodic.items():
        if not same(values["value_maximum"], record["periodicity"][kind]["value_jump"]["sampled_maximum"]):
            failures.append(f"Periodic value gap mismatch: {kind}")
        if not same(values["jacobian_maximum"], record["periodicity"][kind]["jacobian_jump"]["sampled_maximum"]):
            failures.append(f"Periodic Jacobian gap mismatch: {kind}")
        independent_checks["periodic_"+kind+"_values"] = values["value_maximum"] <= tolerance*jnp.maximum(
            1., jnp.max(jnp.linalg.norm(arrays["truth"], axis=1)))
        independent_checks["periodic_"+kind+"_jacobians"] = values["jacobian_maximum"] <= tolerance*jnp.maximum(
            1., jnp.max(jnp.linalg.norm(arrays["truth_jacobian"], axis=(1, 2))))
    boundary = None
    if record["suite"] == "irregular":
        difference = arrays["prediction"][count:]-arrays["truth"][count:]
        boundary = jnp.sqrt(jnp.mean(jnp.sum(difference*difference, axis=1)))
        if not same(boundary, record["boundary"]["absolute_rms"]):
            failures.append("Boundary RMS mismatch")
    original_failed_checks = [key for key, passed in record["scientific_checks"].items() if not passed]
    failed_checks = original_failed_checks + ["independent:"+key for key, passed in independent_checks.items() if not bool(scalar(passed))]
    return dict(cell=record["cell"], suite=record["suite"], dimension=record["dimension"],
        order=record["requested_value_order"], kernel=record["kernel"], level=record["level"],
        nodes=record["points"], nodes_sha256=record["nodes_sha256"],
        h=h, nominal_spacing=record["cloud"]["nominal_spacing"],
        minimum_node_separation=record["geometry"]["minimum_node_separation"],
        half_minimum_node_separation=record["geometry"]["half_minimum_node_separation"],
        errors=metrics, boundary_absolute_rms=boundary,
        scattered_geometry_evidence=geometry_evidence,
        periodicity=periodic, periodic_reference_scales=periodic_reference_scales,
        nodal_relative_error=nodal_error,
        scientific_checks_passed=not failed_checks, failed_scientific_checks=failed_checks,
        original_passed=record["passed"], original_scientific_checks=record["scientific_checks"],
        original_failed_scientific_checks=original_failed_checks,
        independent_scientific_checks=independent_checks,
        verification_passed=not failures, verification_failures=failures,
        geometry_plot=(dict(nodes=arrays["nodes"],
            nodal_speed=jnp.linalg.norm(arrays["nodal_values"], axis=1))
            if record["suite"] == "irregular" and record["level"] == 3
            and record["requested_value_order"] == 6 and record["kernel"] == "phs" else None),
        arrays_sha256=record["arrays_sha256"], record_sha256=digest(path),
        timings=record["timings"], diagnostics=record["diagnostics"],
        query_diagnostics=record["query_diagnostics"])


def oracle_recheck_report(folder, study, source_root, protocol, protocol_hash, rows,
                         source_check, device):
    """Authenticate supplementary evidence without replacing an original flag."""
    result=dict(provided=True,accepted=False,mode=None,cells=[],failures=[],
        qualification="Original flags and complete/pass counts remain unchanged. Supplemental acceptance is reported separately.")
    def require(condition,message):
        if not condition:
            raise ValueError(message)
    try:
        manifest_path,completion_path=folder/"manifest.json",folder/"completion.json"
        manifest=json.loads(manifest_path.read_text())
        completed=json.loads(completion_path.read_text())
        mode=manifest["mode"]
        result.update(mode=mode,manifest_sha256=digest(manifest_path),completion_sha256=digest(completion_path))
        require(manifest["schema"]=="fastol_nodal_polynomial_oracle_recheck_v1","Unknown supplement schema")
        require(mode in ("certificate-only","full-rebuild"),"Unknown supplemental evidence mode")
        require(source_check["passed"],"Original frozen source authentication failed")
        require(Path(manifest["study"]).resolve()==study.resolve(),"Supplement refers to another study")
        require(Path(manifest["source_root"]).resolve()==source_root.resolve(),"Supplement refers to another source tree")
        require(manifest["script_sha256"]==REVIEWED_ORACLE_RECHECK_SHA256,"Supplement was produced by an unreviewed recheck script")
        require(manifest["protocol_sha256"]==protocol_hash,"Supplement protocol pin mismatch")
        require(manifest["protocol_file_sha256"]==digest(study/"protocol.json"),"Supplement protocol-file hash mismatch")
        original_completion_path=study/"completion.json"
        original_completion=json.loads(original_completion_path.read_text())
        require(manifest["completion_file_sha256"]==digest(original_completion_path),"Original completion pin mismatch")
        require(protocol["configuration_count"]==60 and len(rows)==60 and {r["cell"] for r in rows}==expected_cells(),
                "Supplement requires all60 verified/readable cells")
        require(original_completion["protocol_sha256"]==protocol_hash and original_completion["completed_cells"]==60,
                "Original study completion disagrees")
        require({r["cell"]:r["passed"] for r in original_completion["results"]}=={r["cell"]:r["original_passed"] for r in rows},
                "Original completion/pass census disagrees")
        require(manifest["tolerance"]==protocol["tolerance"]==1e-8,"Supplement changed the tolerance")
        require(not manifest["unresolved_other_failures"] and not completed["unresolved"],"Supplement contains unresolved other failures")
        require(completed["mode"]==mode and completed["accepted"] and completed["original_flags_unchanged"],
                "Supplement completion did not explicitly accept its evidence with original flags unchanged")
        require(manifest["original_result_sha256"]=={r["cell"]:r["record_sha256"] for r in rows},
                "Original result-file hashes disagree")
        eligible={r["cell"] for r in rows if r["original_failed_scientific_checks"]==[NODAL_ORACLE_FLAG]}
        require(set(manifest["eligible_cells"])==eligible and len(manifest["eligible_cells"])==len(eligible),
                "Supplement eligible-cell census disagrees")
        allowed={NODAL_ORACLE_FLAG,"independent:independent_nodal_requested_polynomial"}
        for row in rows:
            require(row["verification_passed"],f"Original artifact verification failed: {row['cell']}")
            require(row["original_passed"]==(not row["original_failed_scientific_checks"]),
                    f"Original pass flag disagrees with its saved checks: {row['cell']}")
            require(not row["failed_scientific_checks"] or (row["cell"] in eligible and set(row["failed_scientific_checks"])<=allowed),
                    f"Unresolved other scientific check: {row['cell']}")
        checked={r["cell"]:r for r in completed["checked"]}
        require(set(checked)==eligible and len(completed["checked"])==len(eligible),"Supplement completed-cell census disagrees")
        if mode=="certificate-only":
            require(manifest["audited_certificate_residual_sources"]==REVIEWED_NODAL_RESIDUAL_SOURCES,
                    "Certificate residual formula was not the reviewed producer")
            require(all(protocol["source_sha256"].get(name)==pin for name,pin in REVIEWED_NODAL_RESIDUAL_SOURCES.items()),
                    "Certificate producer-source pins disagree")
        for row in rows:
            if row["cell"] not in eligible:
                continue
            key=row["cell"]
            path=folder/key/"result.json"
            supplemental=json.loads(path.read_text())
            require(supplemental==checked[key],f"Supplement result/completion disagreement: {key}")
            require(supplemental["original_scientific_checks"]==row["original_scientific_checks"]
                    and supplemental["original_passed"]==row["original_passed"]
                    and supplemental["original_failed_flags"]==[NODAL_ORACLE_FLAG],f"Original flags were not preserved: {key}")
            require(supplemental["original_arrays_sha256"]==row["arrays_sha256"],f"Original NPZ pin mismatch: {key}")
            require(all(supplemental["checks"].values()),f"Supplement saved a failed check: {key}")
            name="certificate_arrays.npz" if mode=="certificate-only" else "recheck_arrays.npz"
            artifact=folder/key/name
            require(digest(artifact)==supplemental["supplementary_arrays_sha256"],f"Supplement NPZ hash mismatch: {key}")
            extra,_=load_npz(artifact,device)
            original,_=load_npz(study/"cells"/key/"arrays.npz",device,names=("local_stencil_indices",
                "nodal_diagnostic_query_is_stencil_node","nodal_diagnostic_query_nodal_interpolation_relative_error",
                "independent_nodal_polynomial_generator_relative_rms"))
            delta=original["nodal_diagnostic_query_nodal_interpolation_relative_error"]
            ids=original["local_stencil_indices"]
            count,k=ids.shape
            require(count==row["nodes"] and delta.shape==(count,),f"Original certificate dimensions disagree: {key}")
            self_counts=jnp.sum(ids==jnp.arange(count,dtype=ids.dtype)[:,None],axis=1)
            unique=jnp.all(jnp.diff(jnp.sort(ids,axis=1),axis=1)>0,axis=1)
            valid=jnp.all((ids>=0)&(ids<count),axis=1)
            p=row["dimension"]*k
            epsilon=jnp.asarray(jnp.finfo(jnp.float64).eps,dtype=jnp.float64)
            gamma=p*epsilon/(1-p*epsilon)
            mathematical=jnp.sqrt(jnp.asarray(p,dtype=jnp.float64))*delta
            fp=jnp.sqrt(jnp.asarray(p,dtype=jnp.float64))*(delta+gamma*(1+delta))
            common=jnp.all(original["nodal_diagnostic_query_is_stencil_node"])&jnp.all(self_counts==1)&jnp.all(unique&valid)
            common &= jnp.all(jnp.isfinite(delta)&(delta>=0.))&(p*epsilon<1)
            common &= jnp.all(jnp.isfinite(mathematical)&(mathematical<=1e-8))
            require(bool(scalar(common)),f"Independent all-node certificate hypotheses failed: {key}")
            require(bool(scalar(jnp.all(extra["saved_nodal_map_relative_residual"]==delta))),f"Supplement residual array changed: {key}")
            if mode=="certificate-only":
                require(supplemental["mode"]==mode and supplemental["certificate_passed"],f"Certificate result mode/outcome disagrees: {key}")
                require(bool(scalar(jnp.all(jnp.isfinite(fp)&(fp<=1e-8)))),f"Independent FP64 certificate failed: {key}")
                require(same(extra["mathematical_all_nodal_fields_bound"],mathematical)
                        and same(extra["float64_all_nodal_fields_bound"],fp),f"Certificate bound arrays disagree: {key}")
                require(bool(scalar(jnp.all(extra["local_stencil_indices"]==ids)
                    &jnp.all(extra["self_index_count"]==self_counts)
                    &jnp.all(extra["saved_query_is_stencil_node"]==original["nodal_diagnostic_query_is_stencil_node"])
                    &jnp.all(extra["unique_stencil_indices"]==unique)&jnp.all(extra["valid_stencil_indices"]==valid))),
                    f"Certificate identity arrays disagree: {key}")
                classification="saved-residual certificate; no newly measured monomial error"
            else:
                require(supplemental.get("mode","full-rebuild")==mode and supplemental["corrected_oracle_passed"],
                        f"Measured re-evaluation outcome disagrees: {key}")
                for field in ("input_scaled_generator_rms","prescaled_input_scaled_generator_rms","pointwise_prescaled_generator_maximum"):
                    value=extra[field]
                    require(bool(scalar(jnp.all(jnp.isfinite(value)&(value>=0.)&(value<=1e-8)))),f"Measured corrected oracle failed: {key}/{field}")
                require(bool(scalar(jnp.all(extra["saved_legacy_generator_rms"]==original["independent_nodal_polynomial_generator_relative_rms"]))),
                        f"Original legacy errors changed: {key}")
                require(same(extra["universal_all_nodal_fields_bound"],mathematical),f"Measured supplement all-field bound disagrees: {key}")
                classification="measured independent monomial re-evaluation with input scaling"
            result["cells"].append(dict(cell=key,accepted=True,classification=classification,
                mode=mode,original_flags_unchanged=True,result_sha256=digest(path),arrays_sha256=digest(artifact),
                mathematical_all_field_bound_maximum=jnp.max(mathematical),
                float64_all_field_bound_maximum=jnp.max(fp) if mode=="certificate-only" else None,
                saved_result=supplemental))
            del extra,original
        result["accepted"]=True
        result["eligible_cells"]=sorted(eligible)
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
    return result


def fitted_rate(h, error):
    x, y = jnp.log(h), jnp.log(error)
    x, y = x-jnp.mean(x), y-jnp.mean(y)
    return jnp.sum(x*y)/jnp.sum(x*x)


def convergence_curves(rows):
    curves = []
    for dimension in (2, 3):
        for kernel in ("phs", "matern"):
            for order in (2, 4, 6):
                cells = sorted((r for r in rows if r["suite"] == "irregular"
                    and r["dimension"] == dimension and r["kernel"] == kernel and r["order"] == order),
                    key=lambda r: r["level"])
                if not cells:
                    continue
                h = jnp.stack([r["h"] for r in cells])
                nominal_h = jnp.asarray([r["nominal_spacing"] for r in cells], dtype=jnp.float64)
                error = jnp.stack([r["errors"]["relative_rms"] for r in cells])
                complete = len(cells) == 4
                curves.append(dict(dimension=dimension, kernel=kernel, order=order,
                    label=f"Target order {order}", cells=[r["cell"] for r in cells],
                    checked=all(r["verification_passed"] and r["scientific_checks_passed"] for r in cells),
                    four_levels_available=complete, h=h, error=error,
                    nominal_h=nominal_h,
                    boundary_error=jnp.stack([r["boundary_absolute_rms"] for r in cells]),
                    divergence=jnp.stack([r["errors"]["divergence_normalized_rms"] for r in cells]),
                    reference_slope=order, reference=error[-1]*(h/h[-1])**order,
                    nominal_reference=error[-1]*(nominal_h/nominal_h[-1])**order,
                    successive_rates=jnp.diff(jnp.log(error))/jnp.diff(jnp.log(h)),
                    all_four_rate=fitted_rate(h, error) if complete else None,
                    last_three_rate=fitted_rate(h[-3:], error[-3:]) if complete else None,
                    all_four_nominal_spacing_rate=fitted_rate(nominal_h, error) if complete else None,
                    last_three_nominal_spacing_rate=fitted_rate(nominal_h[-3:], error[-3:]) if complete else None,
                    successive_nominal_spacing_rates=jnp.diff(jnp.log(error))/jnp.diff(jnp.log(nominal_h))))
    return curves


def cloud_geometry_table(rows):
    """One row per saved physical cloud, shared by all orders and kernels.

    Separation is the recorded all-node minimum; sampled fill distance uses
    the independent fixed query set and is not a continuum supremum bound.
    """
    clouds = {}
    for row in sorted(rows, key=lambda r: (r["dimension"], r["suite"], r["level"])):
        key = (row["suite"], row["dimension"], row["nodes_sha256"])
        if key in clouds:
            continue
        half_separation = jnp.asarray(row["half_minimum_node_separation"], dtype=jnp.float64)
        clouds[key] = dict(suite=row["suite"], dimension=row["dimension"], level=row["level"],
            spatial_node_count_M=row["nodes"], nodes_sha256=row["nodes_sha256"],
            nominal_spacing=row["nominal_spacing"], sampled_fill_distance=row["h"],
            minimum_node_separation=row["minimum_node_separation"],
            half_minimum_node_separation=half_separation,
            sampled_fill_to_half_separation_ratio=jnp.where(half_separation > 0,
                row["h"] / half_separation, jnp.nan))
    return list(clouds.values())


def periodic_plot_arrays(rows):
    plots = []
    for dimension in (2, 3):
        series = []
        for kernel, label in (("phs", "PHS"), ("matern", "Matérn")):
            cells = sorted((r for r in rows if r["suite"] == "periodic"
                and r["dimension"] == dimension and r["kernel"] == kernel), key=lambda r: r["order"])
            if not cells:
                continue
            for key, scope in (("value_maximum", "values"), ("jacobian_maximum", "Jacobians")):
                series.append(dict(label=f"{label}, {scope}",
                    order=jnp.asarray([r["order"] for r in cells], dtype=jnp.int32),
                    reference_scale=jnp.stack([r["periodic_reference_scales"][key] for r in cells]),
                    gap=jnp.stack([jnp.max(jnp.stack([g[key] for g in r["periodicity"].values()]))
                        /r["periodic_reference_scales"][key] for r in cells])))
        if series:
            plots.append(dict(dimension=dimension, series=series,
                scope="Maximum over saved face, edge, corner and joint-shift identifications, normalized separately for values and Jacobians",
                normalization="Each absolute velocity gap is divided by max(1,max_q||u(q)||_2); each absolute Jacobian gap by max(1,max_q||Du(q)||_F), using the exact reference field on the saved queries. Absolute gaps remain in the cell reports.",
                ylabel="Maximum normalized periodic gap"))
    return plots


@jax.jit
def covariance_energy_batch(wy, xy, y, wz, xz, z, modes):
    """Independent bounded Fourier-factor contraction of local increment maps."""
    def mode_block(total, block):
        wave, frame = block
        phase_y = jnp.einsum("qki,mi->qkm", xy, wave)
        phase_z = jnp.einsum("qki,mi->qkm", xz, wave)
        cy = jnp.einsum("qakb,mbt,qkm->qmat", wy, frame, jnp.cos(phase_y))
        sy = jnp.einsum("qakb,mbt,qkm->qmat", wy, frame, jnp.sin(phase_y))
        cz = jnp.einsum("qakb,mbt,qkm->qmat", wz, frame, jnp.cos(phase_z))
        sz = jnp.einsum("qakb,mbt,qkm->qmat", wz, frame, jnp.sin(phase_z))
        py, pz = y @ wave.T, z @ wave.T
        tyc, tys = jnp.cos(py)[:, :, None, None]*frame, jnp.sin(py)[:, :, None, None]*frame
        tzc, tzs = jnp.cos(pz)[:, :, None, None]*frame, jnp.sin(pz)[:, :, None, None]*frame
        norm = lambda c, s: jnp.sum(c*c+s*s, axis=(1, 2, 3))
        energy = jnp.stack((norm(cy-cz, sy-sz), norm(tyc-tzc, tys-tzs),
            norm(cy-tyc, sy-tys), norm(cz-tzc, sz-tzs),
            norm(cy-cz-tyc+tzc, sy-sz-tys+tzs)), axis=1)
        return total+energy, None
    initial = jnp.zeros((len(y), 5), dtype=y.dtype)
    return jax.lax.scan(mode_block, initial, modes)[0]


def covariance_energies(reference, inputs, diagnostics, y, z, *, pair_batch=4, mode_batch=64):
    count = len(y)
    weights, ids = diagnostics["weights"], diagnostics["stencil_indices"]
    if weights.shape[0] != 2*count:
        raise ValueError("Saved statistical map must contain all y then all z rows")
    coordinates = inputs["nodes"][ids]
    padding = (-len(reference["weights"])) % mode_batch
    wave = jnp.pad(reference["wavevectors"], ((0, padding), (0, 0))).reshape(-1, mode_batch, 3)
    frame = reference["frames"]*jnp.sqrt(reference["weights"])[:, None, None]
    frame = jnp.pad(frame, ((0, padding), (0, 0), (0, 0))).reshape(-1, mode_batch, 3, 2)
    outputs = []
    for start in range(0, count, pair_batch):
        valid = min(pair_batch, count-start)
        take = jnp.minimum(jnp.arange(pair_batch)+start, count-1)
        energy = covariance_energy_batch(weights[take], coordinates[take], y[take],
            weights[count+take], coordinates[count+take], z[take], (wave, frame))
        outputs.append(energy[:valid])
    return jnp.concatenate(outputs)


def statistical_spatial_checks(arrays, degree, dimension):
    """Recheck the complete requested polynomial space from saved arrays."""
    required=("retained_rank","requested_rank","finite","fallback_activated",
        "solver_backward_error","interpolation_relative_residual",
        "retained_polynomial_relative_error","requested_polynomial_relative_error")
    missing=[key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"Missing statistical spatial diagnostics: {missing}")
    requested=dimension*math.comb(degree+dimension,dimension)-math.comb(degree+dimension-1,dimension)
    checks=dict(full_requested_rank=jnp.all(arrays["retained_rank"]==requested),
        declared_requested_rank=jnp.all(arrays["requested_rank"]==requested),
        finite_local_maps=jnp.all(arrays["finite"]),
        finite_saved_spatial_arrays=jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in arrays.values()])),
        no_pseudoinverse_fallback=~jnp.any(arrays["fallback_activated"]),
        original_system_backward_error=jnp.all(arrays["solver_backward_error"]<=1e-10),
        interpolation=jnp.all(arrays["interpolation_relative_residual"]<=1e-8),
        retained_polynomial_reproduction=jnp.all(arrays["retained_polynomial_relative_error"]<=1e-8),
        requested_polynomial_reproduction=jnp.all(arrays["requested_polynomial_relative_error"]<=1e-8))
    if "unreachable_euclidean_stencil_fallback_rows" in arrays:
        checks["zero_unreachable_stencil_fallback"] = arrays["unreachable_euclidean_stencil_fallback_rows"]==0
    return dict(checks=checks,passed=jnp.all(jnp.stack(list(checks.values()))),
        expected_polynomial_dimension=requested,
        retained_rank_minimum=jnp.min(arrays["retained_rank"]),
        requested_polynomial_error_maximum=jnp.max(arrays["requested_polynomial_relative_error"]))


def statistical_periodic_groups(arrays, groups):
    """Verify actual face/edge/corner coordinates, then reduce both field gaps."""
    prediction,jacobian=arrays["prediction"],arrays["jacobian"]
    truth,true_jacobian=arrays["truth"],arrays["truth_jacobian"]
    queries=arrays["queries"]
    count=len(queries)
    scales=(jnp.maximum(jnp.max(jnp.linalg.norm(prediction,axis=-1)),1.),
        jnp.maximum(jnp.max(jnp.linalg.norm(jacobian,axis=(1,2))),1.),
        jnp.maximum(jnp.max(jnp.linalg.norm(truth,axis=-1)),1.),
        jnp.maximum(jnp.max(jnp.linalg.norm(true_jacobian,axis=(1,2))),1.))
    geometry_checks=[]
    coverage={}
    for group in groups:
        if group["kind"] not in ("face","edge","corner","wrapped"):
            raise ValueError("Unknown statistical periodic probe kind")
        start,stop,first,last=(group[key] for key in ("start","stop","reference_start","reference_stop"))
        if not all(isinstance(value,int) for value in (start,stop,first,last)) or not (0<=start<stop<=count and 0<=first<last<=count and stop-start==last-first):
            raise ValueError("Statistical periodic probe indices are invalid")
        query,reference=queries[start:stop],queries[first:last]
        if group["kind"]=="wrapped":
            shift=tuple(group["shift"])
            if len(shift)!=3 or any(int(value)!=value for value in shift):
                raise ValueError("A periodic joint shift is not an integer3-vector")
            geometry_checks.append(jnp.all(jnp.abs(query-reference-jnp.asarray(shift,dtype=jnp.float64)*(2*jnp.pi))<=1e-12))
            coverage.setdefault("wrapped",set()).add(shift)
        else:
            axes,mask=tuple(group["axes"]),tuple(group["identification"])
            codimension={"face":1,"edge":2,"corner":3}[group["kind"]]
            if len(axes)!=codimension or len(set(axes))!=codimension or len(mask)!=codimension or any(axis not in (0,1,2) for axis in axes) or any(side not in (0,1) for side in mask):
                raise ValueError("A face/edge/corner probe has invalid axes or identifications")
            for axis,side in zip(axes,mask):
                geometry_checks.append(jnp.all(jnp.abs(query[:,axis]-side*(2*jnp.pi))<=1e-12))
            for axis in set((0,1,2))-set(axes):
                geometry_checks.append(jnp.all(query[:,axis]==reference[:,axis]))
            coverage.setdefault((group["kind"],axes),set()).add(mask)
    for kind,codimension in (("face",1),("edge",2),("corner",3)):
        for axes in itertools.combinations(range(3),codimension):
            if coverage.get((kind,axes))!=set(itertools.product((0,1),repeat=codimension)):
                raise ValueError("Actual statistical boundary probes do not cover every identification")
    expected_shifts=set(itertools.product((-1,0,1),repeat=3))|{(2,-1,1),(-2,2,-1)}
    if coverage.get("wrapped")!=expected_shifts:
        raise ValueError("Statistical wrapped probes lack the complete joint-shift set")
    result={}
    for kind in ("face","edge","corner","wrapped"):
        differences=[[],[],[],[]]
        pairs=0
        for group in groups:
            if group["kind"]!=kind or group["start"]==group["reference_start"]:
                continue
            a=slice(group["reference_start"],group["reference_stop"])
            b=slice(group["start"],group["stop"])
            differences[0].append(jnp.linalg.norm(prediction[b]-prediction[a],axis=-1))
            differences[1].append(jnp.linalg.norm(jacobian[b]-jacobian[a],axis=(1,2)))
            differences[2].append(jnp.linalg.norm(truth[b]-truth[a],axis=-1))
            differences[3].append(jnp.linalg.norm(true_jacobian[b]-true_jacobian[a],axis=(1,2)))
            pairs+=group["stop"]-group["start"]
        values=[jnp.max(jnp.concatenate(value,axis=0))/scale for value,scale in zip(differences,scales)]
        result[kind]=dict(pairs=pairs,reconstructed_value_relative_gap=values[0],
            reconstructed_jacobian_relative_gap=values[1],exact_value_relative_gap=values[2],
            exact_jacobian_relative_gap=values[3],
            passed=(values[0]<=1e-8)&(values[1]<=1e-8)&(values[2]<=1e-10)&(values[3]<=1e-10))
    return result,jnp.all(jnp.stack(geometry_checks))


def statistics_report(folder, source_root, device):
    """Rebuild structure functions and slopes from authenticated saved arrays."""
    protocol = json.loads((folder/"protocol.json").read_text())
    protocol_hash = hashlib.sha256(canonical(protocol)).hexdigest()
    sources = verify_sources(source_root, protocol["source_sha256"])
    failures = list(sources["failures"])
    scattered = protocol.get("schema") == "fastol_spatial_statistics_scattered_v4"
    if scattered and (protocol.get("map_batch") != 32 or protocol.get("repetitions") != 11):
        failures.append("The complete scattered statistics protocol requires native batches 32 and 11 warm repetitions")
    if scattered:
        required = dict(spacings=[.26,.19,.14,.101],maximum_wavenumber=1024,
            draws=8,bins=32,directions=8,pair_batch=4,mode_batch=64,field_batch=256,
            periodic_n=12,statistical_policies=["phs","ram_local_tau16","fixed_physical_band"],
            expected_statistical_cells=36,expected_periodic_statistical_cells=9)
        for name,value in required.items():
            if protocol.get(name) != value:
                failures.append("The full scattered statistics protocol changed: " + name)
        if sorted(protocol["orders"]) != [2,4,6]:
            failures.append("The scattered statistics study omitted a requested order")
    reference_path = folder/"reference.npz"
    if digest(reference_path) != protocol["reference_npz_sha256"]:
        raise ValueError("Statistics reference NPZ hash mismatch")
    reference, reference_hashes = load_npz(reference_path, device)
    for name, expected in protocol["reference_array_sha256"].items():
        if reference_hashes.get(name) != expected:
            failures.append(f"Statistics reference input hash mismatch: {name}")
    rows, periodic, failures_saved = [], [], []
    for path in sorted(folder.rglob("result.json")):
        record = json.loads(path.read_text())
        case = record.get("case", str(path.parent.relative_to(folder)))
        local_failures = []
        try:
            if record["protocol_sha256"] != protocol_hash:
                raise ValueError("Statistics protocol hash mismatch")
            if record["reference_npz_sha256"] != protocol["reference_npz_sha256"]:
                raise ValueError("Statistics reference pin mismatch")
            input_path = (folder/record["input_artifact"]).resolve()
            if not input_path.is_relative_to(folder.resolve()):
                raise ValueError("Unscoped statistics input artifact")
            if digest(input_path) != record["input_artifact_sha256"]:
                raise ValueError("Statistics shared input NPZ hash mismatch")
            inputs, input_hashes = load_npz(input_path, device)
            if scattered:
                if len(record["timings"]["warm_value_evaluation_seconds"]) != 11:
                    local_failures.append("Missing statistical warm timing repetitions")
                if "geometry_exact_projected_boundary" in inputs:
                    geometry_evidence = scattered_geometry_evidence(inputs, record["cloud"])
                    for name,value in geometry_evidence.items():
                        if name.endswith("passed") or name.endswith("layer") or name.endswith("finite"):
                            if not bool(scalar(value)):
                                local_failures.append("Independent statistical geometry screen failed: " + name)
                        elif not same(value, record["cloud"][name]):
                            local_failures.append("Statistical geometry reduction disagrees: " + name)
            artifact = path.parent/"statistics.npz"
            diagnostics = path.parent/"spatial_diagnostics.npz"
            if digest(artifact) != record["statistics_npz_sha256"]:
                raise ValueError("Statistics output NPZ hash mismatch")
            if digest(diagnostics) != record["spatial_diagnostics_npz_sha256"]:
                raise ValueError("Statistics spatial-diagnostic NPZ hash mismatch")
            arrays, array_hashes = load_npz(artifact, device)
            diagnostic_arrays, _ = load_npz(diagnostics, device)
            if scattered and "unreachable_euclidean_stencil_fallback_rows" not in diagnostic_arrays:
                local_failures.append("Missing independently saved unreachable-stencil fallback count")
            spatial_checks=statistical_spatial_checks(diagnostic_arrays,record["polynomial_degree"],inputs["nodes"].shape[1])
            if not bool(scalar(spatial_checks["passed"])):
                local_failures.append("Independent full-rank/requested-reproduction/finite spatial checks failed")
            saved_spatial_checks=record.get("numerical_checks",record.get("spatial_checks"))
            if saved_spatial_checks is None or saved_spatial_checks["numerical_checks_passed"]!=bool(scalar(spatial_checks["passed"])):
                local_failures.append("Saved statistical spatial pass flag disagrees with independent diagnostics")
            # Each original-run input pin must resolve to a saved input/output
            # array. Identical names in both artifacts must also be identical.
            for name, expected in record["input_sha256"].items():
                observed = input_hashes.get(name, array_hashes.get(name))
                if observed != expected:
                    local_failures.append(f"Statistics input hash mismatch: {name}")
                if name in input_hashes and name in array_hashes and input_hashes[name] != array_hashes[name]:
                    local_failures.append(f"Statistics duplicated input disagrees: {name}")
            if "pair_reconstructed_s2" not in arrays:
                prediction, jacobian = arrays["prediction"], arrays["jacobian"]
                count = len(arrays["base_queries"])
                legacy_count=count*(1+len(arrays["offsets"]))
                if record.get("legacy_translation_query_count",legacy_count)!=legacy_count:
                    local_failures.append("Joint-translation query count disagrees")
                shifted = prediction[count:legacy_count].reshape(len(arrays["offsets"]), count, *prediction.shape[1:])
                value_gap = jnp.max(jnp.linalg.norm(shifted-prediction[:count][None], axis=-1))
                value_gap /= jnp.maximum(jnp.max(jnp.linalg.norm(prediction[:count], axis=-1)), 1.)
                shifted_jacobian=jacobian[count:legacy_count].reshape(len(arrays["offsets"]),count,*jacobian.shape[1:])
                jacobian_gap=jnp.max(jnp.linalg.norm(shifted_jacobian-jacobian[:count][None],axis=(2,3)))
                jacobian_gap/=jnp.maximum(jnp.max(jnp.linalg.norm(jacobian[:count],axis=(1,2))),1.)
                divergence = jnp.trace(jacobian, axis1=1, axis2=2)
                divergence_gap = jnp.max(jnp.abs(divergence))/jnp.maximum(jnp.max(jnp.abs(jacobian)), 1.)
                true_values = arrays["truth"]
                true_shifted = true_values[count:legacy_count].reshape(len(arrays["offsets"]), count, *true_values.shape[1:])
                true_gap = jnp.max(jnp.linalg.norm(true_shifted-true_values[:count][None], axis=-1))
                true_gap /= jnp.maximum(jnp.max(jnp.linalg.norm(true_values[:legacy_count], axis=-1)), 1.)
                translation_field="reconstructed_joint_translation_periodicity_relative_gap" if "periodic_probe_groups" in record else "reconstructed_face_edge_corner_periodicity_relative_gap"
                if not same(value_gap, record[translation_field]) or not same(true_gap,record["exact_manufactured_prior_periodicity_relative_gap"]):
                    local_failures.append("Statistical joint-translation value-gap reduction disagrees")
                expected_translations=(arrays["base_queries"][None]+arrays["offsets"][:,None]).reshape(-1,3)
                if not bool(scalar(jnp.all(jnp.abs(arrays["queries"][count:legacy_count]-expected_translations)<=1e-12))):
                    local_failures.append("Saved joint translations do not have their declared coordinates")
                if not same(divergence_gap, record["local_normalized_divergence_gap"]):
                    local_failures.append("Periodic statistical divergence reduction disagrees")
                if not bool(scalar((true_gap <= 1e-10) & (value_gap <= 1e-8) & (jacobian_gap<=1e-8) & (divergence_gap <= 1e-8))):
                    local_failures.append("Independent statistical periodicity/divergence checks failed")
                group_checks=None
                true_jacobian_gap=None
                if "periodic_probe_groups" in record:
                    group_path=(folder/record["periodic_probe_groups_artifact"]).resolve()
                    if not group_path.is_relative_to(folder.resolve()) or digest(group_path)!=record["periodic_probe_groups_sha256"]:
                        raise ValueError("Statistical periodic group metadata pin mismatch")
                    group_metadata=json.loads(group_path.read_text())
                    for name in ("periodic_probe_groups","legacy_translation_query_count","actual_probe_query_count"):
                        if group_metadata[name]!=record[name]:
                            raise ValueError("Statistical periodic group metadata disagrees with result")
                    if legacy_count+record["actual_probe_query_count"]!=len(arrays["queries"]) or len(arrays["boundary_queries"])!=record["actual_probe_query_count"]:
                        raise ValueError("Actual boundary-probe query counts disagree")
                    if not bool(scalar(jnp.all(arrays["boundary_queries"]==arrays["queries"][legacy_count:]))):
                        local_failures.append("Actual boundary-query array disagrees with appended probes")
                    if input_hashes.get("true_jacobian")!=array_hashes.get("truth_jacobian"):
                        local_failures.append("Pinned exact periodic Jacobians disagree with saved truth Jacobians")
                    true_jacobian=arrays["truth_jacobian"]
                    true_shifted_jacobian=true_jacobian[count:legacy_count].reshape(shifted_jacobian.shape)
                    true_jacobian_gap=jnp.max(jnp.linalg.norm(true_shifted_jacobian-true_jacobian[:count][None],axis=(2,3)))
                    true_jacobian_gap/=jnp.maximum(jnp.max(jnp.linalg.norm(true_jacobian[:legacy_count],axis=(1,2))),1.)
                    if not same(jacobian_gap,record["reconstructed_joint_translation_jacobian_periodicity_relative_gap"]) or not same(true_jacobian_gap,record["exact_manufactured_prior_jacobian_periodicity_relative_gap"]):
                        local_failures.append("Statistical joint-translation Jacobian-gap reduction disagrees")
                    group_checks,group_geometry=statistical_periodic_groups(arrays,record["periodic_probe_groups"])
                    if not bool(scalar(group_geometry)):
                        local_failures.append("Actual face/edge/corner or wrapped-probe coordinates failed")
                    for kind,values in group_checks.items():
                        saved=record["periodic_group_checks"][kind]
                        if saved["pairs"]!=values["pairs"]:
                            local_failures.append(f"Periodic statistical group pair count disagrees: {kind}")
                        for field in ("reconstructed_value_relative_gap","reconstructed_jacobian_relative_gap","exact_value_relative_gap","exact_jacobian_relative_gap"):
                            if not same(values[field],saved[field]):
                                local_failures.append(f"Periodic statistical group reduction disagrees: {kind}/{field}")
                        if not bool(scalar(values["passed"])) or saved["passed"]!=bool(scalar(values["passed"])):
                            local_failures.append(f"Independent periodic statistical group failed: {kind}")
                    boundary_value=jnp.max(jnp.stack([group_checks[kind]["reconstructed_value_relative_gap"] for kind in ("face","edge","corner")]))
                    boundary_jacobian=jnp.max(jnp.stack([group_checks[kind]["reconstructed_jacobian_relative_gap"] for kind in ("face","edge","corner")]))
                    if not same(boundary_value,record["reconstructed_face_edge_corner_periodicity_relative_gap"]) or not same(boundary_jacobian,record["reconstructed_face_edge_corner_jacobian_periodicity_relative_gap"]):
                        local_failures.append("Actual statistical boundary value/Jacobian maxima disagree")
                    if not bool(scalar(true_jacobian_gap<=1e-10)):
                        local_failures.append("Exact statistical joint-translation Jacobians failed")
                else:
                    local_failures.append("Actual boundary identifications and truth Jacobians are missing; legacy values verify joint translations only")
                if not record["passed"]:
                    local_failures.append("Saved statistical periodicity checks failed")
                periodic.append(dict(case=case, verification_passed=not local_failures,
                    verification_failures=local_failures, periodic_relative_gap=value_gap,
                    joint_translation_relative_gap=value_gap,joint_translation_jacobian_relative_gap=jacobian_gap,
                    exact_joint_translation_jacobian_relative_gap=true_jacobian_gap,
                    actual_boundary_groups=group_checks,legacy_field_scope="periodic_relative_gap denotes joint translations; actual boundary groups are reported separately",
                    maximum_actual_identification_value_gap=(jnp.max(jnp.stack([
                        values["reconstructed_value_relative_gap"] for values in group_checks.values()]))
                        if group_checks is not None else jnp.nan),
                    maximum_actual_identification_jacobian_gap=(jnp.max(jnp.stack([
                        values["reconstructed_jacobian_relative_gap"] for values in group_checks.values()]))
                        if group_checks is not None else jnp.nan),
                    divergence_relative_gap=divergence_gap, independent_spatial_checks=spatial_checks,saved_result=record))
                failures += [f"{case}: {x}" for x in local_failures]
                continue
            radius = arrays["radius"]
            bins, count = len(radius), len(arrays["query_y"])
            directions = count//bins
            if bins*directions != count:
                raise ValueError("Statistical pair count is incompatible with the radius bins")
            phase = (arrays["query_z"]-arrays["query_y"]) @ reference["wavevectors"].T
            exact_pair = 8*(jnp.sin(phase/2)**2 @ reference["weights"])
            saved_pair = arrays["pair_reference_s2"]
            if not same(exact_pair, saved_pair):
                local_failures.append("Exact Fourier increment identity failed on saved statistical pairs")
            energy = covariance_energies(reference, inputs, diagnostic_arrays,
                arrays["query_y"], arrays["query_z"])
            for column, key in enumerate(("pair_reconstructed_s2", "pair_reference_s2",
                "pair_pointwise_mse_y", "pair_pointwise_mse_z")):
                if not same(energy[:, column], arrays[key], rtol=2e-9, atol=2e-12):
                    local_failures.append(f"Saved Fourier-map covariance contraction disagrees: {key}")
            if "pair_increment_mse" in arrays and not same(energy[:, 4],
                    arrays["pair_increment_mse"], rtol=2e-9, atol=2e-12):
                local_failures.append("Saved Fourier-map covariance contraction disagrees: pair_increment_mse")
            predicted = energy[:, 0]
            truth = jnp.mean(exact_pair.reshape(bins, directions), axis=1)
            reconstructed = jnp.mean(predicted.reshape(bins, directions), axis=1)
            if not same(reconstructed, arrays["reconstructed_s2"]) or not same(truth, arrays["reference_s2"]):
                local_failures.append("Saved mean structure functions disagree with pair reductions")
            ratio = predicted/exact_pair
            amplitude_error = jnp.abs(jnp.sqrt(ratio)-1)
            mse_y, mse_z = energy[:, 2], energy[:, 3]
            bound = (jnp.sqrt(mse_y)+jnp.sqrt(mse_z))/jnp.sqrt(exact_pair)
            sharp = jnp.sqrt(energy[:, 4]/exact_pair)
            bound_holds = jnp.all((amplitude_error <= sharp+1e-10) & (sharp <= bound+1e-10))
            if not bool(scalar(bound_holds)):
                local_failures.append("Saved covariance quantities violate the deterministic MSE bound")
            mask = reference["certified_interval"]
            if not same(radius, reference["radius"]):
                local_failures.append("Statistics radius bins changed from the certified reference")
            selected_pairs = jnp.repeat(mask, directions)
            has_interval = jnp.any(mask)
            epsilon = jnp.asarray(record["declared_increment_amplitude_tolerance"], dtype=jnp.float64)
            criterion = jnp.all(jnp.where(selected_pairs, bound <= epsilon, True)) & has_interval
            slope = jnp.diff(jnp.log(reconstructed))/jnp.diff(jnp.log(radius))
            reference_slope = jnp.diff(jnp.log(truth))/jnp.diff(jnp.log(radius))
            slope_mask = mask[:-1] & mask[1:]
            first = jnp.argmax(mask)
            reference_curve = jnp.where(mask, truth[first]*(radius/radius[first])**(2/3), jnp.nan)
            py, pz = arrays["manufactured_prediction_y"], arrays["manufactured_prediction_z"]
            ty, tz = arrays["manufactured_truth_y"], arrays["manufactured_truth_z"]
            velocity_error = jnp.sqrt((jnp.sum((py-ty)**2)+jnp.sum((pz-tz)**2))
                / (jnp.sum(ty**2)+jnp.sum(tz**2)))
            if not same(velocity_error, record["manufactured_relative_velocity_error"]):
                local_failures.append("Manufactured velocity RMS disagrees with saved field arrays")
            draw_truth = jnp.mean(jnp.sum((ty-tz)**2, axis=-1).reshape(bins, directions, -1), axis=1)
            draw_prediction = jnp.mean(jnp.sum((py-pz)**2, axis=-1).reshape(bins, directions, -1), axis=1)
            empirical_mean = jnp.mean(draw_prediction, axis=1)
            empirical_se = jnp.std(draw_prediction, axis=1, ddof=1)/jnp.sqrt(draw_prediction.shape[1])
            if not same(empirical_mean, arrays["empirical_reconstructed_mean"]):
                local_failures.append("Manufactured structure-function mean disagrees with saved field arrays")
            if not same(empirical_se, arrays["empirical_reconstructed_se"]):
                local_failures.append("Manufactured descriptive standard error disagrees with saved field arrays")
            if not record["numerical_checks"]["numerical_checks_passed"]:
                local_failures.append("Saved statistical spatial numerical checks failed")
            rows.append(dict(case=case, order=record["value_order"], policy=record["policy"],
                spacing=record["spacing"], nodes=record["nodes"],
                verification_passed=not local_failures, verification_failures=local_failures,
                numerical_checks_passed=record["numerical_checks"]["numerical_checks_passed"],
                independent_spatial_checks=spatial_checks,
                radius=radius, reference_s2=truth, reconstructed_s2=reconstructed,
                reference_two_thirds=reference_curve, certified_interval=mask,
                slope_radius=jnp.sqrt(radius[:-1]*radius[1:]), slope=slope,
                reference_slope=reference_slope, slope_interval=slope_mask,
                certified_slope=jnp.where(slope_mask, slope, jnp.nan),
                certified_reference_slope=jnp.where(slope_mask, reference_slope, jnp.nan),
                maximum_certified_slope_discrepancy=jnp.where(jnp.any(slope_mask),
                    jnp.max(jnp.where(slope_mask, jnp.abs(slope-reference_slope), 0.)), jnp.nan),
                maximum_certified_per_pair_relative_s2_error=jnp.where(has_interval,
                    jnp.max(jnp.where(selected_pairs,jnp.abs(ratio-1.),0.)),jnp.nan),
                maximum_certified_per_pair_relative_increment_amplitude_error=jnp.where(has_interval,
                    jnp.max(jnp.where(selected_pairs,amplitude_error,0.)),jnp.nan),
                maximum_certified_mean_direction_slope_deviation_from_two_thirds=jnp.where(jnp.any(slope_mask),
                    jnp.max(jnp.where(slope_mask,jnp.abs(slope-2/3),0.)),jnp.nan),
                maximum_certified_per_pair_mse_amplitude_bound=jnp.where(has_interval,
                    jnp.max(jnp.where(selected_pairs,bound,0.)),jnp.nan),
                certified_mse_amplitude_criterion_met=criterion,
                certified_interval_available=has_interval,
                mse_amplitude_tolerance=epsilon, mse_implied_relative_s2_tolerance=2*epsilon+epsilon**2,
                covariance_mse_bound_holds=bound_holds,
                manufactured_relative_velocity_error=velocity_error,
                empirical_mean=empirical_mean, empirical_two_se=2*empirical_se,
                empirical_reference_mean=jnp.mean(draw_truth, axis=1),
                statistics_npz_sha256=record["statistics_npz_sha256"]))
        except Exception as error:
            local_failures.append(f"{type(error).__name__}: {error}")
        failures += [f"{case}: {x}" for x in local_failures]
    for path in sorted(folder.rglob("*failure*.json")):
        failures_saved.append(dict(path=str(path.relative_to(folder)), sha256=digest(path),
            evidence=json.loads(path.read_text())))
    expected = protocol["expected_statistical_cells"]
    if len(rows) != expected:
        failures.append(f"Expected {expected} statistical cells; found {len(rows)} authenticated/readable records")
    expected_periodic = protocol.get("expected_periodic_statistical_cells", 2*len(protocol["orders"]))
    if len(periodic) != expected_periodic:
        failures.append(f"Expected {expected_periodic} periodic statistical cells; found {len(periodic)}")
    completion_path = folder/"completion.json"
    if completion_path.exists():
        completion = json.loads(completion_path.read_text())
        if completion["protocol_sha256"] != protocol_hash:
            failures.append("Statistics completion protocol hash disagrees")
        for relative, expected_digest in completion["result_sha256"].items():
            path = folder/relative
            if not path.is_file() or digest(path) != expected_digest:
                failures.append(f"Statistics completion result hash disagrees: {relative}")
    else:
        failures.append("Statistics completion artifact is missing")
    return dict(protocol_sha256=protocol_hash, source_verification=sources,
        reference_npz_sha256=protocol["reference_npz_sha256"], expected_cells=expected,
        complete=not failures,
        verified_cells=sum(r["verification_passed"] for r in rows), cells=rows, periodicity=periodic,
        failures=failures, saved_execution_failures=failures_saved,
        reference_interval_resolved=protocol["reference_two_thirds_interval_resolved"],
        certified_radius_interval=dict(
            start=jnp.where(jnp.any(reference["certified_interval"]),jnp.min(jnp.where(reference["certified_interval"],reference["radius"],jnp.inf)),jnp.nan),
            end=jnp.where(jnp.any(reference["certified_interval"]),jnp.max(jnp.where(reference["certified_interval"],reference["radius"],-jnp.inf)),jnp.nan),
            scope="The fixed finite-band interval was certified using the exact reference before evaluating reconstruction outcomes"),
        outcome_scopes=dict(relative_s2_error="Maximum per-pair |E_rec/E_ref-1| over pairs in the fixed truth-certified radius interval",
            relative_increment_amplitude_error="Maximum per-pair |sqrt(E_rec/E_ref)-1| on the same fixed interval",
            slope_deviation="Maximum |slope-2/3| for adjacent certified bins of direction-mean reconstructed S2; this is not a per-pair slope",
            mse_amplitude_bound="Maximum per-pair (sqrt(e_y)+sqrt(e_z))/sqrt(E_ref) on that same fixed interval; criterion retains its declared epsilon",
            sufficient_certificate="A met MSE-bound criterion certifies relative increment-amplitude error <= epsilon and relative S2 error <= 2*epsilon+epsilon^2. A criterion not met does not itself establish measured amplitude failure.",
            acceptance="Complete/verified means authenticated, complete and numerically consistent calculations. Measured amplitude/slope outcomes and the sufficient MSE criterion do not determine completion. No reconstruction-slope pass threshold is imposed."),
        qualification="A finite positive periodic Fourier covariance supplies the exact reference. "
            "A resolved finite two-thirds-like interval is not an asymptotic Kolmogorov law or proof of continuous isotropy. "
            "All spatial reconstructions use the same assigned-node stencils: boundary-aware CFPU selection on the irregular domain and periodic minimum images on the torus. The fixed physical band is prior-informed; "
            "it is not learned stencil selection. MSE bounds control increment amplitudes, not log slopes. "
            "Eight-draw bands are descriptive, not rigorous confidence intervals. Restricting fields to the irregular domain imposes no wall condition.")


def proof_metadata(path, source_check, manifest_path=None):
    proof = dict(provided=path is not None,
        qualification="Source pins verify the saved test code. Supplied JUnit records its own outcome; "
            "without a contemporaneous source pin it does not establish which revision executed.")
    if path is None:
        return proof
    tree = ET.parse(path)
    cases = list(tree.getroot().iter("testcase"))
    proof.update(path=str(path), sha256=digest(path), tests=len(cases),
        failures=sum(c.find("failure") is not None or c.find("error") is not None for c in cases),
        skipped=sum(c.find("skipped") is not None for c in cases),
        test_source_sha256=(source_check.get("manifest") or {}).get("source_sha256", {}).get(
            "tests/test_divfree_center_limits.py"))
    proof["all_thirteen_passed_without_skips"] = (proof["tests"] == 13
        and proof["failures"] == 0 and proof["skipped"] == 0)
    required={"test_divfree_center_limits":13,"test_jax_spatial_poisson_periodic":26,
        "test_geometry_device_pipeline":8,"test_cfpu_device_build":4,
        "test_domain_stencil_cache_invalidation":11}
    modules={name:[case for case in cases if name in case.get("classname","").split(".")]
        for name in required}
    proof["required_focused_suites"]={name:dict(expected=expected,observed=len(modules[name]))
        for name,expected in required.items()}
    identities=[(case.get("classname"),case.get("name")) for case in cases]
    checks=dict(all_five_complete_suites=all(len(modules[name])==expected for name,expected in required.items()),
        no_failures_or_skips=proof["failures"]==0 and proof["skipped"]==0,
        no_duplicate_testcase_ids=len(identities)==len(set(identities)))
    frozen_pins=(source_check.get("manifest") or {}).get("source_sha256",{})
    proof["focused_test_source_sha256"]={f"tests/{name}.py":frozen_pins.get(f"tests/{name}.py") for name in required}
    checks["all_test_sources_frozen"]=all(value is not None for value in proof["focused_test_source_sha256"].values())
    checks["execution_manifest_provided"]=manifest_path is not None
    if manifest_path is not None:
        manifest=json.loads(manifest_path.read_text())
        proof.update(execution_manifest_path=str(manifest_path),execution_manifest_sha256=digest(manifest_path),
            execution_manifest=manifest)
        checks.update(
            before_sources_match_freeze=manifest.get("source_before_sha256")==frozen_pins,
            after_sources_match_freeze=manifest.get("source_after_sha256")==frozen_pins,
            junit_pin_matches=manifest.get("junit_sha256")==proof["sha256"],
            cuda_float64_execution=manifest.get("platform")=="cuda" and manifest.get("dtype")=="float64",
            execution_returncode_zero=manifest.get("returncode")==0)
    proof["focused_acceptance_checks"]=checks
    proof["all_focused_gpu_checks_accepted"]=all(checks.values())
    return proof


def write_markdown(report, path):
    lines = ["# Spatial reconstruction study", "",
        f"Verified cell artifacts: {report['verified_cells']}/60. Scientific checks passed: {report['scientific_checks_passed']}/60.",
        f"Full study complete: {report['complete']}.", "",
        "Errors use the saved independent queries. Primary rates are least-squares log–log slopes against the predeclared sampled fill distance. Rates against the prescribed node spacing are additional diagnostics of cloud-to-cloud variation; neither estimate is a convergence theorem.", "",
        "| Dimension | Kernel | Target order | Four-level rate: sampled h | Last-three rate: sampled h | Four-level rate: nominal spacing | Last-three rate: nominal spacing | Checked |",
        "|---|---|---:|---:|---:|---:|---:|---|"]
    supplement=report.get("oracle_recheck")
    if supplement:
        introduction=[
            f"Original saved pass flags: {report['original_saved_checks_passed']}/60; original flags and the `complete` status above are unchanged.",
            f"Supplementary oracle evidence mode: {supplement['mode']}; accepted: {supplement['accepted']}.",
            f"Spatial evidence complete with the explicit supplement: {report['spatial_complete_with_supplemental_evidence']}; combined report complete with it: {report['complete_with_supplemental_evidence']}.",
            "Certificate-only evidence analyzes saved actual nodal-map residuals and a float64 contraction bound. Full-rebuild evidence measures independently generated polynomial fields with corrected input scaling. These are distinct forms of evidence.",
            ""]
        lines[4:4]=introduction
    for curve in report["convergence"]:
        fmt = lambda value: "—" if value is None else f"{value:.3f}"
        lines.append(f"| {curve['dimension']}D | {curve['kernel']} + polynomial | {curve['order']} | "
            f"{fmt(curve['all_four_rate'])} | {fmt(curve['last_three_rate'])} | "
            f"{fmt(curve['all_four_nominal_spacing_rate'])} | {fmt(curve['last_three_nominal_spacing_rate'])} | {curve['checked']} |")
    lines += ["", "## Spatial cloud geometry", "",
        "M counts physical spatial nodes. Each cloud appears once, shared by all orders and kernels. The separation is the saved minimum pair distance, with q equal to half that distance. The sampled h is the maximum nearest-node distance on the fixed independent query set; h/q is a sampled ratio, not a certified global mesh ratio.", "",
        "| Domain | Dimension | M | Nominal spacing | Sampled h | Minimum separation | Sampled h/q |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    geometry_number = lambda value: "—" if value is None else f"{value:.5e}"
    for cloud in report["cloud_geometry"]:
        lines.append(f"| {cloud['suite']} | {cloud['dimension']}D | {cloud['spatial_node_count_M']} | "
            f"{geometry_number(cloud['nominal_spacing'])} | {geometry_number(cloud['sampled_fill_distance'])} | "
            f"{geometry_number(cloud['minimum_node_separation'])} | "
            f"{geometry_number(cloud['sampled_fill_to_half_separation_ratio'])} |")
    if report["aggregate"]:
        aggregate = report["aggregate"]
        lines += ["", f"Maximum independently checked normalized RMS divergence: {aggregate['maximum_normalized_rms_divergence']:.3e}."]
        if aggregate["maximum_periodic_value_gap"] is not None:
            lines += [f"Maximum periodic value gap: {aggregate['maximum_periodic_value_gap']:.3e}; maximum Jacobian gap: {aggregate['maximum_periodic_jacobian_gap']:.3e}."]
    lines += ["", "Local divergence is checked with each assigned stencil held fixed. Cubic kernel centers are excluded from classical derivative claims. The construction does not enforce continuity across stencil interfaces or a complete boundary trace.", "",
        "The shared boundary-aware selector records unreachable Euclidean fill separately from a linear-solver fallback. This study requires zero such unreachable fills, and independently checks admission of every node by the CFPU visibility layer.", "",
        "Reference curves have slopes 2, 4 and 6 and pass through each curve's finest available point. They are visual references.", "",
        "Warm timings cover application of prepared maps; preparation includes its first compilation. Array accounting is not peak device memory."]
    if report["failures"]:
        lines += ["", "## Original failed or missing evidence (retained)", ""]
        lines += [f"- {failure}" for failure in report["failures"]]
    if supplement:
        lines += ["", "## Supplementary nodal-oracle evidence", "", supplement["qualification"]]
        lines += [f"- {cell['cell']}: {cell['classification']}; supplemental acceptance: {cell['accepted']}."
                  for cell in supplement["cells"]]
        lines += [f"- Unresolved supplement: {failure}" for failure in supplement["failures"]]
    stats = report.get("statistics")
    if stats:
        lines += ["", "## Increment statistics", "", stats["qualification"],
            f"Verified statistical cells: {stats['verified_cells']}/{stats['expected_cells']}."]
        interval=stats["certified_radius_interval"]
        number=lambda value:"—" if value is None else f"{value:.3e}"
        interval_available=(stats["reference_interval_resolved"] and interval["start"] is not None
            and interval["end"] is not None)
        if interval_available:
            lines += ["",f"The reference-certified finite interval is r ∈ [{number(interval['start'])}, {number(interval['end'])}]. It is fixed from the exact reference, independently of reconstruction outcomes."]
        else:
            lines += ["", "The exact reference did not resolve the prescribed finite two-thirds-like interval. Interval-specific amplitude, slope and certificate outcomes are unavailable; this is not a measured failure of a reconstruction on a certified interval. Full structure-function curves and numerical verification remain available in the JSON report."]
        lines += ["The S₂ and increment-amplitude errors below are maxima over individual pairs in that interval. The slope deviation is the maximum departure from 2/3 of the direction-mean reconstructed S₂ slope between adjacent certified radius bins. No reconstruction-slope pass threshold is imposed. These quantities do not assert an asymptotic Kolmogorov law.",
            "The MSE criterion is a sufficient amplitude certificate: its bound must be at most the declared ε. A certificate not met does not itself show measured amplitude failure. Verified completion concerns numerical validity and complete evidence; large amplitude or slope errors remain reported scientific outcomes.", "",
            "| M | Target order | Spatial scales | Max pair relative S₂ error | Max pair relative amplitude error | Max mean-slope deviation from 2/3 | Max pair MSE amplitude bound | Sufficient certificate |",
            "|---:|---:|---|---:|---:|---:|---:|---|"]
        policies={"phs":"PHS + polynomial", "ram_local_tau16":"Matérn, stencil-relative scales (τ=16)",
            "fixed_physical_band":"Matérn, fixed physical scales"}
        for cell in sorted(stats["cells"],key=lambda row:(row["nodes"],row["order"],row["policy"])):
            available=interval_available and cell["certified_interval_available"]
            criterion=("met" if cell["certified_mse_amplitude_criterion_met"] else "not met") if available else "unavailable"
            interval_number=lambda value:number(value) if available else "—"
            if not cell["verification_passed"]:
                criterion+="; unverified"
            lines.append(f"| {cell['nodes']} | {cell['order']} | {policies.get(cell['policy'],cell['policy'].replace('_',' '))} | "
                f"{interval_number(cell['maximum_certified_per_pair_relative_s2_error'])} | "
                f"{interval_number(cell['maximum_certified_per_pair_relative_increment_amplitude_error'])} | "
                f"{interval_number(cell['maximum_certified_mean_direction_slope_deviation_from_two_thirds'])} | "
                f"{interval_number(cell['maximum_certified_per_pair_mse_amplitude_bound'])} | {criterion} (ε={number(cell['mse_amplitude_tolerance'])}) |")
        lines += ["", "Periodic manufactured-prior checks for the same three spatial reconstructions:", "",
            "| Target order | Reconstruction | Max periodic value gap | Max periodic Jacobian gap | Normalized divergence gap | Verified |",
            "|---:|---|---:|---:|---:|---|"]
        for cell in sorted(stats["periodicity"],key=lambda row:(row["saved_result"]["value_order"],row["saved_result"]["policy"])):
            saved=cell["saved_result"]
            lines.append(f"| {saved['value_order']} | {policies.get(saved['policy'],saved['policy'].replace('_',' '))} | "
                f"{number(cell['maximum_actual_identification_value_gap'])} | "
                f"{number(cell['maximum_actual_identification_jacobian_gap'])} | "
                f"{number(cell['divergence_relative_gap'])} | {'yes' if cell['verification_passed'] else 'no'} |")
        if stats["failures"]:
            lines.append("")
        lines += [f"- {failure}" for failure in stats["failures"]]
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--statistics", type=Path)
    parser.add_argument("--statistics-source-root", type=Path)
    parser.add_argument("--proof-xml", type=Path)
    parser.add_argument("--proof-manifest", type=Path,
        help="Pinned source-before/source-after, actual CUDA64 runtime and JUnit execution metadata")
    parser.add_argument("--oracle-recheck", type=Path,
        help="Authenticate supplementary nodal-oracle evidence while retaining all original flags")
    parser.add_argument("--gpu-lock-file", type=Path)
    args = parser.parse_args()
    if args.statistics and args.statistics_source_root is None:
        parser.error("--statistics requires --statistics-source-root")
    if args.proof_manifest and args.proof_xml is None:
        parser.error("--proof-manifest requires --proof-xml")
    output = args.output or args.study.parent/"report"
    if output.exists():
        parser.error(f"Choose a new report folder; existing reports are preserved: {output}")
    lock = args.gpu_lock_file or args.study.parent/".fastol_spatial_gpu.lock"
    if lock.exists():
        parser.error(f"GPU study lock exists; do not overlap numerical processes: {lock}")
    device = jax.devices("gpu")[0]
    if device.platform != "gpu" or "nvidia" not in device.device_kind.lower() or not jax.config.x64_enabled:
        raise RuntimeError("NVIDIA CUDA float64 is required; no CPU analysis fallback")
    jax.config.update("jax_default_device", device)
    protocol = json.loads((args.study/"protocol.json").read_text())
    protocol_hash = hashlib.sha256(canonical(protocol)).hexdigest()
    source_check = verify_sources(args.source_root, protocol["source_sha256"])
    failures = list(source_check["failures"])
    if protocol.get("schema") == "fastol_spatial_properties_scattered_v2":
        required = dict(interior_queries_2d=4096,interior_queries_3d=32768,
            boundary_queries_2d=512,boundary_queries_3d=2048,batch_size=32,warm_repetitions=11,
            irregular_spacing_2d=[.103,.0675,.0445,.0295],irregular_spacing_3d=[.22,.16,.118,.088],
            periodic_grid_2d=32,periodic_grid_3d=12)
        for name,value in required.items():
            if protocol.get(name) != value:
                failures.append("The full scattered convergence/property protocol changed: " + name)
    if protocol.get("configuration_count") != 60:
        failures.append("The saved protocol is not the complete prescribed 60-cell study")
    expected, seen, rows = expected_cells(), set(), []
    for path in sorted((args.study/"cells").glob("*/result.json")):
        key = path.parent.name
        if key in seen or key not in expected:
            failures.append(f"Unexpected or duplicate cell: {key}")
            continue
        seen.add(key)
        try:
            row = verify_cell(path, protocol, protocol_hash, device)
            if row["cell"] != key:
                raise ValueError("Cell identity disagrees with folder")
            rows.append(row)
            failures += [f"{key}: {x}" for x in row["verification_failures"]]
            failures += [f"{key}: scientific check failed: {x}" for x in row["failed_scientific_checks"]]
        except Exception as error:
            failures.append(f"{key}: {type(error).__name__}: {error}")
    missing = sorted(expected-seen)
    failures += [f"Missing completed cell: {key}" for key in missing]
    saved_failures = []
    for path in sorted(args.study.rglob("*failure*.json")):
        saved_failures.append(dict(path=str(path.relative_to(args.study)), sha256=digest(path),
            evidence=json.loads(path.read_text())))
    supplement=None
    if args.oracle_recheck:
        supplement=oracle_recheck_report(args.oracle_recheck,args.study,args.source_root,
            protocol,protocol_hash,rows,source_check,device)
    covered=set()
    if supplement and supplement["accepted"]:
        for cell in supplement["cells"]:
            covered.update(f"{cell['cell']}: scientific check failed: {flag}" for flag in
                (NODAL_ORACLE_FLAG,"independent:independent_nodal_requested_polynomial"))
        by_cell={cell["cell"]:cell for cell in supplement["cells"]}
        for row in rows:
            if row["cell"] in by_cell:
                row["supplemental_oracle_evidence"]=dict(mode=supplement["mode"],accepted=True,
                    classification=by_cell[row["cell"]]["classification"],original_flags_unchanged=True)
    supplemental_complete=(len(rows)==60 and bool(supplement and supplement["accepted"])
        and not [failure for failure in failures if failure not in covered])
    proof=proof_metadata(args.proof_xml,source_check,args.proof_manifest)
    if protocol.get("schema")=="fastol_spatial_properties_scattered_v2" and not proof.get("all_focused_gpu_checks_accepted",False):
        failures.append("The full five-suite CUDA float64 regression proof is missing or unverified")
    curves = convergence_curves(rows)
    periodic_gaps = [g for r in rows for g in r["periodicity"].values()]
    aggregate = None if not rows else dict(
        maximum_normalized_rms_divergence=jnp.max(jnp.stack([r["errors"]["divergence_normalized_rms"] for r in rows])),
        maximum_periodic_value_gap=(jnp.max(jnp.stack([g["value_maximum"] for g in periodic_gaps])) if periodic_gaps else None),
        maximum_periodic_jacobian_gap=(jnp.max(jnp.stack([g["jacobian_maximum"] for g in periodic_gaps])) if periodic_gaps else None))
    report = dict(schema="fastol_spatial_report_v1", created_utc=datetime.now(timezone.utc).isoformat(),
        reporter_source_sha256=digest(Path(__file__)),
        study=str(args.study.resolve()), protocol_sha256=protocol_hash,
        protocol_file_sha256=digest(args.study/"protocol.json"), source_verification=source_check,
        proof=proof, device=str(device), precision="float64",
        spatial_complete=len(rows) == 60 and not failures,
        complete=len(rows) == 60 and not failures, partial=args.partial,
        spatial_complete_with_supplemental_evidence=supplemental_complete,
        complete_with_supplemental_evidence=supplemental_complete,
        original_saved_checks_passed=sum(r["original_passed"] for r in rows),
        verified_cells=sum(r["verification_passed"] for r in rows),
        scientific_checks_passed=sum(r["scientific_checks_passed"] for r in rows),
        failures=failures, missing_cells=missing, saved_execution_failures=saved_failures,
        cells=rows, convergence=curves, cloud_geometry=cloud_geometry_table(rows),
        cloud_geometry_scope="Saved physical-node separation and independently sampled fill distance; sampled h/q is not a certified global mesh ratio.",
        aggregate=aggregate,
        periodicity=[r for r in rows if r["suite"] == "periodic"],
        periodicity_plot=periodic_plot_arrays(rows),
        oracle_recheck=supplement,
        labels=dict(h="Sampled fill distance, h", error="Relative RMS velocity error",
            divergence="Normalized RMS divergence", boundary="Boundary RMS velocity error",
            phs="PHS + polynomial", matern="Multiscale Matérn + polynomial"))
    if args.statistics:
        report["statistics"] = statistics_report(args.statistics, args.statistics_source_root, device)
        report["complete"] = report["spatial_complete"] and report["statistics"]["complete"]
        report["complete_with_supplemental_evidence"] = supplemental_complete and report["statistics"]["complete"]
    report = exported(report)
    output.mkdir(parents=True, exist_ok=False)
    (output/"spatial_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    write_markdown(report, output/"spatial_report.md")
    print(json.dumps(dict(report=str(output), complete=report["complete"],
        complete_with_supplemental_evidence=report["complete_with_supplemental_evidence"],
        supplemental_oracle_accepted=bool(report["oracle_recheck"] and report["oracle_recheck"]["accepted"]),
        verified_cells=report["verified_cells"], scientific_checks_passed=report["scientific_checks_passed"])))
    if not args.partial and not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
