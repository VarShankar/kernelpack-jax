"""Strictly accept one completed FastOL cell and refresh finite scalar tables.

Standard library only. Never initializes CUDA/JAX, reads numerical arrays,
launches jobs, watches, polls, modifies production code or campaign artifacts.
Previous acceptance records reuse their full audits only while artifact hashes
and file identity metadata remain consistent. Run once per completed cell.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys


AUDIT_SCHEMA = "matched_gpu_one_cell_audit_v1"
MODELS = ("fixed_affine", "trainable_affine", "fixed_no_poly", "trainable_no_poly")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf8")).hexdigest()


def read_json(path):
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def local_path(name):
    if sys.platform == "win32" and name.startswith("/mnt/"):
        parts = name.split("/")
        return Path(parts[2].upper() + ":/" + "/".join(parts[3:]))
    return Path(name)


def stamp(path):
    value = path.stat()
    return {"size": value.st_size, "mtime_ns": value.st_mtime_ns}


def atomic_json(path, value):
    temporary = path.with_name(path.name + "." + str(os.getpid()) + ".tmp")
    with temporary.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def immutable_json(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_and_import_frozen_validator(root, manifest):
    snapshot = root / "source_snapshot"
    expected = manifest["runtime"]["source_sha256"]
    if not expected:
        raise RuntimeError("The campaign has no frozen source/native manifest")
    for name, checksum in expected.items():
        path = (snapshot / name).resolve()
        if not path.is_relative_to(snapshot.resolve()) or not path.is_file() or sha256(path) != checksum:
            raise RuntimeError("Frozen source/native fingerprint mismatch: " + name)
    validator_path = snapshot / "scripts/run_fastol_gpu_campaign.py"
    spec = importlib.util.spec_from_file_location("fastol_frozen_acceptance", validator_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # This additionally rejects an unmanifested production Python file.
    module.verify_snapshot(snapshot, expected)
    if tuple(module.MODELS) != MODELS:
        raise RuntimeError("Unexpected four-model comparison in the frozen validator")
    return module, expected


def fingerprint_record(path, checksum, *, relative_to=None):
    return {"path": path.relative_to(relative_to).as_posix() if relative_to else str(path),
        "sha256": checksum, **stamp(path)}


def capture_boundary_stamps(root, folder):
    """Reject changes racing the full audit and its later cache metadata."""
    protocol = read_json(folder / "protocol.json")
    result = read_json(folder / "results.json")
    external = {**protocol["source_files"]}
    external.update({protocol[path_key]: protocol[hash_key] for path_key, hash_key in (
        ("split_file", "split_sha256"), ("selection_csv", "selection_csv_sha256"),
        ("baseline_csv", "baseline_csv_sha256"))})
    paths = [local_path(name) for name in external]
    paths.extend(folder / name for name in ("results.json", "selection.json", "protocol.json", "shared_geometry.npz"))
    for name in MODELS:
        prediction = (folder / result["models"][name]["prediction_file"]).resolve()
        if not prediction.is_relative_to(folder.resolve()):
            raise RuntimeError("Prediction path escapes its immutable cell")
        paths.extend((prediction, folder / ("selected_geometry_" + name + ".npz")))
    return {path.resolve(): stamp(path) for path in paths}


def make_model_metrics(name, model, choice, n):
    native = model["native_factor_storage"]
    visible = model["counted_storage"]["visible_model_array_bytes"]
    owned = native["owned_csr_bytes"]
    graph = native["graph_dense_buffer_bytes"]
    if owned != 4 * (n + 1) + 12 * native["actual_nnz"]:
        raise RuntimeError("Native immutable float64 CSR allocation formula differs: " + name)
    if graph != 2 * n * choice["config"]["rhs_block_size"] * 8:
        raise RuntimeError("Native RHS/solution capacity allocation formula differs: " + name)
    durations = model["warm_prediction_seconds"]
    seconds = choice["training_validation_seconds"]
    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError("Positive training/validation timing is missing: " + name)
    estimate = visible + native["permanent_device_bytes"] + owned + graph
    return {"model": name, "validation_error": choice["validation_error"],
        "test_error": model["test_relative_field_error"],
        "kernel_density": native["actual_nnz"] / n**2,
        "actual_kernel_nnz": native["actual_nnz"], "native_factor_nnz": native["factor_nnz"],
        "mean_dimension_requested": choice["mean_dimension"],
        "polynomial_columns": choice["polynomial_columns"],
        "mean_basis_sha256": choice["mean_basis_sha256"],
        "visible_model_array_bytes": visible,
        "native_permanent_device_ESTIMATE_bytes": native["permanent_device_bytes"],
        "native_peak_device_ESTIMATE_bytes": native["peak_device_bytes"],
        "native_owned_CSR_requested_bytes": owned,
        "native_RHS_solution_requested_bytes": graph,
        "accounted_resident_storage_ESTIMATE_bytes": estimate,
        "accounted_resident_storage_ESTIMATE_decimal_GB": estimate / 10**9,
        "padded_JAX_entry_bytes_already_in_visible": native["padded_jax_entry_bytes"],
        "warm_100query_seconds": durations,
        "warm_100query_min_seconds": min(durations),
        "warm_100query_median_seconds": statistics.median(durations),
        "warm_100query_max_seconds": max(durations),
        "training_validation_seconds": seconds, "training_validation_minutes": seconds / 60,
        "operator_setup_seconds": model["operator_setup_seconds"],
        "predictor_preparation_seconds": model["preparation_seconds"],
        "first_prediction_compile_capture_seconds": model["first_prediction_compile_capture_seconds"],
        "accuracy_checks": model["accuracy_checks"],
        "gpu_transfer_guard_passed": model["gpu_transfer_guard_passed"],
        "graph_replays_during_timing": model["graph_replays_during_timing"],
        "observed_cudss_reordering": native["reordering"],
        "selected_model_sha256": model["selected_model_sha256"],
        "prediction_sha256": model["prediction_sha256"],
        "geometry_sha256": model["geometry_sha256"]}


def build_acceptance(root, cell, manifest, expected, validated):
    folder = root / "cells" / cell["key"]
    results = read_json(folder / "results.json")
    selection = read_json(folder / "selection.json")
    protocol = read_json(folder / "protocol.json")
    artifacts = [fingerprint_record(folder / name, sha256(folder / name), relative_to=root)
        for name in ("results.json", "selection.json", "protocol.json")]
    cached = [fingerprint_record(folder / "shared_geometry.npz", selection["shared_geometry_sha256"], relative_to=root)]
    for name in MODELS:
        model, choice = results["models"][name], selection["models"][name]
        cached.extend((fingerprint_record(folder / model["prediction_file"], model["prediction_sha256"], relative_to=root),
            fingerprint_record(folder / ("selected_geometry_" + name + ".npz"), choice["geometry_sha256"], relative_to=root)))
    external = {**protocol["source_files"]}
    external.update({protocol[path_key]: protocol[hash_key] for path_key, hash_key in (
        ("split_file", "split_sha256"), ("selection_csv", "selection_csv_sha256"),
        ("baseline_csv", "baseline_csv_sha256"))})
    inputs = [fingerprint_record(local_path(name), checksum) for name, checksum in external.items()]
    # Protect against a mutation racing the just-finished full validation.
    if artifacts[0]["sha256"] != validated["results_sha256"] or artifacts[1]["sha256"] != validated["selection_sha256"]:
        raise RuntimeError("A result/selection changed while recording its full audit")
    return {"audit_schema": AUDIT_SCHEMA, "accepted_utc": utc_now(),
        "cell": cell, "full_validation_performed": True,
        "campaign_manifest_sha256": sha256(root / "manifest.json"),
        "source_manifest_sha256": canonical_sha256(expected),
        "frozen_source_verified_files": len(expected),
        "validator_sha256": expected["scripts/run_fastol_gpu_campaign.py"],
        "reporting_helper_sha256": sha256(Path(__file__).resolve()),
        "canonical_results_sha256": canonical_sha256(results),
        "canonical_selection_sha256": canonical_sha256(selection),
        "canonical_protocol_sha256": canonical_sha256(protocol),
        "rehashed_on_every_aggregate": artifacts,
        "large_artifacts_full_hash_verified_at_acceptance": cached,
        "inputs_full_hash_verified_at_acceptance": inputs,
        "models": {name: make_model_metrics(name, results["models"][name], selection["models"][name], cell["N"]) for name in MODELS},
        "shared_spatial_preparation_seconds": results["shared_spatial_preparation_seconds"],
        "explicit_response_transformation": {name: selection["models"][name]["config"]["response_scaling"] for name in MODELS},
        "timing_note": results["timing_note"], "storage_note": results["storage_note"]}


def verify_cached_acceptance(record, root, expected, manifest_sha):
    if (record.get("audit_schema") != AUDIT_SCHEMA or record.get("full_validation_performed") is not True
            or record.get("campaign_manifest_sha256") != manifest_sha
            or record.get("source_manifest_sha256") != canonical_sha256(expected)
            or record.get("validator_sha256") != expected["scripts/run_fastol_gpu_campaign.py"]):
        raise RuntimeError("Prior acceptance certificate differs from this frozen campaign")
    for artifact in record["rehashed_on_every_aggregate"]:
        path = (root / artifact["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != artifact["sha256"]:
            raise RuntimeError("Previously accepted result/selection/protocol changed: " + str(path))
    folder = root / "cells" / record["cell"]["key"]
    for filename, key in (("results.json", "canonical_results_sha256"),
            ("selection.json", "canonical_selection_sha256"), ("protocol.json", "canonical_protocol_sha256")):
        if canonical_sha256(read_json(folder / filename)) != record.get(key):
            raise RuntimeError("Previously accepted canonical artifact hash differs: " + filename)
    result, selected = read_json(folder / "results.json"), read_json(folder / "selection.json")
    fresh_metrics = {name: make_model_metrics(name, result["models"][name], selected["models"][name], record["cell"]["N"])
        for name in MODELS}
    if fresh_metrics != record["models"]:
        raise RuntimeError("Cached reporting metrics differ from the accepted source JSON")
    for artifact in record["large_artifacts_full_hash_verified_at_acceptance"]:
        path = (root / artifact["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or stamp(path) != {key: artifact[key] for key in ("size", "mtime_ns")}:
            raise RuntimeError("Previously accepted geometry/prediction file metadata changed; full reaudit required: " + str(path))
    for artifact in record["inputs_full_hash_verified_at_acceptance"]:
        path = local_path(artifact["path"])
        if stamp(path) != {key: artifact[key] for key in ("size", "mtime_ns")}:
            raise RuntimeError("Previously accepted input metadata changed; full reaudit required: " + str(path))


def markdown_table(record):
    labels = {"fixed_affine": "Fixed, affine polynomial", "trainable_affine": "Trainable, affine polynomial",
        "fixed_no_poly": "Fixed, no polynomial", "trainable_no_poly": "Trainable, no polynomial"}
    lines = ["| Model | Test error | Density | Visible arrays (GB) | Native permanent estimate (GB) | Native CSR (GB) | RHS/solution buffers (GB) | Accounted estimate (GB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in MODELS:
        m = record["models"][name]
        lines.append(f"| {labels[name]} | {m['test_error']:.6g} | {100*m['kernel_density']:.4g}% | {m['visible_model_array_bytes']/10**9:.4f} | {m['native_permanent_device_ESTIMATE_bytes']/10**9:.4f} | {m['native_owned_CSR_requested_bytes']/10**9:.4f} | {m['native_RHS_solution_requested_bytes']/10**9:.5f} | {m['accounted_resident_storage_ESTIMATE_decimal_GB']:.4f} |")
    lines.extend(["", "| Model | Warm 100-field pass median [min,max] (s) | Train + validation (min) | Post-selection operator rebuild (s) | Auto predictor preparation, accuracy-check predictor (s) | Prediction initialization, two 100-field passes including compilation (s) |",
        "|---|---:|---:|---:|---:|---:|"])
    for name in MODELS:
        m = record["models"][name]
        selection_time = (f"{m['training_validation_minutes']:.4g}"
            if m.get("selection_time_comparable_without_interruption", True)
            else f"Excluded (interrupted); raw {m['training_validation_minutes']:.4g}")
        lines.append(f"| {labels[name]} | {m['warm_100query_median_seconds']:.6g} [{m['warm_100query_min_seconds']:.6g}, {m['warm_100query_max_seconds']:.6g}] | {selection_time} | {m['operator_setup_seconds']:.4g} | {m['predictor_preparation_seconds']:.4g} | {m['first_prediction_compile_capture_seconds']:.4g} |")
    return "\n".join(lines)


def runtime_interruption_record(report_dir):
    """Report a user interruption without changing or correcting raw timings."""
    path = report_dir / "user_pause_state.json"
    if not path.is_file():
        return None
    state = read_json(path)
    duration = state.get("stopped_duration_seconds")
    duration_note = ("Resume is pending; observed stopped duration is not yet recorded."
        if duration is None else f"Observed stopped duration: {duration:.6f} seconds.")
    qualification = ("Fixed/no-polynomial raw selection wall time includes user-requested suspension "
        "and is excluded from uninterrupted selection-time comparisons. Raw numbers are preserved; "
        "no pause duration is subtracted from any timer. Final post-selection operator rebuild, "
        "prediction initialization and warm timings begin after resume and are unaffected by the "
        "suspension. " + duration_note)
    return {"source_file": path.name, "affected_cell": state["current_cell"],
        "affected_model": "fixed_no_poly", "stopped_duration_seconds": duration,
        "qualification": qualification, "source_pause_state": state}


def reporting_record(record, interruption):
    """Annotate reporting copies after the unchanged certificate validation.

    Immutable acceptance records retain their original metrics and hashes.
    No raw timer is replaced, adjusted, or corrected for an interruption.
    """
    result = dict(record)
    result["models"] = {}
    for name in MODELS:
        affected = (interruption is not None
            and record["cell"]["key"] == interruption["affected_cell"]
            and name == interruption["affected_model"])
        result["models"][name] = {
            **record["models"][name],
            "response_scaling": record["explicit_response_transformation"][name],
            "selection_time_comparable_without_interruption": not affected,
            "selection_time_interrupted": affected,
            "selection_interruption_observed_seconds": (
                interruption["stopped_duration_seconds"] if affected else 0.),
        }
    return result


TAYLOR_GREEN_RESPONSE_QUALIFICATION = (
    "All four Taylor-Green variants share the explicit training-input amplitude "
    "normalization (response_scaling=taylor_green_ray): outputs are divided by "
    "the verified input amplitude during fitting, and query amplitude is restored "
    "in the physical Lagrange weights. Affine constant reproduction applies to "
    "amplitude-proportional normalized responses. These four-way results do not "
    "isolate the contribution of normalization; comparisons are conditional on "
    "this same response transformation. Other datasets use unscaled outputs."
)


def aggregate(report_dir, root, manifest, expected):
    accepted_dir = report_dir / "accepted_cells"
    planned = {cell["key"]: cell for cell in manifest["plan"]["cells"]}
    manifest_sha = sha256(root / "manifest.json")
    records = {}
    for path in sorted(accepted_dir.glob("*.json")):
        record = read_json(path)
        key = record["cell"]["key"]
        if key in records or record["cell"] != planned.get(key) or path.stem != key:
            raise RuntimeError("Unexpected or duplicate acceptance cache: " + str(path))
        verify_cached_acceptance(record, root, expected, manifest_sha)
        records[key] = record
    interruption = runtime_interruption_record(report_dir)
    ordered = [reporting_record(records[cell["key"]], interruption)
        for cell in manifest["plan"]["cells"] if cell["key"] in records]
    summary = {"created_utc": utc_now(), "campaign": root.name, "accepted_cell_count": len(ordered),
        "planned_cell_count": len(planned), "frozen_source_verified_files": len(expected),
        "audit_cache_policy": "Each new cell received the unchanged frozen strict full validation, including all input/split/source/native/selection/geometry/prediction hashes. Aggregation rehashes every accepted result/selection/protocol and all frozen source/native files; prior large artifact/input contents are reused only while size and mtime_ns match their full-audit metadata. This is a metadata cache, not a claim that all input/prediction bytes were rehashed on every report.",
        "storage_formula": "visible_model_array_bytes + native_permanent_device_ESTIMATE_bytes + native_owned_CSR_requested_bytes + native_RHS_solution_requested_bytes",
        "storage_qualification": "Accounted component estimate, not measured total/peak GPU allocation. Do not add padded JAX entries or precomputed fields again; they are already visible. Excludes executables, scratch, allocator/pool overhead, other campaign-scope retained objects and opaque driver allocations. Vendor permanent memory is an analysis-time estimate for the configured RHS capacity; visible arrays deduplicate exact data pointers rather than allocation base/extents.",
        "timing_qualification": "Each warm timing repetition is one pass over all 100 physical fields, processed as four native batches of 32 with edge padding to 128; the reported warm median is over 11 synchronized repetitions. Prediction initialization includes two synchronized 100-field passes: the public prediction call and the additional jitted timing-wrapper call, including route preparation, compilation, CUDA graph capture/replay and wrapper overhead. Post-selection operator rebuild excludes training/validation and shared spatial preparation. Auto predictor preparation constructs the separate accuracy-check predictor and may precompute training fields; the warm timing uses a separately prepared weights predictor. Model variants share one process and spatial preparation, and JIT compilation can be reused, so these are not independently cold-process model timings. Archived CPU first-batch times are not matched warm GPU hardware speedups.",
        "response_transformation_qualification": TAYLOR_GREEN_RESPONSE_QUALIFICATION,
        "monitor_started": False, "runtime_interruption_record": interruption, "cells": ordered}
    atomic_json(report_dir / "accepted_summary.json", summary)
    rows = []
    for record in ordered:
        for name in MODELS:
            rows.append({"case": record["cell"]["case"], "N": record["cell"]["N"],
                **{key: value for key, value in record["models"][name].items() if not isinstance(value, (dict, list))}})
    fields = list(rows[0]) if rows else ["case", "N", "model"]
    with (report_dir / "accepted_models.csv").open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    lines = ["# Matched dimension-safe GPU comparison", "", f"{len(ordered)}/{len(planned)} cells accepted.", ""]
    for record in ordered:
        lines.extend(["## " + record["cell"]["key"], "", markdown_table(record), ""])
        if any(model["response_scaling"] == "taylor_green_ray" for model in record["models"].values()):
            lines.extend([TAYLOR_GREEN_RESPONSE_QUALIFICATION, ""])
        if interruption is not None and record["cell"]["key"] == interruption["affected_cell"]:
            lines.extend(["Runtime interruption: " + interruption["qualification"], ""])
    lines.extend([summary["storage_qualification"], "", summary["timing_qualification"], "", summary["audit_cache_policy"], ""])
    (report_dir / "accepted_report.md").write_text("\n".join(lines), encoding="utf8")
    return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    report_dir = Path(__file__).resolve().parent
    parser.add_argument("--cell", required=True)
    parser.add_argument("--campaign-root", type=Path,
        default=report_dir.with_name(report_dir.name.removesuffix("_report")))
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    manifest = read_json(root / "manifest.json")
    cells = {cell["key"]: cell for cell in manifest["plan"]["cells"]}
    if args.cell not in cells:
        parser.error("--cell must name exactly one cell from the frozen campaign manifest")
    folder = root / "cells" / args.cell
    if not (folder / "results.json").is_file():
        parser.error("This cell has no completed result; no partial cell is audited")
    validator, expected = verify_and_import_frozen_validator(root, manifest)
    boundary = capture_boundary_stamps(root, folder)
    validated = validator.validate_complete(folder, manifest["plan"]["repetitions"], cells[args.cell],
        manifest["plan"], expected)
    accepted = build_acceptance(root, cells[args.cell], manifest, expected, validated)
    if any(stamp(path) != before for path, before in boundary.items()):
        raise RuntimeError("An input or immutable artifact changed during the full acceptance audit")
    accepted_dir = report_dir / "accepted_cells"
    accepted_dir.mkdir(exist_ok=True)
    path = accepted_dir / (args.cell + ".json")
    if path.exists():
        previous = read_json(path)
        verify_cached_acceptance(previous, root, expected, sha256(root / "manifest.json"))
        if previous["models"] != accepted["models"]:
            raise RuntimeError("A repeated full audit disagrees with its immutable acceptance record")
    else:
        immutable_json(path, accepted)
    summary = aggregate(report_dir, root, manifest, expected)
    print(json.dumps({"accepted_cell": args.cell, "accepted_total": summary["accepted_cell_count"],
        "planned_total": summary["planned_cell_count"], "acceptance_file": str(path),
        "frozen_source_verified_files": len(expected), "monitor_started": False}, indent=2))
    print(markdown_table(reporting_record(accepted, runtime_interruption_record(report_dir))))


if __name__ == "__main__":
    main()
