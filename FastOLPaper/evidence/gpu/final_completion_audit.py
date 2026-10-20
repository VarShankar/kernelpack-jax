"""One-shot final revalidation and paired report for the completed GPU study.

PREPARED ONLY: execute after explicit authorization and all 40 accepted cells.
Standard library only; no numerical arrays are decoded or evaluated, no JAX or
NumPy is imported, no GPU is initialized, and no process is launched or watched.
The unchanged frozen validator runs on every finished cell. Fresh final-audit
SHA256 results are cached only within this invocation and immutable identities.
Existing numerical artifacts, acceptance certificates and aggregate files are
read-only. Outputs are new final audit/report files beside this script.
"""
from __future__ import annotations

import argparse
import ast
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
import zipfile


MODELS = ("fixed_affine", "trainable_affine", "fixed_no_poly", "trainable_no_poly")
SIZES = (10000, 5000, 1000, 500)
CASES = ("forced_turb", "species_transport", "taylor_green", "merge_vortices_easier",
         "taylor_green_time", "flow_cylinder_laminar", "flow_cylinder_shedding",
         "backward_facing_step", "buoyancy_cavity_flow", "lid_cavity_flow")
MANIFEST_SHA256 = "6ec0c74cc9296fd65f798a6ab42bc8d96cfd00f14726922514cf81830ebe13e2"
HELPER_SHA256 = "10868b76acbdc38c44a96fe5568c1aa8e06591438a24d1d27b951d208d791e2c"
INTERRUPTED_CELL = "backward_facing_step_N10000"
INTERRUPTED_MODEL = "fixed_no_poly"
PAUSE_SECONDS = 1525.294946
COMPARISON_METRICS = (
    ("test_error", "physical_field_error"),
    ("validation_error", "validation_field_error"),
    ("kernel_density", "kernel_density"),
    ("actual_kernel_nnz", "kernel_nnz"),
    ("native_factor_nnz", "native_factor_nnz"),
    ("visible_model_array_bytes", "visible_model_array_bytes"),
    ("native_permanent_device_ESTIMATE_bytes", "native_permanent_estimate_bytes"),
    ("native_peak_device_ESTIMATE_bytes", "native_peak_estimate_bytes"),
    ("native_owned_CSR_requested_bytes", "native_owned_CSR_requested_bytes"),
    ("native_RHS_solution_requested_bytes", "native_RHS_solution_requested_bytes"),
    ("accounted_resident_storage_ESTIMATE_bytes", "accounted_storage_estimate_bytes"),
    ("warm_100query_median_seconds", "warm_100field_pass_median_seconds"),
    ("operator_setup_seconds", "post_selection_operator_rebuild_seconds"),
    ("predictor_preparation_seconds", "accuracy_predictor_preparation_seconds"),
    ("first_prediction_compile_capture_seconds", "prediction_initialization_two_pass_seconds"),
)


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def bound_json(path, ledger):
    ledger.digest(path)
    value = read_json(path)
    ledger.digest(path)
    return value


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def identity(path):
    s = path.stat()
    return {"dev": s.st_dev, "inode": s.st_ino, "size": s.st_size,
            "mtime_ns": s.st_mtime_ns, "ctime_ns": s.st_ctime_ns}


class FinalHashLedger:
    """Fresh content hashes with invocation-local immutable-file memoization.

    No prior audit or acceptance digest seeds this cache. Every distinct resolved
    file is streamed once during this invocation. Every access compares full
    metadata identity; hashing verifies fstat/path identities before and after.
    All identities are checked again at the final boundary. Metadata-based cache
    reuse assumes these read-only files are not covertly rewritten with restored
    identity metadata; it is not protection against a malicious filesystem actor.
    """
    def __init__(self):
        self.entries = {}
        self.requested_aliases = {}
        self.requests = 0
        self.hits = 0
        self.bytes_hashed = 0

    def digest(self, path):
        requested = Path(path).absolute()
        path = requested.resolve(strict=True)
        alias, key = str(requested), str(path)
        require(alias not in self.requested_aliases or self.requested_aliases[alias] == key,
                "A previously hashed path resolves to a different target: " + alias)
        self.requested_aliases[alias] = key
        self.requests += 1
        before = identity(path)
        if key in self.entries:
            entry = self.entries[key]
            require(before == entry["identity"], "File changed after final content hash: " + key)
            self.hits += 1
            return entry["sha256"]
        require(path.is_file(), "Not a regular audit file: " + key)
        if before["size"] >= 1024**3:
            print(json.dumps({"stage": "fresh_shared_file_hash", "path": key,
                              "bytes": before["size"]}), flush=True)
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            opened_id = {"dev": opened.st_dev, "inode": opened.st_ino,
                         "size": opened.st_size, "mtime_ns": opened.st_mtime_ns,
                         "ctime_ns": opened.st_ctime_ns}
            require(opened_id == before, "File identity raced open: " + key)
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            finished = os.fstat(stream.fileno())
            finished_id = {"dev": finished.st_dev, "inode": finished.st_ino,
                           "size": finished.st_size, "mtime_ns": finished.st_mtime_ns,
                           "ctime_ns": finished.st_ctime_ns}
        require(finished_id == before == identity(path), "File changed during final hash: " + key)
        self.entries[key] = {"path": key, "sha256": checksum, "identity": before}
        self.bytes_hashed += before["size"]
        return checksum

    def verify_boundary(self):
        for alias, target in self.requested_aliases.items():
            require(str(Path(alias).resolve(strict=True)) == target,
                    "An audited path alias changed target: " + alias)
        for entry in self.entries.values():
            require(identity(Path(entry["path"])) == entry["identity"],
                    "Final immutable identity changed: " + entry["path"])

    def report(self):
        return {"fresh_unique_content_hashes": len(self.entries), "digest_requests": self.requests,
                "invocation_local_cache_hits": self.hits, "bytes_streamed_for_SHA256": self.bytes_hashed,
                "cache_seeded_from_existing_acceptance": False,
                "requested_path_aliases_verified": self.requested_aliases,
                "fresh_content_hashes": list(self.entries.values())}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def immutable_json(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_csv(path, rows):
    require(rows, "No final comparison rows")
    with path.open("x", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def bound_csv(path, ledger):
    ledger.digest(path)
    value = read_csv(path)
    ledger.digest(path)
    return value


def reviewed_aggregate_texts(helper_path):
    """Obtain the pinned helper's literal qualifications without running aggregate."""
    tree = ast.parse(helper_path.read_text(encoding="utf8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "aggregate")
    fields = {"storage_formula", "storage_qualification", "timing_qualification", "audit_cache_policy"}
    found = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value in fields:
                    found[key.value] = ast.literal_eval(value)
    require(set(found) == fields, "Reviewed helper qualification constants are missing")
    return found


def same_csv_row(actual, expected, context):
    require(set(actual) == set(expected), "CSV columns differ: " + context)
    for key, value in expected.items():
        text = actual[key]
        if isinstance(value, bool):
            matches = text == str(value)
        elif isinstance(value, (int, float)):
            matches = float(text) == value
        elif value is None:
            matches = text == ""
        else:
            matches = text == value
        require(matches, "CSV value differs from freshly validated raw files: " + context + "/" + key)


def zip_member_hash(path, name):
    """Compare exported coordinate bytes without decoding numerical arrays."""
    with zipfile.ZipFile(path) as archive, archive.open(name + ".npy") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_geometry_bytes(folder, selected, response):
    common = ("center", "parallel_basis", "delta", "normalizer_mean", "normalizer_std", "normalizer_zero")
    if response != "none":
        common += ("response_scale", "ray_direction", "ray_zero", "ray_median")
    hashes = {name: {field: zip_member_hash(folder / ("selected_geometry_" + name + ".npz"), field)
                     for field in common + ("basis", "scale")} for name in MODELS}
    for field in common:
        require(len({hashes[name][field] for name in MODELS}) == 1,
                "Fixed/learned coordinate bytes differ: " + str(folder) + "/" + field)
    for suffix in ("affine", "no_poly"):
        a, b = "fixed_" + suffix, "trainable_" + suffix
        for field in ("basis", "scale"):
            require(hashes[a][field] == hashes[b][field], "Paired polynomial coordinate bytes differ")
        require(selected["models"][a]["mean_basis_sha256"] == selected["models"][b]["mean_basis_sha256"],
                "Paired original GPU polynomial matrix fingerprints differ")
    return hashes


def check_split_groups(groups, expected, context):
    require([len(group) for group in groups] == expected, "Wrong split sizes: " + context)
    require(all(all(type(value) is int and value >= 0 for value in group) for group in groups),
            "Invalid split identities: " + context)
    sets = [set(group) for group in groups]
    require(all(len(group) == len(unique) for group, unique in zip(groups, sets)),
            "Duplicate example IDs: " + context)
    require(not any(sets[i] & sets[j] for i in range(3) for j in range(i)),
            "Train/validation/test overlap: " + context)


def audit_nested_splits(raw):
    records = []
    for case in CASES:
        by_size = {}
        anchor_protocol, anchor_result = raw[(case, 10000)]
        for n in SIZES:
            protocol, result = raw[(case, n)]
            logical = [protocol["training_ids"], protocol["validation_ids"], protocol["test_ids"]]
            physical = [protocol["physical_training_validation_ids"][:n],
                        protocol["physical_training_validation_ids"][n:], result["physical_test_ids"]]
            check_split_groups(logical, [n, 100, 100], case + "/" + str(n) + "/logical")
            check_split_groups(physical, [n, 100, 100], case + "/" + str(n) + "/physical")
            require(logical[0] == anchor_protocol["training_ids"][:n], "Logical training sets are not ordered nested prefixes")
            require(physical[0] == anchor_protocol["physical_training_validation_ids"][:n], "Physical training sets are not ordered nested prefixes")
            require(logical[1:] == [anchor_protocol["validation_ids"], anchor_protocol["test_ids"]], "Validation/test logical IDs vary with N")
            require(physical[1:] == [anchor_protocol["physical_training_validation_ids"][10000:], anchor_result["physical_test_ids"]], "Validation/test physical IDs vary with N")
            require(protocol["split_sha256"] == anchor_protocol["split_sha256"], "Saved split file differs across N")
            require(protocol["spatial_options"] == anchor_protocol["spatial_options"]
                    and protocol["input_shape"][1:] == anchor_protocol["input_shape"][1:]
                    and protocol["output_shape"][1:] == anchor_protocol["output_shape"][1:],
                    "Spatial method/resolution or input representation changes across N")
            by_size[str(n)] = {"training_count": n, "validation_count": 100, "test_count": 100,
                "logical_ids_sha256": [canonical(group) for group in logical],
                "physical_ids_sha256": [canonical(group) for group in physical],
                "ordered_training_prefix_of_N10000": True, "same_ordered_validation_test": True,
                "logical_and_physical_splits_disjoint_and_unique": True}
        records.append({"dataset": case, "sizes": by_size, "same_spatial_method_and_resolution": True})
    return records


def paired_rows(records):
    output = []
    for record in records:
        for suffix in ("no_poly", "affine"):
            fixed, learned = record["models"]["fixed_" + suffix], record["models"]["trainable_" + suffix]
            row = {"case": record["cell"]["case"], "N": record["cell"]["N"],
                   "polynomial_setting": suffix, "fixed_model": fixed["model"], "trainable_model": learned["model"],
                   "response_scaling": fixed["response_scaling"],
                   "shared_spatial_preparation_seconds_once_per_cell": record["shared_spatial_preparation_seconds"]}
            require(fixed["response_scaling"] == learned["response_scaling"], "Unmatched response transformation")
            for source, label in COMPARISON_METRICS:
                a, b = fixed[source], learned[source]
                row[label + "_fixed"] = a
                row[label + "_trainable"] = b
                row[label + "_trainable_over_fixed"] = b / a if a else None
            row.update(warm_100field_pass_min_fixed=fixed["warm_100query_min_seconds"],
                warm_100field_pass_max_fixed=fixed["warm_100query_max_seconds"],
                warm_100field_pass_min_trainable=learned["warm_100query_min_seconds"],
                warm_100field_pass_max_trainable=learned["warm_100query_max_seconds"],
                observed_warm_ranges_overlap=max(fixed["warm_100query_min_seconds"], learned["warm_100query_min_seconds"]) <= min(fixed["warm_100query_max_seconds"], learned["warm_100query_max_seconds"]))
            comparable = fixed["selection_time_comparable_without_interruption"] and learned["selection_time_comparable_without_interruption"]
            row.update(selection_wall_seconds_fixed_RAW=fixed["training_validation_seconds"],
                selection_wall_seconds_trainable_RAW=learned["training_validation_seconds"],
                selection_wall_comparable_without_interruption=comparable,
                selection_wall_trainable_over_fixed=(learned["training_validation_seconds"] / fixed["training_validation_seconds"] if comparable else None),
                selection_interruption_observed_seconds=(fixed["selection_interruption_observed_seconds"] + learned["selection_interruption_observed_seconds"]),
                selection_comparison_qualification=("Uninterrupted recorded wall times; training costs differ by design" if comparable else "EXCLUDED: fixed no-polynomial selection contains the user suspension; raw timers retained without subtraction"))
            output.append(row)
    return output


def paired_markdown(rows, summary, records):
    lines = ["# Final matched dimension-safe GPU comparison", "", "40 dataset and training-size combinations; 160 fully checked model results.", "",
        "Each comparison holds the polynomial setting fixed. Ratios are trainable/fixed; values below one mean numerically lower error, density, storage or time. Warm ranges are observed repetitions, not confidence intervals. Small timing differences are not evidence of a reliable speed improvement.", "",
        summary["response_transformation_qualification"], "", summary["storage_qualification"], "", summary["timing_qualification"], ""]
    lines.extend(["## Shared spatial preparation", "", "This setup is shared by all four models and charged once per dataset and training-size cell. It must not be added once per model or twice for the two paired CSV rows.", "",
                  "| Dataset | N | Shared spatial preparation (s) |", "|---|---:|---:|"])
    for record in records:
        lines.append(f"| {record['cell']['case']} | {record['cell']['N']} | {record['shared_spatial_preparation_seconds']:.6g} |")
    lines.append("")
    for suffix, title in (("no_poly", "Without polynomials"), ("affine", "With the same affine polynomial")):
        group = [row for row in rows if row["polynomial_setting"] == suffix]
        lines.extend(["## " + title, "", "| Dataset | N | Error fixed | Error trainable | Error ratio | Density fixed | Density trainable |", "|---|---:|---:|---:|---:|---:|---:|"])
        for r in group:
            error_ratio = (f"{r['physical_field_error_trainable_over_fixed']:.4g}"
                if r['physical_field_error_trainable_over_fixed'] is not None else "Undefined (fixed error is zero)")
            lines.append(f"| {r['case']} | {r['N']} | {r['physical_field_error_fixed']:.6g} | {r['physical_field_error_trainable']:.6g} | {error_ratio} | {100*r['kernel_density_fixed']:.4g}% | {100*r['kernel_density_trainable']:.4g}% |")
        lines.extend(["", "| Dataset | N | Visible arrays GB fixed / trainable | Native permanent estimate GB fixed / trainable | Accounted estimate GB fixed / trainable | Accounted ratio |", "|---|---:|---:|---:|---:|---:|"])
        for r in group:
            lines.append(f"| {r['case']} | {r['N']} | {r['visible_model_array_bytes_fixed']/1e9:.4f} / {r['visible_model_array_bytes_trainable']/1e9:.4f} | {r['native_permanent_estimate_bytes_fixed']/1e9:.4f} / {r['native_permanent_estimate_bytes_trainable']/1e9:.4f} | {r['accounted_storage_estimate_bytes_fixed']/1e9:.4f} / {r['accounted_storage_estimate_bytes_trainable']/1e9:.4f} | {r['accounted_storage_estimate_bytes_trainable_over_fixed']:.4g} |")
        lines.extend(["", "| Dataset | N | Warm 100-field pass seconds fixed / trainable | Warm ratio | Post-selection rebuild seconds fixed / trainable | Prediction initialization seconds fixed / trainable | Selection wall seconds fixed / trainable | Selection ratio |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for r in group:
            a, b = r['selection_wall_seconds_fixed_RAW'], r['selection_wall_seconds_trainable_RAW']
            ratio = f"{r['selection_wall_trainable_over_fixed']:.4g}" if r['selection_wall_comparable_without_interruption'] else "Excluded (interrupted)"
            lines.append(f"| {r['case']} | {r['N']} | {r['warm_100field_pass_median_seconds_fixed']:.6g} / {r['warm_100field_pass_median_seconds_trainable']:.6g} | {r['warm_100field_pass_median_seconds_trainable_over_fixed']:.4g} | {r['post_selection_operator_rebuild_seconds_fixed']:.4g} / {r['post_selection_operator_rebuild_seconds_trainable']:.4g} | {r['prediction_initialization_two_pass_seconds_fixed']:.4g} / {r['prediction_initialization_two_pass_seconds_trainable']:.4g} | {a:.6g} / {b:.6g} | {ratio} |")
        lines.append("")
    lines.extend([summary["runtime_interruption_record"]["qualification"], "",
        "The CSV additionally reports actual kernel nnz, native factor nnz, native peak estimates, CSR and RHS buffers separately. Native peak estimates are not added to permanent estimates. No CPU numerical replay or matched CPU/GPU hardware-speedup claim is made.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--confirm-all-40-complete", action="store_true", required=True)
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--expected-helper-sha256", default=HELPER_SHA256)
    args = parser.parse_args()
    report = Path(__file__).resolve().parent
    root = (args.campaign_root or report.with_name(report.name.removesuffix("_report"))).resolve()
    outputs = [report / name for name in ("completion_audit.json", "final_paired_comparisons.csv", "final_paired_report.md", "final_validated_models.csv", "final_shared_spatial_preparation.csv")]
    require(not any(path.exists() for path in outputs), "Final audit outputs already exist; preserve them and review before any separate rerun")
    ledger = FinalHashLedger()
    require(ledger.digest(root / "manifest.json") == MANIFEST_SHA256, "The frozen campaign manifest differs from the reviewed campaign")
    manifest = bound_json(root / "manifest.json", ledger)
    plan, expected = manifest["plan"], manifest["runtime"]["source_sha256"]
    cells = plan["cells"]
    target = {(case, n) for case in CASES for n in SIZES}
    require(len(cells) == 40 and {(c["case"], c["N"]) for c in cells} == target
            and len({c["key"] for c in cells}) == 40
            and all(c["key"] == c["case"] + "_N" + str(c["N"]) for c in cells), "Expected exactly the ten datasets at all four sizes")
    status = bound_json(root / "status.json", ledger)
    require(status.get("status") == "complete" and status.get("child_pid") is None
            and status.get("current_cell") is None and len(status.get("completed", [])) == 40
            and set(status["completed"]) == {c["key"] for c in cells}, "The queue has not completed all40; this script never waits or launches work")
    require(all((root / "cells" / c["key"] / "results.json").is_file() for c in cells), "Not all40 completed results exist")
    require({p.parent.name for p in (root / "cells").glob("*/results.json")} == {c["key"] for c in cells}, "Unexpected completed cell artifacts outside the frozen plan")
    cert_paths = list((report / "accepted_cells").glob("*.json"))
    require(len(cert_paths) == 40 and {p.stem for p in cert_paths} == {c["key"] for c in cells}, "Exactly40 existing immutable acceptances are required")
    aggregate = bound_json(report / "accepted_summary.json", ledger)
    csv_rows = bound_csv(report / "accepted_models.csv", ledger)
    require(aggregate.get("accepted_cell_count") == 40 and len(aggregate.get("cells", [])) == 40 and len(csv_rows) == 160, "Refresh the final accepted aggregate before this audit; no acceptance is performed here")
    audit = {"started_utc": now(), "passed": False, "purpose": "Final full artifact revalidation, not a numerical rerun",
             "numerical_jobs_started": 0, "GPU_initialized": False, "watcher_started": False}
    try:
        sys.dont_write_bytecode = True
        require(ledger.digest(root / "manifest.json") == MANIFEST_SHA256, "The frozen campaign manifest differs from the reviewed campaign")
        helper_path = report / "report_one_cell.py"
        require(ledger.digest(helper_path) == args.expected_helper_sha256, "Reporting helper changed; obtain a reviewed helper hash before final audit")
        helper = load_module(helper_path, "final_audit_reporting_helper")
        validator_path = root / "source_snapshot/scripts/run_fastol_gpu_campaign.py"
        require(ledger.digest(validator_path) == expected["scripts/run_fastol_gpu_campaign.py"], "Frozen strict validator hash differs")
        validator = load_module(validator_path, "final_audit_frozen_strict_validator")
        require(tuple(validator.MODELS) == MODELS, "Frozen validator models differ")
        # Only the byte-hash provider is substituted. All validator/helper
        # scientific conditions, thresholds and certificate checks are unchanged.
        validator.digest = ledger.digest
        helper.sha256 = ledger.digest
        validator.verify_snapshot(root / "source_snapshot", expected)
        audit.update(manifest_sha256=MANIFEST_SHA256, helper_sha256=args.expected_helper_sha256,
                     validator_sha256=expected["scripts/run_fastol_gpu_campaign.py"],
                     script_sha256=ledger.digest(Path(__file__)), frozen_source_files=len(expected))
        interruption = helper.runtime_interruption_record(report)
        require(interruption is not None and interruption["affected_cell"] == INTERRUPTED_CELL
                and interruption["affected_model"] == INTERRUPTED_MODEL
                and interruption["stopped_duration_seconds"] == PAUSE_SECONDS,
                "The reviewed single suspension metadata changed")
        ledger.digest(report / "user_pause_state.json")
        require(aggregate["runtime_interruption_record"] == interruption, "Aggregate suspension metadata differs from the reviewed runtime annotation")
        for key, text in reviewed_aggregate_texts(helper_path).items():
            require(aggregate[key] == text, "Aggregate qualification differs from the pinned reporting helper: " + key)
        aggregate_sha = ledger.digest(report / "accepted_summary.json")
        csv_sha = ledger.digest(report / "accepted_models.csv")
        md_sha = ledger.digest(report / "accepted_report.md")
        md = (report / "accepted_report.md").read_text(encoding="utf8")
        ledger.digest(report / "accepted_report.md")
        aggregate_index = {c["cell"]["key"]: c for c in aggregate["cells"]}
        csv_index = {(row["case"], int(row["N"]), row["model"]): row for row in csv_rows}
        require(len(aggregate_index) == 40 and set(aggregate_index) == {c["key"] for c in cells}, "Duplicate/missing aggregate cells")
        require(len(csv_index) == 160, "Duplicate aggregate model rows")
        require({line.removeprefix("## ") for line in md.splitlines() if line.startswith("## ")} == {c["key"] for c in cells}
                and all(md.count("## " + c["key"] + "\n") == 1 for c in cells), "Missing or duplicate Markdown cell sections")
        fresh_records, raw_splits, validated_cells = [], {}, []
        scalar_rows = []
        for position, cell in enumerate(cells, 1):
            folder = root / "cells" / cell["key"]
            boundary = helper.capture_boundary_stamps(root, folder)
            checked = validator.validate_complete(folder, plan["repetitions"], cell, plan, expected)
            require(all(helper.stamp(path) == stamp for path, stamp in boundary.items()), "A cell artifact/input changed during final validation")
            result, selected, protocol = (read_json(folder / name) for name in ("results.json", "selection.json", "protocol.json"))
            # Hash result explicitly, as frozen validate_complete returns this
            # digest but its other scientific checks read the complete JSON.
            require(checked["results_sha256"] == ledger.digest(folder / "results.json"), "Final result hash mismatch")
            cert_path = report / "accepted_cells" / (cell["key"] + ".json")
            cert_sha = ledger.digest(cert_path)
            certificate = read_json(cert_path)
            helper.verify_cached_acceptance(certificate, root, expected, MANIFEST_SHA256)
            require(certificate["cell"] == cell, "Acceptance certificate identity differs")
            fresh = {"cell": cell, "models": {name: helper.make_model_metrics(name, result["models"][name], selected["models"][name], cell["N"]) for name in MODELS},
                     "explicit_response_transformation": {name: selected["models"][name]["config"]["response_scaling"] for name in MODELS},
                     "shared_spatial_preparation_seconds": result["shared_spatial_preparation_seconds"],
                     "timing_note": result["timing_note"], "storage_note": result["storage_note"]}
            require(fresh["models"] == certificate["models"], "Fresh metrics differ from immutable acceptance")
            fresh = helper.reporting_record(fresh, interruption)
            require(fresh["models"] == aggregate_index[cell["key"]]["models"], "Aggregate differs from freshly revalidated raw metrics/annotations")
            require(fresh["explicit_response_transformation"] == aggregate_index[cell["key"]]["explicit_response_transformation"], "Aggregate response metadata differs")
            for field in ("shared_spatial_preparation_seconds", "timing_note", "storage_note"):
                require(fresh[field] == certificate[field] == aggregate_index[cell["key"]][field],
                        "Raw shared setup/timing/storage metadata differs: " + cell["key"] + "/" + field)
            section = md.split("## " + cell["key"] + "\n", 1)[1].split("\n## ", 1)[0]
            require(helper.markdown_table(fresh) in section, "Markdown table differs from freshly validated raw values")
            response = "taylor_green_ray" if cell["case"] in ("taylor_green", "taylor_green_time") else "none"
            if response != "none":
                require(helper.TAYLOR_GREEN_RESPONSE_QUALIFICATION in section, "Missing shared Taylor-Green response qualification")
            geometry = check_geometry_bytes(folder, selected, response)
            for name in MODELS:
                model, choice, metrics = result["models"][name], selected["models"][name], fresh["models"][name]
                require(model["accuracy_checks"]["timed_native_solve_count"] == 44
                        and model["accuracy_checks"]["expected_timed_native_solves"] == 44
                        and model["graph_replays_during_timing"] >= 44, "Every timed native32 batch must retain its checked sparse solve and graph replay")
                histories = choice["training_histories"]
                require(dict(choice["training_timings"])["training_steps_executed"] == (0 if name.startswith("fixed_") else 120), "Actual total training-step evidence differs")
                require(len(histories) == (0 if name.startswith("fixed_") else 2), "Wrong actual number of training paths")
                affected = cell["key"] == INTERRUPTED_CELL and name == INTERRUPTED_MODEL
                require(metrics["selection_time_interrupted"] is affected
                        and metrics["selection_time_comparable_without_interruption"] is (not affected)
                        and metrics["selection_interruption_observed_seconds"] == (PAUSE_SECONDS if affected else 0.), "Incorrect interruption annotation")
                require(metrics["training_validation_seconds"] == choice["training_validation_seconds"]
                        and metrics["response_scaling"] == response, "Raw timer/response transformation changed")
                row = {"case": cell["case"], "N": cell["N"], **{key: value for key, value in metrics.items() if not isinstance(value, (dict, list))}}
                same_csv_row(csv_index[(cell["case"], cell["N"], name)], row, cell["key"] + "/" + name)
                scalar_rows.append(row)
            raw_splits[(cell["case"], cell["N"])] = (protocol, result)
            fresh_records.append(fresh)
            validated_cells.append({"cell": cell, "frozen_strict_validation_passed": True,
                "results_sha256": checked["results_sha256"], "selection_sha256": checked["selection_sha256"],
                "protocol_sha256": ledger.digest(folder / "protocol.json"),
                "acceptance_certificate_sha256": cert_sha, "coordinate_member_byte_hashes": geometry})
            print(json.dumps({"stage": "final_full_validation_passed", "position": position, "total": 40, "cell": cell["key"]}), flush=True)
        require(len(scalar_rows) == 160 and len(fresh_records) == 40, "Incomplete final model set")
        require(md.count("Excluded (interrupted); raw") == 1 and interruption["qualification"] in md, "BFS interrupted selection exclusion is missing/duplicated")
        require(aggregate["response_transformation_qualification"] == helper.TAYLOR_GREEN_RESPONSE_QUALIFICATION, "Aggregate response qualification differs")
        require(aggregate.get("monitor_started") is False, "Aggregate monitor policy differs")
        split_checks = audit_nested_splits(raw_splits)
        pairs = paired_rows(fresh_records)
        require(len(pairs) == 80, "Missing final fixed/learned polynomial pair")
        validator.verify_snapshot(root / "source_snapshot", expected)
        require(bound_json(root / "status.json", ledger) == status, "Final queue completion state changed during audit")
        require({p.name for p in (report / "accepted_cells").glob("*.json")} == {p.name for p in cert_paths}, "Immutable acceptance file set changed during final audit")
        require({p.parent.name for p in (root / "cells").glob("*/results.json")} == {c["key"] for c in cells}, "Completed result set changed during final audit")
        ledger.verify_boundary()
        require(not any(name == "numpy" or name.startswith("numpy.") or name == "jax" or name.startswith("jax.") for name in sys.modules), "A numerical runtime was imported by this audit")
        report_text = paired_markdown(pairs, aggregate, fresh_records)
        write_csv(outputs[1], pairs)
        with outputs[2].open("x", encoding="utf8") as stream:
            stream.write(report_text)
        write_csv(outputs[3], scalar_rows)
        shared_spatial_rows = [{"case": record["cell"]["case"], "N": record["cell"]["N"],
            "shared_spatial_preparation_seconds": record["shared_spatial_preparation_seconds"],
            "shared_by_model_count": 4, "charged_once_per_cell": True} for record in fresh_records]
        write_csv(outputs[4], shared_spatial_rows)
        audit.update(passed=True, finished_utc=now(), completed_cells=40, checked_model_rows=160,
            paired_comparisons=80, all_cells=validated_cells, nested_split_checks=split_checks,
            shared_spatial_preparation=shared_spatial_rows,
            input_split_geometry_prediction_hashes="Fresh full content hashes in this invocation, with identity-guarded unique-file memoization only",
            aggregate_files_verified={"accepted_summary.json": aggregate_sha, "accepted_models.csv": csv_sha, "accepted_report.md": md_sha},
            storage_formula=aggregate["storage_formula"], storage_qualification=aggregate["storage_qualification"],
            timing_qualification=aggregate["timing_qualification"], response_qualification=helper.TAYLOR_GREEN_RESPONSE_QUALIFICATION,
            runtime_interruption=interruption, interrupted_selection_pair_excluded_without_subtraction=True,
            training_step_qualification="Learned variants use two60-step penalty paths,120 steps total, plus eight35-step validation paths; fixed shapes use no REML and four35-step validation paths.",
            convergence_qualification="The same ordered validation/test IDs and nested training sets support an empirical four-size sample-count study of the retuned complete workflow. Normalization/PCA and hyperparameters are refit at each N. This is not a fixed-kernel convergence theorem, monotonicity guarantee, or validated asymptotic rate.",
            limits="No GPU evaluation, PDE solve, saved-array numerical recomputation, field-content duplicate test, upstream independence audit, continuous hardware-exclusivity certification or matched CPU numerical replay occurs here. The audit revalidates saved GPU-check evidence and exported bytes, not independent numerical reproduction.",
            final_outputs={path.name: ledger.digest(path) for path in outputs[1:]},
            hash_ledger=ledger.report())
        ledger.verify_boundary()
        immutable_json(outputs[0], audit)
        print(json.dumps({"passed": True, "cells": 40, "model_rows": 160, "paired_comparisons": 80, "completion_audit": str(outputs[0]), "fresh_unique_hashed_files": len(ledger.entries)}, indent=2))
    except BaseException as error:
        audit.update(passed=False, failed_utc=now(), failure=type(error).__name__ + ": " + str(error), hash_ledger=ledger.report())
        if not outputs[0].exists():
            immutable_json(outputs[0], audit)
        raise


if __name__ == "__main__":
    main()
