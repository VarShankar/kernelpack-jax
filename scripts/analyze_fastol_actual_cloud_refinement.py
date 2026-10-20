"""CUDA float64 reanalysis of all 36 original global scattered-cloud cases.

Saved data and reconstructions are immutable.  No nodes, stencils, targets or
weights are regenerated.  Nominal r/h cuts are descriptive, not certificates.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import traceback
import numpy as np  # Archive I/O only.
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

SPACINGS, ORDERS = (.26, .19, .14, .101), (2, 4, 6)
POLICIES = ("phs", "ram_local_tau16", "fixed_physical_band")
RANKS, WIDTHS = {2: 11, 4: 50, 6: 133}, {2: 9, 4: 35, 6: 91}
REFERENCE_SHA = "4294130ac653551052e7d48b867cd368d1c21483be1bd63ba89fff457f6b7e54"
MANIFEST_SHA = "33e32133e2d495a76c8822008c6a52e459ef0e9ca1cf9579f70213fae9bb7b63"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def export(value):
    if isinstance(value, jax.Array):
        return jax.device_get(value).tolist()
    if isinstance(value, dict):
        return {key: export(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [export(item) for item in value]
    return value


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(export(value), handle, indent=2, allow_nan=False)
        handle.write("\n")


def require(condition, message):
    if not bool(jax.device_get(jnp.asarray(condition)).item()):
        raise RuntimeError(message)


def archive(path, keys):
    with np.load(path, allow_pickle=False) as handle:
        return {key: jnp.asarray(handle[key]) for key in keys}


@jax.jit
def calculate(nodes, queries, indices, weights, truth, prediction, mse, radius):
    bins, directions = len(radius), len(truth) // len(radius)
    true, pred, error = (a.reshape(bins, directions) for a in (truth, prediction, mse))
    ratio, eta = pred / true, jnp.sqrt(error / true)
    amplitude = jnp.abs(jnp.sqrt(ratio) - 1.)
    selected_nodes = nodes[indices]
    query_radius = jnp.max(jnp.linalg.norm(selected_nodes - queries[:, None], axis=-1), axis=1)
    blocks = weights.transpose(0, 2, 1, 3)  # query,node,output,input
    lebesgue = jnp.sum(jnp.linalg.svd(blocks, compute_uv=False)[..., 0], axis=1)
    endpoint_radius = jnp.stack((query_radius[:len(truth)], query_radius[len(truth):]), axis=-1).reshape(bins, -1)
    endpoint_lambda = jnp.stack((lebesgue[:len(truth)], lebesgue[len(truth):]), axis=-1).reshape(bins, -1)
    radius_y, radius_z = query_radius[:len(truth)].reshape(bins, directions), query_radius[len(truth):].reshape(bins, directions)
    mean_true, mean_pred = jnp.mean(true, axis=1), jnp.mean(pred, axis=1)
    rows = dict(reference_s2=mean_true, reconstructed_s2=mean_pred,
        relative_error_of_direction_mean_s2=jnp.abs(mean_pred / mean_true - 1),
        mean_direction_relative_s2_error=jnp.mean(jnp.abs(ratio - 1), axis=1),
        worst_direction_relative_s2_error=jnp.max(jnp.abs(ratio - 1), axis=1),
        mean_increment_mse=jnp.mean(error, axis=1),
        normalized_increment_mse=jnp.sum(error, axis=1) / jnp.sum(true, axis=1),
        worst_direction_normalized_increment_mse=jnp.max(error / true, axis=1),
        maximum_query_radius=jnp.max(endpoint_radius, axis=1), median_query_radius=jnp.median(endpoint_radius, axis=1),
        maximum_block_lebesgue_constant=jnp.max(endpoint_lambda, axis=1),
        median_block_lebesgue_constant=jnp.median(endpoint_lambda, axis=1),
        maximum_query_radius_sum_over_separation=jnp.max((radius_y + radius_z) / radius[:, None], axis=1),
        disjoint_query_ball_pair_count=jnp.sum(radius_y + radius_z < radius[:, None], axis=1),
        both_query_radii_at_most_quarter_separation_pair_count=jnp.sum(
            (radius_y <= radius[:, None] / 4) & (radius_z <= radius[:, None] / 4), axis=1))
    slopes = dict(reference_forward_log_slope=jnp.diff(jnp.log(mean_true)) / jnp.diff(jnp.log(radius)),
                  reconstructed_forward_log_slope=jnp.diff(jnp.log(mean_pred)) / jnp.diff(jnp.log(radius)))
    stability = dict(maximum_query_radius=jnp.max(query_radius), median_query_radius=jnp.median(query_radius),
        maximum_block_lebesgue_constant=jnp.max(lebesgue), median_block_lebesgue_constant=jnp.median(lebesgue),
        maximum_amplitude_bound_violation=jnp.max(jnp.maximum(amplitude - eta, 0.)))
    return rows, slopes, stability


def masked_summary(mask, truth, pred, mse):
    selected = jnp.repeat(mask, 8)
    count = jnp.sum(mask)
    if not bool(jax.device_get(count > 0).item()):
        return dict(radius_bin_count=0, pair_count=0, relative_error_of_summed_s2=None,
                    worst_direction_relative_s2_error=None, normalized_increment_mse=None)
    summed_true = jnp.sum(jnp.where(selected, truth, 0.))
    return dict(radius_bin_count=count, pair_count=jnp.sum(selected),
        relative_error_of_summed_s2=jnp.abs(jnp.sum(jnp.where(selected, pred, 0.)) / summed_true - 1),
        worst_direction_relative_s2_error=jnp.max(jnp.where(selected, jnp.abs(pred / truth - 1), -jnp.inf)),
        normalized_increment_mse=jnp.sum(jnp.where(selected, mse, 0.)) / summed_true)


def analyze(data, row, ref):
    inputs = archive(data / row["input_artifact"], ("nodes", "queries", "query_y", "query_z"))
    maps = archive(data / row["case"] / "spatial_diagnostics.npz",
                   ("stencil_indices", "weights", "retained_rank", "requested_rank", "inactive_padding_count"))
    stats = archive(data / row["case"] / "statistics.npz",
                    ("radius", "pair_reference_s2", "pair_reconstructed_s2", "pair_increment_mse"))
    order, radius = row["value_order"], ref["radius"]
    queries = jnp.concatenate((ref["query_y"], ref["query_z"]))
    require(jnp.array_equal(inputs["queries"], queries) & jnp.array_equal(inputs["query_y"], ref["query_y"])
            & jnp.array_equal(inputs["query_z"], ref["query_z"]) & jnp.array_equal(stats["radius"], radius),
            "Physical query locations or radii changed")
    if maps["weights"].shape != (512, 3, WIDTHS[order], 3) or maps["stencil_indices"].shape != (512, WIDTHS[order]):
        raise RuntimeError("Unexpected component-major stencil-map dimensions")
    require(jnp.all((maps["stencil_indices"] >= 0) & (maps["stencil_indices"] < len(inputs["nodes"]))),
            "Saved stencil index is outside its physical cloud")
    require(jnp.all(maps["retained_rank"] == RANKS[order]) & jnp.all(maps["requested_rank"] == RANKS[order])
            & jnp.all(maps["inactive_padding_count"] == 0), "Complete requested polynomial space was not retained")
    truth, pred, mse = (stats[key] for key in ("pair_reference_s2", "pair_reconstructed_s2", "pair_increment_mse"))
    if any(value.shape != (256,) for value in (truth, pred, mse)):
        raise RuntimeError("Expected all 256 original physical query pairs")
    for value in (inputs["nodes"], queries, maps["weights"], truth, pred, mse):
        if value.dtype != jnp.float64:
            raise RuntimeError("Saved numerical inputs must be float64")
        require(jnp.all(jnp.isfinite(value)), "Nonfinite saved numerical input")
    require(jnp.all(truth > 0) & jnp.all(pred >= 0) & jnp.all(mse >= 0), "Invalid saved covariance energy")
    values, slopes, stability = calculate(inputs["nodes"], queries, maps["stencil_indices"], maps["weights"],
                                          truth, pred, mse, radius)
    require(stability["maximum_amplitude_bound_violation"] <= 1e-10, "Exact increment-amplitude MSE bound failed")
    values.update(radius_over_nominal_spacing=radius / row["spacing"],
                  radius_over_maximum_query_radius=radius / values["maximum_query_radius"],
                  maximum_query_radius_over_separation=values["maximum_query_radius"] / radius)
    stability.update(maximum_query_radius_over_nominal_spacing=stability["maximum_query_radius"] / row["spacing"],
                     median_query_radius_over_nominal_spacing=stability["median_query_radius"] / row["spacing"])
    values, slopes, stability = export((values, slopes, stability))
    radii, certified = export((radius, ref["certified_interval"]))
    per_bin = []
    for i in range(32):
        item = dict(case=row["case"], nominal_spacing=row["spacing"], spatial_nodes=row["nodes"],
            target_order=order, polynomial_degree=order - 1, policy=row["policy"], radius_bin=i,
            radius=radii[i], in_original_reference_certified_interval=certified[i])
        item.update({key: value[i] for key, value in values.items()})
        item.update({key: value[i] if i < 31 else None for key, value in slopes.items()})
        per_bin.append(item)
    log_r = jnp.log(radius[-4:]); centered = log_r - jnp.mean(log_r)
    mean_truth, mean_pred = (jnp.mean(a.reshape(32, 8), axis=1) for a in (truth, pred))
    fit = lambda a: jnp.sum(centered * jnp.log(a[-4:])) / jnp.sum(centered**2)
    summary = dict(case=row["case"], spacing=row["spacing"], nodes=row["nodes"], order=order, policy=row["policy"],
        stability=stability, fixed_radius_0p8=per_bin[-1],
        fixed_last_four_radius_bins=dict(radius_lower=radius[-4], radius_upper=radius[-1],
            fitted_reference_log_slope=fit(mean_truth), fitted_reconstructed_log_slope=fit(mean_pred),
            **masked_summary(jnp.arange(32) >= 28, truth, pred, mse)),
        nominal_radius_over_h_at_least_2=masked_summary(radius / row["spacing"] >= 2, truth, pred, mse),
        nominal_radius_over_h_at_least_4=masked_summary(radius / row["spacing"] >= 4, truth, pred, mse))
    return export(summary), per_bin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.study_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False); (output / "cases").mkdir()
    data, frozen = root / "increment_statistics", root / "study_source"
    paths = dict(analysis_source=Path(__file__).resolve(), source_manifest=frozen / "source_manifest.json",
                 protocol=data / "protocol.json", reference=data / "reference.npz")
    before = {}
    try:
        if sha(paths["source_manifest"]) != MANIFEST_SHA:
            raise RuntimeError("Frozen source manifest differs from the verified study")
        manifest = json.loads(paths["source_manifest"].read_text())
        paths.update({"source/" + key: frozen / key for key in manifest["source_sha256"]})
        protocol = json.loads(paths["protocol"].read_text())
        protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        cases = []
        for spacing in SPACINGS:
            for order in ORDERS:
                for policy in POLICIES:
                    case = f"irregular3d_h{spacing:g}_p{order}_{policy}"
                    row = json.loads((data / case / "result.json").read_text())
                    if (row["case"], row["spacing"], row["value_order"], row["policy"]) != (case, spacing, order, policy):
                        raise RuntimeError("Unexpected original case identity")
                    if not row["numerical_checks"]["full_requested_space_retained"] or not row["numerical_checks"]["numerical_checks_passed"]:
                        raise RuntimeError("Original scientific checks failed: " + case)
                    if row["protocol_sha256"] != protocol_hash or row["reference_npz_sha256"] != REFERENCE_SHA:
                        raise RuntimeError("Original protocol/reference provenance differs")
                    for label, path in (("input", data / row["input_artifact"]), ("result", data / case / "result.json"),
                                        ("maps", data / case / "spatial_diagnostics.npz"), ("statistics", data / case / "statistics.npz")):
                        if not path.resolve().is_relative_to(data):
                            raise RuntimeError("Input path escapes the original study")
                        paths[case + "/" + label] = path
                    cases.append(row)
        before = {key: sha(path) for key, path in paths.items()}
        if before["reference"] != REFERENCE_SHA:
            raise RuntimeError("Reference archive changed")
        for key, expected in manifest["source_sha256"].items():
            if before["source/" + key] != expected:
                raise RuntimeError("Frozen producer source changed: " + key)
        for row in cases:
            for label, key in (("input", "input_artifact_sha256"), ("maps", "spatial_diagnostics_npz_sha256"), ("statistics", "statistics_npz_sha256")):
                if before[row["case"] + "/" + label] != row[key]:
                    raise RuntimeError("Original artifact changed: " + row["case"] + "/" + label)
        shutil.copyfile(__file__, output / Path(__file__).name)
        write(output / "hashes_before.json", before)
        devices = jax.devices("gpu")
        if not devices or "nvidia" not in devices[0].device_kind.lower():
            raise RuntimeError("CUDA-only analysis; no CPU numerical fallback")
        jax.config.update("jax_default_device", devices[0])
        ref = archive(paths["reference"], ("radius", "query_y", "query_z", "certified_interval"))
        if ref["radius"].shape != (32,) or ref["query_y"].shape != (256, 3):
            raise RuntimeError("Unexpected original radius/query dimensions")
        require(ref["radius"][-1] == .8, "Largest original separation changed")
        summaries, rows = [], []
        for case in cases:
            summary, per_bin = analyze(data, case, ref)
            summaries.append(summary); rows.extend(per_bin)
            write(output / "cases" / (case["case"] + ".json"), dict(summary=summary, per_bin=per_bin))
            print(json.dumps(dict(case=case["case"], complete=True)), flush=True)
        after = {key: sha(path) for key, path in paths.items()}
        if before != after:
            raise RuntimeError("Authenticated source or input changed during analysis")
        with (output / "per_bin.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        write(output / "analysis.json", dict(complete=True, device=str(devices[0]), dtype="float64", jax_version=jax.__version__, cases=summaries,
            case_count=len(summaries), radius_row_count=len(rows), hashes_before=before, hashes_after=after,
            qualification="Original global scattered clouds only; fixed r=0.8 and fixed last four radii are unchanged. r/h cuts use NOMINAL node spacing, not a resolution guarantee. Last-four slope fits concern S2 versus separation, not convergence versus h. No original full-window outcome is replaced."))
    except BaseException as error:
        write(output / "failure.json", dict(type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
            hashes_before=before, hashes_after={key: sha(path) for key, path in paths.items() if path.exists()}))
        raise


if __name__ == "__main__":
    main()
