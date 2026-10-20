"""Six CUDA refinement cells using the original Fourier field and all pairs.

Only two new GLOBAL scattered clouds are generated. This extends the original
production five-mode, stencil-relative Matérn experiment; it does not fit an
operator, change the target, or run PHS/periodic/global comparison experiments.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback
import argparse
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np  # Archive I/O only.
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import fastol_spatial_statistics as statistics
import fastol_spatial_properties as properties
import fastol_spatial_study_core as spatial

REFERENCE_SHA = "4294130ac653551052e7d48b867cd368d1c21483be1bd63ba89fff457f6b7e54"
ORIGINAL_PROTOCOL_SHA = "9de9a798fd196cb972c6d91733710acd8dbb16d1f5ea6313307e6cddc83e025f"
SPACINGS, ORDERS = (.075, .05), (2, 4, 6)


@jax.jit
def guarded_pair_outputs(points, weights, indices, gradients, fields, y, z, blocks, wavevectors, prior_weights):
    """Compile slice indices and checks with the existing GPU numerical routines."""
    mapping = SimpleNamespace(weights=weights, stencil_indices=indices)
    energies, py, pz, ty, tz = statistics.pair_statistics(points, mapping, y, z, blocks, pair_batch=4)
    values = jnp.transpose(fields, (0, 2, 1))
    predicted = spatial.apply_map_arrays(weights, indices, values)
    jacobian = spatial.apply_map_jacobian_arrays(gradients, indices, values)
    gap = jnp.linalg.norm(jnp.transpose(predicted, (0, 2, 1))-jnp.concatenate((py, pz))) / jnp.maximum(
        jnp.linalg.norm(predicted), jnp.finfo(points.dtype).tiny)
    target_gap = jnp.max(jnp.abs(energies[:, 1]-statistics.exact_increment(wavevectors, prior_weights, z-y)))
    divergence = jnp.linalg.norm(jnp.einsum("qaad->qd", jacobian))/jnp.maximum(jnp.linalg.norm(jacobian), 1.)
    return energies, py, pz, ty, tz, predicted, jacobian, gap, target_gap, divergence


def source_paths():
    root = Path(__file__).resolve().parents[1]
    paths = {key: root/key for key in statistics._sources()}
    paths["scripts/"+Path(__file__).name] = Path(__file__).resolve()
    return paths


def hashes(paths):
    return {key: statistics._digest_file(path) for key, path in paths.items()}


def assert_unchanged(paths, expected):
    if hashes(paths) != expected:
        raise RuntimeError("A pinned source or original input changed during the extension")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.study_root, args.output = args.study_root.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    output, original = args.output, args.study_root/"increment_statistics"
    paths, before, provenance, current_case = {}, {}, {}, None
    try:
        paths = {"current/"+key:path for key,path in source_paths().items()}
        paths.update(reference=original/"reference.npz", original_protocol=original/"protocol.json")
        old_protocol = json.loads(paths["original_protocol"].read_text())
        old_digest = hashlib.sha256(statistics._canonical(old_protocol)).hexdigest()
        if old_digest != ORIGINAL_PROTOCOL_SHA:
            raise RuntimeError("Original full-study protocol changed")
        paths.update({"original_source/"+key:args.study_root/"study_source"/key
                      for key in old_protocol["source_sha256"]})
        before = hashes(paths)
        if before["reference"] != REFERENCE_SHA or old_protocol["reference_npz_sha256"] != REFERENCE_SHA:
            raise RuntimeError("Original Fourier reference changed")
        for key, expected in old_protocol["source_sha256"].items():
            if before["original_source/"+key] != expected:
                raise RuntimeError("Original frozen source changed: "+key)
        if os.environ.get("JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS", "").lower() != "false":
            raise RuntimeError("Launch with JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS=false for static polynomial metadata")
        origins = statistics._frozen_module_origins()
        device = properties.configure_gpu()
        with np.load(paths["reference"], allow_pickle=False) as archive:
            reference = {key:jax.device_put(archive[key], device) for key in archive.files}
        if {key:statistics._digest_array(value) for key,value in reference.items()} != old_protocol["reference_array_sha256"]:
            raise RuntimeError("Original field/query/reference array pins disagree")
        prior = statistics.FourierPrior(**{key:reference[key] for key in (
            "wavevectors", "weights", "frames", "cosine_draws", "sine_draws", "physical_matern_base_length")})
        radius, directions = reference["radius"], reference["directions"]
        y, z = reference["query_y"], reference["query_z"]
        if len(y) != 256 or len(z) != 256 or len(radius) != 32 or len(directions) != 8:
            raise RuntimeError("The complete original pair set is required")
        queries = jnp.concatenate((y,z))
        blocks = statistics._prior_blocks(prior, 64)
        source_hashes = {key[8:]:value for key,value in before.items() if key.startswith("current/")}
        protocol = dict(schema="fastol_local_matern_global_cloud_refinement_v1",
            spacings=SPACINGS, orders=ORDERS, expected_cells=6, dimension=3, cloud_seed=1729,
            reference_npz_sha256=REFERENCE_SHA, original_protocol_sha256=ORIGINAL_PROTOCOL_SHA,
            reference_array_sha256=old_protocol["reference_array_sha256"], source_sha256=source_hashes,
            module_origins=origins, field="unchanged finite Fourier covariance and original eight draws",
            query_pairs=256, query_endpoints=512, pair_batch=4, mode_batch=64, map_batch=32,
            field_batch=256, kernel="five-mode divergence-free Matérn plus complete DF polynomials",
            matern_base_length_factor=16., matern_num_modes=5, matern_gamma=8/3,
            physical_scale_policy="each assigned stencil radius times tau16; no fixed-band substitution",
            assignment="unchanged nearest physical node and shared boundary-aware CFPU stencil rules",
            polynomial_degrees=[1,3,5], polynomial_dimensions=[11,50,133], stencil_sizes=[9,35,91],
            compute_gradients=True, allow_pseudoinverse=False, numerical_divergence_tolerance=1e-9,
            amplitude_tolerance=old_protocol["amplitude_tolerance"], device=str(device), dtype="float64",
            jax_version=jax.__version__, simplified_jaxpr_constants="false",
            timing_qualification="No timing benchmark. Internal shared-routine wall metadata includes compilation and possible GPU contention.",
            scientific_acceptance="Report all physical errors; only numerical, provenance and identity checks gate acceptance.")
        protocol_sha = hashlib.sha256(statistics._canonical(protocol)).hexdigest()
        provenance = dict(protocol_sha256=protocol_sha, source_sha256=source_hashes,
                          reference_npz_sha256=REFERENCE_SHA)
        statistics._write_json(output/"protocol.json", protocol)
        statistics._write_json(output/"manifest_before.json", before)
        # Preserve the exact source used, including this orchestration, without
        # modifying or replacing the original study's frozen source tree.
        for key,path in source_paths().items():
            target = output/"source"/key
            target.parent.mkdir(parents=True, exist_ok=True)
            with path.open("rb") as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            if statistics._digest_file(target) != source_hashes[key]:
                raise RuntimeError("Source snapshot differs: "+key)
        assert_unchanged(paths, before)
        results = []
        for spacing in SPACINGS:
            current_case = None
            assert_unchanged(paths, before)
            print(json.dumps(dict(stage="global_cloud_generation", spacing=spacing)), flush=True)
            try:
                cloud = properties.make_irregular_cloud(3, spacing, seed=1729)
            except BaseException as error:
                properties._preserve_cloud_failure(output, error, source_sha256=source_hashes,
                    protocol_sha256=protocol_sha, specification=dict(dimension=3, spacing=spacing, seed=1729))
                raise
            points = cloud.points
            try:
                sample_executable = jax.jit(lambda p, b: statistics.sample_fields(p, b, point_batch=256)).lower(points, blocks).compile()
                with jax.transfer_guard("disallow"):
                    fields = sample_executable(points, blocks)
                fields.block_until_ready()
                properties._assert_device_tree((points, fields, queries))
                inputs = statistics._input_provenance(args, f"irregular3d_h{spacing:g}_inputs.npz",
                    dict(nodes=points, queries=queries, nodal_values=fields, boundary=cloud.boundary,
                         query_y=y, query_z=z, **(cloud.geometry_evidence or {})))
            except BaseException as error:
                properties._preserve_cloud_failure(output, error, source_sha256=source_hashes,
                    protocol_sha256=protocol_sha, specification=dict(dimension=3, spacing=spacing, seed=1729),
                    additional_arrays=dict(nodes=points, queries=queries, **(cloud.geometry_evidence or {})),
                    stage_metadata=dict(stage="original_field_sampling_or_input_serialization", cloud=cloud.metadata))
                raise
            statistics._write_json(output/f"irregular3d_h{spacing:g}_cloud.json", cloud.metadata)
            print(json.dumps(dict(stage="global_cloud_ready", spacing=spacing, nodes=len(points))), flush=True)
            for order in ORDERS:
                assert_unchanged(paths, before)
                name = f"irregular3d_h{spacing:g}_p{order}_ram_local_tau16"
                current_case = output/name
                current_case.mkdir()
                mapping, report = None, None
                try:
                    mapping = spatial.build_map(points, queries, degree=order-1, kernel="matern",
                        assignment="nearest-node", batch_size=32, stencil_domain=cloud.stencil_domain,
                        allow_pseudoinverse=False, compute_gradients=True,
                        matern_base_length_factor=16., matern_num_modes=5, matern_gamma=8/3)
                    mapping.weights.block_until_ready()
                    properties._assert_device_tree((mapping.weights, mapping.weight_gradients, mapping.diagnostics))
                    if (mapping.metadata["requested_rank"] != {2:11,4:50,6:133}[order]
                        or mapping.stencil_indices.shape[1] != {2:9,4:35,6:91}[order]
                        or mapping.metadata["batch_size"] != 32):
                        raise RuntimeError("Production polynomial/stencil/batch dimensions changed")
                    diagnostics = statistics.summarize_map(mapping)
                    diagnostics_sha = statistics._export_npz(current_case/"spatial_diagnostics.npz",
                        center_indices=mapping.center_indices, stencil_indices=mapping.stencil_indices,
                        weights=mapping.weights, weight_gradients=mapping.weight_gradients, **mapping.diagnostics)
                    statistics._write_json(current_case/"spatial_checks.json", diagnostics)
                    if not diagnostics["numerical_checks_passed"]:
                        raise RuntimeError("Original full-rank, solver or polynomial reproduction checks failed")
                    divergence_ok = (jnp.all(mapping.diagnostics["gradient_finite"])
                        & jnp.all(mapping.diagnostics["derivative_defined"])
                        & jnp.all(mapping.diagnostics["query_node_min_distance"]>1e-12)
                        & jnp.all(mapping.diagnostics["divergence_weight_normalized"]<=1e-9))
                    if not bool(statistics._scalar(divergence_ok)):
                        raise RuntimeError("Fixed-stencil AD divergence check failed away from nodes")
                    pair_args = (points, mapping.weights, mapping.stencil_indices, mapping.weight_gradients,
                                 fields, y, z, blocks, prior.wavevectors, prior.weights)
                    pair_executable = guarded_pair_outputs.lower(*pair_args).compile()
                    with jax.transfer_guard("disallow"):
                        energies,py,pz,ty,tz,predicted,jacobian,gap,target_gap,field_divergence = pair_executable(*pair_args)
                    metrics, arrays = statistics._report_metrics(energies,py,pz,ty,tz,
                        bins=32,directions=8,radius=radius,interval=reference["certified_interval"],
                        epsilon=old_protocol["amplitude_tolerance"])
                    statistics_sha = statistics._export_npz(current_case/"statistics.npz",radius=radius,
                        query_y=y,query_z=z,energies=energies,nodal_route_prediction=predicted,
                        nodal_route_jacobian=jacobian,**arrays)
                    numerical_ok = bool(statistics._scalar((gap<=1e-9)&(target_gap<=1e-11)&(field_divergence<=1e-9)))
                    if not numerical_ok or not metrics["covariance_bound_check_passed"]:
                        raise RuntimeError("Fourier feature/nodal/covariance/field-divergence identity failed")
                    report = dict(case=name,dimension=3,geometry="irregular_F3_domain",spacing=spacing,
                        nodes=len(points),value_order=order,polynomial_degree=order-1,
                        stencil_size=mapping.stencil_indices.shape[1],policy="ram_local_tau16",
                        assignment="nearest-node",kernel="five-scale divergence-free Matern + polynomial",
                        fixed_physical_matern_base_length=None,cloud=cloud.metadata,map_metadata=mapping.metadata,
                        scalar_prior_s2_identity_max_gap=float(statistics._scalar(target_gap)),
                        manufactured_feature_vs_nodal_route_relative_gap=float(statistics._scalar(gap)),
                        manufactured_normalized_divergence=float(statistics._scalar(field_divergence)),
                        reference_has_certified_two_thirds_interval=old_protocol["reference_two_thirds_interval_resolved"],
                        numerical_checks=diagnostics,**metrics,**statistics._cell_pins(provenance,inputs),
                        statistics_npz_sha256=statistics_sha,spatial_diagnostics_npz_sha256=diagnostics_sha,
                        timing_qualification=protocol["timing_qualification"])
                    assert_unchanged(paths, before)
                    statistics._write_json(current_case/"result.json",report)
                    results.append(report)
                    statistics._write_json(output/"progress.json",dict(completed_cells=len(results),
                        expected_cells=6,last_completed=name,results=results),immutable=False)
                    print(json.dumps(dict(stage="refinement_cell_complete",case=name,
                        resolved_s2_error=report["resolved_interval_max_relative_s2_error"])),flush=True)
                except BaseException as error:
                    statistics._preserve_failure(current_case,error,provenance=provenance,inputs=inputs,
                        nodes=points,queries=queries,nodal_values=fields,mapping=mapping,
                        extra_arrays=dict(query_y=y,query_z=z,boundary=cloud.boundary),candidate_report=report)
                    raise
        assert_unchanged(paths, before)
        statistics._write_json(output/"completion.json",dict(completed=True,statistical_cells=6,
            protocol_sha256=protocol_sha,reference_npz_sha256=REFERENCE_SHA,results=results,
            result_sha256={p.relative_to(output).as_posix():statistics._digest_file(p)
                           for p in sorted(output.glob("*/result.json"))},
            hashes_before=before,hashes_after=hashes(paths),
            qualification="Six numerical cells complete. Physical errors are reported without a preservation pass claim. No timing/scalability, operator-learning, PHS, periodic or manuscript result is produced."))
    except BaseException as error:
        statistics._write_json(output/"failure.json",dict(error_type=type(error).__name__,error=str(error),
            traceback=traceback.format_exc(),failed_case=current_case,provenance=provenance,
            hashes_before=before,hashes_after={k:statistics._digest_file(p) for k,p in paths.items() if p.exists()}))
        raise


if __name__ == "__main__":
    main()
