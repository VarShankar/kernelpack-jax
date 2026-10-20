"""Cost-only replay of the frozen FastOL GPU campaign with a fill-reducing training-input order.

Every model is rebuilt from its frozen selection (radius, regularization, shape) without refitting.
Only the order of the training inputs and the cuDSS reordering change, which leaves every prediction
unchanged up to rounding (checked against the campaign's saved predictions). For inputs of numerical
rank one, sorting by the principal coordinate makes the compactly supported kernel matrix banded
(an interval graph), so Cholesky creates no fill with reordering 'none'. For n_z >= 2 the script
keeps the ordering with the fewest factor entries among AMD, cuDSS's default and the sorted order.
Cost columns follow the campaign exactly (scripts/fastol_gpu_campaign.py, report_one_cell.py).
"""
import argparse, json, math, os, statistics, sys, time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--N", type=int, nargs="+", default=[500, 1000, 5000, 10000])
    parser.add_argument("--campaign", type=Path,
        default=Path("/mnt/e/kernelpack-jax/Outputs_and_results/fastol_matched_safe_gpu_20261016/cells"))
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/e/fluid_datasets"))
    parser.add_argument("--baseline-root", type=Path, default=Path("/mnt/e/kernelpack-matlab-validation/experiments/fastol"))
    parser.add_argument("--repetitions", type=int, default=11)
    parser.add_argument("--rule", choices=("band", "fewest", "default"), default="band",
                        help="band: sorted order for n_z=1, fewest factor entries otherwise; fewest: always "
                             "the fewest factor entries; default: cuDSS default (nested dissection)")
    parser.add_argument("--compare", action="store_true", help="measure every candidate ordering")
    parser.add_argument("--compare-models", nargs="+", default=["fixed_affine"])
    args = parser.parse_args()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import numpy as np
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from fastol_gpu_campaign import protocol, load_fields, spatial_options, counted_model_arrays
    from kernelpack.operator_learning import FastOL, KernelTrainingConfig, load_frozen_split
    from kernelpack.operator_learning.trainable import train_compact_kernel, freeze_compact_operator
    from kernelpack.accelerators.jax_spatial import enable_compilation_cache
    enable_compilation_cache()
    device = jax.devices("gpu")[0]
    models = ("fixed_no_poly", "trainable_no_poly", "fixed_affine", "trainable_affine")
    for N in args.N:
        cell = args.campaign / f"{args.case}_N{N}"
        selection = json.loads((cell / "selection.json").read_text())["models"]
        record = protocol(args.baseline_root, args.case, N)
        fixed = record["prescribed"]["sparse_c4_wendland"]
        split = load_frozen_split(record["split_file"])
        training_ids = split["train"][:N].tolist(); test_ids = split["test"].tolist()
        inputs, points, fields, *_ = load_fields(args.data_root, args.case, training_ids, fixed, device)
        q, _, truth, *_ = load_fields(args.data_root, args.case, test_ids, fixed, device)
        options = spatial_options(args.case, fixed)
        reconstructor = FastOL.prepare_spatial_reconstructor(points, fields, spatial_batch_size=32,
                                                              spatial_setup_bytes=1024**3, **options)
        jax.block_until_ready(reconstructor.buffers)
        ids = np.asarray(training_ids)

        def build(name, perm, reordering):
            choice = selection[name]
            config = KernelTrainingConfig(**{**choice["config"], "cudss_reordering": reordering})
            same = bool(np.array_equal(perm, np.arange(N)))     # avoid copying the large input/field arrays
            x_p, y_p = (inputs, fields) if same else (inputs[perm], fields[perm])
            fit = train_compact_kernel(x_p, y_p, training_ids=jnp.asarray(ids[perm], device=device),
                                       radius=float(choice["radius"]), nugget=1e-6, config=config, optimize=False)
            start = time.perf_counter()
            operator = freeze_compact_operator(fit, np.asarray(choice["parameters"], float))
            model = FastOL(x_p, points, y_p, epsilon=1., operator_model=operator, backend="cudss",
                           spatial_batch_size=32, spatial_setup_bytes=1024**3, spatial_reconstructor=reconstructor,
                           **options)
            jax.block_until_ready(model.device_state())
            return fit, model, time.perf_counter() - start

        identity = np.arange(N)
        probe, model, _ = build(models[0], identity, "amd"); model.close()
        z = np.asarray(jax.device_get(probe.z)); n_z = z.shape[1]
        del probe
        order = np.argsort(z[:, 0], kind="stable")
        def measure(name, label, perm, reordering, extra=None):
            _, model, setup = build(name, perm, reordering)
            try:
                predictor = model.prepare_predictor(operator_evaluation="lagrange", spatial_reconstructor=reconstructor)
                jax.block_until_ready(predictor.buffers)
                batch = 32
                model.predict(q, query_batch_size=batch).block_until_ready()
                cached = model._prepared_prediction
                batch_predictor, runner = cached[1], cached[4][batch]
                padded_q = jnp.pad(q, ((0, (-len(q)) % batch), (0, 0)), mode="edge").reshape(-1, batch, q.shape[-1])
                def run(queries, buffers):
                    values = runner(queries, buffers)
                    return values.reshape((-1,) + values.shape[2:])[:len(q)]
                timed_run = jax.jit(run)
                prediction = timed_run(padded_q, batch_predictor.buffers).block_until_ready()
                durations = []
                for _ in range(args.repetitions):
                    t = time.perf_counter()
                    with jax.transfer_guard("disallow"):
                        prediction = timed_run(padded_q, batch_predictor.buffers).block_until_ready()
                    durations.append(time.perf_counter() - t)
                native = model.operator_model.sparse_factor.storage_statistics()
                visible = counted_model_arrays(model, batch_predictor)["visible_model_array_bytes"]
                owned = 4 * (N + 1) + 12 * native["actual_nnz"]
                graph = 2 * N * selection[name]["config"]["rhs_block_size"] * 8
                saved = np.load(cell / f"{name}_predictions.npz")["prediction"]
                pred = np.asarray(jax.device_get(prediction))
                axes = (1, 3) if truth.ndim == 4 else (1, 2)
                err = jnp.linalg.norm(prediction - truth, axis=axes) / jnp.linalg.norm(truth, axis=axes)
                row = dict(case=args.case, N=N, model=name, n_z=int(n_z), input_order=label, cudss_reordering=reordering,
                           actual_kernel_nnz=native["actual_nnz"], native_factor_nnz=native["factor_nnz"],
                           native_permanent_device_bytes=native["permanent_device_bytes"],
                           visible_model_array_bytes=visible, owned_csr_bytes=owned, rhs_solution_bytes=graph,
                           accounted_resident_storage_ESTIMATE_bytes=visible + native["permanent_device_bytes"] + owned + graph,
                           warm_100query_seconds=durations, warm_100query_median_seconds=statistics.median(durations),
                           operator_setup_seconds=setup, test_error=float(jnp.mean(err)),
                           prediction_rel_diff_vs_campaign=float(np.linalg.norm(pred - saved) / np.linalg.norm(saved)))
                row.update(extra or {})
                print(json.dumps(row), flush=True)
            finally:
                model.close()

        candidates = (("input", identity, "amd"), ("input", identity, "default"),
                      ("sorted_z1", order, "none"), ("sorted_z1", order, "default"))
        for name in models:
            if args.compare:
                if name in args.compare_models:
                    for label, perm, reo in candidates:
                        measure(name, label, perm, reo, dict(compare=True))
                continue
            if args.rule == "band" and n_z == 1:
                measure(name, "sorted_z1", order, "none")
                continue
            if args.rule == "default":
                measure(name, "input", identity, "default")
                continue
            trial = []
            for label, perm, reo in candidates[:3]:
                _, m, _ = build(name, perm, reo)
                trial.append((m.operator_model.sparse_factor.storage_statistics()["factor_nnz"], label, perm, reo))
                m.close()
            trial.sort(key=lambda t: t[0])
            measure(name, trial[0][1], trial[0][2], trial[0][3],
                    dict(trial_factor_nnz={f"{t[1]}/{t[3]}": int(t[0]) for t in trial}))
        reconstructor = None


if __name__ == "__main__":
    main()
