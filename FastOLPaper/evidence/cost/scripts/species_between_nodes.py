"""Species transport: reconstruct true velocity fields from the 7000 nodes at the 4388 remaining output locations
(local multiscale Matern, degree 4) and measure the mean relative error against the truth. Run once with the
campaign snapshot (SVD rank reduction) and once with the new code (certified QR / pivoted QR)."""
import json, sys, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from fastol_gpu_campaign import protocol, spatial_options
from kernelpack.operator_learning import FastOL, load_frozen_split
from kernelpack.operator_learning.datasets import load_fastol_paper_dataset
dev = jax.devices("gpu")[0]
case, label = "species_transport", sys.argv[1]
record = protocol(Path("/mnt/e/kernelpack-matlab-validation/experiments/fastol"), case, 500)
fixed = record["prescribed"]["sparse_c4_wendland"]
split = load_frozen_split(record["split_file"])
ids = split["validation"][:50].tolist()
data, *_ = load_fastol_paper_dataset(Path("/mnt/e/fluid_datasets"), case, ids, output_points=11388,
                                     input_points=int(fixed["input_spatial_points"]),
                                     representation=fixed["input_representation"], device=dev)
pts, vel = data.points, data.velocity          # (11388, 3), (50, 11388, 3)
nodes, rest = pts[:7000], pts[7000:]
options = spatial_options(case, fixed)
t = time.perf_counter()
r = FastOL.prepare_spatial_reconstructor(nodes, vel[:, :7000], spatial_queries=rest, spatial_batch_size=32,
                                         spatial_setup_bytes=1024**3, **options)
pred = jax.block_until_ready(r(vel[:, :7000])); setup = time.perf_counter() - t
truth = vel[:, 7000:]
err = jnp.linalg.norm(pred - truth, axis=(1, 2)) / jnp.linalg.norm(truth, axis=(1, 2))
np.save(f"/mnt/e/kernelpack-jax/.venv-wsl/_claude_jobs/logs/species_between_{label}.npy", np.asarray(pred))
print(json.dumps(dict(label=label, fields=len(ids), between_points=int(rest.shape[0]),
                      mean_rel_error=float(jnp.mean(err)), max_rel_error=float(jnp.max(err)), seconds=setup)))
