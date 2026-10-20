"""Cost-only re-timing of the shared spatial preparation of the paper's GPU campaign (N=10000), with the
SVD-free polynomial reduction and the persistent XLA compilation cache. Same node sets, options, fields
and batch settings as scripts/fastol_gpu_campaign.py. Reports the first (compiling) and a warm run."""
import json, sys, time
from pathlib import Path
import jax
jax.config.update("jax_enable_x64", True)
from fastol_gpu_campaign import protocol, load_fields, spatial_options
from kernelpack.operator_learning import FastOL, load_frozen_split
from kernelpack.accelerators.jax_spatial import enable_compilation_cache
enable_compilation_cache()
dev = jax.devices("gpu")[0]
case = sys.argv[1]
N = 10000
record = protocol(Path("/mnt/e/kernelpack-matlab-validation/experiments/fastol"), case, N)
fixed = record["prescribed"]["sparse_c4_wendland"]
split = load_frozen_split(record["split_file"])
_, points, fields, *_ = load_fields(Path("/mnt/e/fluid_datasets"), case, split["train"][:N].tolist(), fixed, dev)
options = spatial_options(case, fixed)
out = dict(case=case, m=int(points.shape[0]), d=int(points.shape[1]), options={k: str(v) for k, v in options.items()})
for run in ("first", "warm"):
    t = time.perf_counter()
    r = FastOL.prepare_spatial_reconstructor(points, fields, spatial_batch_size=32,
                                             spatial_setup_bytes=1024**3, **options)
    jax.block_until_ready(r.buffers)
    out[run + "_s"] = time.perf_counter() - t
    out["buffer_bytes"] = int(sum(r.buffers_sizes))
    del r
print(json.dumps(out), flush=True)
