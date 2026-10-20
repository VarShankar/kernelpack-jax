"""Grid vs dense stencil search on a paper node set (prepared reconstructor buffers identical?)."""
import json, sys
from pathlib import Path
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from fastol_gpu_campaign import protocol, load_fields, spatial_options
from kernelpack.operator_learning import FastOL, load_frozen_split
import kernelpack.operator_learning.local_dfphs as L
dev = jax.devices("gpu")[0]
case = sys.argv[1]
record = protocol(Path("/mnt/e/kernelpack-matlab-validation/experiments/fastol"), case, 500)
fixed = record["prescribed"]["sparse_c4_wendland"]
split = load_frozen_split(record["split_file"])
_, points, fields, *_ = load_fields(Path("/mnt/e/fluid_datasets"), case, split["train"][:4].tolist(), fixed, dev)
options = spatial_options(case, fixed)
out = {}
for label, thr in (("grid", 4096), ("dense", 10**9)):
    L._GRID_KNN_NODES = thr
    r = FastOL.prepare_spatial_reconstructor(points, fields, spatial_batch_size=32, spatial_setup_bytes=1024**3, **options)
    out[label] = [np.asarray(b) for b in r.buffers[:2]]
print(json.dumps(dict(case=case, indices_identical=bool(np.array_equal(out["grid"][1], out["dense"][1])),
                      weights_max_abs_diff=float(np.max(np.abs(out["grid"][0] - out["dense"][0]))))))
