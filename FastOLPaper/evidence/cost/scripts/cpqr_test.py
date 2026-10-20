"""Column-pivoted QR on the GPU for the reduced polynomial factors: speed, and rank agreement with the SVD
on the species-transport stencils (degree 4, 3-D; many rank-deficient)."""
import json, time, sys
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
from jax import lax
jax.config.update("jax_enable_x64", True)
res = {"jax": jax.__version__}
A = jnp.asarray(np.linalg.qr(np.random.default_rng(0).standard_normal((250, 177, 85)), mode="r"))
def timed(name, f, x):
    try:
        g = jax.jit(f); jax.block_until_ready(g(x))
        t = time.perf_counter(); jax.block_until_ready(g(x)); res[name] = time.perf_counter() - t
    except Exception as e:
        res[name] = type(e).__name__ + ": " + str(e)[:160]
timed("cpqr_85x85_x250", lambda a: lax.linalg.qr(a, pivoting=True, full_matrices=False), A)
timed("svd_85x85_x250", lambda a: jnp.linalg.svd(a, compute_uv=False), A)
print(json.dumps(res), flush=True)
if not isinstance(res["cpqr_85x85_x250"], float):
    sys.exit()
# rank agreement on species transport
from fastol_gpu_campaign import protocol, load_fields, spatial_options
from kernelpack.operator_learning import load_frozen_split
from kernelpack.operator_learning.local_dfphs import LocalDFPHSDecoder
from kernelpack.domain.stencils import select_stencils
from kernelpack.divfree.core import divfree_polynomial_stack
dev = jax.devices("gpu")[0]
case = "species_transport"
record = protocol(Path("/mnt/e/kernelpack-matlab-validation/experiments/fastol"), case, 500)
fixed = record["prescribed"]["sparse_c4_wendland"]
split = load_frozen_split(record["split_file"])
_, points, _, *_ = load_fields(Path("/mnt/e/fluid_datasets"), case, split["train"][:2].tolist(), fixed, dev)
dec = LocalDFPHSDecoder(points, **spatial_options(case, fixed))
basis = dec.polynomial_row_basis
def reduced(n):
    c = jnp.mean(n, 0); w = jnp.max(jnp.linalg.norm(n - c, axis=1))
    raw, _ = divfree_polynomial_stack((n - c) / w, poly_degree=dec.order)
    return jnp.linalg.qr(raw @ basis, mode="r")
agree = disagree = 0; t_qr = t_svd = 0.0; ranks = []
for s in range(0, len(points), 250):
    _, nodes, _ = select_stencils(dec.points, points[s:s + 250], k=dec.stencil_size, periodic=dec.periodic,
                                  lengths=dec.maximum - dec.minimum, center_indices=None, use_periodic=False)
    R = jax.jit(jax.vmap(reduced))(nodes)
    t = time.perf_counter(); _, Rp, _ = jax.block_until_ready(jax.jit(lambda a: lax.linalg.qr(a, pivoting=True, full_matrices=False))(R)); t_qr += time.perf_counter() - t
    t = time.perf_counter(); sv = jax.block_until_ready(jax.jit(lambda a: jnp.linalg.svd(a, compute_uv=False))(R)); t_svd += time.perf_counter() - t
    d = jnp.abs(jnp.diagonal(Rp, axis1=-2, axis2=-1))
    r_qr = jnp.sum(d > 1e-12 * d[:, :1], axis=1)
    r_svd = jnp.sum(sv > 1e-12 * sv[:, :1], axis=1)
    agree += int(jnp.sum(r_qr == r_svd)); disagree += int(jnp.sum(r_qr != r_svd)); ranks += np.asarray(r_svd).tolist()
print(json.dumps(dict(case=case, stencils=len(points), rank_agree=agree, rank_disagree=disagree,
                      min_rank=int(min(ranks)), cpqr_s=t_qr, svd_s=t_svd)), flush=True)
