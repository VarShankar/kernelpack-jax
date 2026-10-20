"""Item 3 of the FastOL cost check: local vs global divergence-free reconstruction setup vs number of nodes m.
Global (PPKM-style): dense divergence-free C^4 Matern Gram on all dm unknowns + Cholesky (assembly and factor timed).
Local (paper): FastOL.prepare_spatial_reconstructor, degree 4, cubic PHS, batch 32. Scattered uniform nodes in the unit
square/cube. Cost only; one random field supplies the shapes. Each timing is a single run after a warm-up at small m."""
import json, sys, time
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from kernelpack.divfree.matern import divfree_matern_gram_matrix
from kernelpack.operator_learning.fastol import FastOL

dev = jax.devices("gpu")[0]
d = int(sys.argv[1]); ms = [int(a) for a in sys.argv[2:]]
glob_limit = 30000                     # dm beyond which one dense JAX call is known to run out of memory here

def nodes(m):
    return jnp.asarray(np.random.default_rng(m).random((m, d)), device=dev)

def global_setup(x):
    t = time.perf_counter()
    K = divfree_matern_gram_matrix(x, x, jnp.array([10.0]), jnp.array([1.0])).block_until_ready()
    ta = time.perf_counter() - t
    t = time.perf_counter()
    L = jnp.linalg.cholesky(K + 1e-10 * jnp.eye(K.shape[0])).block_until_ready()
    tf = time.perf_counter() - t
    ok = bool(jnp.all(jnp.isfinite(jnp.diagonal(L))))
    return ta, tf, ok

def local_setup(x):
    f = jnp.asarray(np.random.default_rng(1).standard_normal((1, x.shape[0], d)), device=dev)
    t = time.perf_counter()
    r = FastOL.prepare_spatial_reconstructor(x, f, order=4, use_periodic=False,
                                             spatial_batch_size=32, spatial_setup_bytes=1024**3)
    jax.block_until_ready(r.buffers)
    return time.perf_counter() - t, int(sum(r.buffers_sizes))

# warm-up (compilation) at a small size, not reported
global_setup(nodes(200)); local_setup(nodes(200))
for m in ms:
    row = dict(d=d, m=m, dm=d * m)
    try:
        row["local_setup_s"], row["local_buffer_bytes"] = local_setup(nodes(m))
        row["local_setup_warm_s"], _ = local_setup(nodes(m))
    except Exception as e:
        row["local_error"] = type(e).__name__ + ": " + str(e)[:150]
    if d * m <= glob_limit:
        try:
            ta, tf, ok = global_setup(nodes(m))
            row.update(global_assembly_s=ta, global_cholesky_s=tf, global_ok=ok, global_matrix_bytes=(d * m)**2 * 8)
        except Exception as e:
            row["global_error"] = type(e).__name__ + ": " + str(e)[:150]
    else:
        row["global_skipped"] = "dense dm x dm exceeds single-call device memory"
    print(json.dumps(row), flush=True)
