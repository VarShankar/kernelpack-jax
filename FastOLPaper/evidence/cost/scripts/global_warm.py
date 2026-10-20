"""Warm timings of the global (PPKM-style) divergence-free reconstruction setup: dense dm x dm C^4 Matern
Gram + Cholesky, median of 3 runs after a compiling run at the same size."""
import json, sys, time
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from kernelpack.divfree.matern import divfree_matern_gram_matrix
dev = jax.devices("gpu")[0]
d = int(sys.argv[1])
gram = jax.jit(lambda x: divfree_matern_gram_matrix(x, x, jnp.array([10.0]), jnp.array([1.0])))
chol = jax.jit(lambda K: jnp.linalg.cholesky(K + 1e-10 * jnp.eye(K.shape[0])))
for m in [int(a) for a in sys.argv[2:]]:
    x = jnp.asarray(np.random.default_rng(m).random((m, d)), device=dev)
    try:
        K = gram(x).block_until_ready(); chol(K).block_until_ready()
        ta, tf = [], []
        for _ in range(3):
            t = time.perf_counter(); K = gram(x).block_until_ready(); ta.append(time.perf_counter() - t)
            t = time.perf_counter(); L = chol(K).block_until_ready(); tf.append(time.perf_counter() - t)
        ok = bool(jnp.all(jnp.isfinite(jnp.diagonal(L))))
        row = dict(d=d, m=m, dm=d * m, global_assembly_warm_s=sorted(ta)[1], global_cholesky_warm_s=sorted(tf)[1],
                   global_ok=ok, global_matrix_bytes=(d * m)**2 * 8)
        del K, L
    except Exception as e:
        row = dict(d=d, m=m, dm=d * m, global_error=type(e).__name__ + ": " + str(e)[:120])
    print(json.dumps(row), flush=True)
