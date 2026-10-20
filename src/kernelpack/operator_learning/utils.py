from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
from jax import numpy as jnp, random as jr
import numpy as np


def get_batch(epoch_key: jax.Array, data: tuple[jax.Array, ...], batch_index: int, batch_size: int) -> list[jax.Array]:
    batch: list[jax.Array] = []
    for dat in data:
        dat_perm = jr.permutation(epoch_key, dat)
        batch.append(jax.lax.dynamic_slice_in_dim(dat_perm, batch_index * batch_size, batch_size))
    return batch


def is_trainable(x: object) -> bool:
    return eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating)


def create_lifted_module(base_layer, lift_dim: int, key: jax.Array):
    keys = jr.split(key, int(lift_dim))
    return eqx.filter_vmap(lambda layer_key: base_layer(key=layer_key))(keys)


def shuffle(x: np.ndarray, y: np.ndarray, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    idx = np.arange(len(x))
    np.random.shuffle(idx)
    return x[idx], y[idx]


class UnitGaussianNormalizer:
    def __init__(self, x: jax.Array, axis: int = 0, eps: float = 1e-7):
        self.mean = jnp.mean(x, axis=axis, keepdims=True)
        self.std = jnp.std(x, axis=axis, keepdims=True)
        self.eps = eps

    @partial(jax.jit, static_argnums=(0,))
    def encode(self, x: jax.Array) -> jax.Array:
        return (x - self.mean) / (self.std + self.eps)

    @partial(jax.jit, static_argnums=(0,))
    def decode(self, x: jax.Array) -> jax.Array:
        return (x * (self.std + self.eps)) + self.mean
