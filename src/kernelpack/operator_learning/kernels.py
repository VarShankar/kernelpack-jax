from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial

import equinox as eqx
import jax
from jax import numpy as jnp, random as jr


class KernelBaseClass(ABC):
    @abstractmethod
    def eval(self, x: jax.Array, y: jax.Array) -> jax.Array:
        raise NotImplementedError

    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        if x.ndim == 1 or y.ndim == 1:
            ndims = 1
        else:
            ndims = x.shape[-1]
        x_flat = x.reshape(-1, ndims)
        y_flat = y.reshape(-1, ndims)
        return jax.vmap(jax.vmap(self.eval, (0, None)), (None, 0))(y_flat, x_flat)


class GaussianKernel(eqx.Module, KernelBaseClass):
    scale: jax.Array

    def __init__(self, *, key: jax.Array, **_kwargs):
        key, _ = jr.split(key)
        self.scale = jr.uniform(key, minval=-3.0, maxval=0.0)

    def eval(self, x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.exp(-0.5 * (x - y) ** 2 / jnp.exp(2 * self.scale)).sum()


class AnisotropicGaussianKernel(eqx.Module, KernelBaseClass):
    scale: jax.Array
    ndims: int

    def __init__(self, *, ndims: int, key: jax.Array, **_kwargs):
        scale_dim = int(ndims * (ndims + 1) / 2)
        self.scale = jr.uniform(key, (scale_dim,), minval=-3.0, maxval=0.0)
        self.ndims = int(ndims)

    def eval(self, x: jax.Array, y: jax.Array) -> jax.Array:
        lower = jnp.zeros((self.ndims, self.ndims)).at[jnp.tril_indices(self.ndims)].set(self.scale)
        scale = lower @ lower.T
        r_scaled = (x - y) @ scale @ (x - y)
        return jnp.exp(-0.5 * r_scaled)


class NonstationaryGaussianKernel(eqx.Module, KernelBaseClass):
    scale: eqx.Module
    ndims: int

    def __init__(self, *, ndims: int, latent_dim: int, key: jax.Array, **_kwargs):
        keys = jr.split(key, 2)
        self.scale = eqx.nn.Sequential(
            [
                eqx.nn.Linear(ndims, latent_dim, key=keys[0]),
                eqx.nn.Lambda(jax.nn.selu),
                eqx.nn.Linear(latent_dim, 1, key=keys[1]),
                eqx.nn.Lambda(jax.nn.softplus),
            ]
        )
        self.ndims = int(ndims)

    def eval(self, x: jax.Array, y: jax.Array) -> jax.Array:
        sx = self.scale(x).squeeze()
        sy = self.scale(y).squeeze()
        t1 = jnp.sqrt((2.0 * sx * sy) / (sx**2 + sy**2))
        t2 = jnp.exp(-jnp.sum((x - y) ** 2) / (sx**2 + sy**2))
        return t1 * t2


class GaussianSpectralMixtureKernel(eqx.Module, KernelBaseClass):
    q: int
    weights: jax.Array
    freqs: jax.Array
    scales: jax.Array

    def __init__(self, *, q: int, ndims: int, key: jax.Array, **_kwargs):
        key1, key2, key3 = jr.split(key, 3)
        self.q = int(q)
        self.weights = jr.uniform(key1, (self.q, 1), maxval=0.1, minval=-3.0)
        self.freqs = jr.uniform(key2, (self.q, ndims), maxval=0.1, minval=-3.0)
        self.scales = jr.uniform(key3, (self.q,), maxval=0.1, minval=-3.0)

    def eval(self, x: jax.Array, y: jax.Array) -> jax.Array:
        weights = jax.nn.softplus(self.weights)
        freqs = jax.nn.softplus(self.freqs)
        scales = jax.nn.softplus(self.scales)
        tau = x - y
        cos_term = jnp.cos(freqs @ tau)
        gauss_term = eqx.filter_vmap(lambda scale: jnp.exp(-0.5 * tau**2 / jnp.exp(2 * scale)).sum())(scales)
        return jnp.sum(weights * cos_term * gauss_term)


class NonstationaryGaussianSpectralMixtureKernel(eqx.Module, KernelBaseClass):
    weights: eqx.Module
    q: int

    def __init__(self, *, ndims: int, q: int, latent_dim: int, key: jax.Array, **_kwargs):
        self.q = int(q)
        self.weights = eqx.nn.MLP(
            key=key,
            in_size=ndims,
            out_size=self.q + self.q + (self.q * ndims),
            width_size=latent_dim,
            depth=1,
            activation=jax.nn.selu,
            final_activation=jax.nn.softplus,
        )

    def eval(self, x: jax.Array, y: jax.Array) -> jax.Array:
        q = self.q
        all_x, all_y = self.weights(x), self.weights(y)
        wx, wy = all_x[:q], all_y[:q]
        sx, sy = all_x[q : 2 * q], all_y[q : 2 * q]
        fx, fy = all_x[2 * q :].reshape(q, -1), all_y[2 * q :].reshape(q, -1)
        gibbs = (jnp.sqrt(2.0 * sx * sy) / (sx**2 + sy**2)) * jnp.exp(-(jnp.sum((x - y) ** 2)) / (sx**2 + sy**2))
        cosine = jnp.cos(2.0 * jnp.pi * (fx @ x - fy @ y))
        return (wx * wy * gibbs * cosine).sum()


kernels = {
    "g": GaussianKernel,
    "a_g": AnisotropicGaussianKernel,
    "ns_g": partial(NonstationaryGaussianKernel, latent_dim=8),
    "gsm": partial(GaussianSpectralMixtureKernel, q=2),
    "ns_gsm": partial(NonstationaryGaussianSpectralMixtureKernel, latent_dim=8, q=2),
}
