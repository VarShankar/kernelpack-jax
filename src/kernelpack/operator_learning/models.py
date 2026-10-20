from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
from jax import numpy as jnp, random as jr

from .utils import create_lifted_module


class KNO_REG_GRID_1D(eqx.Module):
    integration_kernels: list[eqx.Module]
    proj_layers: list[eqx.Module]
    pointwise_layers: list[eqx.Module]
    lift_kernel: eqx.Module
    lift_dim: int
    depth: int
    activation: Callable

    def __init__(self, integration_kernel, lift_dim: int, depth: int, in_feats: int, *, key: jax.Array):
        keys = jr.split(key, 2)
        self.integration_kernels = [create_lifted_module(integration_kernel, lift_dim=lift_dim, key=k) for k in jr.split(keys[0], depth)]
        self.pointwise_layers = [eqx.nn.Conv(1, lift_dim, lift_dim, 1, key=subkey) for subkey in jr.split(keys[1], depth)]
        keys = jr.split(keys[0], 4)
        self.proj_layers = [
            eqx.nn.Linear(lift_dim, lift_dim, key=keys[0]),
            eqx.nn.Linear(lift_dim, lift_dim, key=keys[1]),
            eqx.nn.Linear(lift_dim, 1, key=keys[2]),
        ]
        self.lift_kernel = eqx.nn.Linear(in_feats, lift_dim, key=keys[3])
        self.activation = jax.nn.gelu
        self.lift_dim = int(lift_dim)
        self.depth = int(depth)

    def __call__(self, f_x: jax.Array, x_grid: jax.Array, q_weights: jax.Array) -> jax.Array:
        def integration_transform(int_kernel, q_nodes, weights, f_q):
            gram = int_kernel(q_nodes, q_nodes) * weights.T
            return jnp.einsum("q,kq->k", f_q, gram)

        q_nodes = x_grid
        f_q = jnp.concatenate((f_x, q_nodes), axis=-1)
        f_q = eqx.filter_vmap(self.lift_kernel)(f_q)
        f_q = self.activation(f_q)

        for i in range(self.depth - 1):
            f_q_skip = self.pointwise_layers[i](f_q.T).T
            f_q = eqx.filter_vmap(lambda int_kernel, f: integration_transform(int_kernel, q_nodes, q_weights, f), in_axes=(eqx.if_array(0), 1), out_axes=1)(self.integration_kernels[i], f_q)
            f_q = self.activation(f_q + f_q_skip)

        f_q_skip = self.pointwise_layers[-1](f_q.T).T
        f_q = eqx.filter_vmap(lambda int_kernel, f: integration_transform(int_kernel, q_nodes, q_weights, f), in_axes=(eqx.if_array(0), 1), out_axes=1)(self.integration_kernels[-1], f_q)
        f_q = f_q + f_q_skip

        f_q = self.activation(eqx.filter_vmap(self.proj_layers[0])(f_q))
        f_q = self.activation(eqx.filter_vmap(self.proj_layers[1])(f_q))
        f_q = eqx.filter_vmap(self.proj_layers[2])(f_q)
        return f_q.squeeze()


class KNO_DIFFUSION_REACTION(eqx.Module):
    output_kernel: eqx.Module
    integration_kernels: list[eqx.Module]
    proj_layers: list[eqx.Module]
    pointwise_layers: list[eqx.Module]
    lift_kernel: eqx.Module
    lift_dim: int
    depth: int
    activation: Callable

    def __init__(self, output_kernel, integration_kernel, lift_dim: int, depth: int, in_feats: int, *, key: jax.Array):
        keys = jr.split(key, 2)
        self.integration_kernels = [create_lifted_module(integration_kernel, lift_dim=lift_dim, key=k) for k in jr.split(keys[0], depth)]
        self.pointwise_layers = [eqx.nn.Conv(1, lift_dim, lift_dim, 1, key=subkey) for subkey in jr.split(keys[1], depth)]
        keys = jr.split(keys[0], 4)
        self.proj_layers = [
            eqx.nn.Linear(lift_dim, lift_dim, key=keys[0]),
            eqx.nn.Linear(lift_dim, lift_dim, key=keys[1]),
            eqx.nn.Linear(lift_dim, 1, key=keys[2]),
        ]
        self.lift_kernel = eqx.nn.Linear(in_feats, lift_dim, key=keys[3])
        output_key = jr.split(keys[0], 1)[0]
        self.output_kernel = output_kernel(key=output_key)
        self.activation = jax.nn.gelu
        self.lift_dim = int(lift_dim)
        self.depth = int(depth)

    def __call__(self, f_x: jax.Array, y_grid: jax.Array, q_nodes: jax.Array, q_weights: jax.Array) -> jax.Array:
        def integration_transform(int_kernel, nodes, weights, f_q):
            gram = int_kernel(nodes, nodes) * weights.T
            return jnp.einsum("q,kq->k", f_q, gram)

        f_q = jnp.concatenate((f_x, q_nodes), axis=-1)
        f_q = eqx.filter_vmap(self.lift_kernel)(f_q)
        f_q = self.activation(f_q)

        for i in range(self.depth - 1):
            f_q_skip = self.pointwise_layers[i](f_q.T).T
            f_q = eqx.filter_vmap(lambda int_kernel, f: integration_transform(int_kernel, q_nodes, q_weights, f), in_axes=(eqx.if_array(0), 1), out_axes=1)(self.integration_kernels[i], f_q)
            f_q = self.activation(f_q + f_q_skip)

        f_q_skip = self.pointwise_layers[-1](f_q.T).T
        f_q = eqx.filter_vmap(lambda int_kernel, f: integration_transform(int_kernel, q_nodes, q_weights, f), in_axes=(eqx.if_array(0), 1), out_axes=1)(self.integration_kernels[-1], f_q)
        f_q = f_q + f_q_skip

        f_q = self.activation(eqx.filter_vmap(self.proj_layers[0])(f_q))
        f_q = self.activation(eqx.filter_vmap(self.proj_layers[1])(f_q))
        f_q = eqx.filter_vmap(self.proj_layers[2])(f_q).reshape(len(q_nodes), 1)

        kqq = self.output_kernel(q_nodes, q_nodes) + (jnp.eye(len(q_nodes)) * 1e-5)
        kqy = self.output_kernel(q_nodes, y_grid)
        kyq_kqq_inv = jnp.linalg.solve(kqq, kqy).T
        return jnp.einsum("mc,qm->qc", f_q, kyq_kqq_inv)


class KNO_DARCY_PWC(eqx.Module):
    integration_kernels: list[tuple[eqx.Module, ...]]
    lift_kernel: eqx.Module
    depth: int
    proj_layers: list[eqx.Module]
    pointwise_layers: list[eqx.Module]
    d: int
    lift_dim: int
    in_feats: int

    def __init__(self, integration_kernel, depth: int, lift_dim: int, ndims: int, in_feats: int, key: jax.Array):
        keys = jr.split(key, 7)
        self.lift_dim = int(lift_dim)
        self.d = int(ndims)
        self.proj_layers = [
            eqx.nn.Linear(lift_dim, lift_dim, key=keys[0]),
            eqx.nn.Linear(lift_dim, lift_dim, key=keys[1]),
            eqx.nn.Linear(lift_dim, 1, key=keys[2]),
        ]
        self.pointwise_layers = [eqx.nn.Conv(1, lift_dim, lift_dim, 1, key=subkey) for subkey in jr.split(keys[3], depth)]
        self.lift_kernel = eqx.nn.Linear(in_feats, lift_dim, key=keys[4])
        self.integration_kernels = [
            tuple(create_lifted_module(integration_kernel, lift_dim, dim_key) for dim_key in jr.split(subkey, ndims))
            for subkey in jr.split(keys[5], depth)
        ]
        self.in_feats = int(in_feats)
        self.depth = int(depth)

    def __call__(self, f_x: jax.Array, x_grid: jax.Array, q_weights: jax.Array) -> jax.Array:
        def integration_transform(int_kernel, q_nodes, weights, f_q):
            g1 = int_kernel[0](q_nodes, q_nodes) * weights.T
            g2 = int_kernel[1](q_nodes, q_nodes) * weights.T
            return jnp.einsum("ij,ki->kj", f_q, g1) + jnp.einsum("ij,kj->ik", f_q, g2)

        q_nodes = x_grid[:, 0, 0]
        f_q = jnp.concatenate((f_x, x_grid), axis=-1).reshape(-1, self.in_feats)
        f_q = eqx.filter_vmap(self.lift_kernel)(f_q).reshape(len(q_nodes), len(q_nodes), self.lift_dim)

        for i in range(self.depth - 1):
            f_q_skip = self.pointwise_layers[i](f_q.reshape(-1, self.lift_dim).T).T.reshape(f_q.shape)
            f_q = eqx.filter_vmap(lambda int_kernel, f: integration_transform(int_kernel, q_nodes, q_weights, f), in_axes=(eqx.if_array(0), self.d), out_axes=self.d)(self.integration_kernels[i], f_q)
            f_q = jax.nn.gelu(f_q + f_q_skip)

        f_q_skip = self.pointwise_layers[-1](f_q.reshape(-1, self.lift_dim).T).T.reshape(f_q.shape)
        f_q = eqx.filter_vmap(lambda int_kernel, f: integration_transform(int_kernel, q_nodes, q_weights, f), in_axes=(eqx.if_array(0), self.d), out_axes=self.d)(self.integration_kernels[-1], f_q)
        f_q = f_q + f_q_skip

        f_q = f_q.reshape(-1, self.lift_dim)
        f_q = jax.nn.gelu(eqx.filter_vmap(self.proj_layers[0])(f_q))
        f_q = jax.nn.gelu(eqx.filter_vmap(self.proj_layers[1])(f_q))
        return eqx.filter_vmap(self.proj_layers[2])(f_q)


class KNO_DARCY_TRIANGLE(eqx.Module):
    input_kernel: eqx.Module
    output_kernel: eqx.Module
    integration_kernels: list[eqx.Module]
    proj_layers: list[eqx.Module]
    pointwise_layers: list[eqx.Module]
    lift_kernel: eqx.Module
    lift_dim: int
    depth: int
    activation: Callable

    def __init__(self, input_kernel, output_kernel, integration_kernel, lift_dim: int, depth: int, in_feats: int, *, key: jax.Array):
        keys = jr.split(key, 2)
        self.integration_kernels = [create_lifted_module(integration_kernel, lift_dim=lift_dim, key=k) for k in jr.split(keys[0], depth)]
        self.pointwise_layers = [eqx.nn.Conv(1, lift_dim, lift_dim, 1, key=subkey) for subkey in jr.split(keys[1], depth)]
        keys = jr.split(keys[0], 4)
        self.proj_layers = [
            eqx.nn.Linear(lift_dim, lift_dim, key=keys[0]),
            eqx.nn.Linear(lift_dim, lift_dim, key=keys[1]),
            eqx.nn.Linear(lift_dim, 1, key=keys[2]),
        ]
        self.lift_kernel = eqx.nn.Linear(in_feats, lift_dim, key=keys[3])
        io_keys = jr.split(keys[0], 2)
        self.input_kernel = input_kernel(key=io_keys[0])
        self.output_kernel = output_kernel(key=io_keys[1])
        self.activation = jax.nn.gelu
        self.lift_dim = int(lift_dim)
        self.depth = int(depth)

    def __call__(self, f_x: jax.Array, x_grid: jax.Array, y_grid: jax.Array, q_nodes: jax.Array, q_weights: jax.Array) -> jax.Array:
        def integration_transform(int_kernel, nodes, weights, f_q):
            gram = int_kernel(nodes, nodes) * weights.T
            return jnp.einsum("q,kq->k", f_q, gram)

        f_x = eqx.filter_vmap(self.lift_kernel)(jnp.concatenate((f_x, x_grid), axis=-1)).reshape(len(x_grid), self.lift_dim)
        kxx = self.input_kernel(x_grid, x_grid) + (jnp.eye(len(x_grid)) * 1e-5)
        kxq = self.input_kernel(x_grid, q_nodes)
        kqx_kinv = jnp.linalg.solve(kxx, kxq).T
        f_q = jnp.einsum("mc,qm->qc", f_x, kqx_kinv)
        f_q = self.activation(f_q)

        for i in range(self.depth - 1):
            f_q_skip = self.pointwise_layers[i](f_q.T).T
            f_q = eqx.filter_vmap(lambda int_kernel, f: integration_transform(int_kernel, q_nodes, q_weights, f), in_axes=(eqx.if_array(0), 1), out_axes=1)(self.integration_kernels[i], f_q)
            f_q = self.activation(f_q + f_q_skip)

        f_q_skip = self.pointwise_layers[-1](f_q.T).T
        f_q = eqx.filter_vmap(lambda int_kernel, f: integration_transform(int_kernel, q_nodes, q_weights, f), in_axes=(eqx.if_array(0), 1), out_axes=1)(self.integration_kernels[-1], f_q)
        f_q = f_q_skip + f_q
        f_q = self.activation(eqx.filter_vmap(self.proj_layers[0])(f_q))
        f_q = self.activation(eqx.filter_vmap(self.proj_layers[1])(f_q))
        f_q = eqx.filter_vmap(self.proj_layers[2])(f_q)

        kqq = self.output_kernel(q_nodes, q_nodes) + (jnp.eye(len(q_nodes)) * 1e-5)
        kqy = self.output_kernel(q_nodes, y_grid)
        kyq_kqq_inv = jnp.linalg.solve(kqq, kqy).T
        return jnp.einsum("mc,qm->qc", f_q, kyq_kqq_inv)


class KNO_NS_PIPE(eqx.Module):
    integration_kernels: list[tuple[eqx.Module, ...]]
    lift_kernel: eqx.Module
    depth: int
    proj_layers: list[eqx.Module]
    pointwise_layers: list[eqx.Module]
    d: int
    lift_dim: int
    res_1d: int
    activation: Callable

    def __init__(self, integration_kernel, depth: int, lift_dim: int, ndims: int, in_feats: int, res_1d: int, key: jax.Array):
        keys = jr.split(key, 7)
        self.lift_dim = int(lift_dim)
        self.d = int(ndims)
        self.proj_layers = [
            eqx.nn.Linear(lift_dim, lift_dim, key=keys[0]),
            eqx.nn.Linear(lift_dim, lift_dim, key=keys[1]),
            eqx.nn.Linear(lift_dim, 1, key=keys[2]),
        ]
        self.pointwise_layers = [eqx.nn.Conv(1, lift_dim, lift_dim, 1, key=subkey) for subkey in jr.split(keys[3], depth)]
        self.lift_kernel = eqx.nn.Linear(in_feats, lift_dim, key=keys[4])
        self.integration_kernels = [
            tuple(create_lifted_module(integration_kernel, lift_dim, dim_key) for dim_key in jr.split(subkey, ndims))
            for subkey in jr.split(keys[5], depth)
        ]
        self.depth = int(depth)
        self.res_1d = int(res_1d)
        self.activation = jax.nn.gelu

    def __call__(self, q: jax.Array, wx: jax.Array, wy: jax.Array) -> jax.Array:
        grid_1d_y = q[0, :, 1]
        grid_1d_x = q[:, 0, 0]

        def integration_transform(int_kernel, f_q):
            g1 = int_kernel[0](grid_1d_x, grid_1d_x) * wx.T
            g2 = int_kernel[1](grid_1d_y, grid_1d_y) * wy.T
            return jnp.einsum("ij,ki->kj", f_q, g1) + jnp.einsum("ij,kj->ik", f_q, g2)

        f_q = eqx.filter_vmap(self.lift_kernel)(q.reshape(-1, 2)).reshape(self.res_1d, self.res_1d, self.lift_dim)
        for i in range(self.depth - 1):
            f_q_skip = self.pointwise_layers[i](f_q.reshape(-1, self.lift_dim).T).T.reshape(f_q.shape)
            f_q = eqx.filter_vmap(lambda int_kernel, f: integration_transform(int_kernel, f), in_axes=(eqx.if_array(0), self.d), out_axes=self.d)(self.integration_kernels[i], f_q)
            f_q = self.activation(f_q + f_q_skip)

        f_q_skip = self.pointwise_layers[-1](f_q.reshape(-1, self.lift_dim).T).T.reshape(f_q.shape)
        f_q = eqx.filter_vmap(lambda int_kernel, f: integration_transform(int_kernel, f), in_axes=(eqx.if_array(0), self.d), out_axes=self.d)(self.integration_kernels[-1], f_q)
        f_q = f_q + f_q_skip
        f_q = f_q.reshape(-1, self.lift_dim)
        f_q = self.activation(eqx.filter_vmap(self.proj_layers[0])(f_q))
        f_q = self.activation(eqx.filter_vmap(self.proj_layers[1])(f_q))
        return eqx.filter_vmap(self.proj_layers[2])(f_q)


class KNO_NS_3D(eqx.Module):
    integration_kernels: list[tuple[eqx.Module, ...]]
    lift_kernel: eqx.Module
    depth: int
    proj_layers: list[eqx.Module]
    pointwise_layers: list[eqx.Module]
    d: int
    lift_dim: int
    in_feats: int
    res_1d: int
    activation: Callable

    def __init__(self, integration_kernel, depth: int, lift_dim: int, ndims: int, in_feats: int, res_1d: int, key: jax.Array):
        keys = jr.split(key, 7)
        self.lift_dim = int(lift_dim)
        self.d = int(ndims)
        self.proj_layers = [
            eqx.nn.Linear(lift_dim, lift_dim, key=keys[0]),
            eqx.nn.Linear(lift_dim, lift_dim, key=keys[1]),
            eqx.nn.Linear(lift_dim, 1, key=keys[2]),
        ]
        self.pointwise_layers = [eqx.nn.Conv(1, lift_dim, lift_dim, 1, key=subkey) for subkey in jr.split(keys[3], depth)]
        self.lift_kernel = eqx.nn.Linear(in_feats, lift_dim, key=keys[4])
        self.integration_kernels = [
            tuple(create_lifted_module(integration_kernel, lift_dim, dim_key) for dim_key in jr.split(subkey, ndims))
            for subkey in jr.split(keys[5], depth)
        ]
        self.in_feats = int(in_feats)
        self.depth = int(depth)
        self.res_1d = int(res_1d)
        self.activation = jax.nn.gelu

    def __call__(self, f_x: jax.Array, x_grid: jax.Array, q_weights: jax.Array) -> jax.Array:
        def integration_transform(int_kernel, q_nodes, weights, f_q):
            g1 = int_kernel[0](q_nodes, q_nodes) * weights.T
            g2 = int_kernel[1](q_nodes, q_nodes) * weights.T
            g3 = int_kernel[2](q_nodes, q_nodes) * weights.T
            return (
                jnp.einsum("ijk,li->ljk", f_q, g1)
                + jnp.einsum("ijk,lj->ilk", f_q, g2)
                + jnp.einsum("ijk,lk->ijl", f_q, g3)
            )

        q_nodes = x_grid[:, 0, 0, 1]
        f_q = eqx.filter_vmap(self.lift_kernel)(jnp.concatenate((f_x, x_grid), axis=-1).reshape(-1, self.in_feats))
        f_q = f_q.reshape(len(q_nodes), len(q_nodes), len(q_nodes), self.lift_dim)

        for i in range(self.depth - 1):
            f_q_skip = self.pointwise_layers[i](f_q.reshape(-1, self.lift_dim).T).T.reshape(f_q.shape)
            f_q = eqx.filter_vmap(lambda int_kernel, f: integration_transform(int_kernel, q_nodes, q_weights, f), in_axes=(eqx.if_array(0), self.d), out_axes=self.d)(self.integration_kernels[i], f_q)
            f_q = self.activation(f_q + f_q_skip)

        f_q_skip = self.pointwise_layers[-1](f_q.reshape(-1, self.lift_dim).T).T.reshape(f_q.shape)
        f_q = eqx.filter_vmap(lambda int_kernel, f: integration_transform(int_kernel, q_nodes, q_weights, f), in_axes=(eqx.if_array(0), self.d), out_axes=self.d)(self.integration_kernels[-1], f_q)
        f_q = f_q + f_q_skip
        f_q = f_q.reshape(-1, self.lift_dim)
        f_q = self.activation(eqx.filter_vmap(self.proj_layers[0])(f_q))
        f_q = self.activation(eqx.filter_vmap(self.proj_layers[1])(f_q))
        return eqx.filter_vmap(self.proj_layers[2])(f_q)
