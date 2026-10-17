"""Divergence-free Matérn mixtures and local polynomial augmentation.

Each scalar mode is exp(-epsilon*r)*(3+3*epsilon*r+(epsilon*r)**2).
The Cartesian divergence-free differential operator is applied before modes
are added. This is the physical-coordinate convention used by KernelPack MATLAB.
"""
from functools import partial

import jax
import jax.numpy as jnp

from kernelpack.kernels import safe_sqrt, squared_distances
from .core import divfree_polynomial_stack, _rank_reduce_polynomial_stack


@partial(jax.jit, static_argnames=("num_modes",))
def multiscale_matern_parameters(base_length, *, num_modes=5, gamma=8 / 3):
    """Dyadic inverse lengths and relative scalar mixture weights.

    Mode indices start at one, as in MATLAB: lengths = base_length/2**j.
    The common base_length**gamma multiplier cancels on center normalization.
    """
    base_length = jnp.asarray(base_length)
    mode = jnp.arange(1, num_modes + 1, dtype=base_length.dtype)
    inverse_lengths = jnp.exp2(mode) / base_length
    weights = jnp.exp2(-gamma * mode)
    return inverse_lengths, weights


@partial(jax.jit, static_argnames=("normalize_center",))
def divfree_matern_gram_matrix(x, y, inverse_lengths, weights, *, normalize_center=True):
    """Component-major blocks of (-Delta I + grad grad.T) times a mixture.

    All arrays stay on their input device. The closed Cartesian expressions
    avoid differentiating a radial norm at coincident points. Center
    normalization makes each diagonal component at x=y exactly one.
    """
    x, y = jnp.asarray(x), jnp.asarray(y)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] not in (2, 3) or y.shape[1] != x.shape[1]:
        raise ValueError("x and y must be matching 2D or 3D point matrices")
    epsilons = jnp.asarray(inverse_lengths, dtype=x.dtype)
    weights = jnp.asarray(weights, dtype=x.dtype)
    if epsilons.ndim != 1 or not len(epsilons) or weights.shape != epsilons.shape:
        raise ValueError("Each Matérn inverse length requires one mixture weight")
    dim = x.shape[1]
    squared = squared_distances(x, y)
    radius = safe_sqrt(squared)
    if normalize_center:
        weights = weights / ((dim - 1) * jnp.sum(weights * epsilons**2))

    def mode(accumulated, parameters):
        diagonal, cross = accumulated
        epsilon, weight = parameters
        decay = weight * jnp.exp(-epsilon * radius)
        diagonal = diagonal + decay * ((dim - 1) * epsilon**2 * (1 + epsilon * radius)
                                        - epsilon**4 * squared)
        return (diagonal, cross + decay * epsilon**4), None

    initial = (jnp.zeros_like(radius), jnp.zeros_like(radius))
    (diagonal, cross), _ = jax.lax.scan(mode, initial, (epsilons, weights))
    blocks = []
    for a in range(dim):
        da = x[:, a, None] - y[None, :, a]
        blocks.append([cross * da * (x[:, b, None] - y[None, :, b])
                       + (diagonal if a == b else 0.) for b in range(dim)])
    return jnp.block(blocks)


@partial(jax.jit, static_argnames=("poly_degree", "num_modes"))
def prepare_divfree_matern_local_system(nodes, queries, *, poly_degree,
                                        base_length_factor=16., num_modes=5, gamma=8 / 3,
                                        coefficient_basis=None):
    """Local center-normalized multiscale Matérn plus divergence-free polynomials.

    Length scales derive from the actual selected image stencil radius. The
    sampled polynomial reduction, 1e-12 threshold, and inactive-column padding
    match the shared PHS local-system builder exactly.
    """
    center = jnp.mean(nodes, axis=0)
    radius = jnp.max(jnp.linalg.norm(nodes - center, axis=1))
    width = jnp.where(radius > 0, radius, 1.)
    raw, _ = divfree_polynomial_stack((nodes - center) / width, poly_degree=poly_degree)
    p, transform, active, rank_ratio = _rank_reduce_polynomial_stack(
        raw, nodes.shape[1], poly_degree, 1e-12,coefficient_basis=coefficient_basis)
    epsilons, weights = multiscale_matern_parameters(
        base_length_factor * width, num_modes=num_modes, gamma=gamma)
    gram = divfree_matern_gram_matrix(nodes, nodes, epsilons, weights)
    system = jnp.block([[gram, p], [p.T, jnp.diag((~active).astype(nodes.dtype))]])
    qraw, _ = divfree_polynomial_stack((queries - center) / width, poly_degree=poly_degree)
    evaluation = jnp.concatenate([
        divfree_matern_gram_matrix(queries, nodes, epsilons, weights), qraw @ transform], axis=1)
    return system, evaluation, jnp.where(radius > 0, rank_ratio, 0.)
