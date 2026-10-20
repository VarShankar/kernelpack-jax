"""Divergence-free Matérn mixtures and local polynomial augmentation.

Each scalar mode is exp(-epsilon*r)*(3+3*epsilon*r+(epsilon*r)**2).
The Cartesian divergence-free differential operator is applied before modes
are added. This is the physical-coordinate convention used by KernelPack MATLAB.
"""
from functools import partial
import math

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


@partial(jax.jit, static_argnames=("polynomial_degree",))
def divfree_matern_quotient_gram_matrix(
        x, y, inverse_lengths, weights, *, reference_radius, polynomial_degree=5):
    """Stable physical Matérn matrix modulo reproduced DF polynomials.

    This is the SAME constrained interpolation kernel as the unnormalized
    ``divfree_matern_gram_matrix`` when the complete divergence-free polynomial
    space of degree ``polynomial_degree`` is retained.  Supported degrees are
    1, 3, 5.  For a coarse mode (epsilon*reference_radius <= 1/2), subtract its
    divergence-free matrix Taylor terms of degrees 0, ..., degree-1 (even).
    Their rows and columns are divergence-free polynomials of those degrees.
    Consequently B(X,X)=P G P.T and B(x,X)c=0 when P.T c=0: subtraction leaves
    the augmented interpolant and its Lagrange weights unchanged.

    The quotient matrix itself need not be positive definite.  Factor its
    projection onto ker(P.T), not its full matrix.  All requested polynomial
    columns must be retained; rank truncation invalidates the equivalence.

    A single positive finite ``reference_radius`` selects coarse modes for
    the WHOLE local Gram/evaluation system.  Pass the same radius to every
    block.  It should bound node/query radii about one common center.  When
    it does, coarse-mode pair arguments are <=1; a 36-term Taylor evaluation
    avoids cancellation of their small remainders.  Outside that bound an
    algebraically identical direct expression is used, so the kernel is not
    selected separately for individual pairs.  Fine modes are never altered.

    No per-mode or center normalization is performed.  A common positive
    scalar may divide all Gram/evaluation blocks externally; for a broad
    ell**(8/3) mixture, reference_radius**(2/3) is a useful quotient scale.
    Coordinates are physical: under x=c+R*xi use inverse lengths epsilon*R
    and multiply the resulting matrix by R**(-2).
    """
    if polynomial_degree not in (1, 3, 5):
        raise ValueError("The stable Matérn quotient supports polynomial degrees 1, 3, 5")
    x, y = jnp.asarray(x), jnp.asarray(y)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] not in (2, 3) or y.shape[1] != x.shape[1]:
        raise ValueError("x and y must be matching 2D or 3D point matrices")
    dim = x.shape[1]
    epsilons = jnp.asarray(inverse_lengths, dtype=x.dtype)
    weights = jnp.asarray(weights, dtype=x.dtype)
    ref = jnp.asarray(reference_radius, dtype=x.dtype)
    if epsilons.ndim != 1 or not len(epsilons) or weights.shape != epsilons.shape or ref.shape != ():
        raise ValueError("Each inverse length needs one weight and the reference radius must be scalar")
    # Invalid dynamic options propagate NaN rather than silently changing the
    # coarse-mode selection.  Callers validate finite positive physical scales.
    valid = jnp.isfinite(ref) & (ref > 0) & jnp.all(jnp.isfinite(epsilons) & (epsilons > 0))
    valid = valid & jnp.all(jnp.isfinite(weights) & (weights >= 0))
    # Physical clusters can be tiny relative to their coordinate offsets.
    # Subtract coordinates directly; a norm/Gram identity would lose digits.
    delta = x[:, None, :] - y[None, :, :]
    squared = jnp.sum(delta * delta, axis=-1)
    radius = safe_sqrt(squared)
    removed_diagonal = tuple(range(0, polynomial_degree, 2))
    removed_cross = tuple(range(0, max(polynomial_degree - 2, 0), 2))
    # Coefficients are static factorial/index bookkeeping, never host array math.
    coefficients = []
    for n in range(36, -1, -1):
        base = (-1.)**n / math.factorial(n)
        diag = base * ((dim - 1) * (1 - n) - n * (n - 1))
        coefficients.append((0. if n in removed_diagonal else diag,
                             0. if n in removed_cross else base))
    coefficients = jnp.asarray(coefficients, dtype=x.dtype)

    def mode(accumulated, parameters):
        diagonal, cross = accumulated
        epsilon, weight = parameters
        coarse = epsilon * ref <= .5
        t = epsilon * radius
        decay = jnp.exp(-t)
        full_diag = decay * ((dim - 1) * (1 + t) - t*t)
        full_cross = decay
        # Clamp only the UNUSED Taylor branch outside t<=1.  This avoids
        # high powers of large fine-mode arguments in staged JAX branches.
        series_t = jnp.minimum(t, 1.)
        def horner(state, coefficient):
            a, b = state
            return (a * series_t + coefficient[0], b * series_t + coefficient[1]), None
        (series_diag, series_cross), _ = jax.lax.scan(
            horner, (jnp.zeros_like(t), jnp.zeros_like(t)), coefficients)
        polynomial_t = jnp.where(coarse, t, 0.)
        direct_diag, direct_cross = full_diag, full_cross
        for n in removed_diagonal:
            coefficient = ((-1.)**n / math.factorial(n)
                           * ((dim - 1) * (1 - n) - n * (n - 1)))
            direct_diag = direct_diag - coefficient * polynomial_t**n
        for n in removed_cross:
            direct_cross = direct_cross - ((-1.)**n / math.factorial(n)) * polynomial_t**n
        quotient_diag = jnp.where(t <= 1., series_diag, direct_diag)
        quotient_cross = jnp.where(t <= 1., series_cross, direct_cross)
        diagonal = diagonal + weight * epsilon**2 * jnp.where(coarse, quotient_diag, full_diag)
        cross = cross + weight * epsilon**4 * jnp.where(coarse, quotient_cross, full_cross)
        return (diagonal, cross), None

    (diagonal, cross), _ = jax.lax.scan(mode,
        (jnp.zeros_like(radius), jnp.zeros_like(radius)), (epsilons, weights))
    blocks = [[cross * delta[..., a] * delta[..., b]
               + (diagonal if a == b else 0.) for b in range(dim)] for a in range(dim)]
    return jnp.where(valid, jnp.block(blocks), jnp.nan)


@jax.jit
def divfree_matern_anchored_gram_matrix(
        x, y, inverse_lengths, weights, *, anchor, reference_radius):
    """Stable covariance of u(x)-u(anchor) for the original Matérn mixture.

    The four blocks use the SAME constant-only quotient and coarse-mode flags:
    Ka(x,y)=K(x,y)-K(x,a)-K(a,y)+K(a,a).  Removed constants cancel exactly in
    this identity, so this is the original anchored covariance, not a changed
    mode mixture.  It is positive definite at distinct nodes excluding the
    anchor whenever the original matrix kernel is strictly positive definite.

    If the augmented polynomial space contains all constant vector fields,
    using Ka instead of K leaves the exact interpolant and Lagrange weights
    unchanged.  Indeed sum(c_i)=0 makes their kernel expansions differ only
    by a constant vector, which is absorbed by the polynomial augmentation.
    Unlike a degree-five quotient, Ka itself can use ordinary SPD Cholesky.

    ``reference_radius`` must be identical for all four blocks and subsequent
    evaluation blocks.  No individual mode normalization is introduced.
    """
    x, y = jnp.asarray(x), jnp.asarray(y)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] not in (2, 3) or y.shape[1] != x.shape[1]:
        raise ValueError("x and y must be matching 2D or 3D point matrices")
    a = jnp.asarray(anchor, dtype=x.dtype)
    if a.shape == (x.shape[1],):
        a = a[None]
    if a.shape != (1, x.shape[1]):
        raise ValueError("The anchor must be one point in the physical dimension")
    def quotient(left, right):
        return divfree_matern_quotient_gram_matrix(left, right, inverse_lengths, weights,
            reference_radius=reference_radius, polynomial_degree=1)
    xy, xa, ay, aa = quotient(x, y), quotient(x, a), quotient(a, y), quotient(a, a)
    return (xy - jnp.repeat(xa, len(y), axis=1) - jnp.repeat(ay, len(x), axis=0)
            + jnp.repeat(jnp.repeat(aa, len(x), axis=0), len(y), axis=1))


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
