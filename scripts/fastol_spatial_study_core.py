"""Shared CUDA float64 maps and diagnostics for the FastOL spatial study.

This is a study adapter, not a second RBF implementation. Kernel matrices,
polynomial generation/rank reduction and Ram's periodic stencil rules come
from KernelPack. Nearest-node assignment is fixed before differentiation.
The only host work is orchestration, sizes, timing and explicit failure checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
import math
import time
from typing import Callable

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl

from kernelpack.divfree.core import (
    divfree_gram_matrix, divfree_polynomial_stack,
    divfree_polynomial_row_basis, _rank_reduce_polynomial_stack,
)
from kernelpack.divfree.matern import (
    divfree_matern_gram_matrix, multiscale_matern_parameters,
)
from kernelpack.domain.stencils import select_stencils, nearest_periodic_images


class SpatialStudyFailure(RuntimeError):
    """A concrete rejected geometry/solve, retaining its on-device evidence."""

    def __init__(self, message, *, diagnostics=None, center_indices=None):
        super().__init__(message)
        self.diagnostics = diagnostics
        self.center_indices = center_indices


@dataclass
class SpatialSystemCache:
    """Bounded GPU cache of canonical node batches; no numerical host copies.

    Entries that fit are retained. Once full, further entries are not cached,
    so a second scan still reuses the retained first batches rather than
    thrashing an LRU cache. The caller may clear it between scientific cells.
    """
    nodes: jax.Array
    degree: int
    kernel: str
    phs_degree: int
    batch_size: int
    periodic: jax.Array
    lengths: jax.Array
    use_periodic: bool
    base_length: jax.Array
    base_length_factor: float
    num_modes: int
    gamma: float
    allow_pseudoinverse: bool
    max_bytes: int
    stencil_domain: object | None = None
    stencil_geometry_token: tuple | None = None
    entries: dict = field(default_factory=dict)
    used_bytes: int = 0

    def clear(self):
        self.entries.clear()
        self.used_bytes = 0


@dataclass(frozen=True)
class SpatialMap:
    nodes: jax.Array
    original_query_coords: jax.Array
    query_coords: jax.Array
    center_indices: jax.Array
    stencil_indices: jax.Array
    image_coords: jax.Array
    weights: jax.Array                    # (query, output component, node, input component)
    weight_gradients: jax.Array | None     # same axes, then derivative coordinate
    diagnostics: dict[str, jax.Array]
    metadata: dict
    system_cache: SpatialSystemCache

    @property
    def stencil_radii(self):
        return self.diagnostics["stencil_radius"]


def _gpu64(value, device=None, *, check_finite=True):
    if not jax.config.x64_enabled:
        raise ValueError("Spatial study requires JAX_ENABLE_X64=1")
    result = jnp.asarray(value, dtype=jnp.float64, device=device)
    if result.device.platform != "gpu":
        raise ValueError("Spatial study requires CUDA arrays; no CPU numerical fallback")
    if check_finite and not bool(jnp.all(jnp.isfinite(result))):
        raise ValueError("Spatial coordinates/values must be finite")
    return result


def polynomial_dimension(dim, degree):
    return dim * math.comb(degree + dim, dim) - (
        math.comb(degree + dim - 1, dim) if degree else 0)


def _bounds(bounds, dim, device):
    if bounds is None:
        return (jnp.zeros(dim, dtype=bool, device=device),
                jnp.ones(dim, dtype=jnp.float64, device=device))
    if isinstance(bounds, dict):
        low, high = bounds["lower"], bounds["upper"]
        mask = bounds.get("periodic", [True] * dim)
    else:
        low, high = bounds
        mask = [True] * dim
    low, high = _gpu64(low, device), _gpu64(high, device)
    periodic = jnp.asarray(mask, dtype=bool, device=device)
    if low.shape != (dim,) or high.shape != (dim,) or periodic.shape != (dim,):
        raise ValueError("Periodic lower/upper/mask must match spatial dimension")
    if not bool(jnp.all(high > low)):
        raise ValueError("Periodic box lengths must be positive")
    return periodic, high - low


def _relative(error, target):
    numerator = jnp.linalg.norm(error)
    denominator = jnp.linalg.norm(target)
    return numerator / jnp.where(denominator > 0., denominator, 1.)


def _infinity_norm(matrix):
    return jnp.max(jnp.sum(jnp.abs(matrix), axis=-1))


def _stencil_geometry_token(domain):
    """Reuse the shared selector's complete immutable-buffer/policy signature."""
    return None if domain is None else domain._stencil_cache_signature()


@partial(jax.jit, static_argnames=("degree", "kernel", "phs_degree", "num_modes", "allow_pseudoinverse"))
def _prepare_one(images, row_basis, base_length, *, degree, kernel, phs_degree,
                 base_length_factor, num_modes, gamma, allow_pseudoinverse):
    """Use the shared sampled rank policy, and check the original padded system."""
    dim = images.shape[1]
    data_width = dim * len(images)
    center = jnp.mean(images, axis=0)
    radius = jnp.max(jnp.linalg.norm(images - center, axis=1))
    width = jnp.where(radius > 0., radius, 1.)
    raw, _ = divfree_polynomial_stack((images - center) / width, poly_degree=degree)
    p, transform, active, ratio = _rank_reduce_polynomial_stack(
        raw, dim, degree, 1e-12, coefficient_basis=row_basis)
    length = jnp.where(base_length > 0., base_length, base_length_factor * width)
    if kernel == "phs":
        gram = divfree_gram_matrix(images, images, degree=phs_degree)
    else:
        epsilons, mixture = multiscale_matern_parameters(length, num_modes=num_modes, gamma=gamma)
        gram = divfree_matern_gram_matrix(images, images, epsilons, mixture)
    matrix = jnp.block([[gram, p], [p.T, jnp.diag((~active).astype(images.dtype))]])
    identity = jnp.eye(len(matrix), dtype=matrix.dtype)
    lu, pivots = jl.lu_factor(matrix)
    inverse_lu = jl.lu_solve((lu, pivots), identity)
    lu_bad = ~jnp.all(jnp.isfinite(inverse_lu))
    if allow_pseudoinverse:
        inverse = jax.lax.cond(lu_bad, lambda _: jnp.linalg.pinv(matrix),
                               lambda _: inverse_lu, operand=None)
    else:
        inverse = inverse_lu
    coefficients = inverse[:, :data_width]
    residual = matrix @ inverse - identity
    columns = jnp.linalg.norm(residual, axis=0)
    inverse_norm = jnp.linalg.norm(inverse, axis=0)
    backward = jnp.max(columns / (jnp.linalg.norm(matrix) * inverse_norm + 1.))
    raw_condition = jnp.where(lu_bad, jnp.inf,
        _infinity_norm(matrix) * _infinity_norm(inverse))
    # Symmetric diagonal equilibration is used only for a reported condition
    # diagnostic; the actual solver above always uses the original matrix.
    scales = 1. / jnp.sqrt(jnp.maximum(jnp.max(jnp.abs(matrix), axis=1), jnp.finfo(matrix.dtype).tiny))
    equilibrated = scales[:, None] * matrix * scales[None, :]
    equilibrated_inverse = inverse / scales[:, None] / scales[None, :]
    scaled_condition = jnp.where(lu_bad, jnp.inf,
        _infinity_norm(equilibrated) * _infinity_norm(equilibrated_inverse))
    nodal_map = matrix[:data_width] @ coefficients
    interpolation = _relative(nodal_map - identity[:data_width, :data_width],
                              identity[:data_width, :data_width])
    singular = jnp.linalg.svd(raw @ row_basis, full_matrices=False, compute_uv=False)
    singular_ratio = singular / jnp.where(singular[0] > 0., singular[0], 1.)
    near = jnp.any((singular_ratio >= .5e-12) & (singular_ratio <= 2e-12))
    eps = jnp.finfo(matrix.dtype).eps
    diagnostics = {
        "retained_rank": jnp.sum(active),
        "requested_rank": jnp.asarray(row_basis.shape[1], dtype=jnp.int32),
        "inactive_padding_count": jnp.sum(~active),
        "rank_ratio": ratio,
        "rank_threshold_near": near,
        "stencil_radius": radius,
        "condition_number": raw_condition,
        "condition_times_epsilon": raw_condition * eps,
        "equilibrated_condition_number": scaled_condition,
        "equilibrated_condition_times_epsilon": scaled_condition * eps,
        "solver_relative_residual": jnp.max(columns),  # identity columns have norm one
        "solver_backward_error": backward,
        "condition_scaled_solver_error": raw_condition * backward,
        "interpolation_relative_residual": interpolation,
        "lu_nonfinite": lu_bad,
        "fallback_activated": lu_bad & allow_pseudoinverse,
        "system_finite": jnp.all(jnp.isfinite(matrix)) & jnp.all(jnp.isfinite(coefficients)),
    }
    return dict(images=images, center=center, width=width, transform=transform,
                coefficients=coefficients, raw=raw, p=p, active=active,
                base_length=length, diagnostics=diagnostics)


@partial(jax.jit, static_argnames=("degree", "kernel", "phs_degree", "num_modes", "allow_pseudoinverse"))
def _prepare_batch(images, row_basis, base_length, *, degree, kernel, phs_degree,
                   base_length_factor, num_modes, gamma, allow_pseudoinverse):
    # Healthy stencils factor together. Gate the rare fallback outside vmap:
    # a vmapped conditional would compute every pseudoinverse unconditionally.
    options = dict(degree=degree, kernel=kernel, phs_degree=phs_degree,
        base_length_factor=base_length_factor, num_modes=num_modes, gamma=gamma)
    healthy = jax.vmap(lambda cloud: _prepare_one(cloud, row_basis, base_length,
        allow_pseudoinverse=False, **options))(images)
    if not allow_pseudoinverse:
        return healthy
    def repair(_):
        return jax.lax.map(lambda pair: jax.lax.cond(
            pair[1]["diagnostics"]["lu_nonfinite"],
            lambda _: _prepare_one(pair[0], row_basis, base_length,
                allow_pseudoinverse=True, **options),
            lambda _: pair[1], operand=None), (images, healthy))
    return jax.lax.cond(jnp.any(healthy["diagnostics"]["lu_nonfinite"]),
                        repair, lambda _: healthy, operand=None)


def _query_one(query, images, center, width, transform, coefficients, base_length,
               *, degree, kernel, phs_degree, num_modes, gamma):
    if kernel == "phs":
        kernel_values = divfree_gram_matrix(query[None], images, degree=phs_degree)
    else:
        epsilons, mixture = multiscale_matern_parameters(base_length, num_modes=num_modes, gamma=gamma)
        kernel_values = divfree_matern_gram_matrix(query[None], images, epsilons, mixture)
    raw, _ = divfree_polynomial_stack(((query - center) / width)[None], poly_degree=degree)
    polynomial = raw @ transform
    evaluation = jnp.concatenate((kernel_values, polynomial), axis=1)
    dim = images.shape[1]
    weights = (evaluation @ coefficients).reshape(dim, dim, len(images)).transpose(0, 2, 1)
    return weights


@partial(jax.jit, static_argnames=("degree", "kernel", "phs_degree", "num_modes", "compute_gradients"))
def _query_batch(queries, entries, *, degree, kernel, phs_degree, num_modes, gamma, compute_gradients):
    def one(query, entry):
        evaluate = lambda q: _query_one(q, entry["images"], entry["center"], entry["width"],
            entry["transform"], entry["coefficients"], entry["base_length"], degree=degree,
            kernel=kernel, phs_degree=phs_degree, num_modes=num_modes, gamma=gamma)
        weights = evaluate(query)
        dim = query.shape[0]
        # Convert the public point-major map to the shared component-major data stack.
        flat = weights.transpose(0, 2, 1).reshape(dim, -1)
        raw_query, _ = divfree_polynomial_stack(
            ((query - entry["center"]) / entry["width"])[None], poly_degree=degree)
        p_query = raw_query @ entry["transform"]
        retained_error = _relative(flat @ entry["p"] - p_query, p_query)
        requested_error = _relative(flat @ entry["raw"] - raw_query, raw_query)
        distance = jnp.min(jnp.linalg.norm(entry["images"] - query, axis=1))
        node_match = jnp.all(entry["images"] == query, axis=1)
        is_node = jnp.any(node_match)
        nodal_target = jnp.eye(dim, dtype=query.dtype)[:, None, :] * node_match[None, :, None]
        nodal_error = jnp.where(is_node, _relative(weights - nodal_target, nodal_target), 0.)
        defined = jnp.asarray(True) if kernel != "phs" or phs_degree >= 5 else distance > 0.
        diagnostics = dict(entry["diagnostics"])
        diagnostics.update(retained_polynomial_relative_error=retained_error,
            requested_polynomial_relative_error=requested_error,
            polynomial_error_over_condition_roundoff=retained_error / jnp.maximum(
                diagnostics["condition_times_epsilon"], jnp.finfo(query.dtype).eps),
            query_is_stencil_node=is_node, query_nodal_interpolation_relative_error=nodal_error,
            weight_norm=jnp.linalg.norm(weights),
            query_node_min_distance=distance, derivative_defined=defined,
            finite=diagnostics["system_finite"] & jnp.all(jnp.isfinite(weights))
                & jnp.isfinite(retained_error) & jnp.isfinite(requested_error))
        if compute_gradients:
            gradient = jax.jacfwd(evaluate)(query)
            # Cubic center derivatives do not exist classically. Return an
            # explicit mask and NaNs rather than certifying the AD convention.
            gradient = jnp.where(defined, gradient, jnp.nan)
            gradient_finite = jnp.all(jnp.isfinite(gradient))
            diagnostics["gradient_finite"] = gradient_finite
            diagnostics["finite"] &= (~defined) | gradient_finite
            divergence = jnp.einsum("akca->kc", gradient)
            diagnostics["divergence_weight_residual"] = jnp.linalg.norm(divergence)
            diagnostics["divergence_weight_normalized"] = jnp.linalg.norm(divergence) / jnp.maximum(
                jnp.linalg.norm(gradient), jnp.finfo(query.dtype).tiny)
        else:
            gradient = jnp.empty(weights.shape + (0,), dtype=weights.dtype)
        return weights, gradient, diagnostics
    return jax.vmap(one)(queries, entries)


@partial(jax.jit, static_argnames=("degree", "kernel", "phs_degree", "num_modes", "compute_gradients"),
         donate_argnums=(6, 7, 8, 9))
def _query_scatter_batch(entry, node_ids, framed_queries, padded_ids, padded_positions,
                         valid_count, output_weights, output_gradient, output_ids,
                         output_images, output_diagnostics, *, degree, kernel,
                         phs_degree, num_modes, gamma, compute_gradients):
    """Gather, evaluate and update one fixed native query batch in one dispatch.

    Only the four fresh output buffers are donated. Canonical cached entries,
    geometry, query coordinates, and diagnostic buffers remain read-only inputs;
    diagnostic leaves can contain equal/aliased values and are not donated.
    Padded rows evaluate the same valid repeated query as before, but their
    out-of-bounds write IDs are discarded rather than overwriting valid rows.
    """
    selected = jax.tree.map(lambda value: value[padded_positions], entry)
    weights, gradients, diagnostics = _query_batch(framed_queries[padded_ids], selected,
        degree=degree, kernel=kernel, phs_degree=phs_degree, num_modes=num_modes,
        gamma=gamma, compute_gradients=compute_gradients)
    write_ids = jnp.where(jnp.arange(padded_ids.shape[0], dtype=padded_ids.dtype) < valid_count,
                          padded_ids, output_weights.shape[0])
    output_weights = output_weights.at[write_ids].set(weights, mode="drop")
    if compute_gradients:
        output_gradient = output_gradient.at[write_ids].set(gradients, mode="drop")
    output_ids = output_ids.at[write_ids].set(node_ids[padded_positions], mode="drop")
    output_images = output_images.at[write_ids].set(selected["images"], mode="drop")
    if output_diagnostics is None:
        output_diagnostics = {key: jnp.zeros((output_weights.shape[0],) + value.shape[1:],
            dtype=value.dtype) for key, value in diagnostics.items()}
    output_diagnostics = {key: output_diagnostics[key].at[write_ids].set(value, mode="drop")
                          for key, value in diagnostics.items()}
    return output_weights, output_gradient, output_ids, output_images, output_diagnostics


@jax.jit
def apply_map_arrays(weights, indices, values):
    """Jitted public array API; supports .lower(...).compile() without captured arrays."""
    return jnp.einsum("qakc,qkc...->qa...", weights, values[indices],
                      precision=jax.lax.Precision.HIGHEST)


@jax.jit
def apply_map_jacobian_arrays(weight_gradients, indices, values):
    """Result axes are (query, output component, derivative coordinate, ...)."""
    return jnp.einsum("qakcl,qkc...->qal...", weight_gradients, values[indices],
                      precision=jax.lax.Precision.HIGHEST)


def apply_map(spatial_map, values):
    # Metadata-only checks: this wrapper is safe inside a transfer guard.
    # Values/outputs receive their recorded finite checks in the study driver.
    values = _gpu64(values, spatial_map.nodes.device, check_finite=False)
    if values.shape[:2] != spatial_map.nodes.shape:
        raise ValueError("values must have shape (node, component, ...)")
    return apply_map_arrays(spatial_map.weights, spatial_map.stencil_indices, values)


def apply_map_jacobian(spatial_map, values):
    if spatial_map.weight_gradients is None:
        raise ValueError("Map was built with compute_gradients=False")
    values = _gpu64(values, spatial_map.nodes.device, check_finite=False)
    if values.shape[:2] != spatial_map.nodes.shape:
        raise ValueError("values must have shape (node, component, ...)")
    return apply_map_jacobian_arrays(spatial_map.weight_gradients, spatial_map.stencil_indices, values)


def build_map(nodes, queries, degree, kernel, periodic_bounds=None, assignment="nearest-node",
              batch_size=32, *, allow_pseudoinverse=False, compute_gradients=True,
              phs_degree=None, matern_base_length_factor=16., matern_num_modes=5,
              matern_gamma=8/3, matern_base_length=None, system_cache=None,
              system_cache_bytes=512*1024**2, stencil_domain=None,
              progress: Callable | None = None):
    """Build fixed-node-stencil maps using shared KernelPack GPU routines.

    ``degree`` is spatial polynomial degree ell (requested orders 2/4/6 use
    ell=1/3/5), not the desired convergence order. PHS defaults are 3/3/5.
    ``matern_base_length`` optionally fixes one physical mixture length;
    otherwise base_length_factor times each assigned stencil radius is used.
    Diagnostics are on-device arrays, one entry per original query, before
    any user-defined acceptance thresholds. No numerical fallback is hidden.
    """
    started = time.perf_counter()
    nodes = _gpu64(nodes)
    queries = _gpu64(queries, nodes.device)
    if nodes.ndim != 2 or nodes.shape[1] not in (2, 3) or queries.ndim != 2 or queries.shape[1] != nodes.shape[1]:
        raise ValueError("nodes and queries must be spatial matrices in 2D or 3D")
    if not len(queries) or int(degree) != degree or degree < 0 or int(batch_size) != batch_size or batch_size < 1:
        raise ValueError("Nonempty queries, nonnegative integral degree and positive batch_size required")
    if assignment != "nearest-node":
        raise ValueError("This study uses fixed nearest-node stencil assignment only")
    aliases = {"phs": "phs", "matern": "matern", "matern_multiscale": "matern"}
    if kernel not in aliases:
        raise ValueError("kernel must be phs or matern")
    kernel = aliases[kernel]
    degree, batch_size = int(degree), int(batch_size)
    dim = nodes.shape[1]
    rank = polynomial_dimension(dim, degree)
    stencil_size = 2 * math.ceil(rank / dim) + 1
    if len(nodes) < stencil_size:
        raise ValueError(f"Need at least {stencil_size} physical nodes for degree {degree}")
    phs_degree = max(3, degree if degree % 2 else degree - 1) if phs_degree is None else int(phs_degree)
    if phs_degree < 3 or phs_degree % 2 != 1:
        raise ValueError("PHS radial degree must be odd and >=3")
    if not (matern_base_length_factor > 0. and math.isfinite(matern_base_length_factor)
            and int(matern_num_modes) == matern_num_modes and matern_num_modes > 0
            and math.isfinite(matern_gamma)):
        raise ValueError("Invalid Matérn mixture parameters")
    if int(system_cache_bytes) != system_cache_bytes or system_cache_bytes < 0:
        raise ValueError("system_cache_bytes must be a nonnegative integer")
    periodic, lengths = _bounds(periodic_bounds, dim, nodes.device)
    use_periodic = periodic_bounds is not None
    stencil_stats_before = dict(stencil_domain.stencil_stats) if stencil_domain is not None else {}
    if stencil_domain is not None:
        if use_periodic or stencil_domain.stencil_level_set is None:
            raise ValueError("An irregular stencil domain must have boundary-aware geometry and cannot be periodic")
        domain_nodes = stencil_domain.get_tree_points("all")
        if domain_nodes.shape != nodes.shape or not bool(jnp.array_equal(domain_nodes, nodes)):
            raise ValueError("Boundary-aware domain and reconstruction nodes must agree in order and coordinates")
    geometry_token = _stencil_geometry_token(stencil_domain)
    base_length = jnp.asarray(-1., dtype=nodes.dtype, device=nodes.device) if matern_base_length is None else _gpu64(matern_base_length, nodes.device)
    if base_length.shape != () or matern_base_length is not None and not bool(base_length > 0.):
        raise ValueError("Fixed Matérn base length must be a positive scalar")
    if system_cache is None:
        cache = SpatialSystemCache(nodes=nodes, degree=degree, kernel=kernel, phs_degree=phs_degree,
            batch_size=batch_size, periodic=periodic, lengths=lengths, use_periodic=use_periodic,
            base_length=base_length, base_length_factor=float(matern_base_length_factor),
            num_modes=int(matern_num_modes), gamma=float(matern_gamma),
            allow_pseudoinverse=bool(allow_pseudoinverse), max_bytes=int(system_cache_bytes),
            stencil_domain=stencil_domain, stencil_geometry_token=geometry_token)
    else:
        cache = system_cache
        static_match = (cache.degree == degree and cache.kernel == kernel and cache.phs_degree == phs_degree
            and cache.batch_size == batch_size and cache.base_length_factor == matern_base_length_factor
            and cache.num_modes == matern_num_modes and cache.gamma == matern_gamma
            and cache.use_periodic == use_periodic
            and cache.stencil_domain is stencil_domain
            and cache.stencil_geometry_token == geometry_token
            and cache.max_bytes == int(system_cache_bytes)
            and cache.allow_pseudoinverse == allow_pseudoinverse and cache.nodes.shape == nodes.shape
            and cache.nodes.dtype == nodes.dtype and cache.nodes.device == nodes.device)
        if not static_match or not bool(jnp.all(cache.nodes == nodes) & jnp.all(cache.periodic == periodic)
                & jnp.all(cache.lengths == lengths) & (cache.base_length == base_length)):
            raise ValueError("Spatial-system cache does not match nodes, geometry and numerical method")
    nearest, _, valid = select_stencils(nodes, queries, k=1, periodic=periodic,
        lengths=lengths, use_periodic=use_periodic)
    if not bool(jnp.all(valid)):
        raise SpatialStudyFailure("Nearest-node assignment is invalid or ambiguous")
    centers = nearest[:, 0]
    # One device-side stable grouping avoids variable-size nonzero/gather
    # compilations in each canonical node batch. Only integer bucket counts
    # cross to the host to orchestrate fixed-size GPU query batches.
    query_order = jnp.argsort(centers, stable=True)
    bucket_counts = jnp.bincount(centers // batch_size,
        length=math.ceil(len(nodes) / batch_size))
    framed_queries = nearest_periodic_images(queries[:, None, :], nodes[centers], lengths, periodic)[:, 0]
    row_basis = divfree_polynomial_row_basis(dim, degree, dtype=nodes.dtype, device=nodes.device)
    options = dict(degree=degree, kernel=kernel, phs_degree=phs_degree,
                   num_modes=int(matern_num_modes), gamma=matern_gamma)
    output_weights = jnp.zeros((len(queries), dim, stencil_size, dim), dtype=nodes.dtype, device=nodes.device)
    output_gradient = (jnp.zeros(output_weights.shape + (dim,), dtype=nodes.dtype, device=nodes.device)
                       if compute_gradients else None)
    output_ids = jnp.zeros((len(queries), stencil_size), dtype=jnp.int32, device=nodes.device)
    output_images = jnp.zeros((len(queries), stencil_size, dim), dtype=nodes.dtype, device=nodes.device)
    output_diagnostics = None
    built_batches = reused_batches = 0
    query_cursor = 0
    for start in range(0, len(nodes), batch_size):
        stop = min(start + batch_size, len(nodes))
        count = int(bucket_counts[start // batch_size])
        if not count:
            continue
        if start in cache.entries:
            entry, node_ids = cache.entries[start]
            reused_batches += 1
        else:
            node_centers = jnp.minimum(jnp.arange(start, start + batch_size,
                dtype=jnp.int32, device=nodes.device), stop - 1)
            if stencil_domain is None:
                node_ids, images, stencil_valid = select_stencils(nodes, nodes[node_centers], k=stencil_size,
                    periodic=periodic, lengths=lengths, center_indices=node_centers, use_periodic=use_periodic)
            else:
                node_ids, distance = stencil_domain.query_knn(
                    "all", nodes[node_centers], stencil_size, backend="jax")
                images = nodes[node_ids]
                self_count = jnp.sum(node_ids == node_centers[:, None], axis=1)
                ordered = jnp.sort(node_ids, axis=1)
                stencil_valid = (jnp.all(jnp.isfinite(distance), axis=1)
                    & jnp.all((node_ids >= 0) & (node_ids < len(nodes)), axis=1)
                    & jnp.all(ordered[:, 1:] != ordered[:, :-1], axis=1)
                    & (self_count == 1))
            if not bool(jnp.all(stencil_valid)):
                raise SpatialStudyFailure(f"Invalid fixed stencil in node batch [{start},{stop})",
                                          center_indices=node_centers)
            entry = _prepare_batch(images, row_basis, base_length,
                base_length_factor=matern_base_length_factor, allow_pseudoinverse=allow_pseudoinverse,
                **options)
            if not bool(jnp.all(entry["diagnostics"]["system_finite"])) or (
                    not allow_pseudoinverse and bool(jnp.any(entry["diagnostics"]["lu_nonfinite"]))):
                raise SpatialStudyFailure(f"Nonfinite local solve in node batch [{start},{stop})",
                    diagnostics=entry["diagnostics"], center_indices=node_centers)
            built_batches += 1
            size = sum(x.nbytes for x in jax.tree.leaves(entry)) + node_ids.nbytes
            if cache.used_bytes + size <= cache.max_bytes:
                cache.entries[start] = (entry, node_ids)
                cache.used_bytes += size
        for offset in range(0, count, batch_size):
            valid_count = min(batch_size, count - offset)
            chunk = jnp.minimum(jnp.arange(batch_size, dtype=jnp.int32,
                device=nodes.device) + offset, count - 1)
            padded_ids = query_order[query_cursor + chunk]
            padded_positions = centers[padded_ids] - start
            output_weights, output_gradient, output_ids, output_images, output_diagnostics = _query_scatter_batch(
                entry, node_ids, framed_queries, padded_ids, padded_positions, valid_count,
                output_weights, output_gradient, output_ids, output_images, output_diagnostics,
                compute_gradients=compute_gradients, **options)
        query_cursor += count
        if progress is not None:
            progress(dict(stage="spatial_map_node_batch", first_node=start, stop_node=stop,
                          total_nodes=len(nodes), assigned_queries=count, cache_bytes=cache.used_bytes))
    jax.block_until_ready((output_weights, output_gradient, output_diagnostics))
    metadata = dict(degree=degree, kernel=kernel, phs_degree=phs_degree, dimension=dim,
        stencil_size=stencil_size, requested_rank=rank, padded_system_width=dim*stencil_size+rank,
        assignment=assignment, precision="float64", backend="cuda", batch_size=batch_size,
        stencil_selection=("shared boundary-aware CFPU visibility and one-bend paths"
            if stencil_domain is not None else "shared Ram periodic/Euclidean nearest-node rule"),
        boundary_stencil_statistics=({key:value-stencil_stats_before.get(key,0)
            for key,value in stencil_domain.stencil_stats.items()} if stencil_domain is not None else None),
        compute_gradients=compute_gradients, pseudoinverse_allowed=allow_pseudoinverse,
        condition_norm="infinity; symmetric diagonal row-max equilibration reported separately",
        setup_wall_including_compilation_seconds=time.perf_counter()-started,
        built_node_batches=built_batches, reused_node_batches=reused_batches,
        system_cache_bytes=cache.used_bytes, system_cache_budget_bytes=cache.max_bytes,
        matern_length_policy="stencil-relative" if matern_base_length is None else "fixed-physical",
        matern_base_length_factor=float(matern_base_length_factor), matern_num_modes=int(matern_num_modes),
        matern_gamma=float(matern_gamma), cubic_center_derivative="undefined; NaN with explicit mask")
    return SpatialMap(nodes, queries, framed_queries, centers, output_ids, output_images,
                      output_weights, output_gradient, output_diagnostics, metadata, cache)
