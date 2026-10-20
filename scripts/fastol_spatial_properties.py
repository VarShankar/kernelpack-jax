"""Full CUDA-float64 spatial convergence, divergence, and periodicity study.

This is a spatial reconstruction study, not an operator-learning fit.  Analytic
potential fields and independently generated fixed queries are the truth oracle.
Importing this module performs no numerical work; only main() acquires the GPU.
NumPy is used exclusively by the artifact serializer, after device computation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
import gc
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import traceback
import uuid

os.environ["JAX_PLATFORMS"] = "cuda"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp


DOMAIN_FORMULAS = {
    "2": "F=1-x^2-1.4*y^2-.15*x^4-.10*x^2*y^2-.20*y^4+.10*x^3",
    "3": "F=1-x^2-1.3*y^2-1.7*z^2-.10*(x^4+y^4+z^4)-.08*x^2*y^2-.06*y^2*z^2+.10*x*y*z",
}
FIELD_FORMULAS = {
    "2": "psi=F^2*(sin(pi*x)*sin(2*pi*y)+.2*cos(3*pi*x)*sin(pi*y)); u=(psi_y,-psi_x)",
    "3": "A=F^2*(sin(pi*y)*sin(2*pi*z),sin(pi*z)*sin(2*pi*x),sin(pi*x)*sin(2*pi*y)); u=curl(A)",
    "periodic2": "psi=sin(x)*cos(2*y)+.2*cos(3*x)*sin(y); u=(psi_y,-psi_x)",
    "periodic3": "A=(sin(y)*sin(2*z),sin(z)*sin(2*x),sin(x)*sin(2*y)); u=curl(A)",
}


@dataclass
class Cloud:
    points: jax.Array
    metadata: dict
    boundary: jax.Array | None = None
    stencil_domain: object | None = None
    geometry_evidence: dict | None = None


def configure_gpu():
    """Require NVIDIA CUDA; no CPU numerical fallback is available."""
    device = jax.devices("gpu")[0]
    if device.platform != "gpu" or "nvidia" not in device.device_kind.lower():
        raise RuntimeError(f"NVIDIA CUDA is required, got {device}")
    if not jax.config.x64_enabled:
        raise RuntimeError("This study requires jax_enable_x64=True")
    jax.config.update("jax_default_device", device)
    return device


@jax.jit
def level_set(points):
    """The analytic irregular-domain oracle, evaluated entirely on the GPU."""
    x, y = points[..., 0], points[..., 1]
    if points.shape[-1] == 2:
        return 1-x*x-1.4*y*y-.15*x**4-.10*x*x*y*y-.20*y**4+.10*x**3
    z = points[..., 2]
    return (1-x*x-1.3*y*y-1.7*z*z-.10*(x**4+y**4+z**4)
            -.08*x*x*y*y-.06*y*y*z*z+.10*x*y*z)


@partial(jax.jit, static_argnames=("dimension", "count"))
def domain_boundary(dimension, count, phase=0.):
    """Ray-bisection on the smooth star-shaped analytic boundary.

    Monotonic radial growth of 1-F is certified by the quadratics documented in
    the protocol.  The fixed 64 bisections are not chosen from interpolation error.
    """
    i = jnp.arange(count, dtype=jnp.float64)
    if dimension == 2:
        angle = (i+.5)*(2*jnp.pi/count)+phase
        directions = jnp.stack((jnp.cos(angle), jnp.sin(angle)), axis=1)
    else:
        z = 1-2*(i+.5)/count
        angle = i*(jnp.pi*(3-jnp.sqrt(jnp.asarray(5., dtype=jnp.float64))))+phase
        radius = jnp.sqrt(jnp.maximum(1-z*z, 0.))
        directions = jnp.stack((radius*jnp.cos(angle), radius*jnp.sin(angle), z), axis=1)
    lo = jnp.zeros(count, dtype=jnp.float64)
    hi = jnp.full(count, 1.5, dtype=jnp.float64)
    def step(_, state):
        left, right = state
        mid = (left+right)/2
        inside = level_set(mid[:, None]*directions) > 0
        return jnp.where(inside, mid, left), jnp.where(inside, right, mid)
    lo, hi = jax.lax.fori_loop(0, 64, step, (lo, hi))
    return ((lo+hi)/2)[:, None]*directions


def make_irregular_cloud(dimension, spacing, *, seed=1729):
    """Canonical closed-piece PU geometry and shared GPU Poisson MIS nodes."""
    from fastol_scattered_clouds import irregular_cloud
    points, boundary, descriptor, metadata, evidence = irregular_cloud(
        dimension, spacing, seed=seed, level_set=level_set, domain_boundary=domain_boundary)
    return Cloud(points, metadata, boundary, descriptor, evidence)


def fixed_irregular_queries(dimension, n_interior, n_boundary, *, seed=8675309):
    """One independent query set, reused verbatim at every h/order/kernel."""
    key = jax.random.fold_in(jax.random.PRNGKey(seed), dimension)
    candidates = jax.random.uniform(key, (12*n_interior, dimension), dtype=jnp.float64,
                                    minval=-1.2, maxval=1.2)
    mask = level_set(candidates) > 0
    if int(jax.device_get(jnp.count_nonzero(mask))) < n_interior:
        raise RuntimeError("Prescribed rejection pool is too small; no query set was altered")
    indices = jnp.nonzero(mask, size=n_interior)[0]
    return candidates[indices], domain_boundary(dimension, n_boundary, phase=.641267)


def make_periodic_cloud(dimension, n_per_axis, *, seed=1729):
    """Shared minimum-image Poisson MIS; one representative per torus point."""
    from fastol_scattered_clouds import periodic_cloud
    points, metadata = periodic_cloud(dimension, n_per_axis, seed=seed)
    return Cloud(points, metadata)


def _stream(point, periodic):
    x, y = point
    if periodic:
        return jnp.sin(x)*jnp.cos(2*y)+.2*jnp.cos(3*x)*jnp.sin(y)
    return level_set(point)**2*(jnp.sin(jnp.pi*x)*jnp.sin(2*jnp.pi*y)
                               +.2*jnp.cos(3*jnp.pi*x)*jnp.sin(jnp.pi*y))


def _potential(point, periodic):
    x, y, z = point
    factor = jnp.asarray(1., dtype=point.dtype) if periodic else level_set(point)**2
    frequency = jnp.asarray(1. if periodic else jnp.pi, dtype=point.dtype)
    return factor*jnp.stack((jnp.sin(frequency*y)*jnp.sin(2*frequency*z),
                             jnp.sin(frequency*z)*jnp.sin(2*frequency*x),
                             jnp.sin(frequency*x)*jnp.sin(2*frequency*y)))


def _one_field(point, periodic):
    if point.shape[0] == 2:
        derivative = jax.grad(lambda p: _stream(p, periodic))(point)
        return jnp.stack((derivative[1], -derivative[0]))
    derivative = jax.jacfwd(lambda p: _potential(p, periodic))(point)
    return jnp.stack((derivative[2, 1]-derivative[1, 2],
                      derivative[0, 2]-derivative[2, 0],
                      derivative[1, 0]-derivative[0, 1]))


@partial(jax.jit, static_argnames=("periodic",))
def manufactured_field(points, *, periodic=False):
    return jax.vmap(lambda p: _one_field(p, periodic))(points)


@partial(jax.jit, static_argnames=("periodic",))
def manufactured_jacobian(points, *, periodic=False):
    return jax.vmap(jax.jacfwd(lambda p: _one_field(p, periodic)))(points)


def periodic_probes(dimension, *, seed=32452843, count=64, wrapped_count=128):
    """All face/edge/corner identifications and joint integer-period shifts."""
    key = jax.random.fold_in(jax.random.PRNGKey(seed), dimension)
    anchors = jax.random.uniform(key, (count, dimension), dtype=jnp.float64,
                                 minval=.15, maxval=.85)*(2*jnp.pi)
    chunks, groups = [], []
    cursor = 0
    for codimension in range(1, dimension+1):
        for axes in itertools.combinations(range(dimension), codimension):
            anchor = anchors[:1] if codimension == dimension else anchors
            number = len(anchor)
            first = cursor
            for mask in itertools.product((0, 1), repeat=codimension):
                query = anchor
                for axis, side in zip(axes, mask):
                    query = query.at[:, axis].set(side*2*jnp.pi)
                chunks.append(query)
                groups.append(dict(kind=("corner" if codimension == dimension else "face" if codimension == 1 else "edge"),
                    axes=list(axes), identification=list(mask), start=cursor, stop=cursor+number,
                    reference_start=first, reference_stop=first+number))
                cursor += number
    wrapped = jax.random.uniform(jax.random.fold_in(key, 19), (wrapped_count, dimension),
                                 dtype=jnp.float64, minval=.07, maxval=.93)*(2*jnp.pi)
    first = cursor
    shifts = [(0,)*dimension]+[s for s in itertools.product((-1, 0, 1), repeat=dimension) if any(s)]
    shifts += [(2, -1) if dimension == 2 else (2, -1, 1),
               (-2, 2) if dimension == 2 else (-2, 2, -1)]
    for shift in shifts:
        offset = jnp.asarray(shift, dtype=jnp.float64)*(2*jnp.pi)
        chunks.append(wrapped+offset)
        groups.append(dict(kind="wrapped", shift=list(shift), start=cursor,
            stop=cursor+wrapped_count, reference_start=first, reference_stop=first+wrapped_count))
        cursor += wrapped_count
    return jnp.concatenate(chunks, axis=0), groups


def _scalar(value):
    return jax.device_get(value).item()


def _json_safe(value):
    if isinstance(value, jax.Array):
        # Explicit diagnostics serialization; scientific reductions happened
        # on device before this boundary.
        return _json_safe(jax.device_get(value).tolist())
    if type(value).__module__.startswith("numpy") and hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    return value


def _canonical(value):
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest_array(value):
    """Device export and byte hashing are artifact I/O, never a numerical oracle."""
    value.block_until_ready()
    host = jax.device_get(value)
    return hashlib.sha256(host.tobytes(order="C")).hexdigest()


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to replace immutable artifact {path}")
    temporary = path.with_name(path.name+f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(json.dumps(_json_safe(value), indent=2, allow_nan=False).encode()+b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        raise FileExistsError(path)
    os.replace(temporary, path)


def _atomic_npz(path, arrays):
    import numpy as np  # Serialization only.  All array mathematics is JAX/CUDA.
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name+f".tmp.{os.getpid()}")
    exported = {name: jax.device_get(value) for name, value in arrays.items()}
    with temporary.open("xb") as handle:
        np.savez(handle, **exported)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _preserve_cloud_failure(folder, error, *, source_sha256, protocol_sha256, specification,
        additional_arrays=None, stage_metadata=None):
    """Preserve actual geometry/sampler state even before a scientific cell exists."""
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path=folder/("cloud_failure_"+stamp+".json")
    arrays=dict(getattr(error,"evidence",{}),**(additional_arrays or {}))
    artifact=path.with_suffix(".npz")
    _atomic_npz(artifact,{key:value for key,value in arrays.items() if isinstance(value,jax.Array)})
    _atomic_json(path,dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
        specification=specification,stage_metadata=dict(getattr(error,"metadata",{}),**(stage_metadata or {})),
        source_sha256=source_sha256,protocol_sha256=protocol_sha256,
        diagnostic_artifact=artifact.name,diagnostic_artifact_sha256=_digest_file(artifact),
        time_utc=datetime.now(timezone.utc).isoformat()))


def _sources():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), root/"scripts/fastol_spatial_study_core.py",
             root/"scripts/fastol_scattered_clouds.py",
             *(p for p in (root/"src/kernelpack").rglob("*")
               if p.is_file() and p.suffix.lower() in (".py", ".cu", ".cuh", ".cc", ".cpp", ".h", ".hpp", ".pyd", ".so"))]
    return {str(p.relative_to(root)).replace("\\", "/"): _digest_file(p) for p in sorted(set(paths))}


def _check_sources(expected):
    observed = _sources()
    if observed != expected:
        raise RuntimeError("A pinned study or production source changed during execution")


def _assert_device_tree(tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        if isinstance(leaf, jax.Array):
            if leaf.device.platform != "gpu":
                raise RuntimeError("A numerical array is not on CUDA")
            if jnp.issubdtype(leaf.dtype, jnp.floating) and leaf.dtype != jnp.float64:
                raise RuntimeError(f"Unexpected numerical precision {leaf.dtype}")


def _visible_array_bytes(*trees):
    """Shape accounting of distinct Python array objects, not peak GPU memory."""
    seen = set()
    total = 0
    for tree in trees:
        for value in jax.tree_util.tree_leaves(tree):
            if isinstance(value, jax.Array) and id(value) not in seen:
                total += value.nbytes
                seen.add(id(value))
    return total


def geometry_metrics(nodes, queries, *, periodic=False, batch_size=256):
    """Explicit sampled fill distance and full-node minimum separation."""
    def distances(a, b):
        delta = a[:, None, :]-b[None, :, :]
        if periodic:
            delta = delta-2*jnp.pi*jnp.round(delta/(2*jnp.pi))
        return jnp.sum(delta*delta, axis=-1)
    distance_block = jax.jit(distances)
    nearest = []
    for start in range(0, len(queries), batch_size):
        nearest.append(jnp.min(distance_block(queries[start:start+batch_size], nodes), axis=1))
    nearest_squared = jnp.concatenate(nearest)
    separation_squared = jnp.asarray(jnp.inf, dtype=jnp.float64)
    for start in range(0, len(nodes), batch_size):
        squared = distance_block(nodes[start:start+batch_size], nodes)
        own = jnp.arange(start, min(start+batch_size, len(nodes)))
        squared = squared.at[jnp.arange(len(own)), own].set(jnp.inf)
        separation_squared = jnp.minimum(separation_squared, jnp.min(squared))
    return dict(sampled_fill_distance=jnp.sqrt(jnp.max(nearest_squared)),
        rms_nearest_distance=jnp.sqrt(jnp.mean(nearest_squared)),
        minimum_node_separation=jnp.sqrt(separation_squared),
        half_minimum_node_separation=.5*jnp.sqrt(separation_squared),
        exact_query_node_coincidences=jnp.count_nonzero(nearest_squared == 0),
        minimum_query_node_distance=jnp.sqrt(jnp.min(nearest_squared))), nearest_squared


def _diagnostic_summary(diagnostics, expected_rank, tolerance, condition_epsilon_limit):
    required = ("retained_rank", "requested_rank", "condition_number", "condition_times_epsilon",
        "solver_relative_residual", "solver_backward_error", "interpolation_relative_residual",
        "retained_polynomial_relative_error", "requested_polynomial_relative_error",
        "fallback_activated", "lu_nonfinite", "finite", "equilibrated_condition_number",
        "equilibrated_condition_times_epsilon", "rank_threshold_near", "stencil_radius")
    missing = [key for key in required if key not in diagnostics]
    if missing:
        raise RuntimeError(f"Missing mandatory local-system diagnostics: {missing}")
    ranks = diagnostics["retained_rank"]
    summary = dict(stencils=int(ranks.size), expected_rank=expected_rank,
        retained_rank_min=_scalar(jnp.min(ranks)), retained_rank_max=_scalar(jnp.max(ranks)),
        requested_rank_defects=_scalar(jnp.count_nonzero(ranks != expected_rank)),
        retained_rank_histogram=jax.device_get(jnp.bincount(ranks.reshape(-1), length=expected_rank+1)).tolist(),
        fallback_count=_scalar(jnp.count_nonzero(diagnostics["fallback_activated"])),
        lu_nonfinite_count=_scalar(jnp.count_nonzero(diagnostics["lu_nonfinite"])))
    for key in ("condition_number", "condition_times_epsilon", "solver_relative_residual",
                "solver_backward_error", "interpolation_relative_residual",
                "retained_polynomial_relative_error", "requested_polynomial_relative_error",
                "equilibrated_condition_number", "equilibrated_condition_times_epsilon"):
        values = diagnostics[key]
        summary[key+"_max"] = _scalar(jnp.max(values))
        summary[key+"_quantiles_0_50_90_99_100"] = jax.device_get(jnp.quantile(values.reshape(-1),
            jnp.asarray([0., .5, .9, .99, 1.], dtype=jnp.float64))).tolist()
    if "rank_threshold_near" in diagnostics:
        summary["rank_threshold_near_count"] = _scalar(jnp.count_nonzero(diagnostics["rank_threshold_near"]))
    if "stencil_radius" in diagnostics:
        summary["stencil_radius_quantiles_0_50_90_99_100"] = jax.device_get(jnp.quantile(
            diagnostics["stencil_radius"].reshape(-1),jnp.asarray([0.,.5,.9,.99,1.]))).tolist()
    checks = dict(full_requested_rank=jnp.all(ranks == expected_rank),
        declared_requested_rank=jnp.all(diagnostics["requested_rank"] == expected_rank),
        finite_local_systems=jnp.all(diagnostics["finite"]),
        no_pseudoinverse_fallback=jnp.logical_not(jnp.any(diagnostics["fallback_activated"])),
        original_system_solve=jnp.all(diagnostics["solver_relative_residual"] <= tolerance),
        original_system_backward_error=jnp.all(diagnostics["solver_backward_error"] <= tolerance),
        nodal_interpolation=jnp.all(diagnostics["interpolation_relative_residual"] <= tolerance),
        retained_polynomial_reproduction=jnp.all(diagnostics["retained_polynomial_relative_error"] <= tolerance),
        requested_polynomial_reproduction=jnp.all(diagnostics["requested_polynomial_relative_error"] <= tolerance),
        equilibrated_condition_screen=jnp.all(diagnostics["equilibrated_condition_times_epsilon"] <= condition_epsilon_limit))
    return summary, checks


@partial(jax.jit, static_argnames=("degree",))
def polynomial_potential_fields(points, degree):
    """Independent analytic curl oracle spanning the full requested vector space.

    These fields use elementary derivatives of monomial potentials, rather than
    the production polynomial stack or its sampled change of basis. Redundant
    curl generators are retained; zero generators are omitted analytically.
    Columns are spatial fields, so their maximum relative error checks every
    generator rather than hiding a failed column in an aggregate norm.
    """
    dimension = points.shape[1]
    exponents = [e for e in itertools.product(range(degree+2), repeat=dimension)
                 if 1 <= sum(e) <= degree+1]
    powers = jnp.asarray(exponents, dtype=jnp.int32)
    derivatives = []
    for axis in range(dimension):
        decrement = jnp.zeros(dimension, dtype=jnp.int32).at[axis].set(1)
        reduced = jnp.maximum(powers-decrement, 0)
        monomials = jnp.prod(points[:, None, :]**reduced[None], axis=-1)
        derivatives.append(monomials*powers[None, :, axis])
    if dimension == 2:
        return jnp.stack((derivatives[1], -derivatives[0]), axis=1)
    zero = jnp.zeros_like(derivatives[0])
    # curl(phi*e_j), retaining only generators with a nonzero curl.
    curls = (jnp.stack((zero, derivatives[2], -derivatives[1]), axis=1),
             jnp.stack((-derivatives[2], zero, derivatives[0]), axis=1),
             jnp.stack((derivatives[1], -derivatives[0], zero), axis=1))
    fields = []
    for potential_axis, curl in enumerate(curls):
        indices = [i for i, exponent in enumerate(exponents)
                   if any(power for axis, power in enumerate(exponent) if axis != potential_axis)]
        fields.append(curl[:, :, jnp.asarray(indices, dtype=jnp.int32)])
    return jnp.concatenate(fields, axis=-1)


def _independent_polynomial_check(spatial_map, degree, batch_size=256):
    # Local unwrapped image coordinates are essential on a torus: global
    # monomials are not periodic fields and cannot be tested after wrapping.
    column_errors = column_reference = None
    pointwise_maximum = jnp.asarray(0., dtype=jnp.float64)
    for start in range(0, len(spatial_map.query_coords), batch_size):
        stop = min(start+batch_size, len(spatial_map.query_coords))
        images = spatial_map.image_coords[start:stop]
        center = jnp.mean(images, axis=1)
        radius = jnp.max(jnp.linalg.norm(images-center[:, None], axis=2), axis=1)
        radius = jnp.where(radius > 0., radius, 1.)
        local_nodes = (images-center[:, None])/radius[:, None, None]
        local_queries = (spatial_map.query_coords[start:stop]-center)/radius[:, None]
        exact = polynomial_potential_fields(local_queries, degree)
        field_values = polynomial_potential_fields(local_nodes.reshape(-1, images.shape[-1]), degree)
        field_values = field_values.reshape(*images.shape[:2], images.shape[-1], -1)
        predicted = jnp.einsum("qakc,qkcf->qaf", spatial_map.weights[start:stop], field_values)
        error = predicted-exact
        errors = jnp.sum(error*error, axis=(0, 1))
        # Normalize by the magnitude of both input samples and query truth.
        # A high-degree generator can vanish at an assigned node while its
        # stencil values are substantial; query-only normalization is invalid.
        references = (jnp.sum(exact*exact, axis=(0, 1))
            + jnp.sum(jnp.mean(field_values*field_values, axis=1), axis=(0, 1)))
        column_errors = errors if column_errors is None else column_errors+errors
        column_reference = references if column_reference is None else column_reference+references
        pointwise_maximum = jnp.maximum(pointwise_maximum, jnp.max(jnp.abs(error)))
    column_relative = jnp.sqrt(column_errors/jnp.maximum(column_reference, jnp.finfo(jnp.float64).tiny))
    return dict(generators=column_relative.shape[0], maximum_generator_relative_rms=jnp.max(column_relative),
        maximum_component_error=pointwise_maximum, generator_relative_rms=column_relative,
        coordinate_scope="independently differentiated monomial potentials in each local normalized image frame",
        normalization="sqrt(sum_q(||f(q)||^2 + mean_i ||f(x_qi)||^2)) per generator",
        finite=jnp.all(jnp.isfinite(column_relative)))


def _error_metrics(predicted, truth, jacobian, truth_jacobian, derivative_mask):
    difference = predicted-truth
    point_error = jnp.linalg.norm(difference, axis=1)
    truth_energy = jnp.sum(truth*truth)
    divergence = jnp.trace(jacobian, axis1=1, axis2=2)
    selected = jnp.where(derivative_mask, divergence, 0.)
    count = jnp.maximum(jnp.count_nonzero(derivative_mask), 1)
    grad_energy = jnp.sum(jnp.where(derivative_mask[:,None,None],jacobian*jacobian,0.))/count
    return dict(relative_rms=jnp.sqrt(jnp.sum(difference*difference)/jnp.maximum(truth_energy,jnp.finfo(jnp.float64).tiny)),
        absolute_rms=jnp.sqrt(jnp.mean(point_error**2)), sampled_maximum=jnp.max(point_error),
        reference_rms=jnp.sqrt(truth_energy/len(truth)),
        gradient_relative_rms=jnp.sqrt(jnp.sum(jnp.where(derivative_mask[:,None,None],(jacobian-truth_jacobian)**2,0.))/jnp.maximum(jnp.sum(jnp.where(derivative_mask[:,None,None],truth_jacobian**2,0.)),jnp.finfo(jnp.float64).tiny)),
        divergence_absolute_rms=jnp.sqrt(jnp.sum(selected**2)/count),
        divergence_sampled_maximum=jnp.max(jnp.abs(selected)),
        divergence_normalized_rms=jnp.sqrt(jnp.sum(selected**2)/count)/jnp.maximum(1.,jnp.sqrt(grad_energy)),
        divergence_query_count=jnp.count_nonzero(derivative_mask),
        derivative_excluded_count=len(truth)-jnp.count_nonzero(derivative_mask),
        finite=jnp.all(jnp.isfinite(predicted)) & jnp.all(jnp.where(derivative_mask[:,None,None],jnp.isfinite(jacobian),True)))


def _periodicity_metrics(prediction, jacobian, truth, truth_jacobian, groups):
    result = {}
    for kind in ("face", "edge", "corner", "wrapped"):
        value, derivative, exact_value, exact_derivative = [], [], [], []
        pairs = 0
        for group in groups:
            if group["kind"] != kind or group["start"] == group["reference_start"]:
                continue
            a = slice(group["reference_start"],group["reference_stop"])
            b = slice(group["start"],group["stop"])
            value.append(jnp.linalg.norm(prediction[b]-prediction[a],axis=1))
            derivative.append(jnp.linalg.norm(jacobian[b]-jacobian[a],axis=(1,2)))
            exact_value.append(jnp.linalg.norm(truth[b]-truth[a],axis=1))
            exact_derivative.append(jnp.linalg.norm(truth_jacobian[b]-truth_jacobian[a],axis=(1,2)))
            pairs += group["stop"]-group["start"]
        if not value:
            continue
        def reduce(values):
            values=jnp.concatenate(values)
            return dict(rms=jnp.sqrt(jnp.mean(values**2)),sampled_maximum=jnp.max(values))
        result[kind]=dict(pairs=pairs,value_jump=reduce(value),jacobian_jump=reduce(derivative),
                          exact_value_jump=reduce(exact_value),exact_jacobian_jump=reduce(exact_derivative))
    return result


def _source_and_artifact_protocol(args, device):
    from fastol_scattered_clouds import GEOMETRY_POLICY
    return dict(schema="fastol_spatial_properties_scattered_v2", source_sha256=_sources(),
        execution=dict(platform=device.platform,device_kind=device.device_kind,dtype="float64",
                       host_role="orchestration and artifact serialization only",cpu_numerical_fallback=False),
        packages={name:importlib.metadata.version(name) for name in ("jax","jaxlib","numpy")},
        requested_value_orders=args.orders, polynomial_degrees=[p-1 for p in args.orders],
        phs_scalar_exponents={"2":3,"4":3,"6":5},
        stencil_rule="kappa=2*ceil((d*comb(ell+d,d)-comb(ell+d-1,d))/d)+1",
        assignment="nearest-node fixed stencil; differentiation holds that assigned stencil fixed",
        kernels=args.kernels,matern=dict(modes=5,tau=16.,gamma=8/3),
        irregular_domains=DOMAIN_FORMULAS,manufactured_fields=FIELD_FORMULAS,
        cloud_and_stencil_policy=GEOMETRY_POLICY,
        analytic_boundary_regularity="For 1-F=Q+H-C the radial derivative is r*(2Q-3rC+4r^2H)>0; lower quadratics have negative discriminants in both dimensions. Thus every ray crosses F=0 once, with nonzero gradient.",
        irregular_spacing_2d=args.spacing_2d,irregular_spacing_3d=args.spacing_3d,
        periodic_grid_2d=args.periodic_grid_2d,periodic_grid_3d=args.periodic_grid_3d,
        interior_queries_2d=args.queries_2d,interior_queries_3d=args.queries_3d,
        boundary_queries_2d=args.boundary_queries_2d,boundary_queries_3d=args.boundary_queries_3d,
        query_seed=args.query_seed,node_seed=args.node_seed,
        batch_size=args.batch_size,warm_repetitions=args.repetitions,
        tolerance=args.tolerance,condition_epsilon_limit=args.condition_epsilon_limit,
        fixed_queries_independent_of_nodes=True,point_selection_uses_reconstruction_errors=False,
        classical_divergence="away from kernel centers only; no global stencil-interface divergence claim",
        boundary_trace="Manufactured F^2 potential has exact zero velocity; unconstrained reconstruction does not impose the entire boundary trace",
        norms="sampled vector RMS and maximum; not exact continuous L2 or Linfinity",
        h_definition="maximum nearest-node distance over the same independent interior queries; sampled fill distance, not exact continuous fill distance",
        configuration_count=(len(args.dimensions)*len(args.orders)*len(args.kernels)
                             *(4*int("irregular" in args.suites)+int("periodic" in args.suites))),
        periodic_protocol="One representative periodic cloud per dimension, not a second convergence study; all face/edge/corner and joint multi-shift identifications",
        independent_polynomial_oracle="analytic derivatives/curls of all monomial potentials through degree ell+1; no production polynomial evaluator; local image frames on the torus; normalization includes query and stencil input RMS",
        stability_scope="equilibrated infinity-condition times machine epsilon screen plus explicit interpolation-block norm quantiles; no assertion of a uniform norm bound from finite output",
        pass_scope="Per-cell pass means the declared algebraic/property screens passed; empirical refinement slopes are reported, not silently certified by this Boolean",
        memory_scope="visible-array shape accounting, not allocator usage or peak VRAM; distinct Python array objects can share device storage",
        rate_scope="successive sampled-h slopes, not a proven rate; compare with nominal p only alongside rank, residual and stability diagnostics")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--suites",nargs="+",choices=["irregular","periodic"],default=["irregular","periodic"])
    parser.add_argument("--dimensions",nargs="+",type=int,choices=[2,3],default=[2,3])
    parser.add_argument("--orders",nargs="+",type=int,choices=[2,4,6],default=[2,4,6])
    parser.add_argument("--kernels",nargs="+",choices=["phs","matern"],default=["phs","matern"])
    parser.add_argument("--spacing-2d",nargs="+",type=float,default=[.103,.0675,.0445,.0295])
    parser.add_argument("--spacing-3d",nargs="+",type=float,default=[.22,.16,.118,.088])
    parser.add_argument("--periodic-grid-2d",type=int,default=32)
    parser.add_argument("--periodic-grid-3d",type=int,default=12)
    parser.add_argument("--queries-2d",type=int,default=4096)
    parser.add_argument("--queries-3d",type=int,default=32768)
    parser.add_argument("--boundary-queries-2d",type=int,default=512)
    parser.add_argument("--boundary-queries-3d",type=int,default=2048)
    parser.add_argument("--query-seed",type=int,default=8675309)
    parser.add_argument("--node-seed",type=int,default=1729)
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--repetitions",type=int,default=11)
    parser.add_argument("--tolerance",type=float,default=1e-8)
    parser.add_argument("--condition-epsilon-limit",type=float,default=.01)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--gpu-lock-file",type=Path)
    args=parser.parse_args()
    if min(args.queries_2d,args.queries_3d,args.boundary_queries_2d,args.boundary_queries_3d,
           args.batch_size,args.repetitions) < 1 or args.tolerance <= 0 or args.condition_epsilon_limit <= 0:
        parser.error("Counts, batch sizes, repetitions and tolerances must be positive")
    for values in (args.spacing_2d,args.spacing_3d):
        if len(values)!=4 or any(x<=0 for x in values) or any(a<=b for a,b in zip(values,values[1:])):
            parser.error("Irregular studies require exactly four strictly decreasing positive spacings")
    if min(args.periodic_grid_2d,args.periodic_grid_3d)<2:
        parser.error("The single periodic grid in each dimension needs at least two cells per axis")
    for name in ("suites","dimensions","orders","kernels"):
        values=getattr(args,name)
        if len(values)!=len(set(values)):
            parser.error(f"Repeated entries in --{name} are not permitted")
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    lock=args.gpu_lock_file or args.output.parent/".fastol_spatial_gpu.lock"
    owner=dict(pid=os.getpid(),host=socket.gethostname(),token=uuid.uuid4().hex,
               output=str(args.output),started_utc=datetime.now(timezone.utc).isoformat())
    lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open("x",encoding="utf-8") as handle:
        json.dump(owner,handle)
    try:
        return _run(args)
    except BaseException as exc:
        failure=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),
                     time_utc=datetime.now(timezone.utc).isoformat())
        _atomic_json(args.output/("execution_failure_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+".json"),failure)
        raise
    finally:
        if lock.exists() and json.loads(lock.read_text())["token"]==owner["token"]:
            lock.unlink()


def _run(args):
    source_root=Path(__file__).resolve().parents[1]/"src"
    sys.path.insert(0,str(source_root))
    import kernelpack
    if kernelpack.__file__ is None or not Path(kernelpack.__file__).resolve().is_relative_to(source_root):
        raise RuntimeError("KernelPack must be imported from this pinned study worktree, not another installed checkout")
    from fastol_spatial_study_core import build_map,apply_map,apply_map_arrays,apply_map_jacobian_arrays
    device=configure_gpu()
    protocol=_source_and_artifact_protocol(args,device)
    protocol["suites"]=args.suites
    protocol["dimensions"]=args.dimensions
    protocol_hash=hashlib.sha256(_canonical(protocol)).hexdigest()
    path=args.output/"protocol.json"
    if path.exists():
        if not args.resume or hashlib.sha256(_canonical(json.loads(path.read_text()))).hexdigest()!=protocol_hash:
            raise RuntimeError("Existing output has a different protocol/source; use a fresh output directory")
    else:
        _atomic_json(path,protocol)
    def progress(event,**data):
        row=dict(time_utc=datetime.now(timezone.utc).isoformat(),event=event,**data)
        with (args.output/"events.jsonl").open("a",encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(row))+"\n")
        print(json.dumps(_json_safe(row)),flush=True)
    previous={}
    results=[]
    for dimension in args.dimensions:
        nquery=args.queries_2d if dimension==2 else args.queries_3d
        nb=args.boundary_queries_2d if dimension==2 else args.boundary_queries_3d
        interior,boundary=fixed_irregular_queries(dimension,nquery,nb,seed=args.query_seed)
        periodic_query,periodic_groups=periodic_probes(dimension,seed=args.query_seed+17)
        for suite in args.suites:
            levels=(args.spacing_2d if dimension==2 else args.spacing_3d) if suite=="irregular" else [args.periodic_grid_2d if dimension==2 else args.periodic_grid_3d]
            queries=jnp.concatenate((interior,boundary),axis=0) if suite=="irregular" else periodic_query
            truth=manufactured_field(queries,periodic=suite=="periodic")
            truth_jacobian=manufactured_jacobian(queries,periodic=suite=="periodic")
            query_hash=_digest_array(queries)
            for level,resolution in enumerate(levels):
                geometry_start=time.perf_counter()
                try:
                    cloud=(make_irregular_cloud(dimension,resolution,seed=args.node_seed)
                        if suite=="irregular" else make_periodic_cloud(dimension,resolution,seed=args.node_seed))
                except BaseException as error:
                    _preserve_cloud_failure(args.output,error,source_sha256=protocol["source_sha256"],
                        protocol_sha256=protocol_hash,specification=dict(suite=suite,dimension=dimension,
                            level=level,resolution=resolution,seed=args.node_seed))
                    raise
                try:
                    cloud.points.block_until_ready()
                    generation_seconds=time.perf_counter()-geometry_start
                    h_queries=interior if suite=="irregular" else jax.random.uniform(
                        jax.random.fold_in(jax.random.PRNGKey(args.query_seed+71),dimension),(nquery,dimension),
                        dtype=jnp.float64,minval=0.,maxval=2*jnp.pi)
                    geometry_start=time.perf_counter()
                    geometry,nearest_squared=geometry_metrics(cloud.points,h_queries,periodic=suite=="periodic")
                    jax.block_until_ready((geometry,nearest_squared))
                    geometry_seconds=time.perf_counter()-geometry_start
                    node_values=manufactured_field(cloud.points,periodic=suite=="periodic")
                    jax.block_until_ready((node_values,truth,truth_jacobian))
                    cloud_hash=_digest_array(cloud.points)
                    h_query_hash=_digest_array(h_queries)
                except BaseException as error:
                    _preserve_cloud_failure(args.output,error,source_sha256=protocol["source_sha256"],
                        protocol_sha256=protocol_hash,specification=dict(suite=suite,dimension=dimension,
                            level=level,resolution=resolution,seed=args.node_seed),
                        additional_arrays=dict(nodes=cloud.points,queries=queries,truth=truth,
                            truth_jacobian=truth_jacobian,**(cloud.geometry_evidence or {})),
                        stage_metadata=dict(stage="cloud_diagnostics_or_manufactured_nodal_field",cloud=cloud.metadata))
                    raise
                bounds=((0.,)*dimension,(2*math.pi,)*dimension) if suite=="periodic" else None
                for order in args.orders:
                    degree=order-1
                    rank=dimension*math.comb(degree+dimension,dimension)-math.comb(degree+dimension-1,dimension)
                    stencil=2*math.ceil(rank/dimension)+1
                    if len(cloud.points)<stencil:
                        raise RuntimeError("Prescribed cloud cannot contain the requested stencil")
                    for kernel in args.kernels:
                        key=f"{suite}_d{dimension}_p{order}_{kernel}_level{level}"
                        cell=args.output/"cells"/key
                        cell.mkdir(parents=True,exist_ok=True)
                        if (cell/"result.json").exists():
                            if not args.resume:
                                raise FileExistsError(cell/"result.json")
                            saved=json.loads((cell/"result.json").read_text())
                            if saved["protocol_sha256"]!=protocol_hash or saved["nodes_sha256"]!=cloud_hash or saved["queries_sha256"]!=query_hash or saved["sampled_h_queries_sha256"]!=h_query_hash:
                                raise RuntimeError("Existing cell does not match regenerated frozen inputs")
                            if _digest_file(cell/"arrays.npz")!=saved["arrays_sha256"]:
                                raise RuntimeError("Existing cell array artifact changed")
                            results.append(saved)
                            previous[suite,dimension,order,kernel]=(jnp.asarray(saved["geometry"]["sampled_fill_distance"]),jnp.asarray(saved["errors"]["relative_rms"]))
                            progress("cell_resume_verified",cell=key)
                            continue
                        _check_sources(protocol["source_sha256"])
                        progress("cell_started",cell=key,nodes=len(cloud.points),queries=len(queries),degree=degree,stencil_size=stencil)
                        try:
                            maps_start=time.perf_counter()
                            kwargs=dict(degree=degree,kernel=kernel,periodic_bounds=bounds,assignment="nearest-node",
                                batch_size=args.batch_size,allow_pseudoinverse=True,matern_base_length_factor=16.,
                                matern_num_modes=5,matern_gamma=8/3,stencil_domain=cloud.stencil_domain)
                            nodal=build_map(cloud.points,cloud.points,compute_gradients=False,
                                progress=lambda data: progress("nodal_preparation_batch",cell=key,**data),**kwargs)
                            progress("all_nodal_systems_prepared",cell=key,nodes=len(cloud.points))
                            query_map=build_map(cloud.points,queries,compute_gradients=True,
                                system_cache=nodal.system_cache,
                                progress=lambda data: progress("query_preparation_batch",cell=key,**data),**kwargs)
                            preparation_seconds=time.perf_counter()-maps_start
                            progress("query_maps_prepared",cell=key)
                            _assert_device_tree((cloud.points,queries,node_values,query_map.weights,query_map.weight_gradients))
                            compilation_start=time.perf_counter()
                            compiled_value=apply_map_arrays.lower(query_map.weights,query_map.stencil_indices,node_values).compile()
                            value_compile_seconds=time.perf_counter()-compilation_start
                            compilation_start=time.perf_counter()
                            compiled_gradient=apply_map_jacobian_arrays.lower(query_map.weight_gradients,query_map.stencil_indices,node_values).compile()
                            gradient_compile_seconds=time.perf_counter()-compilation_start
                            first_start=time.perf_counter()
                            with jax.transfer_guard("disallow"):
                                prediction=compiled_value(query_map.weights,query_map.stencil_indices,node_values).block_until_ready()
                                jacobian=compiled_gradient(query_map.weight_gradients,query_map.stencil_indices,node_values).block_until_ready()
                            first_seconds=time.perf_counter()-first_start
                            times=[]
                            for _ in range(args.repetitions):
                                start=time.perf_counter()
                                with jax.transfer_guard("disallow"):
                                    prediction=compiled_value(query_map.weights,query_map.stencil_indices,node_values).block_until_ready()
                                times.append(time.perf_counter()-start)
                            nodal_prediction=apply_map(nodal,node_values).block_until_ready()
                            nodal_error=jnp.linalg.norm(nodal_prediction-node_values)/jnp.maximum(jnp.linalg.norm(node_values),jnp.finfo(jnp.float64).tiny)
                            diagnostics,checks=_diagnostic_summary(nodal.diagnostics,rank,args.tolerance,args.condition_epsilon_limit)
                            query_diagnostics,query_checks=_diagnostic_summary(query_map.diagnostics,rank,args.tolerance,args.condition_epsilon_limit)
                            checks.update({"query_"+name:value for name,value in query_checks.items()})
                            unreachable=sum((mapping.metadata.get("boundary_stencil_statistics") or {}).get("fallback_rows",0)
                                for mapping in (nodal,query_map))
                            checks["zero_unreachable_stencil_fallback"] = jnp.asarray(unreachable==0)
                            oracle_start=time.perf_counter()
                            independent_polynomial=_independent_polynomial_check(query_map,degree)
                            independent_nodal_polynomial=_independent_polynomial_check(nodal,degree)
                            jax.block_until_ready((independent_polynomial["generator_relative_rms"],
                                                  independent_nodal_polynomial["generator_relative_rms"]))
                            independent_polynomial_seconds=time.perf_counter()-oracle_start
                            checks["independent_requested_polynomial_reproduction"]=independent_polynomial["maximum_generator_relative_rms"]<=args.tolerance
                            checks["independent_requested_polynomial_finite"]=independent_polynomial["finite"]
                            checks["independent_nodal_requested_polynomial_reproduction"]=independent_nodal_polynomial["maximum_generator_relative_rms"]<=args.tolerance
                            checks["independent_nodal_requested_polynomial_finite"]=independent_nodal_polynomial["finite"]
                            block_norm_sum=jnp.sum(jnp.linalg.norm(query_map.weights,axis=(1,3)),axis=1)
                            weight_norms=dict(definition="sum of Frobenius norms of d-by-d nodal blocks; upper bound on block spectral-sum norm",
                                maximum=_scalar(jnp.max(block_norm_sum)),
                                quantiles_0_50_90_99_100=jax.device_get(jnp.quantile(block_norm_sum,jnp.asarray([0.,.5,.9,.99,1.]))).tolist())
                            checks["finite_interpolation_norms"]=jnp.all(jnp.isfinite(block_norm_sum))
                            checks["unique_physical_nodes"]=geometry["minimum_node_separation"]>0
                            checks["finite_physical_nodes"]=jnp.all(jnp.isfinite(cloud.points))
                            derivative_mask=query_map.diagnostics["derivative_defined"]
                            main_count=nquery if suite=="irregular" else len(queries)
                            errors=_error_metrics(prediction[:main_count],truth[:main_count],jacobian[:main_count],truth_jacobian[:main_count],derivative_mask[:main_count])
                            checks["finite_predictions"]=errors["finite"]
                            checks["nodal_field_interpolation"]=nodal_error<=args.tolerance
                            checks["local_divergence"]=errors["divergence_normalized_rms"]<=args.tolerance
                            checks["defined_derivative_queries_exist"]=jnp.count_nonzero(derivative_mask)>0
                            checks["query_divergence_weights"]=jnp.all(jnp.where(derivative_mask,
                                query_map.diagnostics["divergence_weight_normalized"]<=args.tolerance,True))
                            checks["query_gradient_finite_away_from_centers"]=jnp.all(jnp.where(derivative_mask,
                                query_map.diagnostics["gradient_finite"],True))
                            checks["query_nodal_interpolation"]=jnp.all(jnp.where(query_map.diagnostics["query_is_stencil_node"],
                                query_map.diagnostics["query_nodal_interpolation_relative_error"]<=args.tolerance,True))
                            extra={}
                            if suite=="irregular":
                                boundary_difference=prediction[nquery:]-truth[nquery:]
                                extra["boundary"]=dict(absolute_rms=jnp.sqrt(jnp.mean(jnp.sum(boundary_difference**2,axis=1))),
                                    sampled_maximum=jnp.max(jnp.linalg.norm(boundary_difference,axis=1)),
                                    exact_velocity_maximum=jnp.max(jnp.linalg.norm(truth[nquery:],axis=1)),
                                    maximum_level_set_residual=jnp.max(jnp.abs(level_set(queries[nquery:]))))
                            else:
                                extra["periodicity"]=_periodicity_metrics(prediction,jacobian,truth,truth_jacobian,periodic_groups)
                                for kind,metrics in extra["periodicity"].items():
                                    checks[f"periodic_{kind}_values"]=metrics["value_jump"]["sampled_maximum"]<=args.tolerance*jnp.maximum(1.,jnp.max(jnp.linalg.norm(truth,axis=1)))
                                    checks[f"periodic_{kind}_derivatives"]=metrics["jacobian_jump"]["sampled_maximum"]<=args.tolerance*jnp.maximum(1.,jnp.max(jnp.linalg.norm(truth_jacobian,axis=(1,2))))
                            previous_key=(suite,dimension,order,kernel)
                            slope=None
                            if previous_key in previous:
                                old_h,old_error=previous[previous_key]
                                slope=jnp.log(old_error/errors["relative_rms"])/jnp.log(old_h/geometry["sampled_fill_distance"])
                            previous[previous_key]=(geometry["sampled_fill_distance"],errors["relative_rms"])
                            arrays=dict(nodes=cloud.points,queries=queries,nodal_values=node_values,truth=truth,
                                prediction=prediction,truth_jacobian=truth_jacobian,prediction_jacobian=jacobian,
                                local_stencil_indices=nodal.stencil_indices,query_center_indices=query_map.center_indices,
                                nodal_prediction=nodal_prediction,sampled_h_nearest_squared=nearest_squared)
                            arrays.update(sampled_h_queries=h_queries,query_stencil_indices=query_map.stencil_indices,
                                query_image_coordinates=query_map.image_coords,query_local_coordinates=query_map.query_coords,
                                interpolation_block_frobenius_sum=block_norm_sum,
                                independent_polynomial_generator_relative_rms=independent_polynomial["generator_relative_rms"],
                                independent_nodal_polynomial_generator_relative_rms=independent_nodal_polynomial["generator_relative_rms"])
                            arrays.update(cloud.geometry_evidence or {})
                            arrays["unreachable_euclidean_stencil_fallback_rows"] = jnp.asarray(unreachable,dtype=jnp.int64)
                            arrays.update({"nodal_diagnostic_"+k:v for k,v in nodal.diagnostics.items() if isinstance(v,jax.Array)})
                            arrays.update({"query_diagnostic_"+k:v for k,v in query_map.diagnostics.items() if isinstance(v,jax.Array)})
                            _assert_device_tree((arrays,geometry,errors))
                            _atomic_npz(cell/"arrays.npz",arrays)
                            record=dict(cell=key,suite=suite,dimension=dimension,requested_value_order=order,polynomial_degree=degree,
                                kernel=kernel,level=level,points=len(cloud.points),queries=len(queries),stencil_size=stencil,
                                polynomial_dimension=rank,protocol_sha256=protocol_hash,nodes_sha256=cloud_hash,queries_sha256=query_hash,
                                sampled_h_queries_sha256=h_query_hash,
                                arrays_sha256=_digest_file(cell/"arrays.npz"),cloud=cloud.metadata,
                                geometry={k:_scalar(v) for k,v in geometry.items()},
                                errors={k:_scalar(v) for k,v in errors.items()},nodal_field_relative_error=_scalar(nodal_error),
                                successive_sampled_h_slope=None if slope is None else _scalar(slope),diagnostics=diagnostics,
                                query_diagnostics=query_diagnostics,interpolation_norms=weight_norms,
                                independent_polynomial={k:_scalar(v) if isinstance(v,jax.Array) else v for k,v in independent_polynomial.items() if k!="generator_relative_rms"},
                                independent_nodal_polynomial={k:_scalar(v) if isinstance(v,jax.Array) else v for k,v in independent_nodal_polynomial.items() if k!="generator_relative_rms"},
                                nodal_map_metadata=nodal.metadata,query_map_metadata=query_map.metadata,
                                unreachable_euclidean_stencil_fallback_rows=unreachable,
                                visible_array_accounting=dict(bytes=_visible_array_bytes(arrays,
                                    query_map.weights,query_map.weight_gradients,nodal.weights,
                                    nodal.system_cache.entries),
                                    scope="array-object shape sum; aliases may share buffers; excludes allocator and compilation workspace; not peak VRAM"),
                                scientific_checks={k:_scalar(v) for k,v in checks.items()},
                                passed=_scalar(jnp.all(jnp.stack(list(checks.values())))),
                                timings=dict(cloud_generation_including_first_compilation_seconds=generation_seconds,
                                    geometry_diagnostics_including_compilation_seconds=geometry_seconds,
                                    local_preparation_including_first_compilation_seconds=preparation_seconds,
                                    value_evaluation_compilation_seconds=value_compile_seconds,
                                    derivative_evaluation_compilation_seconds=gradient_compile_seconds,
                                    first_compiled_value_and_derivative_execution_seconds=first_seconds,
                                    independent_polynomial_oracle_including_compilation_seconds=independent_polynomial_seconds,
                                    warm_value_evaluation_seconds=times,
                                    warm_value_median_seconds=_scalar(jnp.median(jnp.asarray(times,dtype=jnp.float64)))),
                                periodic_probe_groups=periodic_groups if suite=="periodic" else None,
                                completed_utc=datetime.now(timezone.utc).isoformat())
                            record.update({k:jax.tree_util.tree_map(lambda v:_scalar(v) if isinstance(v,jax.Array) else v,value) for k,value in extra.items()})
                            _check_sources(protocol["source_sha256"])
                            _atomic_json(cell/"result.json",record)
                            results.append(record)
                            progress("cell_complete",cell=key,passed=record["passed"],error=record["errors"]["relative_rms"],slope=record["successive_sampled_h_slope"])
                            del nodal,query_map,prediction,jacobian,nodal_prediction,arrays,compiled_value,compiled_gradient
                            gc.collect()
                        except BaseException as exc:
                            failure=dict(cell=key,protocol_sha256=protocol_hash,type=type(exc).__name__,message=str(exc),
                                nodes_sha256=cloud_hash,queries_sha256=query_hash,
                                traceback=traceback.format_exc(),time_utc=datetime.now(timezone.utc).isoformat())
                            failure_path=cell/("failure_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+".json")
                            evidence_arrays=dict(nodes=cloud.points,queries=queries,nodal_values=node_values,
                                truth=truth,truth_jacobian=truth_jacobian,sampled_h_queries=h_queries)
                            evidence=getattr(exc,"diagnostics",None)
                            if evidence is not None:
                                evidence_arrays.update({"diagnostic_"+name:value for name,value in evidence.items() if isinstance(value,jax.Array)})
                            centers=getattr(exc,"center_indices",None)
                            if centers is not None:
                                evidence_arrays["center_indices"]=centers
                            evidence_path=failure_path.with_suffix(".npz")
                            _atomic_npz(evidence_path,evidence_arrays)
                            failure["diagnostic_artifact"]=str(evidence_path)
                            failure["diagnostic_artifact_sha256"]=_digest_file(evidence_path)
                            _atomic_json(failure_path,failure)
                            progress("cell_execution_failed",cell=key,failure=str(failure_path))
                            raise
    _check_sources(protocol["source_sha256"])
    if len(results)!=protocol["configuration_count"]:
        raise RuntimeError(f"Expected {protocol['configuration_count']} cells, got {len(results)}")
    summary=dict(protocol_sha256=protocol_hash,completed_cells=len(results),
        scientific_checks_passed=sum(row["passed"] for row in results),
        failed_scientific_cells=[row["cell"] for row in results if not row["passed"]],
        results=[dict(cell=row["cell"],passed=row["passed"],error=row["errors"]["relative_rms"],
                      h=row["geometry"]["sampled_fill_distance"],slope=row["successive_sampled_h_slope"]) for row in results],
        finished_utc=datetime.now(timezone.utc).isoformat())
    completion_path=args.output/"completion.json"
    if completion_path.exists():
        saved_completion=json.loads(completion_path.read_text())
        for field in ("protocol_sha256","completed_cells","scientific_checks_passed","failed_scientific_cells","results"):
            if saved_completion[field]!=summary[field]:
                raise RuntimeError("Existing completion report disagrees with independently reverified cells")
    else:
        _atomic_json(completion_path,summary)
    progress("study_complete",completed_cells=len(results),scientific_checks_passed=summary["scientific_checks_passed"])
    return 0 if not summary["failed_scientific_cells"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
