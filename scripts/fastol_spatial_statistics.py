"""GPU spatial increment statistics, separate from operator-learning benchmarks.

Protocol: exact finite periodic solenoidal Fourier prior with a declared
Kolmogorov shell-energy target; local five-scale Matérn + polynomial maps on
irregular 3D clouds; orders 2/4/6; shared boundary-aware node stencils. Compare the exact
prior increment covariance to its local reconstruction, not to one lucky draw.

The PHS control and two declared Matérn policies are compared:
``phs``, ``ram_local_tau16`` and ``fixed_physical_band``. The last freezes ONE physical five-mode Matérn band
from the prior before spatial refinement. It does not select a different scale
at each query or claim to implement adaptive stencil selection.

All array algebra, random fields, statistics and checks run on CUDA float64.
Host operations are integer index enumeration, orchestration and artifact I/O.
No global dense covariance is stored or factored. Fourier-feature contractions
evaluate the same covariance quadratic forms using only each union of local
stencil supports, in bounded mode and pair batches.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from fastol_spatial_study_core import build_map, apply_map, apply_map_jacobian, apply_map_arrays
from fastol_spatial_properties import (
    configure_gpu, make_irregular_cloud, make_periodic_cloud, level_set, periodic_probes,
    _preserve_cloud_failure,
)

STATISTICAL_POLICIES = ("phs", "ram_local_tau16", "fixed_physical_band")


def _frozen_module_origins():
    root=Path(__file__).resolve().parents[1]
    source_root=root/"src"
    origins={}
    for name in ("kernelpack","kernelpack.accelerators.jax_spatial","kernelpack.domain.core",
        "kernelpack.domain.stencils","kernelpack.geometry.spherical_pu","kernelpack.geometry.cfpu",
        "kernelpack.divfree.core","kernelpack.divfree.matern","kernelpack.kernels.radial"):
        module=importlib.import_module(name)
        if module.__file__ is None or not Path(module.__file__).resolve().is_relative_to(source_root):
            raise RuntimeError("Statistics imported a package outside the pinned source tree: "+name)
        origins[name]=Path(module.__file__).resolve().relative_to(root).as_posix()
    for name in ("fastol_spatial_study_core","fastol_spatial_properties","fastol_scattered_clouds"):
        module=importlib.import_module(name)
        if module.__file__ is None or Path(module.__file__).resolve().parent != root/"scripts":
            raise RuntimeError("Statistics imported an adapter outside the pinned source tree: "+name)
        origins[name]=Path(module.__file__).resolve().relative_to(root).as_posix()
    return origins


@dataclass(frozen=True)
class FourierPrior:
    """u(x)=sum sqrt(w) B[a cos(k.x)+b sin(k.x)], a,b independent N(0,I2)."""
    wavevectors: jax.Array
    weights: jax.Array
    frames: jax.Array
    cosine_draws: jax.Array
    sine_draws: jax.Array
    physical_matern_base_length: jax.Array


def _scalar(value):
    """Only scalar report values cross to the host outside explicit artifact I/O."""
    value.block_until_ready()
    return value.item()


def _json_safe(value):
    if isinstance(value, jax.Array):
        return _json_safe(jax.device_get(value).tolist())
    if type(value).__module__.startswith("numpy") and hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
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
    """Explicit artifact export, with the same C-order byte hash as the smooth study."""
    value.block_until_ready()
    return hashlib.sha256(jax.device_get(value).tobytes(order="C")).hexdigest()


def _write_json(path, data, *, immutable=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        raise FileExistsError(f"Refusing to replace immutable artifact {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(json.dumps(_json_safe(data), indent=2, allow_nan=False).encode() + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    if immutable and path.exists():
        raise FileExistsError(path)
    os.replace(temporary, path)
    return _digest_file(path)


def _export_npz(path, **arrays):
    # Explicit output I/O, never a numerical fallback.
    import numpy as np
    if path.exists():
        raise FileExistsError(f"Refusing to replace immutable artifact {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    exported = {key: jax.device_get(value) for key, value in arrays.items() if value is not None}
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **exported)
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        raise FileExistsError(path)
    os.replace(temporary, path)
    return _digest_file(path)


def _sources():
    root = Path(__file__).resolve().parents[1]
    paths = [root/"scripts"/name for name in (
        "fastol_spatial_statistics.py", "fastol_spatial_study_core.py", "fastol_spatial_properties.py",
        "fastol_scattered_clouds.py")]
    paths.extend(p for p in (root/"src/kernelpack").rglob("*") if p.is_file()
                 and p.suffix.lower() in (".py", ".cu", ".cuh", ".cc", ".cpp", ".h", ".hpp", ".pyd", ".so"))
    return {str(path.relative_to(root)).replace("\\", "/"): _digest_file(path)
            for path in sorted(set(paths))}


def _check_sources(expected):
    if _sources() != expected:
        raise RuntimeError("A pinned statistics, study, or production source changed during execution")


def _input_provenance(args, name, arrays):
    path = args.output/name
    array_hashes = {key: _digest_array(value) for key, value in arrays.items() if value is not None}
    file_hash = _export_npz(path, **arrays)
    return dict(input_artifact=path.relative_to(args.output).as_posix(),
                input_artifact_sha256=file_hash, input_sha256=array_hashes)


def _cell_pins(provenance, inputs):
    return dict(protocol_sha256=provenance["protocol_sha256"],
                reference_npz_sha256=provenance["reference_npz_sha256"], **inputs)


def _warm_value_timing(mapping, fields, repetitions):
    """Synchronized GPU value application; compilation is separately recorded."""
    values = jnp.transpose(fields, (0, 2, 1))
    started = time.perf_counter()
    compiled = apply_map_arrays.lower(mapping.weights, mapping.stencil_indices, values).compile()
    compilation = time.perf_counter() - started
    with jax.transfer_guard("disallow"):
        compiled(mapping.weights, mapping.stencil_indices, values).block_until_ready()
    times = []
    for _ in range(repetitions):
        started = time.perf_counter()
        with jax.transfer_guard("disallow"):
            compiled(mapping.weights, mapping.stencil_indices, values).block_until_ready()
        times.append(time.perf_counter() - started)
    return dict(value_evaluation_compilation_seconds=compilation,
        warm_value_evaluation_seconds=times,
        warm_value_median_seconds=float(_scalar(jnp.median(jnp.asarray(times, dtype=jnp.float64)))))


def _preserve_failure(folder, error, *, provenance, inputs, nodes, queries,
                      nodal_values, mapping=None, extra_arrays=None, candidate_report=None):
    """Save the rejected inputs and on-device exception diagnostics before re-raising."""
    arrays = dict(nodes=nodes, queries=queries, nodal_values=nodal_values)
    if mapping is not None:
        arrays.update(center_indices=mapping.center_indices, stencil_indices=mapping.stencil_indices,
                      weights=mapping.weights)
        arrays.update({"map_"+key: value for key, value in mapping.diagnostics.items()})
    if getattr(error, "center_indices", None) is not None:
        arrays["exception_center_indices"] = error.center_indices
    for key, value in (getattr(error, "diagnostics", None) or {}).items():
        if isinstance(value, jax.Array):
            arrays["exception_"+key] = value
    arrays.update(extra_arrays or {})
    failure = dict(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                   **_cell_pins(provenance, inputs))
    if mapping is not None:
        failure["map_metadata"] = mapping.metadata
    if candidate_report is not None:
        failure["rejected_report"] = candidate_report
    try:
        failure["failure_arrays_npz_sha256"] = _export_npz(folder/"failure_arrays.npz", **arrays)
    except Exception as preservation_error:
        failure["array_preservation_error"] = repr(preservation_error)
    _write_json(folder/"failure.json", failure)


def make_prior(device, *, maximum_wavenumber=1024, draws=8, seed=7013):
    """Positive angular/radial quadrature of E(k) proportional to k^(-5/3).

    Integer primitive directions make every mode exactly 2pi-periodic. Angular
    quadrature has cubic symmetry, not exact continuous rotational invariance.
    Actual directional averages and resolved slopes are checked below.
    """
    directions = []
    for vector in itertools.product(range(-2, 3), repeat=3):
        if vector == (0, 0, 0) or next(v for v in vector if v) < 0:
            continue
        if math.gcd(math.gcd(abs(vector[0]), abs(vector[1])), abs(vector[2])) == 1:
            directions.append(vector)
    # Prespecified integer logarithmic shell representatives. They are wave
    # indices, not a floating-point host computation of the prior weights.
    harmonics=[n for n in (1,2,3,4,5,6,7,8,10,12,14,16,20,24,28,32,40,48,
        56,64,80,96,112,128,160,192,224,256,320,384,448,512,640,768,896,1024)
        if n<=maximum_wavenumber]
    if harmonics[-1]!=maximum_wavenumber:
        harmonics.append(maximum_wavenumber)
    integer_vectors = [tuple(n*v for v in direction)
                       for direction in directions for n in harmonics]
    # Integer enumeration above is topology/metadata; physical weights are GPU algebra.
    k = jax.device_put(jnp.asarray(integer_vectors, dtype=jnp.float64), device)
    lengths = jnp.linalg.norm(k, axis=1)
    angular_norm = jnp.linalg.norm(jax.device_put(jnp.asarray(directions, dtype=k.dtype), device), axis=1)
    shell_radius=angular_norm[:,None]*jnp.asarray(harmonics,dtype=k.dtype)[None]
    boundaries=jnp.sqrt(shell_radius[:,:-1]*shell_radius[:,1:])
    lower=jnp.concatenate([jnp.ones((len(directions),1),dtype=k.dtype),boundaries],axis=1)
    upper=jnp.concatenate([boundaries,jnp.full((len(directions),1),maximum_wavenumber,dtype=k.dtype)],axis=1)
    lower=jnp.clip(lower,1.,maximum_wavenumber)
    upper=jnp.clip(upper,1.,maximum_wavenumber)
    # Exact shell-integral weights for the specified radial E(k)=k^(-5/3).
    raw=(1.5*jnp.maximum(lower**(-2/3)-upper**(-2/3),0.)).reshape(-1)
    raw=jnp.where(lengths<=maximum_wavenumber,raw,0.)
    weights = raw / (2 * jnp.sum(raw))  # tr K(x,x)=2 sum w=1.
    unit = k / lengths[:, None]
    axes = jnp.eye(3, dtype=k.dtype)[jnp.argmin(jnp.abs(unit), axis=1)]
    first = jnp.cross(unit, axes)
    first = first / jnp.linalg.norm(first, axis=1)[:, None]
    frames = jnp.stack([first, jnp.cross(unit, first)], axis=-1)
    key_a, key_b = jax.random.split(jax.random.key(seed))
    cosine = jax.random.normal(key_a, (draws, len(k), 2), dtype=k.dtype)
    sine = jax.random.normal(key_b, cosine.shape, dtype=k.dtype)
    effective_length = jnp.exp(jnp.sum(weights * jnp.log(1 / lengths)) / jnp.sum(weights))
    prior = FourierPrior(k, weights, frames, cosine, sine, 8 * effective_length)
    jax.block_until_ready((k, weights, frames, cosine, sine))
    return prior


def _prior_blocks(prior, mode_batch):
    padding = (-len(prior.weights)) % mode_batch
    def pad(value, axes):
        return jnp.pad(value, axes)
    k = pad(prior.wavevectors, ((0, padding), (0, 0))).reshape(-1, mode_batch, 3)
    b = pad(prior.frames * jnp.sqrt(prior.weights)[:, None, None],
            ((0, padding), (0, 0), (0, 0))).reshape(-1, mode_batch, 3, 2)
    a = pad(prior.cosine_draws, ((0, 0), (0, padding), (0, 0)))
    a = jnp.moveaxis(a.reshape(a.shape[0], -1, mode_batch, 2), 1, 0)
    s = pad(prior.sine_draws, ((0, 0), (0, padding), (0, 0)))
    s = jnp.moveaxis(s.reshape(s.shape[0], -1, mode_batch, 2), 1, 0)
    return k, b, a, s


@jax.jit
def _field_batch(points, blocks):
    k, b, a, s = blocks
    initial = jnp.zeros((len(points), a.shape[1], 3), dtype=points.dtype)
    def step(total, block):
        wave, frame, ca, si = block
        phase = points @ wave.T
        c = jnp.einsum("mf,fdr,efr->med", jnp.cos(phase), frame, ca,
                       precision=jax.lax.Precision.HIGHEST)
        v = jnp.einsum("mf,fdr,efr->med", jnp.sin(phase), frame, si,
                       precision=jax.lax.Precision.HIGHEST)
        return total + c + v, None
    return jax.lax.scan(step, initial, (k, b, a, s))[0]


def sample_fields(points, blocks, *, point_batch=256):
    """Actual, reproducible smooth manufactured GP realizations [point,draw,d]."""
    chunks = []
    for start in range(0, len(points), point_batch):
        part = points[start:start + point_batch]
        padding = point_batch - len(part)
        value = _field_batch(jnp.pad(part, ((0, padding), (0, 0)), mode="edge"), blocks)
        chunks.append(value[:len(part)])
    return jnp.concatenate(chunks)


@jax.jit
def _field_jacobian_batch(points, blocks):
    """Analytic Fourier derivatives [point,out,derivative,draw], without dense AD."""
    k, b, a, s = blocks
    initial = jnp.zeros((len(points), 3, 3, a.shape[1]), dtype=points.dtype)
    def step(total, block):
        wave, frame, ca, si = block
        phase = points @ wave.T
        cosine_derivative = jnp.einsum("mf,fdr,efr,fj->mdje", -jnp.sin(phase), frame, ca, wave,
            precision=jax.lax.Precision.HIGHEST)
        sine_derivative = jnp.einsum("mf,fdr,efr,fj->mdje", jnp.cos(phase), frame, si, wave,
            precision=jax.lax.Precision.HIGHEST)
        return total + cosine_derivative + sine_derivative, None
    return jax.lax.scan(step, initial, (k, b, a, s))[0]


def sample_field_jacobians(points, blocks, *, point_batch=256):
    chunks = []
    for start in range(0, len(points), point_batch):
        part = points[start:start + point_batch]
        padding = point_batch - len(part)
        value = _field_jacobian_batch(jnp.pad(part, ((0, padding), (0, 0)), mode="edge"), blocks)
        chunks.append(value[:len(part)])
    return jnp.concatenate(chunks)


@jax.jit
def _pair_batch(wy, xy, y, wz, xz, z, blocks):
    """Exact local-union covariance quadratic forms via bounded Fourier factors."""
    wave, frame, ca, si = blocks
    count, draws = len(y), ca.shape[1]
    energy = jnp.zeros((count, 5), dtype=y.dtype)
    fields = jnp.zeros((count, draws, 3), dtype=y.dtype)
    initial = energy, fields, fields, fields, fields
    def step(state, block):
        energies, py, pz, ty, tz = state
        k, b, a, s = block
        fy, fz = jnp.einsum("pkd,fd->pkf", xy, k), jnp.einsum("pkd,fd->pkf", xz, k)
        cy = jnp.einsum("pdki,pkf,fir->pfdr", wy, jnp.cos(fy), b,
                        precision=jax.lax.Precision.HIGHEST)
        sy = jnp.einsum("pdki,pkf,fir->pfdr", wy, jnp.sin(fy), b,
                        precision=jax.lax.Precision.HIGHEST)
        cz = jnp.einsum("pdki,pkf,fir->pfdr", wz, jnp.cos(fz), b,
                        precision=jax.lax.Precision.HIGHEST)
        sz = jnp.einsum("pdki,pkf,fir->pfdr", wz, jnp.sin(fz), b,
                        precision=jax.lax.Precision.HIGHEST)
        qy, qz = y @ k.T, z @ k.T
        tcy, tsy = jnp.cos(qy)[:, :, None, None] * b, jnp.sin(qy)[:, :, None, None] * b
        tcz, tsz = jnp.cos(qz)[:, :, None, None] * b, jnp.sin(qz)[:, :, None, None] * b
        norm2 = lambda c, s: jnp.sum(c*c + s*s, axis=(1, 2, 3))
        dc, ds, tc, ts = cy-cz, sy-sz, tcy-tcz, tsy-tsz
        energies = energies + jnp.stack([norm2(dc, ds), norm2(tc, ts),
            norm2(cy-tcy, sy-tsy), norm2(cz-tcz, sz-tsz),norm2(dc-tc,ds-ts)], axis=1)
        apply = lambda c, v: jnp.einsum("pfdr,efr->ped", c, v,
                                       precision=jax.lax.Precision.HIGHEST)
        return (energies, py+apply(cy,a)+apply(sy,s), pz+apply(cz,a)+apply(sz,s),
                ty+apply(tcy,a)+apply(tsy,s), tz+apply(tcz,a)+apply(tsz,s)), None
    return jax.lax.scan(step, initial, (wave, frame, ca, si))[0]


def pair_statistics(points, mapping, y, z, blocks, *, pair_batch=4):
    count = len(y)
    if len(mapping.weights) != 2 * count:
        raise ValueError("Map must contain all y then all z query rows")
    nodal = points[mapping.stencil_indices]
    outputs = [[] for _ in range(5)]
    for start in range(0, count, pair_batch):
        stop = min(count, start+pair_batch)
        def padded(value):
            part = value[start:stop]
            return jnp.pad(part, ((0,pair_batch-len(part)),)+((0,0),)*(part.ndim-1), mode="edge")
        result = _pair_batch(padded(mapping.weights[:count]), padded(nodal[:count]), padded(y),
            padded(mapping.weights[count:]), padded(nodal[count:]), padded(z), blocks)
        for target, value in zip(outputs, result):
            target.append(value[:stop-start])
    return tuple(jnp.concatenate(values) for values in outputs)


@jax.jit
def exact_increment(prior_k, prior_w, separation_vectors):
    # tr(P_k)=2. Stable half-angle form avoids cancellation as r -> 0.
    phase = separation_vectors @ prior_k.T
    return 8 * (jnp.sin(.5*phase)**2 @ prior_w)


def make_pairs(device, *, bins=32, directions=8, seed=9109, r_min=1e-5, r_max=.8):
    kd, ka = jax.random.split(jax.random.key(seed))
    vectors = jax.random.normal(kd, (directions, 3), dtype=jnp.float64)
    vectors = vectors / jnp.linalg.norm(vectors, axis=1)[:, None]
    # Fixed anchors are in the common interior of the analytic irregular domain.
    anchors = .08 * jax.random.uniform(ka, (directions,3), minval=-1., maxval=1., dtype=vectors.dtype)
    radius = jnp.geomspace(r_min, r_max, bins)
    delta = radius[:, None, None]*vectors[None]
    y, z = (anchors[None]-.5*delta).reshape(-1,3), (anchors[None]+.5*delta).reshape(-1,3)
    if not bool(jnp.all(level_set(y)>0) & jnp.all(level_set(z)>0)):
        raise ValueError("Prespecified statistical pair endpoints left the irregular domain")
    return radius, vectors, y, z


def _angular_grid(count, dtype):
    i=jnp.arange(count,dtype=dtype)
    z=1-2*(i+.5)/count
    azimuth=jnp.pi*(3-jnp.sqrt(jnp.asarray(5.,dtype=dtype)))*i
    transverse=jnp.sqrt(jnp.maximum(1-z*z,0.))
    return jnp.stack([transverse*jnp.cos(azimuth),transverse*jnp.sin(azimuth),z],axis=1)


def certify_reference(prior, radius, directions, *, slope_tolerance=.10,
                      minimum_decades=math.log10(4), angular_tolerance=.02):
    values = exact_increment(prior.wavevectors, prior.weights,
                             (radius[:,None,None]*directions[None]).reshape(-1,3))
    directional = values.reshape(len(radius),len(directions))
    # Independent spherical quadrature, not a slope crossing on one direction.
    angular=[]
    for count in (64,128):
        angular_directions=_angular_grid(count,radius.dtype)
        energy=exact_increment(prior.wavevectors,prior.weights,
            (radius[:,None,None]*angular_directions[None]).reshape(-1,3))
        angular.append(jnp.mean(energy.reshape(len(radius),count),axis=1))
    mean=angular[1]
    angular_gap=jnp.abs(angular[0]-mean)/mean
    slopes = jnp.diff(jnp.log(mean))/jnp.diff(jnp.log(radius))
    pair_mean=jnp.mean(directional,axis=1)
    pair_slopes=jnp.diff(jnp.log(pair_mean))/jnp.diff(jnp.log(radius))
    qualifies=((jnp.abs(slopes-2/3)<=slope_tolerance)
               &(jnp.abs(pair_slopes-2/3)<=slope_tolerance)
               &(angular_gap[:-1]<=angular_tolerance)&(angular_gap[1:]<=angular_tolerance))
    # Run lengths remain on GPU; only the scalar-selected interval is exported.
    indices = jnp.arange(len(slopes))
    def step(last_bad, data):
        i, ok = data
        last_bad = jnp.where(ok,last_bad,i)
        start = last_bad+1
        span = jnp.where(ok,jnp.log10(radius[i+1]/radius[start]),0.)
        return last_bad,(start,span)
    _,(starts,spans)=jax.lax.scan(step,jnp.asarray(-1,dtype=indices.dtype),(indices,qualifies))
    end=jnp.argmax(spans); start=starts[end]
    resolved=spans[end]>=minimum_decades
    interval_mask=(indices>=start)&(indices<=end)&resolved
    point_mask=jnp.concatenate([interval_mask[:1],interval_mask])
    # Include the first endpoint of the selected interval, even when start > 0.
    point_mask=point_mask.at[start].set(resolved)
    return dict(mean=mean,directional=directional,slopes=slopes,pair_slopes=pair_slopes,
                angular_relative_gap=angular_gap,
                pair_mean=pair_mean,
                interval_mask=point_mask,resolved=resolved,
                start=start,end=end+1,decades=spans[end])


def summarize_map(mapping):
    diagnostics=mapping.diagnostics
    result={}
    for key,value in diagnostics.items():
        value=jnp.asarray(value)
        if value.dtype==jnp.bool_:
            result[key+"_count"]=int(_scalar(jnp.sum(value)))
        else:
            finite=jnp.isfinite(value)
            result[key+"_all_finite"]=bool(_scalar(jnp.all(finite)))
            result[key+"_min"]=float(_scalar(jnp.min(jnp.where(finite,value,jnp.inf)))) if bool(_scalar(jnp.any(finite))) else None
            result[key+"_max"]=float(_scalar(jnp.max(jnp.where(finite,value,-jnp.inf)))) if bool(_scalar(jnp.any(finite))) else None
    retained=diagnostics["retained_rank"]
    requested=diagnostics["requested_rank"]
    result["truncated_map_count"]=int(_scalar(jnp.sum(retained<requested)))
    finite_diagnostics=jnp.all(jnp.stack([
        jnp.all(jnp.isfinite(jnp.asarray(value))) for value in diagnostics.values()]))
    unreachable=(mapping.metadata.get("boundary_stencil_statistics") or {}).get("fallback_rows",0)
    checks=(jnp.all(diagnostics["finite"])
            & finite_diagnostics
            & jnp.all(retained==requested)
            & ~jnp.any(diagnostics["fallback_activated"])
            & jnp.all(diagnostics["solver_backward_error"]<=1e-10)
            & jnp.all(diagnostics["interpolation_relative_residual"]<=1e-8)
            & jnp.all(diagnostics["retained_polynomial_relative_error"]<=1e-8)
            & jnp.all(diagnostics["requested_polynomial_relative_error"]<=1e-8)
            & jnp.asarray(unreachable==0))
    result["unreachable_euclidean_stencil_fallback_rows"]=unreachable
    result["zero_unreachable_stencil_fallback"]=unreachable==0
    result["numerical_checks_passed"]=bool(_scalar(checks))
    result["finite_diagnostics"]=bool(_scalar(finite_diagnostics))
    result["full_requested_space_retained"]=bool(_scalar(jnp.all(retained==requested)))
    return result


def _report_metrics(energies, py, pz, ty, tz, *, bins, directions, radius, interval, epsilon):
    reconstructed,reference,ey,ez,increment_mse=energies.T
    ratio=reconstructed/reference
    bound=(jnp.sqrt(ey)+jnp.sqrt(ez))/jnp.sqrt(reference)
    observed=jnp.abs(jnp.sqrt(ratio)-1)
    sharp_bound=jnp.sqrt(increment_mse/reference)
    # Direct feature contraction must satisfy the deterministic L2 bound.
    bound_check=jnp.all((observed<=sharp_bound+1e-10)&(sharp_bound<=bound+1e-10))
    empirical_true=jnp.sum((ty-tz)**2,axis=2)
    empirical_pred=jnp.sum((py-pz)**2,axis=2)
    reshape=lambda a:a.reshape(bins,directions,-1)
    truth_draw_bins=jnp.mean(reshape(empirical_true),axis=1)
    pred_draw_bins=jnp.mean(reshape(empirical_pred),axis=1)
    se=lambda a:jnp.std(a,axis=1,ddof=1)/jnp.sqrt(a.shape[1])
    mean_ref=jnp.mean(reference.reshape(bins,directions),axis=1)
    mean_rec=jnp.mean(reconstructed.reshape(bins,directions),axis=1)
    reference_slopes=jnp.diff(jnp.log(mean_ref))/jnp.diff(jnp.log(radius))
    reconstructed_slopes=jnp.diff(jnp.log(mean_rec))/jnp.diff(jnp.log(radius))
    selected_slopes=interval[:-1]&interval[1:]
    selected=jnp.repeat(interval,directions)
    selected_count=jnp.sum(selected)
    max_on=lambda a:jnp.max(jnp.where(selected,a,-jnp.inf))
    metrics=dict(
        maximum_relative_s2_error=float(_scalar(jnp.max(jnp.abs(ratio-1)))),
        maximum_relative_increment_amplitude_error=float(_scalar(jnp.max(observed))),
        maximum_relative_amplitude_mse_bound=float(_scalar(jnp.max(bound))),
        maximum_relative_amplitude_increment_mse_bound=float(_scalar(jnp.max(sharp_bound))),
        covariance_bound_check_passed=bool(_scalar(bound_check)),
        declared_increment_amplitude_tolerance=epsilon,
        resolved_interval_pair_count=int(_scalar(selected_count)),
        resolved_interval_max_relative_s2_error=(float(_scalar(max_on(jnp.abs(ratio-1)))) if int(_scalar(selected_count)) else None),
        resolved_interval_bound_met=(bool(_scalar(jnp.all(jnp.where(selected,bound<=epsilon,True)))) if int(_scalar(selected_count)) else None),
        resolved_interval_maximum_slope_error_against_reference=(float(_scalar(jnp.max(
            jnp.where(selected_slopes,jnp.abs(reconstructed_slopes-reference_slopes),-jnp.inf))))
            if int(_scalar(jnp.sum(selected_slopes))) else None),
        resolved_interval_maximum_slope_deviation_from_two_thirds=(float(_scalar(jnp.max(
            jnp.where(selected_slopes,jnp.abs(reconstructed_slopes-2/3),-jnp.inf))))
            if int(_scalar(jnp.sum(selected_slopes))) else None),
        manufactured_relative_velocity_error=float(_scalar(jnp.sqrt(
            (jnp.sum((py-ty)**2)+jnp.sum((pz-tz)**2))/(jnp.sum(ty**2)+jnp.sum(tz**2))))),
        manufactured_draw_count=py.shape[1],
        ensemble_band_definition="Mean +/- two standard errors across independent manufactured draws; descriptive finite-draw band, not a rigorous confidence interval",
    )
    arrays=dict(reference_s2=mean_ref,reconstructed_s2=mean_rec,
        reference_pair_mean_slopes=reference_slopes,reconstructed_slopes=reconstructed_slopes,
        pair_reference_s2=reference,pair_reconstructed_s2=reconstructed,
        pair_pointwise_mse_y=ey,pair_pointwise_mse_z=ez,pair_amplitude_error_bound=bound,
        pair_increment_mse=increment_mse,pair_sharp_amplitude_error_bound=sharp_bound,
        empirical_reference_mean=jnp.mean(truth_draw_bins,axis=1),
        empirical_reference_se=se(truth_draw_bins),
        empirical_reconstructed_mean=jnp.mean(pred_draw_bins,axis=1),
        empirical_reconstructed_se=se(pred_draw_bins),
        manufactured_prediction_y=py,manufactured_prediction_z=pz,
        manufactured_truth_y=ty,manufactured_truth_z=tz)
    return metrics,arrays


def run_irregular(args, prior, blocks, device, radius, directions, y, z, reference, provenance):
    results=[]
    queries=jnp.concatenate([y,z])
    for spacing in args.spacings:
        _check_sources(provenance["source_sha256"])
        cloud_started=time.perf_counter()
        try:
            cloud=make_irregular_cloud(3,spacing,seed=args.cloud_seed)
        except BaseException as error:
            _preserve_cloud_failure(args.output,error,source_sha256=provenance["source_sha256"],
                protocol_sha256=provenance["protocol_sha256"],specification=dict(
                    suite="increment_statistics_irregular",dimension=3,spacing=spacing,seed=args.cloud_seed))
            raise
        points=cloud.points
        try:
            fields=sample_fields(points,blocks,point_batch=args.field_batch)
            jax.block_until_ready(fields)
            shared_seconds=time.perf_counter()-cloud_started
            inputs=_input_provenance(args,f"irregular3d_h{spacing:g}_inputs.npz",
                dict(nodes=points,queries=queries,nodal_values=fields,boundary=cloud.boundary,
                     query_y=y,query_z=z,**(cloud.geometry_evidence or {})))
        except BaseException as error:
            _preserve_cloud_failure(args.output,error,source_sha256=provenance["source_sha256"],
                protocol_sha256=provenance["protocol_sha256"],specification=dict(
                    suite="increment_statistics_irregular",dimension=3,spacing=spacing,seed=args.cloud_seed),
                additional_arrays=dict(nodes=points,queries=queries,**(cloud.geometry_evidence or {})),
                stage_metadata=dict(stage="manufactured_prior_samples_or_input_serialization",cloud=cloud.metadata))
            raise
        _check_sources(provenance["source_sha256"])
        for order in args.orders:
            for policy in STATISTICAL_POLICIES:
                name=f"irregular3d_h{spacing:g}_p{order}_{policy}"
                folder=args.output/name
                folder.mkdir(parents=True,exist_ok=False)
                started=time.perf_counter()
                mapping=None
                report=None
                try:
                    _check_sources(provenance["source_sha256"])
                    options={} if policy!="fixed_physical_band" else {
                        "matern_base_length":prior.physical_matern_base_length}
                    kernel="phs" if policy=="phs" else "matern"
                    mapping=build_map(points,queries,degree=order-1,kernel=kernel,
                        assignment="nearest-node",batch_size=args.map_batch,
                        stencil_domain=cloud.stencil_domain,
                        allow_pseudoinverse=False,compute_gradients=False,
                        matern_base_length_factor=16.,matern_num_modes=5,matern_gamma=8/3,**options)
                    jax.block_until_ready(mapping.weights)
                    setup_seconds=time.perf_counter()-started
                    diagnostics=summarize_map(mapping)
                    _write_json(folder/"spatial_checks.json",diagnostics)
                    diagnostics_sha=_export_npz(folder/"spatial_diagnostics.npz",center_indices=mapping.center_indices,
                        stencil_indices=mapping.stencil_indices,weights=mapping.weights,
                        unreachable_euclidean_stencil_fallback_rows=jnp.asarray(
                            diagnostics["unreachable_euclidean_stencil_fallback_rows"],dtype=jnp.int64),**mapping.diagnostics)
                    if not diagnostics["numerical_checks_passed"]:
                        raise RuntimeError("Original-system or spatial reproduction checks failed; preserved spatial_checks.json")
                    statistic_started=time.perf_counter()
                    energies,py,pz,ty,tz=pair_statistics(points,mapping,y,z,blocks,pair_batch=args.pair_batch)
                    predicted=apply_map(mapping,jnp.transpose(fields,(0,2,1)))
                    predicted=jnp.transpose(predicted,(0,2,1))
                    feature_prediction=jnp.concatenate([py,pz])
                    gap=jnp.linalg.norm(predicted-feature_prediction)/jnp.maximum(jnp.linalg.norm(predicted),jnp.finfo(points.dtype).tiny)
                    target_gap=jnp.max(jnp.abs(energies[:,1]-exact_increment(prior.wavevectors,prior.weights,z-y)))
                    jax.block_until_ready((energies,predicted,gap))
                    statistic_seconds=time.perf_counter()-statistic_started
                    value_timings=_warm_value_timing(mapping,fields,args.repetitions)
                    metrics,arrays=_report_metrics(energies,py,pz,ty,tz,bins=len(radius),directions=len(directions),radius=radius,
                        interval=reference["interval_mask"],epsilon=args.amplitude_tolerance)
                    report=dict(case=name,dimension=3,geometry="irregular_F3_domain",
                        spacing=spacing,nodes=len(points),value_order=order,polynomial_degree=order-1,
                        stencil_size=mapping.stencil_indices.shape[1],policy=policy,
                        assignment="nearest-node",kernel=("divergence-free PHS + polynomial"
                            if policy=="phs" else "five-scale divergence-free Matern + polynomial"),
                        fixed_physical_matern_base_length=(float(_scalar(prior.physical_matern_base_length))
                            if policy=="fixed_physical_band" else None),
                        shared_cloud_and_manufactured_field_seconds=shared_seconds,
                        cloud=cloud.metadata,
                        setup_compile_seconds=setup_seconds,statistics_compile_seconds=statistic_seconds,
                        timings=value_timings,
                        scalar_prior_s2_identity_max_gap=float(_scalar(target_gap)),
                        manufactured_feature_vs_nodal_route_relative_gap=float(_scalar(gap)),
                        reference_has_certified_two_thirds_interval=bool(_scalar(reference["resolved"])),
                        numerical_checks=diagnostics,**metrics,**_cell_pins(provenance,inputs))
                    report["map_metadata"] = mapping.metadata
                    statistics_sha=_export_npz(folder/"statistics.npz",radius=radius,query_y=y,query_z=z,**arrays)
                    report.update(statistics_npz_sha256=statistics_sha,
                                  spatial_diagnostics_npz_sha256=diagnostics_sha)
                    if not (bool(_scalar(gap<=1e-9)) and bool(_scalar(target_gap<=1e-11))
                            and report["covariance_bound_check_passed"]):
                        raise RuntimeError("Covariance/feature/nodal identity failed; numerical result preserved")
                    _check_sources(provenance["source_sha256"])
                    _write_json(folder/"result.json",report)
                    results.append(report)
                    _write_json(args.output/"progress.json",dict(protocol_sha256=provenance["protocol_sha256"],
                        completed_cells=len(results),results=results),immutable=False)
                    print(json.dumps(dict(completed=name,error=report["manufactured_relative_velocity_error"],
                        resolved_s2_error=report["resolved_interval_max_relative_s2_error"],seconds=time.perf_counter()-started)),flush=True)
                except Exception as error:
                    _preserve_failure(folder,error,provenance=provenance,inputs=inputs,nodes=points,
                        queries=queries,nodal_values=fields,mapping=mapping,
                        extra_arrays=dict(query_y=y,query_z=z,boundary=cloud.boundary),candidate_report=report)
                    raise
    return results


def _periodic_group_checks(predicted, jacobian, truth, truth_jacobian, groups):
    """Actual boundary identifications and wrapped probes, for every manufactured draw."""
    scales=(jnp.maximum(jnp.max(jnp.linalg.norm(predicted,axis=-1)),1.),
            jnp.maximum(jnp.max(jnp.linalg.norm(jacobian,axis=(1,2))),1.),
            jnp.maximum(jnp.max(jnp.linalg.norm(truth,axis=-1)),1.),
            jnp.maximum(jnp.max(jnp.linalg.norm(truth_jacobian,axis=(1,2))),1.))
    result={}
    for kind in ("face","edge","corner","wrapped"):
        values,gradients,true_values,true_gradients=[],[],[],[]
        pairs=0
        for group in groups:
            if group["kind"]!=kind or group["start"]==group["reference_start"]:
                continue
            reference=slice(group["reference_start"],group["reference_stop"])
            shifted=slice(group["start"],group["stop"])
            values.append(jnp.linalg.norm(predicted[shifted]-predicted[reference],axis=-1).reshape(-1))
            gradients.append(jnp.linalg.norm(jacobian[shifted]-jacobian[reference],axis=(1,2)).reshape(-1))
            true_values.append(jnp.linalg.norm(truth[shifted]-truth[reference],axis=-1).reshape(-1))
            true_gradients.append(jnp.linalg.norm(truth_jacobian[shifted]-truth_jacobian[reference],axis=(1,2)).reshape(-1))
            pairs+=group["stop"]-group["start"]
        if not values:
            raise RuntimeError(f"Missing actual periodic identification probes: {kind}")
        gaps=[jnp.max(jnp.concatenate(parts))/scale for parts,scale in
              zip((values,gradients,true_values,true_gradients),scales)]
        finite=jnp.all(jnp.isfinite(jnp.stack(gaps)))
        passed=finite&(gaps[0]<=1e-8)&(gaps[1]<=1e-8)&(gaps[2]<=1e-10)&(gaps[3]<=1e-10)
        result[kind]=dict(pairs=pairs,reconstructed_value_relative_gap=float(_scalar(gaps[0])),
            reconstructed_jacobian_relative_gap=float(_scalar(gaps[1])),
            exact_value_relative_gap=float(_scalar(gaps[2])),exact_jacobian_relative_gap=float(_scalar(gaps[3])),
            passed=bool(_scalar(passed)))
    return result


def periodic_diagnostics(args, prior, blocks, device, provenance):
    _check_sources(provenance["source_sha256"])
    try:
        cloud=make_periodic_cloud(3,args.periodic_n,seed=args.cloud_seed)
    except BaseException as error:
        _preserve_cloud_failure(args.output,error,source_sha256=provenance["source_sha256"],
            protocol_sha256=provenance["protocol_sha256"],specification=dict(
                suite="increment_statistics_periodic",dimension=3,n_per_axis=args.periodic_n,seed=args.cloud_seed))
        raise
    key=jax.random.key(4527)
    interior=jax.random.uniform(key,(24,3),minval=0.,maxval=2*jnp.pi,dtype=jnp.float64)
    # Retain the original joint translations as an additional periodicity check.
    offsets=jax.device_put(jnp.asarray(list(itertools.product((0.,2*math.pi),repeat=3)),dtype=jnp.float64),device)
    anchor=jax.device_put(jnp.asarray([[.037,.043,.029]],dtype=jnp.float64),device)
    queries=jnp.concatenate([interior,anchor])
    translated=(queries[None]+offsets[:,None]).reshape(-1,3)
    legacy_queries=jnp.concatenate([queries,translated])
    boundary_queries,groups=periodic_probes(3,count=4,wrapped_count=4)
    groups=[{key:(value+len(legacy_queries) if key in
        ("start","stop","reference_start","reference_stop") else value)
        for key,value in group.items()} for group in groups]
    all_queries=jnp.concatenate([legacy_queries,boundary_queries])
    fields=sample_fields(cloud.points,blocks,point_batch=args.field_batch)
    truth=sample_fields(all_queries,blocks,point_batch=args.field_batch)
    truth_jacobian=sample_field_jacobians(all_queries,blocks,point_batch=args.field_batch)
    inputs=_input_provenance(args,"periodic3d_inputs.npz",dict(nodes=cloud.points,
        queries=all_queries,nodal_values=fields,true_values=truth,true_jacobian=truth_jacobian,
        base_queries=queries,offsets=offsets,boundary_queries=boundary_queries))
    group_path=args.output/"periodic3d_probe_groups.json"
    groups_sha=_write_json(group_path,dict(periodic_probe_groups=groups,
        legacy_translation_query_count=len(legacy_queries),actual_probe_query_count=len(boundary_queries)))
    inputs.update(periodic_probe_groups_artifact=group_path.relative_to(args.output).as_posix(),
                  periodic_probe_groups_sha256=groups_sha)
    _check_sources(provenance["source_sha256"])
    results=[]
    for order,policy in itertools.product(args.orders,STATISTICAL_POLICIES):
        name=f"periodic3d_p{order}_{policy}"
        folder=args.output/name
        folder.mkdir(parents=True,exist_ok=False)
        mapping=None
        row=None
        try:
            _check_sources(provenance["source_sha256"])
            options={} if policy!="fixed_physical_band" else {
                "matern_base_length":prior.physical_matern_base_length}
            kernel="phs" if policy=="phs" else "matern"
            mapping=build_map(cloud.points,all_queries,degree=order-1,kernel=kernel,
                assignment="nearest-node",periodic_bounds=((0.,0.,0.),(2*math.pi,)*3),
                batch_size=args.map_batch,allow_pseudoinverse=False,compute_gradients=True,**options)
            predicted=jnp.transpose(apply_map(mapping,jnp.transpose(fields,(0,2,1))),(0,2,1))
            jacobian=apply_map_jacobian(mapping,jnp.transpose(fields,(0,2,1)))
            divergence=jnp.trace(jacobian,axis1=1,axis2=2)
            original=predicted[:len(queries)]
            shifted=predicted[len(queries):len(legacy_queries)].reshape(len(offsets),len(queries),args.draws,3)
            true_original=truth[:len(queries)]
            true_shifted=truth[len(queries):len(legacy_queries)].reshape(shifted.shape)
            scale=jnp.maximum(jnp.max(jnp.linalg.norm(original,axis=-1)),1.)
            periodic_gap=jnp.max(jnp.linalg.norm(shifted-original[None],axis=-1))/scale
            true_gap=jnp.max(jnp.linalg.norm(true_shifted-true_original[None],axis=-1))/jnp.maximum(
                jnp.max(jnp.linalg.norm(truth[:len(legacy_queries)],axis=-1)),1.)
            original_jacobian=jacobian[:len(queries)]
            shifted_jacobian=jacobian[len(queries):len(legacy_queries)].reshape(
                len(offsets),len(queries),3,3,args.draws)
            true_original_jacobian=truth_jacobian[:len(queries)]
            true_shifted_jacobian=truth_jacobian[len(queries):len(legacy_queries)].reshape(shifted_jacobian.shape)
            jacobian_gap=jnp.max(jnp.linalg.norm(shifted_jacobian-original_jacobian[None],axis=(2,3)))/jnp.maximum(
                jnp.max(jnp.linalg.norm(original_jacobian,axis=(1,2))),1.)
            true_jacobian_gap=jnp.max(jnp.linalg.norm(true_shifted_jacobian-true_original_jacobian[None],axis=(2,3)))/jnp.maximum(
                jnp.max(jnp.linalg.norm(truth_jacobian[:len(legacy_queries)],axis=(1,2))),1.)
            div_scale=jnp.maximum(jnp.max(jnp.abs(jacobian)),1.)
            div_gap=jnp.max(jnp.abs(divergence))/div_scale
            group_checks=_periodic_group_checks(predicted,jacobian,truth,truth_jacobian,groups)
            value_timings=_warm_value_timing(mapping,fields,args.repetitions)
            diagnostics=summarize_map(mapping)
            diagnostics_sha=_export_npz(folder/"spatial_diagnostics.npz",center_indices=mapping.center_indices,
                stencil_indices=mapping.stencil_indices,weights=mapping.weights,
                unreachable_euclidean_stencil_fallback_rows=jnp.asarray(
                    diagnostics["unreachable_euclidean_stencil_fallback_rows"],dtype=jnp.int64),**mapping.diagnostics)
            statistics_sha=_export_npz(folder/"statistics.npz",prediction=predicted,
                jacobian=jacobian,divergence=divergence,truth=truth,truth_jacobian=truth_jacobian,
                queries=all_queries,base_queries=queries,offsets=offsets,boundary_queries=boundary_queries)
            row=dict(case=name,value_order=order,polynomial_degree=order-1,nodes=len(cloud.points),policy=policy,
                kernel=kernel,
                timings=value_timings,cloud=cloud.metadata,
                exact_manufactured_prior_periodicity_relative_gap=float(_scalar(true_gap)),
                exact_manufactured_prior_jacobian_periodicity_relative_gap=float(_scalar(true_jacobian_gap)),
                reconstructed_joint_translation_periodicity_relative_gap=float(_scalar(periodic_gap)),
                reconstructed_joint_translation_jacobian_periodicity_relative_gap=float(_scalar(jacobian_gap)),
                reconstructed_face_edge_corner_periodicity_relative_gap=max(
                    group_checks[kind]["reconstructed_value_relative_gap"] for kind in ("face","edge","corner")),
                reconstructed_face_edge_corner_jacobian_periodicity_relative_gap=max(
                    group_checks[kind]["reconstructed_jacobian_relative_gap"] for kind in ("face","edge","corner")),
                periodic_group_checks=group_checks,periodic_probe_groups=groups,
                legacy_translation_query_count=len(legacy_queries),actual_probe_query_count=len(boundary_queries),
                local_normalized_divergence_gap=float(_scalar(div_gap)),spatial_checks=diagnostics,
                passed=diagnostics["numerical_checks_passed"] and all(group["passed"] for group in group_checks.values())
                    and bool(_scalar((true_gap<=1e-10)&(true_jacobian_gap<=1e-10)
                        &(periodic_gap<=1e-8)&(jacobian_gap<=1e-8)&(div_gap<=1e-8))),
                statistics_npz_sha256=statistics_sha,spatial_diagnostics_npz_sha256=diagnostics_sha,
                map_metadata=mapping.metadata,
                **_cell_pins(provenance,inputs))
            if not row["passed"]:
                raise RuntimeError("Periodic face/edge/corner or fixed-local divergence check failed; record preserved")
            _check_sources(provenance["source_sha256"])
            _write_json(folder/"result.json",row)
            results.append(row)
            _write_json(args.output/"periodicity_progress.json",dict(protocol_sha256=provenance["protocol_sha256"],
                results=results),immutable=False)
        except Exception as error:
            _preserve_failure(folder,error,provenance=provenance,inputs=inputs,nodes=cloud.points,
                queries=all_queries,nodal_values=fields,mapping=mapping,
                extra_arrays=dict(true_values=truth,true_jacobian=truth_jacobian,
                    base_queries=queries,offsets=offsets,boundary_queries=boundary_queries),candidate_report=row)
            raise
    _write_json(args.output/"periodicity.json",dict(protocol_sha256=provenance["protocol_sha256"],results=results))
    return results


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--orders",type=int,nargs="+",default=[2,4,6])
    parser.add_argument("--spacings",type=float,nargs="+",default=[.26,.19,.14,.101])
    parser.add_argument("--maximum-wavenumber",type=int,default=1024)
    parser.add_argument("--draws",type=int,default=8)
    parser.add_argument("--bins",type=int,default=32)
    parser.add_argument("--directions",type=int,default=8)
    parser.add_argument("--pair-batch",type=int,default=4)
    parser.add_argument("--mode-batch",type=int,default=64)
    parser.add_argument("--map-batch",type=int,default=32)
    parser.add_argument("--repetitions",type=int,default=11)
    parser.add_argument("--field-batch",type=int,default=256)
    parser.add_argument("--periodic-n",type=int,default=12)
    parser.add_argument("--cloud-seed",type=int,default=1729)
    parser.add_argument("--amplitude-tolerance",type=float,default=.1)
    parser.add_argument("--reference-only",action="store_true",
                        help="Only certify/save the fixed analytic prior; no clouds or local maps")
    args=parser.parse_args()
    if (any(p not in (2,4,6) for p in args.orders) or len(set(args.orders))!=len(args.orders)
            or any(h<=0 for h in args.spacings) or args.maximum_wavenumber<8 or args.draws<2
            or args.bins<4 or args.directions<3 or any(v<1 for v in
                (args.pair_batch,args.mode_batch,args.map_batch,args.field_batch,args.periodic_n,args.repetitions))
            or not 0<args.amplitude_tolerance<1):
        parser.error("Invalid full-study protocol parameters")
    args.output.mkdir(parents=True,exist_ok=False)
    module_origins=_frozen_module_origins()
    source_sha256=_sources()
    device=configure_gpu()
    if device.platform!="gpu" or not jax.config.x64_enabled:
        raise RuntimeError("CUDA float64 required; no CPU numerical fallback")
    prior=make_prior(device,maximum_wavenumber=args.maximum_wavenumber,draws=args.draws)
    blocks=_prior_blocks(prior,args.mode_batch)
    radius,directions,y,z=make_pairs(device,bins=args.bins,directions=args.directions)
    reference=certify_reference(prior,radius,directions)
    transverse_gap=jnp.max(jnp.abs(jnp.einsum("md,mdr->mr",prior.wavevectors,prior.frames))
        /jnp.linalg.norm(prior.wavevectors,axis=1)[:,None])
    frame_gap=jnp.max(jnp.abs(jnp.einsum("mdr,mds->mrs",prior.frames,prior.frames)-jnp.eye(2,dtype=radius.dtype)))
    variance_gap=jnp.abs(2*jnp.sum(prior.weights)-1)
    prior_checks=(jnp.all(prior.weights>=0)&jnp.all(jnp.isfinite(prior.weights))
                  &(transverse_gap<=1e-12)&(frame_gap<=1e-12)&(variance_gap<=1e-12))
    reference_arrays=dict(wavevectors=prior.wavevectors,weights=prior.weights,
        frames=prior.frames,cosine_draws=prior.cosine_draws,sine_draws=prior.sine_draws,
        physical_matern_base_length=prior.physical_matern_base_length,
        radius=radius,directions=directions,query_y=y,query_z=z,
        reference_s2=reference["mean"],reference_directional_s2=reference["directional"],
        reference_slopes=reference["slopes"],pair_reference_slopes=reference["pair_slopes"],
        reference_angular_relative_gap=reference["angular_relative_gap"],
        certified_interval=reference["interval_mask"])
    reference_array_sha256={key:_digest_array(value) for key,value in reference_arrays.items()}
    reference_npz_sha256=_export_npz(args.output/"reference.npz",**reference_arrays)
    _check_sources(source_sha256)
    protocol={key:(str(value) if isinstance(value,Path) else value) for key,value in vars(args).items()}
    from fastol_scattered_clouds import GEOMETRY_POLICY
    protocol.update(
        schema="fastol_spatial_statistics_scattered_v4",source_sha256=source_sha256,
        cloud_and_stencil_policy=GEOMETRY_POLICY,
        module_origins=module_origins,
        statistical_policies=STATISTICAL_POLICIES,
        reference_npz_sha256=reference_npz_sha256,reference_array_sha256=reference_array_sha256,
        array_hash_definition="SHA256 of exported raw C-order bytes; shapes and dtypes stored in the pinned NPZ",
        device=str(device),dtype="float64",prior="finite periodic solenoidal Fourier covariance",
        prior_proof="K=sum nonnegative w B B^T cos(k.(x-y)); integer k and k^T B=0 prove PSD, periodicity and divergence freedom",
        spectral_target=f"primitive integer directions with logarithmic integer harmonics to kmax{args.maximum_wavenumber}; positive exact integrals of k^(-5/3) over geometric radial shell cells; finite angular/radial quadrature, not exact continuous isotropy",
        prior_padded_modes=len(prior.weights),prior_active_modes=int(_scalar(jnp.sum(prior.weights>0))),
        reference_two_thirds_interval_resolved=bool(_scalar(reference["resolved"])),
        reference_interval_decades=float(_scalar(reference["decades"])),
        reference_interval_start=float(_scalar(radius[reference["start"]])),
        reference_interval_end=float(_scalar(radius[reference["end"]])),
        reference_slope_tolerance=.10,reference_minimum_interval_decades=math.log10(4),
        reference_angular_quadrature_counts=[64,128],reference_angular_relative_tolerance=.02,
        reference_max_angular_relative_gap=float(_scalar(jnp.max(reference["angular_relative_gap"]))),
        reference_small_separation_slopes=[float(_scalar(v)) for v in reference["slopes"][:3]],
        reference_quadratic_small_separation_check=bool(_scalar(jnp.all(jnp.abs(reference["slopes"][:2]-2)<.05))),
        prior_numeric_checks_passed=bool(_scalar(prior_checks)),
        prior_transversality_relative_gap=float(_scalar(transverse_gap)),
        prior_frame_orthonormality_gap=float(_scalar(frame_gap)),
        prior_unit_velocity_variance_gap=float(_scalar(variance_gap)),
        finite_band_qualification="Finite smooth modes imply S2=O(r^2) near0; only the numerically certified finite intermediate interval may be called two-thirds-like",
        baseline="Shared boundary-aware node stencils (periodic minimum images on torus), five-scale Matern, Ram local tau16 scale convention",
        variant="Same assigned node stencils and polynomials; one frozen physical five-mode band, no query-dependent basis or adaptive stencil claim",
        phs_control="Same assigned node stencils and polynomials, divergence-free PHS scalar exponent3/3/5 for orders2/4/6; no scale parameter",
        numerical_acceptance="All diagnostics finite, full requested polynomial rank retained, no pseudoinverse fallback, backward error <=1e-10, interpolation and retained/requested polynomial errors <=1e-8",
        periodic_probe_protocol=dict(dimension=3,count=4,wrapped_count=4,seed=32452843,
            scope="Actual face/edge/corner identifications and signed/joint multi-period wraps, in addition to the original positive joint translations",
            comparisons="Values and analytic Jacobians of every manufactured draw; local divergence away from stencil interfaces",
            group_normalization="All-query maximum vector norm or Frobenius Jacobian norm, each floored at1, separately for reconstruction and truth",
            exact_tolerance=1e-10,reconstructed_tolerance=1e-8),
        fixed_physical_matern_base_length=float(_scalar(prior.physical_matern_base_length)),
        expected_statistical_cells=len(args.spacings)*len(args.orders)*len(STATISTICAL_POLICIES),
        expected_periodic_statistical_cells=len(args.orders)*len(STATISTICAL_POLICIES),
        statistical_pair_count=len(y),statistical_query_map_count=2*len(y))
    protocol_sha256=hashlib.sha256(_canonical(protocol)).hexdigest()
    _write_json(args.output/"protocol.json",protocol)
    provenance=dict(protocol_sha256=protocol_sha256,source_sha256=source_sha256,
        reference_npz_sha256=reference_npz_sha256)
    if not protocol["prior_numeric_checks_passed"]:
        raise RuntimeError("Prior PSD-factor/transversality/normalization checks failed; protocol preserved")
    print(json.dumps(dict(reference_ready=True,two_thirds_interval=protocol["reference_two_thirds_interval_resolved"],
        declared_cells=protocol["expected_statistical_cells"])),flush=True)
    if args.reference_only:
        _write_json(args.output/"reference_completion.json",dict(reference_only=True,
            protocol_sha256=protocol_sha256,reference_npz_sha256=reference_npz_sha256,
            reference_kolmogorov_interval_resolved=protocol["reference_two_thirds_interval_resolved"],
            qualification="Analytic-prior certification only; no spatial property result"))
        return
    try:
        rows=run_irregular(args,prior,blocks,device,radius,directions,y,z,reference,provenance)
        periodic=periodic_diagnostics(args,prior,blocks,device,provenance)
        _check_sources(source_sha256)
        result_sha256={path.relative_to(args.output).as_posix():_digest_file(path)
            for path in sorted(args.output.glob("*/result.json"))}
        _write_json(args.output/"completion.json",dict(completed=True,statistical_cells=len(rows),
            protocol_sha256=protocol_sha256,reference_npz_sha256=reference_npz_sha256,
            result_sha256=result_sha256,
            reference_kolmogorov_interval_resolved=protocol["reference_two_thirds_interval_resolved"],
            periodicity_passed=all(row["passed"] for row in periodic),results=rows,
            qualification="Completion means all declared calculations/checks completed; it does not mean every model preserves increments within tolerance"))
    except Exception as error:
        _write_json(args.output/"failure.json",dict(error_type=type(error).__name__,error=str(error),
            traceback=traceback.format_exc(),protocol_sha256=protocol_sha256,
            reference_npz_sha256=reference_npz_sha256))
        raise


if __name__=="__main__":
    main()
