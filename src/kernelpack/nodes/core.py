from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Callable
import os

import numpy as np
from jax import jit, lax, vmap
import jax.numpy as jnp

from kernelpack.accelerators import (
    WarpBridsonOptions,
    WarpUnavailableError,
    warp_available,
    warp_bridson_sample,
)
from kernelpack.domain import DomainDescriptor
from kernelpack.geometry import EmbeddedSurface, PiecewiseSmoothEmbeddedSurface, RBFLevelSet, distance_matrix


@jit
def _nearest_boundary_distance_impl(points: jnp.ndarray, boundary_points: jnp.ndarray) -> jnp.ndarray:
    return jnp.min(distance_matrix(points, boundary_points), axis=1)


def _nearest_boundary_distance(points: jnp.ndarray | np.ndarray, boundary_points: jnp.ndarray | np.ndarray) -> jnp.ndarray:
    pts = jnp.asarray(points, dtype=float)
    boundary = jnp.asarray(boundary_points, dtype=float)
    if boundary.shape[0] == 0:
        return jnp.full(pts.shape[0], jnp.inf, dtype=float)
    return _nearest_boundary_distance_impl(pts, boundary)


@jit
def _clip_mask_from_phi(
    phi: jnp.ndarray,
    keep_inside: bool,
    tolerance: float,
    boundary_clearance: float,
    min_signed_distance: float,
    max_signed_distance: float,
) -> jnp.ndarray:
    primary = jnp.where(
        keep_inside,
        phi >= (boundary_clearance - tolerance),
        phi <= (tolerance - boundary_clearance),
    )
    return primary & (phi >= min_signed_distance) & (phi <= max_signed_distance)


@partial(jit, static_argnums=(1,))
def _radical_inverse_array(indices: jnp.ndarray, base: int) -> jnp.ndarray:
    indices = jnp.asarray(indices, dtype=jnp.int32)
    values = jnp.zeros(indices.shape[0], dtype=float)
    denom = jnp.ones(indices.shape[0], dtype=float)

    def cond_fun(state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        current, _, _ = state
        return jnp.any(current > 0)

    def body_fun(state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        current, value, denom = state
        current, remainder = jnp.divmod(current, base)
        denom = denom * base
        value = value + remainder.astype(float) / denom
        return current, value, denom

    _, values, _ = lax.while_loop(cond_fun, body_fun, (indices, values, denom))
    return values


def _fps_initial_state(candidates: jnp.ndarray, selected_seed: jnp.ndarray | None) -> tuple[jnp.ndarray, jnp.ndarray]:
    if selected_seed is None or selected_seed.size == 0:
        center = candidates.mean(axis=0, keepdims=True)
        min_dists = jnp.linalg.norm(candidates - center, axis=1)
        selected = jnp.zeros((0, candidates.shape[1]), dtype=float)
    else:
        selected = jnp.asarray(selected_seed, dtype=float).reshape(-1, candidates.shape[1])
        min_dists = jnp.min(distance_matrix(candidates, selected), axis=1)
    return min_dists, selected


@jit
def _greedy_fps_constant_radius_impl(
    candidates: jnp.ndarray,
    radius: float,
    min_dists: jnp.ndarray,
    has_seed: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    count = candidates.shape[0]
    tol = 1e-12
    active0 = jnp.ones(count, dtype=bool)
    picked0 = -jnp.ones(count, dtype=jnp.int32)

    def cond_fun(state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        active, current_min_dists, _ = state
        score = jnp.where(active, current_min_dists / radius, -jnp.inf)
        return jnp.any(active) & (jnp.max(score) >= 1.0 - tol)

    def body_fun(state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        active, current_min_dists, picked = state
        score = jnp.where(active, current_min_dists / radius, -jnp.inf)
        next_idx = jnp.asarray(jnp.argmax(score), dtype=jnp.int32)
        dist_to_new = jnp.linalg.norm(candidates - candidates[next_idx], axis=1)
        new_active = active & (dist_to_new >= radius * (1.0 - tol))
        new_active = new_active.at[next_idx].set(False)
        next_slot = jnp.sum(picked >= 0, dtype=jnp.int32)
        picked = picked.at[next_slot].set(next_idx)
        return new_active, jnp.minimum(current_min_dists, dist_to_new), picked

    _, final_min_dists, picked = lax.while_loop(cond_fun, body_fun, (active0, min_dists, picked0))
    pick_count = jnp.sum(picked >= 0, dtype=jnp.int32)
    fallback_idx = jnp.asarray(jnp.argmax(final_min_dists), dtype=jnp.int32)
    picked = lax.cond(
        (pick_count == 0) & (~has_seed),
        lambda arr: arr.at[0].set(fallback_idx),
        lambda arr: arr,
        picked,
    )
    pick_count = jnp.maximum(pick_count, jnp.where((pick_count == 0) & (~has_seed), 1, 0))
    return picked, pick_count


@jit
def _greedy_fps_variable_radius_impl(
    candidates: jnp.ndarray,
    radii: jnp.ndarray,
    min_radius: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    count = candidates.shape[0]
    tol = 1e-12
    center = candidates.mean(axis=0, keepdims=True)
    min_dists0 = jnp.linalg.norm(candidates - center, axis=1)
    active0 = jnp.ones(count, dtype=bool)
    picked0 = -jnp.ones(count, dtype=jnp.int32)

    def cond_fun(state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        active, current_min_dists, _ = state
        score = jnp.where(active, current_min_dists / jnp.maximum(radii, min_radius), -jnp.inf)
        return jnp.any(active) & (jnp.max(score) >= 1.0 - tol)

    def body_fun(state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        active, current_min_dists, picked = state
        score = jnp.where(active, current_min_dists / jnp.maximum(radii, min_radius), -jnp.inf)
        next_idx = jnp.asarray(jnp.argmax(score), dtype=jnp.int32)
        dist_to_new = jnp.linalg.norm(candidates - candidates[next_idx], axis=1)
        new_active = active & (dist_to_new >= radii * (1.0 - tol))
        new_active = new_active.at[next_idx].set(False)
        next_slot = jnp.sum(picked >= 0, dtype=jnp.int32)
        picked = picked.at[next_slot].set(next_idx)
        return new_active, jnp.minimum(current_min_dists, dist_to_new), picked

    _, final_min_dists, picked = lax.while_loop(cond_fun, body_fun, (active0, min_dists0, picked0))
    pick_count = jnp.sum(picked >= 0, dtype=jnp.int32)
    fallback_idx = jnp.asarray(jnp.argmax(final_min_dists), dtype=jnp.int32)
    picked = lax.cond(
        pick_count == 0,
        lambda arr: arr.at[0].set(fallback_idx),
        lambda arr: arr,
        picked,
    )
    pick_count = jnp.maximum(pick_count, 1)
    return picked, pick_count


def generate_poisson_nodes_in_box(
    radius_or_func: float | Callable[[jnp.ndarray, float], float],
    x_min: jnp.ndarray,
    x_max: jnp.ndarray,
    *,
    attempts: int = 30,
    seed: int | None = None,
    deterministic: bool | None = None,
    strip_count: int | None = None,
    use_parallel: bool = True,
    min_radius: float | None = None,
    boundary_points: jnp.ndarray | None = None,
    boundary_refinement_fraction: float = 1.0,
    boundary_distance: float = 0.0,
    backend: str = "auto",
) -> tuple[jnp.ndarray, dict[str, object]]:
    x_min_np = np.asarray(x_min, dtype=float).reshape(-1)
    x_max_np = np.asarray(x_max, dtype=float).reshape(-1)
    dim = x_min_np.size
    opts = _parse_poisson_options(
        radius_or_func,
        x_min_np,
        x_max_np,
        attempts=attempts,
        seed=seed,
        deterministic=deterministic,
        strip_count=strip_count,
        use_parallel=use_parallel,
        min_radius=min_radius,
        boundary_points=boundary_points,
        boundary_refinement_fraction=boundary_refinement_fraction,
        boundary_distance=boundary_distance,
    )
    selected_backend = _resolve_poisson_backend(backend, opts)
    if np.any(x_max_np <= x_min_np):
        info = _empty_poisson_info(opts, dim)
        info["backend"] = selected_backend
        return jnp.zeros((0, dim), dtype=float), info

    boxes = _build_strip_boxes(x_min_np, x_max_np, float(opts["split_tol"]), int(opts["strip_count"]))
    clouds = []
    for k, box in enumerate(boxes):
        strip_seed = int((int(opts["base_seed"]) + 104729 * k) % (2**32 - 1))
        if selected_backend == "warp":
            try:
                warp_opts = WarpBridsonOptions(
                    radius=float(opts["radius"]),
                    min_radius=float(opts["min_radius"]),
                    attempts=int(opts["attempts"]),
                    seed=strip_seed,
                    mode=str(opts["mode"]),
                    boundary_refinement_fraction=float(opts["boundary_refinement_fraction"]),
                    boundary_distance=float(opts["boundary_distance"]),
                    boundary_points=np.asarray(opts["boundary_points"], dtype=float),
                    radius_function=opts["rad_func"],
                )
                clouds.append(
                    warp_bridson_sample(
                        box["sample_min"],
                        box["sample_max"],
                        warp_opts,
                    )
                )
                continue
            except WarpUnavailableError:
                if backend != "auto":
                    raise
                selected_backend = "cpu"
        clouds.append(
            _poisson_strip_sample(
                box["sample_min"],
                box["sample_max"],
                opts,
                strip_seed,
            )
        )
    points = _flatten_strip_clouds(clouds, x_min_np, x_max_np)
    info = {
        "dimension": dim,
        "mode": opts["mode"],
        "radius": opts["radius"],
        "min_radius": float(opts["min_radius"]),
        "attempts": int(opts["attempts"]),
        "seed": int(opts["seed"]),
        "deterministic": bool(opts["deterministic"]),
        "strip_count": int(opts["strip_count"]),
        "used_parallel": bool(opts["use_parallel"]),
        "boundary_refinement_fraction": float(opts["boundary_refinement_fraction"]),
        "boundary_distance": float(opts["boundary_distance"]),
        "num_points": int(points.shape[0]),
        "backend": selected_backend,
    }
    return jnp.asarray(points, dtype=float), info


def _parse_poisson_options(
    radius_or_func: float | Callable[[jnp.ndarray, float], float],
    x_min: np.ndarray,
    x_max: np.ndarray,
    *,
    attempts: int,
    seed: int | None,
    deterministic: bool | None,
    strip_count: int | None,
    use_parallel: bool,
    min_radius: float | None,
    boundary_points: jnp.ndarray | np.ndarray | None,
    boundary_refinement_fraction: float,
    boundary_distance: float,
) -> dict[str, object]:
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if x_min.size != x_max.size:
        raise ValueError("x_min and x_max must have the same length")
    if boundary_refinement_fraction <= 0 or boundary_refinement_fraction > 1:
        raise ValueError("boundary_refinement_fraction must be in (0, 1]")
    if boundary_distance < 0:
        raise ValueError("boundary_distance must be nonnegative")

    if boundary_points is None:
        boundary_points_array = np.zeros((0, x_min.size), dtype=float)
    else:
        boundary_points_array = np.asarray(boundary_points, dtype=float)
        if boundary_points_array.ndim != 2 or boundary_points_array.shape[1] != x_min.size:
            raise ValueError("boundary_points must have the same column count as the box dimension")

    had_explicit_seed = seed is not None
    if seed is None:
        seed = int(np.random.randint(1, 2**31))
    if deterministic is None:
        deterministic = had_explicit_seed
    else:
        deterministic = bool(deterministic)

    requested_strip_count = None if strip_count is None else max(1, int(np.floor(strip_count)))
    actual_strip_count = _default_strip_count(bool(use_parallel), deterministic, requested_strip_count)
    used_parallel = bool(use_parallel) and actual_strip_count > 1
    has_boundary_refinement = (
        boundary_points_array.shape[0] > 0
        and boundary_refinement_fraction < 1.0
        and boundary_distance > 0.0
    )

    if callable(radius_or_func):
        if min_radius is None:
            raise ValueError("variable-density sampling requires min_radius")
        min_radius_value = float(min_radius)
        if min_radius_value <= 0:
            raise ValueError("min_radius must be positive")
        radius = float("nan")
        if has_boundary_refinement:
            mode = "variable_radius_with_boundary_refinement"
            grid_radius = boundary_refinement_fraction * min_radius_value
        else:
            mode = "variable_radius"
            grid_radius = min_radius_value
        rad_func = radius_or_func
    else:
        radius = float(radius_or_func)
        if radius <= 0:
            raise ValueError("radius must be positive")
        min_radius_value = radius if min_radius is None else float(min_radius)
        if has_boundary_refinement:
            mode = "fixed_radius_with_boundary_refinement"
            grid_radius = boundary_refinement_fraction * radius
        else:
            mode = "fixed_radius"
            grid_radius = radius
        rad_func = None

    return {
        "radius": radius,
        "rad_func": rad_func,
        "min_radius": float(min_radius_value),
        "attempts": int(attempts),
        "seed": int(seed),
        "base_seed": int(seed) % (2**32),
        "deterministic": deterministic,
        "strip_count": actual_strip_count,
        "use_parallel": used_parallel,
        "mode": mode,
        "boundary_points": boundary_points_array,
        "boundary_refinement_fraction": float(boundary_refinement_fraction),
        "boundary_distance": float(boundary_distance),
        "has_boundary_refinement": has_boundary_refinement,
        "grid_radius": float(grid_radius),
        "split_tol": float(min_radius_value),
    }


def _empty_poisson_info(opts: dict[str, object], dim: int) -> dict[str, object]:
    return {
        "dimension": dim,
        "mode": opts["mode"],
        "radius": opts["radius"],
        "min_radius": float(opts["min_radius"]),
        "attempts": int(opts["attempts"]),
        "seed": int(opts["seed"]),
        "deterministic": bool(opts["deterministic"]),
        "strip_count": int(opts["strip_count"]),
        "used_parallel": False,
        "boundary_refinement_fraction": float(opts["boundary_refinement_fraction"]),
        "boundary_distance": float(opts["boundary_distance"]),
        "num_points": 0,
    }


def _default_strip_count(use_parallel: bool, deterministic: bool, requested_strip_count: int | None) -> int:
    if deterministic:
        return 1
    if requested_strip_count is not None:
        return requested_strip_count
    if not use_parallel:
        return 1
    return 1


def _resolve_poisson_backend(backend: str, opts: dict[str, object]) -> str:
    backend = str(backend).strip().lower()
    supported_mode = str(opts["mode"]) in {
        "fixed_radius",
        "fixed_radius_with_boundary_refinement",
        "variable_radius",
        "variable_radius_with_boundary_refinement",
    }
    if backend == "auto":
        force_cpu = os.environ.get("KERNELPACK_JAX_USE_WARP", "").strip().lower() in {"0", "false", "no", "off"}
        if (not force_cpu) and supported_mode and warp_available():
            return "warp"
        return "cpu"
    if backend == "cpu":
        return "cpu"
    if backend == "warp":
        if not supported_mode:
            raise ValueError("backend='warp' currently supports only fixed-radius Poisson sampling modes")
        return "warp"
    raise ValueError("backend must be one of 'cpu', 'warp', or 'auto'")


def _estimate_variable_radius_lower_bound(
    radius_function: Callable[[jnp.ndarray, float], float],
    x_min: np.ndarray,
    x_max: np.ndarray,
    nominal_radius: float,
    sample_count: int = 2048,
) -> float:
    candidates = np.asarray(
        _build_deterministic_candidate_cloud(
            jnp.asarray(x_min, dtype=float),
            jnp.asarray(x_max, dtype=float),
            max(256, int(sample_count)),
            0,
        ),
        dtype=float,
    )
    radii = np.asarray([float(radius_function(jnp.asarray(point, dtype=float), float(nominal_radius))) for point in candidates], dtype=float)
    positive = radii[np.isfinite(radii) & (radii > 1e-10)]
    if positive.size == 0:
        raise ValueError("radius_function must produce positive finite radii")
    return float(np.min(positive))


def _build_strip_boxes(x_min: np.ndarray, x_max: np.ndarray, split_tol: float, strip_count: int) -> list[dict[str, np.ndarray]]:
    dim0 = (x_max[0] - x_min[0]) / strip_count
    out = []
    for k in range(strip_count):
        sample_min = x_min.copy()
        sample_max = x_max.copy()
        sample_min[0] = x_min[0] + k * dim0
        sample_max[0] = min(x_max[0], sample_min[0] + dim0 - 0.33 * split_tol)
        out.append({"sample_min": sample_min, "sample_max": sample_max})
    return out


def _poisson_strip_sample(
    sample_min: np.ndarray,
    sample_max: np.ndarray,
    opts: dict[str, object],
    seed: int,
) -> np.ndarray:
    if np.any(sample_max <= sample_min):
        return np.zeros((0, sample_min.size), dtype=float)

    dim = sample_min.size
    cell_size = float(opts["grid_radius"]) / np.sqrt(dim)
    grid_size = np.maximum(1, np.ceil((sample_max - sample_min) / cell_size).astype(int))
    grid: dict[tuple[int, ...], int] = {}
    points: list[np.ndarray] = []
    active: list[int] = []
    rng = np.random.Generator(np.random.MT19937(int(seed)))

    x0 = sample_min + rng.random(dim) * (sample_max - sample_min)
    points.append(x0)
    active.append(0)
    grid[_point_to_cell(x0, sample_min, cell_size)] = 0

    # TODO: The CPU Bridson sampler can hit pathological long runtimes for
    # some radius/box combinations near saturation; e.g. the H-channel timing
    # study stalled around h=0.1 with attempts=30, while lower attempts and some
    # neighboring h values finished normally. If this appears again, check
    # whether the active-list loop needs a progress guard or adaptive attempt cap.
    while active:
        pick = int(rng.integers(len(active)))
        active_idx = active[pick]
        base = points[active_idx]
        active_radius = _local_radius(base, opts)
        accepted = False
        for _ in range(int(opts["attempts"])):
            candidate = _propose_candidate(base, active_radius, rng)
            if np.any(candidate < sample_min) or np.any(candidate > sample_max):
                continue
            if _has_conflicting_neighbor(candidate, active_idx, points, grid, sample_min, cell_size, grid_size, opts):
                continue
            idx = len(points)
            points.append(candidate)
            active.append(idx)
            grid[_point_to_cell(candidate, sample_min, cell_size)] = idx
            accepted = True
            break
        if not accepted:
            active[pick] = active[-1]
            active.pop()

    return np.asarray(points, dtype=float)


def _propose_candidate(base: np.ndarray, radius: float, rng: np.random.Generator) -> np.ndarray:
    dim = base.size
    direction = rng.normal(size=dim)
    direction = direction / max(np.linalg.norm(direction), np.finfo(float).eps)
    shell_radius = radius * (1.0 + rng.random() * (2**dim - 1)) ** (1.0 / dim)
    return base + shell_radius * direction


def _local_radius(point: np.ndarray, opts: dict[str, object]) -> float:
    mode = str(opts["mode"])
    if mode == "fixed_radius":
        return float(opts["radius"])
    if mode == "fixed_radius_with_boundary_refinement":
        return _boundary_refined_radius(point, opts)
    if mode == "variable_radius":
        radius = float(opts["rad_func"](jnp.asarray(point, dtype=float), float(opts["min_radius"])))
        return max(radius, float(opts["min_radius"]))
    if mode == "variable_radius_with_boundary_refinement":
        base_radius = float(opts["rad_func"](jnp.asarray(point, dtype=float), float(opts["min_radius"])))
        base_radius = max(base_radius, float(opts["min_radius"]))
        if _boundary_rad_frac(point, opts) < 1.0:
            return float(opts["boundary_refinement_fraction"]) * float(opts["min_radius"])
        return base_radius
    raise ValueError("unknown Poisson sampling mode")


def _boundary_rad_frac(point: np.ndarray, opts: dict[str, object]) -> float:
    if not bool(opts["has_boundary_refinement"]):
        return 1.0
    dist = _nearest_boundary_distance_scalar(point, np.asarray(opts["boundary_points"], dtype=float))
    if dist <= float(opts["boundary_distance"]):
        return float(opts["boundary_refinement_fraction"])
    return 1.0


def _boundary_refined_radius(point: np.ndarray, opts: dict[str, object]) -> float:
    return _boundary_rad_frac(point, opts) * float(opts["radius"])


def _has_conflicting_neighbor(
    point: np.ndarray,
    active_idx: int,
    points: list[np.ndarray],
    grid: dict[tuple[int, ...], int],
    x_min: np.ndarray,
    cell_size: float,
    grid_size: np.ndarray,
    opts: dict[str, object],
) -> bool:
    candidate_radius = _local_radius(point, opts)
    mode = str(opts["mode"])
    if mode == "fixed_radius_with_boundary_refinement":
        radius_for_reach = max(candidate_radius, float(opts["radius"]))
        exclude_active = False
        pairwise = True
    elif mode in {"fixed_radius", "variable_radius", "variable_radius_with_boundary_refinement"}:
        radius_for_reach = candidate_radius
        exclude_active = True
        pairwise = False
    else:
        raise ValueError("unknown Poisson sampling mode")

    idx = np.asarray(_point_to_cell(point, x_min, cell_size))
    reach = max(1, int(np.ceil(radius_for_reach / cell_size)))
    ranges = [
        range(max(1, idx[d] - reach), min(int(grid_size[d]), idx[d] + reach) + 1)
        for d in range(idx.size)
    ]
    for cell in np.array(np.meshgrid(*ranges)).T.reshape(-1, idx.size):
        key = tuple(int(v) for v in cell)
        if key not in grid:
            continue
        j = grid[key]
        if exclude_active and j == active_idx:
            continue
        if pairwise:
            threshold = max(candidate_radius, _local_radius(points[j], opts))
        else:
            threshold = candidate_radius
        if np.linalg.norm(point - points[j]) < threshold:
            return True
    return False


def _nearest_boundary_distance_scalar(point: np.ndarray, boundary_points: np.ndarray) -> float:
    if boundary_points.size == 0:
        return float("inf")
    diffs = boundary_points - point
    return float(np.sqrt(np.nanmin(np.sum(diffs * diffs, axis=1))))


def _flatten_strip_clouds(local_clouds: list[np.ndarray], x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
    if not local_clouds:
        return np.zeros((0, x_min.size), dtype=float)
    any_points = [pts for pts in local_clouds if pts.size]
    if not any_points:
        return np.zeros((0, x_min.size), dtype=float)
    points = np.vstack(any_points)
    in_box = np.all((points >= x_min) & (points <= x_max), axis=1)
    return points[in_box]


def _point_to_cell(point: np.ndarray, x_min: np.ndarray, cell_size: float) -> tuple[int, ...]:
    return tuple(np.maximum(1, np.floor((point - x_min) / cell_size).astype(int) + 1))


def _supports_mapped_closed_curve_generator(geometry: object) -> bool:
    geom_model = getattr(geometry, "geom_model", None)
    if not isinstance(geom_model, dict):
        return False
    return geom_model.get("type") == "closed-curve-sbf" and getattr(geometry, "surf_dim", None) == 1


def _polygon_signed_area(points: jnp.ndarray) -> jnp.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * jnp.sum(x * jnp.roll(y, -1) - jnp.roll(x, -1) * y)


def _polygon_centroid(points: jnp.ndarray) -> jnp.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    cross = x * jnp.roll(y, -1) - jnp.roll(x, -1) * y
    denom = jnp.sum(cross)
    if jnp.abs(denom) <= 1e-14:
        return points.mean(axis=0)
    cx = jnp.sum((x + jnp.roll(x, -1)) * cross) / (3.0 * denom)
    cy = jnp.sum((y + jnp.roll(y, -1)) * cross) / (3.0 * denom)
    return jnp.array([cx, cy], dtype=float)


def _ensure_ccw_polygon(points: jnp.ndarray) -> jnp.ndarray:
    pts = jnp.asarray(points, dtype=float)
    return pts if float(_polygon_signed_area(pts)) >= 0.0 else jnp.flip(pts, axis=0)


def _closed_curve_length(curve: jnp.ndarray) -> float:
    pts = jnp.asarray(curve, dtype=float)
    if pts.shape[0] <= 1:
        return 0.0
    shifted = jnp.vstack([pts[1:], pts[:1]])
    return float(jnp.linalg.norm(shifted - pts, axis=1).sum())


def _orient_closed_curve_samples(
    points: jnp.ndarray,
    normals: jnp.ndarray,
    level_set: RBFLevelSet,
    probe_distance: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    pts = jnp.asarray(points, dtype=float)
    nrmls = jnp.asarray(normals, dtype=float)
    if float(_polygon_signed_area(pts)) < 0.0:
        pts = jnp.flip(pts, axis=0)
        nrmls = jnp.flip(nrmls, axis=0)
    probe = max(float(probe_distance), 1e-4)
    if pts.shape[0] > 0:
        phi_inward = level_set.evaluate(pts - probe * nrmls)
        if float(jnp.mean(phi_inward)) < 0.0:
            nrmls = -nrmls
    return pts, nrmls


@partial(jit, static_argnums=(1,))
def _sample_closed_curve_by_arc_length(curve: jnp.ndarray, target_count: int) -> jnp.ndarray:
    pts = jnp.asarray(curve, dtype=float)
    if pts.shape[0] == 0 or target_count <= 0:
        return jnp.zeros((0, pts.shape[1] if pts.ndim == 2 else 2), dtype=float)
    if pts.shape[0] == 1 or target_count == 1:
        return pts[:1]
    closed = jnp.vstack([pts, pts[:1]])
    seg = jnp.linalg.norm(closed[1:] - closed[:-1], axis=1)
    cum = jnp.concatenate([jnp.array([0.0], dtype=float), jnp.cumsum(seg)])
    total = jnp.maximum(cum[-1], jnp.finfo(float).eps)
    targets = jnp.arange(target_count, dtype=float) * (total / target_count)
    idx = jnp.clip(jnp.searchsorted(cum, targets, side="right") - 1, 0, pts.shape[0] - 1)
    seg_len = jnp.maximum(seg[idx], jnp.finfo(float).eps)
    frac = ((targets - cum[idx]) / seg_len)[:, None]
    return closed[idx] + frac * (closed[idx + 1] - closed[idx])


def _build_triangular_lattice_annulus(
    r_min: float,
    r_max: float,
    ref_spacing: float,
    seed: int,
    *,
    include_center: bool = False,
) -> jnp.ndarray:
    if r_max <= r_min + 1e-12:
        return jnp.zeros((0, 2), dtype=float)
    ref_spacing = max(ref_spacing, 1e-3)
    dy = ref_spacing * np.sqrt(3.0) / 2.0
    shifts = _seed_shifts(seed, 2)
    x_shift = (float(shifts[0]) - 0.5) * ref_spacing
    y_shift = (float(shifts[1]) - 0.5) * dy
    rows: list[np.ndarray] = []
    if include_center and r_min <= 1e-12:
        rows.append(np.zeros((1, 2), dtype=float))
    y_vals = np.arange(-r_max - dy, r_max + 2.0 * dy, dy)
    x_vals = np.arange(-r_max - ref_spacing, r_max + 2.0 * ref_spacing, ref_spacing)
    for j, y_base in enumerate(y_vals):
        y = y_base + y_shift
        row_offset = 0.5 * ref_spacing * (j % 2)
        x = x_vals + x_shift + row_offset
        pts = np.column_stack([x, np.full_like(x, y)])
        rho2 = np.sum(pts * pts, axis=1)
        mask = (rho2 < (r_max - 1e-8) ** 2) & (rho2 >= (r_min - 1e-8) ** 2)
        if np.any(mask):
            rows.append(pts[mask])
    return jnp.asarray(np.vstack(rows), dtype=float) if rows else jnp.zeros((0, 2), dtype=float)


def _build_explicit_shell_layers(
    boundary_curve: jnp.ndarray,
    boundary_normals: jnp.ndarray,
    *,
    clearance: float,
    shell_depth: float,
    shell_spacing: float,
) -> tuple[jnp.ndarray, dict[str, object]]:
    if shell_depth <= clearance + 1e-12:
        return jnp.zeros((0, 2), dtype=float), {
            "shell_depth": float(shell_depth),
            "shell_spacing": float(shell_spacing),
            "shell_layer_count": 0,
            "shell_layer_offsets": jnp.zeros(0, dtype=float),
            "shell_layer_counts": tuple(),
        }
    eps_offset = max(1e-6, 1e-2 * shell_spacing)
    offsets = np.arange(clearance + eps_offset, shell_depth + 0.5 * shell_spacing, shell_spacing, dtype=float)
    if offsets.size == 0 or abs(float(offsets[-1]) - float(shell_depth)) > 0.25 * shell_spacing:
        offsets = np.append(offsets, shell_depth)
    offsets = np.unique(np.clip(offsets, clearance + eps_offset, shell_depth))
    layers: list[jnp.ndarray] = []
    layer_counts: list[int] = []
    layer_spacings: list[float] = []
    for eta in offsets:
        offset_curve = boundary_curve - float(eta) * boundary_normals
        curve_length = _closed_curve_length(offset_curve)
        target_count = max(8, int(np.floor(curve_length / max(shell_spacing, 1e-6))))
        layer = _sample_closed_curve_by_arc_length(offset_curve, target_count)
        layers.append(layer)
        layer_counts.append(int(layer.shape[0]))
        layer_spacings.append(float(curve_length / max(target_count, 1)))
    shell_points = jnp.vstack(layers) if layers else jnp.zeros((0, boundary_curve.shape[1]), dtype=float)
    return shell_points, {
        "shell_depth": float(shell_depth),
        "shell_spacing": float(shell_spacing),
        "shell_layer_count": len(layers),
        "shell_layer_offsets": jnp.asarray(offsets, dtype=float),
        "shell_layer_counts": tuple(layer_counts),
        "shell_layer_mean_spacings": tuple(layer_spacings),
    }


def _build_reference_harmonic_core_cloud(
    core_boundary: jnp.ndarray,
    spacing: float,
    seed: int,
) -> tuple[jnp.ndarray, dict[str, float]]:
    xb = _ensure_ccw_polygon(core_boundary)
    area = max(float(jnp.abs(_polygon_signed_area(xb))), 1e-12)
    scale = max(float(np.sqrt(area / np.pi)), 1e-6)
    centroid = _polygon_centroid(xb)
    base_ref_spacing = min(max(spacing / scale, 1e-3), 0.5)
    radial_margin = min(0.2, max(0.5 * base_ref_spacing, 1e-3))
    r_max = max(0.0, 1.0 - radial_margin)
    points = _build_triangular_lattice_annulus(0.0, r_max, base_ref_spacing, seed, include_center=True)
    return points, {
        "reference_area_scale": scale,
        "reference_spacing": float(base_ref_spacing),
        "reference_inner_radius": float(r_max),
        "reference_core_area": float(area),
        "reference_core_centroid_x": float(centroid[0]),
        "reference_core_centroid_y": float(centroid[1]),
    }


def _build_star_shaped_radial_model(boundary_curve: jnp.ndarray) -> dict[str, jnp.ndarray] | None:
    boundary = _ensure_ccw_polygon(boundary_curve)
    centroid = _polygon_centroid(boundary)
    rel = boundary - centroid[None, :]
    radii = jnp.linalg.norm(rel, axis=1)
    if float(jnp.min(radii)) <= 1e-8:
        return None
    angles = np.unwrap(np.arctan2(np.asarray(rel[:, 1]), np.asarray(rel[:, 0])))
    diffs = np.diff(np.concatenate([angles, [angles[0] + 2.0 * np.pi]]))
    if np.any(diffs <= 1e-6):
        return None
    theta0 = float(angles[0])
    theta = np.mod(angles - theta0, 2.0 * np.pi)
    order = np.argsort(theta)
    theta_sorted = theta[order]
    radii_sorted = np.asarray(radii, dtype=float)[order]
    if theta_sorted[0] > 1e-8:
        theta_sorted = np.concatenate([[0.0], theta_sorted])
        radii_sorted = np.concatenate([[radii_sorted[-1]], radii_sorted])
    if theta_sorted[-1] < 2.0 * np.pi - 1e-8:
        theta_sorted = np.concatenate([theta_sorted, [2.0 * np.pi]])
        radii_sorted = np.concatenate([radii_sorted, [radii_sorted[0]]])
    return {
        "centroid": jnp.asarray(centroid, dtype=float),
        "theta0": jnp.asarray(theta0, dtype=float),
        "theta": jnp.asarray(theta_sorted, dtype=float),
        "radii": jnp.asarray(radii_sorted, dtype=float),
        "boundary_length": jnp.asarray(_closed_curve_length(boundary), dtype=float),
        "min_radius": jnp.asarray(jnp.min(radii), dtype=float),
    }


def _warp_reference_points_by_arc_length_density(
    radial_model: dict[str, jnp.ndarray],
    reference_points: jnp.ndarray,
) -> tuple[jnp.ndarray, dict[str, float]]:
    ref = jnp.asarray(reference_points, dtype=float)
    if ref.shape[0] == 0:
        return ref, {
            "reference_center_excluded": 0.0,
            "reference_angular_warp": 1.0,
        }
    theta_samples = np.asarray(radial_model["theta"], dtype=float)
    radial_samples = np.asarray(radial_model["radii"], dtype=float)
    delta_theta = np.maximum(np.diff(theta_samples), 1e-12)
    radial_prime = np.diff(radial_samples) / delta_theta
    weight = np.sqrt(radial_samples[:-1] ** 2 + radial_prime**2)
    cdf = np.concatenate([[0.0], np.cumsum(weight * delta_theta)])
    if cdf[-1] <= 1e-12:
        return ref, {
            "reference_center_excluded": 0.0,
            "reference_angular_warp": 0.0,
        }
    cdf = cdf / cdf[-1]
    rho = np.asarray(jnp.linalg.norm(ref, axis=1), dtype=float)
    theta_uniform = np.mod(np.arctan2(np.asarray(ref[:, 1]), np.asarray(ref[:, 0])), 2.0 * np.pi) / (2.0 * np.pi)
    theta_warped = np.interp(theta_uniform, cdf, theta_samples)
    warped = jnp.column_stack(
        [
            rho * jnp.cos(theta_warped + float(radial_model["theta0"])),
            rho * jnp.sin(theta_warped + float(radial_model["theta0"])),
        ]
    )
    return warped, {
        "reference_center_excluded": 0.0,
        "reference_angular_warp": 1.0,
    }


@jit
def _map_points_star_shaped_radial(
    centroid: jnp.ndarray,
    theta0: jnp.ndarray,
    theta_samples: jnp.ndarray,
    radial_samples: jnp.ndarray,
    query_points: jnp.ndarray,
) -> jnp.ndarray:
    q = jnp.asarray(query_points, dtype=float)
    rho = jnp.linalg.norm(q, axis=1)
    theta = jnp.mod(jnp.arctan2(q[:, 1], q[:, 0]) - theta0, 2.0 * jnp.pi)
    idx = jnp.clip(jnp.searchsorted(theta_samples, theta, side="right") - 1, 0, theta_samples.shape[0] - 2)
    theta_left = theta_samples[idx]
    theta_right = theta_samples[idx + 1]
    radius_left = radial_samples[idx]
    radius_right = radial_samples[idx + 1]
    denom = jnp.maximum(theta_right - theta_left, 1e-12)
    frac = (theta - theta_left) / denom
    radius = radius_left + frac * (radius_right - radius_left)
    direction = jnp.column_stack([jnp.cos(theta + theta0), jnp.sin(theta + theta0)])
    return centroid[None, :] + (rho * radius)[:, None] * direction


@jit
def _map_points_harmonic_extension(
    boundary_samples: jnp.ndarray,
    query_points: jnp.ndarray,
) -> jnp.ndarray:
    boundary = jnp.asarray(boundary_samples, dtype=float)
    q = jnp.asarray(query_points, dtype=float)
    z = boundary[:, 0] + 1j * boundary[:, 1]
    coeffs = jnp.fft.fft(z) / z.shape[0]
    freq = jnp.arange(z.shape[0], dtype=jnp.int32)
    signed_freq = jnp.where(freq <= z.shape[0] // 2, freq, freq - z.shape[0])
    rho = jnp.minimum(jnp.linalg.norm(q, axis=1), 1.0)
    theta = jnp.arctan2(q[:, 1], q[:, 0])
    basis = (rho[:, None] ** jnp.abs(signed_freq)[None, :]) * jnp.exp(1j * theta[:, None] * signed_freq[None, :])
    mapped = basis @ coeffs
    return jnp.column_stack([jnp.real(mapped), jnp.imag(mapped)])


def _generate_mapped_closed_curve_nodes(
    geometry: object,
    radius: float,
    *,
    seed: int,
    do_outer_refinement: bool,
    outer_fraction_of_h: float,
    outer_refinement_zone_size_as_multiple_of_h: float,
) -> tuple[jnp.ndarray, dict[str, object]]:
    xb, _, level_set = _build_boundary_state(geometry)
    polygon_count = max(6 * int(xb.shape[0]), 192)
    t = jnp.linspace(0.0, 1.0, polygon_count, endpoint=False)
    target_polygon, target_normals = _orient_closed_curve_samples(
        geometry._eval_closed_curve(t),
        geometry._eval_closed_curve_frame(t)[1],
        level_set,
        0.25 * radius,
    )
    clearance = outer_fraction_of_h * radius if do_outer_refinement else radius
    centroid = _polygon_centroid(target_polygon)
    radial_limit = 0.8 * float(jnp.min(jnp.linalg.norm(target_polygon - centroid[None, :], axis=1)))
    shell_depth_target = outer_refinement_zone_size_as_multiple_of_h * radius if do_outer_refinement and outer_fraction_of_h < 1.0 else clearance
    shell_depth = min(max(shell_depth_target, clearance), max(radial_limit, clearance))
    offset_polygon = target_polygon - shell_depth * target_normals
    for _ in range(8):
        phi_offset = level_set.evaluate(offset_polygon)
        if float(jnp.min(phi_offset)) >= 0.0 or shell_depth <= clearance + 1e-12:
            break
        shell_depth = max(clearance, 0.9 * shell_depth)
        offset_polygon = target_polygon - shell_depth * target_normals
    shell_points, shell_info = _build_explicit_shell_layers(
        target_polygon,
        target_normals,
        clearance=clearance,
        shell_depth=shell_depth,
        shell_spacing=clearance,
    )
    radial_model = _build_star_shaped_radial_model(offset_polygon)
    if radial_model is not None:
        reference_points, reference_info = _build_reference_harmonic_core_cloud(
            offset_polygon,
            radius,
            seed,
        )
        ref_mask = jnp.linalg.norm(reference_points, axis=1) > 1e-10
        reference_points = reference_points[ref_mask]
        warped_reference_points, warp_info = _warp_reference_points_by_arc_length_density(radial_model, reference_points)
        core_points = _map_points_star_shaped_radial(
            radial_model["centroid"],
            radial_model["theta0"],
            radial_model["theta"],
            radial_model["radii"],
            warped_reference_points,
        )
        core_method = "star_shaped_radial_arc_warped_lattice"
        mapped_mode = "shell_plus_arc_warped_radial_core"
        reference_info.update(warp_info)
        reference_info["reference_center_excluded"] = float(jnp.sum(~ref_mask))
        reference_points = warped_reference_points
    else:
        reference_points, reference_info = _build_reference_harmonic_core_cloud(
            offset_polygon,
            radius,
            seed,
        )
        core_points = _map_points_harmonic_extension(offset_polygon, reference_points)
        core_method = "harmonic_extension_fourier"
        mapped_mode = "shell_plus_harmonic_core"
    x = core_points if shell_points.shape[0] == 0 else jnp.vstack([shell_points, core_points])
    centroid_phi = float(level_set.evaluate(centroid[None, :])[0])
    info = {
        "dimension": 2,
        "mode": "mapped_closed_curve_hybrid",
        "radius": float(radius),
        "min_radius": float(radius),
        "attempts": 0,
        "seed": int(seed),
        "deterministic": True,
        "strip_count": 1,
        "used_parallel": False,
        "boundary_refinement_fraction": float(outer_fraction_of_h),
        "boundary_distance": float(outer_refinement_zone_size_as_multiple_of_h * radius if do_outer_refinement else 0.0),
        "num_points": int(x.shape[0]),
        "num_candidates": int(reference_points.shape[0] + shell_points.shape[0]),
        "num_shell_points": int(shell_points.shape[0]),
        "num_core_points": int(core_points.shape[0]),
        "mapping_method": mapped_mode,
        "core_mapping_method": core_method,
        "mapping_centroid": centroid,
        "mapping_centroid_phi": centroid_phi,
        "num_boundary_vertices": int(target_polygon.shape[0]),
        "boundary_polygon_source": "dense_closed_curve_eval",
        "clearance_radius": float(clearance),
        "core_boundary_star_shaped": bool(radial_model is not None),
    }
    info.update(reference_info)
    info.update(shell_info)
    return x, info


def _estimate_candidate_count(
    x_min: jnp.ndarray,
    x_max: jnp.ndarray,
    spacing: float,
    variable_radius: bool,
) -> int:
    widths = np.asarray(x_max - x_min, dtype=float)
    volume = float(np.prod(np.maximum(widths, 1e-12)))
    dim = widths.size
    nominal_count = max(1, int(np.ceil(volume / max(spacing**dim, 1e-12))))
    oversample = 12 if variable_radius else 10
    return max(256, oversample * nominal_count)


def _build_deterministic_candidate_cloud(
    x_min: jnp.ndarray,
    x_max: jnp.ndarray,
    count: int,
    seed: int,
) -> jnp.ndarray:
    dim = int(x_min.size)
    bases = _first_primes(dim)
    shifts = _seed_shifts(seed, dim)
    mins = jnp.asarray(x_min, dtype=float)
    widths = jnp.asarray(x_max - x_min, dtype=float)
    indices = jnp.arange(1, count + 1, dtype=jnp.int32)
    coords = []
    for d, base in enumerate(bases):
        seq = jnp.mod(_radical_inverse_array(indices, base) + shifts[d], 1.0)
        coords.append(mins[d] + widths[d] * seq)
    return jnp.stack(coords, axis=1)


def _sample_fixed_radius_with_boundary_refinement(
    x_min: jnp.ndarray,
    x_max: jnp.ndarray,
    coarse_spacing: float,
    refined_spacing: float,
    boundary_points: np.ndarray,
    boundary_distance: float,
    seed: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    refined_candidate_count = _estimate_candidate_count(x_min, x_max, refined_spacing, True) * 3
    refined_candidates = _build_deterministic_candidate_cloud(x_min, x_max, refined_candidate_count, seed + 104729)
    refined_dists = _nearest_boundary_distance(refined_candidates, boundary_points)
    refined_candidates = refined_candidates[refined_dists <= boundary_distance]

    refined_points = _greedy_fps_constant_radius(
        np.asarray(refined_candidates, dtype=float),
        refined_spacing,
    )

    coarse_candidate_count = _estimate_candidate_count(x_min, x_max, coarse_spacing, False)
    coarse_candidates = _build_deterministic_candidate_cloud(x_min, x_max, coarse_candidate_count, seed)
    coarse_dists = _nearest_boundary_distance(coarse_candidates, boundary_points)
    coarse_candidates = coarse_candidates[coarse_dists > boundary_distance]

    coarse_points = _greedy_fps_constant_radius(
        coarse_candidates,
        coarse_spacing,
        selected_seed=refined_points,
    )

    if refined_points.shape[0] > 0 and coarse_points.shape[0] > 0:
        points = jnp.vstack([refined_points, coarse_points])
    elif refined_points.shape[0] > 0:
        points = refined_points
    else:
        points = coarse_points
    candidates = jnp.vstack([jnp.asarray(refined_candidates, dtype=float), jnp.asarray(coarse_candidates, dtype=float)])
    return points, candidates


def _greedy_fps_constant_radius(
    candidates: jnp.ndarray | np.ndarray,
    radius: float,
    selected_seed: jnp.ndarray | np.ndarray | None = None,
) -> jnp.ndarray:
    candidates = jnp.asarray(candidates, dtype=float)
    if candidates.shape[0] == 0:
        return jnp.zeros((0, candidates.shape[1]), dtype=float)
    min_dists, selected = _fps_initial_state(candidates, None if selected_seed is None else jnp.asarray(selected_seed, dtype=float))
    picked, pick_count = _greedy_fps_constant_radius_impl(candidates, float(radius), min_dists, selected.shape[0] > 0)
    return candidates[picked[: int(pick_count)]]


def _farthest_point_sample_with_radius(
    candidates: jnp.ndarray,
    radius_or_func: float | Callable[[jnp.ndarray, float], float],
    min_radius: float,
    boundary_points: jnp.ndarray,
    boundary_refinement_fraction: float,
    boundary_distance: float,
) -> jnp.ndarray:
    if candidates.shape[0] == 0:
        return jnp.zeros((0, candidates.shape[1]), dtype=float)
    candidates = jnp.asarray(candidates, dtype=float)
    radii = _candidate_radii(
        candidates,
        radius_or_func,
        min_radius,
        boundary_points,
        boundary_refinement_fraction,
        boundary_distance,
    )
    picked, pick_count = _greedy_fps_variable_radius_impl(candidates, radii, float(min_radius))
    return candidates[picked[: int(pick_count)]]


def _candidate_radii(
    candidates: jnp.ndarray,
    radius_or_func: float | Callable[[jnp.ndarray, float], float],
    min_radius: float,
    boundary_points: jnp.ndarray,
    boundary_refinement_fraction: float,
    boundary_distance: float,
) -> jnp.ndarray:
    candidates = jnp.asarray(candidates, dtype=float)
    if callable(radius_or_func):
        radii = vmap(lambda point: radius_or_func(point, float(min_radius)))(candidates)
        radii = jnp.maximum(radii, min_radius)
    else:
        radii = jnp.full(candidates.shape[0], float(radius_or_func), dtype=float)
    boundary_points = jnp.asarray(boundary_points, dtype=float)
    if boundary_points.shape[0] > 0 and boundary_refinement_fraction < 1.0 and boundary_distance > 0.0:
        dists = _nearest_boundary_distance(candidates, boundary_points)
        refined = dists <= boundary_distance
        radii = jnp.where(refined, jnp.minimum(radii, boundary_refinement_fraction * min_radius), radii)
    return radii


def _first_primes(count: int) -> list[int]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < count:
        is_prime = True
        for p in primes:
            if candidate % p == 0:
                is_prime = False
                break
            if p * p > candidate:
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


def _seed_shifts(seed: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(dim)


def clip_points_by_geometry(
    x: jnp.ndarray,
    geometry: object,
    *,
    keep: str = "inside",
    tolerance: float = 0.0,
    boundary_clearance: float = 0.0,
    min_signed_distance: float = -jnp.inf,
    max_signed_distance: float = jnp.inf,
    auto_build_level_set: bool = True,
    use_parallel: bool = True,
    chunk_size: int = 5000,
    min_parallel_points: int = 20000,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    del use_parallel, chunk_size, min_parallel_points
    x = jnp.asarray(x, dtype=float)
    level_set = _ensure_geometry_level_set(geometry, auto_build_level_set)
    phi = level_set.evaluate(x)
    keep_l = keep.lower()
    if keep_l not in {"inside", "outside"}:
        raise ValueError("keep must be inside or outside")
    keep_mask = _clip_mask_from_phi(
        phi,
        keep_l == "inside",
        float(tolerance),
        float(boundary_clearance),
        float(min_signed_distance),
        float(max_signed_distance),
    )
    return x[keep_mask], keep_mask, phi


def _ensure_geometry_level_set(geometry: object, auto_build: bool) -> RBFLevelSet:
    if not hasattr(geometry, "get_level_set"):
        raise ValueError("geometry object must provide get_level_set")
    level_set = geometry.get_level_set()
    needs_build = not isinstance(level_set, RBFLevelSet) or level_set.n == 0
    if needs_build:
        if not auto_build:
            raise ValueError("geometry level set is not built")
        if hasattr(geometry, "build_level_set_from_geometric_model"):
            geometry.build_level_set_from_geometric_model(None)
        elif hasattr(geometry, "build_level_set"):
            geometry.build_level_set()
        else:
            raise ValueError("geometry cannot build a level set")
        level_set = geometry.get_level_set()
    return level_set


def bounding_box_extents(geometry: object, prefer_uniform: bool = True) -> tuple[jnp.ndarray, jnp.ndarray]:
    box = None
    if prefer_uniform and hasattr(geometry, "get_uniform_bounding_box"):
        box = geometry.get_uniform_bounding_box()
        if box is None or jnp.asarray(box).size == 0:
            if hasattr(geometry, "compute_uniform_bounding_box"):
                geometry.compute_uniform_bounding_box()
                box = geometry.get_uniform_bounding_box()
    if (box is None or jnp.asarray(box).size == 0) and hasattr(geometry, "get_bounding_box"):
        box = geometry.get_bounding_box()
        if box is None or jnp.asarray(box).size == 0:
            if hasattr(geometry, "compute_bounding_box"):
                geometry.compute_bounding_box()
                box = geometry.get_bounding_box()
    if box is None or jnp.asarray(box).size == 0:
        if prefer_uniform and hasattr(geometry, "get_uniform_sample_sites"):
            box = geometry.get_uniform_sample_sites()
        elif hasattr(geometry, "get_sample_sites"):
            box = geometry.get_sample_sites()
        elif hasattr(geometry, "get_bdry_nodes"):
            box = geometry.get_bdry_nodes()
        else:
            raise ValueError("geometry does not expose bounding data")
    box = jnp.asarray(box, dtype=float)
    return box.min(axis=0), box.max(axis=0)


@dataclass
class DomainNodeGenerator:
    xi: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    xb: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    xg: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    nrmls: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    xi_orig: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    xi_pds_raw: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    s_dim: int = 0
    last_info: dict[str, object] = field(default_factory=dict)
    descriptor: DomainDescriptor = field(default_factory=DomainDescriptor)

    def generate_poisson_nodes(self, radius: float, x_min: jnp.ndarray, x_max: jnp.ndarray, **kwargs: object) -> None:
        x, info = generate_poisson_nodes_in_box(radius, x_min, x_max, **kwargs)
        self.xi = x
        self.xb = jnp.zeros((0, x.shape[1]))
        self.xg = jnp.zeros((0, x.shape[1]))
        self.nrmls = jnp.zeros((0, x.shape[1]))
        self.xi_orig = x
        self.xi_pds_raw = x
        self.s_dim = x.shape[1]
        self.last_info = info
        self.descriptor = DomainDescriptor()

    def generate_interior_nodes_from_geometry(
        self,
        geometry: object,
        radius: float,
        *,
        radius_function: Callable[[jnp.ndarray, float], float] | None = None,
        do_outer_refinement: bool = False,
        outer_fraction_of_h: float = 1.0,
        outer_refinement_zone_size_as_multiple_of_h: float = 2.0,
        **kwargs: object,
    ) -> None:
        xb, nrmls, level_set = _build_boundary_state(geometry)
        sampler_kwargs = dict(kwargs)
        mapped_cloud_preferred = bool(sampler_kwargs.pop("mapped_cloud_preferred", False))
        use_mapped_closed_curve = (
            radius_function is None
            and mapped_cloud_preferred
            and _supports_mapped_closed_curve_generator(geometry)
        )
        if use_mapped_closed_curve:
            seed = int(sampler_kwargs.pop("seed", 0) or 0)
            x_raw, info = _generate_mapped_closed_curve_nodes(
                geometry,
                radius,
                seed=seed,
                do_outer_refinement=do_outer_refinement,
                outer_fraction_of_h=outer_fraction_of_h,
                outer_refinement_zone_size_as_multiple_of_h=outer_refinement_zone_size_as_multiple_of_h,
            )
        else:
            x_min, x_max = bounding_box_extents(geometry, True)
            sampler_input = radius if radius_function is None else radius_function
            if radius_function is None:
                sampler_kwargs["min_radius"] = radius
            else:
                sampler_kwargs.setdefault(
                    "min_radius",
                    _estimate_variable_radius_lower_bound(radius_function, np.asarray(x_min, dtype=float), np.asarray(x_max, dtype=float), float(radius)),
                )
            if do_outer_refinement:
                sampler_kwargs["boundary_points"] = xb
                sampler_kwargs["boundary_refinement_fraction"] = outer_fraction_of_h
                sampler_kwargs["boundary_distance"] = outer_refinement_zone_size_as_multiple_of_h * radius
            x_raw, info = generate_poisson_nodes_in_box(sampler_input, x_min, x_max, **sampler_kwargs)
        self.xi = x_raw
        self.xb = jnp.zeros((0, x_raw.shape[1]))
        self.xg = jnp.zeros((0, x_raw.shape[1]))
        self.nrmls = jnp.zeros((0, x_raw.shape[1]))
        self.xi_orig = x_raw
        self.xi_pds_raw = x_raw
        self.s_dim = x_raw.shape[1]
        self.last_info = info

        clearance = outer_fraction_of_h * radius if do_outer_refinement else radius
        self.clip_to_geometry(geometry, keep="inside", boundary_clearance=clearance)
        self.last_info["min_active_radius"] = clearance
        self.last_info["outer_refinement"] = {
            "enabled": bool(do_outer_refinement),
            "refinement_fraction": outer_fraction_of_h,
            "zone_size": outer_refinement_zone_size_as_multiple_of_h * radius,
            "boundary_distance": outer_refinement_zone_size_as_multiple_of_h * radius,
            "mapped_cloud": bool(use_mapped_closed_curve),
        }
        self.last_info["boundary_node_count"] = xb.shape[0]
        self.last_info["boundary_level_set_built"] = level_set is not None

    def clip_to_geometry(self, geometry: object, **kwargs: object) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x, mask, phi = clip_points_by_geometry(self.xi_pds_raw, geometry, **kwargs)
        self.xi = x
        self.xi_orig = x
        self.s_dim = x.shape[1]
        self.last_info["clip_mask"] = mask
        self.last_info["clip_count"] = x.shape[0]
        self.last_info["clip_levelset_values"] = phi[mask]
        return x, mask, phi

    def build_domain_descriptor_from_geometry(self, geometry: object, radius: float, **kwargs: object) -> DomainDescriptor:
        self.generate_interior_nodes_from_geometry(geometry, radius, **kwargs)
        xb, nrmls, level_set = _build_boundary_state(geometry)
        xg = xb + 0.5 * radius * nrmls
        descriptor = DomainDescriptor()
        descriptor.set_nodes(self.xi, xb, xg)
        descriptor.set_normals(nrmls)
        descriptor.set_sep_rad(float(radius))
        descriptor.set_outer_level_set(level_set)
        descriptor.set_boundary_level_sets([level_set])
        descriptor.build_structs()
        self.xb = xb
        self.xg = xg
        self.nrmls = nrmls
        self.descriptor = descriptor
        self.last_info["boundary_node_count"] = xb.shape[0]
        self.last_info["ghost_node_count"] = xg.shape[0]
        self.last_info["sep_radius"] = radius
        return descriptor

    def _build_anisotropic_domain_descriptor_from_geometry(
        self,
        geometry: object,
        radii: jnp.ndarray,
        min_radius: float,
        **kwargs: object,
    ) -> DomainDescriptor:
        sampler_kwargs = dict(kwargs)
        do_outer_refinement = bool(sampler_kwargs.pop("do_outer_refinement", False))
        outer_fraction_of_h = float(sampler_kwargs.pop("outer_fraction_of_h", 1.0))
        zone_multiple = float(sampler_kwargs.pop("outer_refinement_zone_size_as_multiple_of_h", 2.0))
        xb = jnp.asarray(geometry.get_bdry_nodes(), dtype=float)
        nrmls = jnp.asarray(geometry.get_bdry_nrmls(), dtype=float)
        level_set = geometry.get_level_set()
        if not isinstance(level_set, RBFLevelSet) or level_set.n == 0:
            geometry.build_level_set()
            level_set = geometry.get_level_set()

        x_min, x_max = bounding_box_extents(geometry, True)
        scaled_min = x_min / radii
        scaled_max = x_max / radii
        if do_outer_refinement:
            sampler_kwargs["boundary_points"] = jnp.asarray(geometry.get_uniform_bdry_nodes(), dtype=float) / radii
            sampler_kwargs["boundary_refinement_fraction"] = outer_fraction_of_h
            sampler_kwargs["boundary_distance"] = zone_multiple
        x_scaled, info = generate_poisson_nodes_in_box(
            1.0,
            scaled_min,
            scaled_max,
            min_radius=1.0,
            **sampler_kwargs,
        )
        x_raw = x_scaled * radii
        self.xi = x_raw
        self.xb = jnp.zeros((0, x_raw.shape[1]))
        self.xg = jnp.zeros((0, x_raw.shape[1]))
        self.nrmls = jnp.zeros((0, x_raw.shape[1]))
        self.xi_orig = x_raw
        self.xi_pds_raw = x_raw
        self.s_dim = x_raw.shape[1]
        self.last_info = info
        clearance = outer_fraction_of_h * min_radius if do_outer_refinement else min_radius
        self.clip_to_geometry(geometry, keep="inside", boundary_clearance=clearance)
        xg = xb + 0.5 * min_radius * nrmls
        descriptor = DomainDescriptor()
        descriptor.set_nodes(self.xi, xb, xg)
        descriptor.set_normals(nrmls)
        descriptor.set_sep_rad(float(min_radius))
        descriptor.set_outer_level_set(level_set)
        descriptor.set_boundary_level_sets([level_set])
        descriptor.build_structs()
        self.xb = xb
        self.xg = xg
        self.nrmls = nrmls
        self.descriptor = descriptor
        self.last_info["mode"] = "anisotropic_radii_with_boundary_refinement" if do_outer_refinement else "anisotropic_radii"
        self.last_info["anisotropic_radii"] = radii
        self.last_info["min_active_radius"] = clearance
        self.last_info["outer_refinement"] = {
            "enabled": do_outer_refinement,
            "refinement_fraction": outer_fraction_of_h,
            "zone_size": zone_multiple * min_radius,
            "boundary_distance": zone_multiple * min_radius,
            "scaled_boundary_distance": zone_multiple,
            "mapped_cloud": False,
        }
        self.last_info["boundary_node_count"] = xb.shape[0]
        self.last_info["ghost_node_count"] = xg.shape[0]
        self.last_info["sep_radius"] = float(min_radius)
        return descriptor

    def generate_piecewise_smooth_domain_nodes_by_segment(
        self,
        bdry_segments: list[jnp.ndarray],
        flip_normal: list[bool] | jnp.ndarray,
        radius: float,
        *,
        method: int = 1,
        supersample_fac: int = 2,
        smooth_normals: bool = False,
        smooth_neighborhood: int = 0,
        radius_function: Callable[[jnp.ndarray, float], float] | None = None,
        radii: list[float] | jnp.ndarray | None = None,
        bdry_mode: int = 0,
        **kwargs: object,
    ) -> DomainDescriptor:
        radii_arr = _validate_anisotropic_radii(radii, jnp.asarray(bdry_segments[0]).shape[1])
        fit_radius: float | jnp.ndarray = radii_arr if radii_arr is not None else radius
        effective_radius = float(jnp.min(radii_arr)) if radii_arr is not None else float(radius)
        _validate_piecewise_fit_bounds_from_segments(bdry_segments, None)
        piecewise_boundary = PiecewiseSmoothEmbeddedSurface()
        piecewise_boundary.generate_piecewise_smooth_surface_by_segment(
            bdry_segments,
            flip_normal,
            fit_radius,
            method=method,
            supersample_fac=supersample_fac,
            bdry_mode=bdry_mode,
            smooth_normals=smooth_normals,
            smooth_neighborhood=smooth_neighborhood,
        )
        _validate_piecewise_fit_bounds_from_segments(bdry_segments, piecewise_boundary)
        piecewise_boundary.build_level_set()
        if radii_arr is not None:
            descriptor = self._build_anisotropic_domain_descriptor_from_geometry(piecewise_boundary, radii_arr, effective_radius, **kwargs)
            self.last_info["piecewise_mode"] = "by_segment"
            self.last_info["segment_count"] = len(bdry_segments)
            self.last_info["piecewise_radius_function"] = False
            self.last_info["anisotropic_radii"] = radii_arr
            self.last_info["bdry_mode"] = int(bdry_mode)
            return descriptor
        if radius_function is not None:
            kwargs["radius_function"] = radius_function
        descriptor = self.build_domain_descriptor_from_geometry(piecewise_boundary, effective_radius, **kwargs)
        self.last_info["piecewise_mode"] = "by_segment"
        self.last_info["segment_count"] = len(bdry_segments)
        self.last_info["piecewise_radius_function"] = radius_function is not None
        return descriptor

    def generate_piecewise_smooth_domain_nodes_hierarchical(
        self,
        bdry_desc: list[jnp.ndarray],
        is_bdry_curve: list[bool] | jnp.ndarray,
        is_smooth_curve: list[bool] | jnp.ndarray,
        constants: list[float] | jnp.ndarray,
        constant_dim: list[int] | jnp.ndarray,
        flip_normal: list[bool] | jnp.ndarray,
        radius: float,
        *,
        method: int = 1,
        supersample_fac: int = 2,
        smooth_normals: bool = False,
        smooth_neighborhood: int = 0,
        radius_function: Callable[[jnp.ndarray, float], float] | None = None,
        radii: list[float] | jnp.ndarray | None = None,
        bdry_mode: int = 0,
        **kwargs: object,
    ) -> DomainDescriptor:
        if len(bdry_desc) == 0:
            raise ValueError("bdry_desc must contain at least one segment")
        radii_arr = _validate_anisotropic_radii(radii, 3)
        fit_radius: float | jnp.ndarray = radii_arr if radii_arr is not None else radius
        effective_radius = float(jnp.min(radii_arr)) if radii_arr is not None else float(radius)
        raw_size = _hierarchical_raw_boundary_size(bdry_desc, is_bdry_curve, constants, constant_dim)
        dim = int(jnp.asarray(bdry_desc[0]).shape[1])
        if dim != 2:
            # The C++ hierarchical entry point is specifically the 3D route:
            # 2D boundary curves may first be expanded into 3D surface patches.
            dim = 3
        segments: list[EmbeddedSurface] = []
        omit: list[bool] = []
        hierarchical_face_sample_counts: list[int] = []
        for idx, pts_in in enumerate(bdry_desc):
            pts = jnp.asarray(pts_in, dtype=float)
            if bool(is_bdry_curve[idx]) and bool(is_smooth_curve[idx]):
                base_nodes = self._generate_hierarchical_face_nodes(pts, radius, seed=int(kwargs.get("seed", 0) or 0) + idx)
                lifted = _lift_constant_dimension(base_nodes, int(constant_dim[idx]), float(constants[idx]))
                face = EmbeddedSurface()
                face.set_data_sites(lifted)
                face.set_sample_sites(lifted)
                face.compute_bounding_box()
                curve_box = pca_axis_aligned_extents_2d(pts)
                face_area_box = float(curve_box[0] * curve_box[1])
                num_samples = max(1, int(face_area_box / max(effective_radius * effective_radius, float(jnp.finfo(float).eps))))
                hierarchical_face_sample_counts.append(num_samples)
                face_build_radius: float | jnp.ndarray = fit_radius if int(bdry_mode) != 0 and radii_arr is not None else effective_radius
                face.build_geometric_model_ps(3, face_build_radius, lifted.shape[0], num_samples, method, supersample_fac)
                if bool(flip_normal[idx]):
                    face.flip_normals()
                _orient_constant_face_normals(face, int(constant_dim[idx]), -1.0 if bool(flip_normal[idx]) else 1.0)
                segments.append(face)
                omit.append(True)
            else:
                segments.append(EmbeddedSurface())
                omit.append(False)

        piecewise_boundary = PiecewiseSmoothEmbeddedSurface()
        piecewise_boundary.set_segments(segments)
        piecewise_boundary.generate_piecewise_smooth_surface_by_segment_with_omission(
            bdry_desc,
            omit,
            flip_normal,
            fit_radius,
            method=method,
            supersample_fac=supersample_fac,
            bdry_mode=bdry_mode,
            smooth_normals=smooth_normals,
            smooth_neighborhood=smooth_neighborhood,
        )
        _validate_assembled_size(raw_size, piecewise_boundary, "hierarchical piecewise-smooth surface")
        piecewise_boundary.build_level_set()
        if radii_arr is not None:
            descriptor = self._build_anisotropic_domain_descriptor_from_geometry(piecewise_boundary, radii_arr, effective_radius, **kwargs)
            self.last_info["piecewise_mode"] = "hierarchical"
            self.last_info["segment_count"] = len(bdry_desc)
            self.last_info["hierarchical_omitted_segments"] = sum(1 for item in omit if item)
            self.last_info["hierarchical_face_sample_counts"] = hierarchical_face_sample_counts
            self.last_info["piecewise_radius_function"] = False
            self.last_info["anisotropic_radii"] = radii_arr
            self.last_info["bdry_mode"] = int(bdry_mode)
            return descriptor
        if radius_function is not None:
            kwargs["radius_function"] = radius_function
        descriptor = self.build_domain_descriptor_from_geometry(piecewise_boundary, effective_radius, **kwargs)
        self.last_info["piecewise_mode"] = "hierarchical"
        self.last_info["segment_count"] = len(bdry_desc)
        self.last_info["hierarchical_omitted_segments"] = sum(1 for item in omit if item)
        self.last_info["hierarchical_face_sample_counts"] = hierarchical_face_sample_counts
        self.last_info["piecewise_radius_function"] = radius_function is not None
        return descriptor

    def _generate_hierarchical_face_nodes(self, boundary_curve: jnp.ndarray, radius: float, *, seed: int = 0) -> jnp.ndarray:
        surface = EmbeddedSurface()
        surface.set_data_sites(jnp.asarray(boundary_curve, dtype=float))
        temp_spacing = _closed_curve_length(boundary_curve) / max(int(boundary_curve.shape[0]), 1)
        surface.build_closed_geometric_model_ps(2, float(temp_spacing), boundary_curve.shape[0])
        surface.build_level_set_from_geometric_model()
        generator = DomainNodeGenerator()
        generator.generate_interior_nodes_from_geometry(
            surface,
            float(temp_spacing),
            seed=seed,
            strip_count=4,
            do_outer_refinement=True,
            outer_fraction_of_h=0.5,
            outer_refinement_zone_size_as_multiple_of_h=2.0,
            mapped_cloud_preferred=False,
        )
        return jnp.vstack([generator.get_interior_nodes(), surface.get_sample_sites()])

    def get_interior_nodes(self) -> jnp.ndarray:
        return self.xi

    def get_bdry_nodes(self) -> jnp.ndarray:
        return self.xb

    def get_ghost_nodes(self) -> jnp.ndarray:
        return self.xg

    def get_nrmls(self) -> jnp.ndarray:
        return self.nrmls

    def get_raw_poisson_interior_nodes(self) -> jnp.ndarray:
        return self.xi_pds_raw

    def get_domain_descriptor(self) -> DomainDescriptor:
        return self.descriptor


def _closed_curve_length(x: jnp.ndarray) -> float:
    pts = jnp.asarray(x, dtype=float)
    if pts.shape[0] <= 1:
        return 1.0
    shifted = jnp.vstack([pts[1:], pts[:1]])
    return float(jnp.linalg.norm(shifted - pts, axis=1).sum())


def pca_axis_aligned_extents_2d(x: jnp.ndarray) -> jnp.ndarray:
    pts = jnp.asarray(x, dtype=float)
    if pts.shape[0] == 0:
        return jnp.zeros(2, dtype=float)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    return maxs - mins


def _validate_anisotropic_radii(radii: list[float] | jnp.ndarray | None, dim: int) -> jnp.ndarray | None:
    if radii is None:
        return None
    arr = jnp.asarray(radii, dtype=float).reshape(-1)
    if arr.shape[0] != dim:
        raise ValueError("anisotropic radii must have one entry per coordinate dimension")
    if bool(jnp.any(arr <= 0.0)):
        raise ValueError("anisotropic radii must be positive")
    return arr


def _raw_boundary_size_from_segments(segments: list[jnp.ndarray]) -> float:
    if len(segments) == 0:
        return 0.0
    pts = jnp.vstack([jnp.asarray(seg, dtype=float) for seg in segments])
    if pts.shape[0] == 0:
        return 0.0
    ext = pts.max(axis=0) - pts.min(axis=0)
    if pts.shape[1] == 2:
        return float(2.0 * (ext[0] + ext[1]))
    if pts.shape[1] == 3:
        return float(2.0 * (ext[0] * ext[1] + ext[1] * ext[2] + ext[2] * ext[0]))
    raise ValueError("piecewise boundary segments must be 2D or 3D")


def _hierarchical_raw_boundary_size(
    bdry_desc: list[jnp.ndarray],
    is_bdry_curve: list[bool] | jnp.ndarray,
    constants: list[float] | jnp.ndarray,
    constant_dim: list[int] | jnp.ndarray,
) -> float:
    lifted_bounds: list[jnp.ndarray] = []
    for idx, desc_in in enumerate(bdry_desc):
        desc = jnp.asarray(desc_in, dtype=float)
        if bool(is_bdry_curve[idx]):
            c = float(constants[idx])
            cd = int(constant_dim[idx])
            mins2 = desc.min(axis=0)
            maxs2 = desc.max(axis=0)
            if cd == 0:
                mins = jnp.array([c, mins2[0], mins2[1]], dtype=float)
                maxs = jnp.array([c, maxs2[0], maxs2[1]], dtype=float)
            elif cd == 1:
                mins = jnp.array([mins2[0], c, mins2[1]], dtype=float)
                maxs = jnp.array([maxs2[0], c, maxs2[1]], dtype=float)
            elif cd == 2:
                mins = jnp.array([mins2[0], mins2[1], c], dtype=float)
                maxs = jnp.array([maxs2[0], maxs2[1], c], dtype=float)
            else:
                raise ValueError("constant_dim must be 0, 1, or 2")
            lifted_bounds.extend([mins, maxs])
        else:
            lifted_bounds.extend([desc.min(axis=0), desc.max(axis=0)])
    bounds = jnp.vstack(lifted_bounds)
    ext = bounds.max(axis=0) - bounds.min(axis=0)
    return float(2.0 * (ext[0] * ext[1] + ext[1] * ext[2] + ext[2] * ext[0]))


def _validate_piecewise_fit_bounds_from_segments(bdry_segments: list[jnp.ndarray], piecewise_boundary: PiecewiseSmoothEmbeddedSurface | None) -> None:
    if piecewise_boundary is None:
        return
    raw_size = _raw_boundary_size_from_segments(bdry_segments)
    _validate_assembled_size(raw_size, piecewise_boundary, "piecewise-smooth surface")


def _validate_assembled_size(raw_size: float, piecewise_boundary: PiecewiseSmoothEmbeddedSurface, label: str) -> None:
    if raw_size <= 0.0:
        return
    piecewise_boundary.compute_uniform_bounding_box()
    box = jnp.asarray(piecewise_boundary.get_uniform_bounding_box(), dtype=float)
    if box.size == 0:
        return
    ext = box.max(axis=0) - box.min(axis=0)
    if ext.shape[0] == 2:
        assembled_size = float(2.0 * (ext[0] + ext[1]))
    elif ext.shape[0] == 3:
        assembled_size = float(2.0 * (ext[0] * ext[1] + ext[1] * ext[2] + ext[2] * ext[0]))
    else:
        return
    if assembled_size > 2.0 * raw_size:
        raise RuntimeError(
            f"Something went wrong fitting the {label}. The assembled bounding box is too big. "
            f"Prior bdry size = {raw_size}. New bdry size = {assembled_size}."
        )


def _lift_constant_dimension(points: jnp.ndarray, constant_dim: int, constant_value: float) -> jnp.ndarray:
    pts = jnp.asarray(points, dtype=float)
    c = jnp.full((pts.shape[0], 1), float(constant_value), dtype=float)
    if constant_dim == 0:
        return jnp.column_stack([c, pts])
    if constant_dim == 1:
        return jnp.column_stack([pts[:, 0], c[:, 0], pts[:, 1]])
    if constant_dim == 2:
        return jnp.column_stack([pts, c])
    raise ValueError("constant_dim must be 0, 1, or 2")


def _orient_constant_face_normals(surface: EmbeddedSurface, constant_dim: int, sign: float) -> None:
    if constant_dim not in {0, 1, 2}:
        return
    target_sign = 1.0 if sign >= 0.0 else -1.0
    nr = jnp.asarray(surface.get_nrmls(), dtype=float)
    unr = jnp.asarray(surface.get_uniform_nrmls(), dtype=float)
    if nr.shape[0] and float(jnp.mean(nr[:, constant_dim])) * target_sign < 0.0:
        surface.flip_normals()
        nr = jnp.asarray(surface.get_nrmls(), dtype=float)
        unr = jnp.asarray(surface.get_uniform_nrmls(), dtype=float)
    axis = jnp.zeros((1, 3), dtype=float).at[0, constant_dim].set(target_sign)
    if nr.shape[0] and float(jnp.mean(jnp.abs(nr[:, constant_dim]))) > 0.95:
        surface.nrmls = jnp.broadcast_to(axis, nr.shape)
    if unr.shape[0] and float(jnp.mean(jnp.abs(unr[:, constant_dim]))) > 0.95:
        surface.uniform_nrmls = jnp.broadcast_to(axis, unr.shape)


def _build_boundary_state(geometry: object) -> tuple[jnp.ndarray, jnp.ndarray, RBFLevelSet]:
    if hasattr(geometry, "get_uniform_bdry_nodes"):
        xb = geometry.get_uniform_bdry_nodes()
        nrmls = geometry.get_uniform_bdry_nrmls()
    elif hasattr(geometry, "get_uniform_sample_sites"):
        xb = geometry.get_uniform_sample_sites()
        nrmls = geometry.get_uniform_nrmls()
    elif hasattr(geometry, "get_bdry_nodes"):
        xb = geometry.get_bdry_nodes()
        nrmls = geometry.get_bdry_nrmls()
    else:
        xb = geometry.get_sample_sites()
        nrmls = geometry.get_nrmls()
    level_set = geometry.get_level_set()
    if not isinstance(level_set, RBFLevelSet) or level_set.n == 0:
        if hasattr(geometry, "build_level_set_from_geometric_model"):
            geometry.build_level_set_from_geometric_model(None)
        else:
            geometry.build_level_set()
        level_set = geometry.get_level_set()
    return jnp.asarray(xb, dtype=float), jnp.asarray(nrmls, dtype=float), level_set
