from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kernelpack.geometry import EmbeddedSurface
from kernelpack.geometry.core import build_planar_parametric_eval_nodes_2d, project_to_best_fit_plane


H_CHANNEL_CURVE_AMPLITUDE = 0.06
ICE_CAP_HEIGHT = 0.5
DOMED_SQUARE_LENGTH = 1.2
DOMED_SQUARE_DOME_DEPTH = 0.6
DOMED_SQUARE_HALF_WIDTH = 0.6
DOMED_SQUARE_HALF_HEIGHT = 0.35
DOMED_SQUARE_EXPONENT = 4.0
DOMED_SQUARE_PATCHES = ("west", "y_min", "y_max", "z_min", "z_max", "shoulder", "dome")


def _normalized_arc_length_to_curve_parameter(s: np.ndarray, amplitude: float) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    if abs(amplitude) < np.finfo(float).eps:
        return s
    grid = np.linspace(0.0, 1.0, 10001)
    speed = np.sqrt(4.0 + (amplitude * np.pi * np.cos(np.pi * grid)) ** 2)
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(grid))])
    cumulative = cumulative / cumulative[-1]
    return np.interp(s, cumulative, grid)


def h_channel_exact_segments() -> np.ndarray:
    return np.asarray(
        [
            [[0.0, 0.0], [2.0, 0.0]],
            [[0.0, 1.0], [2.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.4]],
            [[2.0, 0.0], [2.0, 0.4]],
            [[0.0, 0.6], [0.0, 1.0]],
            [[2.0, 0.6], [2.0, 1.0]],
            [[0.9, 0.4], [0.9, 0.6]],
            [[1.1, 0.4], [1.1, 0.6]],
            [[0.0, 0.4], [0.9, 0.4]],
            [[0.0, 0.6], [0.9, 0.6]],
            [[1.1, 0.4], [2.0, 0.4]],
            [[1.1, 0.6], [2.0, 0.6]],
        ],
        dtype=float,
    )


def sample_h_channel_segment(segment_index: int, n: int, *, arc_length_parameter: bool = False) -> np.ndarray:
    param = np.linspace(0.0, 1.0, n, endpoint=True)
    if segment_index == 0:
        u = _normalized_arc_length_to_curve_parameter(param, -H_CHANNEL_CURVE_AMPLITUDE) if arc_length_parameter else param
        x = 2.0 * u
        y = -H_CHANNEL_CURVE_AMPLITUDE * np.sin(np.pi * u)
        return np.column_stack([x, y])
    if segment_index == 1:
        u = _normalized_arc_length_to_curve_parameter(param, H_CHANNEL_CURVE_AMPLITUDE) if arc_length_parameter else param
        x = 2.0 * u
        y = 1.0 + H_CHANNEL_CURVE_AMPLITUDE * np.sin(np.pi * u)
        return np.column_stack([x, y])
    u = param
    a, b = h_channel_exact_segments()[segment_index]
    return (1.0 - u[:, None]) * a[None, :] + u[:, None] * b[None, :]


def relative_errors(fitted: np.ndarray, exact: np.ndarray) -> dict[str, float]:
    err = fitted - exact
    component_l2 = np.linalg.norm(err, axis=0) / np.maximum(np.linalg.norm(exact, axis=0), np.finfo(float).eps)
    component_linf = np.max(np.abs(err), axis=0) / np.maximum(np.max(np.abs(exact), axis=0), np.finfo(float).eps)
    return {
        "rel_l2": float(np.linalg.norm(err.ravel()) / np.maximum(np.linalg.norm(exact.ravel()), np.finfo(float).eps)),
        "rel_linf": float(np.max(np.abs(err)) / np.maximum(np.max(np.abs(exact)), np.finfo(float).eps)),
        "component_rel_l2_max": float(np.max(component_l2)),
        "component_rel_linf_max": float(np.max(component_linf)),
        "abs_rms": float(np.sqrt(np.mean(err * err))),
        "abs_max": float(np.max(np.abs(err))),
    }


def h_channel_parametric_errors(seed_count: int, eval_count: int, radius: float) -> dict[str, float]:
    fitted_all = []
    exact_all = []
    start = time.perf_counter()
    segment_count = len(h_channel_exact_segments())
    for segment_index in range(segment_count):
        data = sample_h_channel_segment(segment_index, seed_count)
        surface = EmbeddedSurface()
        surface.set_data_sites(jnp.asarray(data))
        surface.build_geometric_model_ps(2, radius, data.shape[0], eval_count, method=2, supersample_fac=1)
        u = jnp.linspace(0.0, 1.0, eval_count)
        fitted_all.append(np.asarray(surface.evaluate_parametric(u)))
        exact_all.append(sample_h_channel_segment(segment_index, eval_count, arc_length_parameter=True))
    elapsed = time.perf_counter() - start
    metrics = relative_errors(np.vstack(fitted_all), np.vstack(exact_all))
    metrics["nd_build_points"] = int(segment_count * seed_count)
    metrics["nb_eval_points"] = int(segment_count * eval_count)
    metrics["actual_seed_count"] = metrics["nd_build_points"]
    metrics["build_seconds"] = elapsed
    return metrics


def h_channel_parametric_points(seed_count: int, eval_count: int, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fitted_all = []
    exact_all = []
    seed_all = []
    for segment_index in range(len(h_channel_exact_segments())):
        data = sample_h_channel_segment(segment_index, seed_count)
        surface = EmbeddedSurface()
        surface.set_data_sites(jnp.asarray(data))
        surface.build_geometric_model_ps(2, radius, data.shape[0], eval_count, method=2, supersample_fac=1)
        u = jnp.linspace(0.0, 1.0, eval_count)
        fitted_all.append(np.asarray(surface.evaluate_parametric(u)))
        exact_all.append(sample_h_channel_segment(segment_index, eval_count, arc_length_parameter=True))
        seed_all.append(data)
    return np.vstack(fitted_all), np.vstack(exact_all), np.vstack(seed_all)


def cap_seed_points(seed_count: int) -> np.ndarray:
    base_y = 1.0 - ICE_CAP_HEIGHT
    if seed_count <= 1:
        return np.asarray([[0.0, 1.0, 0.0]])
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    i = np.arange(seed_count, dtype=float)
    y = 1.0 - (1.0 - base_y) * i / max(seed_count - 1, 1)
    theta = golden_angle * i
    radius = np.sqrt(np.maximum(1.0 - y * y, 0.0))
    return np.column_stack([radius * np.cos(theta), y, radius * np.sin(theta)])


def disk_seed_points(seed_count: int) -> np.ndarray:
    base_y = 1.0 - ICE_CAP_HEIGHT
    base_radius = math.sqrt(1.0 - base_y * base_y)
    rings = max(3, int(round(math.sqrt(seed_count))))
    points = [np.asarray([[0.0, base_y, 0.0]])]
    for i, radius in enumerate(np.linspace(base_radius / max(rings - 1, 1), base_radius, rings - 1, endpoint=True), start=1):
        count = max(8, int(round(2.0 * math.pi * radius / max(base_radius / rings, 1e-12))))
        theta = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
        points.append(np.column_stack([radius * np.cos(theta), np.full_like(theta, base_y), radius * np.sin(theta)]))
    data = np.vstack(points)
    if data.shape[0] <= seed_count:
        return data
    keep = np.round(np.linspace(0, data.shape[0] - 1, seed_count)).astype(int)
    return data[keep]


def exact_cap_from_uv(data: np.ndarray, uv: np.ndarray) -> np.ndarray:
    base_y = 1.0 - ICE_CAP_HEIGHT
    _, origin, basis = project_to_best_fit_plane(jnp.asarray(data))
    projected = np.asarray(origin) + uv @ np.asarray(basis).T
    x = projected[:, 0]
    z = projected[:, 2]
    y = np.sqrt(np.maximum(1.0 - x * x - z * z, base_y * base_y))
    return np.column_stack([x, y, z])


def exact_disk_from_uv(data: np.ndarray, uv: np.ndarray) -> np.ndarray:
    _, origin, basis = project_to_best_fit_plane(jnp.asarray(data))
    projected = np.asarray(origin) + uv @ np.asarray(basis).T
    exact = projected.copy()
    exact[:, 1] = 0.7
    return exact


def fit_surface_patch(data: np.ndarray, eval_count: int, radius: float) -> tuple[np.ndarray, np.ndarray]:
    surface = EmbeddedSurface()
    surface.set_data_sites(jnp.asarray(data))
    surface.build_geometric_model_ps(3, radius, data.shape[0], eval_count, method=2, supersample_fac=1)
    uv_eval = np.asarray(build_planar_parametric_eval_nodes_2d(jnp.asarray(data), eval_count))
    fitted = np.asarray(surface.evaluate_parametric(jnp.asarray(uv_eval)))
    return fitted, uv_eval


def ice_cap_parametric_errors(seed_count: int, eval_count: int, radius: float) -> dict[str, float]:
    start = time.perf_counter()
    cap_data = cap_seed_points(seed_count)
    disk_data = disk_seed_points(seed_count)
    cap_fitted, cap_uv = fit_surface_patch(cap_data, eval_count, radius)
    disk_fitted, disk_uv = fit_surface_patch(disk_data, eval_count, radius)
    fitted = np.vstack([cap_fitted, disk_fitted])
    exact = np.vstack([exact_cap_from_uv(cap_data, cap_uv), exact_disk_from_uv(disk_data, disk_uv)])
    elapsed = time.perf_counter() - start
    metrics = relative_errors(fitted, exact)
    metrics["nd_build_points"] = int(cap_data.shape[0] + disk_data.shape[0])
    metrics["nb_eval_points"] = int(cap_uv.shape[0] + disk_uv.shape[0])
    metrics["actual_seed_count"] = metrics["nd_build_points"]
    metrics["build_seconds"] = elapsed
    return metrics


def ice_cap_parametric_points(seed_count: int, eval_count: int, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cap_data = cap_seed_points(seed_count)
    disk_data = disk_seed_points(seed_count)
    cap_fitted, cap_uv = fit_surface_patch(cap_data, eval_count, radius)
    disk_fitted, disk_uv = fit_surface_patch(disk_data, eval_count, radius)
    fitted = np.vstack([cap_fitted, disk_fitted])
    exact = np.vstack([exact_cap_from_uv(cap_data, cap_uv), exact_disk_from_uv(disk_data, disk_uv)])
    return fitted, exact, np.vstack([cap_data, disk_data])


def _select_evenly(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.shape[0] <= count:
        return points
    indices = np.round(np.linspace(0, points.shape[0] - 1, count)).astype(int)
    return points[indices]


def _rectangle_points(
    count: int,
    first: tuple[float, float],
    second: tuple[float, float],
    make_points,
) -> np.ndarray:
    side = max(2, int(math.ceil(math.sqrt(count))))
    u = np.linspace(first[0], first[1], side)
    v = np.linspace(second[0], second[1], side)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    return _select_evenly(make_points(uu.ravel(), vv.ravel()), count)


def _domed_square_level(y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return (np.abs(y) / DOMED_SQUARE_HALF_WIDTH) ** DOMED_SQUARE_EXPONENT + (
        np.abs(z) / DOMED_SQUARE_HALF_HEIGHT
    ) ** DOMED_SQUARE_EXPONENT


def _domed_square_dome_points(count: int) -> np.ndarray:
    indices = np.arange(max(1, count), dtype=float)
    radius = np.sqrt((indices + 0.5) / max(float(count), 1.0))
    theta = indices * math.pi * (3.0 - math.sqrt(5.0))
    power = 2.0 / DOMED_SQUARE_EXPONENT
    y = DOMED_SQUARE_HALF_WIDTH * radius**power * np.sign(np.cos(theta)) * np.abs(np.cos(theta)) ** power
    z = DOMED_SQUARE_HALF_HEIGHT * radius**power * np.sign(np.sin(theta)) * np.abs(np.sin(theta)) ** power
    level = _domed_square_level(y, z)
    x = DOMED_SQUARE_LENGTH + DOMED_SQUARE_DOME_DEPTH * np.sqrt(np.maximum(0.0, 1.0 - level))
    return np.column_stack([x, y, z])


def _domed_square_shoulder_points(count: int) -> np.ndarray:
    side = max(8, int(math.ceil(math.sqrt(4 * count))))
    while True:
        y = np.linspace(-DOMED_SQUARE_HALF_WIDTH, DOMED_SQUARE_HALF_WIDTH, side)
        z = np.linspace(-DOMED_SQUARE_HALF_HEIGHT, DOMED_SQUARE_HALF_HEIGHT, side)
        yy, zz = np.meshgrid(y, z, indexing="ij")
        keep = _domed_square_level(yy.ravel(), zz.ravel()) >= 1.0
        points = np.column_stack([np.full(np.count_nonzero(keep), DOMED_SQUARE_LENGTH), yy.ravel()[keep], zz.ravel()[keep]])
        if points.shape[0] >= count or side >= 4096:
            return _select_evenly(points, count)
        side *= 2


def domed_square_patch_points(patch: str, count: int) -> np.ndarray:
    if patch == "west":
        return _rectangle_points(
            count,
            (-DOMED_SQUARE_HALF_WIDTH, DOMED_SQUARE_HALF_WIDTH),
            (-DOMED_SQUARE_HALF_HEIGHT, DOMED_SQUARE_HALF_HEIGHT),
            lambda y, z: np.column_stack([np.zeros_like(y), y, z]),
        )
    if patch == "y_min":
        return _rectangle_points(
            count,
            (0.0, DOMED_SQUARE_LENGTH),
            (-DOMED_SQUARE_HALF_HEIGHT, DOMED_SQUARE_HALF_HEIGHT),
            lambda x, z: np.column_stack([x, np.full_like(x, -DOMED_SQUARE_HALF_WIDTH), z]),
        )
    if patch == "y_max":
        return _rectangle_points(
            count,
            (0.0, DOMED_SQUARE_LENGTH),
            (-DOMED_SQUARE_HALF_HEIGHT, DOMED_SQUARE_HALF_HEIGHT),
            lambda x, z: np.column_stack([x, np.full_like(x, DOMED_SQUARE_HALF_WIDTH), z]),
        )
    if patch == "z_min":
        return _rectangle_points(
            count,
            (0.0, DOMED_SQUARE_LENGTH),
            (-DOMED_SQUARE_HALF_WIDTH, DOMED_SQUARE_HALF_WIDTH),
            lambda x, y: np.column_stack([x, y, np.full_like(x, -DOMED_SQUARE_HALF_HEIGHT)]),
        )
    if patch == "z_max":
        return _rectangle_points(
            count,
            (0.0, DOMED_SQUARE_LENGTH),
            (-DOMED_SQUARE_HALF_WIDTH, DOMED_SQUARE_HALF_WIDTH),
            lambda x, y: np.column_stack([x, y, np.full_like(x, DOMED_SQUARE_HALF_HEIGHT)]),
        )
    if patch == "shoulder":
        return _domed_square_shoulder_points(count)
    if patch == "dome":
        return _domed_square_dome_points(count)
    raise ValueError(f"unknown domed-square patch {patch}")


def _fit_domed_square_patch(patch: str, seed_count: int, eval_count: int, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = domed_square_patch_points(patch, seed_count)
    exact = domed_square_patch_points(patch, eval_count)
    surface = EmbeddedSurface()
    surface.set_data_sites(jnp.asarray(data))
    surface.build_geometric_model_ps(3, radius, data.shape[0], eval_count, method=2, supersample_fac=1)
    _, origin, basis = project_to_best_fit_plane(jnp.asarray(data))
    uv_eval = (exact - np.asarray(origin)) @ np.asarray(basis)
    fitted = np.asarray(surface.evaluate_parametric(jnp.asarray(uv_eval)))
    return fitted, exact, data


def domed_square_parametric_errors(
    seed_count: int,
    eval_count: int,
    radius: float,
    *,
    patch_diagnostics: bool = False,
    dome_interior_levels: tuple[float, ...] = (0.8, 0.9, 0.95),
) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    fitted_all = []
    exact_all = []
    seed_all = []
    patch_rows: list[dict[str, float | int | str]] = []
    start = time.perf_counter()
    for patch in DOMED_SQUARE_PATCHES:
        fitted, exact, seeds = _fit_domed_square_patch(patch, seed_count, eval_count, radius)
        fitted_all.append(fitted)
        exact_all.append(exact)
        seed_all.append(seeds)
        if patch_diagnostics:
            patch_rows.append(
                {
                    "geometry": "domed_square",
                    "patch": patch,
                    "sample": "full",
                    "seed_count": seed_count,
                    "eval_count": int(exact.shape[0]),
                    "included_eval_points": int(exact.shape[0]),
                    **relative_errors(fitted, exact),
                }
            )
            if patch == "dome":
                level = _domed_square_level(exact[:, 1], exact[:, 2])
                for interior_level in dome_interior_levels:
                    interior = level <= interior_level
                    if not np.any(interior):
                        raise ValueError("dome interior diagnostic selected no evaluation points")
                    patch_rows.append(
                        {
                            "geometry": "domed_square",
                            "patch": patch,
                            "sample": f"level<={interior_level:g}",
                            "seed_count": seed_count,
                            "eval_count": int(exact.shape[0]),
                            "included_eval_points": int(np.count_nonzero(interior)),
                            **relative_errors(fitted[interior], exact[interior]),
                        }
                    )
    elapsed = time.perf_counter() - start
    metrics = relative_errors(np.vstack(fitted_all), np.vstack(exact_all))
    metrics["nd_build_points"] = int(sum(points.shape[0] for points in seed_all))
    metrics["nb_eval_points"] = int(sum(points.shape[0] for points in exact_all))
    metrics["actual_seed_count"] = metrics["nd_build_points"]
    metrics["build_seconds"] = elapsed
    return metrics, patch_rows


def domed_square_parametric_points(seed_count: int, eval_count: int, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fitted_all = []
    exact_all = []
    seed_all = []
    for patch in ("west", "y_min", "y_max", "z_min", "z_max", "shoulder", "dome"):
        fitted, exact, seeds = _fit_domed_square_patch(patch, seed_count, eval_count, radius)
        fitted_all.append(fitted)
        exact_all.append(exact)
        seed_all.append(seeds)
    return np.vstack(fitted_all), np.vstack(exact_all), np.vstack(seed_all)


def print_rows(rows: list[dict[str, float | int | str]]) -> None:
    columns = [
        "geometry",
        "dim",
        "seed_count",
        "actual_seed_count",
        "nd_build_points",
        "nb_eval_points",
        "eval_count",
        "build_seconds",
        "rel_l2",
        "rel_linf",
        "component_rel_l2_max",
        "component_rel_linf_max",
        "abs_rms",
        "abs_max",
    ]
    print(",".join(columns))
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                vals.append(f"{value:.8e}")
            else:
                vals.append(str(value))
        print(",".join(vals))


def write_rows(rows: list[dict[str, float | int | str]], path: Path) -> None:
    columns = [
        "geometry",
        "dim",
        "seed_count",
        "actual_seed_count",
        "nd_build_points",
        "nb_eval_points",
        "eval_count",
        "build_seconds",
        "rel_l2",
        "rel_linf",
        "component_rel_l2_max",
        "component_rel_linf_max",
        "abs_rms",
        "abs_max",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote csv to {path}")


def write_patch_rows(rows: list[dict[str, float | int | str]], path: Path) -> None:
    columns = [
        "geometry",
        "patch",
        "sample",
        "seed_count",
        "eval_count",
        "included_eval_points",
        "rel_l2",
        "rel_linf",
        "component_rel_l2_max",
        "component_rel_linf_max",
        "abs_rms",
        "abs_max",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote domed-square patch diagnostics to {path}")


def plot_rows(rows: list[dict[str, float | int | str]], plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    names = sorted({str(row["geometry"]) for row in rows})
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for name in names:
        group = sorted((row for row in rows if row["geometry"] == name), key=lambda row: int(row.get("nd_build_points", row["actual_seed_count"])))
        counts = np.asarray([row.get("nd_build_points", row["actual_seed_count"]) for row in group], dtype=float)
        rel_l2 = np.asarray([row["rel_l2"] for row in group], dtype=float)
        rel_linf = np.asarray([row["rel_linf"] for row in group], dtype=float)
        dim = int(group[0]["dim"])
        x = np.sqrt(counts) if dim == 3 else counts
        ax.loglog(x, rel_l2, "o-", label=f"{name} rel. l2")
        ax.loglog(x, rel_linf, "s--", label=f"{name} rel. linf")
    ax.set_xlabel("interpolant build points N_d (sqrt(N_d) for 3D)")
    ax.set_ylabel("relative Cartesian-coordinate error")
    ax.set_title("Parametric geometry convergence")
    ax.grid(alpha=0.2, which="both")
    ax.legend(loc="best")
    fig.savefig(plot_dir / "parametric_geometry_convergence.png", dpi=200)
    plt.close(fig)


def plot_parametric_shape(
    geometry_name: str,
    seed_count: int,
    eval_count: int,
    radius: float,
    plot_dir: Path,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    if geometry_name == "h_channel":
        fitted, exact, seeds = h_channel_parametric_points(seed_count, eval_count, radius)
    elif geometry_name == "ice_cap":
        fitted, exact, seeds = ice_cap_parametric_points(seed_count, eval_count, radius)
    elif geometry_name == "domed_square":
        fitted, exact, seeds = domed_square_parametric_points(seed_count, eval_count, radius)
    else:
        raise ValueError(f"unknown geometry {geometry_name}")

    path = plot_dir / f"{geometry_name}_parametric_seed{seed_count}.png"
    if exact.shape[1] == 2:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True, sharex=True, sharey=True)
        all_points = np.vstack([exact, fitted, seeds])
        pad = 0.05 * np.maximum(np.ptp(all_points, axis=0), 1.0e-12)
        mins = all_points.min(axis=0) - pad
        maxs = all_points.max(axis=0) + pad

        axes[0].plot(exact[:, 0], exact[:, 1], ".", color="#374151", ms=2)
        axes[0].plot(seeds[:, 0], seeds[:, 1], "o", color="#2563eb", ms=3, fillstyle="none")
        axes[0].set_title("Exact parametric map")
        axes[1].plot(fitted[:, 0], fitted[:, 1], ".", color="#dc2626", ms=2)
        axes[1].plot(seeds[:, 0], seeds[:, 1], "o", color="#2563eb", ms=3, fillstyle="none")
        axes[1].set_title("KernelPack fitted map")
        for ax in axes:
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(alpha=0.2)
    elif exact.shape[1] == 3:
        fig = plt.figure(figsize=(12, 5), constrained_layout=True)
        ax_exact = fig.add_subplot(1, 2, 1, projection="3d")
        ax_fit = fig.add_subplot(1, 2, 2, projection="3d")
        all_points = np.vstack([exact, fitted, seeds])
        mins = all_points.min(axis=0)
        maxs = all_points.max(axis=0)
        span = np.maximum(maxs - mins, 1.0e-12)
        pad = 0.05 * span
        mins = mins - pad
        maxs = maxs + pad

        ax_exact.scatter(exact[:, 0], exact[:, 1], exact[:, 2], s=2, color="#374151", alpha=0.35)
        ax_exact.scatter(seeds[:, 0], seeds[:, 1], seeds[:, 2], s=10, facecolors="none", edgecolors="#2563eb")
        ax_exact.set_title("Exact parametric map")
        ax_fit.scatter(fitted[:, 0], fitted[:, 1], fitted[:, 2], s=2, color="#dc2626", alpha=0.45)
        ax_fit.scatter(seeds[:, 0], seeds[:, 1], seeds[:, 2], s=10, facecolors="none", edgecolors="#2563eb")
        ax_fit.set_title("KernelPack fitted map")
        for ax in (ax_exact, ax_fit):
            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])
            ax.set_zlim(mins[2], maxs[2])
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(elev=24, azim=-58)
            try:
                ax.set_box_aspect(span)
            except Exception:
                pass
    else:
        raise ValueError(f"Cannot plot {exact.shape[1]}D data")

    fig.suptitle(f"{geometry_name}, seed nodes={seed_count}")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parametric-coordinate geometry convergence study.")
    parser.add_argument("--geometry", choices=("h_channel", "ice_cap", "domed_square", "both", "all"), default="both")
    parser.add_argument("--seed-counts", nargs="+", type=int, default=[40, 80, 160, 320])
    parser.add_argument("--eval-count", type=int, default=800)
    parser.add_argument("--radius", type=float, default=0.08)
    parser.add_argument(
        "--domed-square-patch-diagnostics",
        action="store_true",
        help="Write per-patch domed-square errors, including a dome interior metric.",
    )
    parser.add_argument(
        "--dome-interior-levels",
        nargs="+",
        type=float,
        default=None,
        help="Maximum dome levels included in interior diagnostics; each must be in (0, 1).",
    )
    parser.add_argument(
        "--dome-interior-level",
        type=float,
        default=None,
        help="Deprecated single-value form of --dome-interior-levels.",
    )
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    args = parser.parse_args()

    if args.dome_interior_level is not None and args.dome_interior_levels is not None:
        parser.error("use only one of --dome-interior-level and --dome-interior-levels")
    dome_interior_levels = (
        [args.dome_interior_level]
        if args.dome_interior_level is not None
        else (args.dome_interior_levels or [0.8, 0.9, 0.95])
    )
    if any(not 0.0 < level < 1.0 for level in dome_interior_levels):
        parser.error("each dome interior level must lie strictly between 0 and 1")

    rows: list[dict[str, float | int | str]] = []
    patch_rows: list[dict[str, float | int | str]] = []
    for seed_count in args.seed_counts:
        if args.geometry in ("h_channel", "both", "all"):
            metrics = h_channel_parametric_errors(seed_count, args.eval_count, args.radius)
            rows.append({"geometry": "h_channel", "dim": 2, "seed_count": seed_count, "actual_seed_count": seed_count, "eval_count": args.eval_count, **metrics})
        if args.geometry in ("ice_cap", "both", "all"):
            metrics = ice_cap_parametric_errors(seed_count, args.eval_count, args.radius)
            rows.append({"geometry": "ice_cap", "dim": 3, "seed_count": seed_count, "eval_count": args.eval_count, **metrics})
        if args.geometry in ("domed_square", "all"):
            metrics, diagnostics = domed_square_parametric_errors(
                seed_count,
                args.eval_count,
                args.radius,
                patch_diagnostics=args.domed_square_patch_diagnostics,
                dome_interior_levels=tuple(dome_interior_levels),
            )
            rows.append({"geometry": "domed_square", "dim": 3, "seed_count": seed_count, "eval_count": args.eval_count, **metrics})
            patch_rows.extend(diagnostics)

    print_rows(rows)
    if args.csv_output is not None:
        write_rows(rows, args.csv_output)
        if patch_rows:
            patch_path = args.csv_output.with_name(f"{args.csv_output.stem}_patches.csv")
            write_patch_rows(patch_rows, patch_path)
    if args.plot_dir is not None:
        plot_rows(rows, args.plot_dir)
        for seed_count in args.seed_counts:
            if args.geometry in ("h_channel", "both", "all"):
                plot_parametric_shape("h_channel", seed_count, args.eval_count, args.radius, args.plot_dir)
            if args.geometry in ("ice_cap", "both", "all"):
                plot_parametric_shape("ice_cap", seed_count, args.eval_count, args.radius, args.plot_dir)
            if args.geometry in ("domed_square", "all"):
                plot_parametric_shape("domed_square", seed_count, args.eval_count, args.radius, args.plot_dir)


if __name__ == "__main__":
    main()
