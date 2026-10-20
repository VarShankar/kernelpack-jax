from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

from jax import jit, lax
import jax
import jax.numpy as jnp
import numpy as np

from kernelpack.accelerators import WarpUnavailableError, warp_assembled_boundary_keep_mask, warp_available


@jit
def _distance_matrix_impl(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    diff = x[:, None, :] - y[None, :, :]
    return jnp.linalg.norm(diff, axis=2)


def distance_matrix(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=float)
    y = jnp.asarray(y, dtype=float)
    return _distance_matrix_impl(x, y)


@jit
def normalize_rows(x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=float)
    norms = jnp.linalg.norm(x, axis=1, keepdims=True)
    norms = jnp.where(norms > 0, norms, 1.0)
    return x / norms


@partial(jit, static_argnums=(1,))
def phs_kernel(r: jnp.ndarray, degree: int) -> jnp.ndarray:
    r = jnp.asarray(r, dtype=float)
    if degree % 2 == 0:
        return jnp.where(r > 0, r**degree * jnp.log(r + 2e-16), 0.0)
    return r**degree


def wrap_periodic_parameter(t: jnp.ndarray) -> jnp.ndarray:
    return jnp.mod(jnp.asarray(t, dtype=float), 1.0)


@jit
def periodic_chord_distance(theta1: jnp.ndarray, theta2: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    theta1 = jnp.asarray(theta1, dtype=float).reshape(-1, 1)
    theta2 = jnp.asarray(theta2, dtype=float).reshape(1, -1)
    delta = theta1 - theta2
    return jnp.sqrt(jnp.maximum(2.0 - 2.0 * jnp.cos(delta), 0.0)), delta


@jit
def sphere_chord_distance(x: jnp.ndarray, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    x = normalize_rows(jnp.asarray(x, dtype=float))
    y = normalize_rows(jnp.asarray(y, dtype=float))
    dots = jnp.clip(x @ y.T, -1.0, 1.0)
    return jnp.sqrt(jnp.maximum(2.0 - 2.0 * dots, 0.0)), dots


def chord_length_param(x: jnp.ndarray, closed: bool) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=float)
    if x.shape[0] <= 1:
        return jnp.zeros(x.shape[0], dtype=float)
    diffs = jnp.diff(x, axis=0)
    seg = jnp.linalg.norm(diffs, axis=1)
    if closed:
        seg = jnp.concatenate([seg, jnp.array([jnp.linalg.norm(x[0] - x[-1])])])
        s = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(seg[:-1])])
        return s / seg.sum()
    s = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(seg)])
    total = s[-1]
    return s / jnp.where(total > 0.0, total, 1.0)


def cart2sph_rows(x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=float)
    hxy = jnp.hypot(x[:, 0], x[:, 1])
    az = jnp.arctan2(x[:, 1], x[:, 0])
    el = jnp.arctan2(x[:, 2], hxy)
    r = jnp.linalg.norm(x, axis=1)
    return jnp.column_stack([az, el, r])


def fibonacci_sphere(n: int) -> jnp.ndarray:
    i = jnp.arange(n, dtype=float)
    phi = (1.0 + jnp.sqrt(5.0)) / 2.0
    theta = 2.0 * jnp.pi * i / phi
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = jnp.sqrt(jnp.maximum(1.0 - z * z, 0.0))
    return jnp.column_stack([r * jnp.cos(theta), r * jnp.sin(theta), z])


def pca_oriented_bounding_box(x: jnp.ndarray) -> dict[str, jnp.ndarray]:
    x = jnp.asarray(x, dtype=float)
    center = x.mean(axis=0)
    shifted = x - center
    _, _, vh = jnp.linalg.svd(shifted, full_matrices=False)
    local = shifted @ vh.T
    mins = local.min(axis=0)
    maxs = local.max(axis=0)
    axes = [jnp.asarray([mins[d], maxs[d]], dtype=float) for d in range(x.shape[1])]
    corners_local = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, x.shape[1])
    corners = corners_local @ vh + center
    return {"p": corners, "V": vh.T, "D": maxs - mins}


def weighted_sample_elimination_mis(x: jnp.ndarray, radius: float) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=float)
    if x.shape[0] == 0:
        return jnp.zeros(0, dtype=bool)
    if radius <= 0:
        keep = jnp.zeros(x.shape[0], dtype=bool)
        stride = max(1, x.shape[0] // max(1, min(x.shape[0], 64)))
        return keep.at[::stride].set(True)
    return _weighted_sample_elimination_mis_impl(x, float(radius))


@jit
def _weighted_sample_elimination_mis_impl(
    x: jnp.ndarray, radius: float, feature_flags: jnp.ndarray | None = None,
) -> jnp.ndarray:
    d = distance_matrix(x, x)
    n = x.shape[0]
    idx = jnp.arange(n)
    nonself = idx[:, None] != idx[None, :]
    c = jnp.minimum(d, 2.0 * radius)
    weights = jnp.sum(jnp.where(nonself & (d <= 2.0 * radius), (1.0 - c / (2.0 * radius)) ** 8, 0.0), axis=1)
    if feature_flags is not None:
        weights = jnp.where(feature_flags, jnp.inf, weights)
    conflict = nonself & (d <= radius)
    higher_priority = (weights[None, :] > weights[:, None]) | (
        (weights[None, :] == weights[:, None]) & (idx[None, :] < idx[:, None])
    )
    keep0 = jnp.zeros(n, dtype=bool)
    active0 = jnp.ones(n, dtype=bool)

    def cond_fun(state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        active, _, had_winner, iteration = state
        return jnp.any(active) & had_winner & (iteration < n)

    def body_fun(state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        active, keep, _, iteration = state
        dominated = jnp.any(conflict & active[None, :] & higher_priority, axis=1)
        round_winner = active & ~dominated
        removed_by_winner = jnp.any(conflict & round_winner[:, None], axis=0)
        active = active & ~round_winner & ~removed_by_winner
        keep = keep | round_winner
        return active, keep, jnp.any(round_winner), iteration + 1

    _, keep, _, _ = lax.while_loop(cond_fun, body_fun, (active0, keep0, jnp.array(True), jnp.array(0)))
    return keep


def resample_closed_curve_by_arc_length(curve: jnp.ndarray, target_count: int) -> jnp.ndarray:
    curve = jnp.asarray(curve, dtype=float)
    n = curve.shape[0]
    if n == 0:
        return jnp.zeros(0, dtype=jnp.uint32)
    if n == 1 or target_count <= 1:
        return jnp.array([0], dtype=jnp.uint32)
    return _resample_closed_curve_by_arc_length_impl(curve, int(target_count))


def _resample_closed_curve_by_arc_length_impl(curve: jnp.ndarray, target_count: int) -> jnp.ndarray:
    n = curve.shape[0]
    shifted = jnp.vstack([curve[1:], curve[:1]])
    seg_lens = jnp.linalg.norm(shifted - curve, axis=1)
    total = seg_lens.sum()
    if float(total) <= jnp.finfo(float).eps:
        count = min(n, target_count)
        return jnp.unique(jnp.round(jnp.linspace(0, n - 1, count)).astype(jnp.uint32))
    cum_len = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(seg_lens)])
    targets = jnp.arange(target_count, dtype=float) * (total / target_count)
    left = jnp.clip(jnp.searchsorted(cum_len, targets, side="right") - 1, 0, n - 1)
    right = (left + 1) % n
    left_dist = jnp.abs(cum_len[left] - targets)
    right_pos = jnp.where(left < n - 1, cum_len[left + 1], total)
    right_dist = jnp.abs(right_pos - targets)
    inds = jnp.where(right_dist < left_dist, right, left)
    return jnp.unique(inds.astype(jnp.uint32))


def project_to_best_fit_plane(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x = jnp.asarray(x, dtype=float)
    origin = x.mean(axis=0)
    shifted = x - origin
    denom = jnp.maximum(x.shape[0] - 1, 1)
    cov = (shifted.T @ shifted) / denom
    _, eigvecs = jnp.linalg.eigh(cov)
    basis = eigvecs[:, -2:]
    pivot = jnp.argmax(jnp.abs(basis), axis=0)
    signs = jnp.sign(basis[pivot, jnp.arange(basis.shape[1])])
    signs = jnp.where(signs == 0.0, 1.0, signs)
    signs = signs * jnp.array([1.0, -1.0])
    basis = basis * signs[None, :]
    uv = shifted @ basis
    return uv, origin, basis


def _convex_hull_vertices_2d(points: np.ndarray) -> np.ndarray:
    pts = np.unique(np.asarray(points, dtype=float), axis=0)
    if pts.shape[0] <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def points_in_convex_hull_2d(points: jnp.ndarray, hull_points: jnp.ndarray) -> jnp.ndarray:
    pts = np.asarray(points, dtype=float)
    hull = _convex_hull_vertices_2d(np.asarray(hull_points, dtype=float))
    if hull.shape[0] <= 2:
        mins = hull.min(axis=0)
        maxs = hull.max(axis=0)
        return jnp.asarray(np.all((pts >= mins) & (pts <= maxs), axis=1))
    tol = 1e-12
    keep = np.ones(pts.shape[0], dtype=bool)
    for i in range(hull.shape[0]):
        a = hull[i]
        b = hull[(i + 1) % hull.shape[0]]
        edge = b - a
        rel = pts - a
        keep &= (edge[0] * rel[:, 1] - edge[1] * rel[:, 0]) >= -tol
    return jnp.asarray(keep)


def build_planar_parametric_nodes_2d(x: jnp.ndarray, n: int, *_args: object) -> jnp.ndarray:
    uv, _, _ = project_to_best_fit_plane(jnp.asarray(x, dtype=float))
    if uv.shape[0] <= n:
        return uv
    idx = jnp.round(jnp.linspace(0, uv.shape[0] - 1, n)).astype(int)
    return uv[idx]


def build_planar_parametric_eval_nodes_2d(x: jnp.ndarray, n: int) -> jnp.ndarray:
    uv, _, _ = project_to_best_fit_plane(jnp.asarray(x, dtype=float))
    return _build_planar_parametric_eval_nodes_2d_pds(uv, int(n))


def _build_planar_parametric_eval_nodes_2d_pds(uv: jnp.ndarray, n: int) -> jnp.ndarray:
    uv_np = np.asarray(uv, dtype=float)
    if uv_np.shape[0] == 0 or n <= 0:
        return jnp.zeros((0, 2), dtype=float)
    mins = uv_np.min(axis=0)
    maxs = uv_np.max(axis=0)
    width = max(float(abs(maxs[0] - mins[0])), 1.0e-12)
    height = max(float(abs(maxs[1] - mins[1])), 1.0e-12)
    radius = float(np.sqrt(width * height / max(2 * n, 1)))
    last = np.zeros((0, 2), dtype=float)
    for attempt in range(5):
        samples = _poisson_disk_sample_box_2d(mins, maxs, radius, seed=attempt, attempts=5)
        if samples.shape[0] > 0:
            mask = np.asarray(points_in_convex_hull_2d(jnp.asarray(samples, dtype=float), uv), dtype=bool)
            samples = samples[mask]
        last = samples
        if samples.shape[0] >= max(1, n // 2) or attempt == 4:
            break
        radius *= 0.85
    return jnp.asarray(last, dtype=float)


def _poisson_disk_sample_box_2d(
    x_min: np.ndarray,
    x_max: np.ndarray,
    radius: float,
    *,
    seed: int,
    attempts: int,
) -> np.ndarray:
    if radius <= 0.0:
        return np.zeros((0, 2), dtype=float)
    rng = np.random.Generator(np.random.MT19937(int(seed)))
    x_min = np.asarray(x_min, dtype=float)
    x_max = np.asarray(x_max, dtype=float)
    span = np.maximum(x_max - x_min, 0.0)
    if float(np.max(span)) <= 0.0:
        return x_min.reshape(1, 2)
    cell = radius / np.sqrt(2.0)
    grid_shape = np.maximum(np.ceil(span / cell).astype(int), 1)
    grid = -np.ones((grid_shape[0], grid_shape[1]), dtype=int)
    points: list[np.ndarray] = []
    active: list[int] = []

    def grid_coord(point: np.ndarray) -> tuple[int, int]:
        coord = np.floor((point - x_min) / cell).astype(int)
        coord = np.minimum(np.maximum(coord, 0), grid_shape - 1)
        return int(coord[0]), int(coord[1])

    def is_valid(point: np.ndarray) -> bool:
        if np.any(point < x_min) or np.any(point > x_max):
            return False
        gx, gy = grid_coord(point)
        for ix in range(max(gx - 2, 0), min(gx + 3, grid_shape[0])):
            for iy in range(max(gy - 2, 0), min(gy + 3, grid_shape[1])):
                point_index = grid[ix, iy]
                if point_index >= 0 and np.linalg.norm(point - points[point_index]) < radius:
                    return False
        return True

    first = x_min + span * rng.random(2)
    points.append(first)
    active.append(0)
    grid[grid_coord(first)] = 0
    while active:
        active_slot = int(rng.integers(len(active)))
        base_index = active[active_slot]
        base = points[base_index]
        accepted = False
        for _ in range(max(1, attempts)):
            angle = 2.0 * np.pi * rng.random()
            distance = radius * (1.0 + rng.random())
            candidate = base + distance * np.array([np.cos(angle), np.sin(angle)])
            if is_valid(candidate):
                points.append(candidate)
                active.append(len(points) - 1)
                grid[grid_coord(candidate)] = len(points) - 1
                accepted = True
                break
        if not accepted:
            active.pop(active_slot)
    return np.asarray(points, dtype=float)


@partial(jit, static_argnums=(2,))
def _eval_closed_curve_model(theta_model: jnp.ndarray, weights: jnp.ndarray, degree: int, t: jnp.ndarray) -> jnp.ndarray:
    tw = wrap_periodic_parameter(jnp.asarray(t, dtype=float).reshape(-1))
    theta = 2.0 * jnp.pi * tw
    r, _ = periodic_chord_distance(theta, theta_model)
    return phs_kernel(r, degree) @ weights


@partial(jit, static_argnums=(2,))
def _eval_closed_curve_frame_model(theta_model: jnp.ndarray, weights: jnp.ndarray, degree: int, t: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    tw = wrap_periodic_parameter(jnp.asarray(t, dtype=float).reshape(-1))
    theta = 2.0 * jnp.pi * tw
    r, delta = periodic_chord_distance(theta, theta_model)
    dphi_degree_one = jnp.where(r > 0, jnp.sin(delta) / jnp.maximum(r, jnp.finfo(float).eps), 0.0)
    dphi_higher = degree * jnp.sin(delta) * r ** jnp.maximum(degree - 2, 0)
    dphi_higher = jnp.where(r == 0, 0.0, dphi_higher)
    dphi = lax.cond(degree == 1, lambda _: dphi_degree_one, lambda _: dphi_higher, operand=None)
    xt = (2.0 * jnp.pi) * (dphi @ weights)
    n = normalize_rows(jnp.column_stack([xt[:, 1], -xt[:, 0]]))
    return xt, n


@partial(jit, static_argnums=(1, 2, 3, 4))
def _build_closed_curve_geometric_model_ps_impl(
    data: jnp.ndarray,
    ntarget: int,
    method: int,
    supersample_fac: int,
    degree: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    data = jnp.asarray(data, dtype=float)
    t = chord_length_param(data, True)
    theta = 2.0 * jnp.pi * t
    r, _ = periodic_chord_distance(theta, theta)
    k = phs_kernel(r, degree)
    reg = 1e-12 * jnp.maximum(1.0, jnp.abs(k).max(initial=0.0))
    weights = jnp.linalg.solve(k + reg * jnp.eye(k.shape[0]), data)
    _, data_normals = _eval_closed_curve_frame_model(theta, weights, degree, t)
    ns = max(round(supersample_fac * 1.5 * ntarget), ntarget) if method == 1 else ntarget
    ts = jnp.linspace(0.0, 1.0, ns, endpoint=False)
    ptss = _eval_closed_curve_model(theta, weights, degree, ts)
    _, nr = _eval_closed_curve_frame_model(theta, weights, degree, ts)
    return theta, weights, data_normals, ptss, nr


@partial(jit, static_argnums=(3,))
def _eval_open_curve_model(u_model: jnp.ndarray, weights: jnp.ndarray, poly_coeffs: jnp.ndarray, degree: int, u: jnp.ndarray) -> jnp.ndarray:
    u = jnp.clip(jnp.asarray(u, dtype=float).reshape(-1), 0.0, 1.0)
    # In one parameter dimension use abs directly: norm(0) introduces a
    # spurious NaN under AD, while |du|^3 has a finite first derivative.
    r = jnp.abs(u[:, None] - u_model[None, :])
    return phs_kernel(r, degree) @ weights + jnp.column_stack([jnp.ones(u.size), u]) @ poly_coeffs


@partial(jit, static_argnums=(3,))
def _eval_open_curve_frame_model(u_model: jnp.ndarray, weights: jnp.ndarray, poly_coeffs: jnp.ndarray, degree: int, u: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    u = jnp.clip(jnp.asarray(u, dtype=float).reshape(-1), 0.0, 1.0)
    du = u[:, None] - u_model[None, :]
    r = jnp.abs(du)
    dphi_degree_one = jnp.sign(du)
    dphi_higher = degree * jnp.sign(du) * r ** jnp.maximum(degree - 1, 0)
    dphi_higher = jnp.where(r == 0, 0.0, dphi_higher)
    dphi = lax.cond(degree == 1, lambda _: dphi_degree_one, lambda _: dphi_higher, operand=None)
    xt = dphi @ weights + poly_coeffs[1]
    n = normalize_rows(jnp.column_stack([xt[:, 1], -xt[:, 0]]))
    return xt, n


@partial(jit, static_argnums=(1, 2, 3, 4, 7))
def _build_open_curve_geometric_model_ps_impl(
    data: jnp.ndarray,
    ntarget: int,
    method: int,
    supersample_fac: int,
    degree: int,
    rad_arr: jnp.ndarray,
    min_rad: float,
    anisotropic: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if degree not in (1, 2, 3):
        raise ValueError("affine open-curve augmentation requires a PHS degree at most 3")
    data = jnp.asarray(data, dtype=float)
    u = chord_length_param(data, False)
    r = jnp.abs(u[:, None] - u[None, :])
    k = phs_kernel(r, degree)
    p = jnp.column_stack([jnp.ones(u.size), u])
    reg = 1e-12 * jnp.maximum(1.0, jnp.abs(k).max(initial=0.0))
    a = jnp.block([[k + reg * jnp.eye(u.size), p], [p.T, jnp.zeros((2, 2))]])
    coeffs = jnp.linalg.solve(a, jnp.vstack([data, jnp.zeros((2, data.shape[1]))]))
    weights = coeffs[: u.size]
    poly_coeffs = coeffs[u.size :]
    ns = max(round(supersample_fac * 1.5 * ntarget), ntarget) if method == 1 else ntarget
    us = jnp.linspace(0.0, 1.0, ns)
    pts = _eval_open_curve_model(u, weights, poly_coeffs, degree, us)
    _, nr = _eval_open_curve_frame_model(u, weights, poly_coeffs, degree, us)
    # Open-curve endpoints are actual geometric features. Give them priority
    # during thinning, rather than mislabeling surviving interior samples.
    endpoints = (jnp.arange(len(pts)) == 0) | (jnp.arange(len(pts)) == len(pts)-1)
    if method == 1 and anisotropic:
        keep = _weighted_sample_elimination_mis_impl(pts / rad_arr[None, :], 1.0, endpoints)
        keep_uniform = _weighted_sample_elimination_mis_impl(pts, min_rad, endpoints)
    elif method == 1:
        keep = _weighted_sample_elimination_mis_impl(pts, min_rad, endpoints)
        keep_uniform = keep
    else:
        keep = jnp.ones(pts.shape[0], dtype=bool)
        keep_uniform = keep
    fallback = jnp.arange(pts.shape[0]) == 0
    keep = jnp.where(jnp.any(keep), keep, fallback)
    keep_uniform = jnp.where(jnp.any(keep_uniform), keep_uniform, fallback)
    return u, weights, poly_coeffs, pts, nr, keep, keep_uniform


@partial(jit, static_argnums=(2,))
def _eval_closed_surface_model(unit_centers: jnp.ndarray, weights: jnp.ndarray, degree: int, uv: jnp.ndarray) -> jnp.ndarray:
    uv = jnp.asarray(uv, dtype=float)
    unit_query = jnp.column_stack([jnp.cos(uv[:, 1]) * jnp.cos(uv[:, 0]), jnp.cos(uv[:, 1]) * jnp.sin(uv[:, 0]), jnp.sin(uv[:, 1])])
    r, _ = sphere_chord_distance(unit_query, unit_centers)
    return phs_kernel(r, degree) @ weights


@partial(jit, static_argnums=(2,))
def _eval_closed_surface_frame_model(unit_centers: jnp.ndarray, weights: jnp.ndarray, degree: int, tangent_step: float, uv: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    uv = jnp.asarray(uv, dtype=float)
    h = tangent_step
    uv_up = uv.at[:, 0].add(h)
    uv_um = uv.at[:, 0].add(-h)
    tu = (_eval_closed_surface_model(unit_centers, weights, degree, uv_up) - _eval_closed_surface_model(unit_centers, weights, degree, uv_um)) / (2.0 * h)
    uv_vp = uv.at[:, 1].set(jnp.minimum(uv[:, 1] + h, jnp.pi / 2.0))
    uv_vm = uv.at[:, 1].set(jnp.maximum(uv[:, 1] - h, -jnp.pi / 2.0))
    denom = jnp.maximum((uv_vp[:, 1] - uv_vm[:, 1])[:, None], jnp.finfo(float).eps)
    tv = (_eval_closed_surface_model(unit_centers, weights, degree, uv_vp) - _eval_closed_surface_model(unit_centers, weights, degree, uv_vm)) / denom
    n = normalize_rows(jnp.cross(tu, tv))
    return tu, tv, n


@partial(jit, static_argnums=(1, 2, 3, 4))
def _build_closed_surface_geometric_model_ps_impl(
    data: jnp.ndarray,
    ntarget: int,
    method: int,
    supersample_fac: int,
    degree: int,
    rad: float,
    tangent_step: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    data = jnp.asarray(data, dtype=float)
    center = data.mean(axis=0)
    local = data - center
    uvw = cart2sph_rows(local)
    unit_centers = local / jnp.maximum(uvw[:, 2:3], jnp.finfo(float).eps)
    r, _ = sphere_chord_distance(unit_centers, unit_centers)
    k = phs_kernel(r, degree)
    reg = 1e-12 * jnp.maximum(1.0, jnp.abs(k).max(initial=0.0))
    weights = jnp.linalg.solve(k + reg * jnp.eye(k.shape[0]), data)
    ns = max(supersample_fac * ntarget, ntarget) if method == 1 else ntarget
    xyz = fibonacci_sphere(ns)
    uv = cart2sph_rows(xyz)[:, :2]
    ptss = _eval_closed_surface_model(unit_centers, weights, degree, uv)
    _, _, nr = _eval_closed_surface_frame_model(unit_centers, weights, degree, tangent_step, uv)
    if method == 1:
        keep = _weighted_sample_elimination_mis_impl(ptss, rad)
    else:
        keep = jnp.ones(ptss.shape[0], dtype=bool)
    return center, unit_centers, weights, ptss, nr, keep


@partial(jit, static_argnums=(4, 8, 9))
def _build_open_surface_geometric_model_ps_impl(
    data: jnp.ndarray,
    uv: jnp.ndarray,
    uv_eval: jnp.ndarray,
    rad_arr: jnp.ndarray,
    degree: int,
    min_rad: float,
    origin: jnp.ndarray,
    basis: jnp.ndarray,
    method: int,
    anisotropic: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    del origin, basis
    if degree not in (1, 2, 3):
        raise ValueError("affine surface-patch augmentation requires a PHS degree at most 3")
    data = jnp.asarray(data, dtype=float)
    uv = jnp.asarray(uv, dtype=float)
    uv_eval = jnp.asarray(uv_eval, dtype=float)
    r = distance_matrix(uv, uv)
    k = phs_kernel(r, degree)
    p = jnp.column_stack([jnp.ones(uv.shape[0]), uv])
    reg = 1e-12 * jnp.maximum(1.0, jnp.abs(k).max(initial=0.0))
    a = jnp.block([[k + reg * jnp.eye(uv.shape[0]), p], [p.T, jnp.zeros((3, 3))]])
    coeffs = jnp.linalg.solve(a, jnp.vstack([data, jnp.zeros((3, data.shape[1]))]))
    weights = coeffs[: uv.shape[0]]
    poly_coeffs = coeffs[uv.shape[0] :]
    pts = _eval_open_surface_model(uv, weights, poly_coeffs, degree, uv_eval)
    _, _, nr = _eval_open_surface_frame_model(uv, weights, poly_coeffs, degree, uv_eval)
    if method == 1 and anisotropic:
        keep = _weighted_sample_elimination_mis_impl(pts / rad_arr[None, :], 1.0)
        keep_uniform = _weighted_sample_elimination_mis_impl(pts, min_rad)
    elif method == 1:
        keep = _weighted_sample_elimination_mis_impl(pts, min_rad)
        keep_uniform = keep
    else:
        keep = jnp.ones(pts.shape[0], dtype=bool)
        keep_uniform = keep
    fallback = jnp.arange(pts.shape[0]) == 0
    keep = jnp.where(jnp.any(keep), keep, fallback)
    keep_uniform = jnp.where(jnp.any(keep_uniform), keep_uniform, fallback)
    return weights, poly_coeffs, pts, nr, keep, keep_uniform


@partial(jit, static_argnums=(3,))
def _eval_open_surface_model(uv_model: jnp.ndarray, weights: jnp.ndarray, poly_coeffs: jnp.ndarray, degree: int, uv: jnp.ndarray) -> jnp.ndarray:
    uv = jnp.asarray(uv, dtype=float)
    if degree == 3:
        # r^3 is differentiable at a center. Form it from squared distances
        # so AD does not encounter the undefined derivative of norm(0).
        squared = jnp.sum((uv[:, None, :] - uv_model[None, :, :]) ** 2, axis=2)
        kernel = squared ** 1.5
    else:
        kernel = phs_kernel(distance_matrix(uv, uv_model), degree)
    return kernel @ weights + jnp.column_stack([jnp.ones(uv.shape[0]), uv]) @ poly_coeffs


@partial(jit, static_argnums=(3,))
def _eval_open_surface_frame_model(uv_model: jnp.ndarray, weights: jnp.ndarray, poly_coeffs: jnp.ndarray, degree: int, uv: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    uv = jnp.asarray(uv, dtype=float)
    du = uv[:, 0:1] - uv_model[None, :, 0]
    dv = uv[:, 1:2] - uv_model[None, :, 1]
    r = jnp.sqrt(du * du + dv * dv)
    dphi_degree_one = jnp.where(r > 0, 1.0 / jnp.maximum(r, jnp.finfo(float).eps), 0.0)
    dphi_higher = degree * r ** jnp.maximum(degree - 2, 0)
    dphi_higher = jnp.where(r == 0, 0.0, dphi_higher)
    dphi = lax.cond(degree == 1, lambda _: dphi_degree_one, lambda _: dphi_higher, operand=None)
    tu = (dphi * du) @ weights + poly_coeffs[1]
    tv = (dphi * dv) @ weights + poly_coeffs[2]
    n = normalize_rows(jnp.cross(tu, tv))
    return tu, tv, n


@partial(jit, static_argnums=(4,))
def _evaluate_level_set_model_impl(
    xe: jnp.ndarray,
    centers: jnp.ndarray,
    weights: jnp.ndarray,
    poly_coeffs: jnp.ndarray,
    degree: int,
    mean_potential: float,
) -> jnp.ndarray:
    r = distance_matrix(xe, centers)
    return phs_kernel(r, degree) @ weights + jnp.column_stack([jnp.ones(xe.shape[0]), xe]) @ poly_coeffs - mean_potential


@partial(jit, static_argnums=(4,))
def _evaluate_level_set_gradient_impl(
    xe: jnp.ndarray,
    centers: jnp.ndarray,
    weights: jnp.ndarray,
    poly_coeffs: jnp.ndarray,
    degree: int,
) -> jnp.ndarray:
    r = distance_matrix(xe, centers)
    delta = xe[:, None, :] - centers[None, :, :]
    radial = degree * r ** max(degree - 2, 0)
    invr = jnp.where(r > 0, 1.0 / r, 0.0)
    scaled = (radial * invr)[..., None] * delta
    return jnp.einsum("ijd,j->id", scaled, weights) + poly_coeffs[1:]


def _newton_projection_step(
    level_set: "RBFLevelSet",
    opts: dict[str, float],
    state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x, converged, stalled, iterations = state
    active = ~(converged | stalled)
    phi = level_set.evaluate(x)
    grad = level_set.evaluate_gradient(x)
    g2 = jnp.sum(grad * grad, axis=1)
    safe_g2 = jnp.where(g2 > opts["gradient_tolerance"] ** 2, g2, 1.0)
    step = -(phi[:, None] / safe_g2[:, None]) * grad
    step_norm = jnp.linalg.norm(step, axis=1, keepdims=True)
    capped = jnp.where(
        jnp.isfinite(opts["max_step_norm"]) & (step_norm > opts["max_step_norm"]),
        step * (opts["max_step_norm"] / jnp.maximum(step_norm, jnp.finfo(float).eps)),
        step,
    )
    new_x = jnp.where(active[:, None], x + capped, x)
    newly_converged = active & ((jnp.abs(phi) <= opts["value_tolerance"]) | (jnp.linalg.norm(capped, axis=1) <= opts["step_tolerance"]))
    newly_stalled = active & (g2 <= opts["gradient_tolerance"] ** 2)
    iterations = jnp.where(active, iterations + 1, iterations)
    return new_x, converged | newly_converged, stalled | newly_stalled, iterations


@partial(jit, static_argnums=(4, 9))
def _project_level_set_model_newton_impl(
    initial_points: jnp.ndarray,
    centers: jnp.ndarray,
    weights: jnp.ndarray,
    poly_coeffs: jnp.ndarray,
    degree: int,
    mean_potential: float,
    value_tolerance: float,
    step_tolerance: float,
    gradient_tolerance: float,
    max_iterations: int,
    max_step_norm: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x0 = jnp.asarray(initial_points, dtype=float)
    converged0 = jnp.zeros(x0.shape[0], dtype=bool)
    stalled0 = jnp.zeros(x0.shape[0], dtype=bool)
    iterations0 = jnp.zeros(x0.shape[0], dtype=int)

    def body_fun(_i: int, state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x, converged, stalled, iterations = state
        active = ~(converged | stalled)
        phi = jnp.nan_to_num(
            _evaluate_level_set_model_impl(x, centers, weights, poly_coeffs, degree, mean_potential),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        grad = jnp.nan_to_num(
            _evaluate_level_set_gradient_impl(x, centers, weights, poly_coeffs, degree),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        g2 = jnp.sum(grad * grad, axis=1)
        safe_g2 = jnp.where(g2 > gradient_tolerance**2, g2, 1.0)
        step = -(phi[:, None] / safe_g2[:, None]) * grad
        step_norm = jnp.linalg.norm(step, axis=1, keepdims=True)
        capped = jnp.where(
            jnp.isfinite(max_step_norm) & (step_norm > max_step_norm),
            step * (max_step_norm / jnp.maximum(step_norm, jnp.finfo(float).eps)),
            step,
        )
        capped = jnp.nan_to_num(capped, nan=0.0, posinf=0.0, neginf=0.0)
        x_next = jnp.where(active[:, None], x + capped, x)
        newly_converged = active & ((jnp.abs(phi) <= value_tolerance) | (jnp.linalg.norm(capped, axis=1) <= step_tolerance))
        newly_stalled = active & (g2 <= gradient_tolerance**2)
        iterations_next = jnp.where(active, iterations + 1, iterations)
        return x_next, converged | newly_converged, stalled | newly_stalled, iterations_next

    x, _converged, stalled, iterations = lax.fori_loop(
        0,
        max_iterations,
        body_fun,
        (x0, converged0, stalled0, iterations0),
    )
    final_phi = _evaluate_level_set_model_impl(x, centers, weights, poly_coeffs, degree, mean_potential)
    return x, final_phi, iterations, stalled


def _total_degree_multi_indices(dim: int, degree: int) -> jnp.ndarray:
    rows: list[list[int]] = []

    def rec(prefix: list[int], remaining_dim: int, remaining_degree: int) -> None:
        if remaining_dim == 1:
            rows.append(prefix + [remaining_degree])
            return
        for value in range(remaining_degree + 1):
            rec(prefix + [value], remaining_dim - 1, remaining_degree - value)

    for total in range(degree + 1):
        rec([], dim, total)
    return jnp.asarray(rows, dtype=int)


@jit
def _monomial_matrix(x: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=float)
    alpha = jnp.asarray(alpha, dtype=int)
    if alpha.shape[0] == 0:
        return jnp.zeros((x.shape[0], 0), dtype=float)
    return jnp.prod(x[:, None, :] ** alpha[None, :, :], axis=2)


@jit
def _gradient_monomial_matrix(x: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
    blocks = []
    for d in range(x.shape[1]):
        coeff = alpha[:, d]
        reduced = alpha.at[:, d].add(-1)
        valid = coeff > 0
        vals = jnp.where(valid[None, :], coeff[None, :] * _monomial_matrix(x, jnp.maximum(reduced, 0)), 0.0)
        blocks.append(vals)
    return jnp.vstack(blocks)


@partial(jit, static_argnums=(1,))
def _phs_drbfor(r: jnp.ndarray, degree: int) -> jnp.ndarray:
    if degree % 2 == 0:
        return jnp.where(r > 0, r ** (degree - 2) * (degree * jnp.log(r + 2e-16) + 1.0), 0.0)
    return jnp.where(r > 0, degree * r ** (degree - 2), 0.0)


@partial(jit, static_argnums=(2,))
def _phs_d2_same(r: jnp.ndarray, delta: jnp.ndarray, degree: int) -> jnp.ndarray:
    if degree % 2 == 0:
        return jnp.where(
            r > 0,
            r ** (degree - 2) * (1.0 + degree * jnp.log(r + 2e-16))
            + r ** (degree - 4) * delta * delta * (2.0 * degree + degree * degree * jnp.log(r + 2e-16) - 2.0 * degree * jnp.log(r + 2e-16) - 2.0),
            0.0,
        )
    return jnp.where(r > 0, degree * r ** (degree - 2) + degree * (degree - 2.0) * delta * delta * r ** (degree - 4), 0.0)


@partial(jit, static_argnums=(3,))
def _phs_d2_cross(r: jnp.ndarray, delta_a: jnp.ndarray, delta_b: jnp.ndarray, degree: int) -> jnp.ndarray:
    if degree % 2 == 0:
        return jnp.where(
            r > 0,
            r ** (degree - 4) * delta_a * delta_b * (2.0 * degree + degree * degree * jnp.log(r + 2e-16) - 2.0 * degree * jnp.log(r + 2e-16) - 2.0),
            0.0,
        )
    return jnp.where(r > 0, degree * (degree - 2.0) * delta_a * delta_b * r ** (degree - 4), 0.0)


@partial(jit, static_argnums=(2,))
def _curl_free_gram(x: jnp.ndarray, y: jnp.ndarray, degree: int) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=float)
    y = jnp.asarray(y, dtype=float)
    dim = x.shape[1]
    n = x.shape[0]
    m = y.shape[0]
    delta = x[:, None, :] - y[None, :, :]
    r = jnp.linalg.norm(delta, axis=2)
    rows = []
    for a in range(dim):
        cols = []
        for b in range(dim):
            block = _phs_d2_same(r, delta[:, :, a], degree) if a == b else _phs_d2_cross(r, delta[:, :, a], delta[:, :, b], degree)
            cols.append(block)
        rows.append(jnp.hstack(cols))
    return -jnp.vstack(rows).reshape((dim * n, dim * m))


@partial(jit, static_argnums=(5,))
def _curl_free_raw_potential(xe: jnp.ndarray, centers: jnp.ndarray, c_rbf: jnp.ndarray, c_poly: jnp.ndarray, alpha: jnp.ndarray, degree: int) -> jnp.ndarray:
    r = distance_matrix(xe, centers)
    drbf = _phs_drbfor(r, degree)
    pot = jnp.zeros(xe.shape[0], dtype=float)
    for d in range(xe.shape[1]):
        diff = xe[:, None, d] - centers[None, :, d]
        pot = pot - (drbf * diff) @ c_rbf[:, d]
    pot = pot + _monomial_matrix(xe, alpha) @ c_poly
    return pot


@partial(jit, static_argnums=(7,))
def _curl_free_zero_mean_potential(
    xe: jnp.ndarray,
    centers: jnp.ndarray,
    c_rbf: jnp.ndarray,
    c_poly: jnp.ndarray,
    res_rbf: jnp.ndarray,
    res_const: float,
    alpha: jnp.ndarray,
    degree: int,
) -> jnp.ndarray:
    raw = _curl_free_raw_potential(xe, centers, c_rbf, c_poly, alpha, degree)
    r = distance_matrix(xe, centers)
    residual = phs_kernel(r, 1) @ res_rbf + res_const
    return -(raw - residual)


@partial(jit, static_argnums=(5,))
def _curl_free_field_impl(
    xe: jnp.ndarray,
    centers: jnp.ndarray,
    c_rbf: jnp.ndarray,
    c_poly: jnp.ndarray,
    alpha: jnp.ndarray,
    degree: int,
) -> jnp.ndarray:
    cf_a = _curl_free_gram(xe, centers, degree)
    eval_loc = cf_a @ c_rbf.T.reshape(-1)
    eval_loc = eval_loc + _gradient_monomial_matrix(xe, alpha) @ c_poly
    return eval_loc.reshape((xe.shape[1], xe.shape[0])).T


@partial(jit, static_argnums=(7,))
def _curl_free_level_set_gradient_impl(
    xe: jnp.ndarray,
    centers: jnp.ndarray,
    c_rbf: jnp.ndarray,
    c_poly: jnp.ndarray,
    res_rbf: jnp.ndarray,
    res_const: float,
    alpha: jnp.ndarray,
    degree: int,
) -> jnp.ndarray:
    del res_const
    grad = -_curl_free_field_impl(xe, centers, c_rbf, c_poly, alpha, degree)
    r = distance_matrix(xe, centers)
    drbf_res = _phs_drbfor(r, 1)
    diff = xe[:, None, :] - centers[None, :, :]
    grad = grad + jnp.einsum("mn,mnd,n->md", drbf_res, diff, res_rbf)
    return grad


@partial(jit, static_argnums=(7, 12))
def _project_curl_free_level_set_newton_impl(
    initial_points: jnp.ndarray,
    centers: jnp.ndarray,
    c_rbf: jnp.ndarray,
    c_poly: jnp.ndarray,
    res_rbf: jnp.ndarray,
    res_const: float,
    alpha: jnp.ndarray,
    degree: int,
    value_tolerance: float,
    step_tolerance: float,
    gradient_tolerance: float,
    max_step_norm: float,
    max_iterations: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x0 = jnp.asarray(initial_points, dtype=float)
    converged0 = jnp.zeros(x0.shape[0], dtype=bool)
    stalled0 = jnp.zeros(x0.shape[0], dtype=bool)
    iterations0 = jnp.zeros(x0.shape[0], dtype=int)

    def body_fun(_i: int, state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x, converged, stalled, iterations = state
        active = ~(converged | stalled)
        phi = jnp.nan_to_num(
            _curl_free_zero_mean_potential(x, centers, c_rbf, c_poly, res_rbf, res_const, alpha, degree),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        grad = jnp.nan_to_num(
            _curl_free_level_set_gradient_impl(x, centers, c_rbf, c_poly, res_rbf, res_const, alpha, degree),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        g2 = jnp.sum(grad * grad, axis=1)
        safe_g2 = jnp.where(g2 > gradient_tolerance**2, g2, 1.0)
        step = -(phi[:, None] / safe_g2[:, None]) * grad
        step_norm = jnp.linalg.norm(step, axis=1, keepdims=True)
        capped = jnp.where(
            jnp.isfinite(max_step_norm) & (step_norm > max_step_norm),
            step * (max_step_norm / jnp.maximum(step_norm, jnp.finfo(float).eps)),
            step,
        )
        capped = jnp.nan_to_num(capped, nan=0.0, posinf=0.0, neginf=0.0)
        x_next = jnp.where(active[:, None], x + capped, x)
        newly_converged = active & ((jnp.abs(phi) <= value_tolerance) | (jnp.linalg.norm(capped, axis=1) <= step_tolerance))
        newly_stalled = active & (g2 <= gradient_tolerance**2)
        iterations_next = jnp.where(active, iterations + 1, iterations)
        return x_next, converged | newly_converged, stalled | newly_stalled, iterations_next

    x, _converged, stalled, iterations = lax.fori_loop(
        0,
        max_iterations,
        body_fun,
        (x0, converged0, stalled0, iterations0),
    )
    final_phi = _curl_free_zero_mean_potential(x, centers, c_rbf, c_poly, res_rbf, res_const, alpha, degree)
    return x, final_phi, iterations, stalled


@partial(jit, static_argnums=(3,))
def _solve_zero_mean_curl_free_potential_impl(
    x: jnp.ndarray,
    nr: jnp.ndarray,
    alpha: jnp.ndarray,
    spline_degree: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    dim = x.shape[1]
    n = x.shape[0]
    cf_a = _curl_free_gram(x, x, spline_degree)
    cf_p = _gradient_monomial_matrix(x, alpha)
    reg = 1e-12 * jnp.maximum(1.0, jnp.abs(cf_a).max(initial=0.0))
    a = jnp.block([[cf_a + reg * jnp.eye(dim * n), cf_p], [cf_p.T, jnp.zeros((alpha.shape[0], alpha.shape[0]))]])
    rhs = jnp.concatenate([nr.T.reshape(-1), jnp.zeros(alpha.shape[0])])
    coeffs = jnp.linalg.solve(a, rhs)
    c_rbf = coeffs[: dim * n].reshape(dim, n).T
    c_poly = coeffs[dim * n :]
    pot = _curl_free_raw_potential(x, x, c_rbf, c_poly, alpha, spline_degree)
    r = distance_matrix(x, x)
    ar = phs_kernel(r, 1)
    pr = jnp.ones((n, 1), dtype=float)
    a_res = jnp.block([[ar, pr], [pr.T, jnp.zeros((1, 1))]])
    res = jnp.linalg.solve(a_res, jnp.concatenate([pot, jnp.zeros(1)]))
    return c_rbf, c_poly, res[:n], res[n]


@dataclass
class RBFLevelSet:
    n: int = 0
    ell: int = 1
    dim: int = 0
    m_spline_degree: int = 3
    npoly: int = 0
    xd: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    nrd: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    ls_xd: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    mean_potential: float = 0.0
    centers: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    values: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    weights: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    poly_coeffs: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    mode: str = "scalar_offset"
    cfi_centers: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    cfi_rbf_coeffs: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    cfi_poly_coeffs: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    cfi_res_rbf_coeffs: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    cfi_res_constant: float = 0.0
    cfi_multi_indices: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0), dtype=int))
    cfpu_model: object = None

    def build_level_set_from_cfpu(self,segments,x,nr,*,options=None):
        from .cfpu import build_cfpu,CFPUOptions
        self.cfpu_model=build_cfpu(segments,x,nr,options=options or CFPUOptions())
        self.mode='cfpu'; self.xd=jnp.asarray(x); self.nrd=jnp.asarray(nr)
        self.n=len(x); self.dim=self.xd.shape[1]

    def build_level_set_from_cfi(self, x: jnp.ndarray, nr: jnp.ndarray, spline_degree: int = 3) -> None:
        x = jnp.asarray(x, dtype=float)
        nr = normalize_rows(jnp.asarray(nr, dtype=float))
        self.dim = x.shape[1]
        self.n = x.shape[0]
        self.m_spline_degree = spline_degree
        self.npoly = self.dim + 1
        self.xd = x
        self.nrd = nr
        self.ls_xd = jnp.zeros(self.n, dtype=float)
        self._build_zero_mean_curl_free_potential(x, nr, spline_degree)

    def _build_scalar_offset_level_set(self, x: jnp.ndarray, nr: jnp.ndarray, spline_degree: int) -> None:
        sep = self._estimate_offset_distance(x)
        inside = x - sep * nr
        outside = x + sep * nr
        self.centers = jnp.vstack([x, inside, outside])
        self.values = jnp.concatenate([jnp.zeros(self.n), sep * jnp.ones(self.n), -sep * jnp.ones(self.n)])
        nc = self.centers.shape[0]
        p = jnp.column_stack([jnp.ones(nc), self.centers])
        r = distance_matrix(self.centers, self.centers)
        k = phs_kernel(r, self.m_spline_degree)
        reg = 1e-12 * jnp.maximum(1.0, jnp.abs(k).max(initial=0.0))
        a = jnp.block([[k + reg * jnp.eye(nc), p], [p.T, jnp.zeros((self.dim + 1, self.dim + 1))]])
        rhs = jnp.concatenate([self.values, jnp.zeros(self.dim + 1)])
        coeffs = jnp.linalg.solve(a, rhs)
        self.weights = coeffs[:nc]
        self.poly_coeffs = coeffs[nc:]
        self.mean_potential = float(self.evaluate(x).mean())
        self.mode = "scalar_offset"

    def _build_zero_mean_curl_free_potential(self, x: jnp.ndarray, nr: jnp.ndarray, spline_degree: int) -> None:
        dim = int(x.shape[1])
        alpha = _total_degree_multi_indices(dim, 2)[1:]
        c_rbf, c_poly, res_rbf, res_constant = _solve_zero_mean_curl_free_potential_impl(x, nr, alpha, spline_degree)
        self.mode = "curl_free_potential"
        self.cfi_centers = x
        self.cfi_rbf_coeffs = c_rbf
        self.cfi_poly_coeffs = c_poly
        self.cfi_res_rbf_coeffs = res_rbf
        self.cfi_res_constant = float(res_constant)
        self.cfi_multi_indices = alpha
        self.centers = x
        self.weights = c_rbf.reshape(-1)
        self.poly_coeffs = c_poly

    def evaluate(self, xe: jnp.ndarray) -> jnp.ndarray:
        if self.mode=='cfpu':
            result=self.cfpu_model.evaluate(xe)
            if not isinstance(result,jax.core.Tracer) and not np.isfinite(np.asarray(result)).all():
                raise ValueError('CFPU far-field sign unresolved; refine geometry quadrature/cover')
            return result
        if self.mode == "curl_free_potential":
            return _curl_free_zero_mean_potential(
                jnp.asarray(xe, dtype=float),
                self.cfi_centers,
                self.cfi_rbf_coeffs,
                self.cfi_poly_coeffs,
                self.cfi_res_rbf_coeffs,
                self.cfi_res_constant,
                self.cfi_multi_indices,
                self.m_spline_degree,
            )
        return self.evaluate_model(self.get_evaluation_model(), xe)

    def evaluate_gradient(self, xe: jnp.ndarray) -> jnp.ndarray:
        if self.mode=='cfpu': return self.cfpu_model.gradient(xe)
        if self.mode == "curl_free_potential":
            return _curl_free_level_set_gradient_impl(
                jnp.asarray(xe, dtype=float),
                self.cfi_centers,
                self.cfi_rbf_coeffs,
                self.cfi_poly_coeffs,
                self.cfi_res_rbf_coeffs,
                self.cfi_res_constant,
                self.cfi_multi_indices,
                self.m_spline_degree,
            )
        xe = jnp.asarray(xe, dtype=float)
        return _evaluate_level_set_gradient_impl(
            xe,
            self.centers,
            self.weights,
            self.poly_coeffs,
            self.m_spline_degree,
        )

    def project_to_surface_newton(self, initial_points: jnp.ndarray, options: dict[str, float] | None = None) -> dict[str, jnp.ndarray]:
        opts = {
            "value_tolerance": 1e-12,
            "step_tolerance": 1e-12,
            "gradient_tolerance": 1e-14,
            "max_step_norm": jnp.inf,
            "max_iterations": 20,
        }
        if options:
            opts.update(options)
        if self.mode=='cfpu':
            def step(_,state):
                x,counts=state
                value=self.cfpu_model.evaluate(x); gradient=self.cfpu_model.gradient(x)
                norm2=jnp.sum(gradient*gradient,axis=1)
                active=(jnp.abs(value)>opts['value_tolerance']) & (norm2>opts['gradient_tolerance']**2)
                displacement=value[:,None]*gradient/jnp.maximum(norm2[:,None],opts['gradient_tolerance']**2)
                length=jnp.linalg.norm(displacement,axis=1)
                displacement*=jnp.minimum(1.,opts['max_step_norm']/jnp.maximum(length,1e-300))[:,None]
                return x-jnp.where(active[:,None],displacement,0.),counts+active
            x,iterations=lax.fori_loop(0,int(opts['max_iterations']),step,
                (jnp.asarray(initial_points,float),jnp.zeros(len(initial_points),dtype=int)))
            values=self.evaluate(x)
            return dict(points=x,level_set_values=values,iterations=iterations,
                converged=jnp.abs(values)<=opts['value_tolerance'],stalled=jnp.abs(values)>opts['value_tolerance'])
        if self.mode == "curl_free_potential":
            x, final_phi, iterations, stalled = _project_curl_free_level_set_newton_impl(
                initial_points,
                self.cfi_centers,
                self.cfi_rbf_coeffs,
                self.cfi_poly_coeffs,
                self.cfi_res_rbf_coeffs,
                self.cfi_res_constant,
                self.cfi_multi_indices,
                int(self.m_spline_degree),
                float(opts["value_tolerance"]),
                float(opts["step_tolerance"]),
                float(opts["gradient_tolerance"]),
                float(opts["max_step_norm"]),
                int(opts["max_iterations"]),
            )
            return {
                "points": x,
                "level_set_values": final_phi,
                "iterations": iterations,
                "converged": jnp.abs(final_phi) <= opts["value_tolerance"],
                "stalled": stalled,
            }
        x, final_phi, iterations, stalled = _project_level_set_model_newton_impl(
            initial_points,
            self.centers,
            self.weights,
            self.poly_coeffs,
            int(self.m_spline_degree),
            float(self.mean_potential),
            float(opts["value_tolerance"]),
            float(opts["step_tolerance"]),
            float(opts["gradient_tolerance"]),
            int(opts["max_iterations"]),
            float(opts["max_step_norm"]),
        )
        return {
            "points": x,
            "level_set_values": final_phi,
            "iterations": iterations,
            "converged": jnp.abs(final_phi) <= opts["value_tolerance"],
            "stalled": stalled,
        }

    def is_point_in_surface(self, xe: jnp.ndarray, tol: float = 1e-3) -> jnp.ndarray:
        return (self.evaluate(xe) >= 0.5 * tol).astype(jnp.uint32)

    def is_point_outside_surface(self, xe: jnp.ndarray, tol: float = 1e-3) -> jnp.ndarray:
        return (self.evaluate(xe) <= -0.5 * tol).astype(jnp.uint32)

    def get_evaluation_model(self) -> dict[str, jnp.ndarray | float | int]:
        if self.mode=='cfpu': return dict(mode='cfpu',model=self.cfpu_model)
        if self.mode == "curl_free_potential":
            return {
                "mode": self.mode,
                "centers": self.cfi_centers,
                "rbf_coeffs": self.cfi_rbf_coeffs,
                "poly_coeffs": self.cfi_poly_coeffs,
                "res_rbf_coeffs": self.cfi_res_rbf_coeffs,
                "res_constant": self.cfi_res_constant,
                "multi_indices": self.cfi_multi_indices,
                "m_spline_degree": self.m_spline_degree,
            }
        return {
            "centers": self.centers,
            "weights": self.weights,
            "poly_coeffs": self.poly_coeffs,
            "m_spline_degree": self.m_spline_degree,
            "mean_potential": self.mean_potential,
        }

    @staticmethod
    def evaluate_model(model: dict[str, jnp.ndarray | float | int], xe: jnp.ndarray) -> jnp.ndarray:
        xe = jnp.asarray(xe, dtype=float)
        if model.get('mode')=='cfpu': return model['model'].evaluate(xe)
        if model.get("mode") == "curl_free_potential":
            return _curl_free_zero_mean_potential(
                xe,
                jnp.asarray(model["centers"], dtype=float),
                jnp.asarray(model["rbf_coeffs"], dtype=float),
                jnp.asarray(model["poly_coeffs"], dtype=float),
                jnp.asarray(model["res_rbf_coeffs"], dtype=float),
                float(model["res_constant"]),
                jnp.asarray(model["multi_indices"], dtype=int),
                int(model["m_spline_degree"]),
            )
        centers = jnp.asarray(model["centers"], dtype=float)
        weights = jnp.asarray(model["weights"], dtype=float)
        poly_coeffs = jnp.asarray(model["poly_coeffs"], dtype=float)
        degree = int(model["m_spline_degree"])
        mean_potential = float(model["mean_potential"])
        return _evaluate_level_set_model_impl(xe, centers, weights, poly_coeffs, degree, mean_potential)

    @staticmethod
    def _estimate_offset_distance(x: jnp.ndarray) -> float:
        if x.shape[0] < 2:
            return 1e-2
        d = distance_matrix(x, x)
        d = d.at[jnp.diag_indices(x.shape[0])].set(jnp.inf)
        return float(jnp.maximum(0.5 * d.min(axis=1).min(), 1e-3))


@dataclass
class EmbeddedSurface:
    data_sites: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    data_site_nrmls: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    uniform_sample_sites: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    sample_sites: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    sample_sites_s: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    uniform_nrmls: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    nrmls: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    geom_model: dict[str, object] = field(default_factory=dict)
    nd: int = 0
    n: int = 0
    surf_dim: int = 0
    sep_rad: float = jnp.nan
    level_set: RBFLevelSet = field(default_factory=RBFLevelSet)
    tangent_step: float = 1e-5
    cbox: dict[str, jnp.ndarray] = field(default_factory=dict)
    ubox: dict[str, jnp.ndarray] = field(default_factory=dict)

    def set_data_sites(self, data_sites: jnp.ndarray) -> None:
        self.data_sites = jnp.asarray(data_sites, dtype=float)
        self.nd = self.data_sites.shape[0]
        self.surf_dim = self.data_sites.shape[1] - 1

    def set_sample_sites(self, sample_sites: jnp.ndarray) -> None:
        self.sample_sites = jnp.asarray(sample_sites, dtype=float)
        self.n = self.sample_sites.shape[0]

    def build_closed_geometric_model_ps(self, dim: int, rad: float, nb: int, ne: int | None = None, method: int = 1, supersample_fac: int = 2) -> None:
        self.sep_rad = rad
        self.nd = min(nb, self.data_sites.shape[0])
        data = self.data_sites[: self.nd]
        degree = 5
        ntarget = ne if ne is not None else self._estimate_evaluation_count(dim, rad, True)
        if dim == 2:
            theta, weights, data_normals, ptss, nr = _build_closed_curve_geometric_model_ps_impl(
                data,
                int(ntarget),
                int(method),
                int(supersample_fac),
                int(degree),
            )
            self.geom_model = {"type": "closed-curve-sbf", "degree": degree, "theta": theta, "weights": weights}
            self.data_site_nrmls = data_normals
            self.sample_sites_s = ptss
            curve_length = jnp.linalg.norm(jnp.vstack([ptss[1:], ptss[:1]]) - ptss, axis=1).sum()
            target_spacing = max(float(rad), float(jnp.finfo(float).eps))
            target_count = max(2, round(float(curve_length) / target_spacing))
            keep_inds = resample_closed_curve_by_arc_length(ptss, target_count).astype(int)
            self.sample_sites = ptss[keep_inds]
            self.nrmls = nr[keep_inds]
            self.uniform_sample_sites = self.sample_sites
            self.uniform_nrmls = self.nrmls
        elif dim == 3:
            center, unit_centers, weights, ptss, nr, keep = _build_closed_surface_geometric_model_ps_impl(
                data,
                int(ntarget),
                int(method),
                int(supersample_fac),
                int(degree),
                float(rad),
                float(self.tangent_step),
            )
            self.geom_model = {"type": "closed-surface-sbf", "degree": degree, "center": center, "unit_centers": unit_centers, "weights": weights}
            self.sample_sites_s = ptss
            self.sample_sites = ptss[keep]
            self.nrmls = nr[keep]
            self.uniform_sample_sites = self.sample_sites
            self.uniform_nrmls = self.nrmls
        else:
            raise ValueError("unsupported dimension")
        self.n = self.sample_sites.shape[0]

    def build_geometric_model_ps(
        self,
        dim: int,
        rad: float | jnp.ndarray,
        nb: int,
        ne: int | None = None,
        method: int = 1,
        supersample_fac: int = 2,
    ) -> None:
        rad_arr = jnp.asarray(rad, dtype=float).reshape(-1)
        anisotropic = rad_arr.size > 1
        min_rad = float(jnp.min(rad_arr)) if anisotropic else float(rad_arr[0])
        self.sep_rad = min_rad
        self.nd = min(nb, self.data_sites.shape[0])
        data = self.data_sites[: self.nd]
        # Both open curves and surface patches use affine augmentation.
        # Cubic PHS is compatible with that space for either spacing mode.
        degree = 3
        ntarget = ne if ne is not None else self._estimate_evaluation_count(dim, min_rad, False)
        if dim == 2:
            u, weights, poly_coeffs, pts, nr, keep, keep_uniform = _build_open_curve_geometric_model_ps_impl(
                data,
                int(ntarget),
                int(method),
                int(supersample_fac),
                int(degree),
                rad_arr,
                float(min_rad),
                bool(anisotropic),
            )
            self.geom_model = {"type": "open-curve-rbf", "degree": degree, "u": u, "rbf_weights": weights, "poly_coeffs": poly_coeffs}
            self.sample_sites = pts[keep]
            self.nrmls = nr[keep]
            self.uniform_sample_sites = pts[keep_uniform]
            self.uniform_nrmls = nr[keep_uniform]
        elif dim == 3:
            uv = build_planar_parametric_nodes_2d(data, self.nd)
            _, origin, basis = project_to_best_fit_plane(data)
            candidate_count = max(int(supersample_fac) * int(ntarget), int(ntarget)) if method == 1 else int(ntarget)
            uv_eval = build_planar_parametric_eval_nodes_2d(data, candidate_count)
            weights, poly_coeffs, pts, nr, keep, keep_uniform = _build_open_surface_geometric_model_ps_impl(
                data,
                uv,
                uv_eval,
                rad_arr,
                int(degree),
                float(min_rad),
                origin,
                basis,
                int(method),
                bool(anisotropic),
            )
            self.geom_model = {"type": "surface-patch-rbf", "degree": degree, "uv": uv, "rbf_weights": weights, "poly_coeffs": poly_coeffs, "origin": origin, "basis": basis}
            self.sample_sites = pts[keep]
            self.nrmls = nr[keep]
            self.uniform_sample_sites = pts[keep_uniform]
            self.uniform_nrmls = nr[keep_uniform]
        else:
            raise ValueError("unsupported dimension")
        self.n = self.sample_sites.shape[0]

    def build_level_set_from_geometric_model(self, lambdas: jnp.ndarray | None = None) -> None:
        if lambdas is None or len(jnp.atleast_1d(lambdas)) == 0:
            pts = self.uniform_sample_sites
            nr = self.uniform_nrmls
        else:
            vals = jnp.asarray(lambdas, dtype=float)
            if self.geom_model["type"] == "closed-curve-sbf":
                pts = self._eval_closed_curve(vals)
                _, nr = self._eval_closed_curve_frame(vals)
            elif self.geom_model["type"] == "closed-surface-sbf":
                pts = self._eval_closed_surface(vals)
                _, _, nr = self._eval_closed_surface_frame(vals)
            elif self.geom_model["type"] in ("parametric-pu", "spherical-pu"):
                pts = self.evaluate_parametric(vals)
                nr = self.evaluate_parametric_frame(vals)[-1]
            elif self.surf_dim == 1:
                pts = self._eval_open_curve(vals)
                _, nr = self._eval_open_curve_frame(vals)
            else:
                pts = self._eval_open_surface(vals)
                _, _, nr = self._eval_open_surface_frame(vals)
        self.level_set = RBFLevelSet()
        self.level_set.build_level_set_from_cfi(pts, nr)

    def compute_bounding_box(self) -> None:
        self.cbox = pca_oriented_bounding_box(self.sample_sites)

    def compute_uniform_bounding_box(self) -> None:
        self.ubox = pca_oriented_bounding_box(self.uniform_sample_sites)

    def flip_normals(self) -> None:
        self.nrmls = -self.nrmls
        self.uniform_nrmls = -self.uniform_nrmls
        self.data_site_nrmls = -self.data_site_nrmls
        if self.geom_model.get('type') in ('parametric-pu','spherical-pu'):
            self.geom_model['normal_sign']=-self.geom_model.get('normal_sign',1.)

    def get_sample_sites(self) -> jnp.ndarray:
        return self.sample_sites

    def get_uniform_sample_sites(self) -> jnp.ndarray:
        return self.uniform_sample_sites

    def get_nrmls(self) -> jnp.ndarray:
        return self.nrmls

    def get_uniform_nrmls(self) -> jnp.ndarray:
        return self.uniform_nrmls

    def get_bounding_box(self) -> jnp.ndarray:
        return self.cbox.get("p", jnp.zeros((0, self.sample_sites.shape[1] if self.sample_sites.size else 0)))

    def get_uniform_bounding_box(self) -> jnp.ndarray:
        return self.ubox.get("p", jnp.zeros((0, self.sample_sites.shape[1] if self.sample_sites.size else 0)))

    def get_level_set(self) -> RBFLevelSet:
        return self.level_set

    def evaluate_parametric(self, params: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the fitted parametric geometry through JIT/device kernels."""

        geom_type = self.geom_model.get("type")
        if geom_type=='affine-face':
            return jnp.asarray(self.geom_model['origin'])+jnp.asarray(params)@jnp.asarray(self.geom_model['basis']).T
        if geom_type == "closed-curve-sbf":
            return self._eval_closed_curve(params)
        if geom_type == "closed-surface-sbf":
            return self._eval_closed_surface(params)
        if geom_type == "open-curve-rbf":
            return self._eval_open_curve(params)
        if geom_type == "surface-patch-rbf":
            return self._eval_open_surface(params)
        if geom_type in ("parametric-pu", "spherical-pu"):
            return self.geom_model["model"].evaluate(params)
        raise ValueError("parametric geometry model has not been built")

    def evaluate_parametric_frame(self, params: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
        """Evaluate tangents/normals for the fitted parametric geometry."""

        geom_type = self.geom_model.get("type")
        if geom_type=='affine-face':
            basis=jnp.asarray(self.geom_model['basis']); normal=jnp.asarray(self.geom_model['normal'])
            return (jnp.broadcast_to(basis[:,0],(len(params),3)),jnp.broadcast_to(basis[:,1],(len(params),3)),
                    jnp.broadcast_to(normal,(len(params),3)))
        if geom_type == "closed-curve-sbf":
            return self._eval_closed_curve_frame(params)
        if geom_type == "closed-surface-sbf":
            return self._eval_closed_surface_frame(params)
        if geom_type == "open-curve-rbf":
            return self._eval_open_curve_frame(params)
        if geom_type == "surface-patch-rbf":
            return self._eval_open_surface_frame(params)
        if geom_type in ("parametric-pu", "spherical-pu"):
            jac,nr=self.geom_model["model"].frame(params)
            nr=nr*self.geom_model.get('normal_sign',1.)
            return (jac[:,:,0],jac[:,:,1],nr) if self.surf_dim==2 else (jac[:,:,0],nr)
        raise ValueError("parametric geometry model has not been built")

    def build_parametric_pu_model(self, parameters, *, outlines=None, radius=None,
            validation_parameters=None, validation_positions=None,
            validation_normals=None, options=None, seed=0, backend="auto",
            supersample_fac=2,initial_cells=None):
        """Fit a PU coordinate model; optionally supersample its actual footprint.

        An explicit parameter domain avoids guessing a face from its convex hull.
        The physical feature graph remains the responsibility of the hierarchy.
        """
        from .parametric_pu import fit_parametric_pu,PUOptions
        model=fit_parametric_pu(parameters,self.data_sites,
            validation_parameters=validation_parameters,validation_positions=validation_positions,
            validation_normals=validation_normals,options=options or PUOptions(),
            initial_cells=initial_cells,
            parameter_bounds=None if outlines is None else (jnp.asarray(outlines[0]).min(axis=0),jnp.asarray(outlines[0]).max(axis=0)))
        self.geom_model=dict(type="parametric-pu",degree=3,model=model,
            polynomial_basis_terms=self.surf_dim+1,input_parameters=jnp.asarray(parameters),
            outlines=outlines,normal_sign=1.)
        self.geom_model['fit_context']=dict(outlines=outlines,radius=radius,
            validation_parameters=validation_parameters,validation_positions=validation_positions,
            validation_normals=validation_normals,options=options,seed=seed,backend=backend,supersample_fac=supersample_fac)
        if radius is None:
            # Bounding-box PCA needs at least the ambient dimension in rows.
            # Fit-only builds avoid the full node cloud but retain enough small
            # evaluation samples for the geometry object's metadata.
            params=jnp.asarray(parameters)[:self.surf_dim+1]
        else:
            if outlines is None:
                raise ValueError("PU sampling requires a parameter footprint")
            from kernelpack.nodes.hierarchical import sample_parametric_face,sample_parametric_curve
            if self.surf_dim==1:
                params=sample_parametric_curve(model,outlines,float(radius),supersample_fac=supersample_fac)
            else:
                params=sample_parametric_face(model,outlines,float(radius),seed=seed,
                    backend=backend,supersample_fac=supersample_fac)
            self.sep_rad=float(radius)
        self.set_sample_sites(model.evaluate(params))
        _,self.nrmls=model.frame(params)
        if not bool(jnp.all(jnp.isfinite(self.sample_sites))) or not bool(jnp.all(jnp.isfinite(self.nrmls))):
            raise ValueError("PU model does not cover the physical parameter footprint")
        self.uniform_sample_sites=self.sample_sites; self.uniform_nrmls=self.nrmls
        self.compute_bounding_box()

    def build_spherical_pu_model(self, data, *, radius=None, seed=0, supersample_fac=2, initial_cover=None):
        """Fit a closed piece over S^{d-1} by the spherical-cap partition of unity (spherical_pu.py) and,
        given a radius, supersample it; piecewise assembly thins the samples in physical space.

        data: spherical_pu.ClosedSurfaceData. The piece has no trimming loops (outlines=None), hence no
        features. Its normals point away from the center of the parameterization; flip_normals reverses them.
        """
        from .spherical_pu import fit_spherical_pu, sample_sphere_piece
        self.set_data_sites(data.positions)
        model = fit_spherical_pu(data.directions, data.positions,
            validation_directions=data.validation_directions, validation_positions=data.validation_positions,
            validation_normals=data.validation_normals, options=data.options, initial_cover=initial_cover)
        self.geom_model = dict(type="spherical-pu", degree=3, model=model, outlines=None, normal_sign=1.,
            input_parameters=jnp.asarray(data.directions), data=data,
            fit_context=dict(radius=radius, seed=seed, supersample_fac=supersample_fac))
        if radius is None:
            params = jnp.asarray(data.directions)[: self.surf_dim + 2]
        else:
            params = sample_sphere_piece(model, float(radius), seed=seed, supersample_fac=supersample_fac)
            self.sep_rad = float(radius)
        self.set_sample_sites(model.evaluate(params))
        _, self.nrmls = model.frame(params)
        if not bool(jnp.all(jnp.isfinite(self.sample_sites))) or not bool(jnp.all(jnp.isfinite(self.nrmls))):
            raise ValueError("spherical PU model produced nonfinite samples")
        self.uniform_sample_sites = self.sample_sites; self.uniform_nrmls = self.nrmls
        self.compute_bounding_box()

    def get_n(self) -> int:
        return self.n

    def _estimate_evaluation_count(self, dim: int, rad: float, closed: bool) -> int:
        x = self.data_sites[: self.nd]
        if dim == 2:
            if x.shape[0] < 2:
                return 8
            dx = jnp.diff(jnp.vstack([x, x[:1]]) if closed else x, axis=0)
            measure = jnp.linalg.norm(dx, axis=1).sum()
            return max(int(jnp.ceil(measure / max(rad, float(jnp.finfo(float).eps)))), 8)
        mins = x.min(axis=0)
        maxs = x.max(axis=0)
        ext = jnp.maximum(maxs - mins, jnp.finfo(float).eps)
        area = 2.0 * (ext[0] * ext[1] + ext[0] * ext[2] + ext[1] * ext[2])
        if not closed:
            area *= 0.5
        return max(int(jnp.ceil(area / max(rad * rad, float(jnp.finfo(float).eps)))), 16)

    def _eval_closed_curve(self, t: jnp.ndarray) -> jnp.ndarray:
        return _eval_closed_curve_model(
            jnp.asarray(self.geom_model["theta"], dtype=float),
            jnp.asarray(self.geom_model["weights"], dtype=float),
            int(self.geom_model["degree"]),
            t,
        )

    def _eval_closed_curve_frame(self, t: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return _eval_closed_curve_frame_model(
            jnp.asarray(self.geom_model["theta"], dtype=float),
            jnp.asarray(self.geom_model["weights"], dtype=float),
            int(self.geom_model["degree"]),
            t,
        )

    def _eval_open_curve(self, u: jnp.ndarray) -> jnp.ndarray:
        return _eval_open_curve_model(
            jnp.asarray(self.geom_model["u"], dtype=float),
            jnp.asarray(self.geom_model["rbf_weights"], dtype=float),
            jnp.asarray(self.geom_model["poly_coeffs"], dtype=float),
            int(self.geom_model["degree"]),
            u,
        )

    def _eval_open_curve_frame(self, u: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return _eval_open_curve_frame_model(
            jnp.asarray(self.geom_model["u"], dtype=float),
            jnp.asarray(self.geom_model["rbf_weights"], dtype=float),
            jnp.asarray(self.geom_model["poly_coeffs"], dtype=float),
            int(self.geom_model["degree"]),
            u,
        )

    def _eval_closed_surface(self, uv: jnp.ndarray) -> jnp.ndarray:
        return _eval_closed_surface_model(
            jnp.asarray(self.geom_model["unit_centers"], dtype=float),
            jnp.asarray(self.geom_model["weights"], dtype=float),
            int(self.geom_model["degree"]),
            uv,
        )

    def _eval_closed_surface_frame(self, uv: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return _eval_closed_surface_frame_model(
            jnp.asarray(self.geom_model["unit_centers"], dtype=float),
            jnp.asarray(self.geom_model["weights"], dtype=float),
            int(self.geom_model["degree"]),
            float(self.tangent_step),
            uv,
        )

    def _eval_open_surface(self, uv: jnp.ndarray) -> jnp.ndarray:
        return _eval_open_surface_model(
            jnp.asarray(self.geom_model["uv"], dtype=float),
            jnp.asarray(self.geom_model["rbf_weights"], dtype=float),
            jnp.asarray(self.geom_model["poly_coeffs"], dtype=float),
            int(self.geom_model["degree"]),
            uv,
        )

    def _eval_open_surface_frame(self, uv: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return _eval_open_surface_frame_model(
            jnp.asarray(self.geom_model["uv"], dtype=float),
            jnp.asarray(self.geom_model["rbf_weights"], dtype=float),
            jnp.asarray(self.geom_model["poly_coeffs"], dtype=float),
            int(self.geom_model["degree"]),
            uv,
        )


@dataclass
class PiecewiseSmoothEmbeddedSurface:
    segments: list[EmbeddedSurface] = field(default_factory=list)
    xb: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    xb_uniform: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    nrmls: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    nrmls_uniform: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    segment_map: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=int))
    corner_flags: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=float))
    feature_sites: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 3)))
    feature_normals: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 3)))
    feature_priorities: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0, dtype=int))
    level_set: RBFLevelSet = field(default_factory=RBFLevelSet)
    cbox: dict[str, jnp.ndarray] = field(default_factory=dict)
    ubox: dict[str, jnp.ndarray] = field(default_factory=dict)
    sep_rad: float = 0.
    assembly_options: dict = field(default_factory=dict)
    seed: int = 0   # seed of the random samplers that built this surface (feature enrichment reuses it)
    derived_features: tuple = ()

    def set_segments(self, segments: list[EmbeddedSurface]) -> None:
        self.segments = list(segments)

    def generate_piecewise_smooth_surface_by_segment(
        self,
        bdry_segments: list[jnp.ndarray],
        flip_normal: list[bool] | jnp.ndarray,
        radius: float | jnp.ndarray,
        method: int = 1,
        supersample_fac: int = 2,
        _mode: int = 2,
        bdry_mode: int = 0,
        smooth_normals: bool = False,
        smooth_neighborhood: int = 0,
    ) -> None:
        from .parametric_pu import ParametricSurfaceData
        from .spherical_pu import ClosedSurfaceData
        self.segments = []
        raw=[item.positions if isinstance(item,(ParametricSurfaceData,ClosedSurfaceData)) else item for item in bdry_segments]
        radii = _normalize_piecewise_radii(radius, jnp.asarray(raw[0]).shape[1])
        min_radius = float(jnp.min(radii))
        build_radius: float | jnp.ndarray = radii if int(bdry_mode) != 0 and radii.size > 1 else min_radius
        dim = jnp.asarray(raw[0]).shape[1]
        for index, (seg_pts, flip) in enumerate(zip(bdry_segments, flip_normal)):
            if isinstance(seg_pts,ClosedSurfaceData):
                if not bool(jnp.all(radii==radii[0])):
                    raise ValueError('PU pieces currently require isotropic physical sampling')
                seg=EmbeddedSurface()
                seg.build_spherical_pu_model(seg_pts,radius=min_radius,seed=int(self.seed)+index,supersample_fac=supersample_fac)
                if flip: seg.flip_normals()
                self.segments.append(seg)
            elif isinstance(seg_pts,ParametricSurfaceData):
                if not bool(jnp.all(radii==radii[0])):
                    raise ValueError('PU pieces currently require isotropic physical sampling')
                seg=EmbeddedSurface(); seg.set_data_sites(seg_pts.positions)
                seg.build_parametric_pu_model(seg_pts.parameters,outlines=seg_pts.outlines,radius=min_radius,
                    validation_parameters=seg_pts.validation_parameters,validation_positions=seg_pts.validation_positions,
                    validation_normals=seg_pts.validation_normals,options=seg_pts.options,supersample_fac=supersample_fac)
                if flip: seg.flip_normals()
                self.segments.append(seg)
            else:
                self.segments.append(self._build_segment(jnp.asarray(seg_pts, dtype=float), bool(flip), dim, min_radius, build_radius, method, supersample_fac))
        self._assemble_boundary(min_radius, smooth_normals, smooth_neighborhood, use_uniform=bool(int(bdry_mode) != 0 and radii.size > 1))

    def generate_piecewise_smooth_surface_by_segment_with_omission(
        self,
        bdry_segments: list[jnp.ndarray],
        omit: list[bool] | jnp.ndarray,
        flip_normal: list[bool] | jnp.ndarray,
        radius: float | jnp.ndarray,
        method: int = 1,
        supersample_fac: int = 2,
        _mode: int = 2,
        bdry_mode: int = 0,
        smooth_normals: bool = False,
        smooth_neighborhood: int = 0,
    ) -> None:
        radii = _normalize_piecewise_radii(radius, jnp.asarray(bdry_segments[0]).shape[1])
        min_radius = float(jnp.min(radii))
        build_radius: float | jnp.ndarray = radii if int(bdry_mode) != 0 and radii.size > 1 else min_radius
        if len(self.segments) != len(bdry_segments):
            padded = [EmbeddedSurface() for _ in bdry_segments]
            for i, seg in enumerate(self.segments[: len(padded)]):
                padded[i] = seg
            self.segments = padded
        for idx, (seg_pts, skip, flip) in enumerate(zip(bdry_segments, omit, flip_normal)):
            if not bool(skip):
                pts = jnp.asarray(seg_pts, dtype=float)
                self.segments[idx] = self._build_segment(pts, bool(flip), int(pts.shape[1]), min_radius, build_radius, method, supersample_fac)
        self._assemble_boundary(min_radius, smooth_normals, smooth_neighborhood, use_uniform=bool(int(bdry_mode) != 0 and radii.size > 1))

    def _build_segment(
        self,
        seg_pts: jnp.ndarray,
        flip: bool,
        dim: int,
        min_radius: float,
        build_radius: float | jnp.ndarray,
        method: int,
        supersample_fac: int,
    ) -> EmbeddedSurface:
        seg = EmbeddedSurface()
        seg.set_data_sites(seg_pts)
        seg.set_sample_sites(seg_pts)
        seg.compute_bounding_box()
        box = seg.get_bounding_box()
        ext = box.max(axis=0) - box.min(axis=0)
        if dim == 2:
            bdry_size = 2.0 * (ext[0] + ext[1])
            seg_n = max(8, round(float(bdry_size) / min_radius))
        else:
            bdry_size = 2.0 * (ext[0] * ext[1] + ext[0] * ext[2] + ext[1] * ext[2])
            seg_n = max(16, round(float(bdry_size) / max(min_radius * min_radius, float(jnp.finfo(float).eps))))
        seg.build_geometric_model_ps(dim, build_radius, len(seg_pts), seg_n, method, supersample_fac)
        if flip:
            seg.flip_normals()
        return seg

    def _assemble_boundary(self, radius: float, smooth_normals: bool, smooth_neighborhood: int, use_uniform: bool = False) -> None:
        self.sep_rad=float(radius)
        self.assembly_options=dict(smooth_normals=smooth_normals,smooth_neighborhood=smooth_neighborhood,use_uniform=use_uniform)
        # Sites, normals, ordering and thinning stay on the device; the host only sees
        # per-piece counts (static shapes) and the validity checks' booleans.
        from kernelpack.accelerators.jax_spatial import compact_rows, row_norm
        def gather(sites_of, normals_of):
            sites = [jnp.asarray(sites_of(seg), dtype=float) for seg in self.segments]
            xb = jnp.concatenate(sites)
            nr = jnp.concatenate([jnp.asarray(normals_of(seg), dtype=float) for seg in self.segments])
            seg_map = jnp.concatenate([jnp.full(x.shape[0],index,dtype=jnp.int32)
                                       for index,x in enumerate(sites)])
            # Samples on the pieces have priority 0; features come first (below).
            return xb, nr, seg_map, jnp.zeros(xb.shape[0], dtype=float)

        xb, nr, segment_map, corner = gather(lambda s: s.get_sample_sites(), lambda s: s.get_nrmls())
        if self.feature_sites.shape[0]:
            features = jnp.asarray(self.feature_sites, dtype=float)
            normals = jnp.asarray(self.feature_normals, dtype=float)
            priorities = jnp.asarray(self.feature_priorities, dtype=int)
        else:
            # Features derived from the mapped trimming loops (geometry/features.py).
            from .features import derive_features
            features, normals, priorities = derive_features(self.segments, radius)
        self.derived_features = (features, normals, priorities)
        n_features = int(features.shape[0])
        if n_features:
            if features.shape != normals.shape or priorities.shape != (n_features,):
                raise ValueError("feature sites, normals and priorities must have matching shapes")
            if (features.shape[1] != xb.shape[1] or not bool(jnp.all(jnp.isfinite(normals)))
                    or not bool(jnp.all(jnp.isfinite(features))) or bool(jnp.any(priorities <= 0))):
                raise ValueError("features need matching dimensions, finite coordinates/normals and positive priorities")
            if bool(jnp.any(row_norm(normals) == 0)):
                raise ValueError("feature normals must be nonzero")
            # Larger feature priorities precede smaller ones, before face points (0).
            # This also makes duplicate elimination respect feature priority.
            order = jnp.argsort(-priorities, stable=True)
            feature_normals = normals[order] / row_norm(normals[order])[:, None]
            xb = jnp.concatenate([features[order], xb])
            nr = jnp.concatenate([feature_normals, nr])
            segment_map = jnp.concatenate([jnp.full(n_features,-1,dtype=jnp.int32),segment_map])
            corner = jnp.concatenate([priorities[order].astype(float), corner])
        keep = _assembled_boundary_keep_mask(xb, corner, float(radius))
        self.xb, self.nrmls, self.corner_flags = compact_rows(keep, xb, nr, corner)
        self.segment_map = compact_rows(keep, jnp.asarray(segment_map))
        if use_uniform:
            xb_uniform, nr_uniform, _, uniform_corner = gather(lambda s: s.get_uniform_sample_sites(), lambda s: s.get_uniform_nrmls())
            if n_features:
                xb_uniform = jnp.concatenate([features[order], xb_uniform])
                nr_uniform = jnp.concatenate([feature_normals, nr_uniform])
                uniform_corner = jnp.concatenate([priorities[order].astype(float), uniform_corner])
            keep_uniform = _assembled_boundary_keep_mask(xb_uniform, uniform_corner, float(radius))
            self.xb_uniform, self.nrmls_uniform = compact_rows(keep_uniform, xb_uniform, nr_uniform)
        else:
            self.xb_uniform = self.xb
            self.nrmls_uniform = self.nrmls
        if smooth_normals and smooth_neighborhood > 1:
            smoothed = self._smooth_normals(self.xb, self.nrmls, smooth_neighborhood)
            self.nrmls = jnp.where((self.corner_flags > 0)[:, None], self.nrmls, smoothed)
            if use_uniform:
                uniform_smoothed = self._smooth_normals(self.xb_uniform, self.nrmls_uniform, smooth_neighborhood)
                self.nrmls_uniform = jnp.where((compact_rows(keep_uniform, uniform_corner) > 0)[:, None],
                                               self.nrmls_uniform, uniform_smoothed)
            else:
                self.nrmls_uniform = self.nrmls

    def build_level_set(self,method=None,*,options=None) -> None:
        self.level_set = RBFLevelSet()
        if method is None:
            kinds=[seg.geom_model.get('type') for seg in self.segments]
            method='cfpu' if ('parametric-pu' in kinds or 'spherical-pu' in kinds or all(kind=='affine-face' for kind in kinds)) else 'cfi'
        if method=='cfpu':
            from .cfpu import CFPUOptions,CFPUCoverError
            from .parametric_pu import refine_cover_cells,PUOptions
            options=options or CFPUOptions(); history=[]
            for iteration in range(options.max_cover_refinements+1):
                try:
                    from .feature_sampling import feature_graded_cfpu_sites
                    fit,nr,sampling=feature_graded_cfpu_sites(self,factor=options.feature_grade_factor,
                        bandwidth=options.feature_bandwidth,seed=int(self.seed),backend='gpu')
                    self.level_set.build_level_set_from_cfpu(self.segments,fit,nr,options=options)
                    self.level_set.cfpu_model.diagnostics['shared_refinement_history']=history
                    self.level_set.cfpu_model.diagnostics['feature_sampling']=sampling
                    break
                except CFPUCoverError as error:
                    if iteration==options.max_cover_refinements or not error.parent_patches: raise
                    history.append(dict(iteration=iteration,reason=str(error),requested_parents=error.parent_patches))
                    for piece in sorted(set(p for p,_ in error.parent_patches)):
                        segment=self.segments[piece]; model=segment.geom_model
                        indices={patch for p,patch in error.parent_patches if p==piece}
                        if model['type']=='spherical-pu':
                            from .spherical_pu import split_caps
                            pu=model['model']; cover=split_caps(pu.centers,pu.betas,indices)
                            sign=model.get('normal_sign',1.); context=model['fit_context']
                            segment.build_spherical_pu_model(model['data'],initial_cover=cover,**context)
                            if sign<0: segment.flip_normals()
                        elif model['type']=='parametric-pu':
                            cells=refine_cover_cells(model['model'].cells,indices)
                            sign=model.get('normal_sign',1.); parameters=model['input_parameters']; context=model['fit_context']
                            segment.build_parametric_pu_model(parameters,initial_cells=cells,**context)
                            if sign<0: segment.flip_normals()
                        else:
                            cells=refine_cover_cells(model['cover_cells'],indices)
                            if len(cells)>PUOptions().max_patches: raise ValueError('shared affine cover exceeded max_patches')
                            model['cover_cells']=cells
                            model['cover_centers']=np.array([(a+b)/2 for a,b in cells])
                            model['cover_widths']=np.array([(b-a)*PUOptions().overlap/2 for a,b in cells])
                    self._assemble_boundary(self.sep_rad,**self.assembly_options)
        elif method=='cfi':
            self.level_set.build_level_set_from_cfi(self.xb_uniform, self.nrmls_uniform)
        else: raise ValueError('level-set method must be cfpu or cfi')

    def compute_bounding_box(self) -> None:
        self.cbox = pca_oriented_bounding_box(self.xb)

    def compute_uniform_bounding_box(self) -> None:
        self.ubox = pca_oriented_bounding_box(self.xb_uniform)

    def get_level_set(self) -> RBFLevelSet:
        return self.level_set

    def get_bdry_nodes(self) -> jnp.ndarray:
        return self.xb

    def get_uniform_bdry_nodes(self) -> jnp.ndarray:
        return self.xb_uniform

    def get_bdry_nrmls(self) -> jnp.ndarray:
        return self.nrmls

    def get_uniform_bdry_nrmls(self) -> jnp.ndarray:
        return self.nrmls_uniform

    def get_corner_flags(self) -> jnp.ndarray:
        return self.corner_flags

    def get_sample_sites(self) -> jnp.ndarray:
        return self.xb

    def get_uniform_sample_sites(self) -> jnp.ndarray:
        return self.xb_uniform

    def get_nrmls(self) -> jnp.ndarray:
        return self.nrmls

    def get_uniform_nrmls(self) -> jnp.ndarray:
        return self.nrmls_uniform

    def get_bounding_box(self) -> jnp.ndarray:
        return self.cbox.get("p", jnp.zeros((0, self.xb.shape[1] if self.xb.size else 0)))

    def get_uniform_bounding_box(self) -> jnp.ndarray:
        return self.ubox.get("p", jnp.zeros((0, self.xb_uniform.shape[1] if self.xb_uniform.size else 0)))

    def _smooth_normals(self, x: jnp.ndarray, nr: jnp.ndarray, neighborhood: int) -> jnp.ndarray:
        return _smooth_normals_impl(jnp.asarray(x, dtype=float), jnp.asarray(nr, dtype=float), int(neighborhood))


def _normalize_piecewise_radii(radius: float | jnp.ndarray, dim: int) -> jnp.ndarray:
    radii = jnp.asarray(radius, dtype=float).reshape(-1)
    if radii.size == 1:
        return jnp.full(dim, float(radii[0]), dtype=float)
    if radii.size != dim:
        raise ValueError("anisotropic radii must have one entry per coordinate dimension")
    if bool(jnp.any(radii <= 0.0)):
        raise ValueError("radii must be positive")
    return radii


def _assembled_boundary_keep_mask(x: jnp.ndarray, corner_flags: jnp.ndarray, radius: float) -> jnp.ndarray:
    """Greedy spacing thinning in priority order (larger corner flag first, then input order).

    Computed as the lexicographically first independent set of the conflict
    graph ``|x_i - x_j| < radius*(1-1e-12)`` in parallel rounds on the device;
    identical to the sequential pass in ``_assembled_boundary_keep_mask_impl``.
    """
    from kernelpack.accelerators.jax_spatial import bucket, conflict_lists, greedy_independent_set, pad_rows_device
    pts = jnp.asarray(x, dtype=float)
    n = pts.shape[0]
    if n == 0:
        return jnp.zeros(0, dtype=bool)
    if radius <= 0:
        return jnp.ones(n, dtype=bool)
    n_pad = bucket(n)
    corners = pad_rows_device(jnp.asarray(corner_flags, dtype=float).reshape(-1), n_pad, -jnp.inf)
    valid, rank = _keep_mask_rank(corners, n)
    neighbors = conflict_lists(pad_rows_device(pts, n_pad, 0.0), jnp.where(valid, float(radius), 0.0), valid,
                               scale=1.0 - 1.0e-12)
    keep, _ = greedy_independent_set(neighbors, rank, excluded=~valid)
    return keep[:n]


@jit
def _keep_mask_rank(corners, n):
    """Rank = position in the stable order of decreasing corner flag; padding (-inf) keeps its index."""
    rows = jnp.arange(corners.shape[0])
    order = jnp.argsort(-corners, stable=True)
    return rows < n, jnp.zeros(corners.shape[0], jnp.int32).at[order].set(rows.astype(jnp.int32))


def _assembled_boundary_keep_mask_warp(x: jnp.ndarray, corner_flags: jnp.ndarray, radius: float) -> jnp.ndarray:
    pts = np.asarray(x, dtype=float)
    corners = np.asarray(corner_flags, dtype=float).reshape(-1)
    keep = warp_assembled_boundary_keep_mask(pts, corners, float(radius))
    return jnp.asarray(keep)


@jit
def _assembled_boundary_keep_mask_impl(x: jnp.ndarray, corner_flags: jnp.ndarray, radius: float) -> jnp.ndarray:
    n = x.shape[0]
    idx = jnp.arange(n)
    d = distance_matrix(x, x)
    order = jnp.argsort(-corner_flags, stable=True)
    spacing_tol = radius * (1.0 - 1.0e-12)

    def spacing_body(pos, keep):
        i = order[pos]
        close = (d[i] < spacing_tol) & (idx != i)
        return jnp.where(keep[i], keep & ~close, keep)

    # One pass avoids pre-eliminating candidates in a duplicate pass and then
    # removing their winners during spacing, which leaves avoidable holes.
    return lax.fori_loop(0, n, spacing_body, jnp.ones(n, dtype=bool))


@partial(jit, static_argnums=(2,))
def _smooth_normals_impl(x: jnp.ndarray, nr: jnp.ndarray, neighborhood: int) -> jnp.ndarray:
    d = distance_matrix(x, x)
    k = min(neighborhood, x.shape[0])
    take = jnp.argsort(d, axis=1)[:, :k]
    nri = nr[take]
    ref = nri[:, :1, :]
    align = jnp.where(jnp.sum(nri * ref, axis=2, keepdims=True) < 0, -1.0, 1.0)
    return normalize_rows(jnp.sum(nri * align, axis=1))
