from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import warp as wp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    wp = None


class WarpUnavailableError(RuntimeError):
    pass


def warp_available() -> bool:
    return wp is not None


def _require_warp():
    if wp is None:  # pragma: no cover - optional dependency
        raise WarpUnavailableError(
            "NVIDIA Warp is not installed. Install kernelpack-jax with the 'warp' extra "
            "or install 'warp-lang' manually to use backend='warp'."
        )
    wp.init()
    return wp


@dataclass(frozen=True)
class _WarpDeviceConfig:
    device: str = "cuda"


@dataclass(frozen=True)
class WarpBridsonOptions:
    radius: float
    min_radius: float
    attempts: int
    seed: int
    mode: str
    boundary_refinement_fraction: float
    boundary_distance: float
    boundary_points: np.ndarray
    radius_function: object | None = None


def _warp_config() -> _WarpDeviceConfig:
    _require_warp()
    try:
        device = "cuda" if wp.is_cuda_available() else "cpu"
    except Exception:  # pragma: no cover - defensive
        device = "cpu"
    return _WarpDeviceConfig(device=device)


def _as_2d_float32(points: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("expected a 2D point cloud")
    return np.ascontiguousarray(arr)


def _pad_to_vec3(points: np.ndarray) -> np.ndarray:
    pts = _as_2d_float32(points)
    if pts.shape[1] == 3:
        return pts
    if pts.shape[1] != 2:
        raise ValueError("Warp backends currently support only 2D or 3D point clouds")
    padded = np.zeros((pts.shape[0], 3), dtype=np.float32)
    padded[:, :2] = pts
    return padded


def _vec3_from_point(point: np.ndarray) -> np.ndarray:
    padded = _pad_to_vec3(np.asarray(point, dtype=np.float32).reshape(1, -1))
    return padded[0]


if wp is not None:

    @wp.func
    def _norm2(v: wp.vec3) -> float:
        return wp.dot(v, v)


    @wp.func
    def _norm(v: wp.vec3) -> float:
        return wp.sqrt(_norm2(v))


    @wp.func
    def _sample_direction_and_radius(state: wp.uint32, dim: int, base_radius: float) -> wp.vec3:
        if dim == 2:
            theta = 2.0 * 3.141592653589793 * wp.randf(state)
            shell = base_radius * wp.sqrt(1.0 + 3.0 * wp.randf(state))
            return wp.vec3(shell * wp.cos(theta), shell * wp.sin(theta), 0.0)

        # 3D case
        u = 2.0 * wp.randf(state) - 1.0
        theta = 2.0 * 3.141592653589793 * wp.randf(state)
        shell = base_radius * wp.pow(1.0 + 7.0 * wp.randf(state), 1.0 / 3.0)
        xy = wp.sqrt(wp.max(0.0, 1.0 - u * u))
        return wp.vec3(shell * xy * wp.cos(theta), shell * xy * wp.sin(theta), shell * u)


    @wp.func
    def _in_box(p: wp.vec3, box_min: wp.vec3, box_max: wp.vec3, dim: int) -> bool:
        if p[0] < box_min[0] or p[0] > box_max[0]:
            return False
        if p[1] < box_min[1] or p[1] > box_max[1]:
            return False
        if dim == 3 and (p[2] < box_min[2] or p[2] > box_max[2]):
            return False
        return True


    @wp.func
    def _boundary_refined_radius(
        boundary_grid: wp.uint64,
        boundary_points: wp.array(dtype=wp.vec3),
        point: wp.vec3,
        search_radius: float,
        coarse_radius: float,
        refined_radius: float,
        has_boundary_refinement: int,
    ) -> float:
        if has_boundary_refinement == 0:
            return coarse_radius
        for boundary_idx in wp.hash_grid_query(boundary_grid, point, search_radius):
            if _norm(point - boundary_points[boundary_idx]) <= search_radius:
                return refined_radius
        return coarse_radius


    @wp.kernel
    def _point_to_reference_sqdist_kernel(
        points: wp.array2d(dtype=float),
        reference: wp.array(dtype=float),
        dim: int,
        out: wp.array(dtype=float),
    ):
        i = wp.tid()
        accum = float(0.0)
        for d in range(dim):
            diff = points[i, d] - reference[d]
            accum += diff * diff
        out[i] = accum


    @wp.kernel
    def _pairwise_sqdist_kernel(
        query_points: wp.array2d(dtype=float),
        points: wp.array2d(dtype=float),
        dim: int,
        num_points: int,
        out: wp.array(dtype=float),
    ):
        tid = wp.tid()
        i = tid // num_points
        j = tid - i * num_points
        accum = float(0.0)
        for d in range(dim):
            diff = query_points[i, d] - points[j, d]
            accum += diff * diff
        out[tid] = accum


    @wp.kernel
    def _count_ball_neighbors_kernel(
        query_points: wp.array(dtype=wp.vec3),
        points: wp.array(dtype=wp.vec3),
        grid: wp.uint64,
        radius: float,
        dim: int,
        counts: wp.array(dtype=int),
    ):
        i = wp.tid()
        q = query_points[i]
        radius_sq = radius * radius
        count = int(0)
        for j in wp.hash_grid_query(grid, q, radius):
            diff = q - points[j]
            dist_sq = diff[0] * diff[0] + diff[1] * diff[1]
            if dim == 3:
                dist_sq += diff[2] * diff[2]
            if dist_sq <= radius_sq:
                count += 1
        counts[i] = count


    @wp.kernel
    def _fill_ball_neighbors_kernel(
        query_points: wp.array(dtype=wp.vec3),
        points: wp.array(dtype=wp.vec3),
        grid: wp.uint64,
        radius: float,
        dim: int,
        offsets: wp.array(dtype=int),
        flat_indices: wp.array(dtype=int),
        flat_distances: wp.array(dtype=float),
    ):
        i = wp.tid()
        q = query_points[i]
        radius_sq = radius * radius
        write = offsets[i]
        for j in wp.hash_grid_query(grid, q, radius):
            diff = q - points[j]
            dist_sq = diff[0] * diff[0] + diff[1] * diff[1]
            if dim == 3:
                dist_sq += diff[2] * diff[2]
            if dist_sq <= radius_sq:
                flat_indices[write] = j
                flat_distances[write] = wp.sqrt(wp.max(float(0.0), dist_sq))
                write += 1


    @wp.kernel
    def _initialize_knn_kernel(
        indices: wp.array2d(dtype=int),
        distances: wp.array2d(dtype=float),
        k: int,
    ):
        i = wp.tid()
        for slot in range(k):
            indices[i, slot] = -1
            distances[i, slot] = 3.402823466e38


    @wp.kernel
    def _fill_knn_neighbors_kernel(
        query_points: wp.array(dtype=wp.vec3),
        points: wp.array(dtype=wp.vec3),
        grid: wp.uint64,
        radius: float,
        dim: int,
        k: int,
        indices: wp.array2d(dtype=int),
        distances: wp.array2d(dtype=float),
    ):
        i = wp.tid()
        q = query_points[i]
        radius_sq = radius * radius
        for j in wp.hash_grid_query(grid, q, radius):
            diff = q - points[j]
            dist_sq = diff[0] * diff[0] + diff[1] * diff[1]
            if dim == 3:
                dist_sq += diff[2] * diff[2]
            if dist_sq > radius_sq or dist_sq >= distances[i, k - 1]:
                continue

            slot = k - 1
            while slot > 0 and dist_sq < distances[i, slot - 1]:
                distances[i, slot] = distances[i, slot - 1]
                indices[i, slot] = indices[i, slot - 1]
                slot -= 1
            distances[i, slot] = dist_sq
            indices[i, slot] = j


    @wp.kernel
    def _assembled_boundary_duplicate_keep_kernel(
        points: wp.array(dtype=wp.vec3),
        grid: wp.uint64,
        radius: float,
        dim: int,
        num_points: int,
        keep: wp.array(dtype=int),
    ):
        radius_sq = radius * radius
        for i in range(num_points):
            if keep[i] == 0:
                continue
            p = points[i]
            for j in wp.hash_grid_query(grid, p, radius):
                if j == i or keep[j] == 0:
                    continue
                diff = p - points[j]
                dist_sq = diff[0] * diff[0] + diff[1] * diff[1]
                if dim == 3:
                    dist_sq += diff[2] * diff[2]
                if dist_sq < radius_sq:
                    keep[j] = 0


    @wp.kernel
    def _assembled_boundary_spacing_keep_kernel(
        points: wp.array(dtype=wp.vec3),
        corner_flags: wp.array(dtype=int),
        grid: wp.uint64,
        radius: float,
        dim: int,
        num_points: int,
        keep: wp.array(dtype=int),
    ):
        radius_sq = radius * radius
        for phase in range(2):
            for i in range(num_points):
                current_corner = corner_flags[i] != 0
                if phase == 0 and not current_corner:
                    continue
                if phase == 1 and current_corner:
                    continue
                if keep[i] == 0:
                    continue

                p = points[i]
                current_loses = int(0)
                for j in wp.hash_grid_query(grid, p, radius):
                    if j == i or keep[j] == 0:
                        continue
                    diff = p - points[j]
                    dist_sq = diff[0] * diff[0] + diff[1] * diff[1]
                    if dim == 3:
                        dist_sq += diff[2] * diff[2]
                    if dist_sq < radius_sq:
                        if (not current_corner) and corner_flags[j] != 0:
                            current_loses = 1
                            break
                        keep[j] = 0
                if current_loses != 0:
                    keep[i] = 0


    @wp.kernel
    def _generate_provisional_candidates_kernel(
        seed: int,
        epoch: int,
        dim: int,
        attempts: int,
        coarse_radius: float,
        refined_radius: float,
        boundary_distance: float,
        has_boundary_refinement: int,
        box_min: wp.vec3,
        box_max: wp.vec3,
        active_indices: wp.array(dtype=int),
        points: wp.array(dtype=wp.vec3),
        point_radii: wp.array(dtype=float),
        existing_grid: wp.uint64,
        boundary_grid: wp.uint64,
        boundary_points: wp.array(dtype=wp.vec3),
        provisional_points: wp.array(dtype=wp.vec3),
        provisional_radii: wp.array(dtype=float),
        provisional_valid: wp.array(dtype=int),
    ):
        tid = wp.tid()
        point_idx = active_indices[tid]
        base = points[point_idx]
        base_radius = point_radii[point_idx]

        provisional_valid[tid] = 0
        provisional_radii[tid] = 0.0
        provisional_points[tid] = base

        for attempt_idx in range(attempts):
            state = wp.rand_init(seed + epoch * 104729, tid * attempts + attempt_idx)
            offset = _sample_direction_and_radius(state, dim, base_radius)
            candidate = base + offset
            if not _in_box(candidate, box_min, box_max, dim):
                continue

            candidate_radius = _boundary_refined_radius(
                boundary_grid,
                boundary_points,
                candidate,
                boundary_distance,
                coarse_radius,
                refined_radius,
                has_boundary_refinement,
            )

            valid = int(1)
            for neighbor_idx in wp.hash_grid_query(existing_grid, candidate, coarse_radius):
                if neighbor_idx == point_idx:
                    continue
                threshold = coarse_radius
                if has_boundary_refinement != 0:
                    threshold = wp.max(candidate_radius, point_radii[neighbor_idx])
                if _norm(candidate - points[neighbor_idx]) < threshold:
                    valid = 0
                    break

            if valid != 0:
                provisional_points[tid] = candidate
                provisional_radii[tid] = candidate_radius
                provisional_valid[tid] = 1
                break


    @wp.kernel
    def _filter_provisional_candidates_kernel(
        coarse_radius: float,
        has_boundary_refinement: int,
        provisional_grid: wp.uint64,
        provisional_points: wp.array(dtype=wp.vec3),
        provisional_radii: wp.array(dtype=float),
        provisional_valid: wp.array(dtype=int),
        accepted: wp.array(dtype=int),
    ):
        tid = wp.tid()
        if provisional_valid[tid] == 0:
            accepted[tid] = 0
            return

        p = provisional_points[tid]
        r = provisional_radii[tid]
        keep = int(1)
        for other_idx in wp.hash_grid_query(provisional_grid, p, coarse_radius):
            if other_idx == tid or provisional_valid[other_idx] == 0:
                continue
            if other_idx < tid:
                threshold = coarse_radius
                if has_boundary_refinement != 0:
                    threshold = wp.max(r, provisional_radii[other_idx])
                if _norm(p - provisional_points[other_idx]) < threshold:
                    keep = 0
                    break
        accepted[tid] = keep


    @wp.kernel
    def _generate_candidate_attempts_kernel(
        seed: int,
        epoch: int,
        dim: int,
        attempts: int,
        box_min: wp.vec3,
        box_max: wp.vec3,
        active_indices: wp.array(dtype=int),
        points: wp.array(dtype=wp.vec3),
        point_radii: wp.array(dtype=float),
        attempt_points: wp.array(dtype=wp.vec3),
        attempt_valid: wp.array(dtype=int),
    ):
        tid = wp.tid()
        local_active_idx = tid // attempts
        attempt_idx = tid - local_active_idx * attempts
        point_idx = active_indices[local_active_idx]
        base = points[point_idx]
        base_radius = point_radii[point_idx]
        state = wp.rand_init(seed + epoch * 104729, tid + attempt_idx * 8191)
        offset = _sample_direction_and_radius(state, dim, base_radius)
        candidate = base + offset
        if _in_box(candidate, box_min, box_max, dim):
            attempt_points[tid] = candidate
            attempt_valid[tid] = 1
        else:
            attempt_points[tid] = base
            attempt_valid[tid] = 0


def _warp_point_to_reference_sqdist(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    _require_warp()
    cfg = _warp_config()
    pts = _as_2d_float32(points)
    ref = np.asarray(reference, dtype=np.float32).reshape(-1)
    if pts.shape[1] != ref.shape[0]:
        raise ValueError("reference dimension does not match points")
    wp_points = wp.array2d(pts, dtype=float, device=cfg.device)
    wp_reference = wp.array(ref, dtype=float, device=cfg.device)
    wp_out = wp.zeros(pts.shape[0], dtype=float, device=cfg.device)
    wp.launch(
        _point_to_reference_sqdist_kernel,
        dim=pts.shape[0],
        inputs=[wp_points, wp_reference, int(pts.shape[1]), wp_out],
        device=cfg.device,
    )
    wp.synchronize_device(cfg.device)
    return np.asarray(wp_out.numpy(), dtype=np.float32)


def _warp_pairwise_sqdist(query_points: np.ndarray, points: np.ndarray) -> np.ndarray:
    _require_warp()
    cfg = _warp_config()
    q = _as_2d_float32(query_points)
    p = _as_2d_float32(points)
    if q.shape[1] != p.shape[1]:
        raise ValueError("query and reference dimensions do not match")
    flat_size = int(q.shape[0] * p.shape[0])
    wp_q = wp.array2d(q, dtype=float, device=cfg.device)
    wp_p = wp.array2d(p, dtype=float, device=cfg.device)
    wp_out = wp.zeros(flat_size, dtype=float, device=cfg.device)
    wp.launch(
        _pairwise_sqdist_kernel,
        dim=flat_size,
        inputs=[wp_q, wp_p, int(q.shape[1]), int(p.shape[0]), wp_out],
        device=cfg.device,
    )
    wp.synchronize_device(cfg.device)
    return np.asarray(wp_out.numpy(), dtype=np.float32).reshape(q.shape[0], p.shape[0])


def warp_exact_knn(query_points: np.ndarray, points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact KNN with a GPU hash grid and compact host transfer."""

    _require_warp()
    cfg = _warp_config()
    q64 = np.atleast_2d(np.asarray(query_points, dtype=np.float64))
    p64 = np.atleast_2d(np.asarray(points, dtype=np.float64))
    q = _as_2d_float32(query_points)
    p = _as_2d_float32(points)
    if q.shape[1] != p.shape[1]:
        raise ValueError("query and reference dimensions do not match")
    if p.shape[0] == 0:
        return np.zeros((q.shape[0], 0), dtype=np.int32), np.zeros((q.shape[0], 0), dtype=np.float32)
    k = min(int(k), int(p.shape[0]))
    candidate_count = min(int(p.shape[0]), max(k + 8, 2 * k))
    dim = int(q.shape[1])
    q3 = _pad_to_vec3(q)
    p3 = _pad_to_vec3(p)
    extent = np.maximum(np.ptp(p3[:, :dim], axis=0), np.finfo(np.float32).eps)
    measure = float(np.prod(extent))
    if dim == 2:
        radius = 1.5 * np.sqrt(candidate_count * measure / (np.pi * p.shape[0]))
    else:
        radius = 1.5 * (3.0 * candidate_count * measure / (4.0 * np.pi * p.shape[0])) ** (1.0 / 3.0)
    diameter = float(np.linalg.norm(extent))
    radius = max(float(radius), 32.0 * float(np.finfo(np.float32).eps) * max(diameter, 1.0))

    wp_q = wp.array(q3, dtype=wp.vec3, device=cfg.device)
    wp_p = wp.array(p3, dtype=wp.vec3, device=cfg.device)
    while True:
        sample_min = np.min(p3, axis=0) - radius
        sample_max = np.max(p3, axis=0) + radius
        grid_nx, grid_ny, grid_nz = _prepare_hash_grid_capacity(sample_min, sample_max, radius)
        grid = wp.HashGrid(grid_nx, grid_ny, grid_nz, device=cfg.device)
        grid.build(wp_p, radius)
        counts_wp = wp.zeros(q.shape[0], dtype=int, device=cfg.device)
        wp.launch(
            _count_ball_neighbors_kernel,
            dim=q.shape[0],
            inputs=[wp_q, wp_p, grid.id, radius, dim, counts_wp],
            device=cfg.device,
        )
        wp.synchronize_device(cfg.device)
        if int(np.min(np.asarray(counts_wp.numpy(), dtype=np.int32))) >= candidate_count:
            break
        radius *= 2.0
        if radius > 2.0 * max(diameter, 1.0):
            raise RuntimeError("Warp hash-grid KNN could not enclose enough reference points")

    indices_wp = wp.empty((q.shape[0], candidate_count), dtype=int, device=cfg.device)
    distances_wp = wp.empty((q.shape[0], candidate_count), dtype=float, device=cfg.device)
    wp.launch(
        _initialize_knn_kernel,
        dim=q.shape[0],
        inputs=[indices_wp, distances_wp, candidate_count],
        device=cfg.device,
    )
    wp.launch(
        _fill_knn_neighbors_kernel,
        dim=q.shape[0],
        inputs=[wp_q, wp_p, grid.id, radius, dim, candidate_count, indices_wp, distances_wp],
        device=cfg.device,
    )
    wp.synchronize_device(cfg.device)
    indices = np.asarray(indices_wp.numpy(), dtype=np.int32)
    if np.any(indices < 0):
        raise RuntimeError("Warp hash-grid KNN returned an incomplete neighborhood")
    candidate_distances = np.linalg.norm(q64[:, None, :] - p64[indices], axis=2)
    order = np.lexsort((indices, candidate_distances), axis=1)[:, :k]
    rows = np.arange(q.shape[0], dtype=np.int32)[:, None]
    selected_indices = indices[rows, order]
    selected_distances = candidate_distances[rows, order]
    return selected_indices.astype(np.int32), selected_distances.astype(np.float64)


def warp_exact_ball(
    query_points: np.ndarray,
    points: np.ndarray,
    radius: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    _require_warp()
    cfg = _warp_config()
    q = _as_2d_float32(query_points)
    p = _as_2d_float32(points)
    if q.shape[1] != p.shape[1]:
        raise ValueError("query and reference dimensions do not match")
    if p.shape[0] == 0:
        return [np.zeros((0,), dtype=np.int32) for _ in range(q.shape[0])], [np.zeros((0,), dtype=np.float32) for _ in range(q.shape[0])]
    if radius < 0.0:
        return [np.zeros((0,), dtype=np.int32) for _ in range(q.shape[0])], [np.zeros((0,), dtype=np.float32) for _ in range(q.shape[0])]

    dim = int(q.shape[1])
    q3 = _pad_to_vec3(q)
    p3 = _pad_to_vec3(p)
    sample_min = np.minimum(np.min(p3, axis=0), np.min(q3, axis=0)) - float(radius)
    sample_max = np.maximum(np.max(p3, axis=0), np.max(q3, axis=0)) + float(radius)
    grid_nx, grid_ny, grid_nz = _prepare_hash_grid_capacity(sample_min, sample_max, max(float(radius), np.finfo(np.float32).eps))
    wp_q = wp.array(q3, dtype=wp.vec3, device=cfg.device)
    wp_p = wp.array(p3, dtype=wp.vec3, device=cfg.device)
    grid = wp.HashGrid(grid_nx, grid_ny, grid_nz, device=cfg.device)
    grid.build(wp_p, max(float(radius), np.finfo(np.float32).eps))

    counts_wp = wp.zeros(q.shape[0], dtype=int, device=cfg.device)
    wp.launch(
        _count_ball_neighbors_kernel,
        dim=q.shape[0],
        inputs=[wp_q, wp_p, grid.id, float(radius), dim, counts_wp],
        device=cfg.device,
    )
    wp.synchronize_device(cfg.device)
    counts = np.asarray(counts_wp.numpy(), dtype=np.int32)
    offsets = np.empty(q.shape[0] + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    total = int(offsets[-1])
    if total == 0:
        return [np.zeros((0,), dtype=np.int32) for _ in range(q.shape[0])], [np.zeros((0,), dtype=np.float32) for _ in range(q.shape[0])]

    offsets_wp = wp.array(offsets, dtype=int, device=cfg.device)
    flat_indices_wp = wp.zeros(total, dtype=int, device=cfg.device)
    flat_distances_wp = wp.zeros(total, dtype=float, device=cfg.device)
    wp.launch(
        _fill_ball_neighbors_kernel,
        dim=q.shape[0],
        inputs=[wp_q, wp_p, grid.id, float(radius), dim, offsets_wp, flat_indices_wp, flat_distances_wp],
        device=cfg.device,
    )
    wp.synchronize_device(cfg.device)
    flat_indices = np.asarray(flat_indices_wp.numpy(), dtype=np.int32)
    flat_distances = np.asarray(flat_distances_wp.numpy(), dtype=np.float32)
    index_rows: list[np.ndarray] = []
    dist_rows: list[np.ndarray] = []
    for i in range(q.shape[0]):
        start = int(offsets[i])
        end = int(offsets[i + 1])
        ids = flat_indices[start:end]
        d = flat_distances[start:end]
        order = np.argsort(ids, kind="stable")
        index_rows.append(ids[order])
        dist_rows.append(d[order])
    return index_rows, dist_rows


def warp_assembled_boundary_keep_mask(points: np.ndarray, corner_flags: np.ndarray, radius: float) -> np.ndarray:
    _require_warp()
    cfg = _warp_config()
    pts = _as_2d_float32(points)
    corners = np.asarray(corner_flags, dtype=np.int32).reshape(-1)
    if corners.shape[0] != pts.shape[0]:
        raise ValueError("corner_flags length must match the number of points")
    if pts.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    if radius <= 0.0:
        return np.ones(pts.shape[0], dtype=bool)

    dim = int(pts.shape[1])
    p3 = _pad_to_vec3(pts)
    spacing_radius = float(radius) * (1.0 - 1.0e-12)
    duplicate_radius = 0.2 * float(radius)
    grid_radius = max(float(radius), np.finfo(np.float32).eps)
    sample_min = np.min(p3, axis=0) - grid_radius
    sample_max = np.max(p3, axis=0) + grid_radius
    grid_nx, grid_ny, grid_nz = _prepare_hash_grid_capacity(sample_min, sample_max, grid_radius)

    wp_points = wp.array(p3, dtype=wp.vec3, device=cfg.device)
    wp_corners = wp.array(corners, dtype=int, device=cfg.device)
    keep_wp = wp.array(np.ones(pts.shape[0], dtype=np.int32), dtype=int, device=cfg.device)
    grid = wp.HashGrid(grid_nx, grid_ny, grid_nz, device=cfg.device)
    grid.build(wp_points, grid_radius)

    wp.launch(
        _assembled_boundary_duplicate_keep_kernel,
        dim=1,
        inputs=[wp_points, grid.id, duplicate_radius, dim, int(pts.shape[0]), keep_wp],
        device=cfg.device,
    )
    wp.launch(
        _assembled_boundary_spacing_keep_kernel,
        dim=1,
        inputs=[wp_points, wp_corners, grid.id, spacing_radius, dim, int(pts.shape[0]), keep_wp],
        device=cfg.device,
    )
    wp.synchronize_device(cfg.device)
    return np.asarray(keep_wp.numpy(), dtype=np.int32).astype(bool)


def warp_nearest_neighbor_distances(query_points: np.ndarray, points: np.ndarray) -> np.ndarray:
    q = _as_2d_float32(query_points)
    p = _as_2d_float32(points)
    if p.shape[0] == 0:
        return np.full(q.shape[0], np.inf, dtype=np.float32)
    sqdist = _warp_pairwise_sqdist(q, p)
    return np.sqrt(np.maximum(np.min(sqdist, axis=1), 0.0)).astype(np.float32)


def warp_pairwise_distances(query_points: np.ndarray, points: np.ndarray) -> np.ndarray:
    q = _as_2d_float32(query_points)
    p = _as_2d_float32(points)
    if q.shape[0] == 0 or p.shape[0] == 0:
        return np.zeros((q.shape[0], p.shape[0]), dtype=np.float32)
    sqdist = _warp_pairwise_sqdist(q, p)
    return np.sqrt(np.maximum(sqdist, 0.0)).astype(np.float32)


def _estimate_max_points(sample_min: np.ndarray, sample_max: np.ndarray, min_radius: float, dim: int) -> int:
    cell_size = min_radius / np.sqrt(dim)
    grid_size = np.maximum(1, np.ceil((sample_max[:dim] - sample_min[:dim]) / cell_size).astype(int))
    return int(np.prod(grid_size))


def _prepare_hash_grid_capacity(sample_min: np.ndarray, sample_max: np.ndarray, cell_size: float) -> tuple[int, int, int]:
    dims = np.maximum(1, np.ceil((sample_max[:3] - sample_min[:3]) / cell_size).astype(int))
    return int(dims[0]), int(dims[1]), int(dims[2])


def _host_nearest_boundary_distance(point: np.ndarray, boundary_points: np.ndarray) -> float:
    if boundary_points.size == 0:
        return float("inf")
    diffs = boundary_points[:, : point.shape[0]] - point[None, :]
    return float(np.sqrt(np.min(np.sum(diffs * diffs, axis=1))))


def _host_local_radius(point: np.ndarray, opts: WarpBridsonOptions) -> float:
    point = np.asarray(point, dtype=float)
    mode = str(opts.mode)
    if mode == "fixed_radius":
        return float(opts.radius)
    if mode == "fixed_radius_with_boundary_refinement":
        if (
            opts.boundary_points.size > 0
            and opts.boundary_distance > 0.0
            and _host_nearest_boundary_distance(point, opts.boundary_points) <= opts.boundary_distance
        ):
            return float(opts.boundary_refinement_fraction) * float(opts.radius)
        return float(opts.radius)
    if mode == "variable_radius":
        if opts.radius_function is None:
            raise ValueError("variable_radius mode requires radius_function")
        return max(float(opts.radius_function(point, float(opts.min_radius))), float(opts.min_radius))
    if mode == "variable_radius_with_boundary_refinement":
        if opts.radius_function is None:
            raise ValueError("variable_radius_with_boundary_refinement mode requires radius_function")
        base = max(float(opts.radius_function(point, float(opts.min_radius))), float(opts.min_radius))
        if (
            opts.boundary_points.size > 0
            and opts.boundary_distance > 0.0
            and _host_nearest_boundary_distance(point, opts.boundary_points) <= opts.boundary_distance
        ):
            return float(opts.boundary_refinement_fraction) * float(opts.min_radius)
        return base
    raise ValueError(f"unsupported Warp Bridson mode: {mode}")


def warp_bridson_sample_fixed_radius(
    sample_min: np.ndarray,
    sample_max: np.ndarray,
    opts: WarpBridsonOptions,
) -> np.ndarray:
    _require_warp()
    cfg = _warp_config()
    sample_min = _vec3_from_point(sample_min)
    sample_max = _vec3_from_point(sample_max)
    dim = 2 if abs(float(sample_max[2] - sample_min[2])) < 1e-12 else 3
    if dim not in {2, 3}:
        raise ValueError("Warp Bridson sampler currently supports only 2D or 3D boxes")

    if opts.mode not in {"fixed_radius", "fixed_radius_with_boundary_refinement"}:
        raise ValueError("Warp Bridson sampler currently supports only fixed-radius modes")

    coarse_radius = float(opts.radius)
    refined_radius = (
        coarse_radius
        if opts.mode == "fixed_radius"
        else float(opts.boundary_refinement_fraction) * coarse_radius
    )
    min_radius = refined_radius
    max_points = _estimate_max_points(sample_min, sample_max, min_radius, dim)
    if max_points <= 0:
        return np.zeros((0, dim), dtype=np.float32)

    rng = np.random.default_rng(int(opts.seed))
    x0 = sample_min[:dim] + rng.random(dim, dtype=np.float32) * (sample_max[:dim] - sample_min[:dim])
    boundary_points = _pad_to_vec3(opts.boundary_points) if opts.boundary_points.size else np.zeros((0, 3), dtype=np.float32)
    has_boundary_refinement = int(
        opts.mode == "fixed_radius_with_boundary_refinement"
        and boundary_points.shape[0] > 0
        and opts.boundary_distance > 0.0
        and opts.boundary_refinement_fraction < 1.0
    )

    if dim == 2:
        x0 = np.array([x0[0], x0[1], 0.0], dtype=np.float32)
    else:
        x0 = np.array(x0, dtype=np.float32)

    x0_radius = coarse_radius
    if has_boundary_refinement:
        sqdist = np.sum((boundary_points[:, :dim] - x0[:dim]) ** 2, axis=1)
        if sqdist.size and np.sqrt(np.min(sqdist)) <= opts.boundary_distance:
            x0_radius = refined_radius

    points_host = np.zeros((max_points, 3), dtype=np.float32)
    radii_host = np.zeros(max_points, dtype=np.float32)
    points_host[0] = x0
    radii_host[0] = np.float32(x0_radius)
    point_count = 1
    active_host = np.array([0], dtype=np.int32)

    points_wp = wp.zeros(max_points, dtype=wp.vec3, device=cfg.device)
    radii_wp = wp.zeros(max_points, dtype=float, device=cfg.device)
    wp.copy(points_wp[:1], wp.array(points_host[:1], dtype=wp.vec3, device=cfg.device))
    wp.copy(radii_wp[:1], wp.array(radii_host[:1], dtype=float, device=cfg.device))

    active_wp = wp.zeros(max_points, dtype=int, device=cfg.device)
    provisional_points_wp = wp.zeros(max_points, dtype=wp.vec3, device=cfg.device)
    provisional_radii_wp = wp.zeros(max_points, dtype=float, device=cfg.device)
    provisional_valid_wp = wp.zeros(max_points, dtype=int, device=cfg.device)
    accepted_wp = wp.zeros(max_points, dtype=int, device=cfg.device)

    grid_nx, grid_ny, grid_nz = _prepare_hash_grid_capacity(sample_min, sample_max, coarse_radius)
    existing_grid = wp.HashGrid(grid_nx, grid_ny, grid_nz, device=cfg.device)
    provisional_grid = wp.HashGrid(grid_nx, grid_ny, grid_nz, device=cfg.device)
    boundary_grid = wp.HashGrid(grid_nx, grid_ny, grid_nz, device=cfg.device)

    boundary_wp = wp.array(boundary_points, dtype=wp.vec3, device=cfg.device)
    if boundary_points.shape[0] > 0:
        boundary_cell = max(float(opts.boundary_distance), coarse_radius)
        boundary_grid.build(boundary_wp, boundary_cell)

    epoch = 0
    while active_host.size > 0 and point_count < max_points:
        active_count = int(active_host.size)
        active_wp_slice = wp.array(active_host, dtype=int, device=cfg.device)
        wp.copy(active_wp[:active_count], active_wp_slice)

        existing_grid.build(points_wp[:point_count], coarse_radius)

        wp.launch(
            _generate_provisional_candidates_kernel,
            dim=active_count,
            inputs=[
                int(opts.seed),
                epoch,
                dim,
                int(opts.attempts),
                coarse_radius,
                refined_radius,
                float(opts.boundary_distance),
                has_boundary_refinement,
                wp.vec3(*sample_min),
                wp.vec3(*sample_max),
                active_wp,
                points_wp,
                radii_wp,
                wp.uint64(existing_grid.id),
                wp.uint64(boundary_grid.id if boundary_points.shape[0] > 0 else 0),
                boundary_wp,
                provisional_points_wp,
                provisional_radii_wp,
                provisional_valid_wp,
            ],
            device=cfg.device,
        )

        provisional_grid.build(provisional_points_wp[:active_count], coarse_radius)
        wp.launch(
            _filter_provisional_candidates_kernel,
            dim=active_count,
            inputs=[
                coarse_radius,
                has_boundary_refinement,
                wp.uint64(provisional_grid.id),
                provisional_points_wp,
                provisional_radii_wp,
                provisional_valid_wp,
                accepted_wp,
            ],
            device=cfg.device,
        )
        wp.synchronize_device(cfg.device)

        provisional_points_host = np.asarray(provisional_points_wp.numpy()[:active_count], dtype=np.float32)
        provisional_radii_host = np.asarray(provisional_radii_wp.numpy()[:active_count], dtype=np.float32)
        provisional_valid_host = np.asarray(provisional_valid_wp.numpy()[:active_count], dtype=np.int32)
        accepted_host = np.asarray(accepted_wp.numpy()[:active_count], dtype=np.int32)

        next_active: list[int] = []
        new_points: list[np.ndarray] = []
        new_radii: list[float] = []
        for local_idx, point_idx in enumerate(active_host):
            if provisional_valid_host[local_idx] == 0:
                continue
            next_active.append(int(point_idx))
            if accepted_host[local_idx] != 0:
                new_points.append(provisional_points_host[local_idx].copy())
                new_radii.append(float(provisional_radii_host[local_idx]))

        start = point_count
        for p, r in zip(new_points, new_radii):
            if point_count >= max_points:
                break
            points_host[point_count] = p
            radii_host[point_count] = r
            next_active.append(point_count)
            point_count += 1

        if point_count > start:
            wp.copy(
                points_wp[start:point_count],
                wp.array(points_host[start:point_count], dtype=wp.vec3, device=cfg.device),
            )
            wp.copy(
                radii_wp[start:point_count],
                wp.array(radii_host[start:point_count], dtype=float, device=cfg.device),
            )

        active_host = np.asarray(next_active, dtype=np.int32)
        epoch += 1

    return np.asarray(points_host[:point_count, :dim], dtype=np.float32)


def _warp_bridson_sample_variable_radius(
    sample_min: np.ndarray,
    sample_max: np.ndarray,
    opts: WarpBridsonOptions,
) -> np.ndarray:
    sample_min = _vec3_from_point(sample_min)
    sample_max = _vec3_from_point(sample_max)
    dim = 2 if abs(float(sample_max[2] - sample_min[2])) < 1e-12 else 3
    if dim not in {2, 3}:
        raise ValueError("Warp Bridson sampler currently supports only 2D or 3D boxes")

    min_radius = float(opts.min_radius)
    max_points = _estimate_max_points(sample_min, sample_max, min_radius, dim)
    if max_points <= 0:
        return np.zeros((0, dim), dtype=np.float32)

    rng = np.random.default_rng(int(opts.seed))
    x0 = sample_min[:dim] + rng.random(dim, dtype=np.float32) * (sample_max[:dim] - sample_min[:dim])
    points: list[np.ndarray] = [np.asarray(x0, dtype=np.float32)]
    radii: list[float] = [float(_host_local_radius(points[0], opts))]
    active: list[int] = [0]

    while active and len(points) < max_points:
        pick = int(rng.integers(len(active)))
        active_idx = int(active[pick])
        base = points[active_idx]
        active_radius = float(radii[active_idx])
        candidates = np.zeros((int(opts.attempts), dim), dtype=np.float32)
        valid_mask = np.zeros(int(opts.attempts), dtype=bool)
        for attempt_idx in range(int(opts.attempts)):
            direction = rng.normal(size=dim).astype(np.float32)
            direction /= max(float(np.linalg.norm(direction)), np.finfo(float).eps)
            shell_radius = active_radius * float((1.0 + rng.random() * (2**dim - 1)) ** (1.0 / dim))
            candidate = base + shell_radius * direction
            if np.any(candidate < sample_min[:dim]) or np.any(candidate > sample_max[:dim]):
                continue
            candidates[attempt_idx] = candidate
            valid_mask[attempt_idx] = True

        if not np.any(valid_mask):
            active[pick] = active[-1]
            active.pop()
            continue

        candidate_radii = np.zeros(int(opts.attempts), dtype=np.float32)
        valid_indices = np.flatnonzero(valid_mask)
        for idx in valid_indices:
            candidate_radii[idx] = np.float32(_host_local_radius(candidates[idx], opts))

        existing_points = np.asarray(points, dtype=np.float32)
        existing_radii = np.asarray(radii, dtype=np.float32)
        dist_existing = warp_pairwise_distances(candidates[valid_indices], existing_points)

        accepted_point = None
        accepted_radius = None
        for row_idx, attempt_idx in enumerate(valid_indices):
            threshold = np.maximum(candidate_radii[attempt_idx], existing_radii)
            mask = np.ones(existing_points.shape[0], dtype=bool)
            mask[active_idx] = False
            if not np.any(dist_existing[row_idx][mask] < threshold[mask]):
                accepted_point = candidates[attempt_idx].copy()
                accepted_radius = float(candidate_radii[attempt_idx])
                break

        if accepted_point is None:
            active[pick] = active[-1]
            active.pop()
            continue

        points.append(accepted_point)
        radii.append(accepted_radius)
        active.append(len(points) - 1)

    return np.asarray(points[:max_points], dtype=np.float32)


def warp_bridson_sample(
    sample_min: np.ndarray,
    sample_max: np.ndarray,
    opts: WarpBridsonOptions,
) -> np.ndarray:
    if opts.mode in {"fixed_radius", "fixed_radius_with_boundary_refinement"}:
        return warp_bridson_sample_fixed_radius(sample_min, sample_max, opts)
    if opts.mode in {"variable_radius", "variable_radius_with_boundary_refinement"}:
        return _warp_bridson_sample_variable_radius(sample_min, sample_max, opts)
    raise ValueError(f"unsupported Warp Bridson mode: {opts.mode}")
