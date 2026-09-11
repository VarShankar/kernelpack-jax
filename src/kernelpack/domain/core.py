from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
import os
import numpy as np

from jax import jit, lax
import jax.numpy as jnp

from kernelpack.accelerators import WarpUnavailableError, warp_available, warp_exact_ball, warp_exact_knn
from kernelpack.geometry import distance_matrix


@partial(jit, static_argnames=("k",))
def _jax_exact_knn(
    query_points: jnp.ndarray,
    points: jnp.ndarray,
    point_active: jnp.ndarray,
    *,
    k: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    query_sq = jnp.sum(query_points * query_points, axis=1, keepdims=True)
    point_sq = jnp.sum(points * points, axis=1)[None, :]
    squared_distances = jnp.maximum(query_sq + point_sq - 2.0 * (query_points @ points.T), 0.0)
    squared_distances = jnp.where(point_active[None, :], squared_distances, jnp.inf)
    negative_distances, indices = lax.top_k(-squared_distances, k)
    return indices, jnp.sqrt(jnp.maximum(-negative_distances, 0.0))


@dataclass
class TreeStruct:
    points: jnp.ndarray
    searcher: object | None
    has_searcher: bool


@dataclass
class DomainDescriptor:
    xi: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    xb: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    xg: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    x: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    xf: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    nrmls: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    sep_rad: float = jnp.nan
    tall_tree: TreeStruct | None = None
    int_bdry_tree: TreeStruct | None = None
    bdry_tree: TreeStruct | None = None
    outer_level_set: object | None = None
    boundary_level_sets: list[object] = field(default_factory=list)
    interior_active: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,), dtype=bool))
    boundary_active: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,), dtype=bool))
    ghost_active: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,), dtype=bool))

    def set_nodes(self, int_nodes: jnp.ndarray, bdry_nodes: jnp.ndarray, ghost_nodes: jnp.ndarray | None = None) -> None:
        int_nodes = jnp.asarray(int_nodes, dtype=float)
        bdry_nodes = jnp.asarray(bdry_nodes, dtype=float)
        if ghost_nodes is None:
            dim = int_nodes.shape[1] if int_nodes.size else bdry_nodes.shape[1]
            ghost_nodes = jnp.zeros((0, dim), dtype=float)
        self.xi = int_nodes
        self.xb = bdry_nodes
        self.xg = jnp.asarray(ghost_nodes, dtype=float)
        self.interior_active = jnp.ones((self.xi.shape[0],), dtype=bool)
        self.boundary_active = jnp.ones((self.xb.shape[0],), dtype=bool)
        self.ghost_active = jnp.ones((self.xg.shape[0],), dtype=bool)
        self._set_total_nodes()

    def set_nodes_from_host(
        self,
        int_nodes: np.ndarray,
        bdry_nodes: np.ndarray,
        ghost_nodes: np.ndarray | None = None,
    ) -> None:
        """Populate a dynamic snapshot without shape-specialized JAX concatenations."""

        int_host = np.asarray(int_nodes, dtype=float)
        bdry_host = np.asarray(bdry_nodes, dtype=float)
        if ghost_nodes is None:
            dim = int_host.shape[1] if int_host.size else bdry_host.shape[1]
            ghost_host = np.zeros((0, dim), dtype=float)
        else:
            ghost_host = np.asarray(ghost_nodes, dtype=float)
        dim = int_host.shape[1] if int_host.size else bdry_host.shape[1]
        physical_host = np.vstack([int_host, bdry_host]) if (int_host.size or bdry_host.size) else np.zeros((0, dim))
        all_host = np.vstack([physical_host, ghost_host]) if (physical_host.size or ghost_host.size) else np.zeros((0, dim))
        self.xi = jnp.asarray(int_host)
        self.xb = jnp.asarray(bdry_host)
        self.xg = jnp.asarray(ghost_host)
        self.x = jnp.asarray(physical_host)
        self.xf = jnp.asarray(all_host)
        self.interior_active = jnp.ones((int_host.shape[0],), dtype=bool)
        self.boundary_active = jnp.ones((bdry_host.shape[0],), dtype=bool)
        self.ghost_active = jnp.ones((ghost_host.shape[0],), dtype=bool)

    def set_active_masks(
        self,
        interior_active: jnp.ndarray,
        boundary_active: jnp.ndarray,
        ghost_active: jnp.ndarray,
    ) -> None:
        masks = [
            jnp.asarray(interior_active, dtype=bool).reshape(-1),
            jnp.asarray(boundary_active, dtype=bool).reshape(-1),
            jnp.asarray(ghost_active, dtype=bool).reshape(-1),
        ]
        expected = [self.xi.shape[0], self.xb.shape[0], self.xg.shape[0]]
        if any(mask.size != count for mask, count in zip(masks, expected)):
            raise ValueError("active masks must match interior, boundary, and ghost capacities")
        self.interior_active, self.boundary_active, self.ghost_active = masks

    def set_normals(self, nrmls: jnp.ndarray) -> None:
        nrmls = jnp.asarray(nrmls, dtype=float)
        if nrmls.shape[0] != self.xb.shape[0]:
            raise ValueError("boundary normals must match boundary node count")
        self.nrmls = nrmls

    def set_outer_level_set(self, level_set: object) -> None:
        self.outer_level_set = level_set

    def set_sep_rad(self, sep_rad: float) -> None:
        self.sep_rad = float(sep_rad)

    def set_boundary_level_sets(self, level_sets: list[object]) -> None:
        self.boundary_level_sets = level_sets

    def build_structs(self) -> None:
        self.tall_tree = self._build_tree_struct(self.xf)
        self.int_bdry_tree = self._build_tree_struct(self.x)
        self.bdry_tree = self._build_tree_struct(self.xb)

    def query_knn(
        self,
        tree_mode: str,
        query_points: jnp.ndarray,
        k: int,
        *,
        backend: str = "auto",
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        _, points = self._get_tree_data(tree_mode)
        point_active = self.get_tree_active_mask(tree_mode)
        query_points = jnp.atleast_2d(jnp.asarray(query_points, dtype=float))
        if points.size == 0:
            return jnp.zeros((query_points.shape[0], 0), dtype=int), jnp.zeros((query_points.shape[0], 0), dtype=float)
        k = min(int(k), points.shape[0])
        if k > int(jnp.count_nonzero(point_active)):
            raise ValueError("KNN stencil size exceeds the number of active tree points")
        selected_backend = _resolve_backend(backend)
        if selected_backend == "warp":
            try:
                active_indices = np.flatnonzero(np.asarray(point_active, dtype=bool))
                order_np, distances_np = warp_exact_knn(
                    np.asarray(query_points),
                    np.asarray(points)[active_indices],
                    k,
                )
                return (
                    jnp.asarray(active_indices[order_np], dtype=int),
                    jnp.asarray(distances_np, dtype=float),
                )
            except WarpUnavailableError:
                if backend == "auto":
                    selected_backend = "cpu"
                else:
                    raise
        return _jax_exact_knn(query_points, points, point_active, k=k)

    def query_ball(
        self,
        tree_mode: str,
        query_points: jnp.ndarray,
        radius: float,
        *,
        backend: str = "auto",
    ) -> tuple[list[jnp.ndarray], list[jnp.ndarray]]:
        _, points = self._get_tree_data(tree_mode)
        point_active = self.get_tree_active_mask(tree_mode)
        query_points = jnp.atleast_2d(jnp.asarray(query_points, dtype=float))
        if points.size == 0:
            return [jnp.zeros(0, dtype=int) for _ in range(query_points.shape[0])], [jnp.zeros(0, dtype=float) for _ in range(query_points.shape[0])]
        selected_backend = _resolve_backend(backend)
        if selected_backend == "warp" and not bool(jnp.all(point_active)):
            selected_backend = "jax"
        if selected_backend == "warp":
            try:
                idx_rows, dist_rows = warp_exact_ball(np.asarray(query_points, dtype=float), np.asarray(points, dtype=float), float(radius))
                return [jnp.asarray(ids, dtype=int) for ids in idx_rows], [jnp.asarray(d, dtype=float) for d in dist_rows]
            except WarpUnavailableError:
                if backend != "auto":
                    raise
        d = jnp.where(point_active[None, :], distance_matrix(query_points, points), jnp.inf)
        indices = []
        distances = []
        for row in d:
            mask = row <= radius
            indices.append(jnp.flatnonzero(mask))
            distances.append(row[mask])
        return indices, distances

    def query_ball_padded(
        self,
        tree_mode: str,
        query_points: jnp.ndarray,
        radius: float,
        *,
        k_min: int = 0,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return fixed-shape ball-query data suitable for JAX device kernels.

        The legacy `query_ball` API intentionally returns ragged Python lists.
        This API keeps the neighbor test on JAX arrays and pads the result to
        all tree points, with optional `k_min` nearest-neighbor enrichment.
        """

        _, points = self._get_tree_data(tree_mode)
        point_active = self.get_tree_active_mask(tree_mode)
        query_points = jnp.atleast_2d(jnp.asarray(query_points, dtype=float))
        if points.size == 0:
            shape = (query_points.shape[0], 0)
            return jnp.zeros(shape, dtype=int), jnp.zeros(shape, dtype=float), jnp.zeros(shape, dtype=bool)
        d = jnp.where(point_active[None, :], distance_matrix(query_points, points), jnp.inf)
        valid = d <= float(radius)
        k = min(max(int(k_min), 0), int(points.shape[0]))
        if k > 0:
            nearest = jnp.argsort(d, axis=1)[:, :k]
            rows = jnp.broadcast_to(jnp.arange(query_points.shape[0], dtype=int)[:, None], nearest.shape)
            valid = valid.at[rows, nearest].set(True)
        order = jnp.argsort(jnp.where(valid, d, jnp.inf), axis=1)
        ids = order
        dist = jnp.take_along_axis(d, ids, axis=1)
        mask = jnp.take_along_axis(valid, ids, axis=1)
        return ids, dist, mask

    def get_tree_points(self, tree_mode: str) -> jnp.ndarray:
        return self._get_tree_data(tree_mode)[1]

    def get_tree_globals(self, tree_mode: str) -> jnp.ndarray:
        tree_mode = _normalize_tree_mode(tree_mode)
        if tree_mode == "all":
            return jnp.arange(1, self.xf.shape[0] + 1)
        if tree_mode == "interior_boundary":
            return jnp.arange(1, self.x.shape[0] + 1)
        if tree_mode == "boundary":
            return self.get_num_interior_nodes() + jnp.arange(1, self.xb.shape[0] + 1)
        raise ValueError(f"unknown tree mode {tree_mode}")

    def get_tree_active_mask(self, tree_mode: str) -> jnp.ndarray:
        tree_mode = _normalize_tree_mode(tree_mode)
        if tree_mode == "all":
            return jnp.concatenate([self.interior_active, self.boundary_active, self.ghost_active])
        if tree_mode == "interior_boundary":
            return jnp.concatenate([self.interior_active, self.boundary_active])
        if tree_mode == "boundary":
            return self.boundary_active
        raise ValueError(f"unknown tree mode {tree_mode}")

    def get_physical_active_mask(self) -> jnp.ndarray:
        return jnp.concatenate([self.interior_active, self.boundary_active])

    def get_boundary_active_mask(self) -> jnp.ndarray:
        return self.boundary_active

    def get_all_active_mask(self) -> jnp.ndarray:
        return jnp.concatenate([self.interior_active, self.boundary_active, self.ghost_active])

    def get_interior_nodes(self) -> jnp.ndarray:
        return self.xi

    def get_bdry_nodes(self) -> jnp.ndarray:
        return self.xb

    def get_ghost_nodes(self) -> jnp.ndarray:
        return self.xg

    def get_int_bdry_nodes(self) -> jnp.ndarray:
        return self.x

    def get_all_nodes(self) -> jnp.ndarray:
        return self.xf

    def get_nrmls(self) -> jnp.ndarray:
        return self.nrmls

    def get_outer_level_set(self) -> object | None:
        return self.outer_level_set

    def get_boundary_level_sets(self) -> list[object]:
        return self.boundary_level_sets

    def get_sep_rad(self) -> float:
        return self.sep_rad

    def get_dim(self) -> int:
        return int(self.xf.shape[1])

    def get_num_total_nodes(self) -> int:
        return int(self.xf.shape[0])

    def get_num_int_bdry_nodes(self) -> int:
        return int(self.x.shape[0])

    def get_num_interior_nodes(self) -> int:
        return int(self.xi.shape[0])

    def get_num_bdry_nodes(self) -> int:
        return int(self.xb.shape[0])

    def _set_total_nodes(self) -> None:
        dim = self.xi.shape[1] if self.xi.size else self.xb.shape[1]
        if self.xg.size == 0:
            self.xg = jnp.zeros((0, dim))
        self.x = jnp.vstack([self.xi, self.xb]) if (self.xb.size or self.xi.size) else jnp.zeros((0, dim))
        self.xf = jnp.vstack([self.x, self.xg]) if (self.xg.size or self.x.size) else jnp.zeros((0, dim))

    def _get_tree_data(self, tree_mode: str) -> tuple[TreeStruct | None, jnp.ndarray]:
        tree_mode = _normalize_tree_mode(tree_mode)
        if tree_mode == "all":
            return self.tall_tree, self.xf
        if tree_mode == "interior_boundary":
            return self.int_bdry_tree, self.x
        if tree_mode == "boundary":
            return self.bdry_tree, self.xb
        raise ValueError(f"unknown tree mode {tree_mode}")

    @staticmethod
    def _build_tree_struct(points: jnp.ndarray) -> TreeStruct:
        points = jnp.asarray(points, dtype=float)
        return TreeStruct(points=points, searcher=None, has_searcher=False)


def _normalize_tree_mode(mode: str) -> str:
    mode = str(mode).lower()
    aliases = {
        "all": "all",
        "all_nodes": "all",
        "full": "all",
        "interior_boundary": "interior_boundary",
        "interior+boundary": "interior_boundary",
        "int_bdry": "interior_boundary",
        "boundary": "boundary",
        "bdry": "boundary",
        "boundary_only": "boundary",
    }
    if mode not in aliases:
        raise ValueError(f"unknown tree mode {mode}")
    return aliases[mode]


def _resolve_backend(backend: str) -> str:
    backend_l = str(backend).lower()
    if backend_l == "auto":
        use_warp = os.environ.get("KERNELPACK_JAX_USE_WARP", "").strip().lower()
        if use_warp in {"1", "true", "yes", "on"} and warp_available():
            return "warp"
        return "jax"
    if backend_l == "cpu":
        return "jax"
    if backend_l not in {"jax", "warp"}:
        raise ValueError(f"unknown backend {backend}")
    return backend_l
