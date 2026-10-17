from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
import os
import numpy as np

import jax
from jax import jit, lax
import jax.numpy as jnp

from kernelpack.accelerators import WarpUnavailableError, warp_available, warp_exact_ball, warp_exact_knn
from kernelpack.geometry import distance_matrix
from .stencils import exact_knn as _jax_exact_knn, select_stencils, nearest_periodic_images

# Exact kNN: dense GPU top-k in row blocks up to this many query-point pairs (measured faster than the
# cell-blocked search for n_s up to 4*241 on 3e4 nodes), cell-blocked search beyond.
_DENSE_KNN_MAX_PAIRS = 10 ** 10
_DENSE_KNN_BLOCK_ENTRIES = 1 << 27     # distances per dense block (1 GiB in float64)


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
    # Boundary-aware stencils (see kernelpack.domain.boundary_aware); None = Euclidean nearest neighbours.
    stencil_level_set: object | None = None
    stencil_stats: dict = field(default_factory=dict)
    periodic_bounds: jnp.ndarray | None = None
    periodic_dims: jnp.ndarray | None = None

    def set_periodic(self,bounds=None,periodic_dims=None):
        """Enable rectangular periodic stencil geometry; None disables it.

        This configures selection/local coordinates, not PDE boundary rows.
        Boundary-aware ray filtering cannot currently be combined with it.
        """
        if bounds is None:
            self.periodic_bounds=None
            self.periodic_dims=None
            return
        if self.stencil_level_set is not None:
            raise ValueError("Periodic and level-set-filtered stencils cannot be combined")
        bounds=jnp.asarray(bounds,dtype=self.x.dtype)
        dim=self.x.shape[1]
        mask=jnp.ones(dim,dtype=bool) if periodic_dims is None else jnp.asarray(periodic_dims,dtype=bool)
        if bounds.shape!=(2,dim) or mask.shape!=(dim,):
            raise ValueError("Periodic bounds/mask must match spatial dimension")
        lengths=bounds[1]-bounds[0]
        if not bool(jnp.all(jnp.isfinite(bounds))&jnp.all(jnp.where(mask,lengths>0,lengths>=0))):
            raise ValueError("Periodic dimensions require finite bounds of positive length")
        self.periodic_bounds=bounds
        self.periodic_dims=mask

    def stencil_coordinates(self,tree_mode,queries,indices):
        """Coordinates for local fits; indices continue to address physical DOFs."""
        nodes=self.get_tree_points(tree_mode)[indices]
        if self.periodic_bounds is None:
            return nodes
        return nearest_periodic_images(nodes,queries,
            self.periodic_bounds[1]-self.periodic_bounds[0],self.periodic_dims)

    def _query_distances(self,points,queries):
        if self.periodic_bounds is None:
            return distance_matrix(queries,points)
        import jax
        images=jax.vmap(lambda q:nearest_periodic_images(points,q,
            self.periodic_bounds[1]-self.periodic_bounds[0],self.periodic_dims))(queries)
        return jnp.linalg.norm(images-queries[:,None,:],axis=-1)

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

    def set_stencil_level_set(self, model: object, h: float | None = None, **options) -> None:
        """Select stencils by ray tracing against the implicit layer `model` (a CFPU model); None disables."""
        if model is None:
            self.stencil_level_set = None
            return
        if self.periodic_bounds is not None:
            raise ValueError("Periodic and level-set-filtered stencils cannot be combined")
        from kernelpack.domain.boundary_aware import StencilLevelSet
        self.stencil_level_set = StencilLevelSet(model, self.sep_rad if h is None else h, **options)
        self._stencil_phi_nodes = {}
        self._stencil_memo = {}

    def _boundary_aware_knn(self, tree_mode: str, query_points, k: int, backend: str):
        from kernelpack.domain.boundary_aware import boundary_aware_knn
        tree_mode = _normalize_tree_mode(tree_mode)
        _, points = self._get_tree_data(tree_mode)
        n_tree = int(points.shape[0])
        K = min(4 * int(k), n_tree)
        idx, dist = self.query_knn(tree_mode, query_points, K, backend=backend, _euclidean=True)
        idx, dist = np.asarray(idx), np.asarray(dist)
        q = np.atleast_2d(np.asarray(query_points, dtype=float))
        if not (np.linalg.norm(np.asarray(points)[idx[:, 0]] - q, axis=1) <= 1e-12 * (1 + np.abs(q).max())).all():
            return jnp.asarray(idx[:, :k]), jnp.asarray(dist[:, :k])   # centers that are not tree points: Euclidean
        owner = np.arange(n_tree)
        if tree_mode == "all":
            ni, nb, ng = self.get_num_interior_nodes(), self.get_num_bdry_nodes(), int(self.xg.shape[0])
            if ng == nb:
                owner[ni + nb:] = ni + np.arange(ng)   # ghost i belongs to boundary node i
        key = (tree_mode, int(k), q.shape, hash(q.tobytes()))
        memo = getattr(self, "_stencil_memo", {})
        if key in memo:
            return memo[key]
        cache = getattr(self, "_stencil_phi_nodes", {})
        if tree_mode not in cache:
            cache[tree_mode] = self.stencil_level_set(np.asarray(points))
            self._stencil_phi_nodes = cache
        stats = {}
        out_i, out_d = boundary_aware_knn(np.asarray(points), owner, cache[tree_mode], idx, dist, int(k),
                                          self.stencil_level_set, stats=stats)
        self.stencil_stats = {key: self.stencil_stats.get(key, 0) + val for key, val in stats.items()} | {
            "queries": self.stencil_stats.get("queries", 0) + 1}
        memo[key] = (jnp.asarray(out_i, dtype=int), jnp.asarray(out_d, dtype=float))
        self._stencil_memo = memo
        return memo[key]

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
        _euclidean: bool = False,
        center_indices: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self.periodic_bounds is not None:
            if backend not in {"auto","jax","warp","cpu"}:
                raise ValueError("Unknown neighbor backend")
            _,points=self._get_tree_data(tree_mode)
            queries=jnp.atleast_2d(jnp.asarray(query_points,dtype=points.dtype))
            if points.size==0:
                return jnp.zeros((len(queries),0),dtype=int),jnp.zeros((len(queries),0),dtype=points.dtype)
            ids,images,valid=select_stencils(points,queries,k=min(int(k),len(points)),
                lengths=self.periodic_bounds[1]-self.periodic_bounds[0],periodic=self.periodic_dims,
                point_active=self.get_tree_active_mask(tree_mode),
                center_indices=center_indices,use_periodic=True)
            if not bool(jnp.all(valid)):
                raise ValueError("Invalid or ambiguous periodic center, or insufficient distinct active stencil nodes")
            return ids,jnp.linalg.norm(images-queries[:,None,:],axis=-1)
        if self.stencil_level_set is not None and not _euclidean:
            return self._boundary_aware_knn(tree_mode, query_points, k, backend)
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
        if isinstance(query_points, jax.core.Tracer) or isinstance(points, jax.core.Tracer):
            return _jax_exact_knn(query_points, points, point_active, k=k)
        if query_points.shape[0] * points.shape[0] > _DENSE_KNN_MAX_PAIRS:
            # Very large clouds: cell-blocked exact search, O(N k) work (accelerators/jax_spatial.grid_knn).
            # Below this size the dense GPU top-k is faster for the large stencils of high orders.
            from kernelpack.accelerators.jax_spatial import grid_knn
            idx, dist = grid_knn(np.asarray(query_points), np.asarray(points), k, valid=np.asarray(point_active, bool))
            return jnp.asarray(idx, dtype=int), jnp.asarray(dist, dtype=float)
        # Dense top-k in row blocks, so that at most _DENSE_KNN_BLOCK_ENTRIES distances are held at once.
        rows = max(1, int(_DENSE_KNN_BLOCK_ENTRIES // max(1, points.shape[0])))
        if rows >= query_points.shape[0]:
            return _jax_exact_knn(query_points, points, point_active, k=k)
        parts = [_jax_exact_knn(query_points[s:s + rows], points, point_active, k=k)
                 for s in range(0, query_points.shape[0], rows)]
        return jnp.concatenate([a for a, _ in parts]), jnp.concatenate([b for _, b in parts])

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
        if self.periodic_bounds is not None:
            selected_backend="jax"
        if selected_backend == "warp" and not bool(jnp.all(point_active)):
            selected_backend = "jax"
        if selected_backend == "warp":
            try:
                idx_rows, dist_rows = warp_exact_ball(np.asarray(query_points, dtype=float), np.asarray(points, dtype=float), float(radius))
                return [jnp.asarray(ids, dtype=int) for ids in idx_rows], [jnp.asarray(d, dtype=float) for d in dist_rows]
            except WarpUnavailableError:
                if backend != "auto":
                    raise
        d = jnp.where(point_active[None, :], self._query_distances(points,query_points), jnp.inf)
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
        d = jnp.where(point_active[None, :], self._query_distances(points,query_points), jnp.inf)
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

    def tree_indices_for_globals(self, tree_mode, global_ids):
        """Map one-based global node identities to active tree rows on device.

        A center outside this tree (or an inactive center) has index -1.
        Coordinate equality never substitutes for node identity.
        """
        tree_globals = self.get_tree_globals(tree_mode)
        global_ids = jnp.asarray(global_ids)
        if tree_globals.size == 0:
            return jnp.full(global_ids.shape, -1, dtype=jnp.int32)
        # Every supported tree has contiguous global numbering; avoid sorting
        # or an N-by-N coordinate comparison for this identity-only mapping.
        indices = global_ids - tree_globals[0]
        clipped = jnp.clip(indices, 0, tree_globals.size - 1)
        found = (indices >= 0) & (indices < tree_globals.size)
        found = found & self.get_tree_active_mask(tree_mode)[clipped]
        return jnp.where(found, indices, -1).astype(jnp.int32)

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
