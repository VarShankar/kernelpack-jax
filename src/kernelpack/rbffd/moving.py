from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
import os

from jax import jit, lax
import jax.numpy as jnp

from kernelpack.accelerators import warp_available
from kernelpack.domain import DomainDescriptor, MovingDomainDescriptor

from .core import (
    OpProperties,
    StencilProperties,
    _build_legendre_basis_data,
    _evaluate_factored_rbf_interpolants,
    _factor_batched_rbf_interpolants,
    _pick_centers,
    _batched_rbf_weights,
)


@partial(jit, static_argnames=("batch_size",))
def _fresh_row_batches(active, reusable, *, batch_size):
    """Fixed-capacity device compaction; sentinel rows are dropped on scatter."""
    count=active.shape[0]
    rows=jnp.nonzero(active & ~reusable,size=count,fill_value=count)[0]
    rows=jnp.pad(rows,(0,(-count)%batch_size),constant_values=count)
    return rows.reshape(-1,batch_size)


def _interpolation_signature(properties):
    """Snapshot method parameters; the properties object itself is mutable."""
    return (properties.n,properties.dim,properties.ell,properties.spline_degree)


@jit
def _match_previous_rows(
    current_ids: jnp.ndarray,
    previous_ids: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    order = jnp.argsort(previous_ids)
    sorted_ids = previous_ids[order]
    positions = jnp.searchsorted(sorted_ids, current_ids, side="left")
    clipped = jnp.minimum(positions, sorted_ids.size - 1)
    found = (positions < sorted_ids.size) & (sorted_ids[clipped] == current_ids)
    return jnp.where(found, order[clipped], 0), found


@partial(jit, static_argnames=("use_coefficients",))
def _reusable_mask(
    target_ids: jnp.ndarray,
    tree_ids: jnp.ndarray,
    target_points: jnp.ndarray,
    stencil_points: jnp.ndarray,
    stencil_ids: jnp.ndarray,
    current_neu: jnp.ndarray,
    current_dir: jnp.ndarray,
    previous_target_ids: jnp.ndarray,
    previous_tree_ids: jnp.ndarray,
    previous_target_points: jnp.ndarray,
    previous_stencil_points: jnp.ndarray,
    previous_knn_indices: jnp.ndarray,
    previous_neu: jnp.ndarray,
    previous_dir: jnp.ndarray,
    *,
    use_coefficients: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    old_rows, candidates = _match_previous_rows(target_ids, previous_target_ids)
    same_centers = jnp.all(target_points == previous_target_points[old_rows], axis=1)
    previous_stencil_ids = previous_tree_ids[previous_knn_indices[old_rows]]
    same_ids = jnp.all(stencil_ids == previous_stencil_ids, axis=1)
    same_points = jnp.all(
        stencil_points == previous_stencil_points[old_rows],
        axis=(1, 2),
    )
    if use_coefficients:
        same_coefficients = (
            (current_neu == previous_neu[old_rows])
            & (current_dir == previous_dir[old_rows])
        )
    else:
        same_coefficients = jnp.ones_like(candidates)
    return candidates & same_centers & same_ids & same_points & same_coefficients, old_rows


def _neighbor_backend() -> str:
    requested = os.environ.get("KERNELPACK_JAX_NEIGHBOR_BACKEND", "").strip().lower()
    if requested in {"jax", "warp"}:
        return requested
    return "warp" if warp_available() else "jax"


def _point_set_data(
    moving_domain: MovingDomainDescriptor,
    point_set: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    domain = moving_domain.get_discretization_domain()
    mode = StencilProperties.normalize_point_set(point_set)
    physical_ids = moving_domain.get_discretization_physical_ids()
    if mode == "interior_boundary":
        return domain.get_int_bdry_nodes(), physical_ids, domain.get_physical_active_mask()
    if mode == "boundary":
        ni = domain.get_num_interior_nodes()
        return domain.get_bdry_nodes(), physical_ids[ni:], domain.get_boundary_active_mask()
    if mode == "all":
        return domain.get_all_nodes(), moving_domain.get_discretization_all_ids(), domain.get_all_active_mask()
    raise ValueError(f"unknown point set {point_set}")

def _tree_data(
    moving_domain: MovingDomainDescriptor,
    tree_mode: str,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    domain = moving_domain.get_discretization_domain()
    mode = StencilProperties.normalize_tree_mode(tree_mode)
    physical_ids = moving_domain.get_discretization_physical_ids()
    if mode == "interior_boundary":
        return domain.get_int_bdry_nodes(), physical_ids, domain.get_physical_active_mask()
    if mode == "boundary":
        ni = domain.get_num_interior_nodes()
        return domain.get_bdry_nodes(), physical_ids[ni:], domain.get_boundary_active_mask()
    if mode == "all":
        return domain.get_all_nodes(), moving_domain.get_discretization_all_ids(), domain.get_all_active_mask()
    raise ValueError(f"unknown tree mode {tree_mode}")


@dataclass
class MovingDifferentiationUpdater:
    """Selectively refresh rows of a moving RBF-FD operator.

    Identity checks, row compaction and fresh-row assembly stay on device.
    Fixed batches reuse the shared JAX RBF-FD weight kernels.
    """

    stencil_properties: StencilProperties
    operator: str = "lap"
    point_set: str = "interior_boundary"
    tree_mode: str = "all"
    assembly_mode: str = "fd"
    operator_properties: OpProperties = field(
        default_factory=lambda: OpProperties(decompose=False, store_weights=True, record_stencils=False)
    )
    weights: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    knn_indices: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0), dtype=int))
    target_ids: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,), dtype=jnp.int64))
    tree_ids: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,), dtype=jnp.int64))
    target_points: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    stencil_points: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0, 0)))
    neu_coeff: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    dir_coeff: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)))
    last_reused_row_count: jnp.ndarray | int = 0
    last_recomputed_row_count: jnp.ndarray | int = 0
    assembly_batch_size: int = 128
    _method_signature: tuple | None = field(default=None,init=False,repr=False)

    def __post_init__(self) -> None:
        self.operator = str(self.operator).lower()
        self.point_set = StencilProperties.normalize_point_set(self.point_set)
        self.tree_mode = StencilProperties.normalize_tree_mode(self.tree_mode)
        self.assembly_mode = str(self.assembly_mode).lower()
        if self.assembly_mode not in {"fd", "standard", "fdo", "overlapped"}:
            raise ValueError("assembly_mode must be 'fd' or 'fdo'")
        self.stencil_properties.point_set = self.point_set
        self.stencil_properties.tree_mode = self.tree_mode

    def initialize(
        self,
        moving_domain: MovingDomainDescriptor,
        *,
        neu_coeff: jnp.ndarray | None = None,
        dir_coeff: jnp.ndarray | None = None,
    ) -> None:
        self._refresh(moving_domain, neu_coeff=neu_coeff, dir_coeff=dir_coeff, allow_reuse=False)

    def update(
        self,
        moving_domain: MovingDomainDescriptor,
        *,
        neu_coeff: jnp.ndarray | None = None,
        dir_coeff: jnp.ndarray | None = None,
    ) -> None:
        self._refresh(moving_domain, neu_coeff=neu_coeff, dir_coeff=dir_coeff, allow_reuse=True)

    def _refresh(
        self,
        moving_domain: MovingDomainDescriptor,
        *,
        neu_coeff: jnp.ndarray | None,
        dir_coeff: jnp.ndarray | None,
        allow_reuse: bool,
    ) -> None:
        domain = moving_domain.get_discretization_domain()
        target_points, target_ids, target_active = _point_set_data(moving_domain, self.point_set)
        tree_points, tree_ids, _tree_active = _tree_data(moving_domain, self.tree_mode)
        _, _, center_globals, _ = _pick_centers(domain, self.point_set)
        center_indices = domain.tree_indices_for_globals(self.tree_mode, center_globals)
        center_indices = jnp.where(target_active, center_indices, -1)
        knn_indices, _ = domain.query_knn(
            self.tree_mode,
            target_points,
            self.stencil_properties.n,
            backend=_neighbor_backend(),
            center_indices=center_indices,
        )
        stencil_points = domain.stencil_coordinates(self.tree_mode,target_points,knn_indices)
        stencil_ids = tree_ids[knn_indices]
        row_count = target_points.shape[0]
        stencil_size = knn_indices.shape[1]
        signature=(_interpolation_signature(self.stencil_properties),self.operator,
            self.point_set,self.tree_mode,self.assembly_mode,self.operator_properties.selectdim)

        use_boundary = self.operator in {"bc", "boundary"}
        current_neu = jnp.zeros((row_count,), dtype=float) if neu_coeff is None else jnp.asarray(neu_coeff, dtype=float).reshape(-1)
        current_dir = jnp.zeros((row_count,), dtype=float) if dir_coeff is None else jnp.asarray(dir_coeff, dtype=float).reshape(-1)
        if use_boundary and (current_neu.size != row_count or current_dir.size != row_count):
            raise ValueError("boundary coefficients must match the moving boundary row count")

        reusable = jnp.zeros((row_count,), dtype=bool)
        old_rows = jnp.zeros((row_count,), dtype=int)
        reuse_method=signature==self._method_signature
        if allow_reuse and reuse_method and self.target_ids.size and self.weights.shape[1] == stencil_size:
            reusable, old_rows = _reusable_mask(
                target_ids,
                tree_ids,
                target_points,
                stencil_points,
                stencil_ids,
                current_neu,
                current_dir,
                self.target_ids,
                self.tree_ids,
                self.target_points,
                self.stencil_points,
                self.knn_indices,
                self.neu_coeff,
                self.dir_coeff,
                use_coefficients=use_boundary,
            )
            reusable = reusable & target_active

        weights = jnp.zeros((row_count, stencil_size), dtype=float)
        if allow_reuse and reuse_method and self.target_ids.size and self.weights.shape[1] == stencil_size:
            weights = jnp.where(reusable[:, None], self.weights[old_rows], weights)
        if row_count:
            batches=_fresh_row_batches(target_active,reusable,batch_size=max(1,int(self.assembly_batch_size)))
            def step(current,rows):
                def assemble(value):
                    safe=jnp.where(rows<row_count,rows,rows[0])
                    batch_weights=self._assemble_rows(domain,safe,knn_indices[safe],
                        current_neu if use_boundary else None,current_dir if use_boundary else None)
                    return value.at[rows].set(batch_weights,mode="drop")
                return lax.cond(jnp.any(rows<row_count),assemble,lambda value:value,current),None
            weights=lax.scan(step,weights,batches)[0]

        self.weights = weights
        self.knn_indices = knn_indices
        self.target_ids = target_ids
        self.tree_ids = tree_ids
        self.target_points = target_points
        self.stencil_points = stencil_points
        self.neu_coeff = current_neu
        self.dir_coeff = current_dir
        self.last_reused_row_count = jnp.count_nonzero(reusable)
        self.last_recomputed_row_count = jnp.count_nonzero(target_active & ~reusable)
        self._method_signature=signature

    def _assemble_rows(
        self,
        domain: DomainDescriptor,
        active_rows: jnp.ndarray,
        knn_indices: jnp.ndarray,
        neu_coeff: jnp.ndarray | None,
        dir_coeff: jnp.ndarray | None,
    ) -> jnp.ndarray:
        centers,_,_,normals=_pick_centers(domain,self.point_set)
        coordinates=domain.stencil_coordinates(self.tree_mode,centers[active_rows],knn_indices)
        values=_batched_rbf_weights(coordinates,self.stencil_properties,self.operator,
            self.operator_properties,
            normals=normals[active_rows] if normals is not None else None,
            neu_coeff=neu_coeff[active_rows] if neu_coeff is not None else None,
            dir_coeff=dir_coeff[active_rows] if dir_coeff is not None else None)
        return values.reshape((active_rows.size,knn_indices.shape[1]))

    def get_sparse_op(self):
        from kernelpack.solvers._common import SparseCOOMatrix

        row_count = int(self.weights.shape[0])
        col_count = int(self.tree_ids.size)
        rows = jnp.repeat(jnp.arange(row_count, dtype=int), self.weights.shape[1])
        cols = self.knn_indices.reshape(-1)
        return SparseCOOMatrix(
            indices=jnp.column_stack([rows, cols]),
            values=self.weights.reshape(-1),
            shape=(row_count, col_count),
        )

    def get_op(self) -> jnp.ndarray:
        sparse = self.get_sparse_op()
        return jnp.zeros(sparse.shape, dtype=float).at[sparse.indices[:, 0], sparse.indices[:, 1]].add(sparse.values)


@dataclass
class MovingDomainLocalInterpolator:
    """Local physical-node RBF interpolant with cached owner-stencil factors."""

    domain: DomainDescriptor
    stencil_properties: StencilProperties
    previous: "MovingDomainLocalInterpolator | None" = None
    source_ids: jnp.ndarray | None = None
    source_points: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)), init=False)
    stencil_indices: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0), dtype=int), init=False)
    local_coordinates: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,0,0)),init=False)
    lu: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0, 0)), init=False)
    pivots: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0), dtype=int), init=False)
    width: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,)), init=False)
    centers: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)), init=False)
    index_set: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0), dtype=int), init=False)
    recurrence_a: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)), init=False)
    recurrence_b: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)), init=False)
    last_reused_row_count: jnp.ndarray | int = field(default=0, init=False)
    last_recomputed_row_count: jnp.ndarray | int = field(default=0, init=False)
    factor_batch_size: int = 512
    _periodic_bounds: jnp.ndarray | None = field(default=None, init=False, repr=False)
    _periodic_dims: jnp.ndarray | None = field(default=None, init=False, repr=False)
    _method_signature: tuple | None = field(default=None,init=False,repr=False)

    def __post_init__(self) -> None:
        self._periodic_bounds = self.domain.periodic_bounds
        self._periodic_dims = self.domain.periodic_dims
        self._method_signature = _interpolation_signature(self.stencil_properties)
        self.source_points = self.domain.get_int_bdry_nodes()
        if self.source_ids is None:
            self.source_ids = jnp.arange(self.source_points.shape[0], dtype=jnp.int64)
        else:
            self.source_ids = jnp.asarray(self.source_ids, dtype=jnp.int64).reshape(-1)
        if self.source_ids.size != self.source_points.shape[0]:
            raise ValueError("source_ids must match the physical-node capacity")
        self.stencil_indices, _ = self.domain.query_knn(
            "interior_boundary",
            self.source_points,
            self.stencil_properties.n,
            backend=_neighbor_backend(),
            center_indices=jnp.where(self.domain.get_physical_active_mask(),
                jnp.arange(len(self.source_points),dtype=jnp.int32),-1),
        )
        stencil_points = self.domain.stencil_coordinates("interior_boundary",self.source_points,self.stencil_indices)
        self.local_coordinates=stencil_points
        self.index_set, self.recurrence_a, self.recurrence_b = _build_legendre_basis_data(
            self.stencil_properties.dim,
            self.stencil_properties.ell,
        )
        active = self.domain.get_physical_active_mask()
        reusable = jnp.zeros((self.source_points.shape[0],), dtype=bool)
        old_rows = jnp.zeros((self.source_points.shape[0],), dtype=int)
        reuse_method=(self.previous is not None and self.previous.source_ids.size
            and self.previous._method_signature==self._method_signature)
        if reuse_method:
            old_rows, found = _match_previous_rows(self.source_ids, self.previous.source_ids)
            same_centers = jnp.all(self.source_points == self.previous.source_points[old_rows], axis=1)
            stencil_ids = self.source_ids[self.stencil_indices]
            old_stencil_ids = self.previous.source_ids[self.previous.stencil_indices[old_rows]]
            same_ids = jnp.all(stencil_ids == old_stencil_ids, axis=1)
            old_stencil_points = self.previous.local_coordinates[old_rows]
            same_points = jnp.all(stencil_points == old_stencil_points, axis=(1, 2))
            reusable = (found & same_centers & same_ids & same_points) & active

        # Derive dimensions from the assembled basis, not the mutable npoly
        # convenience field (which may be stale after changing ell).
        system_size = self.stencil_indices.shape[1] + self.index_set.shape[0]
        self.lu = jnp.zeros((self.source_points.shape[0], system_size, system_size), dtype=float)
        self.pivots = jnp.zeros((self.source_points.shape[0], system_size), dtype=jnp.int32)
        self.width = jnp.ones((self.source_points.shape[0],), dtype=float)
        self.centers = jnp.zeros_like(self.source_points)
        if reuse_method:
            reusable_device = reusable
            old_lu = self.previous.lu[old_rows]
            old_pivots = self.previous.pivots[old_rows]
            old_width = self.previous.width[old_rows]
            old_centers = self.previous.centers[old_rows]
            self.lu = jnp.where(reusable_device[:, None, None], old_lu, self.lu)
            self.pivots = jnp.where(reusable_device[:, None], old_pivots, self.pivots)
            self.width = jnp.where(reusable_device, old_width, self.width)
            self.centers = jnp.where(reusable_device[:, None], old_centers, self.centers)

        batch_size = max(1, int(self.factor_batch_size))
        capacity=len(self.source_points)
        if capacity:
            batches=_fresh_row_batches(active,reusable,batch_size=batch_size)
            def step(state,rows):
                def factor(current):
                    safe=jnp.where(rows<capacity,rows,rows[0])
                    values=_factor_batched_rbf_interpolants(stencil_points[safe],self.index_set,
                        self.recurrence_a,self.recurrence_b,self.stencil_properties.spline_degree)
                    return tuple(old.at[rows].set(new,mode="drop") for old,new in zip(current,values))
                return lax.cond(jnp.any(rows<capacity),factor,lambda current:current,state),None
            self.lu,self.pivots,self.width,self.centers=lax.scan(step,
                (self.lu,self.pivots,self.width,self.centers),batches)[0]
        self.last_reused_row_count = jnp.count_nonzero(reusable)
        self.last_recomputed_row_count = jnp.count_nonzero(active & ~reusable)

    def evaluate(self, nodal_values: jnp.ndarray, query_points: jnp.ndarray) -> jnp.ndarray:
        bounds = self.domain.periodic_bounds
        dims = self.domain.periodic_dims
        changed = (bounds is None) != (self._periodic_bounds is None)
        if not changed and bounds is not None:
            changed = (bounds.shape != self._periodic_bounds.shape
                or dims is None or self._periodic_dims is None
                or dims.shape != self._periodic_dims.shape
                or not bool(jnp.all(bounds == self._periodic_bounds)
                    & jnp.all(dims == self._periodic_dims)))
        if changed:
            raise ValueError("Periodic configuration changed; rebuild the local interpolator")
        if self._method_signature != _interpolation_signature(self.stencil_properties):
            raise ValueError("Interpolation method changed; rebuild the local interpolator")
        nodal_values = jnp.asarray(nodal_values, dtype=float).reshape(-1)
        query_points = jnp.atleast_2d(jnp.asarray(query_points, dtype=float))
        if nodal_values.size != self.domain.get_num_int_bdry_nodes():
            raise ValueError("nodal values must match the source physical-node count")
        if query_points.shape[0] == 0:
            return jnp.zeros((0,), dtype=nodal_values.dtype)

        owner_indices, _ = self.domain.query_knn(
            "interior_boundary",
            query_points,
            1,
            backend=_neighbor_backend(),
        )
        owner_indices = owner_indices[:, 0]
        stencil_indices = self.stencil_indices[owner_indices]
        stencil_points = self.local_coordinates[owner_indices]
        if self._periodic_bounds is not None:
            from kernelpack.domain.stencils import nearest_periodic_images
            query_points=nearest_periodic_images(query_points[:,None,:],self.source_points[owner_indices],
                self._periodic_bounds[1]-self._periodic_bounds[0],self._periodic_dims)[:,0,:]
        weights = _evaluate_factored_rbf_interpolants(
            self.lu,
            self.pivots,
            self.width,
            self.centers,
            stencil_points,
            query_points,
            owner_indices,
            self.index_set,
            self.recurrence_a,
            self.recurrence_b,
            self.stencil_properties.spline_degree,
            self.stencil_indices.shape[1],
        )
        return jnp.sum(weights * nodal_values[stencil_indices], axis=1)
