from __future__ import annotations

from dataclasses import dataclass, field

from jax import jit
import jax.numpy as jnp
import numpy as np

from kernelpack.geometry import distance_matrix

from .core import DomainDescriptor


@jit
def _outside_inflated_boundary(
    points: jnp.ndarray,
    boundary: jnp.ndarray,
    outward_normals: jnp.ndarray,
    h: float,
) -> jnp.ndarray:
    inflated = boundary + 0.75 * h * outward_normals
    nearest = jnp.argmin(distance_matrix(points, inflated), axis=1)
    displacement = points - inflated[nearest]
    return jnp.sum(displacement * outward_normals[nearest], axis=1) > 1e-12


def _stable_entered(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    return current[~np.isin(current, previous)]


def _stable_persisted(current: np.ndarray, previous: np.ndarray) -> np.ndarray:
    return current[np.isin(current, previous)]


def _generated_ids(points: jnp.ndarray, offset: np.int64) -> np.ndarray:
    points_np = np.asarray(points, dtype=float)
    ids = np.zeros(points_np.shape[0], dtype=np.int64)
    for row, point in enumerate(points_np):
        quantized = np.rint(point * 1e10).astype(np.int64)
        powers = np.asarray([104729**k for k in range(point.size)], dtype=object)
        hashed = int(abs(np.sum(quantized.astype(object) * powers)) % int(8e14))
        ids[row] = offset + np.int64(hashed)
    return ids


@dataclass
class MovingDomainDescriptor:
    """Active snapshot of a fixed background domain with moving holes."""

    background_domain: DomainDescriptor
    current_domain: DomainDescriptor = field(init=False)
    discretization_domain: DomainDescriptor = field(init=False)
    current_physical_ids: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,), dtype=jnp.int64))
    current_all_ids: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,), dtype=jnp.int64))
    entered_physical_ids: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,), dtype=jnp.int64))
    left_physical_ids: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,), dtype=jnp.int64))
    persisted_physical_ids: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,), dtype=jnp.int64))
    entered_all_ids: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,), dtype=jnp.int64))
    left_all_ids: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,), dtype=jnp.int64))
    persisted_all_ids: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,), dtype=jnp.int64))
    active_background_interior: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,), dtype=int))
    active_background_boundary: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,), dtype=int))
    active_background_ghost: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,), dtype=int))
    moving_interior_nodes: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    moving_boundary_nodes: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    moving_boundary_normals: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    moving_ghost_nodes: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    discretization_physical_ids: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,), dtype=jnp.int64))
    discretization_all_ids: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0,), dtype=jnp.int64))
    fixed_moving_boundary_capacity: int | None = None
    capacity_growth_count: int = 0
    _background_interior_ids: np.ndarray = field(init=False, repr=False)
    _background_boundary_ids: np.ndarray = field(init=False, repr=False)
    _background_ghost_ids: np.ndarray = field(init=False, repr=False)
    _moving_interior_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64), repr=False)
    _moving_boundary_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64), repr=False)
    _moving_ghost_ids: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int64), repr=False)

    def __post_init__(self) -> None:
        source = self.background_domain
        h = source.get_sep_rad()
        normalized = DomainDescriptor()
        outer_ghost = source.get_bdry_nodes() + 0.25 * h * source.get_nrmls()
        normalized.set_nodes(source.get_interior_nodes(), source.get_bdry_nodes(), outer_ghost)
        normalized.set_normals(source.get_nrmls())
        normalized.set_sep_rad(h)
        normalized.set_outer_level_set(source.get_outer_level_set())
        normalized.set_boundary_level_sets(source.get_boundary_level_sets())
        normalized.build_structs()
        self.background_domain = normalized

        ni = normalized.get_num_interior_nodes()
        nb = normalized.get_num_bdry_nodes()
        ng = normalized.get_ghost_nodes().shape[0]
        ids = np.arange(1, ni + nb + ng + 1, dtype=np.int64)
        self._background_interior_ids = ids[:ni]
        self._background_boundary_ids = ids[ni : ni + nb]
        self._background_ghost_ids = ids[ni + nb :]
        dim = normalized.get_dim()
        self.update_state(
            np.arange(ni, dtype=int),
            np.arange(nb, dtype=int),
            np.arange(ng, dtype=int),
            jnp.zeros((0, dim)),
            jnp.zeros((0, dim)),
            jnp.zeros((0, dim)),
            jnp.zeros((0, dim)),
        )

    def configure_fixed_capacity(self, moving_boundary_capacity: int) -> None:
        capacity = int(moving_boundary_capacity)
        current_count = int(self.moving_boundary_nodes.shape[0])
        if capacity < current_count:
            raise ValueError("moving-boundary capacity cannot be smaller than the current boundary")
        self.fixed_moving_boundary_capacity = capacity
        self._update_discretization_domain()

    def update_from_embedded_surfaces(self, surfaces: object | list[object] | tuple[object, ...]) -> None:
        if surfaces is None:
            surface_list: list[object] = []
        elif isinstance(surfaces, (list, tuple)):
            surface_list = list(surfaces)
        else:
            surface_list = [surfaces]

        background = self.background_domain
        h = background.get_sep_rad()
        layer_offset = 0.5 * h
        background_interior = background.get_interior_nodes()
        keep = jnp.ones((background_interior.shape[0],), dtype=bool)
        for surface in surface_list:
            keep = keep & _outside_inflated_boundary(
                background_interior,
                surface.get_uniform_sample_sites(),
                surface.get_uniform_nrmls(),
                h,
            )
        active_interior = np.flatnonzero(np.asarray(keep))
        active_boundary = np.arange(background.get_num_bdry_nodes(), dtype=int)
        active_ghost = np.arange(background.get_ghost_nodes().shape[0], dtype=int)

        dim = background.get_dim()
        moving_interior_parts: list[jnp.ndarray] = []
        moving_boundary_parts: list[jnp.ndarray] = []
        moving_normal_parts: list[jnp.ndarray] = []
        moving_interior_ids: list[np.ndarray] = []
        moving_boundary_ids: list[np.ndarray] = []
        moving_ghost_ids: list[np.ndarray] = []
        id_base = np.int64(background.get_num_total_nodes())
        for surface_index, surface in enumerate(surface_list, start=1):
            boundary = jnp.asarray(surface.get_uniform_sample_sites(), dtype=float)
            domain_normals = -jnp.asarray(surface.get_uniform_nrmls(), dtype=float)
            count = int(boundary.shape[0])
            boundary_ids = id_base + np.int64(surface_index * 1_000_000) + np.arange(1, count + 1, dtype=np.int64)
            moving_boundary_parts.append(boundary)
            moving_normal_parts.append(domain_normals)
            moving_boundary_ids.append(boundary_ids)
            moving_ghost_ids.append(np.int64(1_000_000_000) + boundary_ids)
            moving_interior_parts.append(boundary - layer_offset * domain_normals)
            moving_interior_ids.append(
                id_base
                + np.int64(100_000_000 + surface_index * 1_000_000)
                + np.arange(1, count + 1, dtype=np.int64)
            )

        moving_interior = (
            jnp.asarray(np.vstack([np.asarray(part) for part in moving_interior_parts]))
            if moving_interior_parts
            else jnp.zeros((0, dim))
        )
        moving_boundary = (
            jnp.asarray(np.vstack([np.asarray(part) for part in moving_boundary_parts]))
            if moving_boundary_parts
            else jnp.zeros((0, dim))
        )
        moving_normals = (
            jnp.asarray(np.vstack([np.asarray(part) for part in moving_normal_parts]))
            if moving_normal_parts
            else jnp.zeros((0, dim))
        )
        moving_ghost = jnp.asarray(
            np.asarray(moving_boundary) + layer_offset * np.asarray(moving_normals)
        )
        self.update_state(
            active_interior,
            active_boundary,
            active_ghost,
            moving_interior,
            moving_boundary,
            moving_normals,
            moving_ghost,
            np.concatenate(moving_interior_ids) if moving_interior_ids else None,
            np.concatenate(moving_boundary_ids) if moving_boundary_ids else None,
            np.concatenate(moving_ghost_ids) if moving_ghost_ids else None,
        )

    def update_state(
        self,
        active_interior: np.ndarray,
        active_boundary: np.ndarray,
        active_ghost: np.ndarray,
        moving_interior: jnp.ndarray,
        moving_boundary: jnp.ndarray,
        moving_normals: jnp.ndarray,
        moving_ghost: jnp.ndarray,
        moving_interior_ids: np.ndarray | None = None,
        moving_boundary_ids: np.ndarray | None = None,
        moving_ghost_ids: np.ndarray | None = None,
    ) -> None:
        background = self.background_domain
        active_interior = np.unique(np.asarray(active_interior, dtype=int))
        active_boundary = np.unique(np.asarray(active_boundary, dtype=int))
        active_ghost = np.unique(np.asarray(active_ghost, dtype=int))
        dim = background.get_dim()
        moving_interior = jnp.asarray(np.asarray(moving_interior, dtype=float).reshape((-1, dim)))
        moving_boundary = jnp.asarray(np.asarray(moving_boundary, dtype=float).reshape((-1, dim)))
        moving_normals = jnp.asarray(np.asarray(moving_normals, dtype=float).reshape((-1, dim)))
        moving_ghost = jnp.asarray(np.asarray(moving_ghost, dtype=float).reshape((-1, dim)))
        if moving_boundary.shape[0] != moving_normals.shape[0] or moving_boundary.shape[0] != moving_ghost.shape[0]:
            raise ValueError("moving boundary nodes, normals, and ghosts must align")

        base_count = np.int64(background.get_num_total_nodes())
        if moving_interior_ids is None:
            moving_interior_ids = _generated_ids(moving_interior, base_count + np.int64(100_000_000))
        if moving_boundary_ids is None:
            moving_boundary_ids = base_count + np.arange(1, moving_boundary.shape[0] + 1, dtype=np.int64)
        if moving_ghost_ids is None:
            moving_ghost_ids = np.int64(1_000_000_000) + moving_boundary_ids

        interior = np.vstack(
            [np.asarray(background.get_interior_nodes())[active_interior], np.asarray(moving_interior)]
        )
        boundary = np.vstack(
            [np.asarray(background.get_bdry_nodes())[active_boundary], np.asarray(moving_boundary)]
        )
        normals = np.vstack(
            [np.asarray(background.get_nrmls())[active_boundary], np.asarray(moving_normals)]
        )
        ghosts = np.vstack(
            [np.asarray(background.get_ghost_nodes())[active_ghost], np.asarray(moving_ghost)]
        )
        current = DomainDescriptor()
        current.set_nodes_from_host(interior, boundary, ghosts)
        current.set_normals(normals)
        current.set_sep_rad(background.get_sep_rad())
        current.set_outer_level_set(background.get_outer_level_set())
        current.set_boundary_level_sets(background.get_boundary_level_sets())
        current.build_structs()

        interior_ids = np.concatenate([self._background_interior_ids[active_interior], np.asarray(moving_interior_ids)])
        boundary_ids = np.concatenate([self._background_boundary_ids[active_boundary], np.asarray(moving_boundary_ids)])
        ghost_ids = np.concatenate([self._background_ghost_ids[active_ghost], np.asarray(moving_ghost_ids)])
        previous_physical = np.asarray(self.current_physical_ids, dtype=np.int64)
        previous_all = np.asarray(self.current_all_ids, dtype=np.int64)
        current_physical = np.concatenate([interior_ids, boundary_ids])
        current_all = np.concatenate([current_physical, ghost_ids])

        self.current_domain = current
        self.current_physical_ids = jnp.asarray(current_physical, dtype=jnp.int64)
        self.current_all_ids = jnp.asarray(current_all, dtype=jnp.int64)
        self.entered_physical_ids = jnp.asarray(_stable_entered(current_physical, previous_physical), dtype=jnp.int64)
        self.left_physical_ids = jnp.asarray(_stable_entered(previous_physical, current_physical), dtype=jnp.int64)
        self.persisted_physical_ids = jnp.asarray(_stable_persisted(current_physical, previous_physical), dtype=jnp.int64)
        self.entered_all_ids = jnp.asarray(_stable_entered(current_all, previous_all), dtype=jnp.int64)
        self.left_all_ids = jnp.asarray(_stable_entered(previous_all, current_all), dtype=jnp.int64)
        self.persisted_all_ids = jnp.asarray(_stable_persisted(current_all, previous_all), dtype=jnp.int64)
        self.active_background_interior = jnp.asarray(active_interior, dtype=int)
        self.active_background_boundary = jnp.asarray(active_boundary, dtype=int)
        self.active_background_ghost = jnp.asarray(active_ghost, dtype=int)
        self.moving_interior_nodes = moving_interior
        self.moving_boundary_nodes = moving_boundary
        self.moving_boundary_normals = moving_normals
        self.moving_ghost_nodes = moving_ghost
        self._moving_interior_ids = np.asarray(moving_interior_ids, dtype=np.int64)
        self._moving_boundary_ids = np.asarray(moving_boundary_ids, dtype=np.int64)
        self._moving_ghost_ids = np.asarray(moving_ghost_ids, dtype=np.int64)
        self._update_discretization_domain()

    def _update_discretization_domain(self) -> None:
        if self.fixed_moving_boundary_capacity is None:
            self.discretization_domain = self.current_domain
            self.discretization_physical_ids = self.current_physical_ids
            self.discretization_all_ids = self.current_all_ids
            return

        background = self.background_domain
        capacity = self.fixed_moving_boundary_capacity
        moving_count = int(self.moving_boundary_nodes.shape[0])
        if moving_count > capacity:
            capacity = max(moving_count, int(np.ceil(1.5 * capacity)))
            self.fixed_moving_boundary_capacity = capacity
            self.capacity_growth_count += 1
        dim = background.get_dim()
        ni = background.get_num_interior_nodes()
        nb = background.get_num_bdry_nodes()
        ng = background.get_ghost_nodes().shape[0]

        active_interior = np.zeros((ni,), dtype=bool)
        active_interior[np.asarray(self.active_background_interior, dtype=int)] = True
        active_boundary = np.zeros((nb,), dtype=bool)
        active_boundary[np.asarray(self.active_background_boundary, dtype=int)] = True
        active_ghost = np.zeros((ng,), dtype=bool)
        active_ghost[np.asarray(self.active_background_ghost, dtype=int)] = True
        moving_active = np.arange(capacity) < moving_count

        def padded_rows(values: jnp.ndarray) -> np.ndarray:
            padded = np.zeros((capacity, dim), dtype=float)
            if moving_count:
                padded[:moving_count] = np.asarray(values, dtype=float)
            return padded

        interior = np.vstack([np.asarray(background.get_interior_nodes()), padded_rows(self.moving_interior_nodes)])
        boundary = np.vstack([np.asarray(background.get_bdry_nodes()), padded_rows(self.moving_boundary_nodes)])
        ghosts = np.vstack([np.asarray(background.get_ghost_nodes()), padded_rows(self.moving_ghost_nodes)])
        normals = np.vstack([np.asarray(background.get_nrmls()), padded_rows(self.moving_boundary_normals)])
        interior_mask = np.concatenate([active_interior, moving_active])
        boundary_mask = np.concatenate([active_boundary, moving_active])
        ghost_mask = np.concatenate([active_ghost, moving_active])

        padded = DomainDescriptor()
        padded.set_nodes(interior, boundary, ghosts)
        padded.set_normals(normals)
        padded.set_active_masks(interior_mask, boundary_mask, ghost_mask)
        padded.set_sep_rad(background.get_sep_rad())
        padded.set_outer_level_set(background.get_outer_level_set())
        padded.set_boundary_level_sets(background.get_boundary_level_sets())
        padded.build_structs()

        def padded_ids(values: np.ndarray, offset: int) -> np.ndarray:
            result = -(np.int64(offset) + np.arange(1, capacity + 1, dtype=np.int64))
            result[: values.size] = values
            return result

        interior_ids = np.concatenate(
            [self._background_interior_ids, padded_ids(self._moving_interior_ids, 100_000_000)]
        )
        boundary_ids = np.concatenate(
            [self._background_boundary_ids, padded_ids(self._moving_boundary_ids, 200_000_000)]
        )
        ghost_ids = np.concatenate(
            [self._background_ghost_ids, padded_ids(self._moving_ghost_ids, 300_000_000)]
        )
        self.discretization_domain = padded
        self.discretization_physical_ids = jnp.asarray(np.concatenate([interior_ids, boundary_ids]))
        self.discretization_all_ids = jnp.asarray(np.concatenate([interior_ids, boundary_ids, ghost_ids]))

    def get_current_domain(self) -> DomainDescriptor:
        return self.current_domain

    def get_background_domain(self) -> DomainDescriptor:
        return self.background_domain

    def get_discretization_domain(self) -> DomainDescriptor:
        return self.discretization_domain

    def get_discretization_physical_ids(self) -> jnp.ndarray:
        return self.discretization_physical_ids

    def get_discretization_all_ids(self) -> jnp.ndarray:
        return self.discretization_all_ids

    def get_current_physical_ids(self) -> jnp.ndarray:
        return self.current_physical_ids

    def get_current_all_ids(self) -> jnp.ndarray:
        return self.current_all_ids

    def get_entered_physical_ids(self) -> jnp.ndarray:
        return self.entered_physical_ids

    def get_left_physical_ids(self) -> jnp.ndarray:
        return self.left_physical_ids

    def get_persisted_physical_ids(self) -> jnp.ndarray:
        return self.persisted_physical_ids

    def get_entered_all_ids(self) -> jnp.ndarray:
        return self.entered_all_ids

    def get_left_all_ids(self) -> jnp.ndarray:
        return self.left_all_ids

    def get_persisted_all_ids(self) -> jnp.ndarray:
        return self.persisted_all_ids
