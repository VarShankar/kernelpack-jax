from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from kernelpack.domain import DualNodeDomainDescriptor

from .core import DomainNodeGenerator


@dataclass
class DualNodeDomainGenerator:
    velocity_generator: DomainNodeGenerator = field(default_factory=DomainNodeGenerator)
    pressure_generator: DomainNodeGenerator = field(default_factory=DomainNodeGenerator)

    def generate_smooth_domain_nodes(self, geometry: object, velocity_radius: float, pressure_radius: float, **kwargs: object) -> None:
        self.velocity_generator.build_domain_descriptor_from_geometry(geometry, velocity_radius, **kwargs)
        self.pressure_generator.build_domain_descriptor_from_geometry(geometry, pressure_radius, **kwargs)

    def generate_smooth_domain_nodes_auto_pressure(
        self,
        geometry: object,
        velocity_radius: float,
        max_pressure_fraction: float,
        min_pressure_nodes: int = 0,
        **kwargs: object,
    ) -> dict[str, object]:
        if not (0.0 < max_pressure_fraction < 1.0):
            raise ValueError("max_pressure_fraction must lie strictly between 0 and 1")
        self.velocity_generator.build_domain_descriptor_from_geometry(geometry, velocity_radius, **kwargs)
        velocity_descriptor = self.velocity_generator.get_domain_descriptor()
        velocity_nodes = velocity_descriptor.get_num_int_bdry_nodes()
        pressure_target = max(1, int(np.floor(max_pressure_fraction * velocity_nodes)))
        if pressure_target < min_pressure_nodes:
            raise ValueError("requested pressure-node fraction is incompatible with the stencil-size requirement")

        low_radius = max(float(velocity_radius), 1.0e-8)
        high_radius = 1.25 * low_radius
        pressure_count = np.inf
        expansion_iters = 0
        while pressure_count > pressure_target and expansion_iters < 30:
            self.pressure_generator.build_domain_descriptor_from_geometry(geometry, high_radius, **kwargs)
            pressure_count = self.pressure_generator.get_domain_descriptor().get_num_int_bdry_nodes()
            if pressure_count > pressure_target:
                low_radius = high_radius
                high_radius = 1.2 * high_radius
            expansion_iters += 1
        if pressure_count > pressure_target:
            raise ValueError("unable to select a pressure node cloud satisfying the requested density ratio")

        best_radius = high_radius
        for _ in range(20):
            trial_radius = 0.5 * (low_radius + high_radius)
            self.pressure_generator.build_domain_descriptor_from_geometry(geometry, trial_radius, **kwargs)
            trial_count = self.pressure_generator.get_domain_descriptor().get_num_int_bdry_nodes()
            if trial_count <= pressure_target:
                best_radius = trial_radius
                high_radius = trial_radius
            else:
                low_radius = trial_radius

        self.pressure_generator.build_domain_descriptor_from_geometry(geometry, best_radius, **kwargs)
        while self.pressure_generator.get_domain_descriptor().get_num_int_bdry_nodes() > pressure_target:
            best_radius = 1.02 * best_radius
            self.pressure_generator.build_domain_descriptor_from_geometry(geometry, best_radius, **kwargs)
        best_count = self.pressure_generator.get_domain_descriptor().get_num_int_bdry_nodes()
        if best_count < min_pressure_nodes:
            raise ValueError("automatic pressure coarsening produced too few pressure nodes for the requested stencil")
        return {
            "pressure_radius": best_radius,
            "num_velocity_nodes": velocity_nodes,
            "num_pressure_nodes": best_count,
        }

    def create_dual_node_domain_descriptor(self, strip_pressure_ghosts: bool = True) -> DualNodeDomainDescriptor:
        dual = DualNodeDomainDescriptor()
        dual.set_velocity_domain(self.velocity_generator.get_domain_descriptor())
        pressure_domain = self.pressure_generator.get_domain_descriptor()
        if strip_pressure_ghosts:
            pressure_domain = _strip_ghosts(pressure_domain)
        dual.set_pressure_domain(pressure_domain)
        return dual

    def get_velocity_generator(self) -> DomainNodeGenerator:
        return self.velocity_generator

    def get_pressure_generator(self) -> DomainNodeGenerator:
        return self.pressure_generator


def _strip_ghosts(domain):
    stripped = domain.__class__()
    dim = domain.get_dim()
    stripped.set_nodes(domain.get_interior_nodes(), domain.get_bdry_nodes(), jnp.zeros((0, dim)))
    stripped.set_normals(domain.get_nrmls())
    stripped.set_sep_rad(domain.get_sep_rad())
    stripped.set_outer_level_set(domain.get_outer_level_set())
    stripped.set_boundary_level_sets(domain.get_boundary_level_sets())
    stripped.build_structs()
    return stripped
