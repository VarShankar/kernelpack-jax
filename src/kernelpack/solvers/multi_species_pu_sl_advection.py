from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax.numpy as jnp

from .pu_sl_advection import AdvectionBoundaryCondition, PUSLAdvectionSolver


@dataclass
class MultiSpeciesPUSLAdvectionSolver:
    num_species: int = 0
    solver: PUSLAdvectionSolver = field(default_factory=PUSLAdvectionSolver)

    def clear_advection_boundary_condition(self) -> None:
        self.solver.clear_advection_boundary_condition()

    def set_tangential_flow_boundary(self, normal_velocity_tolerance: float = 1.0e-10) -> None:
        self.solver.set_tangential_flow_boundary(normal_velocity_tolerance)

    def set_periodic_boundary(self, periodic_patches: object, normal_velocity_tolerance: float = 1.0e-10) -> None:
        self.solver.set_periodic_boundary(periodic_patches, normal_velocity_tolerance)

    def set_inflow_dirichlet_boundary(
        self,
        inflow_value: Callable[..., jnp.ndarray],
        normal_velocity_tolerance: float = 1.0e-10,
    ) -> None:
        self.solver.set_inflow_dirichlet_boundary(inflow_value, normal_velocity_tolerance)

    def get_advection_boundary_condition(self) -> AdvectionBoundaryCondition:
        return self.solver.get_advection_boundary_condition()

    def returns_distributed_state(self) -> bool:
        return self.solver.returns_distributed_state()

    def get_output_range(self) -> tuple[int, int]:
        return self.solver.get_output_range()

    def init(
        self,
        domain,
        xi: int,
        dlt: float,
        num_species: int,
        patch_spacing_factor: float = 0.0,
        patch_radius_factor: float = 0.0,
    ) -> None:
        if num_species <= 0:
            raise ValueError("MultiSpeciesPUSLAdvectionSolver requires a positive species count.")
        self.num_species = int(num_species)
        self.solver.init(domain, xi, dlt, patch_spacing_factor, patch_radius_factor)

    def get_num_species(self) -> int:
        return self.num_species

    def get_output_nodes(self) -> jnp.ndarray:
        return self.solver.get_output_nodes()

    def project_initial(self, rho0: Callable[[jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
        self._ensure_initialized()
        x = self.solver.get_output_nodes()
        values = jnp.asarray(rho0(x), dtype=float)
        if values.ndim == 1:
            values = values.reshape(x.shape[0], -1)
        if values.shape != (x.shape[0], self.num_species):
            raise ValueError("Initial callback must return an (n_nodes, n_species) array.")
        return self.solver.project_samples(values)

    def project_constant(self, value: float) -> jnp.ndarray:
        self._ensure_initialized()
        return self.solver.project_constant(value, self.num_species)

    def project_constants(self, values: jnp.ndarray) -> jnp.ndarray:
        self._ensure_initialized()
        values = jnp.asarray(values, dtype=float).reshape(1, -1)
        if values.shape[1] != self.num_species:
            raise ValueError("project_constants received the wrong number of species values.")
        return self.solver.project_constant(1.0, self.num_species) * values

    def project_samples(self, nodal_samples: jnp.ndarray) -> jnp.ndarray:
        self._ensure_initialized()
        self._validate_species_matrix(nodal_samples, "project_samples")
        return self.solver.project_samples(nodal_samples)

    def evaluate_at_nodes(self, coeffs: jnp.ndarray) -> jnp.ndarray:
        self._ensure_initialized()
        self._validate_species_matrix(coeffs, "evaluate_at_nodes")
        return self.solver.evaluate_at_nodes(coeffs)

    def evaluate_at_points(self, coeffs: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        self._ensure_initialized()
        self._validate_species_matrix(coeffs, "evaluate_at_points")
        return self.solver.evaluate_at_points(coeffs, x)

    def forward_sl_step(self, tn: float, coeffs_old: jnp.ndarray, velocity, rk) -> jnp.ndarray:
        self._ensure_initialized()
        self._validate_species_matrix(coeffs_old, "forward_sl_step")
        return self.solver.forward_sl_step(tn, coeffs_old, velocity, rk)

    def backward_sl_step(self, tn: float, coeffs_old: jnp.ndarray, velocity, rk) -> jnp.ndarray:
        self._ensure_initialized()
        self._validate_species_matrix(coeffs_old, "backward_sl_step")
        return self.solver.backward_sl_step(tn, coeffs_old, velocity, rk)

    def backward_sl_rollout(
        self,
        t0: float,
        coeffs0: jnp.ndarray,
        velocity,
        nsteps: int,
        rk=None,
        *,
        return_history: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        self._ensure_initialized()
        self._validate_species_matrix(coeffs0, "backward_sl_rollout")
        return self.solver.backward_sl_rollout(t0, coeffs0, velocity, nsteps, rk, return_history=return_history)

    def set_step_size(self, dlt: float) -> None:
        self.solver.set_step_size(dlt)

    def reset_solve_stats(self) -> None:
        self.solver.reset_solve_stats()

    def get_solve_stats(self) -> dict[str, object]:
        return self.solver.get_solve_stats()

    def get_num_patches(self) -> int:
        return self.solver.get_num_patches()

    def get_num_dofs_per_species(self) -> int:
        return self.solver.get_num_dofs()

    def get_total_dofs(self) -> int:
        self._ensure_initialized()
        return self.num_species * self.solver.get_num_dofs()

    def get_patch_radius(self) -> float:
        return self.solver.get_patch_radius()

    def get_patch_spacing(self) -> float:
        return self.solver.get_patch_spacing()

    def total_mass(self, coeffs: jnp.ndarray, species: int) -> float:
        self._ensure_initialized()
        self._validate_species_matrix(coeffs, "total_mass")
        return self.solver.total_mass(coeffs, species)

    def total_masses(self, coeffs: jnp.ndarray) -> jnp.ndarray:
        self._ensure_initialized()
        self._validate_species_matrix(coeffs, "total_masses")
        return jnp.asarray([self.solver.total_mass(coeffs, j) for j in range(self.num_species)], dtype=float)

    def get_domain_measure(self) -> float:
        return self.solver.get_domain_measure()

    def _ensure_initialized(self) -> None:
        if self.num_species <= 0:
            raise ValueError("MultiSpeciesPUSLAdvectionSolver must be initialized before use.")

    def _validate_species_matrix(self, values: jnp.ndarray, where: str) -> None:
        arr = jnp.asarray(values, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != self.num_species:
            raise ValueError(f"MultiSpeciesPUSLAdvectionSolver {where} received the wrong number of species columns.")
