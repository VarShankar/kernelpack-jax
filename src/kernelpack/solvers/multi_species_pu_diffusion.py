from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
import jax.numpy as jnp

from kernelpack.domain import DomainDescriptor
from .pu_diffusion import PUDiffusionSolver


@dataclass
class MultiSpeciesPUDiffusionSolver:
    domain: DomainDescriptor = field(default_factory=DomainDescriptor)
    xi: int = 0
    dt: float = 0.0
    nu: float = 0.0
    num_omp_threads: int = 1
    solvers: list[PUDiffusionSolver] = field(default_factory=list)
    num_species: int = 0

    def spawn_rollout_copy(self) -> MultiSpeciesPUDiffusionSolver:
        clone = copy(self)
        clone.solvers = [solver.spawn_rollout_copy() for solver in self.solvers]
        return clone

    def init(self, domain: DomainDescriptor, xi: int, dlt: float, d_coeff: float, num_omp_threads: int = 1) -> None:
        self.domain = domain
        self.xi = xi
        self.dt = dlt
        self.nu = d_coeff
        self.num_omp_threads = num_omp_threads
        self.solvers = []
        self.num_species = 0

    def set_step_size(self, dlt: float) -> None:
        self.dt = dlt
        for solver in self.solvers:
            solver.set_step_size(dlt)

    def set_initial_state(self, u0: jnp.ndarray) -> None:
        u0 = jnp.asarray(u0, dtype=float)
        self._ensure_solvers(int(u0.shape[1]))
        self.solvers[0].set_initial_state(u0)

    def set_state_history(self, *states: jnp.ndarray) -> None:
        states_2d = [jnp.asarray(state, dtype=float) for state in states]
        num_species = int(states_2d[0].shape[1])
        self._ensure_solvers(num_species)
        for state in states_2d[1:]:
            if int(state.shape[1]) != num_species:
                raise ValueError("All state-history matrices must have the same species count.")
        self.solvers[0].set_state_history(*states_2d)

    def bdf1_step(self, t: float, forcing, neu_coeff_func, dir_coeff_func, bc) -> jnp.ndarray:
        return self._step_columns(t, forcing, neu_coeff_func, dir_coeff_func, bc, "bdf1_step")

    def bdf2_step(self, t: float, forcing, neu_coeff_func, dir_coeff_func, bc) -> jnp.ndarray:
        return self._step_columns(t, forcing, neu_coeff_func, dir_coeff_func, bc, "bdf2_step")

    def bdf3_step(self, t: float, forcing, neu_coeff_func, dir_coeff_func, bc) -> jnp.ndarray:
        return self._step_columns(t, forcing, neu_coeff_func, dir_coeff_func, bc, "bdf3_step")

    def returns_distributed_state(self) -> bool:
        return False

    def get_output_range(self) -> tuple[int, int]:
        if not self.solvers:
            return (0, 0)
        return self.solvers[0].get_output_range()

    def get_output_nodes(self) -> jnp.ndarray:
        if not self.solvers:
            return jnp.zeros((0, 0), dtype=float)
        return self.solvers[0].get_output_nodes()

    def _ensure_solvers(self, num_species: int) -> None:
        if num_species <= 0:
            raise ValueError("MultiSpeciesPUDiffusionSolver requires at least one species.")
        if not self.solvers:
            solver = PUDiffusionSolver()
            solver.init(self.domain, self.xi, self.dt, self.nu, self.num_omp_threads)
            self.solvers = [solver]
            self.num_species = num_species
            return
        if self.num_species != num_species:
            raise ValueError("MultiSpeciesPUDiffusionSolver was initialized for a different number of species.")

    def _step_columns(self, t: float, forcing, neu_coeff_func, dir_coeff_func, bc, step_name: str) -> jnp.ndarray:
        if not self.solvers:
            raise ValueError("MultiSpeciesPUDiffusionSolver requires set_initial_state() before stepping.")
        return jnp.asarray(getattr(self.solvers[0], step_name)(t, forcing, neu_coeff_func, dir_coeff_func, bc), dtype=float)
