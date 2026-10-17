from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import Callable

import jax.numpy as jnp

from kernelpack.domain import DomainDescriptor
from .diffusion import DiffusionSolver
from .pu_sl_advection import PUSLAdvectionSolver


@dataclass
class PUSLFDAdvectionDiffusionSolver:
    domain: DomainDescriptor = field(default_factory=DomainDescriptor)
    xi_sl: int = 0
    xi_fd: int = 0
    dt: float = 0.0
    nu: float = 0.0
    direction: str = "backward"
    advection: PUSLAdvectionSolver = field(default_factory=PUSLAdvectionSolver)
    diffusion: DiffusionSolver = field(
        default_factory=lambda: DiffusionSolver(lap_stencil="rbf", bc_stencil="rbf")
    )
    coeff_nm2: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 1), dtype=float))
    coeff_nm1: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 1), dtype=float))
    coeff_n: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 1), dtype=float))
    completed_steps: int = 0
    enforce_mass_constraint: bool = False
    mass_constraint_target: float = 0.0
    has_explicit_mass_constraint_target: bool = False
    constant_mode: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 1), dtype=float))
    constant_mode_mass: float = 0.0

    def spawn_rollout_copy(self) -> PUSLFDAdvectionDiffusionSolver:
        clone = copy(self)
        clone.diffusion = self.diffusion.spawn_rollout_copy()
        clone.coeff_nm2 = jnp.zeros((0, 1), dtype=float)
        clone.coeff_nm1 = jnp.zeros((0, 1), dtype=float)
        clone.coeff_n = jnp.zeros((0, 1), dtype=float)
        clone.completed_steps = 0
        return clone

    def init(
        self,
        domain: DomainDescriptor,
        xi_sl: int,
        xi_fd: int,
        dt: float,
        nu: float,
        direction: str = "backward",
        num_omp_threads: int = 1,
    ) -> None:
        self.domain = domain
        self.xi_sl = xi_sl
        self.xi_fd = xi_fd
        self.dt = dt
        self.nu = nu
        self.direction = str(direction).lower()
        self.advection.init(domain, xi_sl, dt)
        self.diffusion.init(domain, xi_fd, dt, nu, num_omp_threads)
        self.constant_mode = self.advection.project_constant(1.0, 1)
        self.constant_mode_mass = self.advection.total_mass(self.constant_mode, 0)
        self.coeff_nm2 = jnp.zeros((0, 1), dtype=float)
        self.coeff_nm1 = jnp.zeros((0, 1), dtype=float)
        self.coeff_n = jnp.zeros((0, 1), dtype=float)
        self.completed_steps = 0

    def set_step_size(self, dt: float) -> None:
        self.dt = dt
        self.advection.set_step_size(dt)
        self.diffusion.set_step_size(dt)

    def set_transport_direction(self, direction: str) -> None:
        self.direction = str(direction).lower()

    def enable_homogeneous_neumann_mass_conservation(self, enable: bool = True) -> None:
        self.enable_mass_constraint(enable)

    def enable_mass_constraint(self, enable: bool = True) -> None:
        self.enforce_mass_constraint = bool(enable)
        if enable and not self.has_explicit_mass_constraint_target:
            self.mass_constraint_target = self._mass_of_current_coefficients()

    def disable_mass_constraint(self) -> None:
        self.enforce_mass_constraint = False

    def set_mass_constraint_target(self, target_mass: float) -> None:
        self.mass_constraint_target = float(target_mass)
        self.has_explicit_mass_constraint_target = True
        self.enforce_mass_constraint = True

    def set_initial_state(self, state: jnp.ndarray) -> None:
        local_state = jnp.asarray(state, dtype=float).reshape(-1)
        self.coeff_nm2 = self.advection.project_samples(local_state[:, None])
        self.coeff_nm1 = jnp.zeros((0, 1), dtype=float)
        self.coeff_n = jnp.zeros((0, 1), dtype=float)
        self.completed_steps = 0
        self._initialize_mass_constraint_target(self.coeff_nm2)

    def set_state_history(self, *states: jnp.ndarray) -> None:
        coeffs = [self.advection.project_samples(jnp.asarray(state, dtype=float).reshape(-1, 1)) for state in states]
        if len(coeffs) == 1:
            self.set_initial_state(states[0])
        elif len(coeffs) == 2:
            self.coeff_nm2, self.coeff_nm1 = coeffs
            self.coeff_n = jnp.zeros((0, 1), dtype=float)
            self.completed_steps = 1
            self._initialize_mass_constraint_target(self.coeff_nm1)
        elif len(coeffs) == 3:
            self.coeff_nm2, self.coeff_nm1, self.coeff_n = coeffs
            self.completed_steps = 2
            self._initialize_mass_constraint_target(self.coeff_n)
        else:
            raise ValueError("Expected one, two, or three physical states.")

    def get_output_nodes(self) -> jnp.ndarray:
        return self.advection.get_output_nodes()

    def get_output_range(self) -> tuple[int, int]:
        return self.advection.get_output_range()

    def returns_distributed_state(self) -> bool:
        return self.advection.returns_distributed_state()

    def advection_solver(self) -> PUSLAdvectionSolver:
        return self.advection

    def diffusion_solver(self) -> DiffusionSolver:
        return self.diffusion

    def bdf1_step(
        self,
        t_next: float,
        velocity: Callable[..., jnp.ndarray],
        rk,
        forcing: Callable[..., jnp.ndarray] | jnp.ndarray,
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        bc,
    ) -> jnp.ndarray:
        if self.coeff_nm2.size == 0:
            raise ValueError("bdf1_step requires set_initial_state first")
        transported_coeff_nm2 = self._transport_coefficients(t_next - self.dt, self.coeff_nm2, 1, velocity, rk)
        transported_nm2 = self.advection.evaluate_at_nodes(transported_coeff_nm2).reshape(-1)
        self.diffusion.set_state_history(transported_nm2)
        next_state = self.diffusion.bdf1_step(t_next, forcing, neu_coeff_func, dir_coeff_func, bc)
        coeff_next = self.advection.project_samples(jnp.asarray(next_state, dtype=float).reshape(-1, 1))
        self._enforce_coefficient_mass_target_if_needed(coeff_next)
        self.coeff_nm1 = coeff_next
        self.completed_steps = 1
        return self.current_state()

    def bdf2_step(
        self,
        t_next: float,
        velocity: Callable[..., jnp.ndarray],
        rk,
        forcing: Callable[..., jnp.ndarray] | jnp.ndarray,
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        bc,
    ) -> jnp.ndarray:
        if self.completed_steps < 1 or self.coeff_nm1.size == 0:
            raise ValueError("bdf2_step requires one prior step")
        transported_coeff_nm2 = self._transport_coefficients(t_next - 2.0 * self.dt, self.coeff_nm2, 2, velocity, rk)
        transported_coeff_nm1 = self._transport_coefficients(t_next - self.dt, self.coeff_nm1, 1, velocity, rk)
        transported_nm2 = self.advection.evaluate_at_nodes(transported_coeff_nm2).reshape(-1)
        transported_nm1 = self.advection.evaluate_at_nodes(transported_coeff_nm1).reshape(-1)
        self.diffusion.set_state_history(transported_nm2, transported_nm1)
        next_state = self.diffusion.bdf2_step(t_next, forcing, neu_coeff_func, dir_coeff_func, bc)
        coeff_next = self.advection.project_samples(jnp.asarray(next_state, dtype=float).reshape(-1, 1))
        self._enforce_coefficient_mass_target_if_needed(coeff_next)
        self.coeff_n = coeff_next
        self.completed_steps = 2
        return self.current_state()

    def bdf3_step(
        self,
        t_next: float,
        velocity: Callable[..., jnp.ndarray],
        rk,
        forcing: Callable[..., jnp.ndarray] | jnp.ndarray,
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        bc,
    ) -> jnp.ndarray:
        if self.completed_steps < 2 or self.coeff_n.size == 0:
            raise ValueError("bdf3_step requires two prior steps")
        transported_coeff_nm2 = self._transport_coefficients(t_next - 3.0 * self.dt, self.coeff_nm2, 3, velocity, rk)
        transported_coeff_nm1 = self._transport_coefficients(t_next - 2.0 * self.dt, self.coeff_nm1, 2, velocity, rk)
        transported_coeff_n = self._transport_coefficients(t_next - self.dt, self.coeff_n, 1, velocity, rk)
        transported_nm2 = self.advection.evaluate_at_nodes(transported_coeff_nm2).reshape(-1)
        transported_nm1 = self.advection.evaluate_at_nodes(transported_coeff_nm1).reshape(-1)
        transported_n = self.advection.evaluate_at_nodes(transported_coeff_n).reshape(-1)
        self.diffusion.set_state_history(transported_nm2, transported_nm1, transported_n)
        next_state = self.diffusion.bdf3_step(t_next, forcing, neu_coeff_func, dir_coeff_func, bc)
        coeff_next = self.advection.project_samples(jnp.asarray(next_state, dtype=float).reshape(-1, 1))
        self._enforce_coefficient_mass_target_if_needed(coeff_next)
        self.coeff_nm2 = self.coeff_nm1
        self.coeff_nm1 = self.coeff_n
        self.coeff_n = coeff_next
        self.completed_steps = max(self.completed_steps, 3)
        return self.current_state()

    def current_state(self) -> jnp.ndarray:
        coeffs = self._current_coefficients()
        return jnp.asarray(self.advection.evaluate_at_nodes(coeffs), dtype=float).reshape(-1)

    def total_mass(self, local_state: jnp.ndarray) -> float:
        coeffs = self.advection.project_samples(jnp.asarray(local_state, dtype=float).reshape(-1, 1))
        return self.advection.total_mass(coeffs, 0)

    def _initialize_mass_constraint_target(self, reference_coeffs: jnp.ndarray) -> None:
        if not self.enforce_mass_constraint or self.has_explicit_mass_constraint_target:
            return
        self.mass_constraint_target = self.advection.total_mass(reference_coeffs, 0)

    def _transport_coefficients(
        self,
        t_start: float,
        coeffs: jnp.ndarray,
        num_steps: int,
        velocity: Callable[..., jnp.ndarray],
        rk,
    ) -> jnp.ndarray:
        if num_steps <= 0:
            raise ValueError("transport requires a positive step count")
        original_dt = self.dt
        horizon = num_steps * self.dt
        self.advection.set_step_size(horizon)
        if self.direction == "forward":
            coeffs_next = self.advection.forward_sl_step(t_start, coeffs, velocity, rk)
        else:
            coeffs_next = self.advection.backward_sl_step(t_start, coeffs, velocity, rk)
        self.advection.set_step_size(original_dt)
        coeffs_next = coeffs_next if coeffs_next.ndim == 2 else coeffs_next[:, None]
        self._enforce_coefficient_mass_target_if_needed(coeffs_next)
        return coeffs_next

    def _enforce_coefficient_mass_target_if_needed(self, coeffs: jnp.ndarray) -> None:
        if not self.enforce_mass_constraint:
            return
        current_mass = self.advection.total_mass(coeffs, 0)
        alpha = (self.mass_constraint_target - current_mass) / max(abs(self.constant_mode_mass), 1.0e-14)
        coeffs += alpha * self.constant_mode

    def _current_coefficients(self) -> jnp.ndarray:
        if self.completed_steps <= 0:
            return self.coeff_nm2
        if self.completed_steps == 1:
            return self.coeff_nm1
        return self.coeff_n

    def _mass_of_current_coefficients(self) -> float:
        coeffs = self._current_coefficients()
        if coeffs.size == 0:
            return 0.0
        return self.advection.total_mass(coeffs, 0)
