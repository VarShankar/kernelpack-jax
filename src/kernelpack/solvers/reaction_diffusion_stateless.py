from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from jax import lax
import jax.numpy as jnp

from ._common import (
    build_implicit_rhs,
    build_implicit_system,
    build_implicit_system_sparse,
    evaluate_forcing_callback,
    evaluate_transient_boundary_values,
    solve_dense_system,
    solve_sparse_system_gmres,
    sparse_matrix_to_dense,
    validate_physical_state,
)
from .reaction_diffusion import ReactionDiffusionSolver, _evaluate_reaction_callback, _solve_lineax_system
from copy import copy
from ._cudss import solve_transient_cudss, get_transient_cudss_factor


@dataclass(frozen=True)
class StatelessReactionDiffusionSolver:
    """Functional reaction-diffusion time marcher over a prepared solver.

    The wrapped ``ReactionDiffusionSolver`` owns the assembled KernelPack
    operators. This class only carries BDF state through function arguments, so
    it can use JAX loop primitives without storing intermediate timesteps.
    """

    base: ReactionDiffusionSolver

    @classmethod
    def from_solver(cls, solver: ReactionDiffusionSolver) -> "StatelessReactionDiffusionSolver":
        if solver.n <= 0:
            raise ValueError("StatelessReactionDiffusionSolver requires an initialized ReactionDiffusionSolver")
        if solver.bc_sparse is None:
            raise ValueError("Call prepare_boundary_operator(...) on the base solver before stateless marching")
        if solver.linear_solver == "gmres_ilu" and solver.fixed_preconditioner is None:
            raise ValueError("Call prepare_preconditioner(...) on the base solver before stateless gmres_ilu marching")
        if solver.linear_solver == "cudss":
            solver.prepare_cudss()
            return cls(copy(solver))
        return cls(solver)

    def march_to_final_state(
        self,
        initial_state: jnp.ndarray,
        *,
        steps: int,
        reaction: Callable[..., jnp.ndarray] | jnp.ndarray,
        bc: Callable[..., jnp.ndarray] | jnp.ndarray,
        forcing: Callable[..., jnp.ndarray] | jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Advance ``steps`` times and return only the final physical state."""

        if steps < 0:
            raise ValueError("steps must be nonnegative")
        u0 = validate_physical_state(initial_state, self.base.n)
        if steps == 0:
            return u0
        u1 = self.bdf1_state(u0, self.base.dt, reaction=reaction, bc=bc, forcing=forcing)
        if steps == 1:
            return u1
        u2 = self.bdf2_state(u0, u1, 2.0 * self.base.dt, reaction=reaction, bc=bc, forcing=forcing)
        if steps == 2:
            return u2

        def body(k: int, carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            u_nm2, u_nm1, u_n = carry
            t_next = (k + 1) * self.base.dt
            u_np1 = self.bdf3_state(u_nm2, u_nm1, u_n, t_next, reaction=reaction, bc=bc, forcing=forcing)
            return u_nm1, u_n, u_np1

        _, _, final_state = lax.fori_loop(2, steps, body, (u0, u1, u2))
        return final_state

    def bdf1_state(
        self,
        previous: jnp.ndarray,
        t: float,
        *,
        reaction: Callable[..., jnp.ndarray] | jnp.ndarray,
        bc: Callable[..., jnp.ndarray] | jnp.ndarray,
        forcing: Callable[..., jnp.ndarray] | jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        forcing_term = self._evaluate_forcing_if_present(forcing, t)
        reaction_term = _evaluate_reaction_callback(reaction, t, previous, self.base.x)
        rhs_physical = previous + self.base.dt * (forcing_term + reaction_term)
        return self._solve_state(rhs_physical, t, bc, -self.base.nu * self.base.dt)

    def bdf2_state(
        self,
        u_nm2: jnp.ndarray,
        u_nm1: jnp.ndarray,
        t: float,
        *,
        reaction: Callable[..., jnp.ndarray] | jnp.ndarray,
        bc: Callable[..., jnp.ndarray] | jnp.ndarray,
        forcing: Callable[..., jnp.ndarray] | jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        reaction_nm2 = _evaluate_reaction_callback(reaction, t, u_nm2, self.base.x)
        reaction_nm1 = _evaluate_reaction_callback(reaction, t, u_nm1, self.base.x)
        reaction_extrapolated = 2.0 * reaction_nm1 - reaction_nm2
        rhs_physical = (4.0 / 3.0) * u_nm1 - (1.0 / 3.0) * u_nm2 + (2.0 / 3.0) * self.base.dt * (
            self._evaluate_forcing_if_present(forcing, t) + reaction_extrapolated
        )
        return self._solve_state(rhs_physical, t, bc, -(2.0 / 3.0) * self.base.nu * self.base.dt, bdf_order=2)

    def bdf3_state(
        self,
        u_nm2: jnp.ndarray,
        u_nm1: jnp.ndarray,
        u_n: jnp.ndarray,
        t: float,
        *,
        reaction: Callable[..., jnp.ndarray] | jnp.ndarray,
        bc: Callable[..., jnp.ndarray] | jnp.ndarray,
        forcing: Callable[..., jnp.ndarray] | jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        reaction_nm2 = _evaluate_reaction_callback(reaction, t, u_nm2, self.base.x)
        reaction_nm1 = _evaluate_reaction_callback(reaction, t, u_nm1, self.base.x)
        reaction_n = _evaluate_reaction_callback(reaction, t, u_n, self.base.x)
        reaction_extrapolated = 3.0 * reaction_n - 3.0 * reaction_nm1 + reaction_nm2
        rhs_physical = (18.0 / 11.0) * u_n - (9.0 / 11.0) * u_nm1 + (2.0 / 11.0) * u_nm2 + (6.0 / 11.0) * self.base.dt * (
            self._evaluate_forcing_if_present(forcing, t) + reaction_extrapolated
        )
        return self._solve_state(rhs_physical, t, bc, -(6.0 / 11.0) * self.base.nu * self.base.dt, bdf_order=3)

    def _evaluate_forcing_if_present(
        self,
        forcing: Callable[..., jnp.ndarray] | jnp.ndarray | None,
        t: float,
    ) -> jnp.ndarray:
        if forcing is None:
            return jnp.zeros((self.base.n,), dtype=float)
        return evaluate_forcing_callback(forcing, self.base.nu, t, self.base.x)

    def _solve_state(
        self,
        rhs_physical: jnp.ndarray,
        t: float,
        bc: Callable[..., jnp.ndarray] | jnp.ndarray,
        lap_scale: float,
        *,
        bdf_order: int = 1,
    ) -> jnp.ndarray:
        rhs_boundary = evaluate_transient_boundary_values(
            bc,
            self.base.cached_neu_coeff_,
            self.base.cached_dir_coeff_,
            self.base.nr,
            t,
            self.base.xb,
        )
        rhs = build_implicit_rhs(rhs_physical, rhs_boundary)
        if self.base.linear_solver == "cudss":
            factor = get_transient_cudss_factor(self.base, bdf_order)
            if factor is not None:
                return factor.solve(rhs)[:self.base.n]
        sparse_system = build_implicit_system_sparse(self.base.lap_sparse, self.base.bc_sparse, self.base.n, lap_scale)
        if self.base.linear_solver == "cudss":
            sol = solve_transient_cudss(self.base, sparse_system, rhs, bdf_order)
        elif self.base.linear_solver == "dense":
            system = build_implicit_system(self.base.lap, self.base.bc, self.base.n, lap_scale)
            sol = solve_dense_system(system, rhs)
        elif self.base.linear_solver == "lineax":
            system = build_implicit_system(self.base.lap, self.base.bc, self.base.n, lap_scale) if self.base.lap.size and self.base.bc.size else jnp.zeros((0, 0), dtype=float)
            dense_system = system if system.size else sparse_matrix_to_dense(sparse_system)
            sol = _solve_lineax_system(dense_system, rhs)
        elif self.base.linear_solver in ("gmres", "gmres_ilu"):
            preconditioner = self.base.fixed_preconditioner if self.base.linear_solver == "gmres_ilu" else None
            sol = solve_sparse_system_gmres(sparse_system, rhs, preconditioner=preconditioner)
        else:
            raise ValueError(f"unknown linear solver {self.base.linear_solver}")
        return jnp.asarray(sol[: self.base.n], dtype=float)


ReactionDiffusionSolver_stateless = StatelessReactionDiffusionSolver
