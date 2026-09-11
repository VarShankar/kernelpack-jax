from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from math import comb
import time
from typing import Callable

import jax
from jax import jit
from jax import scipy as jsp
import jax.numpy as jnp
import numpy as np

from kernelpack.domain import DomainDescriptor, MovingDomainDescriptor
from kernelpack.geometry import MovingBoundarySBFModel
from kernelpack.rbffd import (
    MovingDifferentiationUpdater,
    MovingDomainLocalInterpolator,
    StencilProperties,
)

from ._common import SparseCOOMatrix, sparse_matrix_to_dense, sparse_matvec


VelocityCallback = Callable[[float, jnp.ndarray], jnp.ndarray]


@partial(jit, static_argnames=("velocity",))
def rk3_step(t: float, points: jnp.ndarray, dt: float, velocity: VelocityCallback) -> jnp.ndarray:
    k1 = velocity(t, points)
    k2 = velocity(t + dt / 2.0, points + dt * k1 / 2.0)
    k3 = velocity(t + 3.0 * dt / 4.0, points + 3.0 * dt * k2 / 4.0)
    return points + dt * (2.0 * k1 + 3.0 * k2 + 4.0 * k3) / 9.0


def trace_backward(points: jnp.ndarray, arrival_time: float, dt: float, velocity: VelocityCallback) -> jnp.ndarray:
    return rk3_step(arrival_time, points, -dt, velocity)


@partial(jit, static_argnames=("n_physical",))
def _build_sparse_system_arrays(
    lap_indices: jnp.ndarray,
    lap_values: jnp.ndarray,
    boundary_indices: jnp.ndarray,
    boundary_values: jnp.ndarray,
    reaction: jnp.ndarray,
    boundary_active: jnp.ndarray,
    implicit_dt: float,
    diffusivity: float,
    n_physical: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    diagonal = jnp.arange(n_physical, dtype=int)
    identity_indices = jnp.column_stack([diagonal, diagonal])
    boundary_rows = boundary_indices[:, 0] + n_physical
    shifted_boundary_indices = jnp.column_stack([boundary_rows, boundary_indices[:, 1]])
    boundary_diagonal = jnp.arange(boundary_active.size, dtype=int)
    inactive_boundary_indices = jnp.column_stack(
        [boundary_diagonal + n_physical, boundary_diagonal + n_physical]
    )
    indices = jnp.concatenate(
        [lap_indices, identity_indices, shifted_boundary_indices, inactive_boundary_indices], axis=0
    )
    values = jnp.concatenate(
        [
            -implicit_dt * diffusivity * lap_values,
            1.0 - implicit_dt * reaction,
            boundary_values,
            (~boundary_active).astype(boundary_values.dtype),
        ]
    )
    return indices, values


def _build_sparse_system(
    laplacian: SparseCOOMatrix,
    boundary: SparseCOOMatrix,
    reaction: jnp.ndarray,
    boundary_active: jnp.ndarray,
    implicit_dt: float,
    diffusivity: float,
    n_physical: int,
) -> SparseCOOMatrix:
    indices, values = _build_sparse_system_arrays(
        laplacian.indices,
        laplacian.values,
        boundary.indices,
        boundary.values,
        reaction,
        boundary_active,
        implicit_dt,
        diffusivity,
        n_physical,
    )
    return SparseCOOMatrix(
        indices=indices,
        values=values,
        shape=(n_physical + boundary.shape[0], laplacian.shape[1]),
    )


@partial(jit, static_argnames=("shape", "restart", "maxiter"))
def _solve_sparse_gmres(
    indices: jnp.ndarray,
    values: jnp.ndarray,
    rhs: jnp.ndarray,
    guess: jnp.ndarray,
    preconditioner_diagonal: jnp.ndarray,
    tolerance: float,
    *,
    shape: tuple[int, int],
    restart: int,
    maxiter: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    def matvec(vector: jnp.ndarray) -> jnp.ndarray:
        return sparse_matvec(indices, values, shape[0], vector)

    inverse_diagonal = 1.0 / preconditioner_diagonal

    def precondition(vector: jnp.ndarray) -> jnp.ndarray:
        return inverse_diagonal * vector

    return jsp.sparse.linalg.gmres(
        matvec,
        rhs,
        x0=guess,
        tol=tolerance,
        atol=0.0,
        restart=restart,
        maxiter=maxiter,
        M=precondition,
        solve_method="batched",
    )


@partial(jit, static_argnames=("n_physical", "boundary_count"))
def _paper_preconditioner_diagonal_arrays(
    indices: jnp.ndarray,
    values: jnp.ndarray,
    *,
    n_physical: int,
    boundary_count: int,
) -> jnp.ndarray:
    """Build the physical/Schur diagonal directly from device COO arrays."""

    rows = indices[:, 0]
    cols = indices[:, 1]
    physical_mask = (rows < n_physical) & (rows == cols)
    physical_rows = jnp.minimum(rows, n_physical - 1)
    physical_diagonal = jnp.zeros((n_physical,), dtype=values.dtype).at[physical_rows].add(
        jnp.where(physical_mask, values, 0.0)
    )
    epsilon = jnp.finfo(values.dtype).eps
    physical_diagonal = jnp.where(jnp.abs(physical_diagonal) < epsilon, 1.0, physical_diagonal)
    if boundary_count == 0:
        return physical_diagonal

    boundary_rows = rows - n_physical
    boundary_cols = cols - n_physical
    a22_mask = (boundary_rows >= 0) & (boundary_cols >= 0) & (boundary_rows == boundary_cols)
    a22_rows = jnp.clip(boundary_rows, 0, boundary_count - 1)
    a22_diagonal = jnp.zeros((boundary_count,), dtype=values.dtype).at[a22_rows].add(
        jnp.where(a22_mask, values, 0.0)
    )

    # Match A21[b,j] with A12[j,b] by a flattened (b,j) key. Invalid
    # entries sort to a sentinel and contribute zero, keeping all shapes static.
    sentinel = boundary_count * n_physical
    a12_mask = (rows < n_physical) & (cols >= n_physical)
    a12_keys = jnp.where(a12_mask, (cols - n_physical) * n_physical + rows, sentinel)
    permutation = jnp.argsort(a12_keys)
    sorted_keys = a12_keys[permutation]
    sorted_values = values[permutation]

    a21_mask = (rows >= n_physical) & (cols < n_physical)
    a21_keys = jnp.where(a21_mask, (rows - n_physical) * n_physical + cols, sentinel)
    matches = jnp.searchsorted(sorted_keys, a21_keys, side="left")
    matches = jnp.minimum(matches, sorted_keys.size - 1)
    matched = a21_mask & (a21_keys < sentinel) & (sorted_keys[matches] == a21_keys)
    products = jnp.where(
        matched,
        values * sorted_values[matches] / physical_diagonal[jnp.minimum(cols, n_physical - 1)],
        0.0,
    )
    schur_correction = jnp.zeros((boundary_count,), dtype=values.dtype).at[
        jnp.clip(boundary_rows, 0, boundary_count - 1)
    ].add(products)
    schur_diagonal = a22_diagonal - schur_correction
    schur_diagonal = jnp.where(jnp.abs(schur_diagonal) < epsilon, 1.0, schur_diagonal)
    return jnp.concatenate([physical_diagonal, schur_diagonal])


def _paper_preconditioner_diagonal(system: SparseCOOMatrix, n_physical: int) -> jnp.ndarray:
    return _paper_preconditioner_diagonal_arrays(
        system.indices,
        system.values,
        n_physical=n_physical,
        boundary_count=system.shape[0] - n_physical,
    )


@partial(jit, static_argnames=("row_count",))
def _relative_residual(
    indices: jnp.ndarray,
    values: jnp.ndarray,
    row_count: int,
    solution: jnp.ndarray,
    rhs: jnp.ndarray,
) -> jnp.ndarray:
    residual = sparse_matvec(indices, values, row_count, solution) - rhs
    return jnp.linalg.norm(residual) / jnp.maximum(jnp.linalg.norm(rhs), jnp.finfo(float).eps)


def _surface_seed_sites(surface: object) -> jnp.ndarray:
    sites = jnp.asarray(surface.data_sites, dtype=float)
    if sites.size == 0:
        sites = jnp.asarray(surface.get_uniform_sample_sites(), dtype=float)
    return sites


def _paper_stencil_properties(dim: int, ell: int, point_set: str, tree_mode: str) -> StencilProperties:
    npoly = int(comb(dim + ell, dim))
    spline_degree = ell if ell % 2 == 1 else ell - 1
    spline_degree = min(max(spline_degree, 5), 11)
    return StencilProperties(
        n=2 * npoly + 1,
        dim=dim,
        ell=ell,
        spline_degree=spline_degree,
        npoly=npoly,
        point_set=point_set,
        tree_mode=tree_mode,
    )


def _bdf_coefficients(order: int) -> tuple[jnp.ndarray, float]:
    if order == 1:
        return jnp.asarray([1.0]), 1.0
    if order == 2:
        return jnp.asarray([4.0 / 3.0, -1.0 / 3.0]), 2.0 / 3.0
    if order == 3:
        return jnp.asarray([18.0 / 11.0, -9.0 / 11.0, 2.0 / 11.0]), 6.0 / 11.0
    raise ValueError("BDF order must be 1, 2, or 3")


def _normalize_values(values: object, count: int, label: str) -> jnp.ndarray:
    array = jnp.asarray(values, dtype=float)
    if array.ndim == 0 or array.size == 1:
        return jnp.full((count,), array.reshape(-1)[0])
    array = array.reshape(-1)
    if array.size != count:
        raise ValueError(f"{label} callback must return one value per node")
    return array


def _compact_on_host(values: jnp.ndarray, active: jnp.ndarray) -> jnp.ndarray:
    """Extract dynamic public output without compiling one gather per active count."""

    return jnp.asarray(np.asarray(values)[np.asarray(active, dtype=bool)])


def _evaluate_initial(specification: object, time: float, points: jnp.ndarray) -> jnp.ndarray:
    if callable(specification):
        try:
            values = specification(time, points)
        except TypeError:
            values = specification(points)
    else:
        values = specification
    return _normalize_values(values, points.shape[0], "initial-state")


def _evaluate_field(
    specification: object,
    time: float,
    points: jnp.ndarray,
    auxiliary: float | None = None,
) -> jnp.ndarray:
    if callable(specification):
        if auxiliary is not None:
            try:
                values = specification(auxiliary, time, points)
            except TypeError:
                values = specification(time, points)
        else:
            try:
                values = specification(time, points)
            except TypeError:
                values = specification(points)
    else:
        values = specification
    return _normalize_values(values, points.shape[0], "field")


def _evaluate_boundary_value(
    specification: object,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    normals: jnp.ndarray,
    time: float,
    points: jnp.ndarray,
) -> jnp.ndarray:
    if callable(specification):
        try:
            values = specification(alpha, beta, normals, time, points)
        except TypeError:
            try:
                values = specification(time, points)
            except TypeError:
                values = specification(points)
    else:
        values = specification
    return _normalize_values(values, points.shape[0], "boundary-value")


@dataclass
class MovingDomainADRSolver:
    """Semi-Lagrangian RBF-FD ADR solver for domains with moving holes.

    The method follows Shankar, Wright, and Fogelson (JCP 2021): RK3 marker
    motion, cached parametric geometry, carve/refill, local physical-node RBF
    interpolation, selective differentiation updates, and BDF1/2/3 stepping.
    """

    gmres_tolerance: float | None = None
    gmres_restart: int = 10
    gmres_max_iterations: int = 400
    gmres_internal_tolerance_factor: float = 0.1
    linear_solver: str = "gmres"
    dense_fallback_on_failure: bool = False
    fixed_capacity: bool = True
    moving_boundary_capacity: int | None = None
    profile_phases: bool = False
    surface_builder: Callable[..., object] | None = None
    moving_domain: MovingDomainDescriptor | None = field(default=None, init=False)
    surface_models: list[object] = field(default_factory=list, init=False)
    surface_seeds: list[jnp.ndarray] = field(default_factory=list, init=False)
    boundary_models: list[MovingBoundarySBFModel | None] = field(default_factory=list, init=False)
    xi: int = field(default=0, init=False)
    dt: float = field(default=jnp.nan, init=False)
    nu: float = field(default=jnp.nan, init=False)
    current_time: float = field(default=jnp.nan, init=False)
    laplacian_updater: MovingDifferentiationUpdater | None = field(default=None, init=False)
    boundary_updater: MovingDifferentiationUpdater | None = field(default=None, init=False)
    interpolation_history: list[MovingDomainLocalInterpolator] = field(default_factory=list, init=False)
    state_history: list[jnp.ndarray] = field(default_factory=list, init=False)
    time_history: list[float] = field(default_factory=list, init=False)
    completed_steps: int = field(default=0, init=False)
    last_linear_diagnostics: dict[str, object] = field(default_factory=dict, init=False)
    last_update_diagnostics: dict[str, int] = field(default_factory=dict, init=False)
    last_phase_timings: dict[str, float] = field(default_factory=dict, init=False)
    _lap_stencil_properties: StencilProperties | None = field(default=None, init=False, repr=False)
    _boundary_stencil_properties: StencilProperties | None = field(default=None, init=False, repr=False)
    _interp_stencil_properties: StencilProperties | None = field(default=None, init=False, repr=False)

    def init(
        self,
        background_domain: DomainDescriptor,
        embedded_surfaces: object | list[object] | tuple[object, ...],
        xi: int,
        dt: float,
        diffusivity: float,
        *,
        initial_time: float = 0.0,
    ) -> None:
        if xi <= 0 or int(xi) != xi:
            raise ValueError("xi must be a positive integer")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        if not np.isfinite(diffusivity) or diffusivity < 0.0:
            raise ValueError("diffusivity must be nonnegative and finite")
        self.xi = int(xi)
        self.dt = float(dt)
        self.nu = float(diffusivity)
        self.current_time = float(initial_time)
        self.surface_models = list(embedded_surfaces) if isinstance(embedded_surfaces, (list, tuple)) else [embedded_surfaces]
        self.surface_seeds = [_surface_seed_sites(surface) for surface in self.surface_models]
        self.moving_domain = MovingDomainDescriptor(background_domain)
        h = self.moving_domain.get_background_domain().get_sep_rad()
        if self.surface_builder is None:
            self.boundary_models = [MovingBoundarySBFModel(sites) for sites in self.surface_seeds]
            self.surface_models = [model.evaluate(sites, h) for model, sites in zip(self.boundary_models, self.surface_seeds)]
        else:
            self.boundary_models = [None for _ in self.surface_models]
        self.moving_domain.update_from_embedded_surfaces(self.surface_models)
        if self.fixed_capacity:
            if self.moving_boundary_capacity is not None:
                moving_capacity = int(self.moving_boundary_capacity)
            else:
                current_count = int(self.moving_domain.moving_boundary_nodes.shape[0])
                moving_capacity = current_count + max(8, (current_count + 1) // 2)
            self.moving_domain.configure_fixed_capacity(moving_capacity)

        discretization_domain = self.moving_domain.get_discretization_domain()
        dim = discretization_domain.get_dim()
        self._lap_stencil_properties = _paper_stencil_properties(dim, self.xi + 1, "interior_boundary", "all")
        self._boundary_stencil_properties = _paper_stencil_properties(dim, self.xi, "boundary", "all")
        self._interp_stencil_properties = _paper_stencil_properties(
            dim,
            self.xi + 1,
            "interior_boundary",
            "interior_boundary",
        )
        self.laplacian_updater = MovingDifferentiationUpdater(
            self._lap_stencil_properties,
            operator="lap",
            point_set="interior_boundary",
            tree_mode="all",
            assembly_mode="fdo",
            assembly_batch_size=512,
        )
        self.laplacian_updater.initialize(self.moving_domain)
        self.boundary_updater = None
        self.interpolation_history = [
            MovingDomainLocalInterpolator(
                discretization_domain,
                self._interp_stencil_properties,
                source_ids=self.moving_domain.get_discretization_physical_ids(),
            )
        ]
        self.state_history = []
        self.time_history = []
        self.completed_steps = 0
        self.last_linear_diagnostics = {}
        self.last_update_diagnostics = {}

    def set_initial_state(self, initial_state: object) -> None:
        self._require_initialized()
        domain = self.moving_domain.get_discretization_domain()
        points = domain.get_int_bdry_nodes()
        state = _evaluate_initial(initial_state, self.current_time, points)
        self.state_history = [jnp.where(domain.get_physical_active_mask(), state, 0.0)]
        self.time_history = [self.current_time]
        self.interpolation_history = self.interpolation_history[:1]
        self.completed_steps = 0

    def step(
        self,
        next_time: float,
        velocity: VelocityCallback,
        forcing: object = 0.0,
        neu_coeff: object = 0.0,
        dir_coeff: object = 1.0,
        boundary_value: object = 0.0,
        reaction_coeff: object = 0.0,
    ) -> jnp.ndarray:
        self._require_ready()
        if abs(next_time - (self.current_time + self.dt)) > 256.0 * np.finfo(float).eps * max(1.0, abs(next_time)):
            raise ValueError("MovingDomainADRSolver requires its initialized fixed step size")
        phase_start = time.perf_counter()
        phase_timings: dict[str, float] = {}

        def finish_phase(name: str, synchronization_value: object) -> None:
            nonlocal phase_start
            if not self.profile_phases:
                return
            jax.block_until_ready(synchronization_value)
            now = time.perf_counter()
            phase_timings[name] = now - phase_start
            phase_start = now

        self._advance_embedded_surfaces(velocity, next_time)
        finish_phase("surface_motion", self.surface_seeds[-1])
        self.moving_domain.update_from_embedded_surfaces(self.surface_models)
        domain = self.moving_domain.get_discretization_domain()
        finish_phase("domain_update", domain.get_all_nodes())
        self.laplacian_updater.update(self.moving_domain)
        finish_phase("laplacian_update", self.laplacian_updater.weights)

        boundary_points = domain.get_bdry_nodes()
        normals = domain.get_nrmls()
        alpha = _evaluate_field(neu_coeff, next_time, boundary_points)
        beta = _evaluate_field(dir_coeff, next_time, boundary_points)
        if self.boundary_updater is None:
            self.boundary_updater = MovingDifferentiationUpdater(
                self._boundary_stencil_properties,
                operator="bc",
                point_set="boundary",
                tree_mode="all",
                assembly_mode="fd",
            )
            self.boundary_updater.initialize(self.moving_domain, neu_coeff=alpha, dir_coeff=beta)
        else:
            self.boundary_updater.update(self.moving_domain, neu_coeff=alpha, dir_coeff=beta)
        finish_phase("boundary_update", self.boundary_updater.weights)

        order = min(3, len(self.state_history))
        history_coefficients, implicit_scale = _bdf_coefficients(order)
        arrivals = domain.get_int_bdry_nodes()
        all_departure = trace_backward(domain.get_all_nodes(), next_time, self.dt, velocity)
        newest_departure = self.interpolation_history[0].evaluate(self.state_history[0], all_departure)
        departure_values = [newest_departure[: arrivals.shape[0]]]
        for history_index in range(1, order):
            departure = trace_backward(arrivals, next_time, (history_index + 1) * self.dt, velocity)
            departure_values.append(
                self.interpolation_history[history_index].evaluate(self.state_history[history_index], departure)
            )
        rhs_physical = sum(
            history_coefficients[index] * departure_values[index]
            for index in range(order)
        )
        rhs_physical = rhs_physical + implicit_scale * self.dt * _evaluate_field(
            forcing,
            next_time,
            arrivals,
            self.nu,
        )
        physical_active = domain.get_physical_active_mask()
        rhs_physical = jnp.where(physical_active, rhs_physical, 0.0)
        finish_phase("semi_lagrangian_rhs", rhs_physical)

        reaction = _evaluate_field(reaction_coeff, next_time, arrivals)
        reaction = jnp.where(physical_active, reaction, 0.0)
        boundary_active = domain.get_boundary_active_mask()
        laplacian = self.laplacian_updater.get_sparse_op()
        boundary = self.boundary_updater.get_sparse_op()
        system = _build_sparse_system(
            laplacian,
            boundary,
            reaction,
            boundary_active,
            implicit_scale * self.dt,
            self.nu,
            domain.get_num_int_bdry_nodes(),
        )
        rhs_boundary = _evaluate_boundary_value(
            boundary_value,
            alpha,
            beta,
            normals,
            next_time,
            boundary_points,
        )
        rhs_boundary = jnp.where(boundary_active, rhs_boundary, 0.0)
        rhs = jnp.concatenate([rhs_physical, rhs_boundary])
        finish_phase("system_assembly", (system.values, rhs))
        tolerance = self.gmres_tolerance
        if tolerance is None:
            tolerance = min(0.1 * domain.get_sep_rad() ** self.xi, 1e-7)
        guess = jnp.where(domain.get_all_active_mask(), newest_departure, 0.0)
        solution, info, used_fallback = self._solve_system(system, rhs, guess, float(tolerance))
        state = jnp.where(physical_active, solution[: domain.get_num_int_bdry_nodes()], 0.0)
        finish_phase("linear_solve", state)

        current_interpolator = MovingDomainLocalInterpolator(
            domain,
            self._interp_stencil_properties,
            previous=self.interpolation_history[0],
            source_ids=self.moving_domain.get_discretization_physical_ids(),
        )
        finish_phase("interpolator_update", current_interpolator.lu)
        self.interpolation_history = [current_interpolator, *self.interpolation_history[:2]]
        self.state_history = [state, *self.state_history[:2]]
        self.time_history = [float(next_time), *self.time_history[:2]]
        self.current_time = float(next_time)
        self.completed_steps += 1
        self.last_update_diagnostics = {
            "laplacian_rows_reused": self.laplacian_updater.last_reused_row_count,
            "laplacian_rows_recomputed": self.laplacian_updater.last_recomputed_row_count,
            "boundary_rows_reused": self.boundary_updater.last_reused_row_count,
            "boundary_rows_recomputed": self.boundary_updater.last_recomputed_row_count,
            "interpolation_rows_reused": current_interpolator.last_reused_row_count,
            "interpolation_rows_recomputed": current_interpolator.last_recomputed_row_count,
        }
        self.last_linear_diagnostics["info"] = info
        self.last_linear_diagnostics["used_dense_fallback"] = used_fallback
        self.last_phase_timings = phase_timings
        return _compact_on_host(state, physical_active)

    def run(
        self,
        final_time: float,
        velocity: VelocityCallback,
        forcing: object = 0.0,
        neu_coeff: object = 0.0,
        dir_coeff: object = 1.0,
        boundary_value: object = 0.0,
        reaction_coeff: object = 0.0,
    ) -> tuple[jnp.ndarray, list[jnp.ndarray]]:
        step_count = round((final_time - self.current_time) / self.dt)
        if abs(self.current_time + step_count * self.dt - final_time) > 256.0 * np.finfo(float).eps * max(1.0, abs(final_time)):
            raise ValueError("final_time must be an integer number of initialized steps away")
        times = self.current_time + jnp.arange(step_count + 1) * self.dt
        states = [self.current_state()]
        for index in range(step_count):
            states.append(
                self.step(
                    float(times[index + 1]),
                    velocity,
                    forcing,
                    neu_coeff,
                    dir_coeff,
                    boundary_value,
                    reaction_coeff,
                )
            )
        return times, states

    def current_state(self) -> jnp.ndarray:
        if not self.state_history:
            return jnp.zeros((0,))
        mask = self.moving_domain.get_discretization_domain().get_physical_active_mask()
        return _compact_on_host(self.state_history[0], mask)

    def get_output_nodes(self) -> jnp.ndarray:
        self._require_initialized()
        domain = self.moving_domain.get_discretization_domain()
        return _compact_on_host(domain.get_int_bdry_nodes(), domain.get_physical_active_mask())

    def get_current_domain(self) -> DomainDescriptor:
        self._require_initialized()
        return self.moving_domain.get_current_domain()

    def _advance_embedded_surfaces(self, velocity: VelocityCallback, next_time: float) -> None:
        h = self.moving_domain.get_background_domain().get_sep_rad()
        for index, seeds in enumerate(self.surface_seeds):
            moved_seeds = rk3_step(self.current_time, seeds, self.dt, velocity)
            self.surface_seeds[index] = moved_seeds
            if self.surface_builder is None:
                self.surface_models[index] = self.boundary_models[index].evaluate(moved_seeds, h)
            else:
                self.surface_models[index] = self.surface_builder(moved_seeds, h, index, next_time)

    def _solve_system(
        self,
        system: SparseCOOMatrix,
        rhs: jnp.ndarray,
        guess: jnp.ndarray,
        tolerance: float,
    ) -> tuple[jnp.ndarray, int, bool]:
        if system.shape[0] != system.shape[1]:
            raise ValueError("moving-domain ghost closure must produce a square implicit system")
        if self.linear_solver == "dense":
            solution = jnp.linalg.solve(sparse_matrix_to_dense(system), rhs)
            info = 0
            used_fallback = False
            residual = _relative_residual(system.indices, system.values, system.shape[0], solution, rhs)
        elif self.linear_solver == "gmres":
            preconditioner_diagonal = _paper_preconditioner_diagonal(
                system,
                self.moving_domain.get_discretization_domain().get_num_int_bdry_nodes(),
            )
            solution, info_array = _solve_sparse_gmres(
                system.indices,
                system.values,
                rhs,
                guess,
                preconditioner_diagonal,
                self.gmres_internal_tolerance_factor * tolerance,
                shape=system.shape,
                restart=min(self.gmres_restart, system.shape[0]),
                maxiter=self.gmres_max_iterations,
            )
            residual = _relative_residual(system.indices, system.values, system.shape[0], solution, rhs)
            info = info_array
            used_fallback = False
            if self.dense_fallback_on_failure:
                residual_host = float(residual)
                info_host = int(info_array)
                finite_solution = bool(jnp.all(jnp.isfinite(solution))) and np.isfinite(residual_host)
                used_fallback = not finite_solution or (
                    info_host != 0
                    and np.floor(np.log10(max(residual_host, np.finfo(float).tiny)))
                    > np.floor(np.log10(max(tolerance, np.finfo(float).tiny)))
                )
                if used_fallback:
                    solution = jnp.linalg.solve(sparse_matrix_to_dense(system), rhs)
                    residual = _relative_residual(system.indices, system.values, system.shape[0], solution, rhs)
                info = info_host
        else:
            raise ValueError("linear_solver must be 'gmres' or 'dense'")
        if self.linear_solver == "dense" or self.dense_fallback_on_failure:
            residual_host = float(residual)
            if not bool(jnp.all(jnp.isfinite(solution))) or not np.isfinite(residual_host):
                raise RuntimeError("moving-domain linear solve produced nonfinite values")
        self.last_linear_diagnostics = {
            "relative_residual": residual,
            "tolerance": tolerance,
        }
        return solution, info, used_fallback

    def _require_initialized(self) -> None:
        if self.moving_domain is None:
            raise RuntimeError("call init before using the moving-domain solver")

    def _require_ready(self) -> None:
        self._require_initialized()
        if not self.state_history:
            raise RuntimeError("call set_initial_state before stepping")
