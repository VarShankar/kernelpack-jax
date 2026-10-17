from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.io import savemat

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

INV = Path(__file__).resolve().parents[1] / "inverse problems"
if str(INV) not in sys.path:
    sys.path.insert(0, str(INV))

from common import build_domain, default_geometry, default_h, fit_with_optax_jitted, relative_l2
from kernelpack.domain import DomainDescriptor
from kernelpack.geometry import RBFLevelSet
from kernelpack.solvers import EulerianAdvectionDiffusionReactionSolver, StatelessEulerianAdvectionDiffusionReactionSolver
from reaction_diffusion_inverse_convergence_study import SavedDomainMetadata, validate_geometry


DEFAULT_MAT_OUTPUT = Path(__file__).with_name("advection_velocity_inverse_convergence_study.mat")


@dataclass(frozen=True)
class InverseResult:
    order: int
    resolution: str
    h: float
    dt: float
    steps: int
    final_time: float
    node_count: int
    theta_true: np.ndarray
    theta_fit: np.ndarray
    parameter_abs_error: float
    parameter_rel_error: float
    velocity_rel_error: float
    objective_norm: float
    baseline_objective_norm: float
    elapsed_seconds: float
    preprocess_elapsed_seconds: float


def load_domain_file(path: Path, *, rebuild_level_set: bool) -> tuple[DomainDescriptor, SavedDomainMetadata]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        bdry_nodes = jnp.asarray(data["bdry_nodes"], dtype=float)
        normals = jnp.asarray(data["normals"], dtype=float)
        descriptor = DomainDescriptor()
        descriptor.set_nodes(
            jnp.asarray(data["int_nodes"], dtype=float),
            bdry_nodes,
            jnp.asarray(data["ghost_nodes"], dtype=float),
        )
        descriptor.set_normals(normals)
        descriptor.set_sep_rad(float(np.asarray(data["sep_rad"]).item()))
        descriptor.build_structs()
        metadata = SavedDomainMetadata(
            dim=int(np.asarray(data["dim"]).item()),
            geometry=str(np.asarray(data["geometry"]).item()),
            h=float(np.asarray(data["h"]).item()),
            seed=int(np.asarray(data["seed"]).item()),
            sep_rad=float(np.asarray(data["sep_rad"]).item()),
            label=str(np.asarray(data["label"]).item()),
        )
        if rebuild_level_set:
            level_set = RBFLevelSet()
            level_set.build_level_set_from_cfi(bdry_nodes, normals)
            descriptor.set_outer_level_set(level_set)
            descriptor.set_boundary_level_sets([level_set])
    return descriptor, metadata


def timestep_from_h(h: float, final_time: float, dt_factor: float, min_steps: int) -> tuple[float, int]:
    if h <= 0.0:
        raise ValueError("h must be positive for h-dependent timestep selection")
    steps = max(int(min_steps), int(np.ceil(float(final_time) / (float(dt_factor) * float(h)))))
    return float(final_time) / float(steps), steps


def _level_set_values_or_none(domain: DomainDescriptor, x: jnp.ndarray) -> jnp.ndarray | None:
    level_set = domain.get_outer_level_set()
    if level_set is None:
        return None
    return jnp.asarray(level_set.evaluate(x), dtype=float)


def boundary_cutoff(domain: DomainDescriptor, x: jnp.ndarray) -> jnp.ndarray:
    """Smoothly damp velocity and initial scalar data near the boundary.

    In-memory example domains carry an outer level-set object. Saved-domain
    NPZ files do not, so for those cases we fall back to distance from the
    saved boundary nodes.
    """

    h = float(domain.get_sep_rad())
    level_values = _level_set_values_or_none(domain, x)
    if level_values is not None:
        phi = jnp.maximum(level_values, 0.0)
    else:
        xb = jnp.asarray(domain.get_bdry_nodes(), dtype=float)
        distances = jnp.linalg.norm(x[:, None, :] - xb[None, :, :], axis=2)
        phi = jnp.min(distances, axis=1)
    return (phi / (h + phi + 1.0e-14)) ** 2


def _normalize_vector_field(field: jnp.ndarray) -> jnp.ndarray:
    scale = jnp.max(jnp.linalg.norm(field, axis=1))
    return jnp.where(scale > 1.0e-12, field / scale, jnp.zeros_like(field))


def _level_set_values_and_gradients(domain: DomainDescriptor, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    level_set = domain.get_outer_level_set()
    if level_set is None:
        raise ValueError("The incompressible velocity basis requires a rebuilt outer level set.")
    raw_phi = jnp.asarray(level_set.evaluate(x), dtype=float)
    raw_grad = jnp.asarray(level_set.evaluate_gradient(x), dtype=float)
    inside = raw_phi >= 0.0
    return jnp.where(inside, raw_phi, 0.0), jnp.where(inside[:, None], raw_grad, 0.0)


def _c2_not_c3(value: jnp.ndarray) -> jnp.ndarray:
    """C2 but not C3 across value=0, so its gradient is C1 but not C2."""

    return jnp.where(value >= 0.0, value**3, -(value**3))


def _c2_not_c3_derivative(value: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(value >= 0.0, 3.0 * value**2, -3.0 * value**2)


def _mode_and_gradient(basis: str, index: int, x: jnp.ndarray, center: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    shifted = x - center
    pi = jnp.pi
    dim = int(x.shape[1])
    grad = jnp.zeros_like(x)
    if dim == 2:
        x0, y0 = x[:, 0], x[:, 1]
        sx, sy = shifted[:, 0], shifted[:, 1]
        if basis == "smooth":
            if index == 0:
                mode = 0.5 * (sx**2 + sy**2)
                grad = jnp.column_stack([sx, sy])
            elif index == 1:
                mode = jnp.sin(pi * x0) * jnp.sin(pi * y0)
                grad = jnp.column_stack([pi * jnp.cos(pi * x0) * jnp.sin(pi * y0), pi * jnp.sin(pi * x0) * jnp.cos(pi * y0)])
            else:
                mode = sx * sy + 0.2 * sx**3
                grad = jnp.column_stack([sy + 0.6 * sx**2, sx])
        elif basis == "c1":
            if index == 0:
                mode = _c2_not_c3(sx)
                grad = jnp.column_stack([_c2_not_c3_derivative(sx), jnp.zeros_like(sx)])
            elif index == 1:
                mode = _c2_not_c3(sy)
                grad = jnp.column_stack([jnp.zeros_like(sy), _c2_not_c3_derivative(sy)])
            else:
                mode = _c2_not_c3(sy) + 0.2 * sx * sy
                grad = jnp.column_stack([0.2 * sy, _c2_not_c3_derivative(sy) + 0.2 * sx])
        else:
            raise ValueError(f"Unknown velocity basis '{basis}'")
        return mode, grad
    if dim == 3:
        x0, y0, z0 = x[:, 0], x[:, 1], x[:, 2]
        sx, sy, sz = shifted[:, 0], shifted[:, 1], shifted[:, 2]
        if basis == "smooth":
            if index == 0:
                mode = 0.5 * (sx**2 + sy**2)
                grad = jnp.column_stack([sx, sy, jnp.zeros_like(sx)])
            elif index == 1:
                mode = sx * sz
                grad = jnp.column_stack([sz, jnp.zeros_like(sx), sx])
            else:
                mode = sy * sz + 0.15 * sx**2
                grad = jnp.column_stack([0.3 * sx, sz, sy])
        elif basis == "c1":
            if index == 0:
                mode = _c2_not_c3(sx)
                grad = jnp.column_stack([_c2_not_c3_derivative(sx), jnp.zeros_like(sx), jnp.zeros_like(sx)])
            elif index == 1:
                mode = _c2_not_c3(sy)
                grad = jnp.column_stack([jnp.zeros_like(sy), _c2_not_c3_derivative(sy), jnp.zeros_like(sy)])
            else:
                mode = _c2_not_c3(sz) + 0.2 * sx * sy
                grad = jnp.column_stack([0.2 * sy, 0.2 * sx, _c2_not_c3_derivative(sz)])
        else:
            raise ValueError(f"Unknown velocity basis '{basis}'")
        return mode, grad
    raise ValueError("Only dim=2 or dim=3 is supported")


def velocity_basis(domain: DomainDescriptor, x: jnp.ndarray, basis: str) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=float)
    dim = x.shape[1]
    center = jnp.mean(jnp.asarray(domain.get_int_bdry_nodes(), dtype=float), axis=0)
    if dim not in (2, 3):
        raise ValueError("Only dim=2 or dim=3 is supported")
    phi, grad_phi = _level_set_values_and_gradients(domain, x)
    fields = []
    for index in range(3):
        mode, grad_mode = _mode_and_gradient(basis, index, x, center)
        grad_potential = 2.0 * phi[:, None] * grad_phi * mode[:, None] + (phi**2)[:, None] * grad_mode
        if dim == 2:
            field = jnp.column_stack([grad_potential[:, 1], -grad_potential[:, 0]])
        else:
            component = index % 3
            if component == 0:
                field = jnp.column_stack([jnp.zeros_like(phi), -grad_potential[:, 2], grad_potential[:, 1]])
            elif component == 1:
                field = jnp.column_stack([grad_potential[:, 2], jnp.zeros_like(phi), -grad_potential[:, 0]])
            else:
                field = jnp.column_stack([-grad_potential[:, 1], grad_potential[:, 0], jnp.zeros_like(phi)])
        fields.append(_normalize_vector_field(field))
    return jnp.stack(fields, axis=0)


def initial_scalar(domain: DomainDescriptor, x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=float)
    cutoff = boundary_cutoff(domain, x)
    if x.shape[1] == 2:
        mode = jnp.sin(1.3 * x[:, 0] - 0.7 * x[:, 1]) + 0.35 * jnp.cos(0.8 * x[:, 0] + 1.1 * x[:, 1])
    else:
        mode = jnp.sin(1.0 * x[:, 0] - 0.6 * x[:, 1] + 0.4 * x[:, 2]) + 0.30 * jnp.cos(0.5 * x[:, 0] + 0.9 * x[:, 2])
    return cutoff * mode


def theta_reference(dim: int) -> jnp.ndarray:
    return jnp.asarray([1.45, -0.90, 0.60], dtype=float) if dim == 2 else jnp.asarray([1.20, -0.80, 0.50], dtype=float)


def build_solver(domain: DomainDescriptor, order: int, dt: float, nu: float, gamma_scale: float, linear_solver: str) -> EulerianAdvectionDiffusionReactionSolver:
    dim = int(domain.get_dim())
    gamma_ref = gamma_scale * jnp.ones((dim,), dtype=float)
    solver = EulerianAdvectionDiffusionReactionSolver(linear_solver=linear_solver)
    solver.init(domain, order, dt, nu, hyperviscosity_power=2, gamma_components=gamma_ref)
    return solver


def make_rollout(
    solver: EulerianAdvectionDiffusionReactionSolver,
    *,
    dt: float,
    steps: int,
    observation_mode: str,
    velocity_basis_name: str,
) -> tuple[callable, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    neu_coeff = lambda xb: jnp.zeros((xb.shape[0],), dtype=float)
    dir_coeff = lambda xb: jnp.ones((xb.shape[0],), dtype=float)
    solver.prepare_boundary_operator(0.0, neu_coeff, dir_coeff)
    if solver.linear_solver == "gmres_ilu":
        solver.prepare_preconditioner(0.0, neu_coeff, dir_coeff, bdf_order=min(max(steps, 1), 3))
    stateless = StatelessEulerianAdvectionDiffusionReactionSolver.from_solver(solver)
    all_nodes = jnp.asarray(solver.domain.get_all_nodes(), dtype=float)
    x_phys = jnp.asarray(solver.domain.get_int_bdry_nodes(), dtype=float)
    c0_full = initial_scalar(solver.domain, all_nodes)
    basis_phys = velocity_basis(solver.domain, x_phys, velocity_basis_name)
    gamma_ref = solver.gamma_components

    zero_dirichlet = lambda neu_coeffs, dir_coeffs, nr, time, xb: jnp.zeros((xb.shape[0],), dtype=float)
    zero_reaction = lambda time, state, x: jnp.zeros_like(state)

    def velocity_from_theta(theta: jnp.ndarray):
        def velocity(time, x):
            del time, x
            return jnp.einsum("k,knd->nd", theta, basis_phys)

        return velocity

    def rollout(theta: jnp.ndarray) -> jnp.ndarray:
        velocity = velocity_from_theta(theta)
        snapshots = []
        u0 = c0_full
        if steps <= 0:
            return jnp.zeros((0, solver.n), dtype=float)
        u1 = stateless.bdf1_full_state(
            u0,
            0.0,
            dt,
            velocity=velocity,
            reaction=zero_reaction,
            bc=zero_dirichlet,
            forcing=None,
            gamma_components=gamma_ref,
        )
        snapshots.append(u1[: solver.n])
        if steps == 1:
            return snapshots[-1] if observation_mode == "final" else jnp.stack(snapshots, axis=0)
        u2 = stateless.bdf2_full_state(
            u0,
            u1,
            0.0,
            dt,
            2.0 * dt,
            velocity=velocity,
            reaction=zero_reaction,
            bc=zero_dirichlet,
            forcing=None,
            gamma_components=gamma_ref,
        )
        snapshots.append(u2[: solver.n])
        if steps == 2:
            return snapshots[-1] if observation_mode == "final" else jnp.stack(snapshots, axis=0)
        u_nm2, u_nm1, u_n = u0, u1, u2
        for k in range(2, steps):
            u_np1 = stateless.bdf3_full_state(
                u_nm2,
                u_nm1,
                u_n,
                (k - 2) * dt,
                (k - 1) * dt,
                k * dt,
                (k + 1) * dt,
                velocity=velocity,
                reaction=zero_reaction,
                bc=zero_dirichlet,
                forcing=None,
                gamma_components=gamma_ref,
            )
            snapshots.append(u_np1[: solver.n])
            u_nm2, u_nm1, u_n = u_nm1, u_n, u_np1
        return snapshots[-1] if observation_mode == "final" else jnp.stack(snapshots, axis=0)

    return rollout, basis_phys, c0_full, gamma_ref


def run_case(
    *,
    domain: DomainDescriptor,
    metadata: SavedDomainMetadata,
    order: int,
    final_time: float,
    dt_factor: float,
    min_steps: int,
    fixed_steps: int | None,
    nu: float,
    gamma_scale: float,
    linear_solver: str,
    learning_rate: float,
    opt_steps: int,
    observation_mode: str,
    velocity_basis_name: str,
    timing_warmup: bool,
) -> InverseResult:
    if fixed_steps is None:
        dt, steps = timestep_from_h(metadata.h, final_time, dt_factor, min_steps)
    else:
        steps = int(fixed_steps)
        dt = float(final_time) / float(steps)

    theta_true = theta_reference(metadata.dim)
    theta0 = jnp.zeros_like(theta_true)

    def recover_once():
        solver = build_solver(domain, order, dt, nu, gamma_scale, linear_solver)
        rollout, basis_phys, _c0_full, _gamma_ref = make_rollout(
            solver,
            dt=dt,
            steps=steps,
            observation_mode=observation_mode,
            velocity_basis_name=velocity_basis_name,
        )
        observed = jax.lax.stop_gradient(rollout(theta_true))

        def loss_fn(theta: jnp.ndarray) -> jnp.ndarray:
            pred = rollout(theta)
            return jnp.sum((pred - observed) ** 2) / jnp.maximum(jnp.sum(observed**2), 1.0e-14)

        theta_fit, _history, baseline, fitted = fit_with_optax_jitted(
            loss_fn,
            theta0,
            optimizer=optax.adam(learning_rate),
            steps=opt_steps,
        )
        velocity_true = jnp.einsum("k,knd->nd", theta_true, basis_phys)
        velocity_fit = jnp.einsum("k,knd->nd", theta_fit, basis_phys)
        velocity_rel = relative_l2(velocity_fit.reshape(-1), velocity_true.reshape(-1))
        return theta_fit, fitted, baseline, velocity_rel

    if timing_warmup:
        preprocess_start = perf_counter()
        theta_fit, objective_norm, baseline_objective_norm, velocity_rel_error = recover_once()
        jax.block_until_ready(theta_fit)
        preprocess_elapsed_seconds = perf_counter() - preprocess_start
    else:
        preprocess_elapsed_seconds = 0.0

    start = perf_counter()
    theta_fit, objective_norm, baseline_objective_norm, velocity_rel_error = recover_once()
    jax.block_until_ready(theta_fit)
    elapsed_seconds = perf_counter() - start

    param_abs = float(jnp.linalg.norm(theta_fit - theta_true))
    param_rel = float(param_abs / jnp.maximum(jnp.linalg.norm(theta_true), 1.0e-14))
    return InverseResult(
        order=order,
        resolution=metadata.label,
        h=metadata.h,
        dt=dt,
        steps=steps,
        final_time=final_time,
        node_count=int(domain.get_num_int_bdry_nodes()),
        theta_true=np.asarray(theta_true),
        theta_fit=np.asarray(theta_fit),
        parameter_abs_error=param_abs,
        parameter_rel_error=param_rel,
        velocity_rel_error=float(velocity_rel_error),
        objective_norm=float(objective_norm),
        baseline_objective_norm=float(baseline_objective_norm),
        elapsed_seconds=elapsed_seconds,
        preprocess_elapsed_seconds=preprocess_elapsed_seconds,
    )


def export_mat(
    results: list[InverseResult],
    *,
    dim: int,
    geometry: str,
    linear_solver: str,
    final_time: float,
    velocity_basis_name: str,
    output_path: Path,
) -> None:
    orders = sorted({row.order for row in results})
    labels = []
    seen = set()
    for row in results:
        if row.resolution not in seen:
            labels.append(row.resolution)
            seen.add(row.resolution)
    order_to_idx = {order: i for i, order in enumerate(orders)}
    label_to_idx = {label: i for i, label in enumerate(labels)}
    shape = (len(orders), len(labels))
    parameter_rel_error = np.full(shape, np.nan)
    parameter_abs_error = np.full(shape, np.nan)
    velocity_rel_error = np.full(shape, np.nan)
    objective_norm = np.full(shape, np.nan)
    elapsed_seconds = np.full(shape, np.nan)
    preprocess_elapsed_seconds = np.full(shape, np.nan)
    node_counts = np.full(shape, np.nan)
    dt_values = np.full(shape, np.nan)
    steps_values = np.full(shape, np.nan)
    hs = np.full((len(labels),), np.nan)
    theta_true = np.full((len(orders), len(labels), 3), np.nan)
    theta_fit = np.full((len(orders), len(labels), 3), np.nan)
    for row in results:
        i = order_to_idx[row.order]
        j = label_to_idx[row.resolution]
        parameter_rel_error[i, j] = row.parameter_rel_error
        parameter_abs_error[i, j] = row.parameter_abs_error
        velocity_rel_error[i, j] = row.velocity_rel_error
        objective_norm[i, j] = row.objective_norm
        elapsed_seconds[i, j] = row.elapsed_seconds
        preprocess_elapsed_seconds[i, j] = row.preprocess_elapsed_seconds
        node_counts[i, j] = row.node_count
        dt_values[i, j] = row.dt
        steps_values[i, j] = row.steps
        hs[j] = row.h
        theta_true[i, j, : row.theta_true.size] = row.theta_true
        theta_fit[i, j, : row.theta_fit.size] = row.theta_fit

    output_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        output_path,
        {
            "parameter_rel_error": parameter_rel_error,
            "parameter_abs_error": parameter_abs_error,
            "velocity_rel_error": velocity_rel_error,
            "objective_norm": objective_norm,
            "elapsed_seconds": elapsed_seconds,
            "preprocess_elapsed_seconds": preprocess_elapsed_seconds,
            "orders": np.asarray(orders, dtype=float),
            "hs": hs,
            "labels": np.asarray(labels, dtype=object),
            "node_counts": node_counts,
            "dt_values": dt_values,
            "steps_values": steps_values,
            "theta_true": theta_true,
            "theta_fit": theta_fit,
            "geometry": np.asarray([geometry], dtype=object),
            "family": np.asarray([f"velocity_{velocity_basis_name}"], dtype=object),
            "velocity_basis": np.asarray([velocity_basis_name], dtype=object),
            "dimension": np.asarray([[dim]], dtype=float),
            "linear_solver": np.asarray([linear_solver], dtype=object),
            "final_time": np.asarray([[final_time]], dtype=float),
            "equation": np.asarray(["advection velocity inverse"], dtype=object),
        },
    )


def render_markdown(results: list[InverseResult], *, dim: int, geometry: str, linear_solver: str, final_time: float, velocity_basis_name: str) -> str:
    lines = [
        "# Advection Velocity Inverse Convergence Study",
        "",
        f"- dimension: {dim}",
        f"- geometry: {geometry}",
        f"- linear solver: {linear_solver}",
        f"- velocity basis: {velocity_basis_name}",
        f"- final time: {final_time}",
        f"- JAX backend: `{jax.default_backend()}`",
        "",
        "| xi | resolution | h | dt | steps | physical nodes | parameter rel error | velocity rel error | objective | seconds |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {row.order} | {row.resolution} | {row.h:.5f} | {row.dt:.6e} | {row.steps} | {row.node_count} | "
            f"{row.parameter_rel_error:.6e} | {row.velocity_rel_error:.6e} | {row.objective_norm:.6e} | {row.elapsed_seconds:.6f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advection velocity inverse convergence study.")
    parser.add_argument("--dim", type=int, default=2, choices=(2, 3))
    parser.add_argument("--geometry", type=str, default=None)
    parser.add_argument("--orders", type=int, nargs="+", default=[2, 4, 6], help="Eulerian RBF-FD xi values.")
    parser.add_argument("--hs", type=float, nargs="+", default=[0.45, 0.35, 0.25, 0.15])
    parser.add_argument("--domain-files", type=Path, nargs="+", default=None)
    parser.add_argument("--linear-solver", type=str, default="lineax", choices=("gmres_ilu", "dense", "lineax"))
    parser.add_argument("--nu", type=float, default=0.03)
    parser.add_argument("--gamma-scale", type=float, default=-1.0e-4)
    parser.add_argument("--velocity-basis", choices=("smooth", "c1"), default="smooth")
    parser.add_argument(
        "--rebuild-level-set",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rebuild and attach H-channel/ice-cap level sets for saved domains; required by the incompressible velocity basis.",
    )
    parser.add_argument("--final-time", type=float, default=0.1)
    parser.add_argument("--dt-factor", type=float, default=0.25)
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--opt-steps", type=int, default=30)
    parser.add_argument("--observation-mode", choices=("final", "all"), default="final")
    parser.add_argument("--timing-warmup", action="store_true")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output", type=Path, default=None, help="Optional markdown output path. Omit to write only .mat output.")
    parser.add_argument("--mat-output", type=Path, default=DEFAULT_MAT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    geometry_name = args.geometry or default_geometry(args.dim)
    if args.domain_files:
        domain_entries = [load_domain_file(path, rebuild_level_set=args.rebuild_level_set) for path in args.domain_files]
    else:
        validate_geometry(args.dim, geometry_name)
        domain_entries = []
        for h in args.hs:
            domain = build_domain(args.dim, geometry_name, h, args.seed)
            metadata = SavedDomainMetadata(args.dim, geometry_name, float(h), args.seed, float(domain.get_sep_rad()), f"{geometry_name}_d{args.dim}_h{h:.5f}_seed{args.seed}")
            domain_entries.append((domain, metadata))

    results: list[InverseResult] = []
    for domain, metadata in domain_entries:
        for order in args.orders:
            results.append(
                run_case(
                    domain=domain,
                    metadata=metadata,
                    order=order,
                    final_time=args.final_time,
                    dt_factor=args.dt_factor,
                    min_steps=args.min_steps,
                    fixed_steps=args.steps,
                    nu=args.nu,
                    gamma_scale=args.gamma_scale,
                    linear_solver=args.linear_solver,
                    learning_rate=args.lr,
                    opt_steps=args.opt_steps,
                    observation_mode=args.observation_mode,
                    velocity_basis_name=args.velocity_basis,
                    timing_warmup=args.timing_warmup,
                )
            )

    markdown = render_markdown(
        results,
        dim=domain_entries[0][1].dim,
        geometry=geometry_name,
        linear_solver=args.linear_solver,
        final_time=args.final_time,
        velocity_basis_name=args.velocity_basis,
    )
    print(markdown, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    export_mat(
        results,
        dim=domain_entries[0][1].dim,
        geometry=geometry_name,
        linear_solver=args.linear_solver,
        final_time=args.final_time,
        velocity_basis_name=args.velocity_basis,
        output_path=args.mat_output,
    )
    print(f"wrote matlab output to {args.mat_output}")


if __name__ == "__main__":
    main()
