from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import sys
from time import perf_counter

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax
import numpy as np
import optax
import optimistix as optx
from scipy.io import savemat
from scipy.spatial import cKDTree

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

INV = Path(__file__).resolve().parents[1] / "inverse problems"
if str(INV) not in sys.path:
    sys.path.insert(0, str(INV))

from kernelpack.domain import DomainDescriptor
from kernelpack.solvers import StatelessVariablePoissonSolver, VariablePoissonSolver

from common import (
    StaticInverseCase,
    build_domain,
    compile_loss_and_grad,
    default_h,
    manufactured_state_1,
    relative_l2,
    run_optax_jitted_summary,
)
from darcy_model_selection import coefficient_field, manufactured_forcing, model_features

MAX_OPTIMIZATION_ITERATIONS = 60
DEFAULT_LEARNING_RATE = 0.08
RESIDUAL_TOLERANCE = 1.0e-6
DEFAULT_NOISE_STD = 0.05
DEFAULT_NOISE_SEED = 123
NOISE_MODES = ("relative", "absolute")
RESIDUAL_MODES = ("absolute", "relative")
FD_CALLBACK_EPS = 1.0e-5
TWO_D_GEOMETRIES = ("ellipse", "clover", "peanut")
THREE_D_GEOMETRIES = ("ellipsoid", "peanut", "tilted_egg")
FAMILIES = ("constant", "affine", "quadratic_diag", "cross_term", "sin_product")
LM_JACOBIAN_MODES = ("ad", "fd_callback")
SOURCE_MODES = ("manufactured", "fixed")


@dataclass(frozen=True)
class SavedDomainMetadata:
    dim: int
    geometry: str
    h: float
    seed: int
    sep_rad: float
    label: str


@dataclass(frozen=True)
class StudyResult:
    order: int
    resolution: str
    h: float
    node_count: int
    iterations_used: int
    parameter_abs_error: float
    parameter_rel_error: float
    objective_norm: float
    coefficient_abs_error: float
    coefficient_rel_error: float
    state_fit_true_norm: float
    observed_target_error_norm: float
    state_fit_true_over_observed_error: float
    jacobian_sigma_max: float
    jacobian_sigma_min: float
    jacobian_condition: float
    jacobian_singular_values: tuple[float, ...]
    elapsed_seconds: float
    preprocess_elapsed_seconds: float
    stop_reason: str


TRUE_THETA_TABLE = {
    ("constant", 2): jnp.array([0.12], dtype=float),
    ("constant", 3): jnp.array([0.09], dtype=float),
    ("affine", 2): jnp.array([0.08, 0.22, -0.16], dtype=float),
    ("affine", 3): jnp.array([0.06, 0.18, -0.11, 0.09], dtype=float),
    ("quadratic_diag", 2): jnp.array([0.05, 0.18, -0.12, 0.08, -0.04], dtype=float),
    ("quadratic_diag", 3): jnp.array([0.04, 0.15, -0.08, 0.07, 0.09, -0.05], dtype=float),
    ("cross_term", 2): jnp.array([0.10, 0.30, -0.22, 0.12], dtype=float),
    ("cross_term", 3): jnp.array([0.07, 0.20, -0.15, 0.10, 0.14], dtype=float),
    ("sin_product", 2): jnp.array([0.08, 0.22], dtype=float),
    ("sin_product", 3): jnp.array([0.06, 0.20], dtype=float),
}


def valid_geometries(dim: int) -> tuple[str, ...]:
    if dim == 2:
        return TWO_D_GEOMETRIES
    if dim == 3:
        return THREE_D_GEOMETRIES
    raise ValueError("Only dim=2 or dim=3 is supported")


def validate_geometry(dim: int, geometry_name: str) -> None:
    allowed = valid_geometries(dim)
    if geometry_name not in allowed:
        raise ValueError(f"Geometry '{geometry_name}' is not valid for dim={dim}. Choose one of {allowed}.")


def default_geometry(dim: int) -> str:
    return "ellipse" if dim == 2 else "ellipsoid"


def load_domain_file(path: Path) -> tuple[DomainDescriptor, SavedDomainMetadata]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        descriptor = DomainDescriptor()
        descriptor.set_nodes(
            jnp.asarray(data["int_nodes"], dtype=float),
            jnp.asarray(data["bdry_nodes"], dtype=float),
            jnp.asarray(data["ghost_nodes"], dtype=float),
        )
        descriptor.set_normals(jnp.asarray(data["normals"], dtype=float))
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
    return descriptor, metadata


def setup_device_context(setup_device: str):
    if setup_device == "cpu":
        cpu_devices = jax.devices("cpu")
        if not cpu_devices:
            raise RuntimeError("requested CPU setup device, but JAX did not report a CPU device")
        return jax.default_device(cpu_devices[0])
    if setup_device == "default":
        return nullcontext()
    raise ValueError(f"unknown setup device {setup_device}")


def copy_domain_to_device(domain: DomainDescriptor, device) -> DomainDescriptor:
    copied = DomainDescriptor()
    copied.set_nodes(
        jax.device_put(domain.get_interior_nodes(), device),
        jax.device_put(domain.get_bdry_nodes(), device),
        jax.device_put(domain.get_ghost_nodes(), device),
    )
    copied.set_normals(jax.device_put(domain.get_nrmls(), device))
    copied.set_sep_rad(float(domain.get_sep_rad()))
    copied.set_outer_level_set(domain.get_outer_level_set())
    copied.set_boundary_level_sets(domain.get_boundary_level_sets())
    copied.build_structs()
    return copied


def sample_theta(family: str, dim: int) -> jnp.ndarray:
    try:
        return TRUE_THETA_TABLE[(family, dim)]
    except KeyError as exc:
        raise ValueError(f"No true parameter template registered for family={family}, dim={dim}") from exc


def build_cases(family: str, theta_true: jnp.ndarray, x_phys: jnp.ndarray, x_bdry: jnp.ndarray) -> list[StaticInverseCase]:
    state_fns = [manufactured_state_1]
    return [
        StaticInverseCase(
            forcing=manufactured_forcing(family, theta_true, x_phys, fn),
            target=fn(x_phys),
            boundary_target=fn(x_bdry),
        )
        for fn in state_fns
    ]


def add_observation_noise(values: jnp.ndarray, noise_std: float, rng: np.random.Generator, noise_mode: str) -> jnp.ndarray:
    if noise_std <= 0.0:
        return values
    if noise_mode == "relative":
        scale = np.abs(np.asarray(values))
    elif noise_mode == "absolute":
        scale = 1.0
    else:
        raise ValueError(f"Unknown noise mode '{noise_mode}'. Choose one of {NOISE_MODES}.")
    noise = rng.normal(loc=0.0, scale=noise_std, size=values.shape) * scale
    return values + jnp.asarray(noise, dtype=values.dtype)


def maybe_noisify_cases(
    cases: list[StaticInverseCase],
    *,
    noise_std: float,
    noise_seed: int,
    noise_mode: str,
) -> list[StaticInverseCase]:
    if noise_std <= 0.0:
        return cases
    rng = np.random.default_rng(noise_seed)
    noisy_cases = []
    for case in cases:
        noisy_cases.append(
            StaticInverseCase(
                forcing=case.forcing,
                target=add_observation_noise(case.target, noise_std, rng, noise_mode),
                boundary_target=case.boundary_target,
            )
        )
    return noisy_cases


def smooth_observed_values(values: jnp.ndarray, x_phys: jnp.ndarray, radius: float) -> jnp.ndarray:
    if radius <= 0.0:
        return values
    points = np.asarray(x_phys)
    raw_values = np.asarray(values)
    tree = cKDTree(points)
    smoothed = np.empty_like(raw_values)
    for i, neighbors in enumerate(tree.query_ball_point(points, r=radius)):
        if not neighbors:
            smoothed[i] = raw_values[i]
            continue
        neighbor_idx = np.asarray(neighbors, dtype=int)
        distances = np.linalg.norm(points[neighbor_idx] - points[i], axis=1)
        weights = np.exp(-0.5 * (distances / radius) ** 2)
        smoothed[i] = np.sum(weights * raw_values[neighbor_idx]) / np.sum(weights)
    return jnp.asarray(smoothed, dtype=values.dtype)


def maybe_smooth_cases(
    cases: list[StaticInverseCase],
    x_phys: jnp.ndarray,
    *,
    smoothing_radius: float,
) -> list[StaticInverseCase]:
    if smoothing_radius <= 0.0:
        return cases
    return [
        StaticInverseCase(
            forcing=case.forcing,
            target=smooth_observed_values(case.target, x_phys, smoothing_radius),
            boundary_target=case.boundary_target,
        )
        for case in cases
    ]


def fixed_known_source(x: jnp.ndarray) -> jnp.ndarray:
    """Known forcing used by forward-generated inverse observations."""

    phase = 0.8 * x[:, 0] - 0.55 * x[:, 1]
    if x.shape[1] == 3:
        phase = phase + 0.35 * x[:, 2]
    return 1.0 + 0.25 * jnp.sin(phase)


def observation_path(
    observation_dir: Path,
    metadata: SavedDomainMetadata,
    family: str,
    observation_order: int,
    source_mode: str,
) -> Path:
    return observation_dir / (
        f"darcy-{metadata.label}-{family}-source-{source_mode}-order{observation_order}-observation.npz"
    )


def build_forward_observation(
    domain: DomainDescriptor,
    metadata: SavedDomainMetadata,
    family: str,
    theta_true: jnp.ndarray,
    observation_order: int,
    stencil: str,
    sparse_assembly_chunk_size: int | None,
    output_path: Path,
    source_mode: str,
) -> list[StaticInverseCase]:
    x_phys = jnp.asarray(domain.get_int_bdry_nodes(), dtype=float)
    x_bdry = jnp.asarray(domain.get_bdry_nodes(), dtype=float)
    x_all = jnp.asarray(domain.get_all_nodes(), dtype=float)
    if source_mode == "fixed":
        forcing = fixed_known_source(x_phys)
    elif source_mode == "zero":
        forcing = jnp.zeros((domain.get_num_int_bdry_nodes(),), dtype=float)
    else:
        raise ValueError(f"forward observation source mode must be fixed or zero, got {source_mode}")
    boundary = manufactured_state_1(x_bdry)
    neu_coeff = jnp.zeros((domain.get_num_bdry_nodes(),), dtype=float)
    dir_coeff = jnp.ones((domain.get_num_bdry_nodes(),), dtype=float)

    print(
        f"[observation] building forward solver: resolution={metadata.label}, family={family}, "
        f"order={observation_order}, source={source_mode}",
        flush=True,
    )
    solver = VariablePoissonSolver(
        lap_assembler="fd",
        bc_assembler="fd",
        lap_stencil=stencil,
        bc_stencil=stencil,
        linear_solver="scipy_sparse_direct",
    )
    solver.init(
        domain,
        observation_order,
        sparse_assembly_chunk_size=sparse_assembly_chunk_size,
        assemble_dense_operators=False,
    )
    solver.prepare_boundary_operator(neu_coeff, dir_coeff)
    features = model_features(family, x_phys.shape[1], x_all)[0]
    observed = solver.solve(
        forcing,
        jnp.exp(features @ theta_true),
        neu_coeff,
        dir_coeff,
        boundary,
    )["u"]
    observed = jax.block_until_ready(observed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        observed_state=np.asarray(observed),
        forcing=np.asarray(forcing),
        boundary=np.asarray(boundary),
        theta_true=np.asarray(theta_true),
        family=np.asarray(family),
        resolution=np.asarray(metadata.label),
        observation_order=np.asarray(observation_order),
        source_mode=np.asarray(source_mode),
    )
    print(f"[observation] wrote forward state to {output_path}", flush=True)
    return [StaticInverseCase(forcing=forcing, target=observed, boundary_target=boundary)]


def load_or_build_forward_cases(
    domain: DomainDescriptor,
    metadata: SavedDomainMetadata,
    family: str,
    theta_true: jnp.ndarray,
    observation_order: int,
    stencil: str,
    sparse_assembly_chunk_size: int | None,
    observation_dir: Path,
    source_mode: str,
) -> list[StaticInverseCase]:
    path = observation_path(observation_dir, metadata, family, observation_order, source_mode)
    if not path.exists():
        return build_forward_observation(
            domain,
            metadata,
            family,
            theta_true,
            observation_order,
            stencil,
            sparse_assembly_chunk_size,
            path,
            source_mode,
        )

    print(f"[observation] loading cached forward state from {path}", flush=True)
    with np.load(path, allow_pickle=False) as data:
        observed = jnp.asarray(data["observed_state"], dtype=float)
        forcing = jnp.asarray(data["forcing"], dtype=float)
        boundary = jnp.asarray(data["boundary"], dtype=float)
        stored_theta = np.asarray(data["theta_true"], dtype=float)
        stored_family = str(np.asarray(data["family"]).item())
        stored_resolution = str(np.asarray(data["resolution"]).item())
        stored_order = int(np.asarray(data["observation_order"]).item())
        stored_source_mode = str(np.asarray(data["source_mode"]).item())
    if observed.shape != (domain.get_num_int_bdry_nodes(),):
        raise ValueError(f"cached observation {path} has incompatible shape {observed.shape}")
    if forcing.shape != observed.shape or boundary.shape != (domain.get_num_bdry_nodes(),):
        raise ValueError(f"cached observation {path} has incompatible forcing or boundary shape")
    if (
        stored_family != family
        or stored_resolution != metadata.label
        or stored_order != observation_order
        or stored_source_mode != source_mode
    ):
        raise ValueError(f"cached observation metadata does not match requested case: {path}")
    if not np.allclose(stored_theta, np.asarray(theta_true), rtol=0.0, atol=1.0e-14):
        raise ValueError(f"cached observation theta does not match the registered true parameter: {path}")
    return [StaticInverseCase(forcing=forcing, target=observed, boundary_target=boundary)]


def make_fd_callback_residual(
    residual_fn,
    theta0: jnp.ndarray,
    *,
    residual_shape: tuple[int, ...],
    residual_dtype,
    fd_eps: float = FD_CALLBACK_EPS,
    diagnostics: bool = False,
):
    shape_dtype = jax.ShapeDtypeStruct(residual_shape, residual_dtype)
    counters = {"residual": 0, "gradient": 0}

    def _residual_numpy(theta_np: np.ndarray) -> np.ndarray:
        counters["residual"] += 1
        start_time = perf_counter()
        residual, _ = residual_fn(jnp.asarray(theta_np, dtype=theta0.dtype), None)
        value = np.asarray(jax.device_get(residual), dtype=np.asarray(theta_np).dtype)
        if diagnostics:
            elapsed = perf_counter() - start_time
            theta_text = np.array2string(np.asarray(theta_np), precision=6, suppress_small=False)
            print(
                f"[fd-callback] residual eval {counters['residual']}: "
                f"norm={np.linalg.norm(value):.6e}, elapsed={elapsed:.3f}s, theta={theta_text}",
                flush=True,
            )
        return value

    @jax.custom_vjp
    def residual_values(theta_raw: jnp.ndarray) -> jnp.ndarray:
        return jax.pure_callback(_residual_numpy, shape_dtype, theta_raw, vmap_method="sequential")

    def residual_fwd(theta_raw: jnp.ndarray):
        value = residual_values(theta_raw)
        return value, theta_raw

    def residual_bwd(theta_raw: jnp.ndarray, cotangent: jnp.ndarray):
        grad_shape = jax.ShapeDtypeStruct(theta_raw.shape, theta_raw.dtype)

        def _grad_numpy(theta_np: np.ndarray, cotangent_np: np.ndarray) -> np.ndarray:
            counters["gradient"] += 1
            grad_start_time = perf_counter()
            theta_np = np.asarray(theta_np)
            cotangent_np = np.asarray(cotangent_np)
            base_value = _residual_numpy(theta_np)
            jac = np.empty((base_value.size, theta_np.size), dtype=theta_np.dtype)
            if diagnostics:
                print(
                    f"[fd-callback] gradient eval {counters['gradient']}: "
                    f"finite-differencing {theta_np.size} parameter(s)",
                    flush=True,
                )
            for i in range(theta_np.size):
                column_start_time = perf_counter()
                step = fd_eps * max(1.0, abs(float(theta_np[i])))
                theta_hi = theta_np.copy()
                theta_lo = theta_np.copy()
                theta_hi[i] += step
                theta_lo[i] -= step
                jac[:, i] = (_residual_numpy(theta_hi) - _residual_numpy(theta_lo)) / (2.0 * step)
                if diagnostics:
                    print(
                        f"[fd-callback] gradient eval {counters['gradient']}: "
                        f"column {i + 1}/{theta_np.size} done, step={step:.3e}, "
                        f"elapsed={perf_counter() - column_start_time:.3f}s",
                        flush=True,
                    )
            if diagnostics:
                print(
                    f"[fd-callback] gradient eval {counters['gradient']} complete: "
                    f"elapsed={perf_counter() - grad_start_time:.3f}s",
                    flush=True,
                )
            return jac.T @ cotangent_np

        grad = jax.pure_callback(_grad_numpy, grad_shape, theta_raw, cotangent, vmap_method="sequential")
        return (grad,)

    residual_values.defvjp(residual_fwd, residual_bwd)

    def callback_residual_fn(theta_raw: jnp.ndarray, _args):
        return residual_values(theta_raw), None

    return callback_residual_fn


def build_solver(
    domain: DomainDescriptor,
    order: int,
    stencil: str,
    theta0: jnp.ndarray,
    all_features: jnp.ndarray,
    linear_solver: str,
    sparse_assembly_chunk_size: int | None,
) -> StatelessVariablePoissonSolver:
    solver = VariablePoissonSolver(
        lap_assembler="fd",
        bc_assembler="fd",
        lap_stencil=stencil,
        bc_stencil=stencil,
        linear_solver=linear_solver,
    )
    solver.init(
        domain,
        order,
        sparse_assembly_chunk_size=sparse_assembly_chunk_size,
        assemble_dense_operators=linear_solver not in ("gmres_ilu", "scipy_gmres_ilu", "cudss"),
    )
    neu_coeff = jnp.zeros((domain.get_num_bdry_nodes(),), dtype=float)
    dir_coeff = jnp.ones((domain.get_num_bdry_nodes(),), dtype=float)
    solver.prepare_boundary_operator(neu_coeff, dir_coeff)
    if linear_solver in ("gmres_ilu", "scipy_gmres_ilu"):
        preconditioner_name = "exact sparse LU" if linear_solver == "scipy_gmres_ilu" else "sparse ILU"
        print(f"[inverse] building fixed {preconditioner_name} preconditioner at theta0", flush=True)
        solver.prepare_preconditioner(
            lambda xall, feats=all_features, th=theta0: jnp.exp(feats @ th),
            neu_coeff,
            dir_coeff,
            exact_sparse_lu=linear_solver == "scipy_gmres_ilu",
        )
        print(f"[inverse] fixed {preconditioner_name} preconditioner ready", flush=True)
    return solver.freeze()


def build_loss_fn(
    solver: StatelessVariablePoissonSolver,
    all_features: jnp.ndarray,
    target_values: list[jnp.ndarray],
    forcing_values: list[jnp.ndarray],
    boundary_values: list[jnp.ndarray],
    theta_ref: jnp.ndarray,
    theta_reg: float,
    residual_mode: str,
    residual_floor: float,
):
    def loss_fn(theta: jnp.ndarray) -> jnp.ndarray:
        coeff_all = jnp.exp(all_features @ theta)
        losses = []
        for target, rhs_target, rhs_boundary in zip(target_values, forcing_values, boundary_values, strict=False):
            pred = solver.solve_u_from_fields(coeff_all, rhs_target, rhs_boundary)
            if residual_mode == "relative":
                diff = (pred - target) / (jnp.abs(target) + residual_floor)
                losses.append(jnp.mean(diff**2))
            elif residual_mode == "absolute":
                losses.append(relative_l2(pred, target) ** 2)
            else:
                raise ValueError(f"Unknown residual mode '{residual_mode}'. Choose one of {RESIDUAL_MODES}.")
        loss = jnp.mean(jnp.asarray(losses))
        if theta_reg > 0.0:
            loss = loss + theta_reg * jnp.sum((theta - theta_ref) ** 2)
        return loss

    return loss_fn


def build_residual_fn(
    solver: StatelessVariablePoissonSolver,
    all_features: jnp.ndarray,
    target_values: list[jnp.ndarray],
    forcing_values: list[jnp.ndarray],
    boundary_values: list[jnp.ndarray],
    theta_ref: jnp.ndarray,
    theta_reg: float,
    residual_mode: str,
    residual_floor: float,
):
    def residual_fn(theta: jnp.ndarray, _args):
        coeff_all = jnp.exp(all_features @ theta)
        blocks = []
        for target, rhs_target, rhs_boundary in zip(target_values, forcing_values, boundary_values, strict=False):
            pred = solver.solve_u_from_fields(coeff_all, rhs_target, rhs_boundary)
            if residual_mode == "relative":
                blocks.append(((pred - target) / (jnp.abs(target) + residual_floor)).reshape(-1))
            elif residual_mode == "absolute":
                blocks.append((pred - target).reshape(-1))
            else:
                raise ValueError(f"Unknown residual mode '{residual_mode}'. Choose one of {RESIDUAL_MODES}.")
        if theta_reg > 0.0:
            blocks.append(jnp.sqrt(theta_reg) * (theta - theta_ref))
        return jnp.concatenate(blocks, axis=0), None

    return residual_fn


def make_theta_ref(theta0: jnp.ndarray, theta_ref_values: list[float] | None) -> jnp.ndarray:
    if theta_ref_values is None:
        return jnp.zeros_like(theta0)
    theta_ref = jnp.asarray(theta_ref_values, dtype=theta0.dtype)
    if theta_ref.shape != theta0.shape:
        raise ValueError(f"--theta-ref must have {theta0.size} value(s), got {theta_ref.size}")
    return theta_ref


def run_optax_recovery(loss_fn, theta0: jnp.ndarray, learning_rate: float) -> tuple[jnp.ndarray, float, int, str]:
    _, loss_and_grad = compile_loss_and_grad(loss_fn)
    theta_fit, initial_loss, final_loss = run_optax_jitted_summary(
        loss_and_grad,
        theta0,
        optimizer=optax.adam(learning_rate),
        steps=MAX_OPTIMIZATION_ITERATIONS,
    )
    initial_loss = float(initial_loss)
    final_loss = float(final_loss)
    stop_reason = "max_iterations_reached"
    if final_loss <= initial_loss:
        stop_reason = "optimization_finished"
    return theta_fit, float(np.sqrt(max(final_loss, 0.0))), MAX_OPTIMIZATION_ITERATIONS, stop_reason


def build_lm_solver(jac_mode: str):
    linear_solver = lineax.AutoLinearSolver(well_posed=False)
    solver = optx.LevenbergMarquardt(
        atol=1.0e-8,
        rtol=1.0e-8,
        linear_solver=linear_solver,
    )
    solver_options = {"jac": jac_mode}
    tags = frozenset()
    return solver, solver_options, tags


def run_optimistix_recovery(residual_fn, theta0: jnp.ndarray, *, jac_mode: str, diagnostics: bool = False) -> tuple[jnp.ndarray, float, int, str]:
    solver, solver_options, tags = build_lm_solver(jac_mode)
    if diagnostics:
        print(f"[lm] initializing solver with jac={jac_mode}", flush=True)
    residual_struct, aux_struct = jax.eval_shape(residual_fn, theta0, None)
    state = solver.init(residual_fn, theta0, None, solver_options, residual_struct, aux_struct, tags)

    @eqx.filter_jit
    def step_fn(y, state):
        return solver.step(residual_fn, y, None, solver_options, state, tags)

    y = theta0
    aux = None
    result = None
    iterations_used = 0
    stop_reason = "max_iterations_reached"

    for _ in range(MAX_OPTIMIZATION_ITERATIONS):
        y, state, aux = step_fn(y=y, state=state)
        iterations_used += 1
        residual, _ = residual_fn(y, None)
        objective_norm = float(jnp.linalg.norm(residual))
        terminate, result = solver.terminate(residual_fn, y, None, solver_options, state, tags)
        if objective_norm <= RESIDUAL_TOLERANCE:
            stop_reason = "residual_tolerance_reached"
            break
        if bool(terminate):
            stop_reason = "solver_terminated"
            break

    if result is None:
        _, result = solver.terminate(residual_fn, y, None, solver_options, state, tags)
    y_post, aux_post, stats = solver.postprocess(residual_fn, y, aux, None, solver_options, state, tags, result)
    del aux_post, stats
    final_residual, _ = residual_fn(y_post, None)
    return y_post, float(jnp.linalg.norm(final_residual)), iterations_used, stop_reason


def run_forward_fd_lm_eager_recovery(
    residual_fn,
    theta0: jnp.ndarray,
    *,
    initial_damping: float = 1.0e-3,
    relative_step: float = 1.0e-5,
) -> tuple[jnp.ndarray, float, int, str]:
    """Finite-difference LM for residuals that call eager SciPy sparse solves."""

    residual_evaluations = 0

    def residual_only(theta: jnp.ndarray) -> jnp.ndarray:
        nonlocal residual_evaluations
        residual_evaluations += 1
        print(f"[fd-lm-eager] residual evaluation {residual_evaluations} start", flush=True)
        residual, _ = residual_fn(theta, None)
        residual = jax.block_until_ready(residual)
        residual_norm = float(jnp.linalg.norm(residual))
        print(
            f"[fd-lm-eager] residual evaluation {residual_evaluations} done: norm={residual_norm:.6e}",
            flush=True,
        )
        return residual

    def residual_and_fd_jacobian(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        residual = residual_only(theta)
        columns = []
        theta_np = np.asarray(jax.device_get(theta), dtype=float)
        print(
            f"[fd-lm-eager] finite-difference Jacobian start: parameters={theta.size}, "
            f"base_norm={float(jnp.linalg.norm(residual)):.6e}",
            flush=True,
        )
        for i in range(theta.size):
            step_size = relative_step * max(abs(theta_np[i]), 1.0)
            print(
                f"[fd-lm-eager] finite-difference column {i + 1}/{theta.size}: step={step_size:.6e}",
                flush=True,
            )
            direction = jnp.zeros_like(theta).at[i].set(step_size)
            shifted = residual_only(theta + direction)
            columns.append((shifted - residual) / step_size)
        print("[fd-lm-eager] finite-difference Jacobian done", flush=True)
        return residual, jnp.stack(columns, axis=1)

    theta = theta0
    damping = float(initial_damping)
    residual = residual_only(theta)
    objective_norm = float(jnp.linalg.norm(residual))
    print(f"[fd-lm-eager] initial objective={objective_norm:.6e}, damping={damping:.6e}", flush=True)
    stop_reason = "max_iterations_reached"
    iterations_used = 0
    identity = jnp.eye(theta0.size, dtype=theta0.dtype)
    for _ in range(MAX_OPTIMIZATION_ITERATIONS):
        print(
            f"[fd-lm-eager] iteration {iterations_used + 1}/{MAX_OPTIMIZATION_ITERATIONS} start: "
            f"objective={objective_norm:.6e}, damping={damping:.6e}",
            flush=True,
        )
        residual, jacobian = residual_and_fd_jacobian(theta)
        step = jnp.linalg.solve(jacobian.T @ jacobian + damping * identity, -(jacobian.T @ residual))
        trial_theta = theta + step
        trial_residual = residual_only(trial_theta)
        trial_objective_norm = float(jnp.linalg.norm(trial_residual))
        step_norm = float(jax.block_until_ready(jnp.linalg.norm(step)))
        theta_norm = float(jax.block_until_ready(jnp.linalg.norm(theta)))
        iterations_used += 1
        if trial_objective_norm <= objective_norm:
            theta = trial_theta
            objective_norm = trial_objective_norm
            damping = max(damping * 0.3, 1.0e-12)
            print(
                f"[fd-lm-eager] iteration {iterations_used} accepted: "
                f"trial_objective={trial_objective_norm:.6e}, step_norm={step_norm:.6e}, "
                f"theta_norm={theta_norm:.6e}, next_damping={damping:.6e}",
                flush=True,
            )
            if objective_norm <= RESIDUAL_TOLERANCE:
                stop_reason = "residual_tolerance_reached"
                break
            if step_norm <= 1.0e-10 * max(theta_norm, 1.0):
                stop_reason = "step_tolerance_reached"
                break
        else:
            damping = min(damping * 10.0, 1.0e12)
            print(
                f"[fd-lm-eager] iteration {iterations_used} rejected: "
                f"trial_objective={trial_objective_norm:.6e}, current_objective={objective_norm:.6e}, "
                f"step_norm={step_norm:.6e}, theta_norm={theta_norm:.6e}, next_damping={damping:.6e}",
                flush=True,
            )
    print(
        f"[fd-lm-eager] finished: iterations={iterations_used}, residual_evaluations={residual_evaluations}, "
        f"objective={objective_norm:.6e}, stop_reason={stop_reason}",
        flush=True,
    )
    return theta, objective_norm, iterations_used, stop_reason


def coefficient_error_metrics(family: str, theta_true: jnp.ndarray, theta_fit: jnp.ndarray, x_all: jnp.ndarray) -> tuple[float, float]:
    coeff_true = coefficient_field(family, theta_true, x_all)
    coeff_fit = coefficient_field(family, theta_fit, x_all)
    diff = coeff_fit - coeff_true
    abs_err = float(jnp.linalg.norm(diff))
    rel_err = float(abs_err / jnp.linalg.norm(coeff_true))
    return abs_err, rel_err


def inverse_identifiability_diagnostics(
    solver: StatelessVariablePoissonSolver,
    all_features: jnp.ndarray,
    theta_true: jnp.ndarray,
    theta_fit: jnp.ndarray,
    clean_target_values: list[jnp.ndarray],
    target_values: list[jnp.ndarray],
    forcing_values: list[jnp.ndarray],
    boundary_values: list[jnp.ndarray],
) -> tuple[float, float, float, float, float, float, tuple[float, ...]]:
    def state_blocks(theta: jnp.ndarray) -> jnp.ndarray:
        coeff_all = jnp.exp(all_features @ theta)
        blocks = []
        for rhs_target, rhs_boundary in zip(forcing_values, boundary_values, strict=False):
            pred = solver.solve_u_from_fields(coeff_all, rhs_target, rhs_boundary)
            blocks.append(pred.reshape(-1))
        return jnp.concatenate(blocks, axis=0)

    fit_state = state_blocks(theta_fit)
    true_state = state_blocks(theta_true)
    state_fit_true_norm = float(jnp.linalg.norm(fit_state - true_state))

    observed_target_error = jnp.concatenate(
        [(target - clean).reshape(-1) for target, clean in zip(target_values, clean_target_values, strict=False)],
        axis=0,
    )
    observed_target_error_norm = float(jnp.linalg.norm(observed_target_error))
    if observed_target_error_norm > 0.0:
        state_fit_true_over_observed_error = state_fit_true_norm / observed_target_error_norm
    else:
        state_fit_true_over_observed_error = np.nan

    jacobian = jax.jacfwd(state_blocks)(theta_true)
    singular_values = np.asarray(jnp.linalg.svd(jacobian, compute_uv=False), dtype=float)
    if singular_values.size == 0:
        sigma_max = np.nan
        sigma_min = np.nan
        condition = np.nan
    else:
        sigma_max = float(np.max(singular_values))
        sigma_min = float(np.min(singular_values))
        condition = float(sigma_max / sigma_min) if sigma_min > 0.0 else np.inf
    return (
        state_fit_true_norm,
        observed_target_error_norm,
        state_fit_true_over_observed_error,
        sigma_max,
        sigma_min,
        condition,
        tuple(float(value) for value in singular_values),
    )


def _run_case(
    *,
    domain: DomainDescriptor,
    metadata: SavedDomainMetadata,
    family: str,
    order: int,
    stencil: str,
    learning_rate: float,
    linear_solver: str,
    optimizer_name: str,
    noise_std: float,
    noise_seed: int,
    noise_mode: str,
    target_smoothing_radius_factor: float,
    timing_warmup: bool,
    lm_jacobian: str,
    theta_reg: float,
    theta_ref_values: list[float] | None,
    identifiability_diagnostics: bool,
    residual_mode: str,
    residual_floor: float,
    source_mode: str,
    observation_order: int,
    observation_dir: Path,
    sparse_assembly_chunk_size: int | None,
) -> StudyResult:
    x_phys = jnp.asarray(domain.get_int_bdry_nodes(), dtype=float)
    x_bdry = jnp.asarray(domain.get_bdry_nodes(), dtype=float)
    x_all = jnp.asarray(domain.get_all_nodes(), dtype=float)
    dim = x_phys.shape[1]
    theta_true = sample_theta(family, dim)
    all_features = model_features(family, dim, x_all)[0]
    theta0 = jnp.zeros((all_features.shape[1],), dtype=float)
    theta_ref = make_theta_ref(theta0, theta_ref_values)
    if source_mode == "manufactured":
        clean_cases = build_cases(family, theta_true, x_phys, x_bdry)
    elif source_mode in ("fixed", "zero"):
        clean_cases = load_or_build_forward_cases(
            domain,
            metadata,
            family,
            theta_true,
            observation_order,
            stencil,
            sparse_assembly_chunk_size,
            observation_dir,
            source_mode,
        )
    else:
        raise ValueError(f"unknown source mode {source_mode}")
    cases = maybe_noisify_cases(
        clean_cases,
        noise_std=noise_std,
        noise_seed=noise_seed,
        noise_mode=noise_mode,
    )
    smoothing_radius = target_smoothing_radius_factor * metadata.h
    cases = maybe_smooth_cases(cases, x_phys, smoothing_radius=smoothing_radius)
    clean_target_values = [jnp.asarray(case.target, dtype=float) for case in clean_cases]
    target_values = [jnp.asarray(case.target, dtype=float) for case in cases]
    forcing_values = [jnp.asarray(case.forcing, dtype=float) for case in cases]
    boundary_values = [jnp.asarray(case.boundary_target, dtype=float) for case in cases]

    def recover_once() -> tuple[jnp.ndarray, float, int, str]:
        solver = build_solver(
            domain,
            order,
            stencil,
            theta0,
            all_features,
            linear_solver,
            sparse_assembly_chunk_size,
        )
        loss_fn = build_loss_fn(
            solver,
            all_features,
            target_values,
            forcing_values,
            boundary_values,
            theta_ref,
            theta_reg,
            residual_mode,
            residual_floor,
        )
        residual_fn = build_residual_fn(
            solver,
            all_features,
            target_values,
            forcing_values,
            boundary_values,
            theta_ref,
            theta_reg,
            residual_mode,
            residual_floor,
        )
        if optimizer_name == "adam":
            return run_optax_recovery(loss_fn, theta0, learning_rate)
        if optimizer_name == "optimistix_lm":
            jac_mode = "fwd"
            if lm_jacobian == "fd_callback":
                residual_fn = make_fd_callback_residual(
                    residual_fn,
                    theta0,
                    residual_shape=(int(sum(case.target.size for case in cases)),),
                    residual_dtype=cases[0].target.dtype,
                    diagnostics=False,
                )
                jac_mode = "bwd"
            return run_optimistix_recovery(residual_fn, theta0, jac_mode=jac_mode, diagnostics=False)
        if optimizer_name == "forward_fd_lm_eager":
            return run_forward_fd_lm_eager_recovery(residual_fn, theta0)
        raise ValueError(f"Unknown optimizer '{optimizer_name}'")

    if timing_warmup:
        preprocess_start_time = perf_counter()
        recover_once()
        preprocess_elapsed_seconds = perf_counter() - preprocess_start_time
    else:
        preprocess_elapsed_seconds = 0.0

    start_time = perf_counter()
    theta_fit, objective_norm, iterations_used, stop_reason = recover_once()
    elapsed_seconds = perf_counter() - start_time

    param_abs_error = float(jnp.linalg.norm(theta_fit - theta_true))
    param_rel_error = float(param_abs_error / jnp.linalg.norm(theta_true))
    coeff_abs_error, coeff_rel_error = coefficient_error_metrics(family, theta_true, theta_fit, x_all)
    if identifiability_diagnostics:
        if linear_solver == "scipy_gmres_ilu":
            raise ValueError(
                "--identifiability-diagnostics currently requires a differentiable JAX linear solver; "
                "omit it for the scipy_gmres_ilu eager recovery path"
            )
        diagnostic_solver = build_solver(
            domain,
            order,
            stencil,
            theta0,
            all_features,
            linear_solver,
            sparse_assembly_chunk_size,
        )
        (
            state_fit_true_norm,
            observed_target_error_norm,
            state_fit_true_over_observed_error,
            jacobian_sigma_max,
            jacobian_sigma_min,
            jacobian_condition,
            jacobian_singular_values,
        ) = inverse_identifiability_diagnostics(
            diagnostic_solver,
            all_features,
            theta_true,
            theta_fit,
            clean_target_values,
            target_values,
            forcing_values,
            boundary_values,
        )
    else:
        state_fit_true_norm = np.nan
        observed_target_error_norm = np.nan
        state_fit_true_over_observed_error = np.nan
        jacobian_sigma_max = np.nan
        jacobian_sigma_min = np.nan
        jacobian_condition = np.nan
        jacobian_singular_values = ()

    return StudyResult(
        order=order,
        resolution=metadata.label,
        h=metadata.h,
        node_count=int(domain.get_num_int_bdry_nodes()),
        iterations_used=iterations_used,
        parameter_abs_error=param_abs_error,
        parameter_rel_error=param_rel_error,
        objective_norm=objective_norm,
        coefficient_abs_error=coeff_abs_error,
        coefficient_rel_error=coeff_rel_error,
        state_fit_true_norm=state_fit_true_norm,
        observed_target_error_norm=observed_target_error_norm,
        state_fit_true_over_observed_error=state_fit_true_over_observed_error,
        jacobian_sigma_max=jacobian_sigma_max,
        jacobian_sigma_min=jacobian_sigma_min,
        jacobian_condition=jacobian_condition,
        jacobian_singular_values=jacobian_singular_values,
        elapsed_seconds=elapsed_seconds,
        preprocess_elapsed_seconds=preprocess_elapsed_seconds,
        stop_reason=stop_reason,
    )


def run_case(
    *,
    domain: DomainDescriptor,
    setup_device: str,
    **kwargs,
) -> StudyResult:
    if setup_device == "cpu":
        setup_domain = copy_domain_to_device(domain, jax.devices("cpu")[0])
    else:
        setup_domain = domain
    with setup_device_context(setup_device):
        return _run_case(domain=setup_domain, **kwargs)


def export_inverse_mat(
    results: list[StudyResult],
    *,
    family: str,
    stencil: str,
    linear_solver: str,
    optimizer_name: str,
    noise_std: float,
    noise_seed: int,
    noise_mode: str,
    target_smoothing_radius_factor: float,
    theta_reg: float,
    theta_ref_values: list[float] | None,
    residual_mode: str,
    residual_floor: float,
    source_mode: str,
    observation_order: int,
    output_path: Path,
) -> None:
    orders = sorted({row.order for row in results})
    labels = []
    seen = set()
    for row in results:
        if row.resolution not in seen:
            labels.append(row.resolution)
            seen.add(row.resolution)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    order_to_idx = {order: i for i, order in enumerate(orders)}

    shape = (len(orders), len(labels))
    iterations_used = np.full(shape, np.nan, dtype=float)
    parameter_abs_error = np.full(shape, np.nan, dtype=float)
    parameter_rel_error = np.full(shape, np.nan, dtype=float)
    objective_norm = np.full(shape, np.nan, dtype=float)
    coefficient_abs_error = np.full(shape, np.nan, dtype=float)
    coefficient_rel_error = np.full(shape, np.nan, dtype=float)
    state_fit_true_norm = np.full(shape, np.nan, dtype=float)
    observed_target_error_norm = np.full(shape, np.nan, dtype=float)
    state_fit_true_over_observed_error = np.full(shape, np.nan, dtype=float)
    jacobian_sigma_max = np.full(shape, np.nan, dtype=float)
    jacobian_sigma_min = np.full(shape, np.nan, dtype=float)
    jacobian_condition = np.full(shape, np.nan, dtype=float)
    elapsed_seconds = np.full(shape, np.nan, dtype=float)
    preprocess_elapsed_seconds = np.full(shape, np.nan, dtype=float)
    node_counts = np.full(shape, np.nan, dtype=float)
    max_singular_values = max((len(row.jacobian_singular_values) for row in results), default=0)
    jacobian_singular_values = np.full((len(orders), len(labels), max_singular_values), np.nan, dtype=float)

    for row in results:
        i = order_to_idx[row.order]
        j = label_to_idx[row.resolution]
        iterations_used[i, j] = row.iterations_used
        parameter_abs_error[i, j] = row.parameter_abs_error
        parameter_rel_error[i, j] = row.parameter_rel_error
        objective_norm[i, j] = row.objective_norm
        coefficient_abs_error[i, j] = row.coefficient_abs_error
        coefficient_rel_error[i, j] = row.coefficient_rel_error
        state_fit_true_norm[i, j] = row.state_fit_true_norm
        observed_target_error_norm[i, j] = row.observed_target_error_norm
        state_fit_true_over_observed_error[i, j] = row.state_fit_true_over_observed_error
        jacobian_sigma_max[i, j] = row.jacobian_sigma_max
        jacobian_sigma_min[i, j] = row.jacobian_sigma_min
        jacobian_condition[i, j] = row.jacobian_condition
        for k, value in enumerate(row.jacobian_singular_values):
            jacobian_singular_values[i, j, k] = value
        elapsed_seconds[i, j] = row.elapsed_seconds
        preprocess_elapsed_seconds[i, j] = row.preprocess_elapsed_seconds
        node_counts[i, j] = row.node_count

    output_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        output_path,
        {
            "iterations_used": iterations_used,
            "parameter_abs_error": parameter_abs_error,
            "parameter_rel_error": parameter_rel_error,
            "objective_norm": objective_norm,
            "coefficient_abs_error": coefficient_abs_error,
            "coefficient_rel_error": coefficient_rel_error,
            "state_fit_true_norm": state_fit_true_norm,
            "observed_target_error_norm": observed_target_error_norm,
            "state_fit_true_over_observed_error": state_fit_true_over_observed_error,
            "jacobian_sigma_max": jacobian_sigma_max,
            "jacobian_sigma_min": jacobian_sigma_min,
            "jacobian_condition": jacobian_condition,
            "jacobian_singular_values": jacobian_singular_values,
            "elapsed_seconds": elapsed_seconds,
            "preprocess_elapsed_seconds": preprocess_elapsed_seconds,
            "orders": np.asarray(orders, dtype=float),
            "point_cloud_numbers": np.arange(1, len(labels) + 1, dtype=float),
            "labels": np.asarray(labels, dtype=object),
            "node_counts": node_counts,
            "family": np.asarray([family], dtype=object),
            "stencil": np.asarray([stencil], dtype=object),
            "linear_solver": np.asarray([linear_solver], dtype=object),
            "optimizer": np.asarray([optimizer_name], dtype=object),
            "noise_std": np.asarray([[noise_std]], dtype=float),
            "noise_seed": np.asarray([[noise_seed]], dtype=float),
            "noise_mode": np.asarray([noise_mode], dtype=object),
            "target_smoothing_radius_factor": np.asarray([[target_smoothing_radius_factor]], dtype=float),
            "theta_reg": np.asarray([[theta_reg]], dtype=float),
            "theta_ref": np.asarray(theta_ref_values if theta_ref_values is not None else [], dtype=float),
            "residual_mode": np.asarray([residual_mode], dtype=object),
            "residual_floor": np.asarray([[residual_floor]], dtype=float),
            "source_mode": np.asarray([source_mode], dtype=object),
            "observation_order": np.asarray([[observation_order]], dtype=float),
            "max_optimization_iterations": np.asarray([[MAX_OPTIMIZATION_ITERATIONS]], dtype=float),
        },
    )


def render_markdown(
    results: list[StudyResult],
    *,
    family: str,
    stencil: str,
    linear_solver: str,
    optimizer_name: str,
    noise_std: float,
    noise_seed: int,
    noise_mode: str,
    target_smoothing_radius_factor: float,
    theta_reg: float,
    theta_ref_values: list[float] | None,
    residual_mode: str,
    residual_floor: float,
    source_mode: str,
    observation_order: int,
) -> str:
    lines = [
        f"# Darcy Inverse Convergence Study ({family}, {stencil.upper()})",
        "",
        f"- JAX backend: `{jax.default_backend()}`",
        f"- Devices: `{jax.devices()}`",
        f"- Max optimization iterations: `{MAX_OPTIMIZATION_ITERATIONS}`",
        f"- Linear solver: `{linear_solver}`",
        f"- Optimizer: `{optimizer_name}`",
        f"- Source mode: `{source_mode}`",
        f"- Observation order: `{observation_order}`",
        f"- Noise std: `{noise_std}`",
        f"- Noise mode: `{noise_mode}`",
        f"- Noise seed: `{noise_seed}`",
        f"- Target smoothing radius factor: `{target_smoothing_radius_factor}`",
        f"- Residual mode: `{residual_mode}`",
        f"- Residual floor: `{residual_floor}`",
        f"- Theta regularization: `{theta_reg}`",
        f"- Theta reference: `{theta_ref_values if theta_ref_values is not None else 'zeros'}`",
        f"- Timing: `preprocess seconds is 0 unless --timing-warmup is set; seconds is the recovery pass`",
        "",
        "| xi | resolution | h | physical nodes | iterations | param abs err | param rel err | objective norm | coeff abs err | coeff rel err | state delta/noise | jac cond | preprocess seconds | seconds | stop reason |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in results:
        lines.append(
            f"| {row.order} | {row.resolution} | {row.h:.5f} | {row.node_count} | {row.iterations_used} | "
            f"{row.parameter_abs_error:.6e} | {row.parameter_rel_error:.6e} | {row.objective_norm:.6e} | "
            f"{row.coefficient_abs_error:.6e} | {row.coefficient_rel_error:.6e} | "
            f"{row.state_fit_true_over_observed_error:.6e} | {row.jacobian_condition:.6e} | "
            f"{row.preprocess_elapsed_seconds:.6f} | {row.elapsed_seconds:.6f} | {row.stop_reason} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Darcy inverse convergence study using the clean VariablePoissonSolver path.")
    parser.add_argument("--family", type=str, default="affine", choices=FAMILIES)
    parser.add_argument("--dim", type=int, default=2, choices=(2, 3))
    parser.add_argument("--geometry", type=str, default=None)
    parser.add_argument("--orders", type=int, nargs="+", default=[2,4,6])
    parser.add_argument("--hs", type=float, nargs="+", default=[.45,.35,.25,0.15])
    parser.add_argument("--domain-files", type=Path, nargs="+", default=None)
    parser.add_argument("--stencil", type=str, default="rbf", choices=("wls", "rbf"))
    parser.add_argument(
        "--linear-solver",
        type=str,
        default="lineax",
        choices=("gmres_ilu", "scipy_gmres_ilu", "dense", "lineax", "cudss"),
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="optimistix_lm",
        choices=("adam", "optimistix_lm", "forward_fd_lm_eager"),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--noise-std", type=float, default=DEFAULT_NOISE_STD)
    parser.add_argument("--noise-seed", type=int, default=DEFAULT_NOISE_SEED)
    parser.add_argument("--noise-mode", type=str, default="relative", choices=NOISE_MODES)
    parser.add_argument(
        "--target-smoothing-radius-factor",
        type=float,
        default=0.0,
        help="Gaussian pre-smoothing radius for noisy target observations, as a multiple of h. Disabled when zero.",
    )
    parser.add_argument("--theta-reg", type=float, default=0.0, help="Tikhonov regularization weight on theta. Disabled when zero.")
    parser.add_argument("--theta-ref", type=float, nargs="+", default=None, help="Reference theta for --theta-reg. Defaults to zeros.")
    parser.add_argument("--residual-mode", type=str, default="absolute", choices=RESIDUAL_MODES)
    parser.add_argument(
        "--residual-floor",
        type=float,
        default=1.0e-6,
        help="Denominator floor for --residual-mode relative.",
    )
    parser.add_argument(
        "--identifiability-diagnostics",
        action="store_true",
        help="Compute state distinguishability and Jacobian singular-value diagnostics after recovery.",
    )
    parser.add_argument("--lm-jacobian", type=str, default="ad", choices=LM_JACOBIAN_MODES, help="Use full AD through the residual, or a finite-difference callback Jacobian.")
    parser.add_argument("--timing-warmup", action="store_true", help="Run an extra warmup/compile recovery pass before the measured pass.")
    parser.add_argument("--source-mode", choices=SOURCE_MODES, default="fixed")
    parser.add_argument(
        "--observation-order",
        type=int,
        default=6,
        help="Discretization order used to generate cached forward observations.",
    )
    parser.add_argument(
        "--observation-dir",
        type=Path,
        default=Path("Alberto_outputs") / "darcy_reference_observations",
    )
    parser.add_argument("--setup-device", choices=("default", "cpu"), default="default")
    parser.add_argument("--sparse-assembly-chunk-size", type=int, default=None)
    parser.add_argument("--mat-output", type=Path, default=Path(__file__).resolve().with_name("darcy_inverse_convergence_study-clover-cross-noisy-lineax.mat"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.linear_solver == "scipy_gmres_ilu" and args.optimizer != "forward_fd_lm_eager":
        raise ValueError(
            "--linear-solver scipy_gmres_ilu requires --optimizer forward_fd_lm_eager "
            "because SciPy solves are not differentiable JAX traces"
        )
    if args.source_mode == "zero":
        raise ValueError(
            "the overall permeability scale is unidentifiable with zero forcing and Dirichlet data; "
            "use --source-mode fixed"
        )
    results: list[StudyResult] = []

    if args.domain_files:
        domains = [load_domain_file(path) for path in args.domain_files]
    else:
        geometry_name = args.geometry or default_geometry(args.dim)
        validate_geometry(args.dim, geometry_name)
        hs = args.hs or [default_h(args.dim)]
        domains = []
        for h in hs:
            domain = build_domain(args.dim, geometry_name, h, args.seed)
            metadata = SavedDomainMetadata(
                dim=args.dim,
                geometry=geometry_name,
                h=float(h),
                seed=args.seed,
                sep_rad=float(domain.get_sep_rad()),
                label=f"{geometry_name}_h{h:.5f}",
            )
            domains.append((domain, metadata))

    for order in args.orders:
        for domain, metadata in domains:
            results.append(
                run_case(
                    domain=domain,
                    metadata=metadata,
                    family=args.family,
                    order=order,
                    stencil=args.stencil,
                    learning_rate=args.lr,
                    linear_solver=args.linear_solver,
                    optimizer_name=args.optimizer,
                    noise_std=args.noise_std,
                    noise_seed=args.noise_seed,
                    noise_mode=args.noise_mode,
                    target_smoothing_radius_factor=args.target_smoothing_radius_factor,
                    timing_warmup=args.timing_warmup,
                    lm_jacobian=args.lm_jacobian,
                    theta_reg=args.theta_reg,
                    theta_ref_values=args.theta_ref,
                    identifiability_diagnostics=args.identifiability_diagnostics,
                    residual_mode=args.residual_mode,
                    residual_floor=args.residual_floor,
                    source_mode=args.source_mode,
                    observation_order=args.observation_order,
                    observation_dir=args.observation_dir.resolve(),
                    sparse_assembly_chunk_size=args.sparse_assembly_chunk_size,
                    setup_device=args.setup_device,
                )
            )

    markdown = render_markdown(
        results,
        family=args.family,
        stencil=args.stencil,
        linear_solver=args.linear_solver,
        optimizer_name=args.optimizer,
        noise_std=args.noise_std,
        noise_seed=args.noise_seed,
        noise_mode=args.noise_mode,
        target_smoothing_radius_factor=args.target_smoothing_radius_factor,
        theta_reg=args.theta_reg,
        theta_ref_values=args.theta_ref,
        residual_mode=args.residual_mode,
        residual_floor=args.residual_floor,
        source_mode=args.source_mode,
        observation_order=args.observation_order,
    )
    export_inverse_mat(
        results,
        family=args.family,
        stencil=args.stencil,
        linear_solver=args.linear_solver,
        optimizer_name=args.optimizer,
        noise_std=args.noise_std,
        noise_seed=args.noise_seed,
        noise_mode=args.noise_mode,
        target_smoothing_radius_factor=args.target_smoothing_radius_factor,
        theta_reg=args.theta_reg,
        theta_ref_values=args.theta_ref,
        residual_mode=args.residual_mode,
        residual_floor=args.residual_floor,
        source_mode=args.source_mode,
        observation_order=args.observation_order,
        output_path=args.mat_output,
    )
    print(markdown)
    print(f"wrote_mat={args.mat_output}")


if __name__ == "__main__":
    main()


