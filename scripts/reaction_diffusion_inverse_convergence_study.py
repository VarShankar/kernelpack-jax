from __future__ import annotations

import argparse
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

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

INV = Path(__file__).resolve().parents[1] / "inverse problems"
if str(INV) not in sys.path:
    sys.path.insert(0, str(INV))

from kernelpack.domain import DomainDescriptor
from kernelpack.solvers import ReactionDiffusionSolver

from common import build_domain, compile_loss_and_grad, default_h, relative_l2, run_optax_jitted_summary
from reaction_diffusion_inverse import exact_state, exact_state_laplacian, exact_state_t, softplus_shifted

MAX_OPTIMIZATION_ITERATIONS = 60
DEFAULT_LEARNING_RATE = 0.05
RESIDUAL_TOLERANCE = 1.0e-6
DEFAULT_NOISE_STD = 0.0
DEFAULT_NOISE_SEED = 123
DEFAULT_OUTPUT = Path(__file__).with_name("reaction_diffusion_inverse_convergence_study.md")
DEFAULT_MAT_OUTPUT = Path(__file__).with_name("reaction_diffusion_inverse_convergence_study.mat")
TWO_D_GEOMETRIES = ("ellipse", "clover", "peanut")
THREE_D_GEOMETRIES = ("ellipsoid", "peanut", "tilted_egg")
FAMILIES = ("semilinear", "constant", "affine", "quadratic_diag", "cross_term", "sin_product")
OBSERVATION_MODES = ("final", "all")
LM_JACOBIAN_MODES = ("ad", "fd_callback")
FD_CALLBACK_EPS = 1.0e-5


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
    dt: float
    steps: int
    node_count: int
    iterations_used: int
    parameter_abs_error: float
    parameter_rel_error: float
    objective_norm: float
    reaction_abs_error: float
    reaction_rel_error: float
    elapsed_seconds: float
    preprocess_elapsed_seconds: float
    stop_reason: str


TRUE_THETA_TABLE = {
    ("semilinear", 2): jnp.array([-1.7, -3.6], dtype=float),
    ("semilinear", 3): jnp.array([-1.7, -3.6], dtype=float),
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


def default_geometry(dim: int) -> str:
    return "clover" if dim == 2 else "peanut"


def validate_geometry(dim: int, geometry_name: str) -> None:
    allowed = valid_geometries(dim)
    if geometry_name not in allowed:
        raise ValueError(f"Geometry '{geometry_name}' is not valid for dim={dim}. Choose one of {allowed}.")


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


def sample_theta(family: str, dim: int) -> jnp.ndarray:
    try:
        return TRUE_THETA_TABLE[(family, dim)]
    except KeyError as exc:
        raise ValueError(f"No true parameter template registered for family={family}, dim={dim}") from exc


def effective_theta(family: str, theta_raw: jnp.ndarray) -> jnp.ndarray:
    if family == "semilinear":
        return softplus_shifted(theta_raw, 1.0e-4)
    return theta_raw


def model_features(model_name: str, dim: int, x: jnp.ndarray) -> jnp.ndarray:
    ones = jnp.ones((x.shape[0],), dtype=float)
    if dim == 2:
        sin_product = jnp.sin(jnp.pi * x[:, 0]) * jnp.sin(jnp.pi * x[:, 1])
        table = {
            "constant": jnp.column_stack([ones]),
            "affine": jnp.column_stack([ones, x[:, 0], x[:, 1]]),
            "quadratic_diag": jnp.column_stack([ones, x[:, 0], x[:, 1], x[:, 0] ** 2, x[:, 1] ** 2]),
            "cross_term": jnp.column_stack([ones, x[:, 0], x[:, 1], x[:, 0] * x[:, 1]]),
            "sin_product": jnp.column_stack([ones, sin_product]),
        }
    else:
        sin_product = jnp.sin(jnp.pi * x[:, 0]) * jnp.sin(jnp.pi * x[:, 1]) * jnp.sin(jnp.pi * x[:, 2])
        table = {
            "constant": jnp.column_stack([ones]),
            "affine": jnp.column_stack([ones, x[:, 0], x[:, 1], x[:, 2]]),
            "quadratic_diag": jnp.column_stack([ones, x[:, 0], x[:, 1], x[:, 2], x[:, 0] ** 2, x[:, 2] ** 2]),
            "cross_term": jnp.column_stack([ones, x[:, 0], x[:, 1], x[:, 2], x[:, 0] * x[:, 2]]),
            "sin_product": jnp.column_stack([ones, sin_product]),
        }
    if model_name not in table:
        raise ValueError(f"Unknown reaction family '{model_name}'")
    return table[model_name]


def reaction_field(family: str, theta_raw: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    theta = effective_theta(family, theta_raw)
    return jnp.exp(model_features(family, x.shape[1], x) @ theta)


def reaction_family_value(family: str, theta_raw: jnp.ndarray, state: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    theta = effective_theta(family, theta_raw)
    if family == "semilinear":
        return theta[0] * state + theta[1] * state**3
    return jnp.exp(model_features(family, x.shape[1], x) @ theta) * state**2


def build_reaction_term(family: str, theta_raw: jnp.ndarray):
    return lambda time, state, x: reaction_family_value(family, theta_raw, state, x)


def make_forcing(family: str, nu: float, theta_true_raw: jnp.ndarray):
    def forcing(_nu_value: float, time: float, x: jnp.ndarray) -> jnp.ndarray:
        u = exact_state(time, x)
        return exact_state_t(time, x) - nu * exact_state_laplacian(time, x) - reaction_family_value(family, theta_true_raw, u, x)

    return forcing


def add_observation_noise(values: jnp.ndarray, noise_std: float, rng: np.random.Generator) -> jnp.ndarray:
    if noise_std <= 0.0:
        return values
    noise = rng.normal(loc=0.0, scale=noise_std, size=values.shape)
    return values + jnp.asarray(noise, dtype=values.dtype)


def solver_settings(dim: int) -> tuple[float, int, float]:
    dt = 0.02 if dim == 2 else 0.03
    order = 4 if dim == 2 else 3
    nu = 0.08 if dim == 2 else 0.06
    return dt, order, nu


def build_solver_template(
    domain: DomainDescriptor,
    *,
    order: int,
    dt: float,
    nu: float,
    stencil: str,
    linear_solver: str,
    sparse_assembly_chunk_size: int | None = None,
    gmres_tol: float = 1.0e-8,
    gmres_atol: float = 1.0e-10,
    gmres_restart: int = 40,
    gmres_maxiter: int | None = None,
) -> ReactionDiffusionSolver:
    solver = ReactionDiffusionSolver(
        lap_assembler="fd",
        bc_assembler="fd",
        lap_stencil=stencil,
        bc_stencil=stencil,
        linear_solver=linear_solver,
        gmres_tol=gmres_tol,
        gmres_atol=gmres_atol,
        gmres_restart=gmres_restart,
        gmres_maxiter=gmres_maxiter,
    )
    solver.init(
        domain,
        order,
        dt,
        nu,
        sparse_assembly_chunk_size=sparse_assembly_chunk_size,
        assemble_dense_operators=linear_solver not in ("gmres", "gmres_ilu", "cudss"),
    )
    return solver


def rollout_states(
    solver_template: ReactionDiffusionSolver,
    *,
    family: str,
    theta_raw: jnp.ndarray,
    initial_state: jnp.ndarray,
    forcing,
    exact_states: list[jnp.ndarray],
    dt: float,
    keep_all: bool = True,
):
    solver = solver_template.spawn_rollout_copy()
    solver.set_initial_state(initial_state)
    reaction_term = build_reaction_term(family, theta_raw)
    neu_coeff = lambda xb: jnp.zeros((xb.shape[0],), dtype=float)
    dir_coeff = lambda xb: jnp.ones((xb.shape[0],), dtype=float)
    bc = lambda neu_coeffs, dir_coeffs, nr, time, xb: exact_state(time, xb)

    states = []
    last_state = None
    if exact_states:
        last_state = solver.bdf1_step(dt, reaction_term, neu_coeff, dir_coeff, bc, forcing)
        if keep_all:
            states.append(last_state)
    if len(exact_states) >= 2:
        last_state = solver.bdf2_step(2.0 * dt, reaction_term, neu_coeff, dir_coeff, bc, forcing)
        if keep_all:
            states.append(last_state)
    for k in range(2, len(exact_states)):
        last_state = solver.bdf3_step((k + 1) * dt, reaction_term, neu_coeff, dir_coeff, bc, forcing)
        if keep_all:
            states.append(last_state)
    if not keep_all:
        if last_state is None:
            return []
        return [last_state]
    return states


def build_loss_fn(
    solver_template: ReactionDiffusionSolver,
    *,
    family: str,
    initial_state: jnp.ndarray,
    forcing,
    exact_states: list[jnp.ndarray],
    dt: float,
    observation_mode: str,
):
    def loss_fn(theta_raw: jnp.ndarray) -> jnp.ndarray:
        states = rollout_states(
            solver_template,
            family=family,
            theta_raw=theta_raw,
            initial_state=initial_state,
            forcing=forcing,
            exact_states=exact_states,
            dt=dt,
            keep_all=observation_mode == "all",
        )
        if observation_mode == "final":
            return relative_l2(states[-1], exact_states[-1]) ** 2
        losses = [relative_l2(state, exact_state_k) ** 2 for state, exact_state_k in zip(states, exact_states, strict=False)]
        return jnp.mean(jnp.asarray(losses))

    return loss_fn


def build_residual_fn(
    solver_template: ReactionDiffusionSolver,
    *,
    family: str,
    initial_state: jnp.ndarray,
    forcing,
    exact_states: list[jnp.ndarray],
    dt: float,
    observation_mode: str,
):
    def residual_fn(theta_raw: jnp.ndarray, _args):
        states = rollout_states(
            solver_template,
            family=family,
            theta_raw=theta_raw,
            initial_state=initial_state,
            forcing=forcing,
            exact_states=exact_states,
            dt=dt,
            keep_all=observation_mode == "all",
        )
        if observation_mode == "final":
            return (states[-1] - exact_states[-1]).reshape(-1), None
        blocks = [(state - exact_state_k).reshape(-1) for state, exact_state_k in zip(states, exact_states, strict=False)]
        return jnp.concatenate(blocks, axis=0), None

    return residual_fn


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


def run_optimistix_recovery(
    residual_fn,
    theta0: jnp.ndarray,
    *,
    jac_mode: str,
    diagnostics: bool = False,
) -> tuple[jnp.ndarray, float, int, str]:
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
        if diagnostics:
            print(f"[lm] iteration {iterations_used + 1}: starting step", flush=True)
        step_start_time = perf_counter()
        y, state, aux = step_fn(y=y, state=state)
        if diagnostics:
            print(f"[lm] iteration {iterations_used + 1}: step finished in {perf_counter() - step_start_time:.3f}s", flush=True)
            print(f"[lm] iteration {iterations_used + 1}: theta={np.asarray(y)}", flush=True)
        iterations_used += 1
        residual_start_time = perf_counter()
        residual, _ = residual_fn(y, None)
        objective_norm = float(jnp.linalg.norm(residual))
        if diagnostics:
            print(
                f"[lm] iteration {iterations_used}: residual norm={objective_norm:.6e}, "
                f"residual check elapsed={perf_counter() - residual_start_time:.3f}s",
                flush=True,
            )
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
    if diagnostics:
        print(
            f"[lm] finished after {iterations_used} iteration(s): "
            f"final norm={float(jnp.linalg.norm(final_residual)):.6e}, stop_reason={stop_reason}",
            flush=True,
        )
    return y_post, float(jnp.linalg.norm(final_residual)), iterations_used, stop_reason


def reaction_error_metrics(
    family: str,
    theta_true_raw: jnp.ndarray,
    theta_fit_raw: jnp.ndarray,
    x_phys: jnp.ndarray,
    exact_final: jnp.ndarray,
) -> tuple[float, float]:
    reaction_true = reaction_family_value(family, theta_true_raw, exact_final, x_phys)
    reaction_fit = reaction_family_value(family, theta_fit_raw, exact_final, x_phys)
    diff = reaction_fit - reaction_true
    abs_err = float(jnp.linalg.norm(diff))
    rel_err = float(abs_err / jnp.linalg.norm(reaction_true))
    return abs_err, rel_err


def run_case(
    *,
    domain: DomainDescriptor,
    metadata: SavedDomainMetadata,
    family: str,
    order: int,
    stencil: str,
    linear_solver: str,
    learning_rate: float,
    optimizer_name: str,
    steps: int,
    observation_mode: str,
    lm_jacobian: str,
    noise_std: float,
    noise_seed: int,
    timing_warmup: bool,
    diagnostics: bool,
) -> StudyResult:
    x_phys = jnp.asarray(domain.get_int_bdry_nodes(), dtype=float)
    dim = x_phys.shape[1]
    theta_true_raw = sample_theta(family, dim)
    theta0 = jnp.zeros_like(theta_true_raw)
    dt, _, nu = solver_settings(dim)

    forcing = make_forcing(family, nu, theta_true_raw)
    initial_state = exact_state(0.0, x_phys)
    exact_states = [exact_state((k + 1) * dt, x_phys) for k in range(steps)]
    observed_states = exact_states
    if noise_std > 0.0:
        rng = np.random.default_rng(noise_seed)
        observed_states = [add_observation_noise(state, noise_std, rng) for state in exact_states]
    if observation_mode == "final":
        residual_shape = (int(observed_states[-1].size),)
    else:
        residual_shape = (int(sum(state.size for state in observed_states)),)
    residual_dtype = observed_states[-1].dtype
    if diagnostics:
        print(
            "[case] "
            f"dim={dim}, geometry={metadata.geometry}, h={metadata.h:.5f}, order={order}, "
            f"nodes={domain.get_num_int_bdry_nodes()}, steps={steps}, family={family}, "
            f"linear_solver={linear_solver}, optimizer={optimizer_name}, observation_mode={observation_mode}, "
            f"lm_jacobian={lm_jacobian}, parameters={theta0.size}, residual_size={residual_shape[0]}",
            flush=True,
        )
    neu_coeff = lambda xb: jnp.zeros((xb.shape[0],), dtype=float)
    dir_coeff = lambda xb: jnp.ones((xb.shape[0],), dtype=float)

    def recover_once() -> tuple[jnp.ndarray, float, int, str]:
        solver_template = build_solver_template(domain, order=order, dt=dt, nu=nu, stencil=stencil, linear_solver=linear_solver)
        solver_template.prepare_boundary_operator(dt, neu_coeff, dir_coeff)
        if linear_solver == "gmres_ilu":
            solver_template.prepare_preconditioner(dt, neu_coeff, dir_coeff, nu_ref=nu)
        loss_fn = build_loss_fn(
            solver_template,
            family=family,
            initial_state=initial_state,
            forcing=forcing,
            exact_states=observed_states,
            dt=dt,
            observation_mode=observation_mode,
        )
        residual_fn = build_residual_fn(
            solver_template,
            family=family,
            initial_state=initial_state,
            forcing=forcing,
            exact_states=observed_states,
            dt=dt,
            observation_mode=observation_mode,
        )
        if optimizer_name == "adam":
            return run_optax_recovery(loss_fn, theta0, learning_rate)
        if optimizer_name == "optimistix_lm":
            jac_mode = "fwd"
            if lm_jacobian == "fd_callback":
                residual_fn = make_fd_callback_residual(
                    residual_fn,
                    theta0,
                    residual_shape=residual_shape,
                    residual_dtype=residual_dtype,
                    diagnostics=diagnostics,
                )
                jac_mode = "bwd"
            return run_optimistix_recovery(residual_fn, theta0, jac_mode=jac_mode, diagnostics=diagnostics)
        raise ValueError(f"Unknown optimizer '{optimizer_name}'")

    if timing_warmup:
        preprocess_start_time = perf_counter()
        recover_once()
        preprocess_elapsed_seconds = perf_counter() - preprocess_start_time
    else:
        preprocess_elapsed_seconds = 0.0

    start_time = perf_counter()
    theta_fit_raw, objective_norm, iterations_used, stop_reason = recover_once()
    elapsed_seconds = perf_counter() - start_time

    theta_true = effective_theta(family, theta_true_raw)
    theta_fit = effective_theta(family, theta_fit_raw)
    param_abs_error = float(jnp.linalg.norm(theta_fit - theta_true))
    param_rel_error = float(param_abs_error / jnp.linalg.norm(theta_true))
    exact_final = exact_states[-1]
    reaction_abs_error, reaction_rel_error = reaction_error_metrics(family, theta_true_raw, theta_fit_raw, x_phys, exact_final)

    return StudyResult(
        order=order,
        resolution=metadata.label,
        h=metadata.h,
        dt=dt,
        steps=steps,
        node_count=int(domain.get_num_int_bdry_nodes()),
        iterations_used=iterations_used,
        parameter_abs_error=param_abs_error,
        parameter_rel_error=param_rel_error,
        objective_norm=objective_norm,
        reaction_abs_error=reaction_abs_error,
        reaction_rel_error=reaction_rel_error,
        elapsed_seconds=elapsed_seconds,
        preprocess_elapsed_seconds=preprocess_elapsed_seconds,
        stop_reason=stop_reason,
    )


def export_inverse_mat(
    results: list[StudyResult],
    *,
    dim: int | None = None,
    family: str,
    stencil: str,
    linear_solver: str,
    optimizer_name: str,
    dt: float,
    steps: int,
    observation_mode: str,
    lm_jacobian: str,
    noise_std: float,
    noise_seed: int,
    output_path: Path,
    noise_mode: str = "absolute",
    target_smoothing_radius_factor: float = 0.0,
    theta_reg: float = 0.0,
    theta_ref_values: list[float] | None = None,
    residual_mode: str = "absolute",
    residual_floor: float = 1.0e-6,
    timestep_rule: str = "fixed steps",
    theta_bound: float | None = None,
    nonfinite_residual_penalty: float | None = None,
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
    reaction_abs_error = np.full(shape, np.nan, dtype=float)
    reaction_rel_error = np.full(shape, np.nan, dtype=float)
    elapsed_seconds = np.full(shape, np.nan, dtype=float)
    preprocess_elapsed_seconds = np.full(shape, np.nan, dtype=float)
    node_counts = np.full(shape, np.nan, dtype=float)
    dt_values = np.full(shape, np.nan, dtype=float)
    steps_values = np.full(shape, np.nan, dtype=float)

    for row in results:
        i = order_to_idx[row.order]
        j = label_to_idx[row.resolution]
        iterations_used[i, j] = row.iterations_used
        parameter_abs_error[i, j] = row.parameter_abs_error
        parameter_rel_error[i, j] = row.parameter_rel_error
        objective_norm[i, j] = row.objective_norm
        reaction_abs_error[i, j] = row.reaction_abs_error
        reaction_rel_error[i, j] = row.reaction_rel_error
        elapsed_seconds[i, j] = row.elapsed_seconds
        preprocess_elapsed_seconds[i, j] = row.preprocess_elapsed_seconds
        node_counts[i, j] = row.node_count
        dt_values[i, j] = row.dt
        steps_values[i, j] = row.steps

    data = {
        "orders": np.asarray(orders, dtype=float),
        "point_cloud_numbers": np.arange(1, len(labels) + 1, dtype=float),
        "labels": np.asarray(labels, dtype=object),
        "node_counts": node_counts,
        "dt_values": dt_values,
        "steps_values": steps_values,
        "iterations_used": iterations_used,
        "parameter_abs_error": parameter_abs_error,
        "parameter_rel_error": parameter_rel_error,
        "objective_norm": objective_norm,
        "reaction_abs_error": reaction_abs_error,
        "reaction_rel_error": reaction_rel_error,
        "elapsed_seconds": elapsed_seconds,
        "preprocess_elapsed_seconds": preprocess_elapsed_seconds,
        "family": family,
        "problem_type": "reaction-diffusion",
        "dimension": float("nan") if dim is None else int(dim),
        "stencil": stencil,
        "linear_solver": linear_solver,
        "optimizer": optimizer_name,
        "dt": float(dt),
        "steps": int(steps),
        "timestep_rule": timestep_rule,
        "observation_mode": observation_mode,
        "lm_jacobian": lm_jacobian,
        "noise_std": float(noise_std),
        "noise_seed": int(noise_seed),
        "noise_mode": noise_mode,
        "target_smoothing_radius_factor": float(target_smoothing_radius_factor),
        "theta_reg": float(theta_reg),
        "theta_ref": np.asarray(theta_ref_values if theta_ref_values is not None else [], dtype=float),
        "residual_mode": residual_mode,
        "residual_floor": float(residual_floor),
        "theta_bound": float("nan") if theta_bound is None else float(theta_bound),
        "nonfinite_residual_penalty": float("nan") if nonfinite_residual_penalty is None else float(nonfinite_residual_penalty),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(output_path, data)


def format_markdown(
    results: list[StudyResult],
    *,
    dim: int,
    geometry: str,
    family: str,
    stencil: str,
    linear_solver: str,
    optimizer_name: str,
    dt: float,
    steps: int,
    observation_mode: str,
    lm_jacobian: str,
    noise_std: float,
    noise_seed: int,
    noise_mode: str = "absolute",
    target_smoothing_radius_factor: float = 0.0,
    theta_reg: float = 0.0,
    theta_ref_values: list[float] | None = None,
    residual_mode: str = "absolute",
    residual_floor: float = 1.0e-6,
    timestep_rule: str = "fixed steps",
    theta_bound: float | None = None,
    nonfinite_residual_penalty: float | None = None,
) -> str:
    dt_text = "varies by row" if not np.isfinite(dt) else f"{dt}"
    steps_text = "varies by row" if steps < 0 else f"{steps}"
    lines = [
        f"# Reaction-diffusion inverse convergence study",
        "",
        f"- dimension: {dim}",
        f"- geometry: {geometry}",
        f"- family: {family}",
        f"- stencil: {stencil}",
        f"- linear solver: {linear_solver}",
        f"- optimizer: {optimizer_name}",
        f"- timestep rule: {timestep_rule}",
        f"- dt: {dt_text}",
        f"- steps: {steps_text}",
        f"- observation mode: {observation_mode}",
        f"- LM jacobian: {lm_jacobian}",
        f"- noise std: {noise_std}",
        f"- noise mode: {noise_mode}",
        f"- noise seed: {noise_seed}",
        f"- target smoothing radius factor: {target_smoothing_radius_factor}",
        f"- residual mode: {residual_mode}",
        f"- residual floor: {residual_floor}",
        f"- theta regularization: {theta_reg}",
        f"- theta reference: {theta_ref_values if theta_ref_values is not None else 'zeros'}",
        f"- theta bound: {theta_bound if theta_bound is not None else 'none'}",
        f"- nonfinite residual penalty: {nonfinite_residual_penalty if nonfinite_residual_penalty is not None else 'none'}",
        f"- timing: `preprocess seconds is 0 unless --timing-warmup is set; seconds is the recovery pass`",
        "",
        "| xi | resolution | h | dt | steps | physical nodes | iterations | parameter abs error | parameter rel error | objective norm | reaction abs error | reaction rel error | preprocess seconds | seconds | stop reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.order),
                    row.resolution,
                    f"{row.h:.5f}",
                    f"{row.dt:.6e}",
                    str(row.steps),
                    str(row.node_count),
                    str(row.iterations_used),
                    f"{row.parameter_abs_error:.6e}",
                    f"{row.parameter_rel_error:.6e}",
                    f"{row.objective_norm:.6e}",
                    f"{row.reaction_abs_error:.6e}",
                    f"{row.reaction_rel_error:.6e}",
                    f"{row.preprocess_elapsed_seconds:.6f}",
                    f"{row.elapsed_seconds:.6f}",
                    row.stop_reason,
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inverse convergence study for the stock ReactionDiffusionSolver.")
    parser.add_argument("--dim", type=int, default=2, choices=(2, 3))
    parser.add_argument("--geometry", type=str, default=None)
    parser.add_argument("--family", type=str, default="semilinear", choices=FAMILIES)
    parser.add_argument("--orders", type=int, nargs="+", default=[2,4,6])
    parser.add_argument("--hs", type=float, nargs="+", default=None)
    parser.add_argument("--domain-files", type=Path, nargs="+", default=None)
    parser.add_argument("--stencil", type=str, default="rbf", choices=("wls", "rbf"))
    parser.add_argument("--linear-solver", type=str, default="lineax", choices=("gmres_ilu", "dense", "lineax", "cudss"))
    parser.add_argument("--optimizer", type=str, default="optimistix_lm", choices=("adam", "optimistix_lm"))
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--observation-mode", type=str, default="final", choices=OBSERVATION_MODES)
    parser.add_argument(
        "--lm-jacobian",
        type=str,
        default="ad",
        choices=LM_JACOBIAN_MODES,
        help="Use full JAX AD through the rollout, or a finite-difference callback that treats the rollout as a black box.",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--noise-std", type=float, default=DEFAULT_NOISE_STD)
    parser.add_argument("--noise-seed", type=int, default=DEFAULT_NOISE_SEED)
    parser.add_argument("--timing-warmup", action="store_true", help="Run an extra warmup/compile recovery pass before the measured pass.")
    parser.add_argument("--diagnostics", action="store_true", help="Print LM step timing and fd-callback residual/Jacobian progress.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mat-output", type=Path, default=DEFAULT_MAT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    geometry_name = args.geometry or default_geometry(args.dim)

    results: list[StudyResult] = []
    if args.domain_files:
        domain_entries = [load_domain_file(path) for path in args.domain_files]
    else:
        validate_geometry(args.dim, geometry_name)
        hs = args.hs or [default_h(args.dim)]
        domain_entries = []
        for h in hs:
            domain = build_domain(args.dim, geometry_name, h, args.seed)
            metadata = SavedDomainMetadata(
                dim=args.dim,
                geometry=geometry_name,
                h=float(h),
                seed=args.seed,
                sep_rad=float(domain.get_sep_rad()),
                label=f"{geometry_name}_d{args.dim}_h{h:.5f}_seed{args.seed}",
            )
            domain_entries.append((domain, metadata))

    for domain, metadata in domain_entries:
        dim = metadata.dim
        orders = args.orders or ([4] if dim == 2 else [3])
        for order in orders:
            results.append(
                run_case(
                    domain=domain,
                    metadata=metadata,
                    family=args.family,
                    order=order,
                    stencil=args.stencil,
                    linear_solver=args.linear_solver,
                    learning_rate=args.lr,
                    optimizer_name=args.optimizer,
                    steps=args.steps,
                    observation_mode=args.observation_mode,
                    lm_jacobian=args.lm_jacobian,
                    noise_std=args.noise_std,
                    noise_seed=args.noise_seed,
                    timing_warmup=args.timing_warmup,
                    diagnostics=args.diagnostics,
                )
            )

    dt, _, _ = solver_settings(results and domain_entries[0][1].dim or args.dim)
    markdown = format_markdown(
        results,
        dim=domain_entries[0][1].dim if domain_entries else args.dim,
        geometry=domain_entries[0][1].geometry if domain_entries else geometry_name,
        family=args.family,
        stencil=args.stencil,
        linear_solver=args.linear_solver,
        optimizer_name=args.optimizer,
        dt=dt,
        steps=args.steps,
        observation_mode=args.observation_mode,
        lm_jacobian=args.lm_jacobian,
        noise_std=args.noise_std,
        noise_seed=args.noise_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    export_inverse_mat(
        results,
        dim=domain_entries[0][1].dim if domain_entries else args.dim,
        family=args.family,
        stencil=args.stencil,
        linear_solver=args.linear_solver,
        optimizer_name=args.optimizer,
        dt=dt,
        steps=args.steps,
        observation_mode=args.observation_mode,
        lm_jacobian=args.lm_jacobian,
        noise_std=args.noise_std,
        noise_seed=args.noise_seed,
        output_path=args.mat_output,
    )
    print(markdown, end="")
    print(f"wrote markdown to {args.output}")
    print(f"wrote matlab output to {args.mat_output}")


if __name__ == "__main__":
    main()






