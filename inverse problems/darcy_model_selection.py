from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
from jax import grad, hessian, vmap
import optax

from kernelpack.solvers import VariablePoissonSolver

from common import (
    StaticInverseCase,
    build_domain,
    compile_loss_and_grad,
    default_h,
    manufactured_state_1,
    manufactured_state_2,
    manufactured_state_3,
    manufactured_state_4,
    relative_l2,
    run_optax_jitted,
)


def model_features(model_name: str, dim: int, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Candidate feature dictionaries for the Darcy model-selection study."""

    zeros = jnp.zeros((x.shape[0],), dtype=float)
    ones = jnp.ones((x.shape[0],), dtype=float)
    if dim == 2:
        sin_x = jnp.sin(jnp.pi * x[:, 0])
        sin_y = jnp.sin(jnp.pi * x[:, 1])
        sin_xy = sin_x * sin_y
        table = {
            "constant": (
                jnp.column_stack([ones]),
                jnp.stack([jnp.column_stack([zeros, zeros])], axis=1),
            ),
            "affine": (
                jnp.column_stack([ones, x[:, 0], x[:, 1]]),
                jnp.stack(
                    [
                        jnp.column_stack([zeros, zeros]),
                        jnp.column_stack([ones, zeros]),
                        jnp.column_stack([zeros, ones]),
                    ],
                    axis=1,
                ),
            ),
            "quadratic_diag": (
                jnp.column_stack([ones, x[:, 0], x[:, 1], x[:, 0] ** 2, x[:, 1] ** 2]),
                jnp.stack(
                    [
                        jnp.column_stack([zeros, zeros]),
                        jnp.column_stack([ones, zeros]),
                        jnp.column_stack([zeros, ones]),
                        jnp.column_stack([2.0 * x[:, 0], zeros]),
                        jnp.column_stack([zeros, 2.0 * x[:, 1]]),
                    ],
                    axis=1,
                ),
            ),
            "cross_term": (
                jnp.column_stack([ones, x[:, 0], x[:, 1], x[:, 0] * x[:, 1]]),
                jnp.stack(
                    [
                        jnp.column_stack([zeros, zeros]),
                        jnp.column_stack([ones, zeros]),
                        jnp.column_stack([zeros, ones]),
                        jnp.column_stack([x[:, 1], x[:, 0]]),
                    ],
                    axis=1,
                ),
            ),
            "sin_product": (
                jnp.column_stack([ones, sin_xy]),
                jnp.stack(
                    [
                        jnp.column_stack([zeros, zeros]),
                        jnp.column_stack(
                            [
                                jnp.pi * jnp.cos(jnp.pi * x[:, 0]) * sin_y,
                                jnp.pi * sin_x * jnp.cos(jnp.pi * x[:, 1]),
                            ]
                        ),
                    ],
                    axis=1,
                ),
            ),
        }
    else:
        sin_x = jnp.sin(jnp.pi * x[:, 0])
        sin_y = jnp.sin(jnp.pi * x[:, 1])
        sin_z = jnp.sin(jnp.pi * x[:, 2])
        sin_xyz = sin_x * sin_y * sin_z
        table = {
            "constant": (
                jnp.column_stack([ones]),
                jnp.stack([jnp.column_stack([zeros, zeros, zeros])], axis=1),
            ),
            "affine": (
                jnp.column_stack([ones, x[:, 0], x[:, 1], x[:, 2]]),
                jnp.stack(
                    [
                        jnp.column_stack([zeros, zeros, zeros]),
                        jnp.column_stack([ones, zeros, zeros]),
                        jnp.column_stack([zeros, ones, zeros]),
                        jnp.column_stack([zeros, zeros, ones]),
                    ],
                    axis=1,
                ),
            ),
            "quadratic_diag": (
                jnp.column_stack([ones, x[:, 0], x[:, 1], x[:, 2], x[:, 0] ** 2, x[:, 2] ** 2]),
                jnp.stack(
                    [
                        jnp.column_stack([zeros, zeros, zeros]),
                        jnp.column_stack([ones, zeros, zeros]),
                        jnp.column_stack([zeros, ones, zeros]),
                        jnp.column_stack([zeros, zeros, ones]),
                        jnp.column_stack([2.0 * x[:, 0], zeros, zeros]),
                        jnp.column_stack([zeros, zeros, 2.0 * x[:, 2]]),
                    ],
                    axis=1,
                ),
            ),
            "cross_term": (
                jnp.column_stack([ones, x[:, 0], x[:, 1], x[:, 2], x[:, 0] * x[:, 2]]),
                jnp.stack(
                    [
                        jnp.column_stack([zeros, zeros, zeros]),
                        jnp.column_stack([ones, zeros, zeros]),
                        jnp.column_stack([zeros, ones, zeros]),
                        jnp.column_stack([zeros, zeros, ones]),
                        jnp.column_stack([x[:, 2], zeros, x[:, 0]]),
                    ],
                    axis=1,
                ),
            ),
            "sin_product": (
                jnp.column_stack([ones, sin_xyz]),
                jnp.stack(
                    [
                        jnp.column_stack([zeros, zeros, zeros]),
                        jnp.column_stack(
                            [
                                jnp.pi * jnp.cos(jnp.pi * x[:, 0]) * sin_y * sin_z,
                                jnp.pi * sin_x * jnp.cos(jnp.pi * x[:, 1]) * sin_z,
                                jnp.pi * sin_x * sin_y * jnp.cos(jnp.pi * x[:, 2]),
                            ]
                        ),
                    ],
                    axis=1,
                ),
            ),
        }
    if model_name not in table:
        raise ValueError(f"Unknown model family '{model_name}'")
    return table[model_name]


def coefficient_field(model_name: str, theta: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Positive permeability attached to one candidate model family."""

    feats, _ = model_features(model_name, x.shape[1], x)
    return jnp.exp(feats @ theta)


def coefficient_gradient(model_name: str, theta: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Spatial gradient of the candidate permeability field."""

    coeff = coefficient_field(model_name, theta, x)
    _, feat_grads = model_features(model_name, x.shape[1], x)
    log_grad = jnp.einsum("p,npd->nd", theta, feat_grads)
    return coeff[:, None] * log_grad


def manufactured_forcing(model_name: str, theta_true: jnp.ndarray, x: jnp.ndarray, state_fn) -> jnp.ndarray:
    """Exact Darcy forcing associated with one chosen model family."""

    scalar = lambda x_row: state_fn(x_row[None, :])[0]
    grad_u = vmap(grad(scalar))(x)
    hess_u = vmap(hessian(scalar))(x)
    lap_u = jnp.trace(hess_u, axis1=1, axis2=2)
    coeff = coefficient_field(model_name, theta_true, x)
    grad_coeff = coefficient_gradient(model_name, theta_true, x)
    return -(coeff * lap_u + jnp.sum(grad_coeff * grad_u, axis=1))


def average_case_loss(
    solver: VariablePoissonSolver,
    theta: jnp.ndarray,
    cases: list[StaticInverseCase],
    neu_coeff: jnp.ndarray,
    dir_coeff: jnp.ndarray,
    coeff_eval,
) -> jnp.ndarray:
    """Evaluate the mean relative loss over a batch of static Darcy cases."""

    losses = []
    for case in cases:
        out = solver.solve(
            case.forcing,
            lambda xall, th=theta: coeff_eval(th),
            neu_coeff,
            dir_coeff,
            case.boundary_target,
        )
        losses.append(relative_l2(out["u"], case.target) ** 2)
    return jnp.mean(jnp.asarray(losses))


def fit_model_family(
    solver: VariablePoissonSolver,
    theta0: jnp.ndarray,
    train_cases: list[StaticInverseCase],
    validation_cases: list[StaticInverseCase],
    all_features: jnp.ndarray,
    neu_coeff: jnp.ndarray,
    dir_coeff: jnp.ndarray,
    opt_steps: int,
):
    """Fit one candidate family and score it on both train and validation data."""

    def train_loss_fn(theta: jnp.ndarray) -> jnp.ndarray:
        return average_case_loss(solver, theta, train_cases, neu_coeff, dir_coeff, lambda th: jnp.exp(all_features @ th))

    def validation_loss_fn(theta: jnp.ndarray) -> jnp.ndarray:
        return average_case_loss(solver, theta, validation_cases, neu_coeff, dir_coeff, lambda th: jnp.exp(all_features @ th))

    train_loss_jit, loss_and_grad = compile_loss_and_grad(train_loss_fn)
    validation_loss_jit = jax.jit(validation_loss_fn)
    # Give larger model families enough room to settle so the ranking reflects
    # model quality rather than under-training of the higher-dimensional fit.
    fit_steps = max(opt_steps, 8 * int(theta0.shape[0]))
    theta_star, history = run_optax_jitted(loss_and_grad, theta0, optimizer=optax.adam(0.06), steps=fit_steps)
    return theta_star, train_loss_jit(theta_star), validation_loss_jit(theta_star), history


def run_example(dim: int, geometry_name: str, h: float, opt_steps: int, seed: int) -> None:
    """Rank a shortlist of Darcy models by optimized data misfit."""

    domain = build_domain(dim, geometry_name, h, seed)
    x_phys = jnp.asarray(domain.get_int_bdry_nodes(), dtype=float)
    x_bdry = jnp.asarray(domain.get_bdry_nodes(), dtype=float)
    x_all = jnp.asarray(domain.get_all_nodes(), dtype=float)
    xi = 4 if dim == 2 else 3
    neu_coeff = jnp.zeros((domain.get_num_bdry_nodes(),), dtype=float)
    dir_coeff = jnp.ones((domain.get_num_bdry_nodes(),), dtype=float)
    true_model = "cross_term"
    true_theta = (
        jnp.array([0.08, 0.24, -0.16, 0.42], dtype=float)
        if dim == 2
        else jnp.array([0.07, 0.18, -0.12, 0.09, 0.36], dtype=float)
    )

    state_fns = [manufactured_state_1, manufactured_state_2, manufactured_state_3, manufactured_state_4]
    cases = [
        StaticInverseCase(
            forcing=manufactured_forcing(true_model, true_theta, x_phys, fn),
            target=fn(x_phys),
            boundary_target=fn(x_bdry),
        )
        for fn in state_fns
    ]
    train_cases = cases[:3]
    validation_cases = cases[3:]

    families = ["constant", "affine", "quadratic_diag", "cross_term", "sin_product"]
    summary = []
    for family in families:
        # Each family gets its own fresh solver and parameter dimension so the
        # comparison is "best fit in that family" rather than "best fit under a
        # shared super-model."
        theta0 = jnp.zeros((model_features(family, dim, x_phys)[0].shape[1],), dtype=float)
        all_features = model_features(family, dim, x_all)[0]
        solver = VariablePoissonSolver(
            lap_assembler="fd",
            bc_assembler="fd",
            lap_stencil="wls",
            bc_stencil="wls",
            linear_solver="gmres_ilu",
        )
        solver.init(domain, xi)
        solver.prepare_preconditioner(
            lambda xall, feats=all_features, th=theta0: jnp.exp(feats @ th),
            neu_coeff,
            dir_coeff,
        )
        theta_star, train_loss, validation_loss, history = fit_model_family(
            solver,
            theta0,
            train_cases,
            validation_cases,
            all_features,
            neu_coeff,
            dir_coeff,
            opt_steps,
        )
        summary.append((family, float(validation_loss), float(train_loss), theta_star, history))

    summary.sort(key=lambda item: item[1])
    print(f"darcy model selection dim={dim} geometry={geometry_name}")
    print(f"true_model={true_model}")
    print("ranking_metric=validation_loss")
    for family, validation_loss, train_loss, theta_star, history in summary:
        print(f"family={family:14s} val_loss={validation_loss:.6e} train_loss={train_loss:.6e} params={theta_star}")
        print(f"  loss_history_first_last=({history[0]:.6e}, {history[-1]:.6e})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit several Darcy model families and rank them by inverse misfit.")
    parser.add_argument("--dim", type=int, default=2, choices=(2, 3))
    parser.add_argument("--geometry", type=str, default=None)
    parser.add_argument("--h", type=float, default=None)
    parser.add_argument("--opt-steps", type=int, default=35)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_example(
        dim=args.dim,
        geometry_name=args.geometry or ("clover" if args.dim == 2 else "tilted_egg"),
        h=args.h or default_h(args.dim),
        opt_steps=args.opt_steps,
        seed=args.seed,
    )
