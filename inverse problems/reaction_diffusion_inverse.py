from __future__ import annotations

import argparse

import jax.numpy as jnp
from jax import hessian, random, vmap
import optax

from kernelpack.solvers import ReactionDiffusionSolver

from common import (
    build_domain,
    default_geometry,
    default_h,
    fit_with_optax_jitted,
    manufactured_state_1,
    relative_l2,
    softplus_shifted,
)


def exact_state(time: float, x: jnp.ndarray) -> jnp.ndarray:
    """Manufactured transient used as the exact reaction-diffusion state."""

    return jnp.exp(-0.35 * time) * manufactured_state_1(x)


def exact_state_t(time: float, x: jnp.ndarray) -> jnp.ndarray:
    """Time derivative of the manufactured transient."""

    return -0.35 * exact_state(time, x)


def exact_state_laplacian(time: float, x: jnp.ndarray) -> jnp.ndarray:
    """Laplacian of the manufactured transient."""

    phi = lambda x_row: manufactured_state_1(x_row[None, :])[0]
    hess = vmap(hessian(phi))(x)
    lap_phi = jnp.trace(hess, axis1=1, axis2=2)
    return jnp.exp(-0.35 * time) * lap_phi


def reaction(theta_positive: jnp.ndarray, state: jnp.ndarray) -> jnp.ndarray:
    """Two-parameter semilinear reaction law used in the inverse example."""

    return theta_positive[0] * state + theta_positive[1] * state**3


def make_forcing(nu: float, theta_true_positive: jnp.ndarray):
    """Build the exact forcing that closes the manufactured PDE."""

    def forcing(_nu_value: float, time: float, x: jnp.ndarray) -> jnp.ndarray:
        u = exact_state(time, x)
        return exact_state_t(time, x) - nu * exact_state_laplacian(time, x) - reaction(theta_true_positive, u)

    return forcing

def run_example(dim: int, geometry_name: str, h: float, steps: int, learning_rate: float, opt_steps: int, seed: int) -> None:
    """Learn reaction parameters from a short transient trajectory."""

    domain = build_domain(dim, geometry_name, h, seed)
    dt = 0.02 if dim == 2 else 0.03
    xi = 4 if dim == 2 else 3
    nu = 0.08 if dim == 2 else 0.06
    theta_true_raw = jnp.array([-1.7, -3.6], dtype=float)
    key = random.key(seed)
    theta0 = 0.75 * random.normal(key, shape=(2,), dtype=jnp.float64)
    theta_true = softplus_shifted(theta_true_raw, 1.0e-4)

    solver_template = ReactionDiffusionSolver(
        lap_assembler="fd",
        bc_assembler="fd",
        lap_stencil="wls",
        bc_stencil="wls",
        linear_solver="gmres_ilu",
    )
    solver_template.init(domain, xi, dt, nu)
    x_out = jnp.asarray(domain.get_int_bdry_nodes(), dtype=float)
    initial_state = exact_state(0.0, x_out)
    forcing = make_forcing(nu, theta_true)
    neu_coeff = lambda xb: jnp.zeros((xb.shape[0],), dtype=float)
    dir_coeff = lambda xb: jnp.ones((xb.shape[0],), dtype=float)
    bc = lambda neu_coeffs, dir_coeffs, nr, time, xb: exact_state(time, xb)
    # Freeze the sparse preconditioner at the true diffusion coefficient. The
    # inverse variables only enter through the explicit reaction term in this
    # example, so the implicit matrix itself does not change during training.
    solver_template.prepare_preconditioner(dt, neu_coeff, dir_coeff, nu_ref=nu)
    exact_states = [exact_state((k + 1) * dt, x_out) for k in range(steps)]

    def loss_fn(theta_raw: jnp.ndarray) -> jnp.ndarray:
        # The optimization variable is unconstrained, but the actual reaction
        # coefficients are kept positive by the shifted softplus map.
        theta_positive = softplus_shifted(theta_raw, 1.0e-4)
        solver = solver_template.spawn_rollout_copy()
        solver.set_initial_state(initial_state)
        losses = []
        out = solver.bdf1_step(dt, lambda time, state, x: reaction(theta_positive, state), neu_coeff, dir_coeff, bc, forcing)
        losses.append(relative_l2(out, exact_states[0]) ** 2)
        if steps >= 2:
            out = solver.bdf2_step(2.0 * dt, lambda time, state, x: reaction(theta_positive, state), neu_coeff, dir_coeff, bc, forcing)
            losses.append(relative_l2(out, exact_states[1]) ** 2)
        for k in range(2, steps):
            out = solver.bdf3_step((k + 1) * dt, lambda time, state, x: reaction(theta_positive, state), neu_coeff, dir_coeff, bc, forcing)
            losses.append(relative_l2(out, exact_states[k]) ** 2)
        return jnp.mean(jnp.asarray(losses))

    theta_star, history, baseline, fitted = fit_with_optax_jitted(
        loss_fn,
        theta0,
        optimizer=optax.adam(learning_rate),
        steps=opt_steps,
    )
    theta_init = softplus_shifted(theta0, 1.0e-4)
    theta_fit = softplus_shifted(theta_star, 1.0e-4)

    print(f"reaction-diffusion inverse dim={dim} geometry={geometry_name}")
    print(f"nodes={x_out.shape[0]} ghost={domain.get_ghost_nodes().shape[0]}")
    print(f"theta_true={theta_true}")
    print(f"theta_init={theta_init}")
    print(f"theta_fit={theta_fit}")
    print(f"baseline_loss={float(baseline):.6e} fitted_loss={float(fitted):.6e}")
    print(f"loss_history_first_last=({history[0]:.6e}, {history[-1]:.6e})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn a parametrized transient reaction term with ReactionDiffusionSolver.")
    parser.add_argument("--dim", type=int, default=2, choices=(2, 3))
    parser.add_argument("--geometry", type=str, default=None)
    parser.add_argument("--h", type=float, default=None)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--opt-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_example(
        dim=args.dim,
        geometry_name=args.geometry or ("clover" if args.dim == 2 else "peanut"),
        h=args.h or default_h(args.dim),
        steps=args.steps,
        learning_rate=args.lr,
        opt_steps=args.opt_steps,
        seed=args.seed,
    )
