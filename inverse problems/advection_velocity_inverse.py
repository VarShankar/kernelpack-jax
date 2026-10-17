from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import optax

from kernelpack import geometry
from kernelpack.geometry import PiecewiseSmoothEmbeddedSurface
from kernelpack.nodes import DomainNodeGenerator
from kernelpack.solvers import EulerianAdvectionDiffusionReactionSolver, StatelessEulerianAdvectionDiffusionReactionSolver

from common import (
    build_domain,
    fit_with_optax_jitted,
)


def h_channel_segments(n_sites: int = 40) -> tuple[list[jnp.ndarray], list[bool]]:
    """Piecewise H-channel boundary used by the repo examples."""

    width = 2.0
    height = 1.0
    throat_half_width = 0.1
    throat_half_height = 0.1
    throat_center_x = 0.5 * width
    throat_center_y = 0.5 * height
    segments: list[jnp.ndarray] = []
    flip_normal: list[bool] = []

    def add_horizontal(x0: float, x1: float, y: float, flip: bool) -> None:
        x = jnp.linspace(x0, x1, n_sites)
        segments.append(jnp.column_stack([x, jnp.full_like(x, y)]))
        flip_normal.append(flip)

    def add_vertical(x: float, y0: float, y1: float, flip: bool) -> None:
        y = jnp.linspace(y0, y1, n_sites)
        segments.append(jnp.column_stack([jnp.full_like(y, x), y]))
        flip_normal.append(flip)

    add_horizontal(0.0, width, 0.0, False)
    add_horizontal(0.0, width, height, True)
    add_vertical(0.0, 0.0, throat_center_y - throat_half_height, True)
    add_vertical(width, 0.0, throat_center_y - throat_half_height, False)
    add_vertical(0.0, throat_center_y + throat_half_height, height, True)
    add_vertical(width, throat_center_y + throat_half_height, height, False)
    add_vertical(throat_center_x - throat_half_width, throat_center_y - throat_half_height, throat_center_y + throat_half_height, True)
    add_vertical(throat_center_x + throat_half_width, throat_center_y - throat_half_height, throat_center_y + throat_half_height, False)
    add_horizontal(0.0, throat_center_x - throat_half_width, throat_center_y - throat_half_height, True)
    add_horizontal(0.0, throat_center_x - throat_half_width, throat_center_y + throat_half_height, False)
    add_horizontal(throat_center_x + throat_half_width, width, throat_center_y - throat_half_height, True)
    add_horizontal(throat_center_x + throat_half_width, width, throat_center_y + throat_half_height, False)
    return segments, flip_normal


def build_h_channel_domain(h: float, seed: int):
    surface = PiecewiseSmoothEmbeddedSurface()
    segments, flip_normal = h_channel_segments()
    surface.generate_piecewise_smooth_surface_by_segment(segments, flip_normal, 0.8 * h, method=1, supersample_fac=2)
    surface.build_level_set()
    generator = DomainNodeGenerator()
    return generator.build_domain_descriptor_from_geometry(
        surface,
        h,
        seed=seed,
        strip_count=4,
        do_outer_refinement=True,
        outer_fraction_of_h=0.5,
        outer_refinement_zone_size_as_multiple_of_h=1.5,
    )


def build_ice_cap_domain(h: float, seed: int):
    sphere_radius = 1.0
    cap_height = 0.3
    base_y = sphere_radius - cap_height
    base_radius = jnp.sqrt(sphere_radius**2 - base_y**2)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 120, endpoint=False)
    base_curve = jnp.column_stack([base_radius * jnp.cos(theta), base_radius * jnp.sin(theta)])
    cap = geometry.fibonacci_sphere(1600)
    cap = cap[cap[:, 1] >= base_y]
    base_ring = jnp.column_stack([base_radius * jnp.cos(theta), jnp.full_like(theta, base_y), base_radius * jnp.sin(theta)])
    cap = jnp.vstack([cap, base_ring])
    generator = DomainNodeGenerator()
    return generator.generate_piecewise_smooth_domain_nodes_hierarchical(
        [base_curve, cap],
        [True, False],
        [True, False],
        [float(base_y), -1.0],
        [1, -1],
        [True, False],
        h,
        method=2,
        supersample_fac=4,
        seed=seed,
        strip_count=4,
        do_outer_refinement=True,
        outer_fraction_of_h=0.6,
        outer_refinement_zone_size_as_multiple_of_h=1.5,
    )


def default_advection_geometry(dim: int) -> str:
    return "h-channel" if dim == 2 else "ice-cap"


def default_advection_h(dim: int, geometry_name: str) -> float:
    if geometry_name == "h-channel":
        return 0.10
    if geometry_name == "ice-cap":
        return 0.08
    return 0.18 if dim == 2 else 0.42


def build_advection_domain(dim: int, geometry_name: str, h: float, seed: int):
    if geometry_name == "h-channel":
        if dim != 2:
            raise ValueError("h-channel is a 2D geometry")
        return build_h_channel_domain(h, seed)
    if geometry_name == "ice-cap":
        if dim != 3:
            raise ValueError("ice-cap is a 3D geometry")
        return build_ice_cap_domain(h, seed)
    return build_domain(dim, geometry_name, h, seed)


def _level_set_values(domain, x: jnp.ndarray) -> jnp.ndarray:
    level_set = domain.get_outer_level_set()
    if level_set is None:
        raise ValueError("advection velocity inverse example requires an outer level set")
    return jnp.asarray(level_set.evaluate(x), dtype=float)


def _level_set_gradients(domain, x: jnp.ndarray) -> jnp.ndarray:
    level_set = domain.get_outer_level_set()
    if level_set is None:
        raise ValueError("advection velocity inverse example requires an outer level set")
    return jnp.asarray(level_set.evaluate_gradient(x), dtype=float)


def boundary_cutoff(domain, x: jnp.ndarray) -> jnp.ndarray:
    """Smoothly damp initial scalar data near the boundary."""

    h = float(domain.get_sep_rad())
    phi = jnp.maximum(_level_set_values(domain, x), 0.0)
    return (phi / (h + phi + 1.0e-14)) ** 2


def _normalize_vector_field(field: jnp.ndarray) -> jnp.ndarray:
    scale = jnp.maximum(jnp.max(jnp.linalg.norm(field, axis=1)), 1.0e-14)
    return field / scale


def velocity_basis(domain, x: jnp.ndarray) -> jnp.ndarray:
    """Low-dimensional velocity model used by the inverse problem.

    The basis is built from streamfunctions in 2D and vector potentials in 3D.
    This makes each field incompressible by construction. The potentials are
    multiplied by ``phi_+^2``, so their gradients vanish on the boundary and the
    resulting velocity is also zero there.
    """

    x = jnp.asarray(x, dtype=float)
    dim = x.shape[1]
    phi = jnp.maximum(_level_set_values(domain, x), 0.0)
    grad_phi = jnp.where(phi[:, None] > 0.0, _level_set_gradients(domain, x), 0.0)
    phi2 = phi**2
    grad_phi2 = 2.0 * phi[:, None] * grad_phi
    center = jnp.mean(jnp.asarray(domain.get_int_bdry_nodes(), dtype=float), axis=0)
    shifted = x - center

    def grad_weighted_potential(q: jnp.ndarray, grad_q: jnp.ndarray) -> jnp.ndarray:
        return grad_phi2 * q[:, None] + phi2[:, None] * grad_q

    if dim == 2:
        x0 = shifted[:, 0]
        y0 = shifted[:, 1]
        pi = jnp.pi
        potentials = [
            (
                0.5 * (x0**2 + y0**2),
                jnp.column_stack([x0, y0]),
            ),
            (
                jnp.sin(pi * x[:, 0]) * jnp.sin(pi * x[:, 1]),
                jnp.column_stack([pi * jnp.cos(pi * x[:, 0]) * jnp.sin(pi * x[:, 1]), pi * jnp.sin(pi * x[:, 0]) * jnp.cos(pi * x[:, 1])]),
            ),
            (
                x0 * y0 + 0.2 * x0**3,
                jnp.column_stack([y0 + 0.6 * x0**2, x0]),
            ),
        ]
        fields = []
        for q, grad_q in potentials:
            grad_psi = grad_weighted_potential(q, grad_q)
            fields.append(jnp.column_stack([grad_psi[:, 1], -grad_psi[:, 0]]))
    elif dim == 3:
        x0 = shifted[:, 0]
        y0 = shifted[:, 1]
        z0 = shifted[:, 2]
        potentials = [
            (
                0.5 * (x0**2 + y0**2),
                jnp.column_stack([x0, y0, jnp.zeros_like(x0)]),
                2,
            ),
            (
                x0 * z0,
                jnp.column_stack([z0, jnp.zeros_like(x0), x0]),
                1,
            ),
            (
                y0 * z0 + 0.15 * x0**2,
                jnp.column_stack([0.3 * x0, z0, y0]),
                0,
            ),
        ]
        fields = []
        for q, grad_q, axis in potentials:
            grad_a = grad_weighted_potential(q, grad_q)
            if axis == 0:
                fields.append(jnp.column_stack([jnp.zeros_like(x0), grad_a[:, 2], -grad_a[:, 1]]))
            elif axis == 1:
                fields.append(jnp.column_stack([-grad_a[:, 2], jnp.zeros_like(x0), grad_a[:, 0]]))
            else:
                fields.append(jnp.column_stack([grad_a[:, 1], -grad_a[:, 0], jnp.zeros_like(x0)]))
    else:
        raise ValueError("Only dim=2 or dim=3 is supported")
    return jnp.stack([_normalize_vector_field(field) for field in fields], axis=0)


def initial_scalar(domain, x: jnp.ndarray) -> jnp.ndarray:
    """Boundary-compatible passive scalar used for the synthetic observations."""

    x = jnp.asarray(x, dtype=float)
    cutoff = boundary_cutoff(domain, x)
    if x.shape[1] == 2:
        mode = jnp.sin(1.3 * x[:, 0] - 0.7 * x[:, 1]) + 0.35 * jnp.cos(0.8 * x[:, 0] + 1.1 * x[:, 1])
    else:
        mode = (
            jnp.sin(1.0 * x[:, 0] - 0.6 * x[:, 1] + 0.4 * x[:, 2])
            + 0.30 * jnp.cos(0.5 * x[:, 0] + 0.9 * x[:, 2])
        )
    return cutoff * mode


def run_example(dim: int, geometry_name: str, h: float, steps: int, learning_rate: float, opt_steps: int, seed: int) -> None:
    """Recover a low-dimensional Eulerian velocity field from scalar snapshots."""

    domain = build_advection_domain(dim, geometry_name, h, seed)
    dt = 0.01 if dim == 2 else 0.006
    xi = 1
    nu = 0.03 if dim == 2 else 0.02
    gamma_ref = -1.0e-4 * jnp.ones((dim,), dtype=float)
    theta_true = jnp.asarray([1.45, -0.90, 0.60], dtype=float) if dim == 2 else jnp.asarray([1.20, -0.80, 0.50], dtype=float)
    theta0 = jnp.zeros_like(theta_true)

    solver = EulerianAdvectionDiffusionReactionSolver(linear_solver="gmres_ilu")
    solver.init(domain, xi, dt, nu, hyperviscosity_power=2, gamma_components=gamma_ref)
    all_nodes = jnp.asarray(domain.get_all_nodes(), dtype=float)
    x_phys = jnp.asarray(domain.get_int_bdry_nodes(), dtype=float)
    c0_full = initial_scalar(domain, all_nodes)
    basis_phys = velocity_basis(domain, x_phys)

    neu_coeff = lambda xb: jnp.zeros((xb.shape[0],), dtype=float)
    dir_coeff = lambda xb: jnp.ones((xb.shape[0],), dtype=float)
    zero_dirichlet = lambda neu_coeffs, dir_coeffs, nr, time, xb: jnp.zeros((xb.shape[0],), dtype=float)
    zero_reaction = lambda time, state, x: jnp.zeros_like(state)
    solver.prepare_preconditioner(0.0, neu_coeff, dir_coeff, nu_ref=nu, gamma_components_ref=gamma_ref, bdf_order=3)
    stateless = StatelessEulerianAdvectionDiffusionReactionSolver.from_solver(solver)

    def velocity_from_theta(theta: jnp.ndarray):
        def velocity(time, x):
            del time, x
            return jnp.einsum("k,knd->nd", theta, basis_phys)

        return velocity

    def rollout_snapshots(theta: jnp.ndarray) -> jnp.ndarray:
        velocity = velocity_from_theta(theta)
        snapshots = []
        if steps <= 0:
            return jnp.zeros((0, solver.n), dtype=float)
        u0 = c0_full
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
            return jnp.stack(snapshots, axis=0)
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
            return jnp.stack(snapshots, axis=0)
        u_nm2, u_nm1, u_n = u0, u1, u2
        for k in range(2, steps):
            t_nm2 = (k - 2) * dt
            t_nm1 = (k - 1) * dt
            t_n = k * dt
            t_np1 = (k + 1) * dt
            u_np1 = stateless.bdf3_full_state(
                u_nm2,
                u_nm1,
                u_n,
                t_nm2,
                t_nm1,
                t_n,
                t_np1,
                velocity=velocity,
                reaction=zero_reaction,
                bc=zero_dirichlet,
                forcing=None,
                gamma_components=gamma_ref,
            )
            snapshots.append(u_np1[: solver.n])
            u_nm2, u_nm1, u_n = u_nm1, u_n, u_np1
        return jnp.stack(snapshots, axis=0)

    observed = jax.lax.stop_gradient(rollout_snapshots(theta_true))

    def loss_fn(theta: jnp.ndarray) -> jnp.ndarray:
        predicted = rollout_snapshots(theta)
        per_snapshot = jnp.sum((predicted - observed) ** 2, axis=1) / jnp.sum(observed**2, axis=1)
        return jnp.mean(per_snapshot)

    theta_star, history, baseline, fitted = fit_with_optax_jitted(
        loss_fn,
        theta0,
        optimizer=optax.adam(learning_rate),
        steps=opt_steps,
    )

    print(f"advection velocity inverse dim={dim} geometry={geometry_name}")
    print(f"nodes={domain.get_num_int_bdry_nodes()} ghost={domain.get_ghost_nodes().shape[0]} snapshots={steps}")
    print(f"theta_true={theta_true}")
    print(f"theta_init={theta0}")
    print(f"theta_fit={theta_star}")
    print(f"parameter_error={float(jnp.linalg.norm(theta_star - theta_true)):.6e}")
    print(f"baseline_loss={float(baseline):.6e} fitted_loss={float(fitted):.6e}")
    print(f"loss_history_first_last=({history[0]:.6e}, {history[-1]:.6e})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Learn a low-dimensional Eulerian velocity field from scalar snapshots.")
    parser.add_argument("--dim", type=int, default=2, choices=(2, 3))
    parser.add_argument("--geometry", type=str, default=None, help="Use h-channel in 2D or ice-cap in 3D by default.")
    parser.add_argument("--h", type=float, default=None)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--opt-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    geometry_name = args.geometry or default_advection_geometry(args.dim)
    run_example(
        dim=args.dim,
        geometry_name=geometry_name,
        h=args.h or default_advection_h(args.dim, geometry_name),
        steps=args.steps,
        learning_rate=args.lr,
        opt_steps=args.opt_steps,
        seed=args.seed,
    )
