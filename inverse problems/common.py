from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax
import optax

from kernelpack.geometry import EmbeddedSurface, fibonacci_sphere
from kernelpack.nodes import DomainNodeGenerator


@dataclass(frozen=True)
class ExampleConfig:
    """Lightweight record for a single inverse-example geometry setup."""

    dim: int
    geometry: str
    h: float
    seed: int


@dataclass(frozen=True)
class StaticInverseCase:
    """One static PDE observation bundle for elliptic inverse examples."""

    forcing: jnp.ndarray
    target: jnp.ndarray
    boundary_target: jnp.ndarray


def default_h(dim: int) -> float:
    """Choose a conservative spacing that keeps the examples reasonably fast."""

    return 0.18 if dim == 2 else 0.42


def default_geometry(dim: int) -> str:
    """Pick a smooth default geometry for each ambient dimension."""

    return "ellipse" if dim == 2 else "ellipsoid"


def build_domain(dim: int, geometry_name: str, h: float, seed: int):
    """Build a geometry-clipped domain with boundary refinement and ghost nodes.

    The inverse examples all use the same geometry pipeline so that the PDE
    solver comparisons differ mainly in the unknown parameterization, not in
    how the node set was generated.
    """

    if dim == 2:
        surface = _build_surface_2d(geometry_name)
        strip_count = 5
    elif dim == 3:
        surface = _build_surface_3d(geometry_name)
        strip_count = 4
    else:
        raise ValueError("Only dim=2 or dim=3 is supported")

    generator = DomainNodeGenerator()
    return generator.build_domain_descriptor_from_geometry(
        surface,
        h,
        seed=seed,
        strip_count=strip_count,
        do_outer_refinement=True,
        outer_fraction_of_h=0.5,
        outer_refinement_zone_size_as_multiple_of_h=2.0,
        mapped_cloud_preferred=(dim == 2),
    )


def _build_surface_2d(name: str) -> EmbeddedSurface:
    """Create a small family of smooth planar test domains.

    These are intentionally not all just circles/ellipses so that the inverse
    examples exercise the geometry machinery on mildly nontrivial boundaries.
    """

    t = jnp.linspace(0.0, 2.0 * jnp.pi, 180, endpoint=False)
    if name == "ellipse":
        curve = jnp.column_stack([1.15 * jnp.cos(t), 0.78 * jnp.sin(t)])
        rad = 0.06
    elif name == "clover":
        r = 0.92 + 0.18 * jnp.cos(3.0 * t)
        curve = jnp.column_stack([r * jnp.cos(t), 0.92 * r * jnp.sin(t)])
        rad = 0.055
    elif name == "peanut":
        r = 0.78 + 0.28 * jnp.sin(t) ** 2
        curve = jnp.column_stack([1.05 * r * jnp.cos(t), 0.9 * r * jnp.sin(t)])
        rad = 0.055
    else:
        raise ValueError(f"Unknown 2D geometry '{name}'")

    surface = EmbeddedSurface()
    surface.set_data_sites(curve)
    surface.build_closed_geometric_model_ps(2, rad, curve.shape[0])
    surface.build_level_set_from_geometric_model()
    return surface


def _build_surface_3d(name: str) -> EmbeddedSurface:
    """Create smooth closed surfaces for the 3D inverse examples."""

    points = fibonacci_sphere(420)
    if name == "ellipsoid":
        points = points * jnp.array([1.25, 0.82, 0.67], dtype=float)
        rad = 0.18
    elif name == "peanut":
        scale = 0.82 + 0.32 * points[:, 2] ** 2
        points = points * scale[:, None]
        points = points * jnp.array([1.0, 0.88, 1.12], dtype=float)
        rad = 0.18
    elif name == "tilted_egg":
        points = points * jnp.array([0.9, 0.78, 1.18], dtype=float)
        points = points.at[:, 0].add(0.18 * points[:, 2] ** 2)
        rad = 0.18
    else:
        raise ValueError(f"Unknown 3D geometry '{name}'")

    surface = EmbeddedSurface()
    surface.set_data_sites(points)
    surface.build_closed_geometric_model_ps(3, rad, points.shape[0])
    surface.build_level_set_from_geometric_model()
    return surface


def manufactured_state_1(x: jnp.ndarray) -> jnp.ndarray:
    """First smooth manufactured state used across several inverse examples."""

    dim = x.shape[1]
    if dim == 2:
        return (
            jnp.sin(1.1 * x[:, 0] - 0.8 * x[:, 1])
            + 0.2 * x[:, 0] ** 2
            + 0.1 * x[:, 0] * x[:, 1]
            - 0.05 * x[:, 1] ** 2
        )
    return (
        jnp.sin(0.9 * x[:, 0] - 0.7 * x[:, 1] + 0.5 * x[:, 2])
        + 0.12 * x[:, 0] ** 2
        - 0.08 * x[:, 1] * x[:, 2]
        + 0.07 * x[:, 2] ** 2
    )


def manufactured_state_2(x: jnp.ndarray) -> jnp.ndarray:
    """Second smooth state so the inverse problems are not single-case fits."""

    dim = x.shape[1]
    if dim == 2:
        return (
            jnp.cos(0.75 * x[:, 0] + 1.05 * x[:, 1])
            + 0.08 * x[:, 0] * x[:, 1]
            + 0.06 * x[:, 0] ** 2
        )
    return (
        jnp.cos(0.7 * x[:, 0] + 0.9 * x[:, 1] - 0.6 * x[:, 2])
        + 0.05 * x[:, 0] * x[:, 1]
        + 0.04 * x[:, 1] * x[:, 2]
        + 0.03 * x[:, 0] ** 2
    )


def manufactured_state_3(x: jnp.ndarray) -> jnp.ndarray:
    """Third state with stronger mixed-coordinate structure."""

    dim = x.shape[1]
    if dim == 2:
        xy = x[:, 0] * x[:, 1]
        return (
            jnp.sin(0.6 * x[:, 0]) * jnp.cos(1.15 * x[:, 1])
            + 0.22 * xy
            + 0.09 * xy**2
            - 0.04 * x[:, 0] ** 2
        )
    xz = x[:, 0] * x[:, 2]
    return (
        jnp.sin(0.55 * x[:, 0] - 0.4 * x[:, 1]) * jnp.cos(0.85 * x[:, 2])
        + 0.18 * xz
        + 0.06 * xz**2
        - 0.03 * x[:, 1] ** 2
    )


def manufactured_state_4(x: jnp.ndarray) -> jnp.ndarray:
    """Fourth state used as a held-out validation target."""

    dim = x.shape[1]
    if dim == 2:
        return (
            jnp.cos(0.9 * x[:, 0] - 0.45 * x[:, 1])
            + 0.18 * x[:, 0] * x[:, 1]
            + 0.07 * x[:, 0] ** 2 * x[:, 1]
            - 0.03 * x[:, 1] ** 3
        )
    return (
        jnp.cos(0.8 * x[:, 0] - 0.35 * x[:, 1] + 0.5 * x[:, 2])
        + 0.14 * x[:, 0] * x[:, 2]
        + 0.05 * x[:, 0] * x[:, 1] * x[:, 2]
        - 0.02 * x[:, 1] ** 3
    )


def transient_exact(time: float, x: jnp.ndarray) -> jnp.ndarray:
    """Small transient built from the first manufactured state."""

    return jnp.exp(-0.35 * time) * manufactured_state_1(x)


def run_optax_jitted(
    loss_and_grad,
    theta0: jnp.ndarray,
    *,
    optimizer: optax.GradientTransformation,
    steps: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run a jitted optimization loop using an Optax optimizer."""

    theta0 = jnp.asarray(theta0, dtype=float)
    opt_state0 = optimizer.init(theta0)
    loss_history0 = jnp.zeros((steps,), dtype=float)

    def body_fun(i: int, carry):
        theta, opt_state, loss_history = carry
        loss, grad_theta = loss_and_grad(theta)
        updates, opt_state = optimizer.update(grad_theta, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        loss_history = loss_history.at[i].set(loss)
        return theta, opt_state, loss_history

    theta, _, loss_history = lax.fori_loop(
        0,
        steps,
        body_fun,
        (theta0, opt_state0, loss_history0),
    )
    return theta, loss_history


def run_optax_jitted_summary(
    loss_and_grad,
    theta0: jnp.ndarray,
    *,
    optimizer: optax.GradientTransformation,
    steps: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run a jitted optimization loop and keep only first/last loss scalars."""

    theta0 = jnp.asarray(theta0, dtype=float)
    opt_state0 = optimizer.init(theta0)
    initial_loss = jnp.asarray(jnp.inf, dtype=theta0.dtype)
    final_loss = jnp.asarray(jnp.inf, dtype=theta0.dtype)

    def body_fun(i: int, carry):
        theta, opt_state, first_loss, last_loss = carry
        loss, grad_theta = loss_and_grad(theta)
        updates, opt_state = optimizer.update(grad_theta, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        first_loss = jnp.where(i == 0, loss, first_loss)
        return theta, opt_state, first_loss, loss

    theta, _, first_loss, final_loss = lax.fori_loop(
        0,
        steps,
        body_fun,
        (theta0, opt_state0, initial_loss, final_loss),
    )
    return theta, first_loss, final_loss


def compile_loss_and_grad(loss_fn):
    """Compile both the scalar loss and its gradient once for reuse."""

    loss_jit = jax.jit(loss_fn)
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    return loss_jit, loss_and_grad


def fit_with_optax_jitted(
    loss_fn,
    theta0: jnp.ndarray,
    *,
    optimizer: optax.GradientTransformation,
    steps: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compile, fit, and evaluate a small inverse problem consistently."""

    loss_jit, loss_and_grad = compile_loss_and_grad(loss_fn)
    theta_star, history = run_optax_jitted(loss_and_grad, theta0, optimizer=optimizer, steps=steps)
    baseline = loss_jit(theta0)
    fitted = loss_jit(theta_star)
    return theta_star, history, baseline, fitted


def softplus_shifted(theta: jnp.ndarray, floor: float) -> jnp.ndarray:
    """Map unconstrained parameters to a positive range with a small floor."""

    return floor + jax.nn.softplus(theta)


def relative_l2(u: jnp.ndarray, u_true: jnp.ndarray) -> jnp.ndarray:
    """Relative Euclidean error used in the paper-style example losses."""

    return jnp.linalg.norm(u - u_true) / jnp.linalg.norm(u_true)
