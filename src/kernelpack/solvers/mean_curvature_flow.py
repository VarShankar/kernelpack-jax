from __future__ import annotations

from functools import partial
from typing import NamedTuple

from jax import jit, vmap
from jax import scipy as jsp
import jax.numpy as jnp

from kernelpack.manifold import SurfaceOperators, apply_surface_operator


class MeanCurvatureFlowStepInfo(NamedTuple):
    linear_info: jnp.ndarray
    relative_residual: jnp.ndarray
    velocity: jnp.ndarray


@partial(
    jit,
    static_argnames=("mode", "project_normal", "gmres_restart", "gmres_max_iterations"),
)
def mean_curvature_flow_step(
    points: jnp.ndarray,
    normals: jnp.ndarray,
    operators: SurfaceOperators,
    dt: float,
    *,
    mode: str = "semiimplicit",
    project_normal: bool = True,
    linear_tolerance: float = 1.0e-8,
    gmres_restart: int = 20,
    gmres_max_iterations: int = 200,
) -> tuple[jnp.ndarray, MeanCurvatureFlowStepInfo]:
    """Advance ``X_t = Delta_Gamma X`` on the current point cloud."""

    points = jnp.asarray(points, dtype=float)
    normals = jnp.asarray(normals, dtype=points.dtype)

    if mode == "explicit":
        velocity = apply_surface_operator(
            operators.laplacian_weights,
            operators.neighbors,
            points,
        )
        if project_normal:
            velocity = jnp.sum(velocity * normals, axis=1, keepdims=True) * normals
        updated = points + dt * velocity
        return updated, MeanCurvatureFlowStepInfo(
            linear_info=jnp.asarray(0),
            relative_residual=jnp.asarray(0.0, dtype=points.dtype),
            velocity=velocity,
        )

    if mode != "semiimplicit":
        raise ValueError("mode must be 'explicit' or 'semiimplicit'")

    def system(values: jnp.ndarray) -> jnp.ndarray:
        return values - dt * apply_surface_operator(
            operators.laplacian_weights,
            operators.neighbors,
            values,
        )

    solved, linear_info = vmap(
        lambda rhs, guess: jsp.sparse.linalg.gmres(
            system,
            rhs,
            x0=guess,
            tol=linear_tolerance,
            atol=0.0,
            restart=gmres_restart,
            maxiter=gmres_max_iterations,
            solve_method="batched",
        ),
        in_axes=(1, 1),
        out_axes=(1, 0),
    )(points, points)
    raw_step = solved - points
    if project_normal:
        raw_step = jnp.sum(raw_step * normals, axis=1, keepdims=True) * normals
    updated = points + raw_step
    residual = system(solved) - points
    relative_residual = jnp.linalg.norm(residual) / jnp.maximum(
        jnp.linalg.norm(points),
        jnp.finfo(points.dtype).eps,
    )
    return updated, MeanCurvatureFlowStepInfo(
        linear_info=jnp.max(linear_info),
        relative_residual=relative_residual,
        velocity=raw_step / dt,
    )
