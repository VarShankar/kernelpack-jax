"""Run source-free surface transport on the public IBAMR RBC trajectory."""

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from kernelpack import manifold, solvers


def initial_tracer(material_sites):
    center_one = jnp.asarray([0.25, 0.80, 0.54])
    center_two = jnp.asarray([-0.55, -0.20, 0.81])
    center_one /= jnp.linalg.norm(center_one)
    center_two /= jnp.linalg.norm(center_two)
    distance_one = jnp.arccos(jnp.clip(material_sites @ center_one, -1.0, 1.0))
    distance_two = jnp.arccos(jnp.clip(material_sites @ center_two, -1.0, 1.0))
    return 1.0 + jnp.exp(-(distance_one / 0.28) ** 2) + 0.65 * jnp.exp(-(distance_two / 0.22) ** 2)


def main(trajectory_directory=None, *, xi=6, diffusivity=2.0e-3, steps=None):
    root = Path(__file__).resolve().parents[1]
    folder = Path(trajectory_directory or root / "data" / "ibamr_rbc_3d" / "trajectory")
    trajectory = manifold.IBAMRSurfaceTrajectory(folder, xi=xi)
    frame_count = trajectory.times.size if steps is None else min(trajectory.times.size, int(steps) + 1)
    if frame_count < 2:
        raise ValueError("the RBC capstone requires at least two trajectory frames")
    geometry, _ = trajectory.geometry(0)
    neighbors = manifold.build_surface_stencil_graph(
        trajectory.material_sites,
        manifold.surface_stencil_size(manifold.surface_polynomial_degree(xi)),
    )
    operators, cache = manifold.assemble_tangent_plane_operators(
        geometry.points, geometry.normals, neighbors, xi=xi
    )
    concentration = initial_tracer(trajectory.material_sites)
    history = solvers.initialize_moving_surface_history(geometry.points, concentration)
    target_mass = geometry.quadrature_weights @ concentration
    h = jnp.sqrt(jnp.sum(geometry.quadrature_weights) / concentration.size)
    calibration = manifold.calibrate_surface_hyperviscosity(
        operators, geometry.points, geometry.normals, h, target_order=xi, arnoldi_iterations=12
    )
    reference_dt = float(trajectory.times[1] - trajectory.times[0])

    for frame_index in range(1, frame_count):
        dt = float(trajectory.times[frame_index] - trajectory.times[frame_index - 1])
        if not np.isclose(dt, reference_dt):
            raise ValueError("the RBC BDF driver requires uniformly spaced trajectory frames")
        geometry, _ = trajectory.geometry(frame_index)
        operators, cache, _ = manifold.update_tangent_plane_operators(
            geometry.points,
            geometry.normals,
            neighbors,
            cache,
            xi=xi,
            tolerance=1.0e-6,
        )
        h = jnp.sqrt(jnp.sum(geometry.quadrature_weights) / concentration.size)
        calibration, _, _ = manifold.refresh_surface_hyperviscosity(
            operators,
            geometry.points,
            geometry.normals,
            h,
            calibration,
            drift_threshold=0.15,
            arnoldi_iterations=12,
        )
        history, info = solvers.moving_surface_adr_step(
            history,
            geometry.points,
            operators,
            jnp.zeros(concentration.size),
            geometry.quadrature_weights,
            target_mass,
            reference_dt,
            diffusivity,
            calibration.gamma,
            min(0.1 * float(h) ** xi, 1.0e-7),
            order=min(frame_index, 3),
            hyperviscosity_power=calibration.power,
        )

    concentration = history.concentration[0]
    mass_error = jnp.abs(geometry.quadrature_weights @ concentration - target_mass) / jnp.abs(target_mass)
    output = root / "artifacts" / "moving_surface_rbc_capstone.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        points=np.asarray(geometry.points),
        concentration=np.asarray(concentration),
        time=trajectory.times[frame_count - 1],
    )
    print(f"frames={frame_count}, nodes={concentration.size}")
    print(f"relative mass error: {float(mass_error):.6e}")
    print(f"linear relative residual: {float(info.relative_residual):.6e}")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-directory")
    parser.add_argument("--xi", type=int, default=6)
    parser.add_argument("--diffusivity", type=float, default=2.0e-3)
    parser.add_argument("--steps", type=int)
    arguments = parser.parse_args()
    main(
        arguments.trajectory_directory,
        xi=arguments.xi,
        diffusivity=arguments.diffusivity,
        steps=arguments.steps,
    )
