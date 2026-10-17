"""Fixed reference velocity fields for the advection-diffusion forward and transport inverse studies.

A convergence study must solve the same problem at every resolution, so the velocity cannot depend on
the cloud. We build every velocity from one reference level set phi0 per geometry, the implicit layer
of the coarsest (h = .16, seed 11) geometry, whose CFPU balls cover the whole domain, so phi0 = f_loc
is smooth everywhere in the domain. For a mode m and a direction (2D: the scalar streamfunction; 3D: the
vector potential m e_axis), the field is

    w = curl(phi0_+^2 m),

which is divergence-free and vanishes on the boundary. The center c is the center of the bounding box of
the reference boundary nodes, and each field is divided by a fixed scale: its maximum speed over X0, the
points of a uniform grid (spacing NORMALIZATION_SPACING) inside the exact domain, which does not depend
on any cloud. All of these are fixed per geometry.

The point set X0, the reference level set and every normalization scale are computed once per geometry
(and family) and cached; per cloud, only the fields at the cloud's nodes are evaluated, on the device.
X0 itself is built once on the host (meshgrid + the exact inside test of cfpu_classification_study).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from kernelpack.geometry import CFPUOptions
from cfpu_refinement_study import initial_geometry, resample
from advection_velocity_inverse_convergence_study import _mode_and_gradient

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_H = .16
NORMALIZATION_SPACING = {2: .005, 3: .015}


def normalization_points(geometry: str, dim: int):
    """Uniform grid points inside the exact domain (cloud independent)."""
    from cfpu_classification_study import inside
    from treasure_chest_geometry import L, D, A, B
    lo, hi = (np.array([0., -.06]), np.array([2., 1.06])) if dim == 2 else (np.array([0., -A, -B]), np.array([L + D, A, B]))
    step = NORMALIZATION_SPACING[dim]
    axes = [np.arange(a, b + step / 2, step) for a, b in zip(lo, hi)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, dim)
    return grid[inside(grid, geometry)]


@lru_cache(maxsize=None)
def reference(geometry: str, fit_resolution: int = 41, cfpu_max_sites: int = 512):
    """phi0 (level-set object), center c, and the normalization point set X0 for a geometry."""
    base = initial_geometry(geometry, REFERENCE_H, fit_resolution=fit_resolution)[0]
    surface = resample(base, geometry, REFERENCE_H, seed=11, fit_resolution=fit_resolution)
    surface.build_level_set(options=CFPUOptions(feature_grade_factor=2, max_sites=cfpu_max_sites))
    level_set = surface.get_level_set()
    model = level_set.cfpu_model
    if model.band is not None and model.band["components"] != 0:
        raise ValueError(f"{geometry}: reference CFPU cover does not span the domain; phi0 would not be smooth")
    xb = np.asarray(surface.get_bdry_nodes())
    center = jnp.asarray(.5 * (xb.min(axis=0) + xb.max(axis=0)))
    x0 = jnp.asarray(normalization_points(geometry, xb.shape[1]))
    return level_set, center, x0


def _curl_fields(level_set, center, x, gradient_of_modes):
    """curl(phi0_+^2 m_i) for the modes given as a list of (m, grad m) at x."""
    phi = jnp.maximum(jnp.asarray(level_set.evaluate(x)), 0.)
    grad_phi = jnp.where(phi[:, None] > 0., jnp.asarray(level_set.evaluate_gradient(x)), 0.)
    dim = x.shape[1]; zero = jnp.zeros_like(phi); out = []
    # 2D: (d_y psi, -d_x psi) with psi = phi0_+^2 m.  3D: curl(a e_axis) with a = phi0_+^2 m.
    for i, (m, grad_m) in enumerate(gradient_of_modes):
        g = 2. * phi[:, None] * grad_phi * m[:, None] + (phi ** 2)[:, None] * grad_m      # grad(phi0_+^2 m)
        if dim == 2:
            out.append(jnp.column_stack((g[:, 1], -g[:, 0])))
        else:
            axis = i % 3                                                                 # curl(phi0_+^2 m e_axis)
            out.append([jnp.column_stack((zero, g[:, 2], -g[:, 1])), jnp.column_stack((-g[:, 2], zero, g[:, 0])),
                        jnp.column_stack((g[:, 1], -g[:, 0], zero))][axis])
    return out


def _forward_mode(x, center):
    s = x - center
    if x.shape[1] == 2:
        return .5 * (s[:, 0] ** 2 + s[:, 1] ** 2), s
    return .5 * (s[:, 0] ** 2 + s[:, 2] ** 2), jnp.column_stack((s[:, 0], jnp.zeros_like(s[:, 0]), s[:, 2]))


def _forward_field(geometry: str, points):
    level_set, center, _ = reference(geometry)
    m, gm = _forward_mode(points, center)
    f = _curl_fields(level_set, center, points, [(m, gm)] * 2)
    return f[1] if points.shape[1] == 3 else f[0]                                        # 3D: curl(psi e_y)


@lru_cache(maxsize=None)
def _forward_scale(geometry: str) -> float:
    """max_{X0} |w| of the forward base field (fixed per geometry, evaluated once)."""
    x0 = reference(geometry)[2]
    return float(jnp.max(jnp.linalg.norm(_forward_field(geometry, x0), axis=1)))


def forward_velocity(geometry: str, x):
    """Unit-scale forward AD field w/max_{X0}|w| at points x (the paper's base field, mode q, axis e_y in 3D)."""
    return _forward_field(geometry, jnp.asarray(x)) / _forward_scale(geometry)


def _transport_fields(geometry: str, family: str, points):
    level_set, center, _ = reference(geometry)
    modes = [_mode_and_gradient(family, i, points, center) for i in range(3)]
    return _curl_fields(level_set, center, points, modes)


@lru_cache(maxsize=None)
def _transport_scales(geometry: str, family: str) -> tuple:
    """max_{X0} |w_i| of the three transport basis fields (fixed per geometry and family, evaluated once)."""
    x0 = reference(geometry)[2]
    return tuple(float(jnp.max(jnp.linalg.norm(f, axis=1))) for f in _transport_fields(geometry, family, x0))


def transport_basis(geometry: str, family: str, x):
    """Three fixed unit-scale basis fields (3, n, d) at points x for the transport inverse."""
    scales = _transport_scales(geometry, family)
    return jnp.stack([f / s for f, s in zip(_transport_fields(geometry, family, jnp.asarray(x)), scales)])


@lru_cache(maxsize=None)
def _transport_theta_factor(geometry: str, family: str, theta_reference: tuple, speed: float) -> float:
    theta = jnp.asarray(theta_reference)
    v = jnp.einsum("i,ind->nd", theta, transport_basis(geometry, family, reference(geometry)[2]))
    return speed / float(jnp.max(jnp.linalg.norm(v, axis=1)))


def transport_theta(geometry: str, family: str, theta_reference, speed: float):
    """True coefficients: theta_reference scaled so that max_{X0} |sum_i theta_i w_i| = speed (fixed per geometry)."""
    key = tuple(float(t) for t in np.asarray(theta_reference).reshape(-1))
    return theta_reference * _transport_theta_factor(geometry, family, key, float(speed))
