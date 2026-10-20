"""Shared KernelPack node-generation adapter for the manufactured spatial study.

The exact F2/F3 domains and independent query sets are unchanged. Closed-piece
spherical PU fits provide the canonical geometric model and CFPU visibility
layer. Their boundary samples are projected radially onto the independently
known analytic boundary, so geometry approximation does not change the domain
on which the spatial convergence experiment is defined. Projection is fixed
before any reconstruction is evaluated; no field error enters node selection.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import math

import jax
import jax.numpy as jnp


GEOMETRY_POLICY = dict(
    coordinate_model="shared closed-piece spherical-cap PU: local cubic PHS + affine polynomials",
    geometry_data_2d=512, geometry_data_3d=4096,
    geometry_validation_2d=1024, geometry_validation_3d=8192,
    relative_position_tolerance=1e-4, normal_tolerance=.01,
    boundary_projection="64 fixed ray bisections on exact F=0, followed by shared physical MIS thinning",
    boundary_supersample_factor=2, boundary_radius_factor=1.,
    interior_policy="shared Poisson greedy independent sets, four refill passes, oversample 6",
    interior_exact_guard="F > 0.42*h*norm(grad F)",
    fixed_boundary_clearance="h; max(radius_i,radius_j) conflict rule",
    stencil_policy="shared CFPU visibility and one-bend boundary-aware kNN at assigned physical nodes",
    periodic_policy="shared Poisson MIS with minimum-image conflicts on [0,2*pi)^d",
)


class ScatteredCloudFailure(RuntimeError):
    """A rejected cloud stage, retaining its actual device inputs and outputs."""
    def __init__(self, error, audit):
        super().__init__(f"{audit['stage']}: {type(error).__name__}: {error}")
        self.evidence = dict(audit["evidence"])
        self.metadata = dict(stage=audit["stage"], parameters=audit.get("parameters", {}),
            original_error_type=type(error).__name__, original_error=str(error))
        if hasattr(error, "history"):
            self.metadata["geometry_fit_history"] = error.history
        if hasattr(error, "parent_patches"):
            self.metadata["cfpu_parent_patches"] = error.parent_patches
        if hasattr(error,"evidence"):
            self.evidence.update({key:value for key,value in error.evidence.items()
                if isinstance(value,jax.Array)})
        if hasattr(error,"metadata"):
            self.metadata["inner_failure"] = error.metadata
        surface = audit.get("surface")
        if surface is not None:
            for index, segment in enumerate(surface.segments):
                model = segment.geom_model.get("model")
                if model is not None:
                    self.metadata[f"coordinate_model_{index}_diagnostics"] = getattr(model,"diagnostics",{})
                    for name,value in vars(model).items():
                        if isinstance(value,jax.Array):
                            self.evidence[f"coordinate_model_{index}_{name}"] = value
            cfpu = getattr(surface.level_set, "cfpu_model", None)
            if cfpu is not None:
                self.metadata["cfpu_diagnostics"] = cfpu.diagnostics
                for name,value in vars(cfpu).items():
                    if isinstance(value,jax.Array):
                        self.evidence["cfpu_"+name] = value


def _stage(audit, name, **arrays):
    audit["stage"] = name
    audit["evidence"].update({key:value for key,value in arrays.items() if isinstance(value,jax.Array)})


def _cloud_call(function, audit, *args, **kwargs):
    try:
        return function(*args, audit=audit, **kwargs)
    except ScatteredCloudFailure:
        raise
    except BaseException as error:
        raise ScatteredCloudFailure(error,audit) from error


def _directions(points):
    return points / jnp.linalg.norm(points, axis=1, keepdims=True)


def _normals(level_set, points):
    gradient = jax.jit(jax.vmap(jax.grad(lambda p: level_set(p))))(points)
    return -gradient / jnp.linalg.norm(gradient, axis=1, keepdims=True)


def _project_rays(level_set, points):
    directions = _directions(points)
    low = jnp.zeros(len(points), dtype=jnp.float64)
    high = jnp.full(len(points), 1.5, dtype=jnp.float64)

    def step(_, state):
        left, right = state
        middle = (left + right) / 2
        inside = level_set(middle[:, None] * directions) > 0
        return jnp.where(inside, middle, left), jnp.where(inside, right, middle)

    low, high = jax.lax.fori_loop(0, 64, step, (low, high))
    return ((low + high) / 2)[:, None] * directions


@lru_cache(maxsize=2)
def _coordinate_template(dimension, level_set, domain_boundary):
    audit = dict(stage="analytic_geometry_sample_generation",evidence={},parameters=dict(dimension=dimension))
    return _cloud_call(_coordinate_template_impl,audit,dimension,level_set,domain_boundary)


def _coordinate_template_impl(dimension, level_set, domain_boundary, *, audit):
    """Fit once per dimension from fixed geometry data, independently of h/p/kernel."""
    from kernelpack.geometry.core import EmbeddedSurface, PiecewiseSmoothEmbeddedSurface
    from kernelpack.geometry.spherical_pu import ClosedSurfaceData, SphericalPUOptions

    nfit, ncheck = ((512, 1024) if dimension == 2 else (4096, 8192))
    data = domain_boundary(dimension, nfit, phase=.1817283)
    check = domain_boundary(dimension, ncheck, phase=.7281828)
    _stage(audit,"coordinate_fit",geometry_data=data,geometry_validation_exact_positions=check)
    directions, check_directions = _directions(data), _directions(check)
    check_normals = _normals(level_set, check)
    _stage(audit,"coordinate_fit",geometry_data_directions=directions,
        geometry_validation_directions=check_directions,geometry_validation_exact_normals=check_normals)
    options = SphericalPUOptions(relative_position_tolerance=1e-4, normal_tolerance=.01)
    description = ClosedSurfaceData(positions=data, directions=directions,
        center=jnp.zeros(dimension, dtype=jnp.float64),
        validation_directions=check_directions, validation_positions=check,
        validation_normals=check_normals, options=options)
    segment = EmbeddedSurface()
    segment.build_spherical_pu_model(description, radius=None)
    model = segment.geom_model["model"]
    fitted = model.evaluate(check_directions)
    _, fitted_normals = model.frame(check_directions)
    evidence = dict(geometry_data=data, geometry_validation_directions=check_directions,
        geometry_validation_exact_positions=check, geometry_validation_fitted_positions=fitted,
        geometry_validation_exact_normals=check_normals, geometry_validation_fitted_normals=fitted_normals)
    surface = PiecewiseSmoothEmbeddedSurface()
    surface.set_segments([segment])
    return surface, evidence


def irregular_cloud(dimension, spacing, *, seed, level_set, domain_boundary):
    audit = dict(stage="coordinate_template",evidence={},parameters=dict(
        dimension=dimension,spacing=float(spacing),seed=int(seed)))
    return _cloud_call(_irregular_cloud_impl,audit,dimension,spacing,seed=seed,
        level_set=level_set,domain_boundary=domain_boundary)


def _irregular_cloud_impl(dimension, spacing, *, seed, level_set, domain_boundary, audit):
    from kernelpack.accelerators.jax_spatial import (
        poisson_disk_sample, compact_rows, POISSON_RNG_POLICY,
    )
    from kernelpack.geometry.core import _assembled_boundary_keep_mask
    from kernelpack.geometry.spherical_pu import sample_sphere_piece
    from kernelpack.geometry.cfpu import CFPUOptions
    from kernelpack.domain.core import DomainDescriptor

    if dimension not in (2, 3) or not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("Invalid manufactured scattered-cloud specification")
    template, fixed_evidence = _coordinate_template(dimension, level_set, domain_boundary)
    # Every h starts from the identical accepted coordinate fit; CFPU refinement
    # in one cloud cannot change the coordinate model of any later cloud.
    surface = copy.deepcopy(template)
    audit["surface"] = surface
    surface.seed = int(seed)
    segment = surface.segments[0]
    model = segment.geom_model["model"]
    _stage(audit,"shared_spherical_boundary_sampling",**fixed_evidence)
    directions = sample_sphere_piece(model, float(spacing), seed=int(seed), supersample_fac=2)
    _stage(audit,"coordinate_boundary_evaluation",boundary_directions=directions)
    segment.set_sample_sites(model.evaluate(directions))
    _, segment.nrmls = model.frame(directions)
    segment.uniform_sample_sites = segment.sample_sites
    segment.uniform_nrmls = segment.nrmls
    segment.sep_rad = float(spacing)
    segment.geom_model["fit_context"].update(radius=float(spacing), seed=int(seed), supersample_fac=2)
    surface._assemble_boundary(float(spacing), False, 0)
    _stage(audit,"cfpu_visibility_fit",fitted_boundary=surface.get_bdry_nodes(),
        fitted_boundary_normals=surface.get_bdry_nrmls())
    surface.build_level_set(options=CFPUOptions(max_sites=512, feature_grade_factor=1., farfield="band"))

    fitted_boundary = surface.get_bdry_nodes()
    _stage(audit,"exact_boundary_ray_projection",fitted_boundary=fitted_boundary)
    projected = _project_rays(level_set, fitted_boundary)
    _stage(audit,"projected_boundary_MIS_thinning",projected_boundary=projected)
    keep = _assembled_boundary_keep_mask(projected,
        jnp.zeros(len(projected), dtype=jnp.float64), float(spacing))
    boundary, corresponding_fit = compact_rows(keep, projected, fitted_boundary)
    exact_normals = _normals(level_set, boundary)
    _stage(audit,"interior_Poisson_MIS_sampling",boundary=boundary,exact_boundary_normals=exact_normals)

    @jax.jit
    def accept(points):
        gradient = jax.vmap(jax.grad(lambda p: level_set(p)))(points)
        value = level_set(points)
        return jnp.isfinite(value) & (value > .42 * spacing * jnp.linalg.norm(gradient, axis=1))

    interior = poisson_disk_sample(jnp.full(dimension, -1.2), jnp.full(dimension, 1.2),
        lambda p: jnp.full(p.shape[0], float(spacing), dtype=jnp.float64), float(spacing),
        seed=int(seed) + 104729, accept=accept, fixed_points=boundary,
        fixed_radii=float(spacing), oversample=6., passes=4, require_gpu=True)
    _stage(audit,"exact_domain_and_CFPU_visibility_validation",interior=interior)
    descriptor = DomainDescriptor()
    descriptor.set_nodes(interior, boundary)
    descriptor.set_normals(exact_normals)
    descriptor.set_sep_rad(float(spacing))
    descriptor.set_stencil_level_set(surface.level_set.cfpu_model, float(spacing))
    points = descriptor.get_tree_points("all")
    phi_nodes = descriptor.stencil_level_set(points)
    phi_boundary = phi_nodes[len(interior):]
    _stage(audit,"exact_domain_and_CFPU_visibility_validation",nodes=points,cfpu_node_values=phi_nodes)
    if not bool(jnp.all(jnp.isfinite(phi_nodes)) & jnp.all(phi_nodes >= -descriptor.stencil_level_set.delta)):
        raise ValueError("CFPU visibility layer excludes independently admitted exact-domain nodes")
    if not bool(jnp.all(level_set(interior) > 0) & jnp.all(jnp.isfinite(points))):
        raise ValueError("Shared sampler produced nonfinite points or points outside the exact domain")
    actual_model = surface.segments[0].geom_model["model"]
    actual_fitted = actual_model.evaluate(fixed_evidence["geometry_validation_directions"])
    _, actual_normals = actual_model.frame(fixed_evidence["geometry_validation_directions"])
    evidence = dict(fixed_evidence, geometry_validation_fitted_positions=actual_fitted,
        geometry_validation_fitted_normals=actual_normals,
        geometry_fitted_boundary=corresponding_fit,
        geometry_exact_projected_boundary=boundary, geometry_exact_boundary_normals=exact_normals,
        geometry_cfpu_boundary_values=phi_boundary,geometry_cfpu_node_values=phi_nodes)
    model_diagnostics = surface.segments[0].geom_model["model"].diagnostics
    metadata = dict(dimension=dimension, nominal_spacing=float(spacing), seed=int(seed),
        policy="kernelpack_closed_PU_boundary_and_GPU_Poisson_MIS_with_exact_radial_projection",
        rng_policy=POISSON_RNG_POLICY, interior_count=len(interior), boundary_count=len(boundary),
        fitted_boundary_count=len(fitted_boundary), geometry_policy=GEOMETRY_POLICY,
        coordinate_fit_diagnostics=model_diagnostics,
        cfpu_diagnostics=surface.level_set.cfpu_model.diagnostics,
        maximum_boundary_projection_distance=float(jnp.max(jnp.linalg.norm(boundary-corresponding_fit, axis=1))),
        maximum_geometry_validation_position_error=float(jnp.max(jnp.linalg.norm(
            actual_fitted-fixed_evidence["geometry_validation_exact_positions"], axis=1))),
        maximum_geometry_validation_normal_error=float(jnp.max(jnp.linalg.norm(
            actual_normals-fixed_evidence["geometry_validation_exact_normals"], axis=1))),
        maximum_exact_boundary_residual=float(jnp.max(jnp.abs(level_set(boundary)))),
        maximum_cfpu_exact_boundary_residual=float(jnp.max(jnp.abs(phi_boundary))),
        minimum_cfpu_node_value=float(jnp.min(phi_nodes)),
        boundary_visibility_tolerance=float(descriptor.stencil_level_set.delta))
    return points, boundary, descriptor, metadata, evidence


def periodic_cloud(dimension, n_per_axis, *, seed):
    audit = dict(stage="periodic_Poisson_MIS_sampling",evidence={},parameters=dict(
        dimension=dimension,n_per_axis=n_per_axis,seed=int(seed)))
    return _cloud_call(_periodic_cloud_impl,audit,dimension,n_per_axis,seed=seed)


def _periodic_cloud_impl(dimension, n_per_axis, *, seed, audit):
    from kernelpack.accelerators.jax_spatial import poisson_disk_sample, POISSON_RNG_POLICY
    if dimension not in (2, 3) or n_per_axis < 2:
        raise ValueError("Invalid periodic scattered-cloud specification")
    spacing = 2 * math.pi / n_per_axis
    _stage(audit,"periodic_Poisson_MIS_sampling",box_lower=jnp.zeros(dimension,dtype=jnp.float64),
        box_upper=jnp.full(dimension,2*math.pi,dtype=jnp.float64))
    points = poisson_disk_sample(jnp.zeros(dimension, dtype=jnp.float64),
        jnp.full(dimension, 2 * math.pi, dtype=jnp.float64),
        lambda p: jnp.full(p.shape[0], spacing, dtype=jnp.float64), spacing,
        seed=int(seed), periodic=True, oversample=6., passes=4, require_gpu=True)
    return points, dict(dimension=dimension, n_per_axis=n_per_axis, seed=int(seed),
        policy="kernelpack_GPU_Poisson_MIS_on_2pi_torus", rng_policy=POISSON_RNG_POLICY,
        nominal_spacing=spacing, geometry_policy=GEOMETRY_POLICY)
