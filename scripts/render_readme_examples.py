from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from kernelpack import geometry
from kernelpack.geometry import EmbeddedSurface, PiecewiseSmoothEmbeddedSurface
from kernelpack.nodes import DomainNodeGenerator
from kernelpack.solvers import DiffusionSolver, PoissonSolver


ROOT = Path(__file__).resolve().parents[1]
ICE_CAP_HEIGHT = 0.5
OUTDIR = ROOT / "docs" / "readme_assets"
OUTDIR.mkdir(parents=True, exist_ok=True)


def build_surface(n_sites: int, geom_radius: float) -> EmbeddedSurface:
    t = jnp.linspace(0.0, 2.0 * jnp.pi, n_sites, endpoint=False)
    curve = jnp.column_stack([jnp.cos(t), 0.7 * jnp.sin(t)])
    surface = EmbeddedSurface()
    surface.set_data_sites(curve)
    surface.build_closed_geometric_model_ps(2, geom_radius, curve.shape[0])
    surface.build_level_set_from_geometric_model()
    return surface


def build_domain(*, do_outer_refinement: bool = True) -> tuple[EmbeddedSurface, object]:
    surface = build_surface(120, 0.06)
    generator = DomainNodeGenerator()
    domain = generator.build_domain_descriptor_from_geometry(
        surface,
        0.08,
        seed=17,
        strip_count=5,
        do_outer_refinement=do_outer_refinement,
        outer_fraction_of_h=0.5,
        outer_refinement_zone_size_as_multiple_of_h=2.0,
        mapped_cloud_preferred=False,
    )
    return surface, domain


def variable_density_radius(x, h_min):
    return h_min * (0.55 + 0.9 * (0.5 * (1.0 + jnp.tanh(2.8 * x[0]))))


def save_geometry(surface, plain_domain, refined_domain, variable_domain) -> Path:
    fig = plt.figure(figsize=(13, 8), constrained_layout=True)
    axes = fig.subplot_mosaic([["sites", "boundary", "field"], ["plain", "refined", "variable"]])
    data_sites = jnp.asarray(surface.data_sites)
    xb = jnp.asarray(refined_domain.get_bdry_nodes())
    nr = jnp.asarray(refined_domain.get_nrmls())

    axes["sites"].plot(data_sites[:, 0], data_sites[:, 1], "ko", ms=3)
    axes["sites"].set_title("Input Sites")

    axes["boundary"].plot(xb[:, 0], xb[:, 1], ".", color="#0f766e", ms=4)
    step = max(1, xb.shape[0] // 40)
    axes["boundary"].quiver(
        xb[::step, 0],
        xb[::step, 1],
        nr[::step, 0],
        nr[::step, 1],
        angles="xy",
        scale_units="xy",
        scale=18,
        color="#b91c1c",
        width=0.003,
    )
    axes["boundary"].set_title("Boundary Samples and Normals")

    xg, yg = jnp.meshgrid(jnp.linspace(-1.05, 1.05, 260), jnp.linspace(-0.8, 0.8, 220), indexing="xy")
    field = jnp.vectorize(lambda x, y: variable_density_radius(jnp.array([x, y]), 0.08), signature="(),()->()")(xg, yg)
    phi_grid = surface.get_level_set().evaluate(jnp.column_stack([xg.ravel(), yg.ravel()])).reshape(xg.shape)
    field = jnp.where(phi_grid >= 0.0, field, jnp.nan)
    pcm = axes["field"].pcolormesh(xg, yg, field, shading="auto", cmap="YlGnBu")
    axes["field"].plot(xb[:, 0], xb[:, 1], color="#0f766e", lw=1.2)
    axes["field"].set_title("Variable-Density Target Radius")
    cbar = fig.colorbar(pcm, ax=axes["field"], shrink=0.9)
    cbar.set_label("target h(x)")

    for key, domain, title in [
        ("plain", plain_domain, "Interior Without Outer Refinement"),
        ("refined", refined_domain, "Interior With Boundary Refinement"),
        ("variable", variable_domain, "Interior With Variable Density"),
    ]:
        xi = jnp.asarray(domain.get_interior_nodes())
        xb_local = jnp.asarray(domain.get_bdry_nodes())
        axes[key].plot(xi[:, 0], xi[:, 1], ".", color="#1d4ed8", ms=3, label="Interior")
        axes[key].plot(xb_local[:, 0], xb_local[:, 1], ".", color="#0f766e", ms=3, label="Boundary")
        axes[key].set_title(title)
    axes["variable"].legend(frameon=False, fontsize=9, loc="upper right")

    for ax in axes.values():
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)

    path = OUTDIR / "geometry_domain.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def h_channel_segments(n_sites: int = 40) -> tuple[list[jnp.ndarray], list[bool]]:
    width = 2.0
    height = 1.0
    throat_half_width = 0.1
    throat_half_height = 0.1
    throat_center_x = 0.5 * width
    throat_center_y = 0.5 * height
    x_min = 0.0
    y_min = 0.0

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

    add_horizontal(x_min, x_min + width, y_min, False)
    add_horizontal(x_min, x_min + width, y_min + height, True)
    add_vertical(x_min, y_min, y_min + throat_center_y - throat_half_height, True)
    add_vertical(x_min + width, y_min, y_min + throat_center_y - throat_half_height, False)
    add_vertical(x_min, y_min + throat_center_y + throat_half_height, y_min + height, True)
    add_vertical(x_min + width, y_min + throat_center_y + throat_half_height, y_min + height, False)
    add_vertical(x_min + throat_center_x - throat_half_width, y_min + throat_center_y - throat_half_height, y_min + throat_center_y + throat_half_height, True)
    add_vertical(x_min + throat_center_x + throat_half_width, y_min + throat_center_y - throat_half_height, y_min + throat_center_y + throat_half_height, False)
    add_horizontal(x_min, x_min + throat_center_x - throat_half_width, y_min + throat_center_y - throat_half_height, True)
    add_horizontal(x_min, x_min + throat_center_x - throat_half_width, y_min + throat_center_y + throat_half_height, False)
    add_horizontal(x_min + throat_center_x + throat_half_width, x_min + width, y_min + throat_center_y - throat_half_height, True)
    add_horizontal(x_min + throat_center_x + throat_half_width, x_min + width, y_min + throat_center_y + throat_half_height, False)
    return segments, flip_normal


def build_h_channel_piecewise_domain() -> tuple[PiecewiseSmoothEmbeddedSurface, object]:
    surface = PiecewiseSmoothEmbeddedSurface()
    segments, flip_normal = h_channel_segments()
    surface.generate_piecewise_smooth_surface_by_segment(segments, flip_normal, 0.08, method=1, supersample_fac=2)
    surface.build_level_set()

    generator = DomainNodeGenerator()
    domain = generator.build_domain_descriptor_from_geometry(
        surface,
        0.10,
        seed=11,
        strip_count=4,
        do_outer_refinement=True,
        outer_fraction_of_h=0.5,
        outer_refinement_zone_size_as_multiple_of_h=1.5,
    )
    return surface, domain


def ice_cap_segments() -> tuple[list[jnp.ndarray], list[bool], list[bool], list[float], list[int], list[bool]]:
    sphere_radius = 1.0
    cap_height = ICE_CAP_HEIGHT
    base_y = sphere_radius - cap_height
    base_radius = jnp.sqrt(sphere_radius**2 - base_y**2)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 120, endpoint=False)
    base_curve = jnp.column_stack([base_radius * jnp.cos(theta), base_radius * jnp.sin(theta)])
    cap = geometry.fibonacci_sphere(1600)
    cap = cap[cap[:, 1] >= base_y]
    base_ring = jnp.column_stack([base_radius * jnp.cos(theta), jnp.full_like(theta, base_y), base_radius * jnp.sin(theta)])
    cap = jnp.vstack([cap, base_ring])
    return [base_curve, cap], [True, False], [True, False], [float(base_y), -1.0], [1, -1], [True, True]


def filter_ice_cap_volume(points: jnp.ndarray, tolerance: float = 1.0e-10) -> jnp.ndarray:
    """Keep points inside the analytic spherical cap volume."""
    pts = jnp.asarray(points, dtype=float)
    if pts.size == 0:
        return pts.reshape(0, 3)
    base_y = 1.0 - ICE_CAP_HEIGHT
    radius = jnp.linalg.norm(pts, axis=1)
    mask = (pts[:, 1] >= base_y - tolerance) & (radius <= 1.0 + tolerance)
    return pts[mask]


def build_ice_cap_piecewise_domain():
    bdry_desc, is_bdry_curve, is_smooth_curve, constants, constant_dim, flip_normal = ice_cap_segments()
    generator = DomainNodeGenerator()
    domain = generator.generate_piecewise_smooth_domain_nodes_hierarchical(
        bdry_desc,
        is_bdry_curve,
        is_smooth_curve,
        constants,
        constant_dim,
        flip_normal,
        0.08,
        method=2,
        supersample_fac=4,
        seed=7,
        strip_count=4,
        do_outer_refinement=True,
        outer_fraction_of_h=0.6,
        outer_refinement_zone_size_as_multiple_of_h=1.5,
    )
    xi = filter_ice_cap_volume(generator.xi_pds_raw)
    generator.xi = xi
    generator.xi_orig = xi
    domain.set_nodes(xi, domain.get_bdry_nodes(), domain.get_ghost_nodes())
    return generator, domain


def save_piecewise_domains(h_domain, ice_domain) -> Path:
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    ax_h = fig.add_subplot(1, 2, 1)
    ax_ice = fig.add_subplot(1, 2, 2, projection="3d")
    xi_h = jnp.asarray(h_domain.get_interior_nodes())
    xb_h = jnp.asarray(h_domain.get_bdry_nodes())
    xg_h = jnp.asarray(h_domain.get_ghost_nodes())
    ax_h.plot(xi_h[:, 0], xi_h[:, 1], ".", color="#2563eb", ms=3, label="interior")
    ax_h.plot(xb_h[:, 0], xb_h[:, 1], ".", color="#0f766e", ms=2.5, label="boundary")
    ax_h.plot(xg_h[:, 0], xg_h[:, 1], ".", color="#ca8a04", ms=2.5, label="ghost")
    ax_h.set_title("H-Channel Piecewise Boundary")
    ax_h.set_aspect("equal", adjustable="box")
    ax_h.grid(alpha=0.2)
    ax_h.legend(frameon=False, loc="upper right")

    xi_i = jnp.asarray(ice_domain.get_interior_nodes())
    xb_i = jnp.asarray(ice_domain.get_bdry_nodes())
    xg_i = jnp.asarray(ice_domain.get_ghost_nodes())
    ax_ice.scatter(xi_i[:, 0], xi_i[:, 2], xi_i[:, 1], s=8, c="#2563eb", depthshade=False, label="interior")
    ax_ice.scatter(xb_i[:, 0], xb_i[:, 2], xb_i[:, 1], s=4, c="#0f766e", alpha=0.6, depthshade=False, label="boundary")
    ax_ice.scatter(xg_i[:, 0], xg_i[:, 2], xg_i[:, 1], s=4, c="#ca8a04", alpha=0.45, depthshade=False, label="ghost")
    ax_ice.set_title("3D Ice-Cap Hierarchical Boundary")
    ax_ice.set_xlabel("x")
    ax_ice.set_ylabel("z")
    ax_ice.set_zlabel("y")
    ax_ice.view_init(elev=24, azim=-58)
    ax_ice.legend(frameon=False, loc="upper left")

    path = OUTDIR / "piecewise_domain_nodes.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_poisson(domain) -> Path:
    solver = PoissonSolver(
        lap_assembler="fd",
        bc_assembler="fd",
        lap_stencil="rbf",
        bc_stencil="rbf",
    )
    solver.init(domain, 4)

    u_exact = lambda x: (x[:, 0] ** 2 + x[:, 1] ** 2) ** 2 - (x[:, 0] ** 2 + x[:, 1] ** 2) + 1.0 / 6.0
    forcing = lambda x: 4.0 - 16.0 * (x[:, 0] ** 2 + x[:, 1] ** 2)
    neu_coeff = lambda xb: jnp.ones(xb.shape[0], dtype=float)
    dir_coeff = lambda xb: jnp.zeros(xb.shape[0], dtype=float)
    bc = lambda neu_coeffs, dir_coeffs, nr, xb: jnp.sum(
        jnp.column_stack(
            [
                4.0 * xb[:, 0] * (xb[:, 0] ** 2 + xb[:, 1] ** 2) - 2.0 * xb[:, 0],
                4.0 * xb[:, 1] * (xb[:, 0] ** 2 + xb[:, 1] ** 2) - 2.0 * xb[:, 1],
            ]
        )
        * nr,
        axis=1,
    )

    result = solver.solve(forcing, neu_coeff, dir_coeff, bc)
    x_phys = jnp.asarray(domain.get_int_bdry_nodes())
    u = jnp.asarray(result["u"])
    u_true = u_exact(x_phys)
    u = u - jnp.mean(u - u_true)
    err = u - u_true

    tri = mtri.Triangulation(x_phys[:, 0], x_phys[:, 1])

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    cf0 = axes[0].tricontourf(tri, u, levels=24, cmap="viridis")
    xb = jnp.asarray(domain.get_bdry_nodes())
    axes[0].plot(xb[:, 0], xb[:, 1], "k.", ms=1.5, alpha=0.5)
    axes[0].set_title("Poisson Solution")
    fig.colorbar(cf0, ax=axes[0], shrink=0.9)

    cf1 = axes[1].tricontourf(tri, err, levels=24, cmap="coolwarm")
    axes[1].set_title(f"Poisson Error\nmax |e| = {float(jnp.max(jnp.abs(err))):.2e}")
    fig.colorbar(cf1, ax=axes[1], shrink=0.9)

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15)

    path = OUTDIR / "poisson_solution.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def save_diffusion(domain) -> Path:
    solver = DiffusionSolver(
        lap_assembler="fd",
        bc_assembler="fd",
        lap_stencil="rbf",
        bc_stencil="rbf",
    )
    nu = 0.25
    dt = 0.02
    t_final = 0.50
    nsteps = int(round(t_final / dt))
    solver.init(domain, 4, dt, nu)

    x_phys = jnp.asarray(domain.get_int_bdry_nodes())
    u_exact = lambda time, x: jnp.exp(-time) * (x[:, 0] ** 2 + x[:, 1] ** 2)
    forcing = lambda nu_value, time, x: -jnp.exp(-time) * (x[:, 0] ** 2 + x[:, 1] ** 2) - 4.0 * nu_value * jnp.exp(-time)
    neu_coeff = lambda xb: jnp.zeros(xb.shape[0])
    dir_coeff = lambda xb: jnp.ones(xb.shape[0])
    bc = lambda neu_coeffs, dir_coeffs, nr, time, xb: u_exact(time, xb)

    solver.set_initial_state(u_exact(0.0, x_phys))
    times = [0.0]
    errors = [0.0]

    for step in range(1, nsteps + 1):
        time = step * dt
        if step == 1:
            u_next = solver.bdf1_step(time, forcing, neu_coeff, dir_coeff, bc)
        elif step == 2:
            u_next = solver.bdf2_step(time, forcing, neu_coeff, dir_coeff, bc)
        else:
            u_next = solver.bdf3_step(time, forcing, neu_coeff, dir_coeff, bc)
        times.append(time)
        errors.append(float(jnp.max(jnp.abs(u_next - u_exact(time, x_phys)))))

    u_final = jnp.asarray(solver.current_physical_state())
    u_true_final = u_exact(t_final, x_phys)
    err = u_final - u_true_final
    tri = mtri.Triangulation(x_phys[:, 0], x_phys[:, 1])

    fig = plt.figure(figsize=(8, 8), constrained_layout=True)
    axes = fig.subplot_mosaic([["solution", "error"], ["history", "history"]])
    cf0 = axes["solution"].tricontourf(tri, u_final, levels=24, cmap="viridis")
    axes["solution"].set_title(f"Diffusion at t = {t_final:.2f}")
    fig.colorbar(cf0, ax=axes["solution"], shrink=0.85)

    cf1 = axes["error"].tricontourf(tri, err, levels=24, cmap="coolwarm")
    axes["error"].set_title(f"Final-Time Error\nmax |e| = {float(jnp.max(jnp.abs(err))):.2e}")
    fig.colorbar(cf1, ax=axes["error"], shrink=0.85)

    axes["history"].plot(times, errors, color="#1d4ed8", lw=2)
    axes["history"].scatter(times, errors, color="#0f766e", s=18)
    axes["history"].set_title("Time March to Final Time")
    axes["history"].set_xlabel("t")
    axes["history"].set_ylabel("max nodal error")
    axes["history"].grid(alpha=0.2)

    for ax in (axes["solution"], axes["error"]):
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15)

    path = OUTDIR / "diffusion_solution.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    geometry_surface = build_surface(50, 0.05)
    geometry_generator_plain = DomainNodeGenerator()
    geometry_domain_plain = geometry_generator_plain.build_domain_descriptor_from_geometry(
        geometry_surface,
        0.08,
        seed=17,
        strip_count=5,
        do_outer_refinement=False,
        mapped_cloud_preferred=False,
    )
    geometry_generator_refined = DomainNodeGenerator()
    geometry_domain_refined = geometry_generator_refined.build_domain_descriptor_from_geometry(
        geometry_surface,
        0.08,
        seed=17,
        strip_count=5,
        do_outer_refinement=True,
        outer_fraction_of_h=0.5,
        outer_refinement_zone_size_as_multiple_of_h=2.0,
        mapped_cloud_preferred=False,
    )
    geometry_generator_variable = DomainNodeGenerator()
    geometry_domain_variable = geometry_generator_variable.build_domain_descriptor_from_geometry(
        geometry_surface,
        0.08,
        seed=17,
        strip_count=5,
        radius_function=variable_density_radius,
        mapped_cloud_preferred=False,
    )
    _, solver_domain = build_domain(do_outer_refinement=True)
    _, h_domain = build_h_channel_piecewise_domain()
    _, ice_domain = build_ice_cap_piecewise_domain()
    paths = [
        save_geometry(geometry_surface, geometry_domain_plain, geometry_domain_refined, geometry_domain_variable),
        save_piecewise_domains(h_domain, ice_domain),
        save_poisson(solver_domain),
        save_diffusion(solver_domain),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
