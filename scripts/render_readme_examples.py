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

BACKGROUND = "#fbfaf6"
INK = "#172033"
INTERIOR = "#3157d5"
BOUNDARY = "#0f8b8d"
GHOST = "#f2a541"


def style_figure(fig) -> None:
    fig.patch.set_facecolor(BACKGROUND)


def style_surface_axes(ax) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor(BACKGROUND)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def build_surface(n_sites: int, geom_radius: float, aspect: float = 0.7) -> EmbeddedSurface:
    t = jnp.linspace(0.0, 2.0 * jnp.pi, n_sites, endpoint=False)
    curve = jnp.column_stack([jnp.cos(t), aspect * jnp.sin(t)])
    surface = EmbeddedSurface()
    surface.set_data_sites(curve)
    surface.build_closed_geometric_model_ps(2, geom_radius, curve.shape[0])
    surface.build_level_set_from_geometric_model()
    return surface


def build_domain(*, do_outer_refinement: bool = True) -> tuple[EmbeddedSurface, object]:
    surface = build_surface(160, 0.06, aspect=1.0)
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
    del plain_domain, variable_domain
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    style_figure(fig)
    data_sites = jnp.asarray(surface.data_sites)
    xb = jnp.asarray(refined_domain.get_bdry_nodes())
    nr = jnp.asarray(refined_domain.get_nrmls())
    xi = jnp.asarray(refined_domain.get_interior_nodes())
    xg = jnp.asarray(refined_domain.get_ghost_nodes())

    axes[0].scatter(data_sites[:, 0], data_sites[:, 1], s=14, color="#cbd3df", edgecolors="none")
    axes[0].scatter(xb[:, 0], xb[:, 1], s=12, color=BOUNDARY, edgecolors="none")
    step = max(1, xb.shape[0] // 40)
    axes[0].quiver(
        xb[::step, 0], xb[::step, 1], nr[::step, 0], nr[::step, 1],
        angles="xy", scale_units="xy", scale=13, color="#df5b49",
        width=0.004, headwidth=4,
    )
    axes[0].set_title("Geometry and outward normals", color=INK, weight="semibold", pad=10)

    axes[1].scatter(xi[:, 0], xi[:, 1], s=10, color=INTERIOR, edgecolors="none", label="interior")
    axes[1].scatter(xb[:, 0], xb[:, 1], s=11, color=BOUNDARY, edgecolors="none", label="boundary")
    axes[1].scatter(xg[:, 0], xg[:, 1], s=11, color=GHOST, edgecolors="none", label="ghost")
    axes[1].legend(
        frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.08),
        ncol=3, fontsize=10, markerscale=1.4,
    )
    axes[1].set_title("Boundary-refined node cloud", color=INK, weight="semibold", pad=10)

    for ax in axes:
        style_surface_axes(ax)
    fig.suptitle("From sampled geometry to a meshfree domain", color=INK, weight="bold", fontsize=16)

    path = OUTDIR / "geometry_domain.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=BACKGROUND)
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

    u_exact = lambda x: 1.0 - x[:, 0] ** 2 - x[:, 1] ** 2
    forcing = lambda x: 4.0 * jnp.ones(x.shape[0], dtype=float)
    neu_coeff = lambda xb: jnp.zeros(xb.shape[0], dtype=float)
    dir_coeff = lambda xb: jnp.ones(xb.shape[0], dtype=float)
    bc = lambda neu_coeffs, dir_coeffs, nr, xb: u_exact(xb)

    result = solver.solve(forcing, neu_coeff, dir_coeff, bc)
    x_phys = jnp.asarray(domain.get_int_bdry_nodes())
    u = jnp.asarray(result["u"])
    u_true = u_exact(x_phys)

    tri = mtri.Triangulation(x_phys[:, 0], x_phys[:, 1])

    fig, ax = plt.subplots(figsize=(9.4, 4.8), constrained_layout=True)
    style_figure(fig)
    field = ax.tripcolor(tri, u, shading="gouraud", cmap="viridis")
    xb = jnp.asarray(domain.get_bdry_nodes())
    ax.plot(xb[:, 0], xb[:, 1], color=INK, lw=1.1, alpha=0.8)
    ax.tricontour(tri, u, levels=9, colors="white", linewidths=0.45, alpha=0.38)
    style_surface_axes(ax)
    ax.set_title("GPU-ready meshfree Poisson solution", color=INK, weight="bold", fontsize=16, pad=12)
    colorbar = fig.colorbar(field, ax=ax, shrink=0.82, pad=0.03)
    colorbar.set_label(r"$u_h$", color=INK, weight="semibold")
    colorbar.outline.set_visible(False)

    path = OUTDIR / "poisson_solution.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=BACKGROUND)
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
    geometry_surface, solver_domain = build_domain(do_outer_refinement=True)
    paths = [
        save_geometry(geometry_surface, solver_domain, solver_domain, solver_domain),
        save_poisson(solver_domain),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
