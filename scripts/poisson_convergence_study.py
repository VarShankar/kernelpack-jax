from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp

from kernelpack import domain, geometry, nodes, solvers


@dataclass(frozen=True)
class StudyResult:
    order: int
    resolution: str
    h: float
    node_count: int
    l2_error: float
    linf_error: float


def exact_solution(x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=float)
    return jnp.exp(x[:, 0] + 0.35 * x[:, 1])


def forcing(x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=float)
    return -(1.0 + 0.35**2) * jnp.exp(x[:, 0] + 0.35 * x[:, 1])


def build_ellipse_surface(site_count: int = 120) -> geometry.EmbeddedSurface:
    t = jnp.linspace(0.0, 2.0 * jnp.pi, site_count, endpoint=False)
    curve = jnp.column_stack([jnp.cos(t), 0.8 * jnp.sin(t)])
    surface = geometry.EmbeddedSurface()
    surface.set_data_sites(curve)
    surface.build_closed_geometric_model_ps(2, 0.05, curve.shape[0])
    surface.build_level_set_from_geometric_model()
    return surface


def build_ellipse_domain(h: float, seed: int) -> tuple[domain.DomainDescriptor, str]:
    generator = nodes.DomainNodeGenerator()
    descriptor = generator.build_domain_descriptor_from_geometry(
        build_ellipse_surface(),
        h,
        seed=seed,
        strip_count=5,
        do_outer_refinement=True,
        outer_fraction_of_h=0.5,
        outer_refinement_zone_size_as_multiple_of_h=2.0,
    )
    return descriptor, f"h={h:.5f}"


def build_square_domain(n_side: int) -> tuple[domain.DomainDescriptor, float]:
    xs = jnp.linspace(-1.0, 1.0, n_side)
    h = float(xs[1] - xs[0])
    xg, yg = jnp.meshgrid(xs, xs, indexing="ij")
    pts = jnp.column_stack([xg.ravel(), yg.ravel()])
    on_bdry = (jnp.isclose(jnp.abs(pts[:, 0]), 1.0, atol=1e-12) | jnp.isclose(jnp.abs(pts[:, 1]), 1.0, atol=1e-12))
    xb = pts[on_bdry]
    xi = pts[~on_bdry]

    nr = jnp.zeros_like(xb)
    left = jnp.isclose(xb[:, 0], -1.0, atol=1e-12)
    right = jnp.isclose(xb[:, 0], 1.0, atol=1e-12)
    bottom = jnp.isclose(xb[:, 1], -1.0, atol=1e-12)
    top = jnp.isclose(xb[:, 1], 1.0, atol=1e-12)
    nr = nr + left[:, None] * jnp.array([-1.0, 0.0])
    nr = nr + right[:, None] * jnp.array([1.0, 0.0])
    nr = nr + bottom[:, None] * jnp.array([0.0, -1.0])
    nr = nr + top[:, None] * jnp.array([0.0, 1.0])
    nr = nr / jnp.linalg.norm(nr, axis=1, keepdims=True)
    xghost = xb + 0.5 * h * nr

    descriptor = domain.DomainDescriptor()
    descriptor.set_nodes(xi, xb, xghost)
    descriptor.set_normals(nr)
    descriptor.set_sep_rad(h)
    descriptor.build_structs()
    return descriptor, h


def solve_case(descriptor: domain.DomainDescriptor, order: int, stencil: str) -> tuple[int, float, float]:
    solver = solvers.PoissonSolver(
        lap_assembler="fd",
        bc_assembler="fd",
        lap_stencil=stencil,
        bc_stencil=stencil,
    )
    solver.init(descriptor, order)
    x_phys = jnp.asarray(descriptor.get_int_bdry_nodes(), dtype=float)
    result = solver.solve(
        forcing,
        lambda xb: jnp.zeros(xb.shape[0], dtype=float),
        lambda xb: jnp.ones(xb.shape[0], dtype=float),
        lambda neu_coeff, dir_coeff, nr, xb: exact_solution(xb),
    )
    u_num = jnp.asarray(result["u"], dtype=float)
    u_true = exact_solution(x_phys)
    err = u_num - u_true
    return int(x_phys.shape[0]), float(jnp.linalg.norm(err) / jnp.sqrt(err.size)), float(jnp.max(jnp.abs(err)))


def compute_observed_rates(values: list[float], hs: list[float]) -> list[float]:
    rates = [float("nan")]
    for i in range(1, len(values)):
        rates.append(float(jnp.log(values[i - 1] / values[i]) / jnp.log(hs[i - 1] / hs[i])))
    return rates


def render_markdown(results: list[StudyResult], benchmark: str, stencil: str) -> str:
    lines = [
        f"# Poisson Convergence Study ({benchmark}, {stencil.upper()})",
        "",
        f"- JAX backend: `{jax.default_backend()}`",
        f"- Devices: `{jax.devices()}`",
        "",
        "| xi | resolution | h | physical nodes | L2 error | L2 rate | Linf error | Linf rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for order in sorted({r.order for r in results}):
        order_rows = [r for r in results if r.order == order]
        hs = [r.h for r in order_rows]
        l2s = [r.l2_error for r in order_rows]
        linfs = [r.linf_error for r in order_rows]
        l2_rates = compute_observed_rates(l2s, hs)
        linf_rates = compute_observed_rates(linfs, hs)
        for row, l2_rate, linf_rate in zip(order_rows, l2_rates, linf_rates, strict=True):
            lines.append(
                f"| {row.order} | {row.resolution} | {row.h:.5f} | {row.node_count} | {row.l2_error:.6e} | "
                f"{'' if jnp.isnan(l2_rate) else f'{l2_rate:.3f}'} | "
                f"{row.linf_error:.6e} | "
                f"{'' if jnp.isnan(linf_rate) else f'{linf_rate:.3f}'} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a 2D manufactured-solution Poisson convergence study.")
    parser.add_argument("--orders", nargs="+", type=int, default=[2, 4, 6])
    parser.add_argument("--stencil", choices=["wls", "rbf"], default="wls")
    parser.add_argument("--benchmark", choices=["square", "ellipse"], default="square")
    parser.add_argument("--grid-sizes", nargs="+", type=int, default=[9, 13, 17, 25, 33], help="Only used for the square benchmark.")
    parser.add_argument("--hs", nargs="+", type=float, default=[0.24, 0.18, 0.135, 0.10], help="Only used for the ellipse benchmark.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=None, help="Optional markdown output path.")
    args = parser.parse_args()

    results: list[StudyResult] = []
    if args.benchmark == "square":
        for order in args.orders:
            for n_side in args.grid_sizes:
                descriptor, h = build_square_domain(n_side)
                node_count, l2_error, linf_error = solve_case(descriptor, order, args.stencil)
                results.append(
                    StudyResult(
                        order=order,
                        resolution=str(n_side),
                        h=h,
                        node_count=node_count,
                        l2_error=l2_error,
                        linf_error=linf_error,
                    )
                )
    else:
        for order in args.orders:
            for h in args.hs:
                descriptor, resolution = build_ellipse_domain(h, seed=args.seed)
                node_count, l2_error, linf_error = solve_case(descriptor, order, args.stencil)
                results.append(
                    StudyResult(
                        order=order,
                        resolution=resolution,
                        h=h,
                        node_count=node_count,
                        l2_error=l2_error,
                        linf_error=linf_error,
                    )
                )

    markdown = render_markdown(results, args.benchmark, args.stencil)
    print(markdown)
    if args.output is not None:
        args.output.write_text(markdown + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
