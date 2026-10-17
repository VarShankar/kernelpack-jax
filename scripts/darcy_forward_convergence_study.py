from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import gc
from pathlib import Path
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.io import savemat

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

INV = Path(__file__).resolve().parents[1] / "inverse problems"
if str(INV) not in sys.path:
    sys.path.insert(0, str(INV))

from kernelpack.domain import DomainDescriptor
from kernelpack.nodes import DomainNodeGenerator
from kernelpack.solvers import VariablePoissonSolver

from common import StaticInverseCase, build_domain, default_h, relative_l2
from common import _build_surface_2d
from darcy_model_selection import coefficient_field, manufactured_forcing
from darcy_inverse_convergence_study import (
    FAMILIES,
    THREE_D_GEOMETRIES,
    TWO_D_GEOMETRIES,
    SavedDomainMetadata,
    build_cases,
    default_geometry,
    model_features,
    sample_theta,
    validate_geometry,
)

DEFAULT_OUTPUT = Path(__file__).with_name("darcy_forward_convergence_study.md")
DEFAULT_MAT_OUTPUT = Path(__file__).with_name("darcy_forward_convergence_study-egg-const.mat")


@dataclass(frozen=True)
class ForwardResult:
    order: int
    resolution: str
    h: float
    node_count: int
    mean_relative_l2_error: float
    max_relative_l2_error: float
    elapsed_seconds: float
    preprocess_elapsed_seconds: float


def load_domain_file(path: Path) -> tuple[DomainDescriptor, SavedDomainMetadata]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        descriptor = DomainDescriptor()
        descriptor.set_nodes(
            jnp.asarray(data["int_nodes"], dtype=float),
            jnp.asarray(data["bdry_nodes"], dtype=float),
            jnp.asarray(data["ghost_nodes"], dtype=float),
        )
        descriptor.set_normals(jnp.asarray(data["normals"], dtype=float))
        descriptor.set_sep_rad(float(np.asarray(data["sep_rad"]).item()))
        descriptor.build_structs()
        metadata = SavedDomainMetadata(
            dim=int(np.asarray(data["dim"]).item()),
            geometry=str(np.asarray(data["geometry"]).item()),
            h=float(np.asarray(data["h"]).item()),
            seed=int(np.asarray(data["seed"]).item()),
            sep_rad=float(np.asarray(data["sep_rad"]).item()),
            label=str(np.asarray(data["label"]).item()),
        )
    return descriptor, metadata


def setup_device_context(setup_device: str):
    if setup_device == "cpu":
        cpu_devices = jax.devices("cpu")
        if not cpu_devices:
            raise RuntimeError("requested CPU setup device, but JAX did not report a CPU device")
        return jax.default_device(cpu_devices[0])
    if setup_device == "default":
        return nullcontext()
    raise ValueError(f"unknown setup device {setup_device}")


def copy_domain_to_device(domain: DomainDescriptor, device) -> DomainDescriptor:
    copied = DomainDescriptor()
    copied.set_nodes(
        jax.device_put(domain.get_interior_nodes(), device),
        jax.device_put(domain.get_bdry_nodes(), device),
        jax.device_put(domain.get_ghost_nodes(), device),
    )
    copied.set_normals(jax.device_put(domain.get_nrmls(), device))
    copied.set_sep_rad(float(domain.get_sep_rad()))
    copied.set_outer_level_set(domain.get_outer_level_set())
    copied.set_boundary_level_sets(domain.get_boundary_level_sets())
    copied.build_structs()
    return copied


def build_forward_domain(dim: int, geometry_name: str, h: float, seed: int, *, mapped_cloud_preferred: bool) -> DomainDescriptor:
    if dim == 2 and mapped_cloud_preferred:
        surface = _build_surface_2d(geometry_name)
        generator = DomainNodeGenerator()
        return generator.build_domain_descriptor_from_geometry(
            surface,
            h,
            seed=seed,
            strip_count=5,
            do_outer_refinement=True,
            outer_fraction_of_h=0.5,
            outer_refinement_zone_size_as_multiple_of_h=2.0,
            mapped_cloud_preferred=True,
        )
    return build_domain(dim, geometry_name, h, seed)


def build_solver(
    domain: DomainDescriptor,
    order: int,
    stencil: str,
    linear_solver: str,
    sparse_assembly_chunk_size: int | None,
) -> VariablePoissonSolver:
    solver = VariablePoissonSolver(
        lap_assembler="fd",
        bc_assembler="fd",
        lap_stencil=stencil,
        bc_stencil=stencil,
        linear_solver=linear_solver,
    )
    solver.init(
        domain,
        order,
        sparse_assembly_chunk_size=sparse_assembly_chunk_size,
        assemble_dense_operators=linear_solver not in ("gmres_ilu", "scipy_gmres_ilu", "scipy_sparse_direct", "cudss"),
    )
    return solver


def synchronize_solver_operators(solver: VariablePoissonSolver) -> None:
    """Block on assembled dense/sparse operators so timing includes assembly work."""

    jax.block_until_ready(solver.lap)
    for grad in solver.grad:
        jax.block_until_ready(grad)
    if solver.lap_sparse is not None:
        jax.block_until_ready(solver.lap_sparse.indices)
        jax.block_until_ready(solver.lap_sparse.values)
    for grad_sparse in solver.grad_sparse:
        jax.block_until_ready(grad_sparse.indices)
        jax.block_until_ready(grad_sparse.values)
    if solver.bc.size:
        jax.block_until_ready(solver.bc)
    if solver.bc_sparse is not None:
        jax.block_until_ready(solver.bc_sparse.indices)
        jax.block_until_ready(solver.bc_sparse.values)


def solve_cases(
    *,
    solver: VariablePoissonSolver,
    family: str,
    theta_true: jnp.ndarray,
    cases: list[StaticInverseCase],
    x_all: jnp.ndarray,
    phase: str,
) -> tuple[float, float]:
    all_features = model_features(family, x_all.shape[1], x_all)[0]
    neu_coeff = jnp.zeros((solver.domain.get_num_bdry_nodes(),), dtype=float)
    dir_coeff = jnp.ones((solver.domain.get_num_bdry_nodes(),), dtype=float)
    print(f"[{phase}] preparing boundary operator", flush=True)
    solver.prepare_boundary_operator(neu_coeff, dir_coeff)
    synchronize_solver_operators(solver)
    print(f"[{phase}] boundary operator ready", flush=True)
    if solver.linear_solver in ("gmres_ilu", "scipy_gmres_ilu"):
        print(f"[{phase}] building sparse ILU preconditioner", flush=True)
        solver.prepare_preconditioner(
            lambda xnodes, feats=all_features, th=theta_true: jnp.exp(feats @ th),
            neu_coeff,
            dir_coeff,
        )
        print(f"[{phase}] sparse ILU preconditioner ready", flush=True)

    errors = []
    for case_index, case in enumerate(cases, start=1):
        print(f"[{phase}] solving manufactured state {case_index}/{len(cases)}", flush=True)
        out = solver.solve(
            case.forcing,
            lambda xnodes, fam=family, th=theta_true: coefficient_field(fam, th, xnodes),
            neu_coeff,
            dir_coeff,
            case.boundary_target,
        )
        error = relative_l2(out["u"], case.target)
        jax.block_until_ready(error)
        errors.append(error)
        print(
            f"[{phase}] manufactured state {case_index}/{len(cases)} done: "
            f"relative_l2={float(error):.6e}",
            flush=True,
        )
    errors_arr = jnp.asarray(errors)
    jax.block_until_ready(errors_arr)
    return float(jnp.mean(errors_arr)), float(jnp.max(errors_arr))


def run_case(
    *,
    domain: DomainDescriptor,
    metadata: SavedDomainMetadata,
    family: str,
    order: int,
    stencil: str,
    linear_solver: str,
    setup_device: str,
    sparse_assembly_chunk_size: int | None,
    timing_warmup: bool,
) -> ForwardResult:
    if setup_device == "cpu":
        setup_domain = copy_domain_to_device(domain, jax.devices("cpu")[0])
    else:
        setup_domain = domain
    with setup_device_context(setup_device):
        x_phys = jnp.asarray(setup_domain.get_int_bdry_nodes(), dtype=float)
        x_bdry = jnp.asarray(setup_domain.get_bdry_nodes(), dtype=float)
        x_all = jnp.asarray(setup_domain.get_all_nodes(), dtype=float)
        dim = x_phys.shape[1]
        theta_true = sample_theta(family, dim)
        cases = build_cases(family, theta_true, x_phys, x_bdry)

        preprocess_elapsed_seconds = 0.0
        if timing_warmup:
            preprocess_start_time = perf_counter()
            print("[warmup] building solver", flush=True)
            warmup_solver = build_solver(setup_domain, order, stencil, linear_solver, sparse_assembly_chunk_size)
            solve_cases(
                solver=warmup_solver,
                family=family,
                theta_true=theta_true,
                cases=cases,
                x_all=x_all,
                phase="warmup",
            )
            preprocess_elapsed_seconds = perf_counter() - preprocess_start_time
            del warmup_solver
            gc.collect()

        start_time = perf_counter()
        print("[recovery] building solver", flush=True)
        solver = build_solver(setup_domain, order, stencil, linear_solver, sparse_assembly_chunk_size)
        mean_err, max_err = solve_cases(
            solver=solver,
            family=family,
            theta_true=theta_true,
            cases=cases,
            x_all=x_all,
            phase="recovery",
        )
        elapsed_seconds = perf_counter() - start_time
    return ForwardResult(
        order=order,
        resolution=metadata.label,
        h=metadata.h,
        node_count=int(domain.get_num_int_bdry_nodes()),
        mean_relative_l2_error=mean_err,
        max_relative_l2_error=max_err,
        elapsed_seconds=elapsed_seconds,
        preprocess_elapsed_seconds=preprocess_elapsed_seconds,
    )


def export_forward_mat(
    results: list[ForwardResult],
    *,
    dim: int,
    geometry: str,
    family: str,
    stencil: str,
    linear_solver: str,
    output_path: Path,
) -> None:
    orders = sorted({row.order for row in results})
    labels = []
    seen = set()
    for row in results:
        if row.resolution not in seen:
            labels.append(row.resolution)
            seen.add(row.resolution)
    order_to_idx = {order: i for i, order in enumerate(orders)}
    label_to_idx = {label: i for i, label in enumerate(labels)}
    shape = (len(orders), len(labels))
    relative_l2_errors = np.full(shape, np.nan, dtype=float)
    max_relative_l2_errors = np.full(shape, np.nan, dtype=float)
    elapsed_seconds = np.full(shape, np.nan, dtype=float)
    preprocess_elapsed_seconds = np.full(shape, np.nan, dtype=float)
    node_counts = np.full(shape, np.nan, dtype=float)
    hs = np.full((len(labels),), np.nan, dtype=float)

    for row in results:
        i = order_to_idx[row.order]
        j = label_to_idx[row.resolution]
        relative_l2_errors[i, j] = row.mean_relative_l2_error
        max_relative_l2_errors[i, j] = row.max_relative_l2_error
        elapsed_seconds[i, j] = row.elapsed_seconds
        preprocess_elapsed_seconds[i, j] = row.preprocess_elapsed_seconds
        node_counts[i, j] = row.node_count
        hs[j] = row.h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        output_path,
        {
            "relative_l2_errors": relative_l2_errors,
            "max_relative_l2_errors": max_relative_l2_errors,
            "elapsed_seconds": elapsed_seconds,
            "preprocess_elapsed_seconds": preprocess_elapsed_seconds,
            "orders": np.asarray(orders, dtype=float),
            "hs": hs,
            "point_cloud_numbers": np.arange(1, len(labels) + 1, dtype=float),
            "labels": np.asarray(labels, dtype=object),
            "node_counts": node_counts,
            "geometry": np.asarray([geometry], dtype=object),
            "dimension": np.asarray([[dim]], dtype=float),
            "family": np.asarray([family], dtype=object),
            "stencil": np.asarray([stencil], dtype=object),
            "linear_solver": np.asarray([linear_solver], dtype=object),
            "error_metric": np.asarray(["mean_relative_l2_over_manufactured_states"], dtype=object),
        },
    )


def render_markdown(
    results: list[ForwardResult],
    *,
    dim: int,
    geometry: str,
    family: str,
    stencil: str,
    linear_solver: str,
) -> str:
    lines = [
        f"# Darcy Forward Convergence Study ({family}, {stencil.upper()})",
        "",
        f"- dimension: {dim}",
        f"- geometry: {geometry}",
        f"- family: {family}",
        f"- stencil: {stencil}",
        f"- linear solver: {linear_solver}",
        f"- JAX backend: `{jax.default_backend()}`",
        f"- timing: `preprocess seconds is 0 unless --timing-warmup is set; seconds is the measured solve pass`",
        "",
        "| xi | resolution | h | physical nodes | mean relative L2 | max relative L2 | preprocess seconds | seconds |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {row.order} | {row.resolution} | {row.h:.5f} | {row.node_count} | "
            f"{row.mean_relative_l2_error:.6e} | {row.max_relative_l2_error:.6e} | "
            f"{row.preprocess_elapsed_seconds:.6f} | {row.elapsed_seconds:.6f} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Darcy forward convergence study using the inverse-test manufactured families.")
    parser.add_argument("--family", type=str, default="cross_term", choices=FAMILIES)
    parser.add_argument("--dim", type=int, default=2, choices=(2, 3))
    parser.add_argument("--geometry", type=str, default=None)
    parser.add_argument("--orders", type=int, nargs="+", default=[2,4,6])
    parser.add_argument("--hs", type=float, nargs="+", default=[.45,.35,.25,.15])
    parser.add_argument("--domain-files", type=Path, nargs="+", default=None)
    parser.add_argument("--stencil", type=str, default="rbf", choices=("wls", "rbf"))
    parser.add_argument(
        "--linear-solver",
        type=str,
        default="scipy_gmres_ilu",
        choices=("gmres_ilu", "scipy_gmres_ilu", "scipy_sparse_direct", "dense", "lineax", "cudss"),
    )
    parser.add_argument("--setup-device", choices=("default", "cpu"), default="default")
    parser.add_argument("--sparse-assembly-chunk-size", type=int, default=None)
    parser.add_argument("--timing-warmup", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--mapped-cloud-preferred",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the mapped closed-curve node generator for smooth 2D geometries.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mat-output", type=Path, default=DEFAULT_MAT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    geometry_name = args.geometry or default_geometry(args.dim)
    if args.domain_files:
        domain_entries = [load_domain_file(path) for path in args.domain_files]
    else:
        validate_geometry(args.dim, geometry_name)
        hs = args.hs or [default_h(args.dim)]
        domain_entries = []
        for h in hs:
            domain = build_forward_domain(
                args.dim,
                geometry_name,
                h,
                args.seed,
                mapped_cloud_preferred=args.mapped_cloud_preferred,
            )
            metadata = SavedDomainMetadata(
                dim=args.dim,
                geometry=geometry_name,
                h=float(h),
                seed=args.seed,
                sep_rad=float(domain.get_sep_rad()),
                label=f"{geometry_name}_d{args.dim}_h{h:.5f}_seed{args.seed}",
            )
            domain_entries.append((domain, metadata))

    results: list[ForwardResult] = []
    for domain, metadata in domain_entries:
        orders = args.orders or ([4] if metadata.dim == 2 else [3])
        for order in orders:
            print(
                "[case] "
                f"family={args.family}, geometry={metadata.geometry}, resolution={metadata.label}, "
                f"order={order}, h={metadata.h:.5f}, nodes={domain.get_num_int_bdry_nodes()}, "
                f"stencil={args.stencil}, linear_solver={args.linear_solver}",
                flush=True,
            )
            results.append(
                run_case(
                    domain=domain,
                    metadata=metadata,
                    family=args.family,
                    order=order,
                    stencil=args.stencil,
                    linear_solver=args.linear_solver,
                    setup_device=args.setup_device,
                    sparse_assembly_chunk_size=args.sparse_assembly_chunk_size,
                    timing_warmup=args.timing_warmup,
                )
            )
            row = results[-1]
            print(
                "[case-done] "
                f"family={args.family}, resolution={row.resolution}, order={row.order}, "
                f"mean_relative_l2={row.mean_relative_l2_error:.6e}, max_relative_l2={row.max_relative_l2_error:.6e}, "
                f"preprocess_seconds={row.preprocess_elapsed_seconds:.3f}, seconds={row.elapsed_seconds:.3f}",
                flush=True,
            )

    dim = domain_entries[0][1].dim if domain_entries else args.dim
    geometry = domain_entries[0][1].geometry if domain_entries else geometry_name
    markdown = render_markdown(
        results,
        dim=dim,
        geometry=geometry,
        family=args.family,
        stencil=args.stencil,
        linear_solver=args.linear_solver,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_forward_mat(
        results,
        dim=dim,
        geometry=geometry,
        family=args.family,
        stencil=args.stencil,
        linear_solver=args.linear_solver,
        output_path=args.mat_output,
    )
    print(markdown, end="")
    print(f"wrote matlab output to {args.mat_output}")


if __name__ == "__main__":
    main()
