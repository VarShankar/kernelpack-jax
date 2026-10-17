"""Numerical check of the elliptic error estimate (paper, Section analysis).

For the Darcy problem with the flux (Neumann) condition we assemble the ghost-closed system exactly as
run_cudss_paper_forward.py does (PDE rows at the physical nodes, normal-derivative rows at the boundary nodes,
bordered by the constant), and record, for every geometry, h, order, and family:
  tau  = [A; B] Pi u - [f; g/K]   (truncation error of the PDE and boundary rows; Pi u = u* at all nodes),
  E    = Pi u - U, with U shifted so that its mean over the physical nodes equals that of u*,
  kappa = ||E_phys|| / ||tau||   (effective stability constant),
  lam_L = max_c h^2 sum_k |L_ck|, lam_D = max_c h sum_k |D_a,ck| (scaled stencil sums).
Output: Outputs_and_results/paper_forward_v4_20261011/estimate_check.jsonl
"""
import argparse
import gc
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from kernelpack.accelerators.jax_spatial import enable_compilation_cache
from kernelpack.solvers._common import sparse_matvec
from run_cudss_paper_forward import normal_derivative

enable_compilation_cache()
import darcy_forward_convergence_study as darcy
from darcy_inverse_convergence_study import manufactured_state_1

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "Outputs_and_results/paper_forward_v4_20261011/final_seed11"
REJECTED = {("h_channel", .16, 6)}


def row_abs_sums(sp, n):
    idx = np.asarray(sp.indices)
    return np.bincount(idx[:, 0], weights=np.abs(np.asarray(sp.values)), minlength=n)[:n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--geometries", nargs="+", default=["h_channel", "treasure_chest"])
    p.add_argument("--h", nargs="+", type=float, default=[.16, .12, .08, .06, .04])
    p.add_argument("--orders", nargs="+", type=int, default=[2, 4, 6])
    p.add_argument("--out", type=Path, default=RUN.parent / "estimate_check.jsonl")
    p.add_argument("--domain-dir", type=Path, default=None, help="Clouds {geometry}_h{h:.2f}_s11.npz (default: the run's domains)")
    p.add_argument("--boundary-aware-stencils", action="store_true")
    p.add_argument("--skip-rejected", nargs="*", default=None,
                   help="geometry:h:order triples to skip (default: the Laplacian rejections of the Euclidean run)")
    a = p.parse_args()
    done = set()
    if a.out.exists():
        for line in a.out.read_text().splitlines():
            r = json.loads(line); done.add((r["geometry"], r["h"], r["order"], r["family"]))
    for geometry in a.geometries:
        for h in a.h:
            path = (a.domain_dir / f"{geometry}_h{h:.2f}_s11.npz") if a.domain_dir else (RUN / "domains" / f"{geometry}_h{h:g}.npz")
            domain = darcy.load_domain_file(path)[0]
            if a.boundary_aware_stencils:
                import stencil_level_set
                stencil_level_set.attach(domain, geometry, h)
            xf = domain.get_all_nodes(); x = domain.get_int_bdry_nodes(); xb = domain.get_bdry_nodes()
            n, dim = len(x), x.shape[1]
            for order in a.orders:
                rejected = REJECTED if a.skip_rejected is None else {(g, float(hh), int(o)) for g, hh, o in (t.split(":") for t in a.skip_rejected)}
                if (geometry, h, order) in rejected:
                    continue
                if all((geometry, h, order, f) in done for f in darcy.FAMILIES):
                    continue
                solver = darcy.build_solver(domain, order, "rbf", "cudss", 128)
                solver.prepare_boundary_operator(jnp.ones(len(xb)), jnp.zeros(len(xb)))   # pure Neumann
                lam_L = float(np.max(row_abs_sums(solver.lap_sparse, n)) * h ** 2)
                grads = solver.grad_sparse if isinstance(solver.grad_sparse, (list, tuple)) else [solver.grad_sparse]
                lam_D = float(max(np.max(row_abs_sums(g, n)) for g in grads) * h)
                frozen = solver.freeze()
                pi_u = manufactured_state_1(xf)
                for family in darcy.FAMILIES:
                    if (geometry, h, order, family) in done:
                        continue
                    theta = darcy.sample_theta(family, dim)
                    case = darcy.build_cases(family, theta, x, xb)[0]
                    coeff = darcy.coefficient_field(family, theta, xf)
                    flux_over_k = normal_derivative(manufactured_state_1, xb, domain.get_nrmls())
                    out = frozen.solve_from_fields(coeff, case.forcing, flux_over_k)
                    K = out["system_matrix_sparse"]
                    nf = len(pi_u)
                    rows = K.indices[:, 0] < nf                     # drop the bordering row and column
                    cols = K.indices[:, 1] < nf
                    keep = rows & cols
                    tau = sparse_matvec(K.indices[keep], K.values[keep], nf, pi_u) - out["rhs"][:nf]
                    U = out["full_state"][:nf]
                    U = U + (jnp.mean(pi_u[:n]) - jnp.mean(U[:n]))
                    E = pi_u - U
                    tau_p, tau_bc = tau[:n], tau[n:]
                    rec = dict(geometry=geometry, h=h, order=order, family=family, physical_nodes=n,
                               tau_phys_l2=float(jnp.linalg.norm(tau_p)), tau_bc_l2=float(jnp.linalg.norm(tau_bc)),
                               tau_phys_max=float(jnp.max(jnp.abs(tau_p))),
                               err_phys_l2=float(jnp.linalg.norm(E[:n])), err_ghost_l2=float(jnp.linalg.norm(E[n:])),
                               u_phys_l2=float(jnp.linalg.norm(pi_u[:n])),
                               kappa=float(jnp.linalg.norm(E[:n]) / jnp.linalg.norm(tau_p)),
                               lam_L=lam_L, lam_D=lam_D)
                    with a.out.open("a") as fh:
                        fh.write(json.dumps(rec) + "\n")
                    print(json.dumps(rec), flush=True)
                solver.close_cudss(); del solver, frozen
                jax.clear_caches(); gc.collect()


if __name__ == "__main__":
    main()
