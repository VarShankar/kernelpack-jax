#!/usr/bin/env python3
"""GPU studies that replace the MATLAB/CPU comparisons of the FastOL paper.

Subcommands
-----------
lagrange CASE N
    For the unaugmented operator kernel of Section 3 (fixed shape at the five density budgets and at the
    validated radius, and the fitted shape at its validated radius), evaluates the same models directly
    through coefficients and through operator Lagrange weights, with the local reconstruction of the paper
    and with a global divergence-free C4 Matern reconstruction. Records test errors at the nodes, the
    spatial amplification factors, and the solve residuals. One JSON line per model.
spatial CASE
    Reconstructs the campaign's test predictions at the nodes with several spatial reconstructions and
    measures the error at held-out mesh points between the nodes (global vs local, PHS vs multiscale Matern),
    selects c_sigma for the multiscale kernel on validation fields, and records binned structure functions.

Everything runs on the GPU in double precision with KernelPack-JAX; the operator models are rebuilt from the
frozen campaign selections (Outputs_and_results/fastol_matched_safe_gpu_20261016) and checked against the
campaign's stored predictions before anything is measured.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jsl

CAMPAIGN = Path(os.environ.get("FASTOL_CAMPAIGN_DIR",
                               "/mnt/e/kernelpack-jax/Outputs_and_results/fastol_matched_safe_gpu_20261016"))
DATA_ROOT = Path(os.environ.get("FASTOL_DATA_ROOT", "/mnt/e/fluid_datasets"))
GLOBAL_EP = {  # spatial shape parameter of the global divergence-free C4 Matern reconstruction (PPKM settings)
    "lid_cavity_flow": 0.3, "flow_cylinder_laminar": 0.0636, "flow_cylinder_shedding": 0.0636,
    "backward_facing_step": 0.06, "buoyancy_cavity_flow": 0.8, "merge_vortices_easier": 0.02,
    "taylor_green": 0.02, "taylor_green_time": 1.5, "species_transport": 11.0, "forced_turb": 0.023,
}
PERIODIC = {"taylor_green", "taylor_green_time", "merge_vortices_easier", "forced_turb"}
BLOCK_BYTES = 300_000_000


def log(**kw):
    print(json.dumps(dict(t=time.strftime("%H:%M:%S"), **kw)), flush=True)


def cell(case, n):
    return CAMPAIGN / "cells" / f"{case}_N{n}"


def load(case, ids, protocol, device, *, all_points=False):
    from kernelpack.operator_learning.datasets import load_fastol_paper_dataset, read_fastol_field_gpu, _FILES
    pres = protocol["prescribed"]["sparse_c4_wendland"]
    m = int(pres["num_spatial_points"])
    out_pts = m
    if all_points:
        folder, data_name, fekete_name, *_ = _FILES[case]
        fek = DATA_ROOT / folder / fekete_name
        if not fek.exists():
            fek = DATA_ROOT / case / fekete_name
        out_pts = int(read_fastol_field_gpu(fek, "E", device=device).size)
    data, actual, _ = load_fastol_paper_dataset(DATA_ROOT, case, list(ids), output_points=out_pts,
                                                input_points=int(pres["input_spatial_points"]),
                                                representation=pres["input_representation"], device=device)
    return data.inputs, data.points, data.velocity, m


def dim_of(v):
    return v.shape[-1]


def field_error(prediction, truth):
    axes = (1, 3) if truth.ndim == 4 else (1, 2)
    num = jnp.linalg.norm(prediction - truth, axis=axes)
    den = jnp.linalg.norm(truth, axis=axes)
    return float(jnp.mean(num / jnp.where(den == 0, 1.0, den)))


# ----------------------------------------------------------------------------------------------------------
# operator models rebuilt from the frozen campaign
# ----------------------------------------------------------------------------------------------------------
def build_fits(case, x, y, training_ids, device):
    from kernelpack.operator_learning import KernelTrainingConfig
    from kernelpack.operator_learning.trainable import train_compact_kernel
    scaling = "taylor_green_ray" if case in ("taylor_green", "taylor_green_time") else "none"
    base = dict(training_steps=60, validation_steps=35, cudss_reordering="default",
                validation_start_policy="common_budget", response_scaling=scaling)
    anchor = train_compact_kernel(x, y, training_ids=jnp.asarray(training_ids, device=device), radius=None,
                                  nugget=1e-6, config=KernelTrainingConfig(mean_dimension=15, **base), optimize=False)
    nopoly = train_compact_kernel(x, y, training_ids=jnp.asarray(training_ids, device=device), radius=None,
                                  nugget=1e-6, config=KernelTrainingConfig(mean_dimension=0, **base), optimize=False,
                                  geometry=anchor)
    return anchor, nopoly


def model_list(case, n):
    """(label, theta) for the unaugmented models: five budget radii and the two validated selections."""
    sel = json.loads((cell(case, n) / "selection.json").read_text())["models"]
    fixed = sel["fixed_no_poly"]
    out = []
    for c in fixed["candidates"]:
        if c["origin"] == "training_distance_budget":
            out.append((f"budget_{c['density']:.2f}", c["parameters"], c["density"]))
    out.append(("fixed_selected", fixed["parameters"], fixed["training_kernel_density"]))
    tr = sel["trainable_no_poly"]
    out.append(("fitted_selected", tr["parameters"], tr["training_kernel_density"]))
    return out


def pad_theta(theta, fit):
    t = np.zeros(fit.initial.shape)
    t[: len(theta)] = theta
    return jnp.asarray(t)


class Operator:
    """Lagrange weights and coefficient solves of one frozen unaugmented operator, with the
    Taylor--Green amplitude normalization applied explicitly to both evaluation routes."""

    def __init__(self, fit, theta):
        from kernelpack.operator_learning.trainable import freeze_compact_operator
        self.op = freeze_compact_operator(fit, pad_theta(theta, fit), rhs_block_size=4096)
        self.scaled = hasattr(self.op, "ray_direction") and self.op.ray_direction is not None and hasattr(self.op, "operator")
        self.base = self.op.operator if self.scaled else self.op
        self.response_scale = getattr(fit, "response_scale", None)

    def query_scale(self, q):
        if not self.scaled:
            return jnp.ones(len(q))
        st = self.op.device_state()
        b = st["operator"]
        nq = jnp.where(b["zero"], 0.0, (q - b["mean"]) / b["std_safe"])
        amp = (nq - st["ray_zero"]) @ st["ray_direction"]
        return amp / st["ray_median"]

    def train_scale(self):
        return jnp.ones(len(self.base.inputs)) if not self.scaled else self.response_scale

    def weights(self, q):
        """L (queries x N), including the amplitude normalization: prediction = L @ (training targets)."""
        return self.op.lagrange_map(q)

    def direct(self, q, targets):
        """K_* (K^{-1} targets'), with the same normalization: targets' = targets / train_scale."""
        t = targets / self.train_scale()[:, None]
        c = self.base.solve(t)
        return self.query_scale(q)[:, None] * (self.base.kernel_matrix(q) @ c)

    def close(self):
        self.op.close()


# ----------------------------------------------------------------------------------------------------------
# reconstructions
# ----------------------------------------------------------------------------------------------------------
def component_major(v):
    """(F, m, d) -> (F, d*m); (F, m, T, d) -> (F, T*d*m) with time-major blocks."""
    if v.ndim == 3:
        return jnp.transpose(v, (0, 2, 1)).reshape(v.shape[0], -1)
    return jnp.transpose(v, (0, 2, 3, 1)).reshape(v.shape[0], -1)


def unique_periodic(points, period=2 * math.pi):
    p = np.asarray(jax.device_get(points))
    key = np.round(np.mod(p, period) / 1e-9).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    return np.sort(first)


class GlobalRecon:
    """Global divergence-free C4 Matern interpolant on the nodes (PPKM), applied per sampled time."""

    def __init__(self, nodes, ep):
        from kernelpack.divfree.matern import divfree_matern_gram_matrix
        self.nodes = nodes
        self.ep = ep
        self.gram = divfree_matern_gram_matrix(nodes, nodes, jnp.array([ep]), jnp.array([1.0]))
        self.lu = jsl.lu_factor(self.gram)
        self.dm = self.gram.shape[0]

    def coefficients(self, vals, chunk=1000):
        """vals: (F, T*dm) -> coefficients (F, T*dm), solving per time block, in chunks of fields."""
        out = []
        for s in range(0, vals.shape[0], chunk):
            v = vals[s:s + chunk]
            f = v.shape[0]
            blocks = v.reshape(f, -1, self.dm)
            c = jsl.lu_solve(self.lu, jnp.transpose(blocks, (2, 0, 1)).reshape(self.dm, -1))
            out.append(jnp.transpose(c.reshape(self.dm, f, -1), (1, 2, 0)).reshape(f, -1))
        return jnp.concatenate(out, axis=0) if len(out) > 1 else out[0]

    def evaluate_nodes(self, coeff, chunk=1000):
        out = []
        for s in range(0, coeff.shape[0], chunk):
            c = coeff[s:s + chunk]
            f = c.shape[0]
            out.append(jnp.einsum("ftk,jk->ftj", c.reshape(f, -1, self.dm), self.gram).reshape(f, -1))
        return jnp.concatenate(out, axis=0) if len(out) > 1 else out[0]

    def evaluate_at(self, coeff, points):
        from kernelpack.divfree.matern import divfree_matern_gram_matrix
        e = divfree_matern_gram_matrix(points, self.nodes, jnp.array([self.ep]), jnp.array([1.0]))
        f = coeff.shape[0]
        c = coeff.reshape(f, -1, self.dm)
        return jnp.einsum("ftk,jk->ftj", c, e).reshape(f, -1), e

    def norm_eval(self, nt):
        return float(jnp.linalg.norm(self.gram)) * math.sqrt(nt)


def decoder(points, options):
    from kernelpack.operator_learning.local_dfphs import LocalDFPHSDecoder
    return LocalDFPHSDecoder(points, **options)


# ----------------------------------------------------------------------------------------------------------
# lagrange study
# ----------------------------------------------------------------------------------------------------------
def run_lagrange(case, n, out):
    device = jax.devices("gpu")[0]
    proto = json.loads((cell(case, n) / "protocol.json").read_text())
    tr, te = proto["training_ids"], proto["test_ids"]
    x, pts, v, m = load(case, tr + te, proto, device)
    xt, xq, vt, vq = x[:n], x[n:], v[:n], v[n:]
    log(stage="loaded", case=case, N=n, x=list(x.shape), v=list(v.shape))
    # The operator fit needs the targets only for their shape (and the Taylor--Green scaling, 2D); keep a slice for
    # the 3D data sets so that the training velocities are not duplicated on the GPU.
    anchor, fit = build_fits(case, xt, vt[:, :8] if dim_of(vt) == 3 else vt, tr, device)
    nt = vt.shape[2] if vt.ndim == 4 else 1
    dim = vt.shape[-1]

    # local reconstruction blocks (training coefficients built per stencil block on demand)
    opts = dict(proto["spatial_options"])
    dec = decoder(pts, opts)
    centers = jnp.arange(m)
    width_probe = dec.fit_block(vt[:2], pts[:1], center_indices=centers[:1])
    width = width_probe.coefficients.shape[-1]
    per_block = max(1, int(BLOCK_BYTES // (8 * n * width)))
    blocks = [(s, min(m, s + per_block)) for s in range(0, m, per_block)]

    models = model_list(case, n)
    recs = {label: dict(case=case, N=n, model=label, density=density, theta=list(map(float, theta)),
                        global_ep=GLOBAL_EP[case]) for label, theta, density in models}

    # ---- phase 1: global reconstruction (duplicate periodic nodes removed), all models, then freed
    keep = unique_periodic(pts) if case in PERIODIC else np.arange(m)
    g = GlobalRecon(pts[keep], GLOBAL_EP[case])
    vq_g = vq[:, keep] if case in PERIODIC else vq
    Vt_g = component_major(vt[:, keep] if case in PERIODIC else vt)
    Bg = g.coefficients(Vt_g)                                        # (N, T*dm)
    g.lu = None                                                      # the factorization is not needed again
    recon_g = g.evaluate_nodes(Bg)
    alpha_g = float(jnp.linalg.norm(Bg)) * g.norm_eval(nt) / float(jnp.linalg.norm(recon_g))
    interp_g = float(jnp.linalg.norm(recon_g - Vt_g) / jnp.linalg.norm(Vt_g))
    del recon_g
    log(stage="global_ready", dm=g.dm, alpha=alpha_g, interp_rel=interp_g)

    def unflat(p):
        f = p.shape[0]
        if vq_g.ndim == 4:
            return jnp.transpose(p.reshape(f, nt, dim, -1), (0, 3, 1, 2))
        return jnp.transpose(p.reshape(f, dim, -1), (0, 2, 1))

    for label, theta, _ in models:
        t0 = time.perf_counter()
        op = Operator(fit, theta)
        L = op.weights(xq)                                             # (100, N)
        pred_dir = g.evaluate_nodes(op.direct(xq, Bg))
        pred_lag = g.evaluate_nodes(L @ Bg)
        pred_samp_dir = op.direct(xq, Vt_g)                            # direct on sampled velocities (control)
        recs[label].update(alpha_global=alpha_g, global_dm=g.dm, global_interp_rel=interp_g,
                           global_direct=field_error(unflat(pred_dir), vq_g),
                           global_lagrange=field_error(unflat(pred_lag), vq_g),
                           global_lagrange_samples=field_error(unflat(L @ Vt_g), vq_g),
                           global_direct_samples=field_error(unflat(pred_samp_dir), vq_g),
                           global_coeff_norm=float(jnp.linalg.norm(Bg)),
                           seconds_global=time.perf_counter() - t0)
        del pred_dir, pred_lag, pred_samp_dir, L
        op.close()
        log(stage="global_model_done", model=label, direct=recs[label]["global_direct"],
            lagrange=recs[label]["global_lagrange"])
    dm_global = g.dm
    del g, Bg, Vt_g, vq_g

    # ---- phase 2: local reconstruction, all models
    num2, den2 = 0.0, 0.0
    for s, e in blocks:
        blk = dec.fit_block(vt, pts[s:e], center_indices=centers[s:e])
        b2 = jnp.sum(blk.coefficients ** 2, axis=(1, 2))
        a2 = jnp.sum(blk.evaluation ** 2, axis=(1, 2))
        num2 += float(jnp.sum(b2 * a2))
        den2 += float(jnp.sum(blk.reconstruct(blk.coefficients) ** 2))
        del blk
    alpha_l = math.sqrt(num2) / math.sqrt(den2)
    log(stage="local_ready", blocks=len(blocks), width=width, alpha=alpha_l)

    truth_l = vq
    for label, theta, _ in models:
        t0 = time.perf_counter()
        op = Operator(fit, theta)
        L = op.weights(xq)
        pd = jnp.zeros((len(xq),) + truth_l.shape[1:])
        pl = jnp.zeros_like(pd)
        for s, e in blocks:
            blk = dec.fit_block(vt, pts[s:e], center_indices=centers[s:e])
            b = jnp.transpose(blk.coefficients, (1, 0, 2)).reshape(n, -1)     # (N, stencils*width)
            for which, c in (("d", op.direct(xq, b)), ("l", L @ b)):
                c = c.reshape(len(xq), e - s, width)
                r = blk.reconstruct(jnp.transpose(c, (1, 0, 2)))              # (queries, stencils, T*d)
                r = r.reshape(len(xq), e - s, nt, dim) if truth_l.ndim == 4 else r.reshape(len(xq), e - s, dim)
                if which == "d":
                    pd = pd.at[:, s:e].set(r)
                else:
                    pl = pl.at[:, s:e].set(r)
            del blk, b
        rec = recs[label]
        rec.update(alpha_local=alpha_l, global_dm=dm_global,
                   local_direct=field_error(pd, truth_l), local_lagrange=field_error(pl, truth_l),
                   local_lagrange_samples=field_error(jnp.einsum("qi,i...->q...", L, vt), truth_l),
                   seconds_local=time.perf_counter() - t0)
        rec["seconds"] = rec["seconds_global"] + rec["seconds_local"]
        op.close()
        verify_against_campaign(case, n, label, op, xq, vt, L, truth_l)
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        log(stage="model_done", **{k: rec[k] for k in ("model", "global_direct", "global_lagrange", "local_direct",
                                                       "local_lagrange", "seconds")})


def verify_against_campaign(case, n, label, op, xq, vt, L, truth_l):
    """The rebuilt fixed/fitted selections must reproduce the campaign's stored nodal predictions."""
    name = {"fixed_selected": "fixed_no_poly", "fitted_selected": "trainable_no_poly"}.get(label)
    if name is None:
        return
    z = np.load(cell(case, n) / f"{name}_predictions.npz")
    ours = np.asarray(jax.device_get(jnp.einsum("qi,i...->q...", L, vt)))
    ref = z["prediction"]
    gap = float(np.linalg.norm(ours - ref) / np.linalg.norm(ref))
    log(stage="campaign_check", model=label, relative_gap=gap)
    if not gap < 1e-6:
        raise RuntimeError(f"Rebuilt {name} disagrees with the campaign predictions: {gap:.3e}")


# ----------------------------------------------------------------------------------------------------------
# spatial study
# ----------------------------------------------------------------------------------------------------------
def best_model(case, n):
    sel = json.loads((cell(case, n) / "selection.json").read_text())["models"]
    return min(sel, key=lambda k: sel[k]["validation_error"])


def structure_function(points, fields, periodic, rng, anchors=256, partners=4, bins=12):
    p = np.asarray(jax.device_get(points))
    d = np.linalg.norm(p[:, None] - p[None], axis=-1) if not periodic else np.linalg.norm(
        (lambda z: z - 2 * np.pi * np.round(z / (2 * np.pi)))(p[:, None] - p[None]), axis=-1)
    lo, hi = np.percentile(d[d > 0], 0.5), np.percentile(d[d > 0], 60)
    edges = np.geomspace(lo, hi, bins + 1)
    a = rng.choice(len(p), anchors, replace=False)
    pairs = []
    for i in a:
        for k in range(bins):
            cand = np.where((d[i] >= edges[k]) & (d[i] < edges[k + 1]))[0]
            for j in rng.permutation(cand)[:partners]:
                pairs.append((i, j, k))
    pairs = np.array(pairs)
    out = {}
    for name, f in fields.items():
        f = np.asarray(jax.device_get(f))
        inc = np.sum((f[:, pairs[:, 0]] - f[:, pairs[:, 1]]) ** 2, axis=-1).mean(axis=0)
        out[name] = [float(inc[pairs[:, 2] == k].mean()) for k in range(bins)]
    out["r"] = [float(np.sqrt(edges[k] * edges[k + 1])) for k in range(bins)]
    return out


def run_spatial(case, out, n=None):
    device = jax.devices("gpu")[0]
    n = n or 10000
    proto = json.loads((cell(case, n) / "protocol.json").read_text())
    te = proto["test_ids"]
    _, pts_all, vq_all, m = load(case, te, proto, device, all_points=True)
    nodes, held = pts_all[:m], pts_all[m:]
    if len(held) == 0:
        log(stage="no_held_out_points", case=case)
        return
    truth_held = vq_all[:, m:]
    name = best_model(case, n)
    pred = jnp.asarray(np.load(cell(case, n) / f"{name}_predictions.npz")["prediction"])
    rec = dict(case=case, N=n, model=name, m=m, held_out=len(held))
    opts = dict(proto["spatial_options"])
    # local reconstruction(s) of the predicted nodal values, evaluated between the nodes
    variants = {"local_default": opts}
    if case in ("species_transport", "forced_turb"):
        phs = {k: v for k, v in opts.items() if not k.startswith("matern") and k != "spatial_kernel"}
        variants = {"local_phs": phs, "local_multiscale": opts}
    fields = {"reference": truth_held}
    for vname, o in variants.items():
        dec = decoder(nodes, o)
        r = reconstruct_between(dec, pred, held)
        rec[vname] = field_error(r, truth_held)
        fields[vname] = r
        log(stage="local_done", variant=vname, error=rec[vname])
    if True:  # global divergence-free Matern reconstruction, all ten benchmarks
        keep = unique_periodic(nodes) if case in PERIODIC else np.arange(m)
        g = GlobalRecon(nodes[keep], GLOBAL_EP[case])
        coeff = g.coefficients(component_major(pred[:, keep]))
        val, _ = g.evaluate_at(coeff, held)
        nt = pred.shape[2] if pred.ndim == 4 else 1
        dim = pred.shape[-1]
        f = val.shape[0]
        r = (jnp.transpose(val.reshape(f, nt, dim, -1), (0, 3, 1, 2)) if pred.ndim == 4
             else jnp.transpose(val.reshape(f, dim, -1), (0, 2, 1)))
        rec["global"] = field_error(r, truth_held)
        rec["global_unknowns"] = g.dm
        log(stage="global_done", error=rec["global"])
    if case in ("species_transport", "forced_turb"):
        rng = np.random.default_rng(0)
        rec["structure_functions"] = structure_function(held, fields, case == "forced_turb", rng)
        rec["c_sigma_validation"] = select_c_sigma(case, proto, device, m)
    with open(out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def reconstruct_between(dec, nodal, queries, batch=512):
    parts = []
    for s in range(0, len(queries), batch):
        q = queries[s:s + batch]
        blk = dec.fit_block(nodal, q)
        r = blk.reconstruct(blk.coefficients)                      # (fields, queries, T*d)
        parts.append(r)
    r = jnp.concatenate(parts, axis=1)
    if nodal.ndim == 4:
        return r.reshape(r.shape[0], r.shape[1], nodal.shape[2], nodal.shape[3])
    return r


def select_c_sigma(case, proto, device, m):
    """Multiscale scale factor chosen on the 100 validation fields from their exact nodal values."""
    va = proto["validation_ids"]
    _, pts_all, vv, _ = load(case, va, proto, device, all_points=True)
    nodes, held = pts_all[:m], pts_all[m:]
    res = {}
    for c in (2.0, 4.0, 8.0, 16.0):
        o = dict(proto["spatial_options"], matern_base_length_factor=c)
        res[str(c)] = field_error(reconstruct_between(decoder(nodes, o), vv[:, :m], held), vv[:, m:])
        log(stage="c_sigma", c=c, error=res[str(c)])
    phs = {k: v for k, v in proto["spatial_options"].items() if not k.startswith("matern") and k != "spatial_kernel"}
    res["phs"] = field_error(reconstruct_between(decoder(nodes, phs), vv[:, :m], held), vv[:, m:])
    return res


def run_ranks(case, out, n=500):
    """Retained divergence-free polynomial ranks of the local systems on the nodes (degree four, periodic images)."""
    device = jax.devices("gpu")[0]
    proto = json.loads((cell(case, n) / "protocol.json").read_text())
    _, pts, v, m = load(case, proto["test_ids"][:2], proto, device)
    dec = decoder(pts, dict(proto["spatial_options"]))
    centers = jnp.arange(m)
    n_full = {2: 20, 3: 85}[pts.shape[1]]
    counts, nonfinite = [], 0
    for s in range(0, m, 512):
        e = min(m, s + 512)
        _, _, systems, _, _, _ = dec._prepare_block(pts[s:e], v, center_indices=centers[s:e])
        tail = jnp.diagonal(systems[:, -n_full:, -n_full:], axis1=1, axis2=2)
        counts.append(np.asarray(jax.device_get(n_full - jnp.sum(tail == 1.0, axis=1))))
        # stencils whose batched LU solve is not finite, i.e., where the pseudoinverse fallback would be used
        lu, piv = jax.vmap(jax.scipy.linalg.lu_factor)(systems)
        x = jax.vmap(lambda a, p, b: jax.scipy.linalg.lu_solve((a, p), b))(lu, piv, jnp.ones(systems.shape[:2]))
        nonfinite += int(jnp.sum(~jnp.all(jnp.isfinite(x), axis=1)))
    r = np.concatenate(counts)
    rec = dict(case=case, stencils=int(m), full_rank=n_full, min_rank=int(r.min()), max_rank=int(r.max()),
               truncated=int(np.sum(r < n_full)), nonfinite_lu=nonfinite)
    with open(out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    log(stage="ranks", **rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study", choices=["lagrange", "spatial", "ranks"])
    ap.add_argument("case")
    ap.add_argument("N", nargs="?", type=int)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.study == "lagrange":
        run_lagrange(a.case, a.N, a.out)
    elif a.study == "ranks":
        run_ranks(a.case, a.out)
    else:
        run_spatial(a.case, a.out, a.N)


if __name__ == "__main__":
    main()
