"""Rebuild manuscript figures/tables from the frozen, audited GPU scalar reports.

This is reporting only: it never imports JAX, opens datasets or initializes CUDA.
Run: python build_gpu_results.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, LogLocator

BASE = Path(__file__).resolve().parent
SOURCE = BASE
(SOURCE / "sections").mkdir(exist_ok=True)
# Campaign report written by evidence/gpu/final_completion_audit.py; its tables are copied into gpu_data/.
REPORT = Path(os.environ.get("FASTOL_REPORT_DIR",
                             r"E:\kernelpack-jax\Outputs_and_results\fastol_matched_safe_gpu_20261016_report"))
DATA = SOURCE / "gpu_data"
FIG = SOURCE / "gpu_figures"
DATA.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

INPUTS = ["final_validated_models.csv", "final_paired_comparisons.csv",
          "final_shared_spatial_preparation.csv", "completion_audit.json"]
for name in INPUTS:
    destination = DATA / name
    if REPORT.exists():
        shutil.copy2(REPORT / name, destination)
    if not destination.exists():
        raise FileNotFoundError(destination)

audit = json.loads((DATA / "completion_audit.json").read_text())
assert audit["passed"] and audit["completed_cells"] == 40 and audit["checked_model_rows"] == 160
rows = list(csv.DictReader((DATA / "final_validated_models.csv").open(newline="")))
assert len(rows) == 160
models = ["fixed_no_poly", "trainable_no_poly", "fixed_affine", "trainable_affine"]
NVALUES = [500, 1000, 5000, 10000]
CASES = [
    ("forced_turb", "Forced isotropic turbulence"),
    ("species_transport", "Turbulent species transport"),
    ("merge_vortices_easier", "Merging vortices"),
    ("lid_cavity_flow", "Lid-driven cavity"),
    ("flow_cylinder_laminar", "Cylinder flow (no shedding)"),
    ("flow_cylinder_shedding", "Cylinder flow (vortex shedding)"),
    ("taylor_green", "Taylor--Green vortices (space)"),
    ("taylor_green_time", "Taylor--Green vortices (spacetime)"),
    ("backward_facing_step", "Backward-facing step"),
    ("buoyancy_cavity_flow", "Buoyancy-driven cavity"),
]
lookup = {(row["case"], int(row["N"]), row["model"]): row for row in rows}
assert len(lookup) == 160

# Cost-only replay with the fill-reducing training order (scripts/fastol_gpu_cost_replay.py):
# replace the cost columns; accuracy columns stay from the campaign. The replay checks that
# every prediction matches the campaign's saved prediction to rounding.
REPLAY = DATA / "cost_replay.jsonl"
if REPLAY.exists():
    replayed = [json.loads(line) for line in REPLAY.read_text().splitlines() if line.startswith("{")]
    assert len(replayed) == 160, len(replayed)
    for r in replayed:
        row = lookup[r["case"], int(r["N"]), r["model"]]
        assert r["prediction_rel_diff_vs_campaign"] < 1e-8, r   # the campaign's own consistency tolerance
        assert int(r["actual_kernel_nnz"]) == int(row["actual_kernel_nnz"]), r
        durations = r["warm_100query_seconds"]
        row.update(native_factor_nnz=str(r["native_factor_nnz"]),
                   native_permanent_device_ESTIMATE_bytes=str(r["native_permanent_device_bytes"]),
                   visible_model_array_bytes=str(r["visible_model_array_bytes"]),
                   native_owned_CSR_requested_bytes=str(r["owned_csr_bytes"]),
                   native_RHS_solution_requested_bytes=str(r["rhs_solution_bytes"]),
                   accounted_resident_storage_ESTIMATE_bytes=str(r["accounted_resident_storage_ESTIMATE_bytes"]),
                   accounted_resident_storage_ESTIMATE_decimal_GB=str(r["accounted_resident_storage_ESTIMATE_bytes"] / 1e9),
                   warm_100query_min_seconds=str(min(durations)), warm_100query_median_seconds=str(r["warm_100query_median_seconds"]),
                   warm_100query_max_seconds=str(max(durations)), operator_setup_seconds=str(r["operator_setup_seconds"]),
                   observed_cudss_reordering=r["cudss_reordering"], training_input_order=r["input_order"])
    with (DATA / "final_validated_models_reordered.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
for case, _ in CASES:
    for n in NVALUES:
        for model in models:
            assert (case, n, model) in lookup
        fixed = lookup[case, n, "fixed_affine"]
        learned = lookup[case, n, "trainable_affine"]
        assert fixed["mean_basis_sha256"] == learned["mean_basis_sha256"]
for row in rows:
    assert row["gpu_transfer_guard_passed"] == "True"
    assert row["observed_cudss_reordering"] in (("amd", "default", "none") if REPLAY.exists() else ("amd",))
    assert int(row["graph_replays_during_timing"]) >= 44
    assert abs(float(row["kernel_density"]) - int(row["actual_kernel_nnz"]) / int(row["N"]) ** 2) < 1e-14

# Each PDF is 6.3 inches wide, matching the manuscript's printable figure width.
# Every actual axes box has the same width/height, including the appendix panels.
# Axis titles and ticks are 14 pt, bold, in the exported PDF (no LaTeX raster text).
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 14,
    "axes.labelsize": 14, "axes.labelweight": "bold",
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 11.5, "axes.linewidth": 1.0,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 360,
})
STYLE = {
    "fixed_no_poly": ("#2471a3", "o", "-", "Fixed shape"),
    "trainable_no_poly": ("#ca6b15", "s", "-", "Fitted shape"),
    "fixed_affine": ("#2471a3", "^", "--", "Fixed shape + affine"),
    "trainable_affine": ("#ca6b15", "D", "--", "Fitted shape + affine"),
}
METRICS = {
    "error": ("test_error", 1., "Relative error", "log"),
    "density": ("kernel_density", 100., "Matrix density (%)", "linear"),
    "warm": ("warm_100query_median_seconds", 1000., "Time (ms)", "log"),
    "storage": ("accounted_resident_storage_ESTIMATE_decimal_GB", 1., "Storage (GB)", "log"),
}
outputs = []

def make_plot(metric, cases, suffix):
    key, multiplier, ylabel, scale = METRICS[metric]
    nrows = len(cases) // 2
    # The drawing width is deliberately not cropped: the PDF page and PNG have
    # identical physical dimensions, and the 14 pt source text is retained.
    width, height = 6.30, 0.67 + 0.83 + nrows * 1.55 + (nrows-1) * 0.89
    fig = plt.figure(figsize=(width, height))
    left, right = 0.89, 0.20
    bottom, top = 0.67, 0.83
    gapx, gapy = 0.90, 0.89
    axw = (width-left-right-gapx)/2
    axh = (height-bottom-top-gapy*(nrows-1))/nrows
    axes = []
    for i, (case, title) in enumerate(cases):
        col, row = i % 2, i // 2
        ax = fig.add_axes([(left+col*(axw+gapx))/width,
                           (height-top-(row+1)*axh-row*gapy)/height,
                           axw/width, axh/height])
        axes.append(ax)
        for model in models:
            color, marker, linestyle, label = STYLE[model]
            values = [float(lookup[case, n, model][key])*multiplier for n in NVALUES]
            ax.plot(NVALUES, values, color=color, marker=marker, linestyle=linestyle,
                    lw=1.7, ms=5.1, markerfacecolor="white", markeredgewidth=1.35,
                    label=label, clip_on=False)
        ax.set_xscale("log")
        ax.set_xlim(420, 11800)
        ax.xaxis.set_major_locator(FixedLocator(NVALUES))
        ax.xaxis.set_major_formatter(FixedFormatter(["0.5k", "1k", "5k", "10k"]))
        ax.minorticks_off()
        ax.set_yscale(scale)
        if metric == "density":
            ax.set_ylim(-2, 102)
            ax.set_yticks([0, 25, 50, 75, 100])
        else:
            ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.grid(True, color="#dcdcdc", lw=.65, axis="both")
        ax.set_title(title.replace("--", "–"), pad=8)
        ax.set_xlabel("Training functions, $N$", labelpad=5)
        ax.set_ylabel(ylabel, labelpad=5)
        for tick in ax.get_xticklabels()+ax.get_yticklabels():
            tick.set_fontweight("bold")
        ax.tick_params(axis="both", length=3.5, pad=3)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(.53, 1.002), frameon=False,
               handlelength=2.5, columnspacing=1.6, borderpad=0.)
    stem = FIG / f"gpu_{metric}_{suffix}"
    fig.savefig(stem.with_suffix(".pdf"), metadata={"Creator": "build_gpu_results.py", "CreationDate": None, "ModDate": None})
    fig.savefig(stem.with_suffix(".png"))
    plt.close(fig)
    outputs.extend([stem.with_suffix(".pdf"), stem.with_suffix(".png")])
    return {"metric": metric, "file": stem.name, "page_inches": [width, height],
            "axes_inches": [axw, axh], "label_and_tick_points": 14,
            "cases": [case for case, _ in cases]}

plot_dimensions = []
for metric in METRICS:
    plot_dimensions.append(make_plot(metric, CASES[:4], "main"))
    plot_dimensions.append(make_plot(metric, CASES[4:], "additional"))

def fnum(value, digits=3):
    v = float(value)
    if v == 0:
        return "0"
    exponent = int(f"{v:e}".split("e")[1])
    if exponent < -2 or exponent > 3:
        coefficient = v/(10.**exponent)
        return rf"${coefficient:.{digits-1}f}\times10^{{{exponent}}}$"
    return f"{v:.{digits}g}"

def bold(cell):
    """Bold one table cell (math or text)."""
    if cell.startswith("$") and cell.endswith("$") and cell.count("$") == 2:
        return r"$\mathbf{" + cell[1:-1] + "}$"
    return r"\textbf{" + cell + "}"

def bold_best(cells, values, best=min):
    """Bold every cell whose printed value equals that of the best value (ties at the printed precision)."""
    target = cells[values.index(best(values))]
    return [bold(c) if c == target else c for c in cells]

def metric_table(metric, filename, label, caption, multiplier=1., digits=3):
    lines = [r"\begin{table}[pos=!htbp]", r"\centering\small",
             r"\setlength{\tabcolsep}{4pt}", r"\caption{"+caption+"}",
             r"\label{"+label+"}", r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Data set & Fixed & Fitted & Fixed + affine & Fitted + affine \\", r"\midrule"]
    for case, title in CASES:
        vals = [float(lookup[case,10000,m][metric])*multiplier for m in models]
        lines.append(title+" & "+" & ".join(bold_best([fnum(v,digits) for v in vals], vals))+r" \\")
    lines.extend([r"\bottomrule",r"\end{tabular}",r"\end{table}"])
    path=SOURCE/"sections"/filename
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    outputs.append(path)

metric_table("test_error", "gpu_accuracy_n10000.tex", "tab:gpu-accuracy-n10000",
    "Test error (mean relative velocity error of the predictions for the 100 test inputs) at $N=10000$. The two augmented models use the same affine columns. Each model is fitted and validated separately, and all four use the same local output reconstruction. Both Taylor--Green cases use the amplitude normalization~\\eqref{eq:tg-response-transform}. Bold: smallest test error in each row.")

# Compact full-suite timing/storage table at the largest N. Each cell gives
# accounted decimal GB / synchronized warm ms for 100 fields.
lines=[r"\begin{table}[pos=!htbp]",r"\centering\small",r"\setlength{\tabcolsep}{3pt}",
       r"\caption{Accounted storage in decimal GB / median warm prediction time in ms for 100 fields at $N=10000$. Storage includes the stored model arrays, the solve buffers, and cuDSS's estimate of the memory of its factors. It is not measured peak GPU memory. Times include operator Lagrange evaluation and local output reconstruction. Bold: smallest storage and smallest time in each row.}",
       r"\label{tab:gpu-cost-n10000}",r"\begin{tabular}{lrrrr}",r"\toprule",
       r"Data set & Fixed & Fitted & Fixed + affine & Fitted + affine \\",r"\midrule"]
for case,title in CASES:
    gb=[float(lookup[case,10000,m]['accounted_resident_storage_ESTIMATE_decimal_GB']) for m in models]
    ms=[1000*float(lookup[case,10000,m]['warm_100query_median_seconds']) for m in models]
    gbc=bold_best([f"{v:.3f}" for v in gb], gb); msc=bold_best([f"{v:.1f}" for v in ms], ms)
    lines.append(title+" & "+" & ".join(f"{a} / {b}" for a,b in zip(gbc,msc))+r" \\")
lines += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
path=SOURCE/"sections"/"gpu_cost_n10000.tex";path.write_text("\n".join(lines)+"\n",encoding="utf-8");outputs.append(path)

# Complete N-dependent accuracy and cost ledger. Native fill is the vendor's
# CUDSS_DATA_LU_NNZ statistic, not an independently inferred triangular count.
lines=[]
short={"fixed_no_poly":"Fixed shape", "trainable_no_poly":"Fitted shape", "fixed_affine":"Fixed shape + affine", "trainable_affine":"Fitted shape + affine"}
for index,(case,title) in enumerate(CASES):
    lines += [r"\begin{table}[pos=!htbp]",r"\centering\small",r"\setlength{\tabcolsep}{3.5pt}",
       r"\caption{"+title+r": all models. Test error is the mean relative velocity error of the predictions for the 100 test inputs. Density is $d_K=\mathrm{nnz}(K_{UU})/N^2$ in percent. Factor is the number of entries of the cuDSS Cholesky factor in millions. GB is the accounted storage and ms the median warm time for all 100 fields. Bold: smallest test error at each $N$.}",
       rf"\label{{tab:gpu-all-{case.replace('_','-')}}}",
       r"\begin{tabular}{rlrrrrr}",r"\toprule",
       r"$N$ & Operator model & Test error & Density (\%) & Factor ($10^6$) & GB & ms \\",r"\midrule"]
    for n in NVALUES:
        errs=[float(lookup[case,n,m]['test_error']) for m in models]
        errc=bold_best([fnum(v) for v in errs], errs)
        for model,ec in zip(models,errc):
            row=lookup[case,n,model]
            lines.append(f"{n} & {short[model]} & {ec} & {100*float(row['kernel_density']):.2f} & {int(row['native_factor_nnz'])/1e6:.3f} & {float(row['accounted_resident_storage_ESTIMATE_decimal_GB']):.3f} & {1000*float(row['warm_100query_median_seconds']):.1f}"+r" \\")
        if n!=10000: lines.append(r"\addlinespace[3pt]")
    lines += [r"\bottomrule",r"\end{tabular}",r"\end{table}", ""]
path=SOURCE/"sections"/"gpu_all_models_tables.tex";path.write_text("\n".join(lines),encoding="utf-8");outputs.append(path)

lines=[r"\begin{table}[pos=tbp]",r"\centering\small",r"\setlength{\tabcolsep}{4pt}",
       r"\caption{Offline and initialization costs at $N=10000$. Selection includes kernel fitting, where applicable, and validation. Rebuild assembles and factors the selected operator. Initialization includes two synchronized 100-field prediction passes, compilation, and CUDA graph preparation. The four models share the setup of the output reconstruction and may reuse compiled code. The dash marks the selection time of the fixed-shape backward-facing-step model, which contains an external pause. Its raw time is in the released data. Bold: smallest time in each column for each data set.}",
       r"\label{tab:gpu-offline-n10000}",r"\begin{tabular}{llrrr}",r"\toprule",
       r"Data set & Operator model & Selection (min) & Rebuild (s) & Initialization (s) \\",r"\midrule"]
for case,title in CASES:
    rows=[lookup[case,10000,m] for m in models]
    ok=[r['selection_time_comparable_without_interruption']=='True' for r in rows]
    sel=[float(r['training_validation_minutes']) for r in rows]
    selv=[s for s,o in zip(sel,ok) if o]
    selc=[f"{s:.2f}" if o else "---" for s,o in zip(sel,ok)]
    best_sel=f"{min(selv):.2f}"
    selc=[bold(c) if c==best_sel else c for c in selc]
    reb=[float(r['operator_setup_seconds']) for r in rows]; rebc=bold_best([f"{v:.2f}" for v in reb], reb)
    ini=[float(r['first_prediction_compile_capture_seconds']) for r in rows]; inic=bold_best([f"{v:.2f}" for v in ini], ini)
    for i,model in enumerate(models):
        lines.append((title if i==0 else "")+" & "+short[model]+f" & {selc[i]} & {rebc[i]} & {inic[i]}"+r" \\")
    lines.append(r"\addlinespace[3pt]")
lines += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
path=SOURCE/"sections"/"gpu_offline_n10000.tex";path.write_text("\n".join(lines)+"\n",encoding="utf-8");outputs.append(path)

def decay_rate_table():
    """Least-squares exponent tau in test_error ~ N^(-tau) over all four N."""
    import math
    order = ["forced_turb", "species_transport", "merge_vortices_easier",
             "lid_cavity_flow", "flow_cylinder_laminar", "flow_cylinder_shedding",
             "backward_facing_step", "buoyancy_cavity_flow", "taylor_green", "taylor_green_time"]
    titles = dict(CASES)
    x = [math.log(n) for n in NVALUES]
    xm = sum(x)/len(x)
    lines = [r"\begin{table}[pos=!htbp]", r"\centering\small",
             r"\setlength{\tabcolsep}{4pt}",
             r"\caption{Least-squares exponent $\tau$ of the test error $\propto N^{-\tau}$ over $N\in\{500,1000,5000,10000\}$. Bold: fastest decay in each row.}",
             r"\label{tab:gpu-decay-rates}", r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Data set & Fixed & Fitted & Fixed + affine & Fitted + affine \\", r"\midrule"]
    rates = {}
    for case in order:
        vals = []
        for model in models:
            y = [math.log(float(lookup[case, n, model]["test_error"])) for n in NVALUES]
            ym = sum(y)/len(y)
            slope = sum((a-xm)*(b-ym) for a, b in zip(x, y))/sum((a-xm)**2 for a in x)
            vals.append(-slope)
        rates[case] = vals
        lines.append(titles[case]+" & "+" & ".join(bold_best([f"{v:.2f}".replace("-", "$-$") for v in vals], vals, best=max))+r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path = SOURCE/"sections"/"gpu_decay_rates.tex"
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")
    outputs.append(path)
    return rates

decay_rates = decay_rate_table()

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest={"purpose":"Reporting from the completed frozen GPU study; no numerical method rerun",
          "input_files":{name:digest(DATA/name) for name in INPUTS},
          "completed_cells":40,"checked_model_rows":160,
          "script_sha256":digest(Path(__file__)),
          "plot_dimensions":plot_dimensions,
          "generated_files":{str(path.relative_to(SOURCE)).replace('\\','/'):digest(path) for path in outputs}}
manifest["manuscript_files"] = {
    str(path.relative_to(SOURCE)).replace('\\','/'): digest(path)
    for path in [SOURCE/"sections/results.tex", SOURCE/"sections/gpu_results_appendix.tex",
                 SOURCE/"build_gpu_results.py"] if path.exists()
}
(DATA/"publication_artifact_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
print(json.dumps({"figures":len(plot_dimensions),"files":len(outputs),"rows":len(rows),"source":str(SOURCE)},indent=2))
