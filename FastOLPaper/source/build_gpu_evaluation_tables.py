"""Tables and the structure-function figure from the GPU evaluation study (scripts/fastol_gpu_evaluation_study.py).

Reads ../evidence/gpu_evaluation/{lagrange,spatial}.jsonl and writes
  sections/lagrange_all_table.tex            (Table: direct/Lagrange error ratios)
  sections/amplification_table.tex           (Table: amplification factors)
  sections/spatial_controls_table.tex        (Table: reconstructions between the nodes)
  sections/lagrange_matched_control_table.tex (Appendix table: one global lid-cavity model family)
  sections/fig_structure_functions.tex       (pgfplots figure)
and prints the numbers quoted in the text.
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVID = HERE.parent / "evidence" / "gpu_evaluation" if len(sys.argv) < 2 else Path(sys.argv[1])
SEC = HERE / "sections"
SEC.mkdir(exist_ok=True)
CASES = [("forced_turb", "Forced isotropic turbulence"), ("species_transport", "Turbulent species transport"),
         ("merge_vortices_easier", "Merging vortices"), ("lid_cavity_flow", "Lid-driven cavity"),
         ("flow_cylinder_laminar", "Cylinder flow (no shedding)"), ("flow_cylinder_shedding", "Cylinder flow (vortex shedding)"),
         ("backward_facing_step", "Backward-facing step"), ("buoyancy_cavity_flow", "Buoyancy-driven cavity"),
         ("taylor_green", "Taylor--Green vortices (space)"), ("taylor_green_time", "Taylor--Green vortices (spacetime)")]


def sci(x, digits=1):
    if x == 0:
        return "$0$"
    e = int(f"{x:e}".split("e")[1])
    m = x / 10 ** e
    return f"${m:.{digits}f}\\times10^{{{e}}}$"


def bold(cell):
    """Bold a math cell "$...$", keeping any text after it."""
    i = cell.index("$", 1)
    return r"$\mathbf{" + cell[1:i] + "}$" + cell[i + 1:]


def bold_best(cells, values, best=min):
    """Bold every cell whose printed value equals that of the best value (ties at the printed precision)."""
    target = cells[values.index(best(values))]
    return [bold(c) if c == target else c for c in cells]


def load(name):
    p = EVID / name
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def main():
    lag = load("lagrange.jsonl")
    spa = {r["case"]: r for r in load("spatial.jsonl")}
    by = defaultdict(list)
    for r in lag:
        by[r["case"]].append(r)

    # ---- direct / Lagrange ratios
    rows, allg, alll, numbers = [], [], [], {}
    for case, name in CASES:
        rs = by.get(case, [])
        if not rs:
            continue
        g = [r["global_direct"] / r["global_lagrange"] for r in rs]
        l = [r["local_direct"] / r["local_lagrange"] for r in rs]
        allg += g
        alll += l
        rows.append(f"{name} & {len(g)} & {statistics.median(g):.2f} & {max(g):.1f} & {len(l)} & {sci(max(abs(x - 1) for x in l))} \\\\")
        numbers[case] = dict(global_median=statistics.median(g), global_max=max(g), local_min=min(l), local_max=max(l))
    total = len(allg) + len(alll)
    tab = [r"\begin{table}[pos=tbp]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
           r"\caption{Ratio of the test error of direct evaluation to that of Lagrange evaluation, for the same unaugmented "
           r"operator models (the fixed-shape kernel of Section~\ref{sec:sparse_wendland} at the radii of the five prescribed densities and at its "
           r"validated radius, and the fitted-shape kernel at its validated radius, for $N=500$, $1000$, $5000$, and $10000$) "
           r"with global and local output reconstruction. Values above one favor Lagrange evaluation. Over all "
           f"{total} pairs, the smallest ratio is ${min(allg + alll):.5f}$.}}",
           r"\label{tab:lagrange-all}", r"\begin{tabular}{lrrrrr}", r"\toprule",
           r" & \multicolumn{3}{c}{Global output reconstruction} & \multicolumn{2}{c}{Local output reconstruction} \\",
           r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}",
           r"Data set & Models & Median & Max & Models & Max $|$ratio$-1|$ \\", r"\midrule", *rows,
           r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (SEC / "lagrange_all_table.tex").write_text("\n".join(tab), encoding="utf8", newline="\n")

    # ---- amplification factors (N = 1000)
    arows = []
    for case, name in CASES:
        rs = [r for r in by.get(case, []) if r["N"] == 1000]
        if not rs:
            continue
        gmax = max(r["global_direct"] / r["global_lagrange"] for r in by[case])
        ag, al = bold_best([sci(rs[0]['alpha_global']), f"${rs[0]['alpha_local']:.0f}$"],
                           [rs[0]['alpha_global'], rs[0]['alpha_local']])
        arows.append(f"{name} & {ag} & {al} & ${gmax:.1f}$ \\\\")
    amp = [r"\begin{table}[pos=tbp]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
           r"\caption{Amplification factor $\alpha_{\rm spatial}=\|B\|_F\|(\bA^\star)^{\mathsf T}\|_F/\|B(\bA^\star)^{\mathsf T}\|_F$ "
           r"of~\eqref{eq:spatial-amplification}, which measures the cancellation among the spatial coefficients, for $N=1000$ "
           r"training fields, for the global and for our local output reconstruction (stencils combined as in "
           r"Section~\ref{sec:lagrange_accuracy}), and the largest ratio of the direct to the Lagrange error with global "
           r"output reconstruction from Table~\ref{tab:lagrange-all}. Bold: smaller amplification factor in each row.}",
           r"\label{tab:amplification}", r"\begin{tabular}{lrrr}", r"\toprule",
           r"Data set & Global $\alpha_{\rm spatial}$ & Local $\alpha_{\rm spatial}$ & Largest gain (global) \\", r"\midrule",
           *arows, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (SEC / "amplification_table.tex").write_text("\n".join(amp), encoding="utf8", newline="\n")

    # ---- lid-cavity matched control (validated fixed-shape model, global reconstruction)
    crow = []
    for r in sorted((r for r in by.get("lid_cavity_flow", []) if r["model"] == "fixed_selected"), key=lambda r: r["N"]):
        ev = [r['global_direct'], r['global_lagrange'], r['global_direct_samples']]
        d, lg, ds = bold_best([sci(v, 2) for v in ev], ev)
        crow.append(f"{r['N']} & {d} & {lg} & "
                    f"{r['global_direct'] / r['global_lagrange']:.2f} & {ds} & "
                    f"{sci(r['global_coeff_norm'], 2)} \\\\")
    ctl = [r"\begin{table}[pos=!htbp]", r"\centering\small",
           r"\caption{Test error at the nodes (mean relative velocity error over the 100 test inputs) of direct and Lagrange evaluation of the validated "
           r"unaugmented fixed-shape model of the lid-driven cavity with global output reconstruction, of direct evaluation applied to the "
           r"sampled velocities instead of the coefficients, and the Frobenius norm of the spatial coefficients of the "
           r"training fields. Bold: smallest test error in each row.}",
           r"\label{tab:lagrange-matched-control}", r"\begin{tabular}{rrrrrr}", r"\toprule",
           r"$N$ & Direct & Lagrange & Direct/Lagrange & Direct on samples & $\|B\|_F$ \\", r"\midrule", *crow,
           r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (SEC / "lagrange_matched_control_table.tex").write_text("\n".join(ctl), encoding="utf8", newline="\n")

    # ---- reconstructions between the nodes
    srow = []
    for case, name in CASES:
        r = spa.get(case)
        if r is None:
            continue
        assert r["N"] == 10000 and "global" in r, (case, r["N"])
        lv = r["local_multiscale"] if "local_multiscale" in r else r["local_default"]
        glob, loc = bold_best([sci(r["global"], 2), sci(lv, 2)], [r["global"], lv])
        if "local_multiscale" in r:
            loc += " (multiscale)"
        srow.append(f"{name} & {r['m']} & {r['held_out']} & {glob} & {loc} \\\\")
    st = [r"\begin{table}[pos=tbp]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
          r"\caption{Test error between the nodes. We reconstruct the predicted nodal velocities of the 100 test inputs "
          r"(model with the smallest validation error at $N=10000$) and measure the mean relative velocity error at the "
          r"evaluation points, the mesh points of each data set that are not among its $m$ nodes. Global: the divergence-free Mat\'ern interpolant of~\citet{sharma2026fluids} on all "
          r"nodes. Local: our output reconstruction with the cubic PHS, or with the multiscale Mat\'ern kernel ($c_\sigma=16$) "
          r"where marked (Appendix~\ref{app:multiscale-spatial} compares it with the PHS). Bold: smaller error in each row.}",
          r"\label{tab:spatial-reconstruction-controls}", r"\begin{tabular}{lrrrr}", r"\toprule",
          r"Data set & Nodes $m$ & Evaluation points & Global & Local \\", r"\midrule", *srow,
          r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (SEC / "spatial_controls_table.tex").write_text("\n".join(st), encoding="utf8", newline="\n")

    # ---- structure functions (pgfplots)
    figs = []
    ticks = {"species_transport": "xtick={0.003,0.01,0.03,0.1}, ytick={200,300,500,800}, ",
             "forced_turb": "xtick={0.7,1,2,3}, ytick={0.5,0.7,1,1.4}, "}
    for case, title in (("species_transport", "Turbulent species transport"), ("forced_turb", "Forced isotropic turbulence")):
        r = spa.get(case)
        if not r or "structure_functions" not in r:
            continue
        sf = r["structure_functions"]
        rr = sf["r"]
        plots = []
        for key, style, legend in (("reference", "black, thick", "Reference"),
                                   ("local_phs", "blue, mark=o, mark size=1.2pt", "Local PHS"),
                                   ("local_multiscale", "red, mark=triangle, mark size=1.4pt", "Local multiscale")):
            pts = " ".join(f"({x:.6g},{y:.6g})" for x, y in zip(rr, sf[key]) if y > 0)
            plots.append(f"\\addplot[{style}] coordinates {{{pts}}};\\addlegendentry{{{legend}}}")
        figs.append("\\begin{tikzpicture}\\begin{loglogaxis}[width=0.46\\linewidth, height=0.36\\linewidth, "
                    f"title={{{title}}}, xlabel={{$r$}}, ylabel={{$S_2(r)$}}, legend pos=south east, log ticks with fixed point, "
                    + ticks[case] + "minor tick num=0, legend style={font=\\scriptsize}, tick label style={font=\\scriptsize}, label style={font=\\small}]\n"
                    + "\n".join(plots) + "\n\\end{loglogaxis}\\end{tikzpicture}")
    (SEC / "fig_structure_functions.tex").write_text("% Generated by build_gpu_evaluation_tables.py\n" + "\\hfill".join(figs) + "\n",
                                                     encoding="utf8", newline="\n")

    # ---- relative discrepancy of the binned structure functions
    sfd = {}
    for case in ("species_transport", "forced_turb"):
        r = spa.get(case)
        if r and "structure_functions" in r:
            sf = r["structure_functions"]
            ref = sf["reference"]
            nrm = sum(x * x for x in ref) ** 0.5
            sfd[case] = {k: sum((a - b) ** 2 for a, b in zip(sf[k], ref)) ** 0.5 / nrm
                         for k in ("local_phs", "local_multiscale")}

    # ---- numbers for the text
    print(json.dumps(dict(structure_function_discrepancy=sfd, pairs=total, smallest_ratio=min(allg + alll), local_range=[min(alll), max(alll)],
                          global_max=max(allg), per_case=numbers,
                          c_sigma={c: spa[c].get("c_sigma_validation") for c in ("species_transport", "forced_turb") if c in spa},
                          spatial={c: {k: v for k, v in r.items() if k != "structure_functions"} for c, r in spa.items()}),
                     indent=1))


if __name__ == "__main__":
    main()
