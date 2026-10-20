# Reproducing the FastOL paper

Scripts that reproduce the results, tables, and figures of

> R. Sharma, M. Lowery, H. Owhadi, and V. Shankar,
> *Local and Sparse Kernel Operator Learning for Incompressible Flows: An Operator Lagrange Formulation* (submitted to the Journal of Computational Physics),

a follow-up to *Fluids You Can Trust: Property-Preserving Operator Learning for Incompressible Flows* (arXiv:2602.15472), whose ten benchmark data sets it uses. The repository holds code and scripts only; results are produced by running them.

The method is implemented in `kernelpack.operator_learning`. Install it with `pip install -e ".[fastol]"` and build the cuDSS extension (`python -m kernelpack.sparse.build_cudss`). Every result in the paper was computed on one NVIDIA RTX 5090 in double precision. The scripts take data and output locations as command-line options; `FASTOL_DATA_ROOT` points to the benchmark data sets, and `FASTOL_CAMPAIGN_DIR` and `FASTOL_REPORT_DIR` to the outputs of the operator campaign.

## 1. Results

| Results in the paper | Script (paths from the repository root) | Output read by the builders of step 2 |
| --- | --- | --- |
| Operator models of Section 5.2 (ten benchmarks, four training sizes, four models) | `scripts/run_fastol_gpu_campaign.py`, then `FastOLPaper/evidence/gpu/final_completion_audit.py` | `final_validated_models.csv`, `final_paired_comparisons.csv`, `final_shared_spatial_preparation.csv`, `completion_audit.json` in `FASTOL_REPORT_DIR` |
| Steady cylinder at N = 100 (Table 4) | `FastOLPaper/evidence/gpu_n100/run_n100.sh` | |
| Storage and prediction times with nested-dissection ordering | `scripts/fastol_gpu_cost_replay.py` | `FastOLPaper/source/gpu_data/cost_replay.jsonl` |
| Sections 5.1.4 and 5.3, Appendices A.2, A.5, D.4 | `scripts/fastol_gpu_evaluation_study.py` (see `FastOLPaper/evidence/gpu_evaluation/run_*.sh`) | `FastOLPaper/evidence/gpu_evaluation/{lagrange,spatial,ranks}.jsonl` |
| Cost of the local reconstruction (Section 5.1.5) | `FastOLPaper/evidence/cost/scripts/run_cost_chain*.sh`, then `FastOLPaper/evidence/cost/build_cost_json.py` | `FastOLPaper/evidence/cost/cost_scaling.json` |
| Convergence, divergence, periodicity (Sections 5.1.1, 5.1.2) | `scripts/run_fastol_spatial_validation.py`, then `scripts/report_fastol_spatial_properties.py` | `FastOLPaper/evidence/spatial/report_final/spatial_report.json` |
| Kolmogorov statistics (Section 5.1.3) | `scripts/fastol_spatial_statistics.py`, `scripts/extend_fastol_local_matern_refinement.py`, then `scripts/report_fastol_kolmogorov_refinement.py` | `FastOLPaper/evidence/kolmogorov/report_r1/report.json` |

## 2. Tables and figures

```sh
cd FastOLPaper/source
python build_gpu_results.py               # Tables 9-22 (copies the campaign tables into gpu_data/)
python compare_prior_results.py           # Table 4
python build_gpu_evaluation_tables.py     # Tables 2, 5-8 and Figure 11
python make_diagrams.py                   # Figures 1 and 2
cd matlab
matlab -batch "fastol_paper_plots"        # Figures 3-10 and 20-23; 'fields' draws Figures 12-19 from FASTOL_DATA_ROOT
```

The Python builders write LaTeX tables to `FastOLPaper/source/sections/`, and MATLAB writes the figures to `FastOLPaper/source/figures/` with [export_fig](https://github.com/altmany/export_fig) (set `EXPORT_FIG_DIR`).
