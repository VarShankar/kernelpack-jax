#!/bin/bash
# Regenerates every result, figure, and table of the KernelPack-JAX paper (JCP submission).
#
# Usage (from the repository root, on a machine with an NVIDIA GPU, cuDSS, and the package installed):
#   PYTHON=python bash scripts/paper/reproduce.sh            # everything
#   PYTHON=python bash scripts/paper/reproduce.sh clouds     # one stage: clouds forward euclid estimate
#                                                            #   inverse obstacle geometry stats figures
# The stages are independent given their inputs and skip work whose results already exist.
# Output directories match the paths read by scripts/paper_figures.py. The figures are drawn by
# scripts/matlab/paper_plots.m from Outputs_and_results/paper_figures/plot_data.json (MATLAB).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PYTHON:-python}
export PYTHONPATH=src:scripts JAX_ENABLE_X64=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
C=Outputs_and_results/paper_clouds_v6_20261014
F=Outputs_and_results/paper_forward_v6_20261014
I=Outputs_and_results/paper_inverse_v6_20261014
CO=Outputs_and_results/paper_clouds_obstacle_20261015
IO=Outputs_and_results/paper_inverse_obstacle_20261015/adr
STAGES=${*:-clouds forward euclid estimate inverse obstacle geometry stats figures}
mkdir -p $C $F $I $CO $IO
run() { echo "$(date +'%F %T') $*"; "$@"; }

for stage in $STAGES; do case $stage in
clouds)      # H-channel and treasure chest, h = 0.16 ... 0.04, seed 11 (Sections 2 and 6.1)
  run $PY scripts/generate_paper_clouds.py --seeds 11 --output $C ;;
forward)     # Darcy, reaction-diffusion, advection-diffusion, p = 2, 4, 6 (Section 6.2)
  run $PY scripts/run_cudss_paper_forward.py --output $F/seed11 --domain-dir $C --seed 11 --fit-resolution 41 \
    --cfpu-max-sites 512 --ad-velocity-amplitude 1.18 --ad-dt-factor .25 --ad-dt-cfl-factor .0625 \
    --ad-hyperviscosity-rows far --ad-hyperviscosity-power order --ad-velocity-field reference \
    --boundary-aware-stencils ;;
euclid)      # the same forward problems with Euclidean stencils (stencil-selection comparison, Section 6.2)
  run $PY scripts/run_cudss_paper_forward.py --output $F/euclid --domain-dir $C --seed 11 --fit-resolution 41 \
    --cfpu-max-sites 512 --ad-velocity-amplitude 1.18 --ad-dt-factor .25 --ad-dt-cfl-factor .0625 \
    --ad-hyperviscosity-rows far --ad-hyperviscosity-power order --ad-velocity-field reference ;;
estimate)    # truncation errors and effective stability constants (Section 5, Appendix D)
  run $PY scripts/paper_forward_estimate_check.py --domain-dir $C --boundary-aware-stencils \
    --out $F/estimate_check.jsonl --skip-rejected ;;
inverse)     # Darcy, reaction-diffusion, transport parameter recovery (Section 6.3)
  for prob in darcy rd transport; do
    extra=""
    [ $prob = transport ] && extra="--transport-speed 1.18 --dt-factor .25 --dt-cfl-factor .0625 --hyperviscosity-rows far"
    run $PY scripts/paper_inverse.py --problem $prob --domain-dir $C --output $I/$prob --max-lm-steps 200 \
      --boundary-aware-stencils --warm-start-file $I/$prob/results.jsonl $extra
  done ;;
obstacle)    # treasure chest with an internal obstacle: clouds and ADR reaction recovery (Section 6.5)
  run $PY scripts/generate_paper_clouds.py --geometries obstacle_chest --h .16 .12 .08 .06 .04 --seeds 11 --output $CO
  run $PY scripts/paper_inverse.py --problem adr --domain-dir $CO --geometries obstacle_chest --h .16 .12 .08 .06 .04 \
    --orders 2 4 6 --dt-factor .25 --dt-cfl-factor .0625 --hyperviscosity-rows far --boundary-aware-stencils \
    --warm-start-file $IO/results.jsonl --output $IO ;;
geometry)    # convergence of the parametric and implicit layers (Figure: geometry)
  run $PY scripts/paper_geometry_convergence.py ;;
stats)       # parallel-algorithm and stencil-crossing statistics (Sections 6.1, 6.2), velocity fields
  run $PY scripts/paper_node_stats.py $C $C/node_stats.json
  run $PY scripts/paper_stencil_stats.py $C $C/stencil_stats.jsonl
  run $PY scripts/paper_velocity_fields.py
  run $PY scripts/paper_obstacle_solution_fields.py ;;
figures)     # tables (KernelPackJaxPaper/gen_*.tex), plot data, and the matplotlib node renderings
  run $PY scripts/paper_figures.py
  run $PY scripts/paper_chest_nodes_figure.py 0.08
  run $PY scripts/paper_obstacle_nodes_figure.py 0.04
  echo "Now run scripts/matlab/paper_plots.m and scripts/matlab/obstacle_solution_plots.m in MATLAB to draw KernelPackJaxPaper/Images/plots/*.pdf." ;;
*) echo "unknown stage $stage"; exit 2 ;;
esac; done
