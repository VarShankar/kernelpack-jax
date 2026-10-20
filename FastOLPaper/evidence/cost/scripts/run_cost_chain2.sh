#!/bin/bash
# After lowering the grid-kNN threshold to 4096: equivalence on paper node sets, then re-measure affected sizes.
S=/mnt/e/kernelpack-jax/Outputs_and_results/fastol_matched_safe_gpu_20261016/source_snapshot
W=/mnt/e/kernelpack-jax/.codex/worktrees/fastol-spatial-properties
J=/mnt/e/kernelpack-jax/.venv-wsl/_claude_jobs
cd /mnt/e/kernelpack-jax
export PYTHONPATH=$W/src:$W/scripts
export KERNELPACK_CUDSS_LIBRARY=$S/src/kernelpack/sparse/_cudss_ffi.so
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
for c in species_transport buoyancy_cavity_flow; do
  timeout 2400 .venv-wsl/bin/python $J/grid_equiv_benchmark.py $c 2>&1 | grep -E '^\{|Traceback|Error'
done > $J/logs/cost_grid_equiv.jsonl
{
timeout 3600 .venv-wsl/bin/python $J/spatial_scaling.py 2 5000 10000 2>&1 | grep -E '^\{'
timeout 3600 .venv-wsl/bin/python $J/spatial_scaling.py 3 5000 7000 10000 2>&1 | grep -E '^\{'
} > $J/logs/cost_local_scaling_rerun.jsonl
for c in species_transport buoyancy_cavity_flow; do
  timeout 2400 .venv-wsl/bin/python $J/retime_spatial.py $c 2>&1 | grep -E '^\{|Traceback|Error'
done > $J/logs/cost_retime_rerun.jsonl
echo CHAIN2_DONE
