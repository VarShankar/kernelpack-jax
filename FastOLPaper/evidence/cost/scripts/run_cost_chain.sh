#!/bin/bash
# Sequential GPU jobs for the FastOL cost section (one at a time).
S=/mnt/e/kernelpack-jax/Outputs_and_results/fastol_matched_safe_gpu_20261016/source_snapshot
W=/mnt/e/kernelpack-jax/.codex/worktrees/fastol-spatial-properties
J=/mnt/e/kernelpack-jax/.venv-wsl/_claude_jobs
cd /mnt/e/kernelpack-jax
export PYTHONPATH=$W/src:$W/scripts
export KERNELPACK_CUDSS_LIBRARY=$S/src/kernelpack/sparse/_cudss_ffi.so
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
{
timeout 3600 .venv-wsl/bin/python $J/spatial_scaling.py 2 1000 2000 5000 10000 20000 50000 100000 2>&1 | grep -E '^\{'
timeout 3600 .venv-wsl/bin/python $J/spatial_scaling.py 3 1000 2000 5000 7000 10000 20000 50000 100000 2>&1 | grep -E '^\{'
} > $J/logs/cost_local_scaling.jsonl
{
timeout 1800 .venv-wsl/bin/python $J/global_warm.py 2 1000 2000 5000 10000 12500 2>&1 | grep -E '^\{'
timeout 1800 .venv-wsl/bin/python $J/global_warm.py 3 1000 2000 5000 7000 8000 2>&1 | grep -E '^\{'
} > $J/logs/cost_global_warm.jsonl
for c in forced_turb species_transport merge_vortices_easier lid_cavity_flow flow_cylinder_laminar flow_cylinder_shedding backward_facing_step buoyancy_cavity_flow taylor_green taylor_green_time; do
  timeout 2400 .venv-wsl/bin/python $J/retime_spatial.py $c 2>&1 | grep -E '^\{|Traceback|Error'
done > $J/logs/cost_retime_benchmarks.jsonl
echo CHAIN_DONE
