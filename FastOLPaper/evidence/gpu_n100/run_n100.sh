#!/bin/bash
# Steady cylinder at N=100 with the paper's GPU campaign code (source snapshot); the script copy lives in the
# snapshot's scripts/ so its provenance hashes resolve against the same snapshot.
S=/mnt/e/kernelpack-jax/Outputs_and_results/fastol_matched_safe_gpu_20261016/source_snapshot
J=/mnt/e/kernelpack-jax/.venv-wsl/_claude_jobs
cp $J/n100/fastol_gpu_campaign.py $S/scripts/fastol_gpu_campaign_n100.py
OUT=/mnt/e/kernelpack-jax/Outputs_and_results/fastol_n100_20261019/cells/flow_cylinder_laminar_N100
mkdir -p $(dirname $OUT)
cd /mnt/e/kernelpack-jax
export PYTHONPATH=$S/src:$S/scripts
export KERNELPACK_CUDSS_LIBRARY=$S/src/kernelpack/sparse/_cudss_ffi.so
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
timeout 3600 .venv-wsl/bin/python $S/scripts/fastol_gpu_campaign_n100.py --case flow_cylinder_laminar --N 100 --output $OUT > $J/logs/n100.log 2>&1
echo EXIT $?
grep -E "model_selection_complete|model_test_complete|Traceback|Error" $J/logs/n100.log | cut -c1-300 | tail -12
