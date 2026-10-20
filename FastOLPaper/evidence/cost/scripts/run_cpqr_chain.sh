#!/bin/bash
S=/mnt/e/kernelpack-jax/Outputs_and_results/fastol_matched_safe_gpu_20261016/source_snapshot
W=/mnt/e/kernelpack-jax/.codex/worktrees/fastol-spatial-properties
J=/mnt/e/kernelpack-jax/.venv-wsl/_claude_jobs
export KERNELPACK_CUDSS_LIBRARY=$S/src/kernelpack/sparse/_cudss_ffi.so
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1
cd $W
PYTHONPATH=$W/src:$W/scripts:$W timeout 3000 /mnt/e/kernelpack-jax/.venv-wsl/bin/python -m pytest -q \
  tests/test_divfree.py tests/test_divfree_center_limits.py tests/test_divfree_matern_mixture.py \
  tests/test_divfree_matern_quotient.py tests/test_local_reconstruction_fast_paths.py 2>&1 | tail -3
cd /mnt/e/kernelpack-jax
export PYTHONPATH=$W/src:$W/scripts
for c in species_transport forced_turb; do
  timeout 2400 .venv-wsl/bin/python $J/rank_fastpath_check.py $c 2>&1 | grep -E '^\{|Traceback|Error' | cut -c1-400
done
timeout 2400 .venv-wsl/bin/python $J/retime_spatial.py species_transport 2>&1 | grep -E '^\{|Traceback|Error' | cut -c1-600
echo CPQR_CHAIN_DONE
