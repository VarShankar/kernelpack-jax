#!/bin/bash
S=/mnt/e/kernelpack-jax/Outputs_and_results/fastol_matched_safe_gpu_20261016/source_snapshot
W=/mnt/e/kernelpack-jax/.codex/worktrees/fastol-spatial-properties
J=/mnt/e/kernelpack-jax/.venv-wsl/_claude_jobs
cd /mnt/e/kernelpack-jax
export KERNELPACK_CUDSS_LIBRARY=$S/src/kernelpack/sparse/_cudss_ffi.so
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
PYTHONPATH=$W/src:$W/scripts timeout 2400 .venv-wsl/bin/python $J/species_between_nodes.py new 2>&1 | grep -E '^\{|Traceback|Error' | cut -c1-400
PYTHONPATH=$S/src:$W/scripts timeout 2400 .venv-wsl/bin/python $J/species_between_nodes.py old 2>&1 | grep -E '^\{|Traceback|Error' | cut -c1-400
.venv-wsl/bin/python -c "
import numpy as np
a=np.load('$J/logs/species_between_new.npy'); b=np.load('$J/logs/species_between_old.npy')
print('new_vs_old_rel', float(np.linalg.norm(a-b)/np.linalg.norm(b)))"
