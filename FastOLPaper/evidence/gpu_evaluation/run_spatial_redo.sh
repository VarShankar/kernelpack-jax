#!/bin/bash
# Table 2 redo: Taylor-Green (time) at N=10000, and the global reconstruction for the two 3D flows.
W=/mnt/e/kernelpack-jax/.codex/worktrees/fastol-spatial-properties
OUT=/mnt/e/kernelpack-jax/Outputs_and_results/fastol_gpu_evaluation_20261019
cd $W
export PYTHONPATH=$W/src:$W/scripts
export KERNELPACK_CUDSS_LIBRARY=/mnt/e/kernelpack-jax/Outputs_and_results/fastol_matched_safe_gpu_20261016/source_snapshot/src/kernelpack/sparse/_cudss_ffi.so
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
PY=/mnt/e/kernelpack-jax/.venv-wsl/bin/python
S=scripts/fastol_gpu_evaluation_study.py
cp $S $OUT/fastol_gpu_evaluation_study.py.used
for C in taylor_green_time species_transport forced_turb; do
  echo "spatial $C $(date)"
  $PY $S spatial $C --out $OUT/spatial_redo.jsonl > $OUT/logs/spatial_redo_$C.log 2>&1
  echo "  exit $? $(date)"
done
echo done
