#!/bin/bash
# Rerun of the 3D cells at N=10000 (two-phase memory layout, platform allocator), then a check run on
# forced_turb N=5000 is NOT repeated (it completed). Appends to the same lagrange.jsonl.
W=/mnt/e/kernelpack-jax/.codex/worktrees/fastol-spatial-properties
OUT=/mnt/e/kernelpack-jax/Outputs_and_results/fastol_gpu_evaluation_20261019
LOG=$OUT/logs
cd $W
export PYTHONPATH=$W/src:$W/scripts
export KERNELPACK_CUDSS_LIBRARY=/mnt/e/kernelpack-jax/Outputs_and_results/fastol_matched_safe_gpu_20261016/source_snapshot/src/kernelpack/sparse/_cudss_ffi.so
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
PY=/mnt/e/kernelpack-jax/.venv-wsl/bin/python
S=scripts/fastol_gpu_evaluation_study.py
cp $S $OUT/fastol_gpu_evaluation_study.py.used
for C in species_transport forced_turb; do
  N=10000
  done_count=$(grep -c "\"case\": \"$C\", \"N\": $N," $OUT/lagrange.jsonl 2>/dev/null || echo 0)
  if [ "$done_count" -ge 7 ]; then continue; fi
  grep -v "\"case\": \"$C\", \"N\": $N," $OUT/lagrange.jsonl > $OUT/lagrange.tmp && mv $OUT/lagrange.tmp $OUT/lagrange.jsonl
  echo "lagrange3d $C $N $(date)" >> $LOG/chain.log
  timeout 14400 $PY $S lagrange $C $N --out $OUT/lagrange.jsonl > $LOG/lagrange_${C}_N${N}_3d.log 2>&1
  echo "  exit $? $(date)" >> $LOG/chain.log
done
echo "done3d $(date)" >> $LOG/chain.log
