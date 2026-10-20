#!/bin/bash
# Overnight GPU chain replacing the MATLAB/CPU comparisons of the FastOL paper (one job at a time).
# Resumable: a lagrange cell is skipped when all seven of its models are already in lagrange.jsonl.
W=/mnt/e/kernelpack-jax/.codex/worktrees/fastol-spatial-properties
OUT=/mnt/e/kernelpack-jax/Outputs_and_results/fastol_gpu_evaluation_20261019
LOG=$OUT/logs
mkdir -p $LOG
cd $W
export PYTHONPATH=$W/src:$W/scripts
export KERNELPACK_CUDSS_LIBRARY=/mnt/e/kernelpack-jax/Outputs_and_results/fastol_matched_safe_gpu_20261016/source_snapshot/src/kernelpack/sparse/_cudss_ffi.so
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/mnt/e/kernelpack-jax/.venv-wsl/bin/python
S=scripts/fastol_gpu_evaluation_study.py
cp $S $OUT/fastol_gpu_evaluation_study.py.used
CASES="lid_cavity_flow flow_cylinder_laminar flow_cylinder_shedding backward_facing_step buoyancy_cavity_flow merge_vortices_easier taylor_green taylor_green_time species_transport forced_turb"
echo "start $(date)" >> $LOG/chain.log
for N in 500 1000 5000 10000; do
  for C in $CASES; do
    done_count=$(grep -c "\"case\": \"$C\", \"N\": $N," $OUT/lagrange.jsonl 2>/dev/null || echo 0)
    if [ "$done_count" -ge 7 ]; then continue; fi
    grep -v "\"case\": \"$C\", \"N\": $N," $OUT/lagrange.jsonl > $OUT/lagrange.tmp 2>/dev/null && mv $OUT/lagrange.tmp $OUT/lagrange.jsonl
    echo "lagrange $C $N $(date)" >> $LOG/chain.log
    timeout 10800 $PY $S lagrange $C $N --out $OUT/lagrange.jsonl > $LOG/lagrange_${C}_N$N.log 2>&1
    echo "  exit $? $(date)" >> $LOG/chain.log
  done
done
for C in $CASES; do
  N=10000; [ "$C" = taylor_green_time ] && N=5000
  if grep -q "\"case\": \"$C\"," $OUT/spatial.jsonl 2>/dev/null; then continue; fi
  echo "spatial $C $N $(date)" >> $LOG/chain.log
  timeout 7200 $PY $S spatial $C $N --out $OUT/spatial.jsonl > $LOG/spatial_${C}.log 2>&1
  echo "  exit $? $(date)" >> $LOG/chain.log
done
echo "done $(date)" >> $LOG/chain.log
