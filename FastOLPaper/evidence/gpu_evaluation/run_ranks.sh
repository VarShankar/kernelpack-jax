#!/bin/bash
# Retained polynomial ranks of the local systems on the nodes of each benchmark (GPU), for Appendix A.2.
W=/mnt/e/kernelpack-jax/.codex/worktrees/fastol-spatial-properties
OUT=/mnt/e/kernelpack-jax/Outputs_and_results/fastol_gpu_evaluation_20261019
cd $W
export PYTHONPATH=$W/src:$W/scripts
export KERNELPACK_CUDSS_LIBRARY=/mnt/e/kernelpack-jax/Outputs_and_results/fastol_matched_safe_gpu_20261016/source_snapshot/src/kernelpack/sparse/_cudss_ffi.so
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
mv -f $OUT/ranks.jsonl $OUT/ranks_before_nonfinite.jsonl 2>/dev/null
for C in species_transport forced_turb buoyancy_cavity_flow lid_cavity_flow flow_cylinder_laminar flow_cylinder_shedding backward_facing_step taylor_green merge_vortices_easier taylor_green_time; do
  /mnt/e/kernelpack-jax/.venv-wsl/bin/python scripts/fastol_gpu_evaluation_study.py ranks $C --out $OUT/ranks.jsonl 2>&1 | grep -E '"stage"|Error' | cut -c1-200
done
