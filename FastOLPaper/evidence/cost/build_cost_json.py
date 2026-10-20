"""Collect the reconstruction cost runs (raw JSON lines in this folder) into cost_scaling.json for
matlab/fastol_paper_plots('cost') and the appendix numbers. Cost only; no accuracy is affected.

Raw inputs, produced by the scripts in scripts/ (RTX 5090, fp64, JAX 0.10.0):
  local_scaling.jsonl     spatial_scaling.py: local setup (first run incl. compilation, then warm) and buffers
  global_warm.jsonl       global_warm.py: dense divergence-free Matern Gram + Cholesky, warm median of 3
  retime_benchmarks.jsonl retime_spatial.py: shared spatial preparation of the ten benchmarks (N=10000)
"""
import csv, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
def rows(name):
    return [json.loads(l) for l in (HERE / name).read_text(encoding="utf-8-sig").splitlines() if l.startswith("{")]
local = [dict(d=r["d"], m=r["m"], first_s=r["local_setup_s"], warm_s=r["local_setup_warm_s"], bytes=r["local_buffer_bytes"])
         for r in rows("local_scaling.jsonl") if "local_setup_warm_s" in r]
glob = [dict(d=r["d"], m=r["m"], assembly_s=r["global_assembly_warm_s"], cholesky_s=r["global_cholesky_warm_s"],
             matrix_bytes=r["global_matrix_bytes"]) for r in rows("global_warm.jsonl") if r.get("global_ok")]
old = {r["case"]: float(r["shared_spatial_preparation_seconds"])
       for r in csv.DictReader(open(HERE.parents[1] / "source" / "gpu_data" / "final_shared_spatial_preparation.csv"))
       if r["N"] == "10000"}
bench = [dict(case=r["case"], m=r["m"], d=r["d"], first_s=r["first_s"], warm_s=r["warm_s"],
              campaign_s=old.get(r["case"])) for r in rows("retime_benchmarks.jsonl")]
(HERE / "cost_scaling.json").write_text(json.dumps(dict(local=local, dense_global=glob, benchmarks=bench), indent=1))
print(len(local), len(glob), len(bench))
