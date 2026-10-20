"""Table 4 of the paper: our models against the errors reported in arXiv:2602.15472."""
import csv
from pathlib import Path
P = Path(__file__).resolve().parent / "gpu_data" / "final_validated_models.csv"
rows = list(csv.DictReader(open(P, newline="")))
# prior paper (arXiv 2602.15472, Table 2): N, PPKM, VKM, Geo-FNO, Transolver, Geo-FNO+div, Transolver+div
prior = {
 "flow_cylinder_laminar":  (100,   3.95e-5, 5.32e-5, 1.618e-4, 7.36e-3, 1.555e-4, 5.738e-3),
 "flow_cylinder_shedding": (10000, 3.14e-7, 2.34e-6, 4.783e-5, 1.504e-4, 3.737e-5, 1.523e-4),
 "lid_cavity_flow":        (10000, 1.39e-6, 1.45e-6, 1.509e-4, 2.139e-4, 1.795e-4, 2.108e-4),
 "backward_facing_step":   (500,   1.51e-6, 2.9e-6, 9.949e-5, 7.330e-4, 9.259e-5, 7.247e-4),
 "buoyancy_cavity_flow":   (10000, 6.27e-6, 6.14e-6, 1.421e-4, 2.305e-4, 1.571e-4, 1.932e-4),
 "taylor_green":           (5000,  8.76e-10, 1.68e-9, 4.181e-4, 1.531e-4, 5.070e-4, 6.072e-5),
 "taylor_green_time":      (5000,  8.65e-10, 8.67e-10, 1.004e-3, 3.447e-4, 7.765e-4, 2.177e-4),
 "merge_vortices_easier":  (500,   1.09e-4, 1.09e-4, 1.351e-3, 3.747e-3, 1.124e-3, 3.397e-3),
 "species_transport":      (10000, 2.38e-4, 2.35e-4, 4.958e-4, 2.737e-3, None, None),
 "forced_turb":            (10000, 3.63e-4, 1.55e-4, 1.842e-4, 1.206e-3, 1.486e-4, 1.154e-3),
}
for case, pr in prior.items():
    N = pr[0]
    r = [x for x in rows if x["case"] == case and int(x["N"]) == N]
    if not r:
        r = [x for x in rows if x["case"] == case and int(x["N"]) == 500]; tag = "(N=500!)"
    else: tag = ""
    best = min(r, key=lambda x: float(x["validation_error"]))
    allt = {x["model"]: float(x["test_error"]) for x in r}
    print(f"{case:24s} N={N:5d}{tag} sel={best['model']:18s} test={float(best['test_error']):.3g} ppkm={pr[1]:.3g} ratio_ppkm={float(best['test_error'])/pr[1]:.2f} best_NO={min(v for v in pr[3:] if v):.3g}  all={ {k: f'{v:.3g}' for k,v in allt.items()} }")
