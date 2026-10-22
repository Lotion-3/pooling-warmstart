"""Phase 1: validate the pooling formulation on literature instances.

Global: SCIP on the pq-formulation (with RLT rows). Compare to MINLPLib best_known_obj (1e-4 rel).
Local: IPOPT from 20 random starts, pq without RLT rows (main) and p-formulation (reference).
Writes results/validation.csv.
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gamslib_pool import load_case, read_tables  # noqa: E402
from pooling import build_pyomo, rel_gap, random_start, solve_ipopt, solve_scip  # noqa: E402

logging.getLogger("pyomo").setLevel(logging.ERROR)
CASES = ["haverly1", "haverly2", "haverly3", "bental4", "foulds2", "adhya1", "rt2"]
N_STARTS, TOL, FEAS = 20, 1e-4, 1e-5

minlp = pd.read_csv(ROOT / "data" / "clean" / "pooling_instances.csv").set_index("name")["best_known_obj"]
T = read_tables()
rows = []
for case in CASES:
    inst = load_case(case, T)
    ref = float(minlp[f"pooling_{case}pq"])
    g = solve_scip(inst, time_limit=300)
    row = {"case": case, "minlplib_best_known": ref, "gamslib_sol": inst.meta["best_known"],
           "I": inst.I, "L": inst.L, "J": inst.J, "K": inst.lam.shape[1],
           "scip_status": g["status"], "scip_obj": g["obj"], "scip_gap": g["gap"], "scip_time_s": g["time"],
           "scip_max_viol": g.get("max_viol"), "scip_rel_err": abs(rel_gap(g["obj"], ref)),
           "match_1e-4": abs(rel_gap(g["obj"], ref)) <= TOL}
    for form, rlt, tag in [("pq", False, "ipopt_pq"), ("p", True, "ipopt_p")]:
        m = build_pyomo(inst, form, rlt)
        rng = np.random.default_rng(12345)
        res = [solve_ipopt(inst, m, random_start(inst, rng)) for _ in range(N_STARTS)]
        feas = [r for r in res if r["max_viol"] <= FEAS]
        hits = [r for r in feas if rel_gap(r["obj"], ref) <= TOL]
        row.update({f"{tag}_feasible": len(feas), f"{tag}_hits": len(hits), f"{tag}_hit_rate": len(hits) / N_STARTS,
                    f"{tag}_best": min((r["obj"] for r in feas), default=np.nan),
                    f"{tag}_distinct_local_objs": len({round(r["obj"], 3) for r in feas}),
                    f"{tag}_mean_time_s": float(np.mean([r["time"] for r in res]))})
    rows.append(row)
    print({k: row[k] for k in ["case", "minlplib_best_known", "scip_obj", "match_1e-4", "scip_time_s",
                               "ipopt_pq_hit_rate", "ipopt_p_hit_rate"]}, flush=True)

out = ROOT / "results"
out.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out / "validation.csv", index=False)
print("all match:", all(r["match_1e-4"] for r in rows))
