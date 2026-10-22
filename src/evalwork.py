"""Per-instance evaluation of learned warm starts (run in worker processes)."""
from __future__ import annotations

import logging
import time

import numpy as np
import pyomo.environ as pyo

HIT_TOL, FEAS_TOL = 1e-4, 1e-5


def _pred_metrics(inst, row, f, glob_obj):
    from learn import FLOW_SCALE, L, MAXI, NX, NY, flows_to_pq
    from pooling import evaluate, rel_gap
    q, y, z, x = flows_to_pq(row, f)
    e = evaluate(inst, q, y, z)
    bal = np.abs(x.sum(0) - y.sum(1)).max()            # kbd
    neg = max(0.0, -float(np.min(np.asarray(f) * FLOW_SCALE)))
    return {"obj": e["obj"], "gap": rel_gap(e["obj"], glob_obj), "mass_balance_viol": float(bal),
            "spec_viol": e["viol"]["spec_scaled"], "max_viol": e["max_viol"], "neg_flow": neg,
            "other_viol": max(e["viol"]["src"], e["viol"]["pool_cap"], e["viol"]["prod"])}, (q, y, z)


def evaluate_instance(args):
    logging.getLogger("pyomo").setLevel(logging.ERROR)
    from pooling import build_pyomo, evaluate, rel_gap, solve_ipopt
    from realdata import from_row
    row, glob_obj, f_mlp, f_proj, p_struct, thr = args
    inst = from_row(row)
    out = {"inst_id": row["inst_id"]}
    # (b) raw MLP prediction and (c) projected prediction
    mb, start = _pred_metrics(inst, row, f_mlp, glob_obj)
    out.update({f"b_{k}": v for k, v in mb.items()})
    if f_proj is not None:
        mc, _ = _pred_metrics(inst, row, f_proj, glob_obj)
        out.update({f"c_{k}": v for k, v in mc.items()})
    # (d) MLP prediction as IPOPT warm start (full problem)
    m = build_pyomo(inst, "pq", rlt=False)
    r = solve_ipopt(inst, m, start)
    out.update(d_obj=r["obj"], d_gap=rel_gap(r["obj"], glob_obj), d_viol=r["max_viol"], d_time=r["time"],
               d_term=r["term"], d_hit=bool(r["max_viol"] <= FEAS_TOL and rel_gap(r["obj"], glob_obj) <= HIT_TOL))
    # (e) structure-first: fix predicted inactive arcs to 0, predicted binding specs to equality
    if p_struct is not None:
        t0 = time.perf_counter()
        from learn import J, L, MAXI, NX, NY
        I = inst.I
        act = np.asarray(p_struct) >= thr
        ax = act[:NX].reshape(MAXI, L)[:I]; ay = act[NX:NX + NY].reshape(L, J); az = act[NX + NY:NX + NY + MAXI * J].reshape(MAXI, J)[:I]
        bind = act[NX + NY + MAXI * J:]
        m2 = build_pyomo(inst, "pq", rlt=False)
        for (l, j) in m2.LJ:
            if not ay[l, j]:
                m2.y[l, j].fix(0.0)
        for (i, j) in m2.IJ:
            if not az[i, j]:
                m2.z[i, j].fix(0.0)
        for l in range(L):
            arcs = [i for (i, ll) in m2.IL if ll == l]
            keep = [i for i in arcs if ax[i, l]]
            if ay[l].any() and keep:  # pool in use with >=1 predicted crude: fix the others to 0
                for i in arcs:
                    if not ax[i, l]:
                        m2.q[i, l].fix(0.0)
        m2.speq = pyo.ConstraintList()
        for c in range(len(bind)):
            if bind[c]:
                m2.speq.add(m2.spec[c].body >= 0)  # spec  body <= 0  becomes  body == 0
        q, y, z = start
        q = q.copy(); y = y.copy(); z = z.copy()
        r2 = solve_ipopt(inst, m2, (q, y, z))
        ok = r2["term"] == "optimal"
        out.update(e_obj=r2["obj"], e_gap=rel_gap(r2["obj"], glob_obj), e_viol=r2["max_viol"], e_term=r2["term"],
                   e_time=time.perf_counter() - t0,  # includes building the reduced model
                   e_solver_time=r2["time"],
                   e_hit=bool(ok and r2["max_viol"] <= FEAS_TOL and rel_gap(r2["obj"], glob_obj) <= HIT_TOL))
    return out
