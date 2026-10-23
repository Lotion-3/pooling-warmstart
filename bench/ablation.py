"""Ablation beyond the main protocol: the projection metric.

    python -P bench/ablation.py <domain> [--threads T]

Models (c) and (d) project onto the equality set in ORIGINAL units (identity metric). Here the same KKT
projection is weighted by the training-set output variance, i.e. it finds the closest feasible point in
STANDARDISED units:  y = yhat - K (A yhat + B x - b),  K = D pinv(A D),  D = diag(train std of y).
Constraint satisfaction is identical (both land exactly on the same affine set); only the direction of the
correction changes.
  c_w : model (a) of the main run (saved test predictions) + weighted projection at test time
  d_w : weighted projection layer trained end to end (same seeds, split, backbone and early stopping as (d))
Requires a finished main run (bench/results/<domain>/metrics.json and test_predictions.npz).
Writes bench/results/<domain>/ablation_weighted.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.domain import RESULTS  # noqa: E402
from bench.run import (N_CAP, SPLIT_SEED, Projector, Trainer, _write_json, bootstrap_diff, eval_model,  # noqa: E402
                       gate, load_domain, make_splits, reduce_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--max-epochs", type=int, default=300)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    main_m = json.loads((RESULTS / args.domain / "metrics.json").read_text())
    saved = np.load(RESULTS / args.domain / "test_predictions.npz")
    n_seeds = len(main_m["per_seed"])

    # identical preprocessing to bench/run.py:run_domain
    dom = load_domain(args.domain)
    n0 = dom.X.shape[0]
    if n0 > N_CAP:
        keep = np.sort(np.random.default_rng(SPLIT_SEED).choice(n0, N_CAP, replace=False))
        dom.X, dom.Y = dom.X[keep], dom.Y[keep]
        if dom.split is not None:
            dom.split = dom.split[keep]
    g = gate(dom)
    assert g["status"] == "pass"
    scale = np.asarray(g["scale"])
    Ar, Br, br, *_ = reduce_rows(dom.A, dom.B, dom.b)
    tr, va, te, _ = make_splits(dom)
    assert np.array_equal(te, saved["test_idx"]), "split differs from the main run"
    Yte = dom.Y[te]
    assert np.allclose(Yte, saved["Y"]), "test labels differ from the main run"
    sd = Yte.std(0)
    valid = sd > 1e-12 * np.maximum(1.0, np.abs(Yte).mean(0))

    T = Trainer(dom, tr, va, Ar, Br, br, scale, max_epochs=args.max_epochs)
    W = Projector(Ar, Br, br, w=T.ysd ** 2)           # train std, floored at 1 for constant outputs
    T.proj, T.proj_t = W, W.torch()

    per_seed, p_cw, p_dw = [], [], []
    for s in range(n_seeds):
        Pa = saved["a"][s].astype(np.float64)
        Pcw = W(Pa, dom.X[te])
        net, info = T.fit("d", s)                      # kind "d" uses T.proj_t, now the weighted layer
        Pdw = T.predict(net, "d", te)
        rec = {"seed": s, "c_w": eval_model(dom, te, Pcw, scale, sd, valid),
               "d_w": {**eval_model(dom, te, Pdw, scale, sd, valid), **info}}
        per_seed.append(rec)
        p_cw.append(Pcw)
        p_dw.append(Pdw)
        log(f"[{args.domain}] seed {s}: nRMSE c_w={rec['c_w']['nrmse']:.5f} d_w={rec['d_w']['nrmse']:.5f} "
            f"(main a={main_m['per_seed'][s]['a']['nrmse']:.5f} c={main_m['per_seed'][s]['c']['nrmse']:.5g} "
            f"d={main_m['per_seed'][s]['d']['nrmse']:.5g}) viol% d_w={rec['d_w']['eq_viol_pct_above_1e-5']:.1f} "
            f"epochs {info['epochs']}")

    agg = {}
    for k in ("c_w", "d_w"):
        agg[k] = {}
        for key in ("nrmse", "pooled_nrmse", "eq_viol_max", "eq_viol_pct_above_1e-5"):
            v = np.array([r[k][key] for r in per_seed])
            agg[k][f"{key}_mean"] = float(v.mean())
            agg[k][f"{key}_std"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        b = [r[k]["bounds"] for r in per_seed]
        if b[0] is not None:
            agg[k]["bound_pct_points_mean"] = float(np.mean([x["pct_points"] for x in b]))
    pa = [x.astype(np.float64) for x in saved["a"]]
    pd_ = [x.astype(np.float64) for x in saved["d"]]
    pc = [x.astype(np.float64) for x in saved["c"]]
    out = {"domain": args.domain, "n_seeds": n_seeds, "per_seed": per_seed, "aggregate": agg,
           "bootstrap": {"d_w_minus_a": bootstrap_diff(Yte, p_dw, pa, sd, valid),
                         "d_w_minus_d": bootstrap_diff(Yte, p_dw, pd_, sd, valid),
                         "c_w_minus_c": bootstrap_diff(Yte, p_cw, pc, sd, valid),
                         "c_w_minus_a": bootstrap_diff(Yte, p_cw, pa, sd, valid)},
           "note": ("c_w uses the main run's model (a) predictions saved as float32; d_w retrains with the same "
                    "seeds, split, standardisation and early stopping as (d)")}
    _write_json(RESULTS / args.domain / "ablation_weighted.json", out)
    log(f"[{args.domain}] ablation done")


if __name__ == "__main__":
    main()
