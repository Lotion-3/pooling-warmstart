"""Phase 5: learn warm starts / structure and evaluate on the time-split test set.

  python -P scripts/phase5.py --dataset varied   (instances.parquet       + data/solved/parts)
  python -P scripts/phase5.py --dataset fixed    (instances_fixed.parquet + data/solved/parts_fixed)

Split by date: train 2006-2017, validation 2018-2019, test 2020-2026. Only certified instances
(SCIP optimal or gap <= 1e-4) are used for labels and for evaluation (they define "global optimum").
Outputs under results/<dataset>/.
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import learn as Lr  # noqa: E402
from io_utils import read_parquet_dir, write_parquet  # noqa: E402
from pooling import rel_gap  # noqa: E402

HIT_TOL, FEAS_TOL = 1e-4, 1e-5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["varied", "fixed"], required=True)
    ap.add_argument("--workers", type=int, default=19)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--debug-random-split", action="store_true", help="pipeline test only: random split")
    ap.add_argument("--epochs", type=int, default=300)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = ROOT / "results" / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "phase5_log.txt", "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    inst_file = "instances.parquet" if a.dataset == "varied" else "instances_fixed.parquet"
    parts = "parts" if a.dataset == "varied" else "parts_fixed"
    I = pd.read_parquet(ROOT / "data" / "instances" / inst_file)
    S = read_parquet_dir(ROOT / "data" / "solved" / parts)
    if "error" in S:
        log(f"solver errors: {S['error'].notna().sum()}")
        S = S[S["error"].isna()]
    D = I.merge(S, on="inst_id", how="inner")
    D["year"] = pd.to_datetime(D["date"]).dt.year
    n_all = len(D)
    D = D[D["certified"] & np.isfinite(D["glob_obj"])].reset_index(drop=True)
    log(f"{a.dataset}: {n_all} solved, {len(D)} certified (used)")
    split = np.where(D.year <= 2017, "train", np.where(D.year <= 2019, "val", "test"))
    if a.debug_random_split:
        out = ROOT / "results" / "_debug"; out.mkdir(parents=True, exist_ok=True)
        split = np.random.default_rng(0).choice(["train", "val", "test"], len(D), p=[.7, .15, .15])
    D["split"] = split
    log(str(D.split.value_counts().to_dict()))

    rows = D.to_dict("records")
    X = np.stack([Lr.features(r) for r in rows])
    M = np.stack([Lr.masks(r) for r in rows])
    Y = np.stack([Lr.flows_from_solution(r, r["glob_q"], r["glob_y"], r["glob_z"]) for r in rows])
    Sl = np.stack([Lr.labels_structure(r, r) for r in rows])
    Ms = np.concatenate([M, np.ones((len(M), Lr.NSPEC))], 1)
    tr, va, te = (D.split == "train").values, (D.split == "val").values, (D.split == "test").values
    mu, sd = Lr.standardize(X[tr])
    Xs = (X - mu) / sd
    nin = Xs.shape[1]

    log("training (b) flow MLP")
    mb, vb = Lr.train_model(Lr.FlowNet(nin, project=False), Xs[tr], M[tr], Y[tr], Xs[va], M[va], Y[va], log=log, epochs=a.epochs)
    log("training (c) flow MLP + KKT projection")
    mc, vc = Lr.train_model(Lr.FlowNet(nin, project=True), Xs[tr], M[tr], Y[tr], Xs[va], M[va], Y[va], log=log, epochs=a.epochs)
    log("training (e) structure classifier")
    mclf, ve = Lr.train_model(Lr.MLP(nin, Lr.NF + Lr.NSPEC), Xs[tr], Ms[tr], Sl[tr], Xs[va], Ms[va], Sl[va],
                              loss_kind="bce", log=log, epochs=a.epochs)
    T = lambda z: torch.tensor(z, dtype=torch.float32)
    with torch.no_grad():
        # inference time: single-instance forward passes on 200 test rows (what a warm start would pay)
        ti = np.where(te)[0][:200]
        t_inf = {}
        for tag, fn in [("b", lambda k: mb(T(Xs[k:k+1]), T(M[k:k+1]))), ("c", lambda k: mc(T(Xs[k:k+1]), T(M[k:k+1]))),
                        ("e", lambda k: mclf(T(Xs[k:k+1])))]:
            t1 = time.perf_counter()
            for k in ti:
                fn(k)
            t_inf[tag] = (time.perf_counter() - t1) / max(len(ti), 1)
        Fb = mb(T(Xs), T(M)).numpy()
        Fc = mc(T(Xs), T(M)).numpy()
        Ps = torch.sigmoid(mclf(T(Xs))).numpy()
    torch.save({"b": mb.state_dict(), "c": mc.state_dict(), "e": mclf.state_dict(), "mu": mu, "sd": sd},
               out / "models.pt")

    # classifier accuracy on test (masked entries only)
    pred = Ps >= 0.5
    m_te = Ms[te] > 0
    acc_all = float(((pred[te] == (Sl[te] > 0.5)) & m_te).sum() / m_te.sum())
    arc_m = m_te[:, :Lr.NF]; spec_m = m_te[:, Lr.NF:]
    acc_arc = float(((pred[te][:, :Lr.NF] == (Sl[te][:, :Lr.NF] > 0.5)) & arc_m).sum() / arc_m.sum())
    acc_spec = float(((pred[te][:, Lr.NF:] == (Sl[te][:, Lr.NF:] > 0.5)) & spec_m).sum() / spec_m.sum())
    exact = float(np.mean([np.array_equal(p[m], s[m] > 0.5) for p, s, m in zip(pred[te], Sl[te], m_te)]))
    clf = {"acc_all": acc_all, "acc_arcs": acc_arc, "acc_specs": acc_spec, "exact_pattern": exact,
           "val_loss": {"b": vb, "c": vc, "e": ve}, "inference_s_per_instance": t_inf,
           "n": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())}}
    log(f"classifier test: {clf}")

    # evaluate on test with IPOPT (parallel)
    idx = np.where(te)[0]
    if a.max_test:
        idx = idx[:a.max_test]
    inst_cols = ["inst_id", "name", "crudes", "c", "d", "src_lo", "src_up", "pool_cap", "prod_lo", "prod_up", "lam",
                 "q_up", "y_up", "z_up", "spec_prod", "spec_alpha", "spec_beta", "h"]
    inst_cols += [c + "_shape" for c in inst_cols if c + "_shape" in D.columns]
    jobs = [({k: rows[i][k] for k in inst_cols}, rows[i]["glob_obj"], Fb[i], Fc[i], Ps[i], 0.5) for i in idx]
    log(f"evaluating {len(jobs)} test instances with {a.workers} workers")
    t0 = time.time()
    from evalwork import evaluate_instance
    with mp.get_context("spawn").Pool(a.workers) as pool:
        res = list(pool.imap(evaluate_instance, jobs, chunksize=4))
    log(f"evaluation done in {(time.time()-t0)/60:.1f} min")
    E = pd.DataFrame(res)
    meta = D.loc[idx, ["inst_id", "date", "year", "sample", "glob_obj", "glob_time", "glob_status", "glob_gap",
                       "ip_obj", "ip_hit", "ip_time", "ip_viol", "ip_trap_rate", "act_x", "act_y", "act_z", "bind"]]
    E = meta.reset_index(drop=True).merge(E, on="inst_id")
    # structure distance between (b)-implied active set / classifier and the true optimal structure
    true_s = Sl[idx]; mk = Ms[idx] > 0
    b_act = np.concatenate([(Fb[idx] * Lr.FLOW_SCALE > 1.0), np.zeros((len(idx), Lr.NSPEC), bool)], 1)
    E["ham_clf"] = [int(((p >= 0.5) != (s > 0.5))[m].sum()) for p, s, m in zip(Ps[idx], true_s, mk)]
    E["ham_b_arcs"] = [int((bb[:Lr.NF] != (s[:Lr.NF] > 0.5))[m[:Lr.NF]].sum()) for bb, s, m in zip(b_act, true_s, mk)]
    E["true_struct"] = ["".join("1" if v > 0.5 else "0" for v in s[m]) for s, m in zip(true_s, mk)]
    for c in ["act_x", "act_y", "act_z", "bind", "ip_obj", "ip_hit", "ip_time", "ip_viol"]:
        E[c] = E[c].map(lambda v: list(v))
    write_parquet(E, out / "test_eval.parquet")
    (out / "classifier.json").write_text(json.dumps(clf, indent=1))
    log("saved test_eval.parquet")


if __name__ == "__main__":
    main()
