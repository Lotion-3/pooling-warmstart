"""Benchmark harness: gate, rank check, four models, metrics.

    python -P bench/run.py <domain> [--seeds N] [--threads T] [--max-epochs E]

Models (same 3x256 ReLU MLP backbone, Adam, early stopping on validation):
  a  plain MLP
  b  MLP + soft penalty lam * mean((A y + B x - b) / scale)^2, lam tuned on validation over {0.1, 1, 10}
  c  model (a) followed by the KKT projection at test time only
  d  KKT-hPINN: projection layer y = yhat - A'(AA')^-1 (A yhat + B x - b), trained end to end
Training runs in float32; de-standardisation, projection and all metrics run in float64, and the
projection always acts on outputs in ORIGINAL units.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.domain import RESULTS, Domain, residual, term_scale  # noqa: E402

GATE_PASS = 1e-6
GATE_PROJECT = 1e-4
DEGENERATE_RATIO = 0.8
VIOL_TOL = 1e-5
N_CAP = 50_000
SPLIT_SEED = 12345
LAMBDAS = (0.1, 1.0, 10.0)
SCALE_FLOOR = 1e-3


# ----------------------------------------------------------------------------- constraints

def reduce_rows(A, B, b):
    """Drop redundant rows of A (QR with column pivoting on A').  Returns the reduced system,
    the rank and the singular values of A."""
    s = np.linalg.svd(A, compute_uv=False)
    tol = s.max() * max(A.shape) * np.finfo(float).eps if s.size else 0.0
    rank = int((s > tol).sum())
    _, _, piv = sla.qr(A.T, mode="economic", pivoting=True)
    keep = np.sort(piv[:rank])
    return A[keep], B[keep], b[keep], rank, keep, s


class Projector:
    """Euclidean projection onto {y : A y + B x = b} in original units (float64).
    y = yhat - K (A yhat + B x - b) with K = A'(AA')^-1 = pinv(A) for full-row-rank A."""

    def __init__(self, A, B, b, w=None):
        """w (optional, q): weights of a weighted projection, min sum (y - yhat)^2 / w s.t. the equalities.
        w = None is the plain Euclidean projection in original units used by models (c) and (d)."""
        self.A, self.B, self.b = A, B, b
        if w is None:
            self.K = np.linalg.pinv(A)                 # q x m
        else:
            D = np.sqrt(np.asarray(w, float))
            self.K = D[:, None] * np.linalg.pinv(A * D[None, :])
        self.P = np.eye(A.shape[1]) - self.K @ A       # q x q

    def __call__(self, yhat, x):
        return yhat @ self.P.T - (x @ self.B.T - self.b) @ self.K.T

    def torch(self):
        import torch
        P = torch.tensor(self.P, dtype=torch.float64)
        K = torch.tensor(self.K, dtype=torch.float64)
        B = torch.tensor(self.B, dtype=torch.float64)
        b = torch.tensor(self.b, dtype=torch.float64)
        return lambda yhat, x: yhat @ P.T - (x @ B.T - b) @ K.T


def rel_violation(A, B, b, scale, X, Y):
    """Per-sample max over rows of |A y + B x - b| / (scale + 1e-12)."""
    return (np.abs(residual(A, B, b, X, Y)) / (scale + 1e-12)).max(1)


def gate(dom: Domain) -> dict:
    """Check the identity on all data.  Projects the labels if 1e-6 < residual <= 1e-4."""
    scale = term_scale(dom.A, dom.B, dom.b, dom.X, dom.Y)
    # rows whose terms are (almost) always zero, e.g. a metabolite in an unused pathway, would turn any tiny
    # prediction error into an enormous relative residual: floor the scale at 1e-3 x the median row scale
    floor = SCALE_FLOOR * np.median(scale)
    n_floored = int((scale < floor).sum())
    scale = np.maximum(scale, floor)
    v = rel_violation(dom.A, dom.B, dom.b, scale, dom.X, dom.Y)
    info = {"scale": scale.tolist(), "resid_max": float(v.max()), "resid_mean": float(v.mean()),
            "resid_p99": float(np.quantile(v, 0.99)), "frac_above_1e-6": float((v > GATE_PASS).mean()),
            "labels_projected": False, "rows_scale_floored": n_floored}
    if v.max() <= GATE_PASS:
        info["status"] = "pass"
    elif v.max() <= GATE_PROJECT:
        Ar, Br, br, *_ = reduce_rows(dom.A, dom.B, dom.b)
        dom.Y = Projector(Ar, Br, br)(dom.Y, dom.X)
        v2 = rel_violation(dom.A, dom.B, dom.b, scale, dom.X, dom.Y)
        info.update(labels_projected=True, resid_max_after=float(v2.max()),
                    status="pass" if v2.max() <= GATE_PASS else "fail")
    else:
        info["status"] = "fail"
    return info


# ----------------------------------------------------------------------------- splits

def make_splits(dom: Domain, seed=SPLIT_SEED):
    n = dom.X.shape[0]
    rng = np.random.default_rng(seed)
    if dom.split_kind is None:
        perm = rng.permutation(n)
        a, c = int(0.70 * n), int(0.85 * n)
        return np.sort(perm[:a]), np.sort(perm[a:c]), np.sort(perm[c:]), "random 70/15/15"
    keys, inv, counts = np.unique(dom.split, return_inverse=True, return_counts=True)
    order = np.arange(len(keys)) if dom.split_kind == "time" else rng.permutation(len(keys))
    cum = np.cumsum(counts[order]) / n
    part = np.empty(len(keys), int)
    part[order] = np.where(cum <= 0.70 + 1e-12, 0, np.where(cum <= 0.85 + 1e-12, 1, 2))
    assert len(keys) >= 3, "need at least 3 split keys"
    # with coarse keys a cut can leave val or test empty: give them the last keys in split order
    if (part[order] == 2).sum() == 0:
        part[order[-1]] = 2
    if (part[order] == 1).sum() == 0:
        part[order[np.where(part[order] == 2)[0][0] - 1]] = 1
    lab = part[inv]
    desc = f"{dom.split_kind} split on {len(keys)} keys"
    if dom.split_kind == "time":
        desc += (f": train <= {keys[part == 0].max()}, val {keys[part == 1].min()}..{keys[part == 1].max()},"
                 f" test >= {keys[part == 2].min()}")
    return np.where(lab == 0)[0], np.where(lab == 1)[0], np.where(lab == 2)[0], desc


# ----------------------------------------------------------------------------- models

def _mlp(p, q, width=256, depth=3):
    import torch.nn as nn
    layers, d = [], p
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.ReLU()]
        d = width
    layers.append(nn.Linear(d, q))
    return nn.Sequential(*layers)


class Trainer:
    def __init__(self, dom, tr, va, Ar, Br, br, scale_full, max_epochs=300, patience=20, batch=256):
        import torch
        self.torch = torch
        self.dom, self.tr, self.va = dom, tr, va
        X, Y = dom.X, dom.Y
        self.xmu, self.xsd = X[tr].mean(0), X[tr].std(0)
        self.xsd[self.xsd < 1e-12] = 1.0
        self.ymu, self.ysd = Y[tr].mean(0), Y[tr].std(0)
        self.ysd[self.ysd < 1e-12] = 1.0
        self.proj = Projector(Ar, Br, br)
        self.proj_t = self.proj.torch()
        t64 = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)
        self.Af, self.Bf, self.bf = t64(dom.A), t64(dom.B), t64(dom.b)
        self.scale_t = t64(scale_full + 1e-12)
        self.ymu_t, self.ysd_t = t64(self.ymu), t64(self.ysd)
        self.max_epochs, self.patience, self.batch = max_epochs, patience, batch

    def xs(self, idx):
        return self.torch.tensor((self.dom.X[idx] - self.xmu) / self.xsd, dtype=self.torch.float32)

    def ys(self, idx):
        return self.torch.tensor((self.dom.Y[idx] - self.ymu) / self.ysd, dtype=self.torch.float64)

    def xr(self, idx):
        return self.torch.tensor(self.dom.X[idx], dtype=self.torch.float64)

    def forward(self, net, kind, xs, xr):
        """Returns (standardised output, original-units output), both float64."""
        y = net(xs).double() * self.ysd_t + self.ymu_t       # de-standardise BEFORE projecting
        if kind == "d":
            y = self.proj_t(y, xr)
        return (y - self.ymu_t) / self.ysd_t, y

    def penalty(self, y, xr):
        r = (y @ self.Af.T + xr @ self.Bf.T - self.bf) / self.scale_t
        return (r ** 2).mean()

    def fit(self, kind, seed, lam=0.0):
        torch = self.torch
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = _mlp(self.dom.X.shape[1], self.dom.Y.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        Xtr, Ytr, Rtr = self.xs(self.tr), self.ys(self.tr), self.xr(self.tr)
        Xva, Yva, Rva = self.xs(self.va), self.ys(self.va), self.xr(self.va)

        def objective(xs, ys, xr):
            out_s, y = self.forward(net, kind, xs, xr)
            loss = ((out_s - ys) ** 2).mean()
            if kind == "b":
                loss = loss + lam * self.penalty(y, xr)
            return loss

        best, best_state, bad, ep = np.inf, None, 0, 0
        t0 = time.perf_counter()
        n = len(self.tr)
        for ep in range(1, self.max_epochs + 1):
            net.train()
            perm = rng.permutation(n)
            for i in range(0, n, self.batch):
                j = perm[i:i + self.batch]
                opt.zero_grad()
                loss = objective(Xtr[j], Ytr[j], Rtr[j])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(objective(Xva, Yva, Rva))
            if best_state is None or v < best - 1e-7 * abs(best):
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        return net, {"train_seconds": time.perf_counter() - t0, "epochs": ep, "best_val_objective": best}

    def predict(self, net, kind, idx):
        with self.torch.no_grad():
            _, y = self.forward(net, kind, self.xs(idx), self.xr(idx))
        return y.numpy()


# ----------------------------------------------------------------------------- metrics

def nrmse_parts(Y, P, sd, valid):
    rmse = np.sqrt(((P - Y) ** 2).mean(0))
    return float((rmse[valid] / sd[valid]).mean())


def bound_violation(dom, P, sd):
    lo = dom.y_lo if dom.y_lo is not None else np.full(P.shape[1], -np.inf)
    hi = dom.y_hi if dom.y_hi is not None else np.full(P.shape[1], np.inf)
    if not (np.isfinite(lo).any() or np.isfinite(hi).any()):
        return None
    v = np.maximum(np.maximum(lo - P, P - hi), 0.0)                   # original units
    vn = v / np.where(sd > 0, sd, 1.0)                                # relative to output std
    per = vn.max(1)
    return {"pct_points": float(100 * (per > VIOL_TOL).mean()), "max_abs": float(v.max()),
            "max_rel_std": float(vn.max()), "pct_entries": float(100 * (vn > VIOL_TOL).mean())}


def eval_model(dom, te, P, scale, sd, valid):
    v = rel_violation(dom.A, dom.B, dom.b, scale, dom.X[te], P)
    Yte = dom.Y[te]
    return {"nrmse": nrmse_parts(Yte, P, sd, valid),
            # pooled over outputs in ORIGINAL units: ||P - Y||_F / ||Y - mean(Y)||_F (scale-weighted)
            "pooled_nrmse": float(np.sqrt(((P - Yte) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum())),
            "eq_viol_max": float(v.max()), "eq_viol_mean": float(v.mean()),
            "eq_viol_pct_above_1e-5": float(100 * (v > VIOL_TOL).mean()),
            "bounds": bound_violation(dom, P, sd)}


def bootstrap_diff(Yte, preds_1, preds_2, sd, valid, n_boot=1000, seed=0):
    """95% CI over test points for mean-over-seeds nRMSE(1) - nRMSE(2)."""
    rng = np.random.default_rng(seed)
    e1 = np.stack([(p - Yte) ** 2 for p in preds_1])[:, :, valid]    # S x n x q'
    e2 = np.stack([(p - Yte) ** 2 for p in preds_2])[:, :, valid]
    s = sd[valid]
    n = Yte.shape[0]
    out = np.empty(n_boot)
    for k in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n) / n
        r1 = np.sqrt(np.einsum("snq,n->sq", e1, w)) / s
        r2 = np.sqrt(np.einsum("snq,n->sq", e2, w)) / s
        out[k] = r1.mean() - r2.mean()
    full = (np.sqrt(e1.mean(1)) / s).mean() - (np.sqrt(e2.mean(1)) / s).mean()
    return {"diff": float(full), "ci_lo": float(np.quantile(out, 0.025)), "ci_hi": float(np.quantile(out, 0.975))}


# ----------------------------------------------------------------------------- driver

def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    os.replace(tmp, path)


def append_summary(row: dict, results_dir: Path):
    """Replace this domain's row in summary.csv (lock file guards concurrent writers)."""
    import pandas as pd
    path, lock = results_dir / "summary.csv", results_dir / "summary.csv.lock"
    for _ in range(600):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL)
            break
        except FileExistsError:
            time.sleep(0.1)
    else:
        raise RuntimeError("summary.csv lock held for 60 s")
    try:
        df = pd.read_csv(path) if path.exists() else pd.DataFrame()
        if len(df):
            df = df[df["domain"] != row["domain"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(path, index=False)
    finally:
        os.close(fd)
        os.remove(lock)


def load_domain(name) -> Domain:
    return importlib.import_module(f"bench.domains.{name}").load()


def run_domain(dom: Domain, seeds=None, max_epochs=300, patience=20, results_dir: Path = RESULTS,
               write=True, log=print):
    t_start = time.perf_counter()
    n0 = dom.X.shape[0]
    if n0 > N_CAP:
        keep = np.sort(np.random.default_rng(SPLIT_SEED).choice(n0, N_CAP, replace=False))
        dom.X, dom.Y = dom.X[keep], dom.Y[keep]
        if dom.split is not None:
            dom.split = dom.split[keep]
    n, p, q, m = dom.shape
    out = {"domain": dom.name, "meta": dom.meta, "n_raw": n0, "n": n, "p": p, "q": q, "m": m}

    g = gate(dom)
    out["gate"] = {k: v for k, v in g.items() if k != "scale"}
    scale = np.asarray(g["scale"])
    Ar, Br, br, rank, keep, svals = reduce_rows(dom.A, dom.B, dom.b)
    out["rank"] = {"rank": rank, "m": m, "q": q, "ratio": rank / q, "rows_kept": keep.tolist(),
                   "sv_min_kept": float(svals[rank - 1]) if rank else None, "sv_max": float(svals.max())}
    log(f"[{dom.name}] n={n} p={p} q={q} m={m} rank={rank} ({rank / q:.3f} q); gate {g['status']} "
        f"max={g['resid_max']:.3e} projected={g['labels_projected']}")

    row = {"domain": dom.name, "source": dom.meta.get("source", ""), "tier": dom.meta.get("tier", ""),
           "n": n, "p": p, "q": q, "m": m, "rank": rank, "rank_over_q": round(rank / q, 4),
           "gate_resid_max": g["resid_max"], "labels_projected": g["labels_projected"]}
    status = "ok"
    if g["status"] != "pass":
        status = "failed_gate"
    elif rank >= DEGENERATE_RATIO * q:
        status = "degenerate"
    out["status"] = row["status"] = status
    if status != "ok":
        log(f"[{dom.name}] dropped: {status}")
        if write:
            _write_json(results_dir / dom.name / "metrics.json", out)
            append_summary(row, results_dir)
        return out

    tr, va, te, desc = make_splits(dom)
    out["split"] = {"desc": desc, "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
    Yte = dom.Y[te]
    sd = Yte.std(0)
    valid = sd > 1e-12 * np.maximum(1.0, np.abs(Yte).mean(0))
    out["n_outputs_scored"] = int(valid.sum())
    out["label_bounds_test"] = bound_violation(dom, Yte, sd)
    T = Trainer(dom, tr, va, Ar, Br, br, scale, max_epochs=max_epochs, patience=patience)

    # lambda for (b): tuned once on validation (seed 0) by validation nRMSE
    sd_va = dom.Y[va].std(0)
    valid_va = sd_va > 1e-12 * np.maximum(1.0, np.abs(dom.Y[va]).mean(0))
    tune, nets_b0 = {}, {}
    for lam in LAMBDAS:
        net, info = T.fit("b", 0, lam)
        Pva = T.predict(net, "b", va)
        tune[lam] = {"val_nrmse": nrmse_parts(dom.Y[va], Pva, sd_va, valid_va), **info}
        nets_b0[lam] = (net, info)
        log(f"[{dom.name}] tune lam={lam}: val nRMSE {tune[lam]['val_nrmse']:.5f} ({info['train_seconds']:.1f}s)")
    lam = min(LAMBDAS, key=lambda l: tune[l]["val_nrmse"])
    out["lambda"] = {"chosen": lam, "tuning": {str(k): v for k, v in tune.items()}}

    per_seed, preds = [], {k: [] for k in "abcd"}
    seed_list = list(range(seeds)) if seeds else None
    s = 0
    while True:
        t_seed = time.perf_counter()
        rec = {"seed": s}
        net_a, ia = T.fit("a", s)
        Pa = T.predict(net_a, "a", te)
        if s == 0:
            net_b, ib = nets_b0[lam]
        else:
            net_b, ib = T.fit("b", s, lam)
        Pb = T.predict(net_b, "b", te)
        t0 = time.perf_counter()
        Pc = T.proj(Pa, dom.X[te])
        tc = time.perf_counter() - t0
        net_d, idd = T.fit("d", s)
        Pd = T.predict(net_d, "d", te)
        for k, P_, info in (("a", Pa, ia), ("b", Pb, ib), ("c", Pc, {"train_seconds": ia["train_seconds"] + tc,
                                                                   "projection_seconds": tc}), ("d", Pd, idd)):
            rec[k] = {**eval_model(dom, te, P_, scale, sd, valid), **info}
            preds[k].append(P_)
        rec["d_minus_a"] = rec["d"]["nrmse"] - rec["a"]["nrmse"]
        rec["d_minus_c"] = rec["d"]["nrmse"] - rec["c"]["nrmse"]
        rec["b_minus_a"] = rec["b"]["nrmse"] - rec["a"]["nrmse"]
        rec["seed_seconds"] = time.perf_counter() - t_seed
        per_seed.append(rec)
        log(f"[{dom.name}] seed {s}: nRMSE a={rec['a']['nrmse']:.5f} b={rec['b']['nrmse']:.5f} "
            f"c={rec['c']['nrmse']:.5f} d={rec['d']['nrmse']:.5f}  pooled a={rec['a']['pooled_nrmse']:.5f} "
            f"c={rec['c']['pooled_nrmse']:.5f} d={rec['d']['pooled_nrmse']:.5f}  viol% a={rec['a']['eq_viol_pct_above_1e-5']:.1f}"
            f" c={rec['c']['eq_viol_pct_above_1e-5']:.1f} d={rec['d']['eq_viol_pct_above_1e-5']:.1f} "
            f"({rec['seed_seconds']:.0f}s)")
        if seed_list is None:  # 5 seeds if seed 0 (a + b + d, incl. lambda tuning) took < 5 min, else 3
            first = rec["seed_seconds"] + sum(tune[l]["train_seconds"] for l in LAMBDAS if l != lam)
            seed_list = list(range(5 if first < 300 else 3))
            out["seed_rule"] = {"seed0_seconds": first, "n_seeds": len(seed_list)}
        s += 1
        if s >= len(seed_list):
            break

    out["per_seed"] = per_seed
    agg = {}
    for k in "abcd":
        vals = {key: np.array([r[k][key] for r in per_seed]) for key in
                ("nrmse", "pooled_nrmse", "eq_viol_max", "eq_viol_mean", "eq_viol_pct_above_1e-5", "train_seconds")}
        agg[k] = {f"{key}_mean": float(v.mean()) for key, v in vals.items()}
        agg[k].update({f"{key}_std": float(v.std(ddof=1)) if len(v) > 1 else 0.0 for key, v in vals.items()})
        agg[k]["eq_viol_max_over_seeds"] = float(vals["eq_viol_max"].max())
        bnds = [r[k]["bounds"] for r in per_seed]
        if bnds[0] is not None:
            agg[k]["bound_pct_points_mean"] = float(np.mean([x["pct_points"] for x in bnds]))
            agg[k]["bound_max_abs"] = float(np.max([x["max_abs"] for x in bnds]))
            agg[k]["bound_max_rel_std"] = float(np.max([x["max_rel_std"] for x in bnds]))
    out["aggregate"] = agg
    out["paired"] = {key: {"per_seed": [r[key] for r in per_seed],
                           "mean": float(np.mean([r[key] for r in per_seed])),
                           "n_seeds_negative": int(sum(r[key] < 0 for r in per_seed))}
                     for key in ("d_minus_a", "d_minus_c", "b_minus_a")}
    out["bootstrap"] = {"d_minus_a": bootstrap_diff(Yte, preds["d"], preds["a"], sd, valid),
                        "d_minus_c": bootstrap_diff(Yte, preds["d"], preds["c"], sd, valid),
                        "c_minus_a": bootstrap_diff(Yte, preds["c"], preds["a"], sd, valid)}
    out["total_seconds"] = time.perf_counter() - t_start

    for k in "abcd":
        row[f"nrmse_{k}_mean"] = agg[k]["nrmse_mean"]
        row[f"nrmse_{k}_std"] = agg[k]["nrmse_std"]
        row[f"pooled_nrmse_{k}_mean"] = agg[k]["pooled_nrmse_mean"]
        row[f"viol_pct_{k}"] = agg[k]["eq_viol_pct_above_1e-5_mean"]
        row[f"viol_max_{k}"] = agg[k]["eq_viol_max_over_seeds"]
        row[f"bound_pct_{k}"] = agg[k].get("bound_pct_points_mean", np.nan)
    row.update(lam=lam, seeds=len(per_seed), split=desc, n_test=len(te),
               boot_d_minus_a=out["bootstrap"]["d_minus_a"]["diff"],
               boot_d_minus_a_lo=out["bootstrap"]["d_minus_a"]["ci_lo"],
               boot_d_minus_a_hi=out["bootstrap"]["d_minus_a"]["ci_hi"],
               boot_d_minus_c=out["bootstrap"]["d_minus_c"]["diff"],
               boot_d_minus_c_lo=out["bootstrap"]["d_minus_c"]["ci_lo"],
               boot_d_minus_c_hi=out["bootstrap"]["d_minus_c"]["ci_hi"],
               total_seconds=round(out["total_seconds"], 1))
    if write:
        _write_json(results_dir / dom.name / "metrics.json", out)
        append_summary(row, results_dir)
        np.savez_compressed(results_dir / dom.name / "test_predictions.npz", test_idx=te, Y=Yte,
                            **{k: np.stack(v).astype(np.float32) for k, v in preds.items()})
    log(f"[{dom.name}] done in {out['total_seconds']:.0f}s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain")
    ap.add_argument("--seeds", type=int, default=None, help="override the 5-or-3 seed rule")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--gate-only", action="store_true")
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    dom = load_domain(args.domain)
    if args.gate_only:
        g = gate(dom)
        _, _, _, rank, _, _ = reduce_rows(dom.A, dom.B, dom.b)
        n, p, q, m = dom.shape
        print(json.dumps({"domain": dom.name, "n": n, "p": p, "q": q, "m": m, "rank": rank,
                          "rank_over_q": rank / q, **{k: v for k, v in g.items() if k != "scale"}}, indent=1))
        return
    run_domain(dom, seeds=args.seeds, max_epochs=args.max_epochs,
               log=lambda s: print(time.strftime("%H:%M:%S"), s, flush=True))


if __name__ == "__main__":
    main()
