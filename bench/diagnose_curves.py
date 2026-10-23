"""Diagnostic: per-epoch train / validation MSE (standardised units) for models (a) and (d), seed 0.

    python -P bench/diagnose_curves.py <domain> [--epochs E]

Same preprocessing, backbone and optimiser as bench/run.py, but no early stopping, so the full curves are visible.
Writes bench/results/<domain>/curves.json.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.domain import RESULTS  # noqa: E402
from bench.run import N_CAP, SPLIT_SEED, Trainer, _mlp, _write_json, gate, load_domain, make_splits, reduce_rows  # noqa: E402


def curves(T, kind, seed, epochs):
    torch = T.torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = _mlp(T.dom.X.shape[1], T.dom.Y.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xtr, Ytr, Rtr = T.xs(T.tr), T.ys(T.tr), T.xr(T.tr)
    Xva, Yva, Rva = T.xs(T.va), T.ys(T.va), T.xr(T.va)
    mse = lambda xs, ys, xr: ((T.forward(net, kind, xs, xr)[0] - ys) ** 2).mean()
    tr_c, va_c = [], []
    n = len(T.tr)
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(n)
        for i in range(0, n, T.batch):
            j = perm[i:i + T.batch]
            opt.zero_grad()
            loss = mse(Xtr[j], Ytr[j], Rtr[j])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            tr_c.append(float(mse(Xtr, Ytr, Rtr)))
            va_c.append(float(mse(Xva, Yva, Rva)))
    return {"train_mse": tr_c, "val_mse": va_c, "best_val_epoch": int(np.argmin(va_c)) + 1,
            "best_val": float(min(va_c)), "min_train": float(min(tr_c))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    dom = load_domain(a.domain)
    if dom.X.shape[0] > N_CAP:
        keep = np.sort(np.random.default_rng(SPLIT_SEED).choice(dom.X.shape[0], N_CAP, replace=False))
        dom.X, dom.Y = dom.X[keep], dom.Y[keep]
        if dom.split is not None:
            dom.split = dom.split[keep]
    g = gate(dom)
    Ar, Br, br, *_ = reduce_rows(dom.A, dom.B, dom.b)
    tr, va, te, _ = make_splits(dom)
    T = Trainer(dom, tr, va, Ar, Br, br, np.asarray(g["scale"]))
    out = {"domain": a.domain, "epochs": a.epochs, "seed": 0}
    for kind in "ad":
        t0 = time.perf_counter()
        out[kind] = curves(T, kind, 0, a.epochs)
        print(time.strftime("%H:%M:%S"), a.domain, kind, "best val epoch", out[kind]["best_val_epoch"],
              "best val", round(out[kind]["best_val"], 5), "min train", round(out[kind]["min_train"], 5),
              f"{time.perf_counter() - t0:.0f}s", flush=True)
    _write_json(RESULTS / a.domain / "curves.json", out)


if __name__ == "__main__":
    main()
