"""Phase 5: features, targets and models for warm-start / structure prediction.

Permutation invariance: crudes are stored sorted by API (canonical order), and each crude fills a
fixed slot (max 8, padded + masked). Tank slots are fixed (tanks differ in capacity / cost / links).
Flow vector f = [x (8x3 crude->tank), y (3x2 tank->CDU), z (8x2 crude->CDU)] in kbd / 100.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

MAXI, L, J = 8, 3, 2
NX, NY, NZ = MAXI * L, L * J, MAXI * J
NF = NX + NY + NZ
NSPEC = 4 * J
FLOW_SCALE = 100.0


def _arr(row, k):
    return np.asarray(row[k], float).reshape(tuple(int(s) for s in row[k + "_shape"]))


def features(row) -> np.ndarray:
    I = len(row["crudes"])
    brent = row["brent_europe_usd_per_bbl"]
    q_up, z_up = _arr(row, "q_up"), _arr(row, "z_up")
    per = np.zeros((MAXI, 14))
    cost, worth = np.asarray(row["crude_cost"]), np.asarray(row["crude_worth"])
    lam = _arr(row, "lam")
    per[:I, 0] = row["crude_api"]
    per[:I, 1] = row["crude_s"]
    per[:I, 2] = row["crude_tan"]
    per[:I, 3] = lam[:, 1]
    per[:I, 4] = cost - brent
    per[:I, 5] = worth - brent
    per[:I, 6] = worth - cost
    per[:I, 7] = row["src_up"]
    per[:I, 8:11] = q_up
    per[:I, 11:13] = (z_up > 0)
    per[:I, 13] = 1.0  # slot used
    specs = json.loads(row["cdu_specs"])
    g = [brent,
         row["gasoline_conv_reg_usgc_usd_per_gal"] * 42 - brent, row["ulsd_usgc_usd_per_gal"] * 42 - brent,
         row["jet_kero_usgc_usd_per_gal"] * 42 - brent, row["propane_mont_belvieu_usd_per_gal"] * 42 - brent]
    g += list(row["pool_cap"]) + list(row["h"])
    for c in specs:
        g += [c["cap"], c["min"], c["smax"], c["api_lo"], c["api_hi"], c["tan"]]
    return np.concatenate([per.ravel(), np.asarray(g, float)])


def masks(row) -> np.ndarray:
    I = len(row["crudes"])
    mx = np.zeros((MAXI, L)); mx[:I] = _arr(row, "q_up") > 0
    my = (_arr(row, "y_up") > 0).astype(float)
    mz = np.zeros((MAXI, J)); mz[:I] = _arr(row, "z_up") > 0
    return np.concatenate([mx.ravel(), my.ravel(), mz.ravel()])


def flows_from_solution(row, q, y, z) -> np.ndarray:
    I = len(row["crudes"])
    q = np.asarray(q).reshape(I, L); y = np.asarray(y).reshape(L, J); z = np.asarray(z).reshape(I, J)
    x = q * y.sum(1)[None, :]
    fx = np.zeros((MAXI, L)); fx[:I] = x
    fz = np.zeros((MAXI, J)); fz[:I] = z
    return np.concatenate([fx.ravel(), y.ravel(), fz.ravel()]) / FLOW_SCALE


def labels_structure(row, rec) -> np.ndarray:
    I = len(row["crudes"])
    ax = np.zeros((MAXI, L)); ax[:I] = np.asarray(rec["act_x"]).reshape(I, L)
    ay = np.asarray(rec["act_y"], float)
    az = np.zeros((MAXI, J)); az[:I] = np.asarray(rec["act_z"]).reshape(I, J)
    return np.concatenate([ax.ravel(), ay.ravel(), az.ravel(), np.asarray(rec["bind"], float)])


def flows_to_pq(row, f: np.ndarray):
    """Predicted flow vector (scaled) -> dense pq point for this instance (q from x shares)."""
    I = len(row["crudes"])
    f = np.asarray(f) * FLOW_SCALE
    x = f[:NX].reshape(MAXI, L)[:I]
    y = f[NX:NX + NY].reshape(L, J)
    z = f[NX + NY:].reshape(MAXI, J)[:I]
    q_up = _arr(row, "q_up")
    tot = x.sum(0)
    q = np.zeros((I, L))
    for l in range(L):
        arcs = q_up[:, l] > 0
        if tot[l] > 1e-9:
            q[:, l] = np.where(arcs, x[:, l], 0) / max(x[arcs, l].sum(), 1e-12)
        else:
            q[arcs, l] = 1.0 / arcs.sum()
    return q, y, z, x


def balance_matrix() -> np.ndarray:
    """A f = 0 : sum_i x_il - sum_j y_lj = 0 for each tank."""
    A = np.zeros((L, NF))
    for l in range(L):
        for i in range(MAXI):
            A[l, i * L + l] = 1.0
        for j in range(J):
            A[l, NX + l * J + j] = -1.0
    return A


class MLP(nn.Module):
    def __init__(self, nin, nout, hidden=256, depth=3, dropout=0.1):
        super().__init__()
        layers, d = [], nin
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, nout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FlowNet(nn.Module):
    """MLP flows; optional KKT projection onto {A_m f = 0} restricted to existing arcs (KKThPINN-style)."""
    def __init__(self, nin, project=False, **kw):
        super().__init__()
        self.mlp = MLP(nin, NF, **kw)
        self.project = project
        self.register_buffer("A", torch.tensor(balance_matrix(), dtype=torch.float32))

    def forward(self, x, m):
        f = torch.relu(self.mlp(x)) * m
        if self.project:
            f = kkt_project(f, m, self.A)
        return f


def kkt_project(f, m, A):
    """argmin ||g - f||^2  s.t.  A (m*g) = 0, g = 0 off-arc.  Closed form: g = f - M A^T (A M A^T)^-1 A f."""
    Am = A.unsqueeze(0) * m.unsqueeze(1)                 # (B, L, NF)
    AAt = Am @ Am.transpose(1, 2) + 1e-8 * torch.eye(A.shape[0])
    lam = torch.linalg.solve(AAt, (Am @ f.unsqueeze(-1)))  # (B, L, 1)
    return f - (Am.transpose(1, 2) @ lam).squeeze(-1)


def standardize(train: np.ndarray):
    mu, sd = train.mean(0), train.std(0)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def train_model(model, X, M, Y, Xv, Mv, Yv, loss_kind="mse", epochs=300, lr=1e-3, batch=256, patience=30,
                pos_weight=None, seed=0, log=print):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    T = lambda a: torch.tensor(a, dtype=torch.float32)
    X, M, Y, Xv, Mv, Yv = map(T, (X, M, Y, Xv, Mv, Yv))

    def loss_fn(out, y, m):
        if loss_kind == "mse":
            return (((out - y) ** 2) * m).sum() / m.sum().clamp(min=1)
        w = None if pos_weight is None else torch.tensor(pos_weight, dtype=torch.float32)
        l = nn.functional.binary_cross_entropy_with_logits(out, y, reduction="none", pos_weight=w)
        return (l * m).sum() / m.sum().clamp(min=1)

    def fwd(xb, mb):
        return model(xb, mb) if isinstance(model, FlowNet) else model(xb)

    best, best_state, bad = np.inf, None, 0
    n = len(X)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for k in range(0, n, batch):
            idx = perm[k:k + batch]
            opt.zero_grad()
            loss = loss_fn(fwd(X[idx], M[idx]), Y[idx], M[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(fwd(Xv, Mv), Yv, Mv))
        sched.step(vl)
        if vl < best - 1e-7:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 25 == 0:
            log(f"  epoch {ep:3d} val {vl:.5f} best {best:.5f}")
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    log(f"  stopped at epoch {ep}, best val {best:.5f}")
    return model, best
