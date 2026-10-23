"""Harness tests on the toy domain (known answer).   python -P bench/tests/test_harness.py"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.domain import Domain, residual, term_scale  # noqa: E402
from bench.domains import toy  # noqa: E402
from bench.run import Projector, gate, make_splits, reduce_rows, rel_violation, run_domain  # noqa: E402


def test_projector():
    rng = np.random.default_rng(0)
    q, p = 12, 4
    A0 = rng.normal(size=(3, q))
    A = np.vstack([A0, A0[0] + 2 * A0[1]])            # 4th row redundant
    B0 = rng.normal(size=(3, p))
    B = np.vstack([B0, B0[0] + 2 * B0[1]])
    b0 = rng.normal(size=3)
    b = np.r_[b0, b0[0] + 2 * b0[1]]
    Ar, Br, br, rank, keep, _ = reduce_rows(A, B, b)
    assert rank == 3 and len(keep) == 3
    P = Projector(Ar, Br, br)
    X = rng.normal(size=(50, p))
    Yhat = rng.normal(size=(50, q)) * 100
    Y = P(Yhat, X)
    assert np.abs(residual(A, B, b, X, Y)).max() < 1e-10, "projection must satisfy all rows"
    assert np.allclose(P(Y, X), Y), "projection must be idempotent"
    # it is the closest feasible point: the move is orthogonal to the feasible directions
    N = np.linalg.svd(Ar)[2][rank:].T                 # null space of A
    assert np.abs((Yhat - Y) @ N).max() < 1e-8
    print("projector ok")


def test_gate():
    d = toy.load()
    g = gate(d)
    assert g["status"] == "pass" and g["resid_max"] < 1e-12, g["resid_max"]
    d = toy.load(noise=1e-5)           # solver-tolerance noise -> labels projected, passes
    g = gate(d)
    assert g["labels_projected"] and g["status"] == "pass", g
    v = rel_violation(d.A, d.B, d.b, term_scale(d.A, d.B, d.b, d.X, d.Y), d.X, d.Y)
    assert v.max() < 1e-12
    d = toy.load(noise=1e-2)           # real inconsistency -> fails
    g = gate(d)
    assert g["status"] == "fail" and not g["labels_projected"], g
    print("gate ok")


def test_degenerate_and_splits():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3))
    Y = rng.normal(size=(300, 5))
    A = np.vstack([np.eye(5)[:4]])                    # rank 4 >= 0.8 * 5
    d = Domain("deg", X, Y, A, b=None, B=-np.eye(5)[:4] @ np.zeros((5, 3)))
    d.Y[:, :4] = 0.0
    with tempfile.TemporaryDirectory() as tmp:
        out = run_domain(d, results_dir=Path(tmp), log=lambda s: None)
    assert out["status"] == "degenerate", out["status"]
    # time split: no time key shared between parts, and parts are in time order
    t = np.repeat(np.arange(20), 15)
    d = Domain("t", X, Y, np.ones((1, 5)), b=Y.sum(1).mean() * 0, split=t, split_kind="time")
    tr, va, te, _ = make_splits(d)
    assert t[tr].max() < t[va].min() and t[va].max() < t[te].min()
    assert len(tr) + len(va) + len(te) == 300
    print("degenerate + splits ok")


def test_end_to_end():
    d = toy.load()
    with tempfile.TemporaryDirectory() as tmp:
        out = run_domain(d, seeds=1, max_epochs=150, results_dir=Path(tmp), log=print)
        assert (Path(tmp) / "summary.csv").exists() and (Path(tmp) / "toy" / "metrics.json").exists()
    s = out["per_seed"][0]
    assert out["status"] == "ok"
    assert s["a"]["nrmse"] < 0.05, s["a"]["nrmse"]             # a linear map is easy to learn
    assert s["c"]["eq_viol_max"] < 1e-9 and s["d"]["eq_viol_max"] < 1e-9, (s["c"], s["d"])
    assert s["a"]["eq_viol_pct_above_1e-5"] > 50
    assert s["d"]["nrmse"] < 0.05, s["d"]["nrmse"]
    assert s["c"]["pooled_nrmse"] <= s["a"]["pooled_nrmse"] + 1e-12   # guaranteed in original units
    # projecting onto a set that contains the truth can only reduce the per-sample Euclidean error in
    # ORIGINAL units (nRMSE, which averages per-output errors / std, has no such guarantee)
    d = toy.load()
    Ar, Br, br, *_ = reduce_rows(d.A, d.B, d.b)
    Yhat = d.Y + np.random.default_rng(3).normal(size=d.Y.shape) * d.Y.std(0) * 0.1
    Yp = Projector(Ar, Br, br)(Yhat, d.X)
    assert (((Yp - d.Y) ** 2).sum(1) <= ((Yhat - d.Y) ** 2).sum(1) + 1e-9).all()
    print("end-to-end ok:", {k: round(s[k]["nrmse"], 5) for k in "abcd"})


if __name__ == "__main__":
    test_projector()
    test_gate()
    test_degenerate_and_splits()
    test_end_to_end()
    print("ALL TESTS PASSED")
