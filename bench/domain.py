"""Domain container shared by every benchmark domain.

A domain supplies inputs X (n x p), outputs Y (n x q) and a FIXED linear equality
    A @ y + B @ x = b        (A: m x q, B: m x p, b: m)
that every sample satisfies, plus optional split keys, output bounds and metadata.
Each module in bench/domains/ exposes `load() -> Domain`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parent
RAW = BENCH / "data" / "raw"        # downloaded files (gitignored)
CACHE = BENCH / "data" / "cache"    # generated samples (gitignored, rebuilt by the loaders)
RESULTS = BENCH / "results"


@dataclass
class Domain:
    name: str
    X: np.ndarray
    Y: np.ndarray
    A: np.ndarray
    B: np.ndarray | None = None
    b: np.ndarray | None = None
    split: np.ndarray | None = None       # time or group key per sample
    split_kind: str | None = None         # "time", "group" or None (random 70/15/15)
    y_lo: np.ndarray | None = None        # elementwise bounds on Y (reporting only)
    y_hi: np.ndarray | None = None
    x_names: list | None = None
    y_names: list | None = None
    row_names: list | None = None
    meta: dict = field(default_factory=dict)  # source, license, generation, units, tier

    def __post_init__(self):
        self.X = np.asarray(self.X, np.float64)
        self.Y = np.asarray(self.Y, np.float64)
        self.A = np.atleast_2d(np.asarray(self.A, np.float64))
        n, p = self.X.shape
        n2, q = self.Y.shape
        m = self.A.shape[0]
        assert n == n2, "X and Y must have the same number of rows"
        assert self.A.shape[1] == q, "A must have one column per output"
        self.B = np.zeros((m, p)) if self.B is None else np.atleast_2d(np.asarray(self.B, np.float64))
        self.b = np.zeros(m) if self.b is None else np.asarray(self.b, np.float64).reshape(m)
        assert self.B.shape == (m, p), "B must be m x p"
        if self.split is not None:
            self.split = np.asarray(self.split)
            assert len(self.split) == n and self.split_kind in ("time", "group")
        for k in ("y_lo", "y_hi"):
            v = getattr(self, k)
            if v is not None:
                setattr(self, k, np.broadcast_to(np.asarray(v, np.float64), (q,)).copy())
        assert np.isfinite(self.X).all() and np.isfinite(self.Y).all(), "non-finite data"

    @property
    def shape(self):
        return self.X.shape[0], self.X.shape[1], self.Y.shape[1], self.A.shape[0]


def residual(A, B, b, X, Y):
    """Row residuals A y + B x - b for every sample: (n, m)."""
    return Y @ A.T + X @ B.T - b


def term_scale(A, B, b, X, Y):
    """Typical magnitude of the terms of each constraint row: mean over samples of
    sum_j |A_ij y_j| + sum_k |B_ik x_k| + |b_i|.  Shape (m,)."""
    return (np.abs(Y) @ np.abs(A).T + np.abs(X) @ np.abs(B).T + np.abs(b)).mean(0)
