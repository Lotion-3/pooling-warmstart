"""Toy domain with a known answer: a random linear map Y = X W + c plus 2 exact equalities.

The equalities A y + B x = b hold by construction (B = -A W, b = A c).  Outputs are given very
different magnitudes (1e-2 .. 1e3) so that projecting in standardised space with the original A
would show up as a large violation.
"""
import numpy as np

from bench.domain import Domain


def load(n=6000, p=6, q=10, seed=7, noise=0.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)) * rng.uniform(0.5, 20, p) + rng.uniform(-5, 5, p)
    mag = np.logspace(-2, 3, q)
    W = rng.normal(size=(p, q)) * mag / np.sqrt(p) / X.std(0)[:, None]
    c = rng.normal(size=q) * mag
    Y = X @ W + c
    A = rng.normal(size=(2, q)) / mag          # each output contributes O(1) to each row
    B, b = -A @ W.T, A @ c
    if noise:
        Y = Y + noise * rng.normal(size=Y.shape) * np.abs(Y).mean(0)
    return Domain("toy", X, Y, A, B, b, y_lo=np.full(q, -np.inf), y_hi=np.full(q, np.inf),
                  meta={"source": "synthetic", "license": "n/a", "tier": 0,
                        "generation": "Y = X W + c, A Y + B X = b by construction", "units": "none"})
