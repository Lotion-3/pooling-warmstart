"""SEIRD compartmental epidemic model (scipy solve_ivp).

dS = -beta S I / N,  dE = beta S I / N - sigma E,  dI = sigma E - (gamma + mu) I,  dR = gamma I,  dD = mu I.

Inputs : beta, sigma, gamma, mu, population N and the initial E0, I0, R0 (S0 = N - E0 - I0 - R0, D0 = 0).
Outputs: S, E, I, R, D at 30 days (5, 10, ..., 150).
Equality: S + E + I + R + D - N = 0 at each of the 30 times.
"""
import multiprocessing as mp

import numpy as np
from scipy.integrate import solve_ivp

from bench.domain import CACHE, Domain

OUT_T = np.arange(5.0, 151.0, 5.0)               # days
N_SAMPLES, SEED, WORKERS = 20_000, 404, 4


def rhs(t, s, beta, sigma, gamma, mu, N):
    S, E, I, R, D = s
    f = beta * S * I / N
    return [-f, f - sigma * E, sigma * E - (gamma + mu) * I, gamma * I, mu * I]


def simulate(x):
    beta, sigma, gamma, mu, N, E0, I0, R0 = x
    s0 = [N - E0 - I0 - R0, E0, I0, R0, 0.0]
    sol = solve_ivp(rhs, (0.0, OUT_T[-1]), s0, method="DOP853", t_eval=OUT_T, rtol=1e-10, atol=1e-8,
                    args=(beta, sigma, gamma, mu, N))
    if not sol.success:
        return None
    return sol.y.T.ravel()                         # time-major: (t0: S E I R D), (t1: ...), ...


def sample_inputs(rng):
    beta = rng.uniform(0.1, 0.6, N_SAMPLES)                       # 1/day
    sigma = 1.0 / rng.uniform(2.0, 7.0, N_SAMPLES)                # latent period 2-7 days
    gamma = 1.0 / rng.uniform(4.0, 14.0, N_SAMPLES)               # infectious period 4-14 days
    mu = rng.uniform(0.001, 0.02, N_SAMPLES)                      # 1/day
    N = np.exp(rng.uniform(np.log(1e4), np.log(1e6), N_SAMPLES))
    E0 = N * rng.uniform(1e-4, 1e-2, N_SAMPLES)
    I0 = N * rng.uniform(1e-4, 1e-2, N_SAMPLES)
    R0 = N * rng.uniform(0.0, 0.3, N_SAMPLES)
    return np.column_stack([beta, sigma, gamma, mu, N, E0, I0, R0])


def generate():
    X = sample_inputs(np.random.default_rng(SEED))
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        res = pool.map(simulate, list(X), chunksize=100)
    ok = np.array([r is not None for r in res])
    return X[ok], np.stack([r for r in res if r is not None]), int((~ok).sum())


def load():
    path = CACHE / "epidemic.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        X, Y, n_failed = generate()
        np.savez(path, X=X, Y=Y, n_failed=n_failed)
    z = np.load(path)
    X, Y = z["X"], z["Y"]
    nt = len(OUT_T)
    A = np.kron(np.eye(nt), np.ones((1, 5)))
    B = np.zeros((nt, X.shape[1]))
    B[:, 4] = -1.0
    states = ["S", "E", "I", "R", "D"]
    return Domain(
        "epidemic", X, Y, A, B, np.zeros(nt), y_lo=np.zeros(5 * nt),
        x_names=["beta", "sigma", "gamma", "mu", "N", "E0", "I0", "R0"],
        y_names=[f"{s}_d{t:g}" for t in OUT_T for s in states],
        row_names=[f"population_d{t:g}" for t in OUT_T],
        meta={"source": "synthetic: SEIRD ODE (scipy solve_ivp, DOP853, rtol 1e-10, atol 1e-8)",
              "license": "generated here (MIT, as the repo)", "tier": 1, "units": "people; rates 1/day",
              "generation": ("beta U(0.1,0.6); sigma = 1/U(2,7); gamma = 1/U(4,14); mu U(0.001,0.02); "
                             "N logU(1e4,1e6); E0, I0 = N*U(1e-4,1e-2); R0 = N*U(0,0.3); D0 = 0; outputs at days "
                             f"5..150 step 5. {N_SAMPLES} samples, seed {SEED}; failed integrations: "
                             f"{int(z['n_failed'])}."),
              "x_desc": "beta, sigma, gamma, mu, N, E0, I0, R0",
              "y_desc": "S, E, I, R, D at 30 times (150)"})
