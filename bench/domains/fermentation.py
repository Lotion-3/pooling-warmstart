"""Batch yeast fermentation with a carbon balance (tier 3; scipy solve_ivp).

Three carbon-conserving pathways, written in C-mol so each one conserves carbon by construction:
    growth       : 1 C-mol glucose -> Yxs C-mol biomass + (1 - Yxs) C-mol CO2         (rate mu_max S/(Ks+S) X)
    ethanol      : 3 C-mol glucose -> 2 C-mol ethanol + 1 C-mol CO2  (C6H12O6 -> 2 C2H5OH + 2 CO2)
    glycerol     : 1 C-mol glucose -> 1 C-mol glycerol                                (redox sink)
with Monod uptake for growth, product-inhibited ethanol formation and growth-coupled glycerol.
Inputs : mu_max, Ks, Yxs, q_e (ethanol rate), K_i (ethanol inhibition), q_g, S0, X0.
Outputs: glucose, biomass, ethanol, glycerol and cumulative CO2 (C-mol/L) at 20 times (0.5 .. 48 h).
Equality: total carbon at each time equals the initial carbon, S + X + E + G + CO2 - S0 - X0 = 0.
"""
import multiprocessing as mp

import numpy as np
from scipy.integrate import solve_ivp

from bench.domain import CACHE, Domain

OUT_T = np.linspace(0.5, 48.0, 20)               # h
N_SAMPLES, SEED, WORKERS = 20_000, 606, 6


def rhs(t, s, mu_max, Ks, Yxs, qe, Ki, qg):
    S, X, E, G, C = np.maximum(s, 0.0)
    m = S / (Ks + S)
    r_grow = mu_max * m * X / Yxs                # glucose C-mol/L/h consumed by growth
    r_eth = qe * m * X / (1.0 + E / Ki)          # glucose C-mol/L/h to ethanol
    r_gly = qg * r_grow                          # glucose C-mol/L/h to glycerol
    return [-(r_grow + r_eth + r_gly), Yxs * r_grow, 2.0 / 3.0 * r_eth, r_gly,
            (1.0 - Yxs) * r_grow + 1.0 / 3.0 * r_eth]


def simulate(x):
    mu_max, Ks, Yxs, qe, Ki, qg, S0, X0 = x
    sol = solve_ivp(rhs, (0.0, OUT_T[-1]), [S0, X0, 0.0, 0.0, 0.0], method="LSODA", t_eval=OUT_T,
                    rtol=1e-10, atol=1e-12, args=(mu_max, Ks, Yxs, qe, Ki, qg))
    return sol.y.T.ravel() if sol.success else None


def sample_inputs(rng):
    return np.column_stack([
        rng.uniform(0.1, 0.5, N_SAMPLES),        # mu_max 1/h
        rng.uniform(0.005, 0.05, N_SAMPLES),     # Ks C-mol/L
        rng.uniform(0.3, 0.6, N_SAMPLES),        # Yxs C-mol/C-mol
        rng.uniform(0.5, 3.0, N_SAMPLES),        # q_e C-mol/C-mol/h
        rng.uniform(0.5, 3.0, N_SAMPLES),        # K_i C-mol/L
        rng.uniform(0.02, 0.2, N_SAMPLES),       # q_g (fraction of growth uptake)
        rng.uniform(1.0, 6.0, N_SAMPLES),        # S0 C-mol/L (30-180 g/L glucose)
        rng.uniform(0.005, 0.05, N_SAMPLES)])    # X0 C-mol/L


def generate():
    X = sample_inputs(np.random.default_rng(SEED))
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        res = pool.map(simulate, list(X), chunksize=100)
    ok = np.array([r is not None for r in res])
    return X[ok], np.stack([r for r in res if r is not None]), int((~ok).sum())


def load():
    path = CACHE / "fermentation.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        X, Y, n_failed = generate()
        np.savez(path, X=X, Y=Y, n_failed=n_failed)
    z = np.load(path)
    X, Y = z["X"], z["Y"]
    nt = len(OUT_T)
    A = np.kron(np.eye(nt), np.ones((1, 5)))
    B = np.zeros((nt, X.shape[1]))
    B[:, 6] = B[:, 7] = -1.0
    states = ["glucose", "biomass", "ethanol", "glycerol", "CO2"]
    return Domain(
        "fermentation", X, Y, A, B, np.zeros(nt), y_lo=np.zeros(5 * nt),
        x_names=["mu_max", "Ks", "Yxs", "q_e", "K_i", "q_g", "S0", "X0"],
        y_names=[f"{s}_t{t:.2f}" for t in OUT_T for s in states],
        row_names=[f"carbon_t{t:.2f}" for t in OUT_T],
        meta={"source": "synthetic: batch fermentation ODE in C-mol (scipy solve_ivp, LSODA, rtol 1e-10, atol 1e-12)",
              "license": "generated here (MIT, as the repo)", "tier": 3, "units": "C-mol/L; rates 1/h",
              "generation": ("growth, ethanol and glycerol pathways, each carbon-conserving; mu_max U(0.1,0.5), "
                             "Ks U(0.005,0.05), Yxs U(0.3,0.6), q_e U(0.5,3), K_i U(0.5,3), q_g U(0.02,0.2), "
                             f"S0 U(1,6), X0 U(0.005,0.05); outputs at {nt} times 0.5..48 h. {N_SAMPLES} samples, "
                             f"seed {SEED}; failed integrations: {int(z['n_failed'])}."),
              "x_desc": "6 kinetic parameters and initial glucose and biomass",
              "y_desc": "glucose, biomass, ethanol, glycerol, cumulative CO2 at 20 times (100)"})
