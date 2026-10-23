"""Methane/air autoignition in a constant-pressure adiabatic reactor (Cantera, GRI-Mech 3.0).

Inputs X  : T0 [K], P [atm], phi [-] and the initial mass fractions of the fresh-mixture species
            (CH4, O2, N2, AR).
Outputs Y : mass fractions of all 53 GRI-Mech 3.0 species after a fixed residence time.
Equality  : element mass fractions are conserved and the mass fractions sum to one,
                A_e Y - A_e[:, fresh] Y0_fresh = 0      (one row per element O, H, C, N, Ar)
                1' Y = 1
            with A_ek = n_ek * W_e / W_k.  The initial element mass fractions are nonlinear in phi,
            which is why Y0 of the fresh species is part of X: it makes B fixed.  The sum row equals
            the sum of the element rows (sum_e n_ek W_e = W_k), so rank(A) = 5.
"""
from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np

from bench.domain import CACHE, Domain

N = 20_000
SEED = 20240611
TAU = 1.0e-3                 # residence time [s]
RTOL, ATOL = 1e-10, 1e-20
T_RANGE = (1000.0, 1600.0)   # K
P_RANGE = (1.0, 20.0)        # atm
PHI_RANGE = (0.5, 2.0)
AIR = "O2:0.21, N2:0.78, AR:0.01"   # mole fractions, dry air with argon lumped into 1 %
FRESH = ("CH4", "O2", "N2", "AR")
MECH = "gri30.yaml"
WORKERS = 6
CACHE_FILE = CACHE / "combustion.npz"
CONFIG = f"n={N};seed={SEED};tau={TAU};rtol={RTOL};atol={ATOL};T={T_RANGE};P={P_RANGE};phi={PHI_RANGE};air={AIR}"

_gas = None


def _init_worker():
    global _gas
    import cantera as ct
    _gas = ct.Solution(MECH)


def _simulate(args):
    """One reactor run.  Returns (Y0, Y, T_final, T_eq, ok)."""
    import cantera as ct
    T0, P, phi = args
    gas = _gas
    nsp = gas.n_species
    try:
        gas.TP = T0, P * ct.one_atm
        gas.set_equivalence_ratio(phi, "CH4", AIR)
        Y0 = gas.Y.copy()
        gas.equilibrate("HP")
        T_eq = gas.T
        gas.TPY = T0, P * ct.one_atm, Y0
        r = ct.IdealGasConstPressureReactor(gas, clone=False)
        net = ct.ReactorNet([r])
        net.rtol, net.atol = RTOL, ATOL
        net.advance(TAU)
        Y = r.phase.Y.copy()
        T = r.phase.T
        ok = bool(np.isfinite(Y).all() and np.isfinite(T))
        return Y0, Y, T, T_eq, ok
    except Exception:
        return np.full(nsp, np.nan), np.full(nsp, np.nan), np.nan, np.nan, False


def _generate():
    rng = np.random.default_rng(SEED)
    inp = np.column_stack([rng.uniform(*T_RANGE, N), rng.uniform(*P_RANGE, N), rng.uniform(*PHI_RANGE, N)])
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(WORKERS, initializer=_init_worker) as pool:
        out = pool.map(_simulate, [tuple(r) for r in inp], chunksize=50)
    wall = time.perf_counter() - t0
    Y0 = np.array([o[0] for o in out])
    Y = np.array([o[1] for o in out])
    T = np.array([o[2] for o in out])
    Teq = np.array([o[3] for o in out])
    ok = np.array([o[4] for o in out])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_FILE, inp=inp, Y0=Y0, Y=Y, T=T, Teq=Teq, ok=ok, wall=wall, config=CONFIG)


def _constraint_matrix(gas):
    """A_ek = n_ek W_e / W_k: element mass fraction contribution of species k."""
    W = gas.molecular_weights
    return np.array([[gas.n_atoms(k, e) * gas.atomic_weight(e) / W[k] for k in range(gas.n_species)]
                     for e in gas.element_names])


def load() -> Domain:
    import cantera as ct
    if CACHE_FILE.exists():
        d = np.load(CACHE_FILE)
        if str(d["config"]) != CONFIG:
            d = None
    else:
        d = None
    if d is None:
        _generate()
        d = np.load(CACHE_FILE)

    gas = ct.Solution(MECH)
    species = list(gas.species_names)
    fresh_idx = [species.index(s) for s in FRESH]
    ok = d["ok"]
    n_fail = int((~ok).sum())
    inp, Y0, Y = d["inp"][ok], d["Y0"][ok], d["Y"][ok]
    T, Teq = d["T"][ok], d["Teq"][ok]

    # the fresh mixture contains only the FRESH species
    assert np.abs(np.delete(Y0, fresh_idx, axis=1)).max() == 0.0
    X = np.column_stack([inp, Y0[:, fresh_idx]])

    Ae = _constraint_matrix(gas)
    ne, q = Ae.shape
    A = np.vstack([Ae, np.ones((1, q))])
    B = np.zeros((ne + 1, X.shape[1]))
    B[:ne, 3:] = -Ae[:, fresh_idx]
    b = np.r_[np.zeros(ne), 1.0]

    prog = (T - inp[:, 0]) / (Teq - inp[:, 0])
    n_unign = int((prog < 0.05).sum())
    n_burnt = int((prog >= 0.95).sum())
    n_mid = len(prog) - n_unign - n_burnt

    meta = {
        "source": f"simulated with Cantera {ct.__version__}, GRI-Mech 3.0 ({MECH} bundled with Cantera)",
        "license": "Cantera: BSD-3-Clause. GRI-Mech 3.0: published by UC Berkeley (Smith et al.) and "
                   "freely available for use; no formal open-source license text is distributed with it.",
        "generation": (f"n={N} samples, seed {SEED}; T0 ~ U{T_RANGE} K, P ~ U{P_RANGE} atm, phi ~ U{PHI_RANGE}; "
                       f"CH4/air with air = {AIR} (mole); IdealGasConstPressureReactor (adiabatic) + ReactorNet, "
                       f"advanced to a fixed residence time tau = {TAU} s, rtol={RTOL}, atol={ATOL}; "
                       f"{WORKERS} worker processes; wall time {float(d['wall']):.1f} s; "
                       f"failed integrations dropped: {n_fail}. "
                       f"State mix by temperature progress (T-T0)/(T_eq,HP-T0): unignited (<0.05) {n_unign}, "
                       f"igniting (0.05-0.95) {n_mid}, burnt (>=0.95) {n_burnt}."),
        "units": "T0 in K, P in atm, phi dimensionless, all mass fractions dimensionless (kg/kg)",
        "tier": 1,
        "x_desc": "initial temperature, pressure, equivalence ratio and the initial mass fractions of CH4, O2, N2, AR",
        "y_desc": "mass fractions of all 53 GRI-Mech 3.0 species after the fixed residence time",
        "n_failed": n_fail,
        "state_mix": {"unignited": n_unign, "igniting": n_mid, "burnt": n_burnt},
    }
    return Domain(
        "combustion", X, Y, A, B, b,
        y_lo=np.zeros(q), y_hi=np.ones(q),
        x_names=["T0", "P", "phi"] + [f"Y0_{s}" for s in FRESH],
        y_names=[f"Y_{s}" for s in species],
        row_names=[f"element_{e}" for e in gas.element_names] + ["sum_Y"],
        meta=meta,
    )
