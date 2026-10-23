"""Metabolic domain: parsimonious FBA fluxes of the E. coli core model under random media.

Inputs X are lower bounds of uptake exchanges (plus the ATP maintenance lower bound); outputs Y
are all 95 reaction fluxes of the pFBA solution.  Steady state gives the fixed equality
    S v = 0      (A = stoichiometric matrix, metabolites x reactions; B = 0, b = 0).

pFBA: (1) maximise biomass, (2) fix biomass at its optimum and minimise sum |v|.  The L1 step
can have tied optima (alternative pathways of equal total flux), so a third LP fixes sum |v|
at its optimum and minimises a weighted L1 norm with fixed generic positive weights.  This
selects one optimal pFBA vertex deterministically, so the input -> output map is a function.
"""
from __future__ import annotations

import os
import time

import numpy as np

from bench.domain import CACHE, Domain

N = 20_000
SEED = 2024
WORKERS = 6
BIOMASS = "Biomass_Ecoli_core"
# (reaction id, low, high): lower bound sampled uniformly in [low, high]; mmol/gDW/h
SAMPLED = (
    ("EX_glc__D_e", -20.0, -1.0),   # glucose uptake
    ("EX_o2_e", -30.0, 0.0),        # oxygen uptake (0 = anaerobic)
    ("EX_fru_e", -10.0, 0.0),       # fructose co-substrate
    ("EX_ac_e", -10.0, 0.0),        # acetate co-substrate
    ("EX_nh4_e", -10.0, -1.0),      # ammonium (can limit growth)
    ("EX_pi_e", -10.0, -1.0),       # phosphate (can limit growth)
    ("ATPM", 0.0, 15.0),            # non-growth ATP maintenance demand
)
L1_SLACK = 1e-9      # relative slack when fixing the pFBA L1 optimum for the tie-break step
TOL = 1e-9           # GLPK primal (tol_bnd) and dual (tol_dj) feasibility tolerances
CONFIG = f"v1|{N}|{SEED}|{SAMPLED}|{L1_SLACK}|{TOL}"

_MODEL = None
_WEIGHTS = None


def _load_model():
    import cobra
    model = cobra.io.load_model("textbook")      # bundled e_coli_core, no download
    model.solver = "glpk"
    model.tolerance = TOL
    model.solver.configuration._smcp.tol_dj = TOL
    return model


def _init_worker(weights):
    global _MODEL, _WEIGHTS
    _MODEL, _WEIGHTS = _load_model(), weights


def _solve_one(model, weights, x):
    """pFBA with lexicographic weighted-L1 tie-break.  Returns (fluxes, status)."""
    from optlang.symbolics import Zero

    rxns = model.reactions
    with model:
        for (rid, _, _), v in zip(SAMPLED, x):
            rxns.get_by_id(rid).lower_bound = float(v)
        mu = model.slim_optimize(error_value=np.nan)
        if not np.isfinite(mu):
            return None, "infeasible"
        rxns.get_by_id(BIOMASS).lower_bound = mu
        prob = model.problem
        fr = [v for r in rxns for v in (r.forward_variable, r.reverse_variable)]
        model.objective = prob.Objective(Zero, direction="min", sloppy=True)
        model.objective.set_linear_coefficients({v: 1.0 for v in fr})
        l1 = model.slim_optimize(error_value=np.nan)
        if not np.isfinite(l1):
            return None, "failed_l1"
        cons = prob.Constraint(Zero, ub=l1 + L1_SLACK * max(1.0, l1), name="pfba_l1_fix")
        model.add_cons_vars(cons)
        model.solver.update()
        cons.set_linear_coefficients({v: 1.0 for v in fr})
        model.objective = prob.Objective(Zero, direction="min", sloppy=True)
        model.objective.set_linear_coefficients(
            {v: w for r, w in zip(rxns, weights) for v in (r.forward_variable, r.reverse_variable)})
        if not np.isfinite(model.slim_optimize(error_value=np.nan)):
            return None, "failed_tiebreak"
        pv = model.solver.primal_values
        flux = np.array([pv[r.id] - pv[r.reverse_id] for r in rxns])
    return flux, "optimal"


def _solve_chunk(Xc):
    out = np.full((len(Xc), len(_MODEL.reactions)), np.nan)
    status = []
    for i, x in enumerate(Xc):
        f, s = _solve_one(_MODEL, _WEIGHTS, x)
        status.append(s)
        if f is not None:
            out[i] = f
    return out, status


def _generate(n=N, seed=SEED):
    import multiprocessing as mp

    import cobra

    model = _load_model()
    S = cobra.util.create_stoichiometric_matrix(model, array_type="dense")
    rng = np.random.default_rng(seed)
    weights = rng.uniform(1.0, 2.0, len(model.reactions))
    lo = np.array([s[1] for s in SAMPLED])
    hi = np.array([s[2] for s in SAMPLED])
    Xs, Ys, counts = [], [], {}
    have, n_cand, t0 = 0, 0, time.perf_counter()
    with mp.get_context("spawn").Pool(min(WORKERS, os.cpu_count() or 1),
                                      initializer=_init_worker, initargs=(weights,)) as pool:
        while have < n:
            Xc = lo + (hi - lo) * rng.random((int(1.05 * (n - have)) + 50, len(SAMPLED)))
            res = pool.map(_solve_chunk, np.array_split(Xc, 4 * WORKERS))
            Yc = np.vstack([r[0] for r in res])
            st = [s for r in res for s in r[1]]
            ok = np.array([s == "optimal" for s in st])
            # keep candidates in draw order up to the n-th feasible one
            stop = len(st) if ok.sum() <= n - have else int(np.searchsorted(np.cumsum(ok), n - have)) + 1
            for s in st[:stop]:
                counts[s] = counts.get(s, 0) + 1
            n_cand += stop
            Xs.append(Xc[:stop][ok[:stop]])
            Ys.append(Yc[:stop][ok[:stop]])
            have += int(ok[:stop].sum())
    X, Y = np.vstack(Xs), np.vstack(Ys)
    bounds = np.array([r.bounds for r in model.reactions], float)
    np.savez(CACHE / "metabolic.npz", X=X, Y=Y, A=S, bounds=bounds, config=CONFIG,
             y_names=np.array([r.id for r in model.reactions]),
             row_names=np.array([m.id for m in model.metabolites]),
             n_candidates=n_cand, n_infeasible=counts.get("infeasible", 0),
             n_failed=n_cand - have - counts.get("infeasible", 0),
             gen_seconds=time.perf_counter() - t0)


def load() -> Domain:
    path = CACHE / "metabolic.npz"
    if path.exists():
        with np.load(path) as z:
            stale = str(z["config"]) != CONFIG
        if stale:
            path.unlink()
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        _generate()
    z = dict(np.load(path))
    y_names = [str(s) for s in z["y_names"]]

    # Fixed output bounds: the model's bounds, with each sampled lower bound replaced by the
    # loosest value in its sampling range; the model's +-1000 "unbounded" sentinel -> +-inf.
    lb, ub = z["bounds"][:, 0].copy(), z["bounds"][:, 1].copy()
    for rid, low, _ in SAMPLED:
        lb[y_names.index(rid)] = low
    lb[lb <= -1000] = -np.inf
    ub[ub >= 1000] = np.inf

    meta = {
        "source": "COBRApy bundled e_coli_core model (cobra.io.load_model('textbook'), "
                  "Orth, Fleming & Palsson 2010, EcoSal Plus; BiGG id e_coli_core)",
        "license": "COBRApy LGPL/GPL; e_coli_core model distributed with COBRApy/BiGG, free for "
                   "research use",
        "generation": (
            f"{N} feasible media drawn i.i.d. uniform (seed {SEED}) over lower bounds "
            + ", ".join(f"{r} in [{a:g}, {b:g}]" for r, a, b in SAMPLED)
            + "; all other bounds at model defaults.  Fluxes from pFBA (max biomass, then min "
              "sum |v| at optimal growth) followed by a lexicographic tie-break LP (sum |v| fixed "
              f"to its optimum with relative slack {L1_SLACK:g}, minimise sum w_i |v_i| with fixed "
              "w_i ~ U(1, 2)).  Solver GLPK via optlang, primal and dual feasibility tolerances "
              f"{TOL:g}.  Candidates: {int(z['n_candidates'])}, infeasible dropped: "
              f"{int(z['n_infeasible'])}, solver failures dropped: {int(z['n_failed'])}; "
              f"generation time {float(z['gen_seconds']):.0f} s."),
        "units": "fluxes and bounds in mmol/gDW/h; biomass flux in 1/h",
        "tier": 1,
        "x_desc": "lower bounds of uptake exchange fluxes (negative = uptake) and ATPM lower bound",
        "y_desc": "all 95 reaction fluxes of the pFBA solution (exchange: negative = uptake)",
        "n_candidates": int(z["n_candidates"]),
        "n_infeasible": int(z["n_infeasible"]),
        "n_failed": int(z["n_failed"]),
        "gen_seconds": float(z["gen_seconds"]),
    }
    return Domain("metabolic", z["X"], z["Y"], z["A"], y_lo=lb, y_hi=ub,
                  x_names=[f"{r}_lb" for r, _, _ in SAMPLED], y_names=y_names,
                  row_names=[str(s) for s in z["row_names"]], meta=meta)
