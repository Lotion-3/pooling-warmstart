"""Steady-state separation flowsheet (IDAES-PSE, ideal benzene-toluene-o-xylene VLE, no reactions).

    feed1 --+
            +--> Mixer --> Flash --(vapour)--> product
    feed2 --+                  \
                                +--(liquid)--> Splitter --> out1, out2

Property package: the IDAES example BTX package (idaes.models.properties.activity_coeff_models.
BTX_activity_coeff_VLE, ideal liquid, ideal gas, component-flow state variables "FcTP") with
o-xylene enabled (its parameters are already in the package's master data).  Feeds and the
splitter are liquid-only; the flash uses the two-phase package with the smooth VLE formulation.

Inputs : component molar flows of both feeds, feed T and P, flash T and P, split fraction to out1.
Outputs: component molar flows of the mixer outlet, flash vapour, flash liquid, splitter out1/out2.
Equality: component mole balances per unit (mixer, flash, splitter); feeds enter through B.
"""
from __future__ import annotations

import json
import os
import shutil
import time

import numpy as np

from bench.domain import CACHE, Domain

N_POOL, SEED, N_WORKERS, CHUNK = 20_000, 2024, 4, 250
N_SAMPLES = 20_000   # first N_SAMPLES rows of the input pool are solved
COMPS = ["benzene", "toluene", "o-xylene"]
FEEDS = ["feed1", "feed2"]
STREAMS = ["mix", "vap", "liq", "out1", "out2"]
IPOPT_OPTIONS = {"tol": 1e-10, "constr_viol_tol": 1e-10, "max_iter": 300, "print_level": 0,
                 "nlp_scaling_method": "user-scaling"}

# sampling ranges
FLOW_RANGE = (0.05, 1.0)      # mol/s per component per feed
FEED_T = (300.0, 340.0)       # K, subcooled at the feed pressures below
FEED_P = (1.5e5, 3.0e5)       # Pa
FLASH_P = (0.5e5, 1.2e5)      # Pa
THETA = (0.1, 0.9)            # flash T = T_bubble + theta (T_dew - T_bubble) of the mixed feed at flash P
SPLIT = (0.1, 0.9)            # fraction of flash liquid sent to out1

IDAES_BIN = os.environ.get("IDAES_BIN", os.path.join(os.path.expanduser("~"), "solvers", "idaes_bin"))
IPOPT_EXE = os.path.join(IDAES_BIN, "ipopt.exe")


def _env():
    if IDAES_BIN not in os.environ.get("PATH", ""):
        os.environ["PATH"] = IDAES_BIN + os.pathsep + os.environ.get("PATH", "")  # ipopt.exe needs its DLLs


# ----------------------------------------------------------------------------- flowsheet

def _btxo_block():
    """BTX example package with all three components of its master list (block class created once,
    lazily, so that loading the cached data does not import IDAES)."""
    if "BTXoParameterBlock" in globals():
        return globals()["BTXoParameterBlock"]
    from pyomo.environ import Set
    from idaes.core import Component, declare_process_block_class
    from idaes.models.properties.activity_coeff_models.BTX_activity_coeff_VLE import BTXParameterData

    @declare_process_block_class("BTXoParameterBlock")
    class BTXoParameterData(BTXParameterData):
        def build(self):
            self.add_component("o-xylene", Component())
            super().build()
            self.del_component(self.phase_equilibrium_idx)
            self.phase_equilibrium_idx = Set(initialize=[1, 2, 3])
            self.phase_equilibrium_list = {i + 1: [c, ("Vap", "Liq")] for i, c in enumerate(COMPS)}

    globals()["BTXoParameterData"] = BTXoParameterData
    return globals()["BTXoParameterBlock"]   # set on this module by the decorator


def build_model():
    _env()
    import pyomo.environ as pyo
    from pyomo.network import Arc
    from idaes.core import FlowsheetBlock, MaterialBalanceType
    from idaes.models.unit_models import Flash, Mixer, MomentumMixingType, Separator
    from idaes.models.unit_models.separator import SplittingType
    import idaes.core.util.scaling as iscale

    BTXo = _btxo_block()
    m = pyo.ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.vle = BTXo(valid_phase=("Liq", "Vap"), activity_coeff_model="Ideal", state_vars="FcTP")
    m.fs.liq = BTXo(valid_phase="Liq", activity_coeff_model="Ideal", state_vars="FcTP")
    m.fs.mix = Mixer(property_package=m.fs.liq, inlet_list=FEEDS, momentum_mixing_type=MomentumMixingType.minimize,
                     material_balance_type=MaterialBalanceType.componentTotal)
    m.fs.flash = Flash(property_package=m.fs.vle, has_heat_transfer=True, has_pressure_change=True)
    m.fs.split = Separator(property_package=m.fs.liq, outlet_list=["out1", "out2"], split_basis=SplittingType.totalFlow,
                           material_balance_type=MaterialBalanceType.componentTotal)
    m.fs.s1 = Arc(source=m.fs.mix.outlet, destination=m.fs.flash.inlet)
    m.fs.s2 = Arc(source=m.fs.flash.liq_outlet, destination=m.fs.split.inlet)
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)
    for blk in (m.fs.vle, m.fs.liq):
        blk.set_default_scaling("flow_mol_comp", 1.0)
        blk.set_default_scaling("flow_mol_phase_comp", 1.0)
    iscale.set_scaling_factor(m.fs.flash.heat_duty, 1e-4)
    iscale.set_scaling_factor(m.fs.flash.deltaP, 1e-5)
    iscale.calculate_scaling_factors(m)
    m.ports = {"mix": m.fs.mix.outlet, "vap": m.fs.flash.vap_outlet, "liq": m.fs.flash.liq_outlet,
               "out1": m.fs.split.out1, "out2": m.fs.split.out2}
    return m


def set_inputs(m, x):
    """x = [feed1 flows (3), feed2 flows (3), T1, P1, T2, P2, T_flash, P_flash, split]."""
    nc = len(COMPS)
    for f, feed in enumerate(FEEDS):
        port = getattr(m.fs.mix, feed)
        for i, c in enumerate(COMPS):
            port.flow_mol_comp[0, c].fix(x[f * nc + i])
        port.temperature[0].fix(x[2 * nc + 2 * f])
        port.pressure[0].fix(x[2 * nc + 2 * f + 1])
    m.fs.flash.control_volume.properties_out[0].temperature.fix(x[2 * nc + 4])
    m.fs.flash.control_volume.properties_out[0].pressure.fix(x[2 * nc + 5])
    m.fs.split.split_fraction[0, "out1"].fix(x[2 * nc + 6])


def _copy_port(src, dst):
    import pyomo.environ as pyo
    for c in COMPS:
        dst.flow_mol_comp[0, c].value = max(pyo.value(src.flow_mol_comp[0, c]), 1e-8)
    dst.temperature[0].value = pyo.value(src.temperature[0])
    dst.pressure[0].value = pyo.value(src.pressure[0])


def initialize(m):
    """Sequential unit initialisation at the current inputs."""
    import idaes.logger as idaeslog
    m.fs.mix.initialize(outlvl=idaeslog.CRITICAL)
    _copy_port(m.fs.mix.outlet, m.fs.flash.inlet)
    m.fs.flash.initialize(outlvl=idaeslog.CRITICAL)
    _copy_port(m.fs.flash.liq_outlet, m.fs.split.inlet)
    m.fs.split.initialize(outlvl=idaeslog.CRITICAL)


def outputs(m):
    import pyomo.environ as pyo
    return np.array([pyo.value(m.ports[s].flow_mol_comp[0, c]) for s in STREAMS for c in COMPS])


# ----------------------------------------------------------------------------- sampling

def _psat_params():
    """Saturation-pressure parameters read from the property package (same correlation as IDAES)."""
    import pyomo.environ as pyo
    m = pyo.ConcreteModel()
    m.p = _btxo_block()(valid_phase=("Liq", "Vap"), activity_coeff_model="Ideal", state_vars="FcTP")
    coef = np.array([[pyo.value(m.p.pressure_sat_coeff[c, k]) for k in "ABCD"] for c in COMPS])
    Tc = np.array([pyo.value(m.p.temperature_critical[c]) for c in COMPS])
    Pc = np.array([pyo.value(m.p.pressure_critical[c]) for c in COMPS])
    return coef, Tc, Pc


def _psat(T, coef, Tc, Pc):
    tau = 1.0 - T / Tc
    return Pc * np.exp((coef[:, 0] * tau + coef[:, 1] * tau ** 1.5 + coef[:, 2] * tau ** 3
                        + coef[:, 3] * tau ** 6) / (1.0 - tau))


def _bubble_dew(z, P, prm):
    from scipy.optimize import brentq
    tb = brentq(lambda T: (z * _psat(T, *prm)).sum() - P, 250.0, 540.0, xtol=1e-10)
    td = brentq(lambda T: (z / _psat(T, *prm)).sum() * P - 1.0, 250.0, 540.0, xtol=1e-10)
    return tb, td


def sample_inputs(n, seed=SEED, pool=N_POOL):
    """First n rows of a fixed pool of `pool` input draws (so n can change without reshuffling)."""
    rng = np.random.default_rng(seed)
    nc = len(COMPS)
    n, n_req = pool, n
    F = rng.uniform(*FLOW_RANGE, (n, 2 * nc))
    T1, T2 = rng.uniform(*FEED_T, n), rng.uniform(*FEED_T, n)
    P1, P2 = rng.uniform(*FEED_P, n), rng.uniform(*FEED_P, n)
    Pf = rng.uniform(*FLASH_P, n)
    theta = rng.uniform(*THETA, n)
    sf = rng.uniform(*SPLIT, n)
    prm = _psat_params()
    Tf = np.full(n, np.nan)
    for k in range(n_req):
        tot = F[k, :nc] + F[k, nc:]
        tb, td = _bubble_dew(tot / tot.sum(), Pf[k], prm)
        Tf[k] = tb + theta[k] * (td - tb)
    return np.column_stack([F, T1, P1, T2, P2, Tf, Pf, sf])[:n_req]


def _flash_guess(m, x, prm):
    """Overwrite the flash/splitter iterates with an ideal Rachford-Rice estimate at the specified flash
    T and P (used when a warm start lands on the trivial zero-vapour root)."""
    from scipy.optimize import brentq
    nc = len(COMPS)
    F = x[:nc] + x[nc:2 * nc]
    Tf, Pf, sf = x[2 * nc + 4], x[2 * nc + 5], x[2 * nc + 6]
    psat = _psat(Tf, *prm)
    K = psat / Pf
    z = F / F.sum()
    beta = brentq(lambda b: (z * (K - 1) / (1 + b * (K - 1))).sum(), 1e-9, 1 - 1e-9, xtol=1e-12)
    liq = F * (1 - beta) / (1 + beta * (K - 1))
    vap = F - liq
    tb, td = _bubble_dew(z, Pf, prm)
    fo = m.fs.flash.control_volume.properties_out[0]
    fo.temperature_bubble.value, fo.temperature_dew.value = tb, td
    fo._t1.value = fo._temperature_equilibrium.value = Tf
    for i, c in enumerate(COMPS):
        fo.flow_mol_comp[c].value = F[i]
        fo.flow_mol_phase_comp["Liq", c].value = liq[i]
        fo.flow_mol_phase_comp["Vap", c].value = vap[i]
        fo.mole_frac_phase_comp["Liq", c].value = liq[i] / liq.sum()
        fo.mole_frac_phase_comp["Vap", c].value = vap[i] / vap.sum()
        fo.pressure_sat_comp[c].value = psat[i]
        for blk, frac in ((m.fs.split.mixed_state[0], 1.0), (m.fs.split.out1_state[0], sf),
                          (m.fs.split.out2_state[0], 1.0 - sf)):
            blk.flow_mol_comp[c].value = frac * liq[i]
            blk.flow_mol_phase_comp["Liq", c].value = frac * liq[i]
            blk.mole_frac_phase_comp["Liq", c].value = liq[i] / liq.sum()
            blk.temperature.value, blk.pressure.value = Tf, Pf


# ----------------------------------------------------------------------------- generation

_M = None


def _solve_chunk(Xc):
    """Worker: solve each row of Xc with a warm start from the previous solution; on failure retry from
    a Rachford-Rice estimate of the flash, then the nominal solution, then a fresh sequential
    initialisation.  Returns (Y, converged, solve counts)."""
    global _M
    import logging
    import pyomo.environ as pyo
    logging.getLogger("pyomo").setLevel(logging.ERROR)
    logging.getLogger("idaes").setLevel(logging.ERROR)
    logging.getLogger("idaes.core.util.scaling").setLevel(logging.ERROR)
    if _M is None:
        m = build_model()
        solver = pyo.SolverFactory("ipopt", executable=IPOPT_EXE)
        solver.options.update(IPOPT_OPTIONS)
        x0 = np.array([0.4] * 6 + [320.0, 2e5, 320.0, 2e5, 375.0, 1e5, 0.5])
        set_inputs(m, x0)
        initialize(m)
        res = solver.solve(m, load_solutions=False)
        assert pyo.check_optimal_termination(res), "nominal flowsheet did not converge"
        m.solutions.load_from(res)
        vars_ = list(m.component_data_objects(pyo.Var, descend_into=True))
        nominal = [(v.value, v.fixed) for v in vars_]
        _M = (m, solver, vars_, nominal, _psat_params())
    m, solver, vars_, nominal, prm = _M

    flash_out = m.fs.flash.control_volume.properties_out[0]
    stats = {"warm": 0, "guess": 0, "nominal": 0, "reinit": 0, "failed": 0, "trivial_vapour": 0}

    def attempt():
        res = solver.solve(m, load_solutions=False)
        if not pyo.check_optimal_termination(res):
            return False
        m.solutions.load_from(res)
        # The FcTP formulation also admits a trivial root with zero vapour flow (vapour mole fractions
        # then do not sum to one).  The sampled flash T lies strictly between bubble and dew point, so
        # such a root is not the physical solution: treat it as not converged and retry.
        sy = sum(pyo.value(flash_out.mole_frac_phase_comp["Vap", c]) for c in COMPS)
        sx = sum(pyo.value(flash_out.mole_frac_phase_comp["Liq", c]) for c in COMPS)
        if abs(sy - 1.0) > 1e-6 or abs(sx - 1.0) > 1e-6:
            stats["trivial_vapour"] += 1
            return False
        return True

    def reset():
        """Nominal values and fixed flags (a failed unit initialisation can leave variables fixed)."""
        for v, (val, fixed) in zip(vars_, nominal):
            if fixed:
                v.fix()          # inputs: overwritten by set_inputs
            else:
                v.unfix()
                v.value = val

    reset()   # every chunk starts from the nominal point, so results do not depend on worker scheduling

    Y = np.full((len(Xc), len(STREAMS) * len(COMPS)), np.nan)
    conv = np.zeros(len(Xc), bool)
    for k, x in enumerate(Xc):
        set_inputs(m, x)
        how = "warm" if attempt() else None
        if how is None:
            reset()
            set_inputs(m, x)
            try:
                _flash_guess(m, x, prm)
                how = "guess" if attempt() else None
            except Exception:
                how = None
        if how is None:
            reset()
            set_inputs(m, x)
            how = "nominal" if attempt() else None
        if how is None:
            try:
                reset()
                set_inputs(m, x)
                initialize(m)
                how = "reinit" if attempt() else None
            except Exception:
                how = None
        if how is None:
            stats["failed"] += 1
            reset()                      # restart the next sample from the nominal point
        else:
            stats[how] += 1
            Y[k], conv[k] = outputs(m), True
    return Y, conv, stats


def _run_part(args):
    """Worker task: solve one chunk and save it, so an interrupted generation can resume."""
    i, Xc, part_dir = args
    t0 = time.perf_counter()
    Y, conv, stats = _solve_chunk(Xc)
    np.savez(os.path.join(part_dir, f"part_{i:04d}.npz"), Y=Y, conv=conv,
             stats_keys=list(stats), stats_vals=list(stats.values()), seconds=time.perf_counter() - t0)
    return i


def generate(n=N_SAMPLES, workers=N_WORKERS):
    """Solve all samples on `workers` spawn processes, chunk by chunk (finished chunks are kept in
    CACHE/process_flowsheet_parts until all are present).  Returns X, Y, conv, stats, wall seconds."""
    import multiprocessing as mp
    part_dir = CACHE / "process_flowsheet_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    clock = part_dir / "elapsed.json"
    elapsed = json.loads(clock.read_text())["seconds"] if clock.exists() else 0.0
    t0 = time.perf_counter()
    X = sample_inputs(n)
    todo = [(i, X[s:s + CHUNK], str(part_dir)) for i, s in enumerate(range(0, n, CHUNK))
            if not (part_dir / f"part_{i:04d}.npz").exists()]
    n_chunks = -(-n // CHUNK)
    if todo:
        with mp.get_context("spawn").Pool(workers) as pool:
            for done, i in enumerate(pool.imap_unordered(_run_part, todo, chunksize=1), 1):
                clock.write_text(json.dumps({"seconds": elapsed + time.perf_counter() - t0}))
                print(f"[process_flowsheet] chunk {i} done ({n_chunks - len(todo) + done}/{n_chunks}, "
                      f"{(elapsed + time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    parts = [np.load(part_dir / f"part_{i:04d}.npz") for i in range(n_chunks)]
    Y = np.vstack([z["Y"] for z in parts])
    conv = np.concatenate([z["conv"] for z in parts])
    stats = {str(k): int(sum(z["stats_vals"][j] for z in parts)) for j, k in enumerate(parts[0]["stats_keys"])}
    secs = elapsed + time.perf_counter() - t0
    for z in parts:
        z.close()
    return X, Y, conv, stats, secs


# ----------------------------------------------------------------------------- domain

def constraints():
    nc, ns = len(COMPS), len(STREAMS)
    p = 2 * nc + 7
    col = lambda s, i: STREAMS.index(s) * nc + i
    A, B, rows = [], [], []
    for i, c in enumerate(COMPS):          # mixer: mix - feed1 - feed2 = 0
        a, bb = np.zeros(ns * nc), np.zeros(p)
        a[col("mix", i)] = 1.0
        bb[i] = bb[nc + i] = -1.0
        A.append(a), B.append(bb), rows.append(f"mixer_{c}")
    for i, c in enumerate(COMPS):          # flash: mix - vap - liq = 0
        a = np.zeros(ns * nc)
        a[col("mix", i)], a[col("vap", i)], a[col("liq", i)] = 1.0, -1.0, -1.0
        A.append(a), B.append(np.zeros(p)), rows.append(f"flash_{c}")
    for i, c in enumerate(COMPS):          # splitter: liq - out1 - out2 = 0
        a = np.zeros(ns * nc)
        a[col("liq", i)], a[col("out1", i)], a[col("out2", i)] = 1.0, -1.0, -1.0
        A.append(a), B.append(np.zeros(p)), rows.append(f"splitter_{c}")
    return np.array(A), np.array(B), np.zeros(len(rows)), rows


def load():
    path = CACHE / "process_flowsheet.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        X, Y, conv, stats, secs = generate()
        np.savez(path, X=X, Y=Y, conv=conv, seconds=secs, stats_keys=list(stats), stats_vals=list(stats.values()))
        shutil.rmtree(CACHE / "process_flowsheet_parts", ignore_errors=True)
    z = np.load(path)
    conv = z["conv"]
    X, Y = z["X"][conv], z["Y"][conv]
    n_fail, secs = int((~conv).sum()), float(z["seconds"])
    stats = {str(k): int(v) for k, v in zip(z["stats_keys"], z["stats_vals"])}
    A, B, b, rows = constraints()
    x_names = ([f"{f}_F_{c}" for f in FEEDS for c in COMPS]
               + ["feed1_T", "feed1_P", "feed2_T", "feed2_P", "flash_T", "flash_P", "split_frac_out1"])
    y_names = [f"{s}_F_{c}" for s in STREAMS for c in COMPS]
    return Domain(
        "process_flowsheet", X, Y, A, B, b, y_lo=np.zeros(Y.shape[1]),
        x_names=x_names, y_names=y_names, row_names=rows,
        meta={"source": "synthetic: IDAES-PSE 2.13 flowsheet (Mixer -> Flash -> Splitter), Pyomo 6.10, IPOPT 3.13 "
                        "(IDAES binaries); property package idaes.models.properties.activity_coeff_models."
                        "BTX_activity_coeff_VLE (ideal) with o-xylene enabled",
              "license": ("generated here (MIT, as the repo); IDAES-PSE is distributed under a BSD-style license "
                          "(3-clause BSD terms plus a US DOE / Lawrence Berkeley National Laboratory copyright "
                          "notice, see the IDAES COPYRIGHT.md and LICENSE.md); property data in the package from "
                          "Reid et al. (1987), Perry's handbook and the NIST WebBook as cited in its source"),
              "tier": 1,
              "units": "flows mol/s; temperatures K; pressures Pa; split fraction dimensionless",
              "generation": (f"first {N_SAMPLES} rows of a pool of {N_POOL} input draws, seed {SEED}. Feed "
                             f"component flows U{FLOW_RANGE} mol/s each (sampled directly so the balances are "
                             f"linear in X); feed T U{FEED_T} K and P U{FEED_P} Pa (subcooled liquid feeds; "
                             f"liquid-only package for mixer and splitter); flash P U{FLASH_P} Pa; flash T = "
                             f"T_bubble + theta (T_dew - T_bubble) of the mixed feed at flash P with theta "
                             f"U{THETA} (Raoult's law with the package's P_sat correlation), so the flash is "
                             f"two-phase; flash heat duty and pressure drop free; split fraction U{SPLIT}. Mixer "
                             f"momentum: smooth minimum of the feed pressures. IPOPT 3.13.2 (ma27) options "
                             f"{IPOPT_OPTIONS} with IDAES default scaling factors. Each worker builds and "
                             f"initialises the flowsheet once; every chunk of {CHUNK} samples starts from the "
                             f"nominal solution and each sample is warm-started from the previous one, with "
                             f"retries from (1) an ideal Rachford-Rice estimate of the flash, (2) the nominal "
                             f"solution, (3) a fresh sequential unit initialisation. A converged point on the "
                             f"trivial zero-vapour root (vapour mole fractions not summing to one) counts as not "
                             f"converged and is retried. Samples failing all attempts are dropped: {n_fail} of "
                             f"{len(conv)} (attempt counts {stats}). Generation wall time {secs:.0f} s on "
                             f"{N_WORKERS} worker processes on a shared machine. No sample is dropped on "
                             f"residual."),
              "n_nonconverged": n_fail, "generation_seconds": secs, "solve_stats": stats,
              "x_desc": "component molar flows of feed1 and feed2 (6), feed T and P (4), flash T and P (2), "
                        "split fraction (1)",
              "y_desc": "component molar flows (B, T, X) of mixer outlet, flash vapour, flash liquid, splitter "
                        "out1 and out2 (15)",
              "note": ("closest to the original paper (its case studies were Aspen process surrogates): an "
                       "equation-oriented steady-state separation flowsheet without reactions; the component "
                       "mole balances of mixer, flash and splitter are the hard linear equalities, while VLE, "
                       "energy balances and the split are nonlinear and learned")})
