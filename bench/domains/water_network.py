"""Water distribution network: EPANET Net3 steady-state link flows under random junction demands.

Inputs X  : actual demand at every junction with non-zero base demand (L/s).
Outputs Y : flow in every link (117 pipes + 2 pumps, L/s, positive from start node to end node).
Equality  : mass balance at every junction, inflow - outflow - demand = 0, i.e.
            A y + B x = 0 with A the signed node-link incidence restricted to junctions
            (+1 if the link ends at the junction, -1 if it starts there) and B = -E, where E
            places each demand on its junction row.  Tanks and reservoirs are not constrained.

Each sample is one demand-driven steady-state snapshot (duration 0) solved with WNTR's own
Newton solver (WNTRSimulator), which returns double-precision flows directly (the EPANET
toolkit simulator writes single-precision binary output, which would limit the residual to ~1e-7).
Net3.inp ships with the wntr package (wntr/library/networks), so nothing is downloaded.
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np

from bench.domain import CACHE, Domain

NAME = "water_network"
N = 20_000
SEED = 20240611
WORKERS = 6
CHUNK = 250
G_RANGE = (0.6, 1.4)      # system-wide demand factor
U_RANGE = (0.5, 1.5)      # independent per-junction factor
SOLVER_TOL = 1e-10        # Newton tolerance (inf-norm of the equation residual, SI units)
TO_LPS = 1000.0           # m3/s -> L/s

_WN = None


def _inp_path():
    import wntr
    return os.path.join(os.path.dirname(wntr.__file__), "library", "networks", "Net3.inp")


def _network():
    """Net3 prepared for a single snapshot: controls removed, pump 10 and pump 335 running,
    pipe 330 closed (the state the original controls give while tank 1 is being filled and the
    lake pump is on, 01:00-15:00), demand patterns dropped so demand = base demand x multiplier."""
    import wntr
    logging.getLogger("wntr").setLevel(logging.ERROR)
    wn = wntr.network.WaterNetworkModel(_inp_path())
    wn.options.time.duration = 0
    wn.options.hydraulic.demand_model = "DD"
    for c in list(wn.control_name_list):
        wn.remove_control(c)
    wn.get_link("10").initial_status = wntr.network.LinkStatus.Open
    wn.get_link("335").initial_status = wntr.network.LinkStatus.Open
    wn.get_link("330").initial_status = wntr.network.LinkStatus.Closed
    for _, j in wn.junctions():
        assert len(j.demand_timeseries_list) == 1
        j.demand_timeseries_list[0].pattern_name = None
    return wn


def _structure(wn):
    junc = list(wn.junction_name_list)
    links = list(wn.link_name_list)
    jidx = {j: i for i, j in enumerate(junc)}
    A = np.zeros((len(junc), len(links)))
    for k, name in enumerate(links):
        lk = wn.get_link(name)
        if lk.start_node_name in jidx:
            A[jidx[lk.start_node_name], k] -= 1.0
        if lk.end_node_name in jidx:
            A[jidx[lk.end_node_name], k] += 1.0
    base = np.array([wn.get_node(j).demand_timeseries_list[0].base_value for j in junc])
    dem = np.where(base != 0)[0]                      # junctions that carry a demand
    kinds = [wn.get_link(l).link_type for l in links]
    return junc, links, A, base, dem, kinds


def _simulate(args):
    """Solve one chunk of snapshots.  Returns (row indices, flows m3/s, demands m3/s, ok mask)."""
    global _WN
    import wntr
    if _WN is None:
        _WN = _network()
    wn = _WN
    idx, demands = args                               # demands: (k, n_junctions) in m3/s
    junc, links = list(wn.junction_name_list), list(wn.link_name_list)
    ts = [wn.get_node(j).demand_timeseries_list[0] for j in junc]
    Q = np.full((len(idx), len(links)), np.nan)
    D = np.full((len(idx), len(junc)), np.nan)
    ok = np.zeros(len(idx), bool)
    for r in range(len(idx)):
        wn.reset_initial_values()
        for t, d in zip(ts, demands[r]):
            t.base_value = float(d)
        try:
            res = wntr.sim.WNTRSimulator(wn).run_sim(solver_options={"TOL": SOLVER_TOL},
                                                     convergence_error=True)
        except Exception:
            continue
        Q[r] = res.link["flowrate"].iloc[0][links].to_numpy(np.float64)
        D[r] = res.node["demand"].iloc[0][junc].to_numpy(np.float64)
        ok[r] = np.isfinite(Q[r]).all() and np.isfinite(D[r]).all()
    return idx, Q, D, ok


def _generate(path):
    wn = _network()
    junc, links, A, base, dem, kinds = _structure(wn)
    rng = np.random.default_rng(SEED)
    g = rng.uniform(*G_RANGE, size=(N, 1))
    u = rng.uniform(*U_RANGE, size=(N, len(junc)))
    demands = base * g * u                            # m3/s; zero-base junctions stay zero
    jobs = [(np.arange(s, min(s + CHUNK, N)), demands[s:s + CHUNK]) for s in range(0, N, CHUNK)]
    Q = np.full((N, len(links)), np.nan)
    D = np.full((N, len(junc)), np.nan)
    ok = np.zeros(N, bool)
    t0 = time.perf_counter()
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        for idx, q, d, o in pool.imap_unordered(_simulate, jobs):
            Q[idx], D[idx], ok[idx] = q, d, o
    secs = time.perf_counter() - t0
    # the simulator reports the demand it imposed; in demand-driven mode it must equal the input
    dev = np.abs(D[ok] - demands[ok]).max() if ok.any() else np.nan
    np.savez_compressed(path, X=demands[ok][:, dem] * TO_LPS, Y=Q[ok] * TO_LPS, g=g[ok, 0],
                        n_failed=int((~ok).sum()), gen_seconds=secs, demand_dev=dev)
    print(f"[{NAME}] generated {ok.sum()}/{N} snapshots in {secs:.0f}s "
          f"({(~ok).sum()} failed), max |reported - imposed demand| = {dev:.3e} m3/s", flush=True)


def load() -> Domain:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{NAME}.npz"
    if not path.exists():
        _generate(path)
    z = np.load(path)
    wn = _network()
    junc, links, A, base, dem, kinds = _structure(wn)
    E = np.zeros((len(junc), len(dem)))
    E[dem, np.arange(len(dem))] = 1.0
    B = -E
    q = len(links)
    y_lo = np.full(q, -np.inf)
    y_hi = np.full(q, np.inf)
    for k, (name, kind) in enumerate(zip(links, kinds)):
        if kind == "Pump":
            y_lo[k] = 0.0                              # pumps cannot reverse
    k330 = links.index("330")
    y_lo[k330] = y_hi[k330] = 0.0                      # closed by construction
    return Domain(
        NAME, z["X"], z["Y"], A, B, np.zeros(len(junc)),
        y_lo=y_lo, y_hi=y_hi,
        x_names=[f"demand_{junc[i]}" for i in dem],
        y_names=[f"{kind.lower()}_{name}" for name, kind in zip(links, kinds)],
        row_names=[f"mass_balance_{j}" for j in junc],
        meta={
            "source": "EPANET example network Net3 (bundled with WNTR 1.5.0, wntr/library/networks/Net3.inp)",
            "license": "EPANET example networks: US EPA, public domain; WNTR: Revised BSD",
            "generation": (
                f"{N} demand-driven steady-state snapshots of Net3 solved with wntr.sim.WNTRSimulator "
                f"(Newton, TOL={SOLVER_TOL:g}, double precision). Controls removed; pumps 10 and 335 open, "
                "pipe 330 closed; tank levels at their initial values (fixed heads); patterns dropped. "
                f"Junction demand = base demand x g x u_j, g ~ U{G_RANGE} per sample, "
                f"u_j ~ U{U_RANGE} per junction and sample, seed {SEED}. "
                f"{int(z['n_failed'])} non-converged snapshots dropped; generation took "
                f"{float(z['gen_seconds']):.0f}s with {WORKERS} processes."),
            "units": "L/s for demands and flows",
            "tier": 1,
            "x_desc": f"actual demand at the {len(dem)} junctions with non-zero base demand (L/s)",
            "y_desc": (f"flow in all {q} links ({kinds.count('Pipe')} pipes, {kinds.count('Pump')} pumps), "
                       "L/s, positive from start to end node"),
            "constraint": "junction mass balance: sum(inflow) - sum(outflow) - demand = 0 (92 rows)",
            "simulator": "WNTRSimulator (pure Python/NumPy Newton solver, float64 results)",
        },
    )
