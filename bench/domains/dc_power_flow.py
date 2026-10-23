"""DC power flow on the IEEE 118-bus system (topology and parameters from pandapower's case118).

Inputs X (MW): active load at each of the 99 load buses and active output of the 18 non-slack
generators with non-zero base dispatch.  Outputs Y (MW): active flow on all 186 branches
(173 lines + 13 transformers, positive in the from -> to / hv -> lv direction) and the slack
injection at bus 69 (pandapower bus index 68, the ext_grid bus).

Equality (one row per bus, 118 rows), nodal active-power balance:
    gen_i - load_i - sum(flows leaving i) + sum(flows entering i) = 0
    A = [-C | e_slack]  (C: bus-branch incidence, +1 at the from bus, -1 at the to bus)
    B = [-L | G]        (load / generator incidence),  b = 0.
A is fixed: the incidence of a connected graph has rank 117 and the slack column adds one,
so rank(A) = 118 against q = 187.

DC approximation (standard): flat voltage magnitudes, small angle differences, series
resistance, line charging, shunts and losses ignored.  All phase shifts in case118 are zero.
Branch susceptance b_k = 1 / (x_k * tap_k) in per unit on the 100 MVA system base, as in
MATPOWER's makeBdc.  The flows are computed in per unit and reported in MW.
Four case118 branches between voltage levels (pandapower trafos 7, 9, 11, 12) carry line
charging as a negative i0_percent; pandapower's T-to-pi conversion perturbs their series x by
up to 0.3 %.  Here x comes from vk/vkr alone, which reproduces the original IEEE values
(0.016, 0.0202, 0.2074, 0.00405 p.u.); base-case flows differ from pandapower.rundcpp by at most
0.03 MW for that reason only.
"""
from __future__ import annotations

import time

import numpy as np

from bench.domain import CACHE, Domain

NAME = "dc_power_flow"
N = 20_000
SEED = 20240118
LOAD_RANGE = (0.7, 1.3)   # per-bus multiplier on the base load
GEN_RANGE = (0.7, 1.3)    # per-unit multiplier on the base dispatch of each non-slack generator
CACHE_VERSION = 1


def _network():
    """Topology, branch susceptances (p.u.) and injection maps from pandapower case118."""
    import pandapower.networks as pn

    net = pn.case118()
    sn = float(net.sn_mva)                                   # 100 MVA system base
    bus_ids = net.bus.index.to_numpy()
    pos = {b: i for i, b in enumerate(bus_ids)}             # pandapower index -> row
    nb = len(bus_ids)
    vn = net.bus.vn_kv.to_numpy(float)

    ln = net.line[net.line.in_service]
    tr = net.trafo[net.trafo.in_service]
    assert np.all(ln.parallel == 1) and np.all(tr.parallel == 1)
    assert np.allclose(tr.shift_degree.fillna(0.0), 0.0), "phase shifters not handled"

    # lines: x in ohm -> p.u. on the base impedance of the from bus
    lf = ln.from_bus.map(pos).to_numpy()
    lt = ln.to_bus.map(pos).to_numpy()
    x_line = (ln.x_ohm_per_km * ln.length_km).to_numpy(float) / (vn[lf] ** 2 / sn)
    tap_line = np.ones(len(ln))

    # transformers (pandapower model): x = vk% / 100 * (sn / sn_trafo) * (vn_lv_rated / vn_lv_bus)^2,
    # off-nominal ratio on the hv side = rated ratio / bus ratio * (1 + (tap_pos - tap_neutral) * step%)
    tf = tr.hv_bus.map(pos).to_numpy()
    tt = tr.lv_bus.map(pos).to_numpy()
    vk = tr.vk_percent.to_numpy(float)
    vkr = tr.vkr_percent.to_numpy(float)
    xk = np.sqrt(vk ** 2 - vkr ** 2) / 100.0                 # series reactance only
    x_tr = xk * (sn / tr.sn_mva.to_numpy(float)) * (tr.vn_lv_kv.to_numpy(float) / vn[tt]) ** 2
    ratio = (tr.vn_hv_kv.to_numpy(float) / tr.vn_lv_kv.to_numpy(float)) / (vn[tf] / vn[tt])
    has_tap = tr.tap_pos.notna().to_numpy() & tr.tap_step_percent.notna().to_numpy()
    assert np.all(tr.tap_side[has_tap] == "hv"), "only hv-side ratio taps handled"
    steps = (tr.tap_pos - tr.tap_neutral).fillna(0.0).to_numpy(float)
    tap_tr = ratio * np.where(has_tap, 1.0 + steps * tr.tap_step_percent.fillna(0.0).to_numpy(float) / 100.0, 1.0)

    f = np.concatenate([lf, tf])
    t = np.concatenate([lt, tt])
    bsus = 1.0 / (np.concatenate([x_line, x_tr]) * np.concatenate([tap_line, tap_tr]))
    nbr = len(f)
    C = np.zeros((nb, nbr))
    C[f, np.arange(nbr)] = 1.0
    C[t, np.arange(nbr)] = -1.0

    assert len(net.ext_grid) == 1
    slack = pos[int(net.ext_grid.bus.iloc[0])]

    loads = net.load[net.load.in_service]
    assert not loads.bus.duplicated().any()
    load_bus = loads.bus.map(pos).to_numpy()
    load_p = loads.p_mw.to_numpy(float)

    gens = net.gen[net.gen.in_service]
    assert not gens.bus.duplicated().any() and slack not in set(gens.bus.map(pos))
    gens = gens[gens.p_mw > 0]                               # zero-dispatch units: no P injection
    gen_bus = gens.bus.map(pos).to_numpy()
    gen_p = gens.p_mw.to_numpy(float)

    ids = bus_ids + 1                                        # IEEE 1-based bus numbers
    br_names = [f"flow_line{k}_{ids[a]}-{ids[b]}" for k, (a, b) in enumerate(zip(lf, lt))]
    br_names += [f"flow_trafo{k}_{ids[a]}-{ids[b]}" for k, (a, b) in enumerate(zip(tf, tt))]
    return dict(nb=nb, sn=sn, C=C, bsus=bsus, slack=slack, load_bus=load_bus, load_p=load_p,
                gen_bus=gen_bus, gen_p=gen_p, ids=ids, br_names=br_names,
                gen_pmax=gens.max_p_mw.to_numpy(float), n_gen_total=len(net.gen))


def _matrices(nw):
    nb, nl, ng = nw["nb"], len(nw["load_bus"]), len(nw["gen_bus"])
    e = np.zeros((nb, 1))
    e[nw["slack"], 0] = 1.0
    A = np.hstack([-nw["C"], e])
    L = np.zeros((nb, nl))
    L[nw["load_bus"], np.arange(nl)] = 1.0
    G = np.zeros((nb, ng))
    G[nw["gen_bus"], np.arange(ng)] = 1.0
    B = np.hstack([-L, G])
    return A, B


def _simulate(nw, n, seed):
    """Sample inputs and solve the DC power flow exactly (float64)."""
    rng = np.random.default_rng(seed)
    load = nw["load_p"] * rng.uniform(*LOAD_RANGE, size=(n, len(nw["load_p"])))
    gen = nw["gen_p"] * rng.uniform(*GEN_RANGE, size=(n, len(nw["gen_p"])))
    X = np.hstack([load, gen])

    nb, sn, C, bsus, slack = nw["nb"], nw["sn"], nw["C"], nw["bsus"], nw["slack"]
    P = np.zeros((n, nb))
    np.add.at(P, (slice(None), nw["gen_bus"]), gen)
    np.add.at(P, (slice(None), nw["load_bus"]), -load)
    p_slack = -P.sum(1)                                      # lossless: slack closes the balance
    P[:, slack] += p_slack

    Bbus = C @ (bsus[:, None] * C.T)                         # nb x nb nodal susceptance matrix (p.u.)
    keep = np.delete(np.arange(nb), slack)                   # slack angle = 0
    theta = np.zeros((n, nb))
    theta[:, keep] = np.linalg.solve(Bbus[np.ix_(keep, keep)], (P[:, keep] / sn).T).T
    flows = sn * (theta @ C) * bsus                          # MW, from -> to
    Y = np.hstack([flows, p_slack[:, None]])
    return X, Y


def load(n=N, seed=SEED) -> Domain:
    nw = _network()
    A, B = _matrices(nw)
    path = CACHE / f"{NAME}.npz"
    X = Y = None
    gen_seconds = None
    if path.exists():
        z = np.load(path)
        if (int(z["version"]) == CACHE_VERSION and int(z["seed"]) == seed and z["X"].shape[0] == n
                and z["A"].shape == A.shape and np.array_equal(z["A"], A) and np.array_equal(z["B"], B)):
            X, Y, gen_seconds = z["X"], z["Y"], float(z["gen_seconds"])
    if X is None:
        t0 = time.perf_counter()
        X, Y = _simulate(nw, n, seed)
        gen_seconds = time.perf_counter() - t0
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(path, X=X, Y=Y, A=A, B=B, seed=seed, version=CACHE_VERSION, gen_seconds=gen_seconds)

    ids = nw["ids"]
    x_names = [f"load_bus{ids[i]}" for i in nw["load_bus"]] + [f"gen_bus{ids[i]}" for i in nw["gen_bus"]]
    y_names = nw["br_names"] + [f"p_slack_bus{ids[nw['slack']]}"]
    row_names = [f"balance_bus{ids[i]}" for i in range(nw["nb"])]
    meta = {
        "source": "IEEE 118-bus test case via pandapower.networks.case118() (pandapower 3.x)",
        "license": "pandapower: BSD 3-Clause. IEEE 118-bus case: public test data from the "
                   "UW Power Systems Test Case Archive (no explicit license; widely redistributed).",
        "generation": (f"n={n}, seed={seed}. Each of the {len(nw['load_bus'])} bus loads = base p_mw x "
                       f"U{LOAD_RANGE}; each of the {len(nw['gen_bus'])} non-slack generators with non-zero "
                       f"base dispatch = base p_mw x U{GEN_RANGE} (the other "
                       f"{nw['n_gen_total'] - len(nw['gen_bus'])} gens have zero base dispatch, mostly "
                       "synchronous condensers, and are kept at 0 MW). Slack (bus "
                       f"{ids[nw['slack']]}) injection = total load - total generation. Lossless DC power "
                       "flow: b = 1/(x*tap) p.u. on 100 MVA, theta = Bred^-1 P (float64, slack angle 0), "
                       "flow = b (theta_from - theta_to). R, line charging, shunts ignored; no phase shifters."),
        "generation_seconds": gen_seconds,
        "units": "MW for all inputs and outputs (100 MVA base used internally)",
        "tier": 1,
        "x_desc": "active load per load bus (MW) then active output per dispatched non-slack generator (MW)",
        "y_desc": "active flow per branch (MW, lines then transformers, from/hv -> to/lv) then slack injection (MW)",
    }
    return Domain(NAME, X, Y, A, B, np.zeros(A.shape[0]), x_names=x_names, y_names=y_names,
                  row_names=row_names, meta=meta)
