"""Heat exchanger network with fixed stream flows (tier 3; counter-current effectiveness-NTU, numpy).

Two hot streams (H1, H2) and two cold streams (C1, C2) with FIXED heat-capacity flow rates exchange heat in
four counter-current exchangers arranged as a grid:
    E1: H1 (inlet)      vs C1 (inlet)
    E2: H1 (after E1)   vs C2 (inlet)
    E3: H2 (inlet)      vs C1 (after E1)
    E4: H2 (after E3)   vs C2 (after E2)
Inputs : the 4 stream inlet temperatures (degC) and the 4 exchanger UA values (kW/K).
Outputs: duty Q (kW), hot outlet and cold outlet temperature (degC) of every exchanger (12).
Equality: energy balance on each side of each exchanger,
    Q - CP_hot (T_hot,in - T_hot,out) = 0  and  Q - CP_cold (T_cold,out - T_cold,in) = 0,
linear with a fixed A because the CP values are fixed (inlet temperatures enter via B).
"""
import numpy as np

from bench.domain import CACHE, Domain

CP = {"H1": 30.0, "H2": 15.0, "C1": 20.0, "C2": 25.0}      # kW/K, fixed
# exchanger -> (hot stream, hot source, cold stream, cold source); source = "in" or an exchanger name
NET = {"E1": ("H1", "in", "C1", "in"), "E2": ("H1", "E1", "C2", "in"),
       "E3": ("H2", "in", "C1", "E1"), "E4": ("H2", "E3", "C2", "E2")}
N_SAMPLES, SEED = 20_000, 505
INLETS = ["H1", "H2", "C1", "C2"]


def exchanger(th, tc, cph, cpc, ua):
    """Counter-current effectiveness-NTU: returns Q, hot outlet, cold outlet."""
    cmin, cmax = min(cph, cpc), max(cph, cpc)
    cr, ntu = cmin / cmax, ua / cmin
    eps = (1 - np.exp(-ntu * (1 - cr))) / (1 - cr * np.exp(-ntu * (1 - cr)))
    q = eps * cmin * (th - tc)
    return q, th - q / cph, tc + q / cpc


def simulate(t_in, ua):
    t = dict(zip(INLETS, t_in))
    out = {}
    for k, (h, hs, c, cs) in NET.items():               # NET is in a valid evaluation order
        th = t[h] if hs == "in" else out[hs][1]
        tc = t[c] if cs == "in" else out[cs][2]
        out[k] = exchanger(th, tc, CP[h], CP[c], ua[int(k[1]) - 1])
    return np.concatenate([out[k] for k in NET])


def generate():
    rng = np.random.default_rng(SEED)
    T = np.column_stack([rng.uniform(150, 250, N_SAMPLES), rng.uniform(120, 220, N_SAMPLES),
                         rng.uniform(20, 80, N_SAMPLES), rng.uniform(30, 100, N_SAMPLES)])
    UA = np.exp(rng.uniform(np.log(2.0), np.log(40.0), (N_SAMPLES, 4)))
    Y = np.stack([simulate(T[n], UA[n]) for n in range(N_SAMPLES)])
    return T, UA, Y


def constraints():
    names = list(NET)
    col = {(k, v): 3 * i + j for i, k in enumerate(names) for j, v in enumerate(["Q", "Th", "Tc"])}
    A = np.zeros((8, 12))
    B = np.zeros((8, 8))
    rows = []
    for i, (k, (h, hs, c, cs)) in enumerate(NET.items()):
        r = 2 * i                                          # hot side: Q - CPh*Th_in + CPh*Th_out = 0
        A[r, col[(k, "Q")]] = 1.0
        A[r, col[(k, "Th")]] = CP[h]
        if hs == "in":
            B[r, INLETS.index(h)] = -CP[h]
        else:
            A[r, col[(hs, "Th")]] = -CP[h]
        r += 1                                             # cold side: Q - CPc*Tc_out + CPc*Tc_in = 0
        A[r, col[(k, "Q")]] = 1.0
        A[r, col[(k, "Tc")]] = -CP[c]
        if cs == "in":
            B[r, INLETS.index(c)] = CP[c]
        else:
            A[r, col[(cs, "Tc")]] = CP[c]
        rows += [f"{k}_hot_balance", f"{k}_cold_balance"]
    return A, B, rows


def load():
    path = CACHE / "heat_exchanger_network.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        T, UA, Y = generate()
        np.savez(path, T=T, UA=UA, Y=Y)
    z = np.load(path)
    A, B, rows = constraints()
    # only E1 sees both fresh inlets (H1 >= 150 > C1 <= 80 degC), so only its duty is nonnegative by construction;
    # downstream exchangers can run in reverse when the preheated cold stream is hotter than the hot stream
    lo = np.full(12, -np.inf)
    lo[0] = 0.0
    return Domain(
        "heat_exchanger_network", np.hstack([z["T"], z["UA"]]), z["Y"], A, B, np.zeros(8), y_lo=lo,
        x_names=[f"T_in_{s}" for s in INLETS] + [f"UA_{k}" for k in NET],
        y_names=[f"{k}_{v}" for k in NET for v in ["Q", "Th_out", "Tc_out"]], row_names=rows,
        meta={"source": "synthetic: counter-current effectiveness-NTU exchangers (numpy)",
              "license": "generated here (MIT, as the repo)", "tier": 3, "units": "duty kW; temperatures degC; UA kW/K",
              "generation": (f"fixed CP {CP} kW/K; grid network {NET}; inlet T: H1 U(150,250), H2 U(120,220), "
                             f"C1 U(20,80), C2 U(30,100) degC; UA logU(2,40) kW/K per exchanger. {N_SAMPLES} "
                             f"samples, seed {SEED}."),
              "x_desc": "4 inlet temperatures and 4 exchanger UA values",
              "y_desc": "duty, hot outlet T and cold outlet T of each of the 4 exchangers (12)"})
