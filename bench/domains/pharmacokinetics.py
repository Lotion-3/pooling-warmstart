"""Three-compartment pharmacokinetic model with elimination (linear ODE, scipy matrix exponential).

Central (1), shallow peripheral (2) and deep peripheral (3) compartments; elimination from the central
compartment into a cumulative "eliminated" state.  Four IV bolus doses at fixed times 0, 6, 12, 18 h.

Inputs : the 4 dose amounts (mg) and the 5 rate constants k12, k21, k13, k31, kel (1/h).
Outputs: amount in each compartment and cumulative eliminated (mg) at 20 fixed times (none on a dose time).
Equality: at each output time, A1 + A2 + A3 + E - (sum of doses given before that time) = 0.
"""
import numpy as np
from scipy.linalg import expm

from bench.domain import CACHE, Domain

DOSE_T = np.array([0.0, 6.0, 12.0, 18.0])          # h
OUT_T = np.linspace(0.5, 23.5, 20)                 # h, never equal to a dose time
N_SAMPLES, SEED = 20_000, 303
K_RANGES = {"k12": (0.05, 2.0), "k21": (0.05, 2.0), "k13": (0.01, 0.5), "k31": (0.005, 0.2), "kel": (0.05, 1.0)}


def rate_matrix(k12, k21, k13, k31, kel):
    """ds/dt = M s for s = (A1, A2, A3, E); every column of M sums to zero (mass is conserved)."""
    return np.array([[-(k12 + k13 + kel), k21, k31, 0.0],
                     [k12, -k21, 0.0, 0.0],
                     [k13, 0.0, -k31, 0.0],
                     [kel, 0.0, 0.0, 0.0]])


def simulate(doses, k):
    M = rate_matrix(*k)
    events = sorted([(t, "dose", i) for i, t in enumerate(DOSE_T)] + [(t, "out", i) for i, t in enumerate(OUT_T)])
    s, t_now, out = np.zeros(4), 0.0, np.zeros((len(OUT_T), 4))
    for t, kind, i in events:
        if t > t_now:
            s = expm(M * (t - t_now)) @ s
            t_now = t
        if kind == "dose":
            s[0] += doses[i]
        else:
            out[i] = s
    return out.ravel()                              # time-major: (t0: A1 A2 A3 E), (t1: ...), ...


def generate():
    rng = np.random.default_rng(SEED)
    doses = rng.uniform(10.0, 100.0, (N_SAMPLES, len(DOSE_T)))
    lo = np.log([r[0] for r in K_RANGES.values()])
    hi = np.log([r[1] for r in K_RANGES.values()])
    K = np.exp(rng.uniform(lo, hi, (N_SAMPLES, len(K_RANGES))))
    Y = np.stack([simulate(doses[n], K[n]) for n in range(N_SAMPLES)])
    return doses, K, Y


def load():
    path = CACHE / "pharmacokinetics.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        doses, K, Y = generate()
        np.savez(path, doses=doses, K=K, Y=Y)
    z = np.load(path)
    doses, K, Y = z["doses"], z["K"], z["Y"]
    nt = len(OUT_T)
    A = np.kron(np.eye(nt), np.ones((1, 4)))                     # sum of the 4 states at each time
    given = (DOSE_T[None, :] < OUT_T[:, None]).astype(float)     # doses given before each output time
    B = np.hstack([-given, np.zeros((nt, len(K_RANGES)))])
    states = ["A1", "A2", "A3", "E"]
    return Domain(
        "pharmacokinetics", np.hstack([doses, K]), Y, A, B, np.zeros(nt),
        y_lo=np.zeros(4 * nt),
        x_names=[f"dose_{t:g}h" for t in DOSE_T] + list(K_RANGES),
        y_names=[f"{s}_t{t:.2f}" for t in OUT_T for s in states],
        row_names=[f"mass_t{t:.2f}" for t in OUT_T],
        meta={"source": "synthetic: linear 3-compartment PK model (scipy.linalg.expm)",
              "license": "generated here (MIT, as the repo)", "tier": 1, "units": "amounts mg; rates 1/h",
              "generation": (f"IV boluses at {DOSE_T.tolist()} h with amounts U(10,100) mg; rate constants "
                             f"log-uniform in {K_RANGES}; exact propagation with the matrix exponential between "
                             f"events; outputs at {nt} times linspace(0.5, 23.5). {N_SAMPLES} samples, seed {SEED}."),
              "x_desc": "4 dose amounts and 5 rate constants",
              "y_desc": "A1, A2, A3 and cumulative eliminated E at 20 times (80)"})
