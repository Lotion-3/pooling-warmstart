"""Meshed resistor network (DC nodal analysis in numpy).

15 nodes (node 0 = ground) joined by a ring plus random chords: 32 branches, i.e. 18 independent loops,
so Kirchhoff's current law alone does not fix the branch currents (they also depend on the resistances).

Inputs : current injections at the 14 non-ground nodes (A) and the 32 branch resistances (ohm).
Outputs: the 32 branch currents (A, positive from the branch's first node to its second).
Equality: KCL at every non-ground node,  D_r i - J = 0  with the reduced incidence matrix D_r, which is
fixed by the topology even though the resistances vary.
"""
import numpy as np

from bench.domain import CACHE, Domain

N_NODES, N_BRANCH, N_SAMPLES, SEED = 15, 32, 20_000, 202


def topology(seed=SEED):
    rng = np.random.default_rng(seed)
    edges = [(k, (k + 1) % N_NODES) for k in range(N_NODES)]
    have = {frozenset(e) for e in edges}
    while len(edges) < N_BRANCH:
        a, b = rng.choice(N_NODES, 2, replace=False)
        if frozenset((a, b)) not in have:
            have.add(frozenset((a, b)))
            edges.append((int(a), int(b)))
    D = np.zeros((N_NODES, N_BRANCH))          # +1 where the branch leaves the node, -1 where it enters
    for k, (a, b) in enumerate(edges):
        D[a, k], D[b, k] = 1.0, -1.0
    return edges, D


def generate():
    edges, D = topology()
    Dr = D[1:]
    rng = np.random.default_rng(SEED + 1)
    J = rng.uniform(-1.0, 1.0, (N_SAMPLES, N_NODES - 1))                      # A
    R = np.exp(rng.uniform(np.log(1.0), np.log(100.0), (N_SAMPLES, N_BRANCH)))  # ohm
    I = np.empty((N_SAMPLES, N_BRANCH))
    for n in range(N_SAMPLES):
        g = 1.0 / R[n]
        v = np.linalg.solve((Dr * g) @ Dr.T, J[n])   # node voltages, ground = 0
        I[n] = g * (Dr.T @ v)
    return J, R, I


def load():
    path = CACHE / "circuit.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        J, R, I = generate()
        np.savez(path, J=J, R=R, I=I)
    z = np.load(path)
    J, R, I = z["J"], z["R"], z["I"]
    edges, D = topology()
    m = N_NODES - 1
    return Domain(
        "circuit", np.hstack([J, R]), I, D[1:], np.hstack([-np.eye(m), np.zeros((m, N_BRANCH))]), np.zeros(m),
        x_names=[f"J_n{k}" for k in range(1, N_NODES)] + [f"R_b{k}" for k in range(N_BRANCH)],
        y_names=[f"I_b{k}_{a}-{b}" for k, (a, b) in enumerate(edges)],
        row_names=[f"KCL_n{k}" for k in range(1, N_NODES)],
        meta={"source": "synthetic: DC nodal analysis of a resistor network (numpy)",
              "license": "generated here (MIT, as the repo)", "tier": 1, "units": "currents A; resistances ohm",
              "generation": (f"{N_NODES} nodes, ring + random chords = {N_BRANCH} branches (topology seed {SEED}); "
                             f"injections U(-1,1) A at the {m} non-ground nodes; resistances logU(1,100) ohm per "
                             f"branch per sample; (D_r G D_r^T) v = J solved in float64, i = G D_r^T v. "
                             f"{N_SAMPLES} samples."),
              "x_desc": "node current injections (14) and branch resistances (32)",
              "y_desc": "branch currents (32)"})
