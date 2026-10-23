"""Statically indeterminate 2D truss (linear-elastic bar FEM in numpy).

Geometry: a 5 x 3 grid of joints (1 m spacing) with horizontal, vertical and both diagonal members in
every panel: 15 joints, 38 members.  Pin supports at the two bottom corners leave 26 free DOFs, so the
truss is 12 times statically indeterminate: equilibrium alone does not fix the member forces, which
depend on the member stiffnesses.

Inputs : external loads at the 26 free DOFs (kN) and the 38 member axial stiffnesses EA (MN).
Outputs: the 38 member axial forces N (kN, tension positive).
Equality: joint equilibrium at every free DOF,  sum_e g_e N_e - F = 0  (A = G_free, fixed by geometry).
"""
import numpy as np

from bench.domain import CACHE, Domain

NX, NY, N_SAMPLES, SEED = 5, 3, 20_000, 101
EA_BASE = 200.0   # MN, about a 1000 mm^2 steel bar (E = 200 GPa)


def geometry():
    nodes = np.array([(i, j) for j in range(NY) for i in range(NX)], float)
    idx = lambda i, j: j * NX + i
    members = []
    for j in range(NY):
        for i in range(NX - 1):
            members.append((idx(i, j), idx(i + 1, j)))
    for j in range(NY - 1):
        for i in range(NX):
            members.append((idx(i, j), idx(i, j + 1)))
    for j in range(NY - 1):
        for i in range(NX - 1):
            members.append((idx(i, j), idx(i + 1, j + 1)))
            members.append((idx(i + 1, j), idx(i, j + 1)))
    members = np.array(members)
    fixed = {2 * idx(0, 0), 2 * idx(0, 0) + 1, 2 * idx(NX - 1, 0), 2 * idx(NX - 1, 0) + 1}
    free = np.array([d for d in range(2 * len(nodes)) if d not in fixed])
    # G[:, e]: nodal force pattern of a unit tension in member e (force on joint i points to j)
    G = np.zeros((2 * len(nodes), len(members)))
    L = np.zeros(len(members))
    for e, (i, j) in enumerate(members):
        d = nodes[j] - nodes[i]
        L[e] = np.linalg.norm(d)
        c, s = d / L[e]
        G[[2 * i, 2 * i + 1, 2 * j, 2 * j + 1], e] = [-c, -s, c, s]
    return nodes, members, free, G, L


def solve(G, L, free, F_free, EA):
    k = EA * 1e3 / L                                   # kN/m
    Gf = G[free]
    K = (Gf * k) @ Gf.T                                # free-DOF stiffness
    u = np.linalg.solve(K, F_free)                     # m
    return k * (Gf.T @ u)                              # member forces, kN


def generate():
    nodes, members, free, G, L = geometry()
    rng = np.random.default_rng(SEED)
    F = rng.uniform(-50, 50, (N_SAMPLES, len(free)))   # kN
    EA = EA_BASE * np.exp(rng.uniform(np.log(0.25), np.log(4.0), (N_SAMPLES, len(members))))
    Y = np.stack([solve(G, L, free, F[n], EA[n]) for n in range(N_SAMPLES)])
    return F, EA, Y


def load():
    path = CACHE / "truss.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        F, EA, Y = generate()
        np.savez(path, F=F, EA=EA, Y=Y)
    z = np.load(path)
    F, EA, Y = z["F"], z["EA"], z["Y"]
    nodes, members, free, G, L = geometry()
    X = np.hstack([F, EA])
    A = G[free]
    B = np.hstack([-np.eye(len(free)), np.zeros((len(free), len(members)))])
    dof = lambda d: f"j{d // 2}{'xy'[d % 2]}"
    return Domain(
        "truss", X, Y, A, B, np.zeros(len(free)),
        x_names=[f"F_{dof(d)}" for d in free] + [f"EA_m{e}" for e in range(len(members))],
        y_names=[f"N_m{e}_{i}-{j}" for e, (i, j) in enumerate(members)],
        row_names=[f"equilibrium_{dof(d)}" for d in free],
        meta={"source": "synthetic: linear-elastic bar FEM (numpy)", "license": "generated here (MIT, as the repo)",
              "tier": 1, "units": "loads and member forces kN; EA in MN",
              "generation": (f"{NX}x{NY} joint grid, 1 m spacing, horizontal/vertical/both-diagonal members "
                             f"({len(members)} members), pins at both bottom corners ({len(free)} free DOFs, "
                             f"{len(members) - len(free)}x indeterminate). Loads U(-50,50) kN per free DOF; "
                             f"EA = {EA_BASE} MN x logU(0.25,4) per member; K u = F solved in float64; "
                             f"N = EA/L * elongation. {N_SAMPLES} samples, seed {SEED}."),
              "x_desc": "free-DOF joint loads (26) and member stiffnesses EA (38)",
              "y_desc": "member axial forces (38), tension positive"})
