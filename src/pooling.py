"""Standard pooling problem: sources -> pools -> products, plus direct source -> product arcs.

One instance representation serves both the literature benchmarks and the real-data
crude-blending instances. Quality specs are written in a general linear-in-composition form

    sum_i (alpha_c . lam_i + beta_c) * F_ij <= 0        for spec c on product j = spec_prod[c]

where lam_i are the tracked (volume-blending) qualities of source i and F_ij is the total
flow of source i that reaches product j. The classic bound  lo <= quality_k <= up  is
alpha = -e_k, beta = lo  and  alpha = e_k, beta = -up. Mass-based specs (e.g. sulfur wt%)
use tracked qualities S*SG and SG:  alpha = (1, -Smax), beta = 0.

Formulations
  pq : Tawarmalani & Sahinidis (2002), as in GAMS model library `poolmod.inc`
       (variables q_il fractions, y_lj, z_ij; RLT "pszrlt" and "extensions" constraints).
  p  : Haverly's p-formulation (x_il, y_lj, z_ij, pool qualities p_lk).

Objective is MINIMIZED:  sum_i c_i * outflow_i - sum_j d_j * inflow_j + sum_l h_l * throughput_l.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pyomo.environ as pyo

# IDAES binaries copied out of the Store-Python AppData sandbox (see DECISIONS.md)
IDAES_BIN = os.environ.get("IDAES_BIN", os.path.join(os.path.expanduser("~"), "solvers", "idaes_bin"))
IPOPT_EXE = os.path.join(IDAES_BIN, "ipopt.exe")
if IDAES_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = IDAES_BIN + os.pathsep + os.environ.get("PATH", "")  # ipopt.exe needs its DLLs


@dataclass
class Instance:
    name: str
    c: np.ndarray          # (I,) unit cost of source
    d: np.ndarray          # (J,) unit price of product
    src_lo: np.ndarray     # (I,)
    src_up: np.ndarray     # (I,)
    pool_cap: np.ndarray   # (L,)
    prod_lo: np.ndarray    # (J,)
    prod_up: np.ndarray    # (J,)
    lam: np.ndarray        # (I, K) tracked source qualities
    q_up: np.ndarray       # (I, L) upper bound on q_il; 0 = no arc
    y_up: np.ndarray       # (L, J) upper bound on y_lj; 0 = no arc
    z_up: np.ndarray       # (I, J) upper bound on z_ij; 0 = no arc
    spec_prod: np.ndarray  # (C,) product index of each spec
    spec_alpha: np.ndarray  # (C, K)
    spec_beta: np.ndarray  # (C,)
    h: np.ndarray | None = None  # (L,) pool handling cost per unit throughput
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.h is None:
            self.h = np.zeros(len(self.pool_cap))

    @property
    def I(self): return len(self.c)
    @property
    def L(self): return len(self.pool_cap)
    @property
    def J(self): return len(self.d)

    @property
    def G(self) -> np.ndarray:
        """(C, I) spec coefficient of each source."""
        return self.spec_alpha @ self.lam.T + self.spec_beta[:, None]

    def arcs(self):
        il = [tuple(a) for a in np.argwhere(self.q_up > 0)]
        lj = [tuple(a) for a in np.argwhere(self.y_up > 0)]
        ij = [tuple(a) for a in np.argwhere(self.z_up > 0)]
        return il, lj, ij


# ---------------------------------------------------------------- solution vectors
def unpack(inst: Instance, q, y, z):
    """Dense (I,L), (L,J), (I,J) arrays from dicts or arrays."""
    def dense(v, shape, keys):
        if isinstance(v, np.ndarray):
            return v.astype(float)
        a = np.zeros(shape)
        for k in keys:
            a[k] = v.get(k, 0.0)
        return a
    il, lj, ij = inst.arcs()
    return dense(q, (inst.I, inst.L), il), dense(y, (inst.L, inst.J), lj), dense(z, (inst.I, inst.J), ij)


def evaluate(inst: Instance, q, y, z) -> dict:
    """Objective and constraint violations of a pq-point (no solver involved)."""
    q, y, z = unpack(inst, q, y, z)
    F = z + q @ y                                   # (I, J) source -> product flow
    out_i, in_j = F.sum(1), F.sum(0)
    thr = y.sum(1)
    obj = inst.c @ out_i - inst.d @ in_j + inst.h @ thr
    G = inst.G
    spec = np.array([G[c] @ F[:, inst.spec_prod[c]] for c in range(len(inst.spec_beta))])
    # scale each spec by the flow into its product so violations are in quality units;
    # floor the flow at 1% of product capacity so (near-)unused products don't blow up the ratio
    denom = np.maximum(in_j, 1e-2 * inst.prod_up)[inst.spec_prod]
    spec_scaled = spec / np.maximum(denom, 1e-9)
    has_pool = (inst.q_up > 0).any(0)
    viol = {
        "fraction": float(np.abs(q.sum(0)[has_pool] - 1).max(initial=0)),
        "src": float(np.maximum(np.maximum(inst.src_lo - out_i, out_i - inst.src_up), 0).max(initial=0)),
        "pool_cap": float(np.maximum(thr - inst.pool_cap, 0).max(initial=0)),
        "prod": float(np.maximum(np.maximum(inst.prod_lo - in_j, in_j - inst.prod_up), 0).max(initial=0)),
        "spec_abs": float(np.maximum(spec, 0).max(initial=0)),
        "spec_scaled": float(np.maximum(spec_scaled, 0).max(initial=0)),
        "bounds": float(max(np.maximum(-q, 0).max(), np.maximum(-y, 0).max(), np.maximum(-z, 0).max(initial=0),
                            np.maximum(q - inst.q_up, 0).max(), np.maximum(y - inst.y_up, 0).max(),
                            np.maximum(z - inst.z_up, 0).max(initial=0))),
    }
    spec_slack = -spec_scaled
    return {"obj": float(obj), "F": F, "viol": viol, "max_viol": max(viol[k] for k in viol if k != "spec_abs"),
            "spec_slack": spec_slack}


def binding_specs(inst: Instance, q, y, z, tol=1e-5) -> np.ndarray:
    """Boolean (C,) — spec active (scaled slack <= tol) and its product receives flow."""
    e = evaluate(inst, q, y, z)
    in_j = e["F"].sum(0)
    return (e["spec_slack"] <= tol) & (in_j[inst.spec_prod] > 1e-6)


# ---------------------------------------------------------------- Pyomo models
def build_pyomo(inst: Instance, form: str = "pq", rlt: bool = True) -> pyo.ConcreteModel:
    """rlt=False drops the redundant pq RLT rows (pszrlt, extensions), which are linearly dependent
    on the fraction rows and make IPOPT degenerate. Keep them for relaxation-based solvers."""
    il, lj, ij = inst.arcs()
    m = pyo.ConcreteModel(name=inst.name)
    m.IL, m.LJ, m.IJ = pyo.Set(initialize=il, dimen=2), pyo.Set(initialize=lj, dimen=2), pyo.Set(initialize=ij, dimen=2)
    m.y = pyo.Var(m.LJ, bounds=lambda m, l, j: (0, inst.y_up[l, j]))
    m.z = pyo.Var(m.IJ, bounds=lambda m, i, j: (0, inst.z_up[i, j]))
    pools = sorted({l for _, l in il})
    G = inst.G
    into_l = {l: [i for i, ll in il if ll == l] for l in pools}
    out_l = {l: [j for ll, j in lj if ll == l] for l in pools}

    if form == "pq":
        m.q = pyo.Var(m.IL, bounds=lambda m, i, l: (0, inst.q_up[i, l]))
        def flow(i, j):  # F_ij
            e = sum(m.q[i, l] * m.y[l, j] for (ii, l) in il if ii == i and (l, j) in m.LJ)
            return e + (m.z[i, j] if (i, j) in m.IJ else 0)
        m.fraction = pyo.Constraint(pools, rule=lambda m, l: sum(m.q[i, l] for i in into_l[l]) == 1)
        if rlt:
            m.pszrlt = pyo.Constraint(m.IL, rule=lambda m, i, l:
                                      sum(m.q[i, l] * m.y[l, j] for j in out_l[l]) <= m.q[i, l] * inst.pool_cap[l])
            m.ext = pyo.Constraint(m.LJ, rule=lambda m, l, j: sum(m.q[i, l] * m.y[l, j] for i in into_l[l]) == m.y[l, j])
        else:
            m.cap = pyo.Constraint(pools, rule=lambda m, l: sum(m.y[l, j] for j in out_l[l]) <= inst.pool_cap[l])
    elif form == "p":
        K = inst.lam.shape[1]
        m.x = pyo.Var(m.IL, bounds=lambda m, i, l: (0, min(inst.src_up[i], inst.pool_cap[l])))
        lo = {l: inst.lam[into_l[l]].min(0) for l in pools}
        hi = {l: inst.lam[into_l[l]].max(0) for l in pools}
        m.PK = pyo.Set(initialize=[(l, k) for l in pools for k in range(K)], dimen=2)
        m.p = pyo.Var(m.PK, bounds=lambda m, l, k: (lo[l][k], hi[l][k]))
        m.bal = pyo.Constraint(pools, rule=lambda m, l: sum(m.x[i, l] for i in into_l[l]) == sum(m.y[l, j] for j in out_l[l]))
        m.qbal = pyo.Constraint(m.PK, rule=lambda m, l, k: sum(inst.lam[i, k] * m.x[i, l] for i in into_l[l])
                                == m.p[l, k] * sum(m.y[l, j] for j in out_l[l]))
        m.cap = pyo.Constraint(pools, rule=lambda m, l: sum(m.y[l, j] for j in out_l[l]) <= inst.pool_cap[l])
    else:
        raise ValueError(form)

    def out_i(i):
        if form == "pq":
            e = sum(m.q[i, l] * m.y[l, j] for (ii, l) in il if ii == i for j in out_l[l])
        else:
            e = sum(m.x[i, l] for (ii, l) in il if ii == i)
        return e + sum(m.z[i, j] for (ii, j) in ij if ii == i)

    def in_j(j):
        return sum(m.y[l, j] for (l, jj) in lj if jj == j) + sum(m.z[i, j] for (i, jj) in ij if jj == j)

    I, J = range(inst.I), range(inst.J)
    m.src_lo = pyo.Constraint([i for i in I if inst.src_lo[i] > 0], rule=lambda m, i: out_i(i) >= inst.src_lo[i])
    m.src_up = pyo.Constraint(I, rule=lambda m, i: out_i(i) <= inst.src_up[i])
    m.prod_lo = pyo.Constraint([j for j in J if inst.prod_lo[j] > 0], rule=lambda m, j: in_j(j) >= inst.prod_lo[j])
    m.prod_up = pyo.Constraint(J, rule=lambda m, j: in_j(j) <= inst.prod_up[j])

    C = range(len(inst.spec_beta))
    if form == "pq":
        def spec_rule(m, c):
            j = inst.spec_prod[c]
            return sum(G[c, i] * flow(i, j) for i in I) <= 0
    else:
        def spec_rule(m, c):
            j = inst.spec_prod[c]
            e = sum(G[c, i] * m.z[i, j] for (i, jj) in ij if jj == j)
            e += sum((sum(inst.spec_alpha[c, k] * m.p[l, k] for k in range(inst.lam.shape[1])) + inst.spec_beta[c])
                     * m.y[l, j] for (l, jj) in lj if jj == j)
            return e <= 0
    m.spec = pyo.Constraint(C, rule=spec_rule)
    m.obj = pyo.Objective(expr=sum(inst.c[i] * out_i(i) for i in I) - sum(inst.d[j] * in_j(j) for j in J)
                          + sum(inst.h[l] * sum(m.y[l, j] for j in out_l[l]) for l in pools), sense=pyo.minimize)
    m._form = form
    return m


def pyomo_point(inst: Instance, m):
    """Read a Pyomo model's values back as dense pq arrays (q, y, z)."""
    q = np.zeros((inst.I, inst.L)); y = np.zeros((inst.L, inst.J)); z = np.zeros((inst.I, inst.J))
    for (l, j) in m.LJ: y[l, j] = pyo.value(m.y[l, j], exception=False) or 0.0
    for (i, j) in m.IJ: z[i, j] = pyo.value(m.z[i, j], exception=False) or 0.0
    if m._form == "pq":
        for (i, l) in m.IL: q[i, l] = pyo.value(m.q[i, l], exception=False) or 0.0
    else:
        for (i, l) in m.IL:
            x = pyo.value(m.x[i, l], exception=False) or 0.0
            tot = sum(pyo.value(m.x[ii, ll]) or 0.0 for (ii, ll) in m.IL if ll == l)
            q[i, l] = x / tot if tot > 1e-12 else 0.0
        # pools with zero flow: spread fractions evenly over their arcs
        for l in range(inst.L):
            arcs = np.where(inst.q_up[:, l] > 0)[0]
            if len(arcs) and q[arcs, l].sum() < 1e-12:
                q[arcs, l] = 1.0 / len(arcs)
    return q, y, z


def set_pq_start(inst: Instance, m, q, y, z):
    q, y, z = unpack(inst, q, y, z)
    for (l, j) in m.LJ: m.y[l, j].set_value(float(np.clip(y[l, j], 0, inst.y_up[l, j])), skip_validation=True)
    for (i, j) in m.IJ: m.z[i, j].set_value(float(np.clip(z[i, j], 0, inst.z_up[i, j])), skip_validation=True)
    if m._form == "pq":
        for (i, l) in m.IL: m.q[i, l].set_value(float(np.clip(q[i, l], 0, inst.q_up[i, l])), skip_validation=True)
    else:
        thr = y.sum(1)
        for (i, l) in m.IL: m.x[i, l].set_value(float(q[i, l] * thr[l]), skip_validation=True)
        for (l, k) in m.PK:
            m.p[l, k].set_value(float(q[:, l] @ inst.lam[:, k]), skip_validation=True)


def random_start(inst: Instance, rng: np.random.Generator):
    """Random point inside the variable bounds: Dirichlet fractions per pool, uniform flows."""
    q = np.zeros((inst.I, inst.L))
    for l in range(inst.L):
        arcs = np.where(inst.q_up[:, l] > 0)[0]
        if len(arcs):
            q[arcs, l] = np.minimum(rng.dirichlet(np.ones(len(arcs))), inst.q_up[arcs, l])
    ycap = np.minimum(inst.y_up, np.minimum(inst.pool_cap[:, None], inst.prod_up[None, :]))
    y = rng.uniform(0, 1, ycap.shape) * ycap * (inst.y_up > 0)
    zcap = np.minimum(inst.z_up, np.minimum(inst.src_up[:, None], inst.prod_up[None, :]))
    z = rng.uniform(0, 1, zcap.shape) * zcap * (inst.z_up > 0)
    return q, y, z


_IPOPT = None


def ipopt_solver(options: dict | None = None):
    global _IPOPT
    if _IPOPT is None:
        _IPOPT = pyo.SolverFactory("ipopt", executable=IPOPT_EXE)
    s = _IPOPT
    s.options.update({"tol": 1e-8, "max_iter": 3000, "print_level": 0,
                      "acceptable_tol": 1e-6, "max_cpu_time": 30})
    if options:
        s.options.update(options)
    return s


def solve_ipopt(inst: Instance, m, start=None, options=None) -> dict:
    """One IPOPT run from `start` (q, y, z) on Pyomo model m. Returns evaluated pq-point."""
    import time
    if start is not None:
        set_pq_start(inst, m, *start)
    t0 = time.perf_counter()
    try:
        res = ipopt_solver(options).solve(m, load_solutions=False, tee=False)
        term = str(res.solver.termination_condition)
        if len(res.solution) > 0:
            m.solutions.load_from(res)
    except Exception as e:  # solver crash: record and continue
        term = f"error:{type(e).__name__}"
    t = time.perf_counter() - t0
    q, y, z = pyomo_point(inst, m)
    ev = evaluate(inst, q, y, z)
    return {"term": term, "time": t, "obj": ev["obj"], "max_viol": ev["max_viol"], "q": q, "y": y, "z": z}


# ---------------------------------------------------------------- SCIP (global)
def solve_scip(inst: Instance, time_limit: float = 60.0, gap: float = 1e-6, threads: int = 1,
               start=None, verbose: bool = False) -> dict:
    """Global solve of the pq-formulation with SCIP."""
    import time
    from pyscipopt import Model, quicksum
    il, lj, ij = inst.arcs()
    s = Model(inst.name)
    if not verbose:
        s.hideOutput()
    s.setParam("limits/time", time_limit)
    s.setParam("limits/gap", gap)
    s.setParam("limits/absgap", 1e-9)
    s.setParam("parallel/maxnthreads", threads)
    s.setParam("display/verblevel", 0 if not verbose else 4)
    q = {(i, l): s.addVar(lb=0, ub=float(inst.q_up[i, l]), name=f"q_{i}_{l}") for (i, l) in il}
    y = {(l, j): s.addVar(lb=0, ub=float(inst.y_up[l, j]), name=f"y_{l}_{j}") for (l, j) in lj}
    z = {(i, j): s.addVar(lb=0, ub=float(inst.z_up[i, j]), name=f"z_{i}_{j}") for (i, j) in ij}
    pools = sorted({l for _, l in il})
    into_l = {l: [i for i, ll in il if ll == l] for l in pools}
    out_l = {l: [j for ll, j in lj if ll == l] for l in pools}
    # explicit product variables w_ilj = q_il * y_lj keep every constraint quadratic-with-few-terms
    w = {}
    for (i, l) in il:
        for j in out_l[l]:
            w[i, l, j] = s.addVar(lb=0, ub=float(min(inst.y_up[l, j], inst.q_up[i, l] * inst.y_up[l, j])), name=f"w_{i}_{l}_{j}")
            s.addCons(w[i, l, j] == q[i, l] * y[l, j])
    F = {(i, j): quicksum(w[i, l, j] for l in pools if (i, l) in q and (l, j) in y) + (z[i, j] if (i, j) in z else 0)
         for i in range(inst.I) for j in range(inst.J)}
    for l in pools:
        s.addCons(quicksum(q[i, l] for i in into_l[l]) == 1)
        for j in out_l[l]:
            s.addCons(quicksum(w[i, l, j] for i in into_l[l]) == y[l, j])
        for i in into_l[l]:
            s.addCons(quicksum(w[i, l, j] for j in out_l[l]) <= inst.pool_cap[l] * q[i, l])
        s.addCons(quicksum(y[l, j] for j in out_l[l]) <= inst.pool_cap[l])
    for i in range(inst.I):
        out = quicksum(F[i, j] for j in range(inst.J))
        s.addCons(out <= inst.src_up[i])
        if inst.src_lo[i] > 0:
            s.addCons(out >= inst.src_lo[i])
    for j in range(inst.J):
        inn = quicksum(F[i, j] for i in range(inst.I))
        s.addCons(inn <= inst.prod_up[j])
        if inst.prod_lo[j] > 0:
            s.addCons(inn >= inst.prod_lo[j])
    G = inst.G
    for c in range(len(inst.spec_beta)):
        j = inst.spec_prod[c]
        s.addCons(quicksum(float(G[c, i]) * F[i, j] for i in range(inst.I)) <= 0)
    obj = (quicksum(float(inst.c[i] - inst.d[j]) * F[i, j] for i in range(inst.I) for j in range(inst.J))
           + quicksum(float(inst.h[l]) * y[l, j] for (l, j) in lj))
    s.setObjective(obj, "minimize")
    t0 = time.perf_counter()
    s.optimize()
    t = time.perf_counter() - t0
    status = s.getStatus()
    out = {"status": status, "time": t, "nodes": s.getNNodes()}
    if s.getNSols() > 0:
        sol = s.getBestSol()
        qa = np.zeros((inst.I, inst.L)); ya = np.zeros((inst.L, inst.J)); za = np.zeros((inst.I, inst.J))
        for k, v in q.items(): qa[k] = s.getSolVal(sol, v)
        for k, v in y.items(): ya[k] = s.getSolVal(sol, v)
        for k, v in z.items(): za[k] = s.getSolVal(sol, v)
        out.update(obj=s.getObjVal(), dual=s.getDualbound(), gap=s.getGap(), q=qa, y=ya, z=za,
                   max_viol=evaluate(inst, qa, ya, za)["max_viol"])
    else:
        out.update(obj=np.nan, dual=s.getDualbound(), gap=np.inf)
    return out


def rel_gap(obj, ref):
    return (obj - ref) / max(1.0, abs(ref))
