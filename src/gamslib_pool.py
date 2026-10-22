"""Read the literature pooling instances from GAMS model library `pool` (data/raw/gamslib/pool.254).

GAMS tables are fixed-width: a value belongs to the column header whose text it overlaps.
Defaults follow the GAMS model: missing entries are 0 (so a missing quality upper bound means 0).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from pooling import Instance

ROOT = Path(__file__).resolve().parents[1]
POOL_FILE = ROOT / "data" / "raw" / "gamslib" / "pool.254"


def _expand(label: str) -> list[str]:
    m = re.fullmatch(r"([A-Za-z]+)(\d+)\*\1?(\d+)", label)  # o1*o2, p6*p16
    if m:
        return [f"{m.group(1)}{k}" for k in range(int(m.group(2)), int(m.group(3)) + 1)]
    return [label]


def _parse_table(lines: list[str]) -> dict[tuple, float]:
    """lines = header line + data lines (comments removed). Returns {(row..., col): value}."""
    out = {}
    header = None
    for ln in lines:
        if ln.strip().startswith("+"):
            ln = ln.replace("+", " ", 1)
            header = None
        if header is None:
            header = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", ln)]
            continue
        toks = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", ln)]
        if not toks:
            continue
        row = toks[0][2].rstrip(";")
        for a, b, val in toks[1:]:
            val = val.rstrip(";")
            if not val:
                continue
            # header with largest overlap, else nearest
            best = max(header, key=lambda h: (min(b, h[1]) - max(a, h[0]), -abs((a + b) / 2 - (h[0] + h[1]) / 2)))
            parts = row.split(".")
            heads = _expand(parts[0]) if len(parts) == 1 else None
            for col in _expand(best[2]):
                if len(parts) == 1:
                    for r in heads:
                        out[(r, col)] = float(val)
                else:
                    for r2 in _expand(parts[1]):
                        out[(parts[0], r2, col)] = float(val)
    return out


def read_tables(path: Path = POOL_FILE) -> dict[str, dict]:
    text = path.read_text(encoding="latin-1").splitlines()
    tables, i = {}, 0
    while i < len(text):
        m = re.match(r"^Table\s+(\w+)\(", text[i])
        if m:
            name, body = m.group(1), []
            i += 1
            while i < len(text):
                ln = text[i]
                if not ln.lstrip().startswith("*"):
                    body.append(ln.split("//")[0])
                if ";" in ln:
                    break
                i += 1
            tables[name] = _parse_table([b for b in body if b.strip()])
        i += 1
    sol = {}
    m = re.search(r"Parameter sol\(case\).*?/(.*?)/;", "\n".join(text), re.S)
    for ln in m.group(1).splitlines():
        t = ln.split()
        if len(t) == 2:
            name, v = t
            mm = re.fullmatch(r"([a-z]+)(\d)\*\1(\d)", name)
            names = [f"{mm.group(1)}{k}" for k in range(int(mm.group(2)), int(mm.group(3)) + 1)] if mm else [name]
            for n in names:
                sol[n] = float(v)
    tables["sol"] = sol
    return tables


def load_case(case: str, tables: dict | None = None) -> Instance:
    T = tables or read_tables()
    sel = lambda t: {k[1:]: v for k, v in T[t].items() if k[0] == case}
    comp = sel("ComponentData")
    comps = sorted({k[0] for k in comp}, key=lambda s: int(s[1:]))
    prod = sel("ProductData")
    prods = sorted({k[0] for k in prod}, key=lambda s: int(s[1:]))
    psize = {k[1]: v for k, v in T["Poolsize"].items() if k[0] == case}
    pools = sorted(psize, key=lambda s: int(s[1:]))
    cq = sel("ComponentQuality")
    quals = sorted({k[1] for k, v in cq.items() if k[0] in comps}, key=lambda s: int(s[1:]))
    I, L, J, K = len(comps), len(pools), len(prods), len(quals)
    ci = {c: n for n, c in enumerate(comps)}; li = {p: n for n, p in enumerate(pools)}; ji = {p: n for n, p in enumerate(prods)}
    g = lambda d, k: d.get(k, 0.0)
    c = np.array([g(comp, (x, "price")) for x in comps])
    src_lo = np.array([g(comp, (x, "lo")) for x in comps]); src_up = np.array([g(comp, (x, "up")) for x in comps])
    d = np.array([g(prod, (x, "price")) for x in prods])
    prod_lo = np.array([g(prod, (x, "lo")) for x in prods]); prod_up = np.array([g(prod, (x, "up")) for x in prods])
    lam = np.array([[g(cq, (x, k)) for k in quals] for x in comps])
    q_up = np.zeros((I, L)); y_up = np.zeros((L, J)); z_up = np.zeros((I, J))
    for (cc, pp), v in sel("ComponentPoolFraction").items(): q_up[ci[cc], li[pp]] = v
    for (pp, jj), v in sel("PoolProductBound").items():
        if jj in ji: y_up[li[pp], ji[jj]] = v
    for (cc, jj), v in sel("ComponentProductBound").items():
        if jj in ji: z_up[ci[cc], ji[jj]] = v
    lo, up = sel("ProductQualityLower"), sel("ProductQualityUpper")
    sp, sa, sb = [], [], []
    for j, pj in enumerate(prods):
        for k, qk in enumerate(quals):
            e = np.zeros(K); e[k] = 1
            sp += [j, j]; sa += [-e, e.copy()]; sb += [g(lo, (pj, qk)), -g(up, (pj, qk))]
    return Instance(name=case, c=c, d=d, src_lo=src_lo, src_up=src_up, pool_cap=np.array([psize[p] for p in pools]),
                    prod_lo=prod_lo, prod_up=prod_up, lam=lam, q_up=q_up, y_up=y_up, z_up=z_up,
                    spec_prod=np.array(sp), spec_alpha=np.array(sa), spec_beta=np.array(sb),
                    meta={"best_known": T["sol"].get(case), "comps": comps, "pools": pools, "prods": prods, "quals": quals})
