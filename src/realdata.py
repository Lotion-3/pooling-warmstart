"""Real-data crude-blending pooling instances (Phase 2).

Network: 6-8 crudes -> 3 storage tanks -> 2 crude distillation units (CDU-A sour-tolerant,
CDU-B low-sulfur), plus a few direct crude -> CDU pipelines.
Tracked qualities (volume-blending):  lam = [S*SG, SG, TAN*SG]
  sulfur wt%  (mass-blended):  sum (S_i - Smax) SG_i F_ij <= 0      alpha = (1, -Smax, 0)
  TAN mgKOH/g (mass-blended):  sum (T_i - Tmax) SG_i F_ij <= 0      alpha = (0, -Tmax, 1)
  SG bounds   (volume-blended): SGmin <= SG <= SGmax  (API limits converted to SG)
Crude prices are MODELED: Brent + a_y (API - 35) + b_y (S - 1), see fit_price_model().
Objective (min): sum_i (cost_i - worth_i) F_i + sum_l h_l thr_l   [k$/day with flows in kbd]
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pooling import Instance

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
RAW = ROOT / "data" / "raw"
WATER_SG15 = 0.99904
RESID_FRACTION_OF_BRENT = 0.85
API_REF, S_REF = 35.0, 1.0
CORE = ["brent_europe_usd_per_bbl", "gasoline_conv_reg_nyh_usd_per_gal", "gasoline_conv_reg_usgc_usd_per_gal",
        "ulsd_nyh_usd_per_gal", "ulsd_usgc_usd_per_gal", "jet_kero_usgc_usd_per_gal"]


def api_to_sg(api):
    return 141.5 / (np.asarray(api, float) + 131.5)


# ---------------------------------------------------------------- yields
def _cut_table(path: Path, yield_label: str):
    d = pd.read_excel(path, sheet_name="Summary", header=None)
    s = d.map(lambda v: v.strip() if isinstance(v, str) else v)
    r = [i for i in range(len(s)) if str(s.iat[i, 1]).startswith("Start")][0]
    y = [i for i in range(r, len(s)) if s.iat[i, 1] == yield_label][0]
    cols = []
    for c in range(2, s.shape[1]):
        a, b = s.iat[r, c], s.iat[r + 1, c]
        if str(a) in ("nan", ""):
            break  # BP sheets may carry a second, unlabeled copy to the right
        cols.append((str(a), str(b), pd.to_numeric(d.iat[y, c], errors="coerce")))
    return cols


def _temp(x):
    return {"IBP": -1e9, "C4": 0.0, "C5": 0.0, "FBP": 1e9}.get(x, None) if not x.replace(".", "").isdigit() else float(x)


def crude_yields(raw_file: str, source: str) -> dict:
    """Volume fractions: lpg, naphtha (<=150C), kero (<=250C), gasoil (<=370C), resid (atmospheric residue)."""
    lab = "Yield (% vol)" if source.startswith("Equinor") else "Yield on crude (% vol)"
    cols = [(a, b, v) for a, b, v in _cut_table(ROOT / raw_file, lab) if not (a == "IBP" and b == "FBP")]
    # distillate chain: cuts with numeric/C4 end <= 371; residue: the X-FBP column starting at the chain end
    dist = [(a, b, v) for a, b, v in cols if b != "FBP"]
    out = {"lpg": 0.0, "naphtha": 0.0, "kero": 0.0, "gasoil": 0.0}
    chain_end = None
    for a, b, v in dist:
        tb = _temp(b)
        if tb > 371:  # vacuum cuts belong to the residue
            continue
        if chain_end is not None and _temp(a) < chain_end - 1e-6:
            continue  # overlapping cut
        key = "lpg" if b == "C4" else "naphtha" if tb <= 151 else "kero" if tb <= 251 else "gasoil"
        out[key] += float(np.nan_to_num(v))
        chain_end = tb
    res = [v for a, b, v in cols if b == "FBP" and abs(_temp(a) - chain_end) < 2]
    assert res, f"{raw_file}: no atmospheric residue column starting at {chain_end}"
    out["resid"] = float(res[0])
    tot = sum(out.values())
    if source.startswith("BP") and tot < 100:  # BP chains start at C5: light ends are the remainder
        out["lpg"] += 100 - tot
        tot = 100.0
    return {k: v / tot for k, v in out.items()}


def load_crudes() -> pd.DataFrame:
    c = pd.read_csv(CLEAN / "crudes.csv")
    Y = pd.DataFrame([crude_yields(r.raw_file, r.source) for r in c.itertuples()])
    c = pd.concat([c, Y.add_prefix("y_")], axis=1)
    c["sg"] = c["density_15c_g_cc"] / WATER_SG15
    c["tan_mgkoh_g"] = c["tan_mgkoh_g"].fillna(c["tan_mgkoh_g"].median())
    c["key"] = c["name"].str.lower().str.replace(r" blend| export| \(.*\)", "", regex=True).str.strip()
    c["crude_id"] = [f"{'EQ' if s.startswith('Equinor') else 'BP'}:{n}" for s, n in zip(c.source, c.name)]
    return c


# ---------------------------------------------------------------- prices
def load_prices() -> pd.DataFrame:
    p = pd.read_csv(CLEAN / "prices_daily.csv", parse_dates=["date"], index_col="date")
    p = p.loc["2006-06-14":"2026-10-20"]
    p = p[p[CORE].notna().all(1)].copy()
    # propane (light ends) has occasional gaps inside the core panel: carry last value forward
    p["propane_mont_belvieu_usd_per_gal"] = p["propane_mont_belvieu_usd_per_gal"].ffill().bfill()
    return p


def _eia_monthly(name):
    d = pd.read_excel(RAW / "eia_monthly" / f"{name}.xls", sheet_name="Data 1", header=None)
    body = d.iloc[3:].copy()
    body.columns = ["date"] + list(d.iloc[1, 1:])
    body["date"] = pd.to_datetime(body["date"]).dt.to_period("M")
    return body.set_index("date").apply(pd.to_numeric, errors="coerce"), dict(zip(d.iloc[1, 1:], d.iloc[2, 1:]))


BAND_API = {"I000000208": 17.5, "I000000258": 22.5, "I000000308": 27.5, "I000000358": 32.5,
            "I000000408": 37.5, "I000000458": 42.5, "I000000998": 47.5}


def fit_price_model(crudes: pd.DataFrame, prices: pd.DataFrame) -> dict:
    """diff_{k,t} = landed_{k,t} - Brent_t = c_t + a_y (API_k - 35) + b_y (S_k - 1) + e.

    Observations: EIA landed cost by API band (band-midpoint API; sulfur proxy = median sulfur of
    our 82 assays in that band) and the Iraqi Basrah Light stream (API/S proxy: BP Basrah Medium
    assay, the 2021 successor grade). Month fixed effects c_t absorb freight and Brent basis.
    """
    bands, names = _eia_monthly("PET_PRI_LAND3_K_M")
    streams, snames = _eia_monthly("PET_PRI_LAND2_K_M")
    brent = prices["brent_europe_usd_per_bbl"].groupby(prices.index.to_period("M")).mean()
    edges = [-1, 20.05, 25.05, 30.05, 35.05, 40.05, 45.05, 99]
    band_s = {}
    for k, (lo, hi) in zip(BAND_API, zip(edges[:-1], edges[1:])):
        sel = crudes[(crudes.api > lo) & (crudes.api <= hi)]
        band_s[k] = float(sel.sulfur_pct.median()) if len(sel) else np.nan
    bm = crudes[crudes.crude_id == "BP:Basrah Medium"].iloc[0]
    obs = []
    for k, api in BAND_API.items():
        if np.isnan(band_s[k]):
            continue
        for t, v in bands[k].dropna().items():
            if t in brent.index:
                obs.append((t, api, band_s[k], v - brent[t], f"band_{api}"))
    for t, v in streams["I36100000008"].dropna().items():
        if t in brent.index:
            obs.append((t, bm.api, bm.sulfur_pct, v - brent[t], "basrah_light"))
    df = pd.DataFrame(obs, columns=["month", "api", "s", "diff", "series"])
    df["year"] = df.month.dt.year

    def ols(sub, by_year):
        months = sorted(sub.month.unique())
        mi = {m: n for n, m in enumerate(months)}
        X_fe = np.zeros((len(sub), len(months)))
        X_fe[np.arange(len(sub)), sub.month.map(mi).values] = 1
        years = sorted(sub.year.unique()) if by_year else [None]
        cols = [X_fe]
        for y in years:
            msk = (sub.year == y).values if y is not None else np.ones(len(sub), bool)
            cols += [((sub.api - API_REF) * msk).values[:, None], ((sub.s - S_REF) * msk).values[:, None]]
        X = np.hstack(cols)
        beta, *_ = np.linalg.lstsq(X, sub["diff"].values, rcond=None)
        res = sub["diff"].values - X @ beta
        dof = len(sub) - np.linalg.matrix_rank(X)
        s2 = res @ res / dof
        cov = s2 * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        tss = ((sub["diff"] - sub.groupby("month")["diff"].transform("mean")) ** 2).sum()
        within_r2 = 1 - (res @ res) / tss
        out = {}
        for n, y in enumerate(years):
            ia, ib = len(months) + 2 * n, len(months) + 2 * n + 1
            out[str(y) if y else "pooled"] = {"a": beta[ia], "b": beta[ib], "se_a": se[ia], "se_b": se[ib]}
        return out, {"n_obs": int(len(sub)), "resid_sd": float(np.sqrt(s2)), "within_r2": float(within_r2)}

    pooled, pdiag = ols(df, False)
    yearly, ydiag = ols(df, True)
    joint = {"pooled": pooled["pooled"], "pooled_diag": pdiag, "yearly": yearly, "yearly_diag": ydiag}
    fallback = None
    if pooled["pooled"]["b"] >= 0:
        # Sulfur is not identified with the right sign from these proxies (see DECISIONS.md):
        # refit on API bands only with the sulfur term removed (b = 0); the API slope then carries
        # the heavy/sour discount through the API-sulfur correlation of imported crudes.
        fallback = "api_only_bands"
        df = df[df.series != "basrah_light"].copy()
        df["s"] = S_REF
        pooled, pdiag = ols_api(df, False, ols)
        yearly, ydiag = ols_api(df, True, ols)
    p = pooled["pooled"]
    # per-year coefficients are used only if every year has the economically expected signs
    # (lighter = dearer, sourer = cheaper) and the fit improves; otherwise pooled for all years
    ok = all(v["a"] > 0 and (v["b"] < 0 or fallback) for v in yearly.values())
    use_yearly = ok and ydiag["resid_sd"] < pdiag["resid_sd"]
    coef = {y: ({"a": v["a"], "b": v["b"]} if use_yearly else {"a": p["a"], "b": p["b"]}) for y, v in yearly.items()}
    model = {"form": "crude_price = Brent + a_y*(API-35) + b_y*(S-1)  [$/bbl]",
             "api_ref": API_REF, "s_ref": S_REF, "use_yearly": bool(use_yearly),
             "yearly_signs_ok": bool(ok), "coefficients_used": coef,
             "pooled": p, "pooled_diag": pdiag, "yearly": yearly, "yearly_diag": ydiag,
             "fallback": fallback, "joint_api_sulfur_fit_rejected": joint if fallback else None,
             "band_sulfur_proxy": band_s, "basrah_proxy": {"api": float(bm.api), "s": float(bm.sulfur_pct)},
             "obs_by_series": df.series.value_counts().to_dict()}
    return json.loads(json.dumps(model, default=float)), df


def ols_api(df, by_year, ols):
    """API-only version of the fixed-effects regression (sulfur column is constant, so b is not estimated)."""
    df = df.copy()
    df["s"] = S_REF + 0.0  # (S - S_ref) == 0  ->  the b column is all zeros
    out, diag = ols(df, by_year)
    for v in out.values():
        v["b"], v["se_b"] = 0.0, float("nan")
    return out, diag


def crude_cost(api, s, brent, year, model):
    co = model["coefficients_used"].get(str(year))
    if co is None:  # years outside the fit (e.g. after the last EIA month): nearest year
        ys = sorted(int(y) for y in model["coefficients_used"])
        co = model["coefficients_used"][str(min(ys, key=lambda y: abs(y - year)))]
    return brent + co["a"] * (api - API_REF) + co["b"] * (s - S_REF)


def crude_worth(crudes: pd.DataFrame, prow: pd.Series) -> np.ndarray:
    """Gross product worth per bbl of crude ($/bbl) at one date's prices."""
    gal = 42.0
    px = {"lpg": prow["propane_mont_belvieu_usd_per_gal"] * gal,
          "naphtha": prow["gasoline_conv_reg_usgc_usd_per_gal"] * gal,
          "kero": prow["jet_kero_usgc_usd_per_gal"] * gal,
          "gasoil": prow["ulsd_usgc_usd_per_gal"] * gal,
          "resid": RESID_FRACTION_OF_BRENT * prow["brent_europe_usd_per_bbl"]}
    return sum(crudes[f"y_{k}"].values * v for k, v in px.items())


# ---------------------------------------------------------------- instances
DEFAULT_DESIGN = {
    "n_src": (6, 8), "n_pools": 3, "pool_cap": 60.0, "avail": (15.0, 50.0),
    "cdu": {"A": {"cap": 100.0, "min_frac": 0.3, "smax": 2.0, "api": (26.0, 36.0), "tan": 0.8},
            "B": {"cap": 80.0, "min_frac": 0.3, "smax": 0.8, "api": (30.0, 40.0), "tan": 0.8}},
    "jitter": 0.10, "pool_links": (1, 2), "p_direct": 0.3, "handling": (0.10, 0.30),
    "extra_specs": False,
}


def sample_instance(crudes: pd.DataFrame, rng: np.random.Generator, design: dict) -> dict:
    """Random network + specs (date independent). Returns plain-python parameters."""
    D = design
    n = int(rng.integers(D["n_src"][0], D["n_src"][1] + 1))
    sour = crudes.index[crudes.sulfur_pct > 1.0].tolist()
    lsw = crudes.index[crudes.type == "light sweet"].tolist()
    pick = [int(rng.choice(sour))]
    pick.append(int(rng.choice([i for i in lsw if crudes.key[i] != crudes.key[pick[0]]])))
    while len(pick) < n:
        i = int(rng.integers(len(crudes)))
        if i not in pick and crudes.key[i] not in set(crudes.key[pick]):
            pick.append(i)
    pick = sorted(pick, key=lambda i: crudes.api[i])  # canonical order: by API
    L = D["n_pools"]
    q_up = np.zeros((n, L))
    for i in range(n):
        k = int(rng.integers(D["pool_links"][0], D["pool_links"][1] + 1))
        q_up[i, rng.choice(L, size=min(k, L), replace=False)] = 1.0
    for l in range(L):  # every tank gets at least two crudes
        while q_up[:, l].sum() < 2:
            q_up[int(rng.integers(n)), l] = 1.0
    jit = lambda v: float(v * rng.uniform(1 - D["jitter"], 1 + D["jitter"]))
    cdus = []
    for name in ["A", "B"]:
        c = D["cdu"][name]
        lo, hi = sorted([jit(c["api"][0]), jit(c["api"][1])])
        if hi - lo < 4:
            mid = (lo + hi) / 2; lo, hi = mid - 2, mid + 2
        cap = jit(c["cap"])
        cdus.append({"name": name, "cap": cap, "min": c["min_frac"] * cap, "smax": jit(c["smax"]),
                     "api_lo": lo, "api_hi": hi, "tan": jit(c["tan"]),
                     "s2": jit(c["s2"]) if "s2" in c else None})
    J = 2
    y_up = np.ones((L, J)) * D["pool_cap"]
    z_up = np.where(rng.uniform(size=(n, J)) < D["p_direct"], 1.0, 0.0)
    avail = rng.uniform(*D["avail"], size=n)
    z_up = z_up * np.minimum(avail[:, None], [c["cap"] for c in cdus])
    pool_cap = np.array([jit(D["pool_cap"]) for _ in range(L)])
    return {"crude_idx": pick, "q_up": q_up, "y_up": np.minimum(y_up, pool_cap[:, None]), "z_up": z_up,
            "avail": avail, "pool_cap": pool_cap, "cdus": cdus, "h": rng.uniform(*D["handling"], size=L)}


def build_instance(crudes: pd.DataFrame, net: dict, prow: pd.Series, date, model: dict, name: str) -> Instance:
    cr = crudes.loc[net["crude_idx"]]
    sg = cr.sg.values
    lam = np.column_stack([cr.sulfur_pct.values * sg, sg, cr.tan_mgkoh_g.values * sg])
    worth = crude_worth(cr, prow)
    cost = crude_cost(cr.api.values, cr.sulfur_pct.values, prow["brent_europe_usd_per_bbl"], date.year, model)
    sp, sa, sb = [], [], []
    for j, c in enumerate(net["cdus"]):
        sg_lo, sg_hi = api_to_sg(c["api_hi"]), api_to_sg(c["api_lo"])
        rows = [((1, -c["smax"], 0), 0.0),     # sulfur (mass)
                ((0, 1, 0), -sg_hi),           # SG <= SG(api_lo)
                ((0, -1, 0), sg_lo),           # SG >= SG(api_hi)
                ((0, -c["tan"], 1), 0.0)]      # TAN (mass)
        for a, b in rows:
            sp.append(j); sa.append(a); sb.append(b)
    caps = np.array([c["cap"] for c in net["cdus"]])
    return Instance(name=name, c=cost - worth, d=np.zeros(2), src_lo=np.zeros(len(cost)), src_up=net["avail"],
                    pool_cap=net["pool_cap"], prod_lo=np.array([c["min"] for c in net["cdus"]]), prod_up=caps,
                    lam=lam, q_up=net["q_up"], y_up=net["y_up"], z_up=net["z_up"],
                    spec_prod=np.array(sp), spec_alpha=np.array(sa, float), spec_beta=np.array(sb), h=net["h"],
                    meta={"crudes": cr.crude_id.tolist(), "cost": cost, "worth": worth})


# ---- (de)serialisation to a flat parquet row
ARR = ["c", "d", "src_lo", "src_up", "pool_cap", "prod_lo", "prod_up", "lam", "q_up", "y_up", "z_up",
       "spec_prod", "spec_alpha", "spec_beta", "h"]


def to_row(inst: Instance) -> dict:
    row = {"name": inst.name}
    for k in ARR:
        a = np.asarray(getattr(inst, k))
        row[k] = a.ravel().tolist()
        row[k + "_shape"] = list(a.shape)
    return row


def from_row(row) -> Instance:
    kw = {k: np.asarray(row[k], float).reshape(tuple(int(s) for s in row[k + "_shape"])) for k in ARR}
    kw["spec_prod"] = kw["spec_prod"].astype(int)
    return Instance(name=row["name"], **kw)
