"""U.S. bank call report balance sheets (FDIC BankFind Suite financials API, real data).

Sample = (FDIC certificate CERT, quarter t) for a bank that files in both t and t+1, 2015Q1 - 2024Q4.
Inputs : the 18 balance-sheet items below at quarter-end t.
Outputs: the same 18 items at quarter-end t+1 (components and totals).
Equalities (call report schedule RC, as aggregated by the FDIC, $ thousands):
    ASSET = CHBAL + FREPO + SC + TRADE + LNLSNET + BKPREM + ORE + INTAN + AOA
    LIAB  = DEP + FREPP + TRADEL + OTHBRF + SUBND + ALLOTHL
    ASSET = LIAB + EQTOT          (EQTOT = bank equity EQ + noncontrolling interests EQCONSUB)
All three are checked on the data by the gate.  In ~3.5% of 2015Q2-2021Q1 target rows the published
ASSET and LIAB + EQTOT (= LIABEQ) differ by exactly 1 ($1,000, rounding in the published aggregates);
that is ~1.4e-7 of the row's term scale, so labels are used as published.  In the full panel the
asset-component sum is exact except in 10 bank-quarters: 8 differ by 1-3 and one bank (CERT 23992)
differs by 520 and 501 in 2019Q4 / 2020Q1 (~7e-5 of the term scale); those two pairs are not in
the fixed-seed subsample.

Rows lacking any of the items are dropped: these are insured U.S. branches of foreign banks, which
report no equity capital (EQTOT and several components missing, LIAB = ASSET).

Raw files: python -P bench/scripts/get_bank_balance_sheets.py
"""
import json

import numpy as np
import pandas as pd

from bench.domain import CACHE, RAW, Domain

DIR = RAW / "bank_balance_sheets"
ENDPOINT = "https://api.fdic.gov/banks/financials"
N_MAX = 50_000
SEED = 0

ASSETS = ["CHBAL", "FREPO", "SC", "TRADE", "LNLSNET", "BKPREM", "ORE", "INTAN", "AOA"]
LIABS = ["DEP", "FREPP", "TRADEL", "OTHBRF", "SUBND", "ALLOTHL"]
VARS = ["ASSET"] + ASSETS + ["LIAB"] + LIABS + ["EQTOT"]
DESC = {
    "ASSET": "total assets", "CHBAL": "cash and balances due from depository institutions",
    "FREPO": "federal funds sold and reverse repurchase agreements", "SC": "total securities",
    "TRADE": "trading account assets", "LNLSNET": "loans and leases, net of allowance",
    "BKPREM": "premises and fixed assets", "ORE": "other real estate owned", "INTAN": "intangible assets",
    "AOA": "all other assets", "LIAB": "total liabilities", "DEP": "total deposits",
    "FREPP": "federal funds purchased and repurchase agreements", "TRADEL": "trading liabilities",
    "OTHBRF": "other borrowed money", "SUBND": "subordinated notes and debentures",
    "ALLOTHL": "all other liabilities", "EQTOT": "total equity capital incl. noncontrolling interests",
}
# non-negative by construction; INTAN has a few small negative published values, EQTOT can be negative
NONNEG = [v for v in VARS if v not in ("INTAN", "EQTOT")]


def _quarter(path):
    js = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(js["rows"])
    if df.empty:
        raise ValueError(f"{path.name}: no rows")
    df = df[["CERT"] + VARS].apply(pd.to_numeric, errors="coerce")
    n0 = len(df)
    df = df.dropna()
    return int(js["repdte"]), df.set_index("CERT").astype(np.int64), n0 - len(df), n0, js["downloaded"]


def _build(files, subsample=True):
    """Pairs (bank, t) -> (bank, t+1) over consecutive quarter files, then the fixed-seed subsample."""
    qs = [_quarter(f) for f in files]
    X, Y, key = [], [], []
    for (_, d0, *_), (t1, d1, *_) in zip(qs[:-1], qs[1:]):
        common = d0.index.intersection(d1.index)
        X.append(d0.loc[common, VARS].to_numpy())
        Y.append(d1.loc[common, VARS].to_numpy())
        key.append(np.full(len(common), t1))
    X, Y, key = (np.concatenate(a) for a in (X, Y, key))
    info = {"dates": [q[0] for q in qs], "dropped": sum(q[2] for q in qs), "rows_total": sum(q[3] for q in qs),
            "stamps": sorted({q[4][:10] for q in qs}), "n_pairs": len(key)}
    if subsample and len(key) > N_MAX:  # uniform over (bank, quarter) pairs, fixed seed
        keep = np.sort(np.random.default_rng(SEED).choice(len(key), N_MAX, replace=False))
        X, Y, key = X[keep], Y[keep], key[keep]
    return X, Y, key, info


def _cached(files):
    """Parsing the 40 JSON files takes ~10 s; reuse the arrays while the raw files are unchanged."""
    sig = "|".join(f"{f.name}:{f.stat().st_size}" for f in files) + f"|{','.join(VARS)}|{N_MAX}|{SEED}"
    path = CACHE / "bank_balance_sheets.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if str(z["sig"]) == sig:
                return z["X"], z["Y"], z["key"], json.loads(str(z["info"]))
    X, Y, key, info = _build(files)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, Y=Y, key=key, sig=sig, info=json.dumps(info))
    return X, Y, key, info


def load():
    files = sorted(DIR.glob("financials_*.json"))
    if len(files) < 3:
        raise FileNotFoundError(f"{DIR} has {len(files)} quarter files: run "
                                "python -P bench/scripts/get_bank_balance_sheets.py")
    X, Y, key, info = _cached(files)
    dates, dropped, rows_total, n_pairs = info["dates"], info["dropped"], info["rows_total"], info["n_pairs"]
    stamps = info["stamps"]

    iy = {v: i for i, v in enumerate(VARS)}
    q = len(VARS)
    A = np.zeros((3, q))
    A[0, iy["ASSET"]] = 1.0
    A[0, [iy[v] for v in ASSETS]] = -1.0
    A[1, iy["LIAB"]] = 1.0
    A[1, [iy[v] for v in LIABS]] = -1.0
    A[2, [iy["ASSET"], iy["LIAB"], iy["EQTOT"]]] = [1.0, -1.0, -1.0]
    y_lo = np.full(q, -np.inf)
    y_lo[[iy[v] for v in NONNEG]] = 0.0

    fields = ",".join(["CERT", "REPDTE"] + VARS + ["EQ", "EQCONSUB", "LIABEQ"])
    return Domain(
        "bank_balance_sheets", X, Y, A, None, np.zeros(3), split=key, split_kind="time", y_lo=y_lo,
        x_names=[f"{v}_t" for v in VARS], y_names=[f"{v}_tp1" for v in VARS],
        row_names=["asset_components", "liability_components", "assets_eq_liabilities_plus_equity"],
        meta={"source": (f"FDIC BankFind Suite financials API {ENDPOINT}?filters=REPDTE:<YYYYMMDD>"
                         f"&fields={fields}&limit=10000&sort_by=CERT&sort_order=ASC&format=json, one request "
                         f"per quarter-end {dates[0]}..{dates[-1]} ({len(dates)} quarters); downloaded "
                         f"{', '.join(stamps)}. Underlying data: FFIEC Consolidated Reports of Condition and "
                         "Income (call reports, FFIEC 031/041/051) as aggregated by the FDIC."),
              "license": ("FDIC BankFind data are public information compiled by a U.S. federal agency "
                          "(works of the U.S. government are not subject to copyright, 17 U.S.C. 105); "
                          "the API is open with no key or account. No separate FDIC license text was found in the API "
                          "docs; cite the FDIC as the source."),
              "tier": 2,
              "units": "$ thousands (integers as published), original units, no transform",
              "generation": (f"{rows_total} bank-quarter rows in {len(dates)} quarterly files; {dropped} rows "
                             "dropped for missing items (insured U.S. branches of foreign banks, which report "
                             "no equity). Each sample pairs a bank's quarter t with its quarter t+1 "
                             "(inner join on CERT of consecutive quarter-ends), giving "
                             f"{n_pairs} pairs; {'subsampled uniformly to ' + str(N_MAX) + ' with seed ' + str(SEED) if n_pairs > N_MAX else 'all kept'}. "
                             "Split key = target quarter-end t+1 (YYYYMMDD). Banks that merge or fail drop "
                             "out; balance sheets after a merger are used as reported. EQTOT includes "
                             "noncontrolling interests (EQTOT = EQ + EQCONSUB holds exactly in the data). "
                             "Values are never altered and no sample is removed on its residual. Full-panel check: "
                             "LIABEQ = LIAB + EQTOT in every row; ASSET - LIABEQ = +-1 in 3.5% of 2015Q2-2021Q1 "
                             "rows (rounding); asset-component sum exact except 10 bank-quarters (8 off by 1-3, "
                             "CERT 23992 off by 520 / 501 in 2019Q4 / 2020Q1, not in the subsample); liability "
                             "sum exact except 2 rows off by 1."),
              "x_desc": "quarter-t balance sheet: " + "; ".join(f"{v} {DESC[v]}" for v in VARS) + f" ({q})",
              "y_desc": f"the same {q} items at quarter t+1 (9 asset components + total assets, 6 liability "
                        "components + total liabilities, total equity)"})
