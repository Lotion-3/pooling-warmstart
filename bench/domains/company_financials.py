"""U.S. public-company balance sheets from SEC EDGAR Financial Statement Data Sets (real data).

Sample = (company CIK, fiscal year t): two consecutive annual reports (form 10-K, fp FY) of the same
filer whose balance-sheet dates are 11-13 months apart.  t = calendar year of the year-t balance-sheet
date (num.txt ddate / sub.txt period, both rounded to month end by SEC); t = 2011..2023.
Inputs : year-t consolidated balance-sheet values (see X_TAGS).
Outputs: year-(t+1) Assets, Liabilities, StockholdersEquity, LiabilitiesAndStockholdersEquity.
Equality: Assets - LiabilitiesAndStockholdersEquity = 0  (balance-sheet identity, year t+1).

Values come from num.txt with qtrs = 0 (instant), uom = USD, no coreg and no segments (the
consolidated entity), us-gaap namespace, ddate = the filing's own balance-sheet date.  A filing must
tag all four output concepts (presence filter).  The second identity
    LiabilitiesAndStockholdersEquity = Liabilities + equity incl. NCI + temporary equity + commitments
is measured on the data (counts in meta) but not imposed, because it does not close for a sizeable
share of filers (other mezzanine / custom lines); see meta["generation"].
Even the first identity is not exact in every filing: a few filers tagged inconsistent totals
(typos, wrong figure on the total line), so the gate is expected to fail on these labels as filed.

Raw files: python -P bench/scripts/get_company_financials.py   (Q1 + Q2 zips of 2012 - 2025)
"""
import time
import zipfile

import numpy as np
import pandas as pd

from bench.domain import CACHE, RAW, Domain

DIR = RAW / "company_financials"
BASE = "https://www.sec.gov/files/dera/data/financial-statement-data-sets"
QUARTERS = [f"{y}q{q}" for y in range(2012, 2026) for q in (1, 2)]
CACHE_NPZ = CACHE / "company_financials.npz"
FILINGS = CACHE / "company_financials_filings.parquet"   # per-filing wide table (intermediate)
YEARS = (2011, 2023)    # complete target years: t+1 balance sheets dated 2012..2024
N_CAP = 50_000
SEED = 12345

Y_TAGS = ["Assets", "Liabilities", "StockholdersEquity", "LiabilitiesAndStockholdersEquity"]
# components of the second identity (measured, not imposed) and further year-t inputs
EQ_NCI = "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"
TEMP_TAGS = ["TemporaryEquityCarryingAmountIncludingPortionAttributableToNoncontrollingInterest",
             "TemporaryEquityCarryingAmountAttributableToParent", "TemporaryEquityCarryingAmount",
             "RedeemableNoncontrollingInterestEquityCarryingAmount"]
OTHER_TAGS = ["MinorityInterest", EQ_NCI, "CommitmentsAndContingencies", "AssetsCurrent",
              "LiabilitiesCurrent", "CashAndCashEquivalentsAtCarryingValue"]
ALL_TAGS = Y_TAGS + OTHER_TAGS + TEMP_TAGS
# inputs: the year-t value of every extracted tag; untagged optional items enter as 0
X_TAGS = Y_TAGS + ["MinorityInterest", EQ_NCI, "TemporaryEquity", "CommitmentsAndContingencies",
                   "AssetsCurrent", "LiabilitiesCurrent", "CashAndCashEquivalentsAtCarryingValue"]
NUM_COLS = ["adsh", "tag", "version", "ddate", "qtrs", "uom", "segments", "coreg", "value"]


def _read_zip(path):
    """Consolidated year-end USD balance-sheet values of every 10-K FY filing in one quarterly zip."""
    z = zipfile.ZipFile(path)
    with z.open("sub.txt") as fh:
        sub = pd.read_csv(fh, sep="\t", dtype=str, quoting=3, encoding="utf-8", encoding_errors="replace",
                          usecols=["adsh", "cik", "form", "period", "fy", "fp", "accepted"])
    sub = sub[(sub.form == "10-K") & (sub.fp == "FY")].drop(columns=["form", "fp"])
    per = sub.set_index("adsh")["period"]
    parts = []
    with z.open("num.txt") as fh:
        for ch in pd.read_csv(fh, sep="\t", dtype=str, quoting=3, chunksize=250_000, usecols=NUM_COLS,
                              encoding="utf-8", encoding_errors="replace"):
            ch = ch[ch.tag.isin(ALL_TAGS) & (ch.qtrs == "0") & (ch.uom == "USD") & ch.segments.isna()
                    & ch.coreg.isna() & ch.version.str.startswith("us-gaap/") & ch.adsh.isin(per.index)]
            ch = ch[ch.ddate.values == per.reindex(ch.adsh).values]
            parts.append(ch[["adsh", "tag", "value"]])
    num = pd.concat(parts, ignore_index=True)
    num["value"] = num["value"].astype(float)
    # one value per adsh/tag (key also has version; keep the last row if a filing repeats a tag)
    dup = num.duplicated(["adsh", "tag"], keep=False)
    n_conflict = int(num[dup].groupby(["adsh", "tag"])["value"].nunique().gt(1).sum())
    num = num.drop_duplicates(["adsh", "tag"], keep="last")
    w = num.pivot(index="adsh", columns="tag", values="value").reindex(columns=ALL_TAGS)
    w = sub.set_index("adsh").join(w, how="left")
    w["quarter"] = path.stem
    return w.reset_index(), n_conflict


def _filings(log=print):
    if FILINGS.exists():
        return pd.read_parquet(FILINGS)
    frames, conflicts = [], 0
    for qn in QUARTERS:
        path = DIR / f"{qn}.zip"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run python -P bench/scripts/get_company_financials.py")
        t0 = time.time()
        w, c = _read_zip(path)
        frames.append(w)
        conflicts += c
        log(f"[company_financials] {qn}: {len(w)} 10-K filings ({time.time() - t0:.0f}s)")
    df = pd.concat(frames, ignore_index=True)
    df.attrs["conflicting_duplicates"] = conflicts
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FILINGS, index=False)
    return df


def _build(log=print):
    f = _filings(log)
    stats = {"filings_10k_fy": len(f), "conflicting_duplicate_values": f.attrs.get("conflicting_duplicates")}
    f["period"] = pd.to_datetime(f["period"], format="%Y%m%d")
    f["accepted"] = pd.to_datetime(f["accepted"])
    # one annual report per filer and balance-sheet date: the latest accepted original 10-K
    f = f.sort_values(["cik", "period", "accepted"])
    stats["duplicate_cik_period_dropped"] = int(f.duplicated(["cik", "period"], keep="last").sum())
    f = f.drop_duplicates(["cik", "period"], keep="last")

    # presence filter: the four output concepts must be tagged (no condition on their values' residual)
    has = f[Y_TAGS].notna().all(1)
    stats["unique_cik_period"] = len(f)
    stats["with_all_four_output_tags"] = int(has.sum())
    for t in Y_TAGS:
        stats[f"missing_{t}"] = int(f[t].isna().sum())

    # second identity, measured on every filing with the four tags (diagnostic only)
    g = f[has].copy()
    z = g.fillna({t: 0.0 for t in ALL_TAGS})
    te = np.where(g[TEMP_TAGS[0]].notna(), z[TEMP_TAGS[0]], z[TEMP_TAGS[1]] + z[TEMP_TAGS[2]] + z[TEMP_TAGS[3]])
    eq = np.where(g[EQ_NCI].notna(), z[EQ_NCI], z.StockholdersEquity + z.MinorityInterest)
    r2 = z.LiabilitiesAndStockholdersEquity - z.Liabilities - eq - te - z.CommitmentsAndContingencies
    r1 = g.Assets - g.LiabilitiesAndStockholdersEquity
    rel = lambda r: np.abs(r) / np.maximum(np.abs(g.Assets), 1.0)
    stats["identity1_filings_exact"] = int((r1 == 0).sum())
    stats["identity1_filings_rel_gt_1e-6"] = int((rel(r1) > 1e-6).sum())
    stats["identity2_filings_exact"] = int((r2 == 0).sum())
    stats["identity2_filings_rel_gt_1e-6"] = int((rel(r2) > 1e-6).sum())

    # year-t inputs: untagged optional items enter as 0; temporary equity = the incl.-NCI total when
    # tagged, else the sum of its parts
    z["TemporaryEquity"] = te
    g = z
    g["year"] = g["period"].dt.year
    g = g.sort_values(["cik", "period"]).reset_index(drop=True)
    nxt = g.groupby("cik").shift(-1)
    months = (nxt["period"].dt.year - g["period"].dt.year) * 12 + nxt["period"].dt.month - g["period"].dt.month
    ok = months.between(11, 13)
    stats["pairs_consecutive"] = int(ok.sum())
    # t = 2010 (a late filer in 2012q1) and t = 2024 (only Jan-Jun 2025 year-ends are in the
    # 2025q1/q2 zips) are incomplete years
    in_years = g["year"].between(*YEARS)
    stats["pairs_outside_years_dropped"] = int((ok & ~in_years).sum())
    ok &= in_years
    gaps = months.dropna().astype(int)
    stats["next_filing_gap_months"] = {"11-13": int(gaps.between(11, 13).sum()), "<11": int((gaps < 11).sum()),
                                       ">13": int((gaps > 13).sum())}
    X = g.loc[ok, X_TAGS].to_numpy(np.float64)
    Y = nxt.loc[ok, Y_TAGS].to_numpy(np.float64)
    year = g.loc[ok, "year"].to_numpy(np.int64)
    cik = g.loc[ok, "cik"].astype(np.int64).to_numpy()
    stats["n_before_cap"] = len(Y)
    if len(Y) > N_CAP:
        keep = np.sort(np.random.default_rng(SEED).choice(len(Y), N_CAP, replace=False))
        X, Y, year, cik = X[keep], Y[keep], year[keep], cik[keep]
    import json
    np.savez_compressed(CACHE_NPZ, X=X, Y=Y, year=year, cik=cik, stats=json.dumps(stats))
    return X, Y, year, cik, stats


def load():
    import json
    if CACHE_NPZ.exists():
        d = np.load(CACHE_NPZ)
        X, Y, year, cik, stats = d["X"], d["Y"], d["year"], d["cik"], json.loads(str(d["stats"]))
    else:
        X, Y, year, cik, stats = _build()
    iy = {v: i for i, v in enumerate(Y_TAGS)}
    A = np.zeros((1, len(Y_TAGS)))
    A[0, [iy["Assets"], iy["LiabilitiesAndStockholdersEquity"]]] = [1, -1]
    y_lo = np.full(len(Y_TAGS), -np.inf)
    y_lo[[iy["Assets"], iy["Liabilities"]]] = 0.0
    yrs = f"{year.min()}-{year.max()}"
    return Domain(
        "company_financials", X, Y, A, None, np.zeros(1), split=year, split_kind="time", y_lo=y_lo,
        x_names=[f"{t}_t" for t in X_TAGS], y_names=[f"{t}_tp1" for t in Y_TAGS],
        row_names=["assets_eq_liabilities_and_equity"],
        meta={"source": ("SEC EDGAR Financial Statement Data Sets (DERA), quarterly zips "
                         f"{BASE}/YYYYqN.zip, quarters {QUARTERS[0]}..{QUARTERS[-1]} (Q1 and Q2 of each "
                         "year only, 28 zips, ~2.7 GB), index https://www.sec.gov/dera/data/"
                         "financial-statement-data-sets; downloaded 2026-10-23 (see download_log.txt)"),
              "license": ("Data extracted by the SEC from filers' XBRL exhibits; SEC EDGAR content is "
                          "U.S. federal government information, generally public domain in the U.S. "
                          "(17 U.S.C. 105) with no account needed; underlying statements are the filers' "
                          "own disclosures. Access is subject to the SEC fair-access policy (declared "
                          "User-Agent, < 10 requests/s). SEC states the data sets are provided as-is, "
                          "without warranty of accuracy."),
              "tier": 2,
              "units": "USD (original units, no scaling or log transform)",
              "generation": (
                  "sub.txt: form 10-K with fp FY. num.txt: qtrs 0, uom USD, coreg and segments empty, "
                  "us-gaap version, ddate equal to the filing's period (balance-sheet date). Per filer "
                  "and balance-sheet date the latest accepted 10-K is kept. A filing is used only if it "
                  "tags Assets, Liabilities, StockholdersEquity and LiabilitiesAndStockholdersEquity "
                  "(presence filter only, never a residual filter). Sample = filings t and t+1 of the "
                  "same CIK with balance-sheet dates 11-13 months apart; X = year-t values (untagged "
                  "MinorityInterest / equity incl. NCI / temporary equity / commitments / current items "
                  "/ cash enter as 0), Y = year-(t+1) outputs. Split key t = calendar year of the year-t "
                  f"balance-sheet date ({yrs}). Imposed identity: Assets = LiabilitiesAndStockholdersEquity. "
                  "The second identity LSE = Liabilities + (equity incl. NCI, or StockholdersEquity + "
                  "MinorityInterest) + temporary equity + CommitmentsAndContingencies was measured but "
                  "not imposed (fails for a sizeable share of filings). Labels are the values as filed; Assets "
                  "and LiabilitiesAndStockholdersEquity differ in a small number of filings because of filer "
                  "tagging errors (typed digits, a total tagged with the Liabilities figure); EDGAR does not "
                  "reject XBRL calculation inconsistencies. Only Q1/Q2 zips are used, so firms whose 10-K is "
                  "filed in Jul-Dec (mostly Apr-Sep year-ends) are largely absent. Samples "
                  f"capped at {N_CAP} samples (seed {SEED}). Counts: {json.dumps(stats)}"),
              "x_desc": "year-t consolidated balance-sheet values in USD (" + ", ".join(X_TAGS) + ")",
              "y_desc": "year-(t+1) Assets, Liabilities, StockholdersEquity, LiabilitiesAndStockholdersEquity (USD)",
              "stats": stats})
