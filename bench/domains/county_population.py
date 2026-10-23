"""U.S. county population estimates with components of change (Census Bureau, real data).

Sample = (county, year t) inside one estimates vintage.  Year t is the estimates year July 1 (t-1) to
June 30 t; POPESTIMATE<t> is the July 1, t population.
Inputs : POPESTIMATE<t-1> and the eight year-(t-1) components of change.
Outputs: the year-t components and POPESTIMATE<t>.
Equalities (the Census demographic balancing equation, exact in the published integers):
    NATURALCHG = BIRTHS - DEATHS
    NETMIG     = INTERNATIONALMIG + DOMESTICMIG
    NPOPCHG    = NATURALCHG + NETMIG + RESIDUAL
    POPESTIMATE<t> - POPESTIMATE<t-1> - NPOPCHG<t> = 0      (POPESTIMATE<t-1> enters through B)

Vintages: V2020 (2010 Census base, years 2010..2020) and V2025 (2020 Census base, 2020..2025).
The first year of each vintage (2010, 2020) covers only the April 1 base to July 1 period (3 months)
and starts from ESTIMATESBASE rather than a July estimate; it is excluded as a target and as a lag,
so targets are 2012..2020 (V2020) and 2022..2025 (V2025).  A sample never mixes vintages, and every
target year comes from exactly one vintage (V2020's July 2019 - July 2020 year is kept; V2025's
April-July 2020 period is not used).

Raw files: python -P bench/scripts/get_county_population.py
"""
import numpy as np
import pandas as pd

from bench.domain import RAW, Domain

DIR = RAW / "county_population"
BASE_URL = "https://www2.census.gov/programs-surveys/popest/datasets"
# file, URL, first full estimates year, last year, column-name aliases used in that file
VINTAGES = [
    ("co-est2020-alldata.csv", f"{BASE_URL}/2010-2020/counties/totals/co-est2020-alldata.csv", 2011, 2020,
     {"NPOPCHG": "NPOPCHG_", "NATURALCHG": "NATURALINC"}),
    ("co-est2025-alldata.csv", f"{BASE_URL}/2020-2025/counties/totals/co-est2025-alldata.csv", 2021, 2025, {}),
]
COMP = ["BIRTHS", "DEATHS", "NATURALCHG", "INTERNATIONALMIG", "DOMESTICMIG", "NETMIG", "RESIDUAL", "NPOPCHG"]
Y_VARS = COMP + ["POPESTIMATE"]


def _vintage(fname, first, last, alias):
    path = DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run python -P bench/scripts/get_county_population.py")
    df = pd.read_csv(path, encoding="latin-1", dtype={"STATE": str, "COUNTY": str})
    df = df[df["SUMLEV"] == 50]                       # counties only (040 = state totals)
    col = lambda v, y: f"{alias.get(v, v)}{y}"
    X, Y, yr = [], [], []
    # target t needs a full lag year t-1 >= first, so t runs from first+1
    for t in range(first + 1, last + 1):
        X.append(np.column_stack([df[col("POPESTIMATE", t - 1)]] + [df[col(v, t - 1)] for v in COMP]))
        Y.append(np.column_stack([df[col(v, t)] for v in Y_VARS]))
        yr.append(np.full(len(df), t))
    return np.vstack(X).astype(float), np.vstack(Y).astype(float), np.concatenate(yr), len(df)


def load():
    parts = [_vintage(f, a, b, al) for f, _, a, b, al in VINTAGES]
    X = np.vstack([p[0] for p in parts])
    Y = np.vstack([p[1] for p in parts])
    year = np.concatenate([p[2] for p in parts])
    iy = {v: i for i, v in enumerate(Y_VARS)}
    A = np.zeros((4, len(Y_VARS)))
    A[0, [iy["NATURALCHG"], iy["BIRTHS"], iy["DEATHS"]]] = [1, -1, 1]
    A[1, [iy["NETMIG"], iy["INTERNATIONALMIG"], iy["DOMESTICMIG"]]] = [1, -1, -1]
    A[2, [iy["NPOPCHG"], iy["NATURALCHG"], iy["NETMIG"], iy["RESIDUAL"]]] = [1, -1, -1, -1]
    A[3, [iy["POPESTIMATE"], iy["NPOPCHG"]]] = [1, -1]
    B = np.zeros((4, X.shape[1]))
    B[3, 0] = -1.0                                    # - POPESTIMATE<t-1>
    y_lo = np.full(len(Y_VARS), -np.inf)
    y_lo[[iy["BIRTHS"], iy["DEATHS"], iy["POPESTIMATE"]]] = 0.0
    counts = ", ".join(f"{f}: {n} counties x {b - a} years" for (f, _, a, b, _), (*_, n) in zip(VINTAGES, parts))
    return Domain(
        "county_population", X, Y, A, B, np.zeros(4), split=year, split_kind="time", y_lo=y_lo,
        x_names=["POPESTIMATE_tm1"] + [f"{v}_tm1" for v in COMP],
        y_names=[f"{v}_t" for v in Y_VARS],
        row_names=["natural_change", "net_migration", "population_change", "population_balance"],
        meta={"source": ("U.S. Census Bureau, Population Estimates Program, county population totals and "
                         "components of change, Vintage 2020 (" + VINTAGES[0][1] + ") and Vintage 2025 ("
                         + VINTAGES[1][1] + ")"),
              "license": ("U.S. Census Bureau data are works of the U.S. federal government and in the public "
                          "domain in the U.S. (17 U.S.C. 105); no account or terms acceptance needed. Cite the "
                          "Census Bureau as the source."),
              "tier": 2,
              "units": "persons (integer counts); POPESTIMATE is the July 1 resident population",
              "generation": ("County rows only (SUMLEV 50). One sample per (county, target year t) within a "
                             "vintage; X = July 1 (t-1) population and the year-(t-1) components, Y = year-t "
                             "components and July 1 t population. The partial first year of each vintage "
                             "(April 1 - July 1 2010 / 2020, starting from ESTIMATESBASE) is excluded both as "
                             "target and as lag, so targets are 2012-2020 (V2020, 2010 Census base) and "
                             "2022-2025 (V2025, 2020 Census base); no sample mixes vintages. V2020 names "
                             "NPOPCHG_yyyy and NATURALINCyyyy are mapped to NPOPCHG and NATURALCHG. "
                             "Connecticut appears as its 8 former counties in V2020 and as its 9 planning "
                             f"regions in V2025. {counts}. All four identities hold with zero residual in "
                             "every county-year of both files (checked, including the excluded years)."),
              "x_desc": "July 1 (t-1) population and year-(t-1) births, deaths, natural change, international, "
                        "domestic and net migration, residual, total population change (9)",
              "y_desc": "year-t births, deaths, natural change, international, domestic and net migration, "
                        "residual, total population change, and July 1 t population (9)"})
