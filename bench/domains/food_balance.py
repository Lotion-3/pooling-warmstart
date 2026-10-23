"""FAOSTAT Food Balance Sheets, 2010- series (FAO, real data).

Sample = (country, item, year t), t = 2011..2023.
Inputs : the 13 year-(t-1) quantity elements of that country-item and the year-(t-1) total
         population (1000 persons).  Nothing from year t.
Outputs: the 13 year-t quantity elements (1000 t), FAO element names/codes:
         5511 Production, 5611 Import quantity, 5911 Export quantity, 5072 Stock Variation,
         5301 Domestic supply quantity, 5142 Food, 5521 Feed, 5527 Seed, 5123 Losses,
         5131 Processing, 5154 Other uses (non-food), 5171 Tourist consumption, 5170 Residuals.
Equalities (the FBS supply/utilization identity):
    Production + Import - Export - Stock Variation - Domestic supply = 0
    Domestic supply - (Food + Feed + Seed + Losses + Processing + Other uses
                       + Tourist consumption + Residuals) = 0
Stock Variation is published as the change in stocks (positive = addition to stocks), so it
enters supply with a minus sign; checked on the data (with +SV the first identity fails for
~37% of rows by up to 98,534 kt, with -SV every row closes within 2 kt).

The bulk file publishes quantities as integers of 1000 t, so both identities close only up to
rounding (|residual| <= 2 kt, one row at 54 kt); the gate decides whether that is acceptable.
Gate on the full data: fails.  ~30% of samples have a 1-2 kt rounding residual, which is ~6e-4 of
the mean term magnitude of each row (1664 / 1447 kt); max 3.7e-2 (the 54 kt record).

Raw file: python -P bench/scripts/get_food_balance.py
"""
import zipfile

import numpy as np
import pandas as pd

from bench.domain import CACHE, RAW, Domain

DIR = RAW / "food_balance"
URL = "https://bulks-faostat.fao.org/production/FoodBalanceSheets_E_All_Data_(Normalized).zip"
ZIP = DIR / "FoodBalanceSheets_E_All_Data_(Normalized).zip"
CSV = "FoodBalanceSheets_E_All_Data_(Normalized).csv"
CACHE_FILE = CACHE / "food_balance.npz"

# (element code, short name); order defines the columns of Y and of the lagged part of X
ELEMENTS = [(5511, "production"), (5611, "import"), (5911, "export"), (5072, "stock_variation"),
            (5301, "domestic_supply"), (5142, "food"), (5521, "feed"), (5527, "seed"),
            (5123, "losses"), (5131, "processing"), (5154, "other_uses_nonfood"),
            (5171, "tourist_consumption"), (5170, "residuals")]
POP_ITEM, POP_ELEMENT = 2501, 511
# individual FBS items are 2511..2899; 2501 is Population, codes >= 2900 are Grand Total,
# Vegetal/Animal Products and item groups (2905 Cereals, 2948 Milk, ...)
ITEM_LO, ITEM_HI = 2511, 2899
# area codes >= 5000 are regional/economic aggregates; 351 "China" = mainland + Hong Kong +
# Macao + Taiwan, which are also listed individually (41, 96, 128, 214)
AGGREGATE_AREAS = {351}


def _build():
    if not ZIP.exists():
        raise FileNotFoundError(f"{ZIP} missing: run python -P bench/scripts/get_food_balance.py")
    codes = [c for c, _ in ELEMENTS]
    with zipfile.ZipFile(ZIP) as z, z.open(CSV) as f:
        df = pd.read_csv(f, encoding="latin-1",
                         usecols=["Area Code", "Item Code", "Element Code", "Year", "Unit", "Value"],
                         dtype={"Area Code": np.int32, "Item Code": np.int32, "Element Code": np.int32,
                                "Year": np.int32, "Value": np.float64})
    df = df[(df["Area Code"] < 5000) & ~df["Area Code"].isin(AGGREGATE_AREAS)]
    pop = df[(df["Item Code"] == POP_ITEM) & (df["Element Code"] == POP_ELEMENT)]
    assert (pop["Unit"] == "1000 No").all()
    pop = pop.set_index(["Area Code", "Year"])["Value"]
    q = df[df["Item Code"].between(ITEM_LO, ITEM_HI) & df["Element Code"].isin(codes)]
    assert (q["Unit"] == "1000 t").all()
    assert not q.duplicated(["Area Code", "Item Code", "Element Code", "Year"]).any()
    w = q.pivot(index=["Area Code", "Item Code", "Year"], columns="Element Code", values="Value")
    w = w.reindex(columns=codes)
    # A country-item-year exists in the FBS when it has any published element; Domestic supply is
    # published for every such record.  Elements not published for an existing record are nil /
    # not applicable for that commodity (e.g. Seed for meat) and count as 0: the published
    # balances close (to rounding) only with that reading.
    assert w[5301].notna().all()
    w = w.fillna(0.0)
    lag = w.copy()
    lag.index = lag.index.set_levels(lag.index.levels[2] + 1, level=2)   # (a, i, t-1) -> key t
    both = w.index.intersection(lag.index)                               # needs a t-1 record
    cur, prev = w.loc[both], lag.loc[both]
    area = both.get_level_values(0).to_numpy()
    item = both.get_level_values(1).to_numpy()
    year = both.get_level_values(2).to_numpy()
    p_prev = pop.reindex(pd.MultiIndex.from_arrays([area, year - 1])).to_numpy()
    ok = ~np.isnan(p_prev)                                              # drop on missing population
    X = np.column_stack([prev.to_numpy(), p_prev])[ok]
    Y = cur.to_numpy()[ok]
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_FILE, X=X, Y=Y, area=area[ok], item=item[ok], year=year[ok],
                        n_records=len(w), n_no_lag=len(w) - len(both), n_no_pop=int((~ok).sum()),
                        zip_size=ZIP.stat().st_size)
    return np.load(CACHE_FILE)


def load():
    d = None
    if CACHE_FILE.exists() and ZIP.exists():
        d = np.load(CACHE_FILE)
        if int(d["zip_size"]) != ZIP.stat().st_size:
            d = None
    if d is None:
        d = _build()
    X, Y, year = d["X"], d["Y"], d["year"]
    names = [s for _, s in ELEMENTS]
    iy = {s: i for i, s in enumerate(names)}
    q = len(names)
    A = np.zeros((2, q))
    A[0, [iy["production"], iy["import"], iy["export"], iy["stock_variation"], iy["domestic_supply"]]] = \
        [1, 1, -1, -1, -1]
    A[1, iy["domestic_supply"]] = 1
    A[1, [iy[s] for s in ("food", "feed", "seed", "losses", "processing", "other_uses_nonfood",
                          "tourist_consumption", "residuals")]] = -1
    y_lo = np.full(q, -np.inf)
    y_lo[[iy[s] for s in ("production", "import", "export", "food", "feed", "seed", "losses",
                          "processing", "other_uses_nonfood", "tourist_consumption")]] = 0.0
    n_areas, n_items = len(np.unique(d["area"])), len(np.unique(d["item"]))
    return Domain(
        "food_balance", X, Y, A, None, np.zeros(2), split=year, split_kind="time", y_lo=y_lo,
        x_names=[f"{s}_tm1" for s in names] + ["population_tm1"],
        y_names=[f"{s}_t" for s in names],
        row_names=["supply_balance", "utilization_balance"],
        meta={"source": ("FAOSTAT Food Balances (2010-), dataset FBS, bulk file " + URL + " (FAOSTAT "
                         "update 2025-10-28, downloaded 2026-10-23, 54,840,802 bytes; listed in "
                         "https://bulks-faostat.fao.org/production/datasets_E.json)"),
              "license": ("FAO statistical databases are released under CC BY 4.0 per FAO's open data / "
                          "database terms of use (as understood at download; the bulk file itself carries "
                          "no license text). Cite FAO. FAOSTAT Food Balances. Rome."),
              "tier": 2,
              "units": "quantities in 1000 t (published as integers); population in 1000 persons",
              "generation": (
                  "Individual countries only (area codes < 5000, and the aggregate 351 'China' removed "
                  "in favour of its four listed parts); individual items only (codes 2511-2899; 2501 "
                  "Population and every code >= 2900, i.e. Grand Total, Vegetal/Animal Products and the "
                  "item groups, removed). One sample per (country, item, year t) whose t and t-1 records "
                  "both exist and whose t-1 population is published; elements not published within an "
                  "existing record count as 0 (nil / not applicable), Domestic supply is present in "
                  f"every record. {int(d['n_records'])} country-item-year records, "
                  f"{int(d['n_no_lag'])} without a t-1 record, {int(d['n_no_pop'])} without t-1 "
                  f"population; {len(year)} samples over {n_areas} areas x {n_items} items, targets "
                  f"{year.min()}-{year.max()}. Stock Variation enters supply with a minus sign "
                  "(addition to stocks), verified on the data. Because values are rounded to integer "
                  "kt, the identities close only to within 2 kt in all but one record (Nepal, oranges, "
                  "2014: Processing 54 kt left out of Domestic supply). Labels are the published values."),
              "x_desc": "year-(t-1) Production, Import, Export, Stock Variation, Domestic supply, Food, Feed, "
                        "Seed, Losses, Processing, Other uses, Tourist consumption, Residuals of the same "
                        "country-item (1000 t) and year-(t-1) total population (1000 persons) (14)",
              "y_desc": "year-t Production, Import, Export, Stock Variation, Domestic supply, Food, Feed, "
                        "Seed, Losses, Processing, Other uses, Tourist consumption, Residuals (1000 t) (13)"})
