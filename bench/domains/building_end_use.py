"""Residential annual energy use by end use and fuel (NREL ResStock 2025 release 1, AMY2018; real data).

Sample = one ResStock baseline dwelling-unit model (upgrade 0) in one of 20 states, 2,500 per state.
Inputs : building characteristics (in.* columns): numeric ones parsed from their labels plus one-hot
         encodings of a few key categoricals (fuels, building / HVAC type, climate zone, ...).
Outputs: annual site energy (kWh) of every end use of electricity, natural gas, fuel oil and propane,
         PV generation, the four fuel totals, net electricity, total and net site energy.
Equalities (ResStock reporting identities):
    sum(electricity end uses, PV excluded) - electricity total = 0
    sum(natural gas end uses)              - natural gas total = 0
    sum(fuel oil end uses)                 - fuel oil total    = 0
    sum(propane end uses)                  - propane total     = 0
    electricity + gas + fuel oil + propane totals - site total = 0
    electricity total + PV - electricity net = 0              (PV is reported as a negative number)
    site total        + PV - site net        = 0
The published values are integer kBtu converted to kWh, each rounded on its own, so the end-use sums
miss the totals by a few kBtu (rounding; the harness projects the labels); the last three identities
hold exactly.  The propane row is the tight one: most buildings use little or no propane, so a 1 kBtu
rounding miss is 9.7e-5 of that row's typical term size in this sample, just under the harness's 1e-4
projection limit.  A different draw containing one of the rare 2 kBtu propane misses (3 of 288k
buildings in the first 12 states checked) would exceed it, so changing STATES / PER_STATE / SEED
needs the gate re-checked.  Fuel oil is material only in the Northeast, hence seven Northeast states
(a first 12-state draw sampled by state size, with only MA, NY and PA from the Northeast, gave 1.1e-4 on
the fuel-oil row and 4.0e-4 on the propane row).

Raw files: python -P bench/scripts/get_building_end_use.py
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from bench.domain import CACHE, RAW, Domain

DIR = RAW / "building_end_use"
CACHE_FILE = CACHE / "building_end_use.npz"
CACHE_KEY = "v5"
RELEASE = "2025/resstock_amy2018_release_1"
BASE = ("https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/"
        f"end-use-load-profiles-for-us-building-stock/{RELEASE}")
STATES = ["AZ", "CA", "CO", "CT", "FL", "GA", "IL", "MA", "ME", "MI", "MN", "NC", "NH", "NJ", "NY", "PA",
          "TX", "VA", "WA", "WI"]
PER_STATE = 2500
SEED = 2025
FUELS = ["electricity", "natural_gas", "fuel_oil", "propane"]
SUFFIX = ".energy_consumption..kwh"

# one-hot encoded categoricals (column -> short prefix)
ONEHOT = {
    "in.heating_fuel": "heat_fuel",
    "in.water_heater_fuel": "wh_fuel",
    "in.clothes_dryer": "dryer",
    "in.cooking_range": "range",
    "in.geometry_building_type_recs": "bldg_type",
    "in.hvac_heating_type": "heat_type",
    "in.hvac_cooling_type": "cool_type",
    "in.ashrae_iecc_climate_zone_2004": "climate_zone",
    "in.geometry_foundation_type": "foundation",
    "in.lighting": "lighting",
    "in.misc_hot_tub_spa": "spa",
    "in.misc_pool_heater": "pool_heater",
}
VINTAGE = {"<1940": 1930, "1940s": 1945, "1950s": 1955, "1960s": 1965, "1970s": 1975, "1980s": 1985,
           "1990s": 1995, "2000s": 2005, "2010s": 2015}
USAGE = {"Low": 0.0, "Medium": 1.0, "High": 2.0}


def _num(s, pattern, default=0.0):
    """First capture group of `pattern` in each label as float, `default` where it does not match."""
    return s.astype(str).str.extract(pattern, expand=False).astype(float).fillna(default).to_numpy()


def _flag(s, off=("None",)):
    return (~s.astype(str).isin(off)).to_numpy(float)


def _numeric_features(d):
    """Numeric inputs parsed from ResStock characteristic labels.  Returns (names, columns)."""
    he, ce = d["in.hvac_heating_efficiency"], d["in.hvac_cooling_efficiency"]
    cool = d["in.hvac_cooling_partial_space_conditioning"]
    whe = d["in.water_heater_efficiency"].astype(str)
    win = d["in.windows"].astype(str)
    ev = (d["in.electric_vehicle_ownership"].astype(str) == "Yes").to_numpy(float)
    f = {
        "floor_area_ft2": d["in.sqft..ft2"].to_numpy(float),
        "vintage_year": d["in.vintage"].map(VINTAGE).to_numpy(float),
        "stories": _num(d["in.geometry_stories"], r"^(\d+)"),
        "bedrooms": _num(d["in.bedrooms"], r"^(\d+)"),
        "occupants": _num(d["in.occupants"], r"^(\d+)"),               # "10+" -> 10
        "vacant": (d["in.vacancy_status"].astype(str) == "Vacant").to_numpy(float),
        "heating_setpoint_F": _num(d["in.heating_setpoint"], r"^(\d+)F"),
        "heating_setback_F": _num(d["in.heating_setpoint_offset_magnitude"], r"^(\d+)F"),
        "cooling_setpoint_F": _num(d["in.cooling_setpoint"], r"^(\d+)F"),
        "cooling_setup_F": _num(d["in.cooling_setpoint_offset_magnitude"], r"^(\d+)F"),
        "infiltration_ach50": _num(d["in.infiltration"], r"^([\d.]+) ACH50"),
        "wall_R": _num(d["in.insulation_wall"], r"R-(\d+)"),          # Uninsulated -> 0
        "ceiling_R": _num(d["in.insulation_ceiling"], r"R-(\d+)"),    # None (no attic) / Uninsulated -> 0
        "floor_R": _num(d["in.insulation_floor"], r"R-(\d+)"),
        "window_to_wall_pct": _num(d["in.window_areas"], r"^F(\d+)"),
        "window_panes": np.select([win.str.startswith("Single"), win.str.startswith("Double"),
                                   win.str.startswith("Triple")], [1.0, 2.0, 3.0], 0.0),
        "window_low_e": win.str.contains("Low-E").to_numpy(float),
        "cooled_fraction_pct": np.where(cool.astype(str) == "<10% Conditioned", 5.0, _num(cool, r"^(\d+)%")),
        "garage_cars": _num(d["in.geometry_garage"], r"^(\d+) Car"),
        "pv_kwdc": _num(d["in.pv_system_size"], r"^([\d.]+) kWDC"),
        "latitude": d["in.weather_file_latitude"].to_numpy(float),
        "longitude": d["in.weather_file_longitude"].to_numpy(float),
        "heating_eff_pct": _num(he, r"([\d.]+)% (?:AFUE|Efficiency)"),   # furnaces, boilers, resistance
        "heating_hspf": _num(he, r"([\d.]+) HSPF"),                     # heat pumps
        "cooling_seer": _num(ce, r"SEER ([\d.]+)") + _num(he, r"SEER ([\d.]+)") *
                        ce.astype(str).str.contains("Heat Pump").to_numpy(float),
        "cooling_eer": _num(ce, r"EER ([\d.]+)"),                       # room AC
        "shared_hvac": _flag(d["in.hvac_has_shared_system"]),
        "has_ducts": (d["in.hvac_has_ducts"].astype(str) == "Yes").to_numpy(float),
        "wh_tankless": whe.str.contains("Tankless").to_numpy(float),
        "wh_heat_pump": whe.str.contains("Heat Pump").to_numpy(float),
        "wh_premium": whe.str.contains("Premium").to_numpy(float),
        "wh_solar_thermal": whe.str.contains("Solar Thermal").to_numpy(float),
        "plug_load_pct": _num(d["in.plug_loads"], r"^(\d+)%"),
        "hot_water_fixtures_pct": _num(d["in.hot_water_fixtures"], r"^(\d+)%"),
        "usage_level": d["in.usage_level"].map(USAGE).to_numpy(float),
        "refrigerator_ef": _num(d["in.refrigerator"], r"EF ([\d.]+)"),
        "extra_refrigerator": _flag(d["in.misc_extra_refrigerator"]),
        "freezer": _flag(d["in.misc_freezer"]),
        "dishwasher": _flag(d["in.dishwasher"]),
        "clothes_washer": _flag(d["in.clothes_washer"]),
        "ceiling_fan": _flag(d["in.ceiling_fan"]),
        "pool": _flag(d["in.misc_pool"]),
        "well_pump": _flag(d["in.misc_well_pump"]),
        "gas_fireplace": _flag(d["in.misc_gas_fireplace"]),
        "gas_grill": _flag(d["in.misc_gas_grill"]),
        "gas_lighting": _flag(d["in.misc_gas_lighting"]),
        "ev_owned": ev,
        "ev_miles_per_year": _num(d["in.electric_vehicle_miles_traveled"], r"^(\d+)") * ev,
    }
    return list(f), [np.asarray(v, float) for v in f.values()]


def _onehot(d):
    names, cols = [], []
    for c, pre in ONEHOT.items():
        s = d[c].astype(str)
        for v in sorted(s.unique()):
            names.append(f"{pre}={v}")
            cols.append((s == v).to_numpy(float))
    return names, cols


def _read_state(state, rng):
    path = DIR / f"{state}_upgrade0.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run python -P bench/scripts/get_building_end_use.py")
    pf = pq.ParquetFile(path)
    cols = [c for c in pf.schema_arrow.names
            if c.startswith("in.") or c.endswith(SUFFIX) or c in ("bldg_id", "completed_status")]
    d = pf.read(columns=cols).to_pandas()
    d = d[d["completed_status"] == "Success"].sort_values("bldg_id").reset_index(drop=True)
    return d.iloc[np.sort(rng.choice(len(d), PER_STATE, replace=False))]


def _build():
    rng = np.random.default_rng(SEED)
    d = pd.concat([_read_state(s, rng) for s in STATES], ignore_index=True)
    xn1, xc1 = _numeric_features(d)
    xn2, xc2 = _onehot(d)
    X = np.column_stack(xc1 + xc2)
    x_names = xn1 + xn2
    # outputs: per fuel its end uses (all-zero columns dropped), then totals
    y_cols = []
    for fuel in FUELS:
        eu = sorted(c for c in d.columns if c.startswith(f"out.{fuel}.") and c.endswith(SUFFIX)
                    and c.split(".")[2] not in ("total", "net", "pv"))
        y_cols += [c for c in eu if (d[c] != 0).any()]
        if fuel == "electricity":
            y_cols.append(f"out.electricity.pv{SUFFIX}")
        y_cols.append(f"out.{fuel}.total{SUFFIX}")
        if fuel == "electricity":
            y_cols.append(f"out.electricity.net{SUFFIX}")
    y_cols += [f"out.site_energy.total{SUFFIX}", f"out.site_energy.net{SUFFIX}"]
    Y = d[y_cols].to_numpy(float)
    y_names = [c[len("out."):-len(SUFFIX)] for c in y_cols]
    return X, Y, d["in.state"].to_numpy(str), x_names, y_names


def _cached():
    if CACHE_FILE.exists():
        z = np.load(CACHE_FILE, allow_pickle=False)
        if str(z["key"]) == CACHE_KEY:
            return z["X"], z["Y"], z["state"], list(z["x_names"]), list(z["y_names"])
    X, Y, state, xn, yn = _build()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_FILE, key=CACHE_KEY, X=X, Y=Y, state=state, x_names=np.array(xn),
                        y_names=np.array(yn))
    return X, Y, state, xn, yn


def load():
    X, Y, state, x_names, y_names = _cached()
    iy = {v: i for i, v in enumerate(y_names)}
    q = len(y_names)
    rows, row_names = [], []
    for fuel in FUELS:
        a = np.zeros(q)
        for v in y_names:
            if v.startswith(f"{fuel}.") and v.split(".")[1] not in ("total", "net", "pv"):
                a[iy[v]] = 1.0
        a[iy[f"{fuel}.total"]] = -1.0
        rows.append(a)
        row_names.append(f"{fuel}_end_uses")
    a = np.zeros(q)
    a[[iy[f"{f}.total"] for f in FUELS]] = 1.0
    a[iy["site_energy.total"]] = -1.0
    rows.append(a)
    row_names.append("site_total")
    for tot, net, nm in (("electricity.total", "electricity.net", "electricity_net"),
                         ("site_energy.total", "site_energy.net", "site_net")):
        a = np.zeros(q)
        a[[iy[tot], iy["electricity.pv"]]] = 1.0
        a[iy[net]] = -1.0
        rows.append(a)
        row_names.append(nm)
    A = np.array(rows)

    # consumption is nonnegative by definition; PV is reported <= 0; net quantities can be negative
    signed = {"electricity.pv", "electricity.net", "site_energy.net"}
    y_lo = np.array([-np.inf if v in signed else 0.0 for v in y_names])
    y_hi = np.array([0.0 if v == "electricity.pv" else np.inf for v in y_names])

    n_eu = {f: sum(v.startswith(f"{f}.") and v.split(".")[1] not in ("total", "net", "pv") for v in y_names)
            for f in FUELS}
    n_num = sum("=" not in v for v in x_names)
    file_urls = f"{BASE}/metadata_and_annual_results/by_state/full/parquet/state=XX/XX_upgrade0.parquet"
    return Domain(
        "building_end_use", X, Y, A, None, np.zeros(len(A)), split=state, split_kind="group",
        y_lo=y_lo, y_hi=y_hi, x_names=x_names, y_names=y_names, row_names=row_names,
        meta={"source": ("NREL ResStock 2025 release 1 (resstock_amy2018_release_1, AMY2018 weather), "
                         "End-Use Load Profiles for the U.S. Building Stock, OEDI data lake; per-state baseline "
                         f"metadata_and_annual_results files: {file_urls} (XX = state code). Dataset page: "
                         "https://data.openei.org/submissions/4520"),
              "license": ("Creative Commons Attribution 4.0 (CC BY 4.0), as stated on the OEDI dataset page "
                          "(https://data.openei.org/submissions/4520); anonymous access, cite NREL / the "
                          "End-Use Load Profiles project."),
              "tier": 2,
              "units": ("outputs: annual site energy in kWh (published as integer kBtu x 0.29307107); inputs: "
                        "ft2, year, degF, ACH50, R-value (h ft2 F/Btu), %, kWdc, degrees lat/lon, AFUE %, "
                        "HSPF/SEER/EER, 0/1 flags and one-hot indicators"),
              "generation": (f"States {', '.join(STATES)}; baseline (upgrade 0) models with completed_status "
                             f"Success; {PER_STATE} buildings per state drawn without replacement "
                             f"(numpy default_rng({SEED}), states in the order listed, rows sorted by bldg_id), "
                             f"n = {len(X)}; sampling is equal per state, not by ResStock weight. Numeric inputs "
                             "are parsed from characteristic labels (vintage decade -> midpoint year, setpoints, "
                             "R-values, window-to-wall %, AFUE/HSPF/SEER/EER, usage %, flags for appliances "
                             "present); categoricals are one-hot over the values present in the sample. No "
                             "out.* column is used as input. End-use columns that are zero in every sampled "
                             "building are dropped (fuel oil / gas / propane heat-pump backup). End uses per "
                             f"fuel: {n_eu}. Values are independently rounded integer kBtu, so the four "
                             "end-use identities hold to a few kBtu (rounding; the harness projects labels) and "
                             "the site-total and net identities hold exactly (wood and other fuels are not in the "
                             "site total in this release). Max relative residual per end-use row on this sample: "
                             "electricity 5.5e-5, gas 2.6e-5, fuel oil 4.4e-5, propane 9.7e-5 (close to the 1e-4 "
                             "projection limit; re-check the gate if the sample changes). Group split by state."),
              "x_desc": (f"{len(x_names)} building characteristics: {n_num} numeric "
                         "(floor area, vintage, stories, bedrooms, occupants, setpoints, envelope, HVAC and "
                         "water-heater efficiency, appliance flags, location) and one-hot heating / water-heater "
                         "fuel, dryer, range, building type, heating / cooling type, climate zone, foundation, "
                         "lighting, spa and pool-heater fuel"),
              "y_desc": (f"{q} annual energy outputs (kWh): end uses of electricity, natural gas, fuel oil and "
                         "propane, PV generation (<= 0), the four fuel totals, net electricity, total and net "
                         "site energy")})
