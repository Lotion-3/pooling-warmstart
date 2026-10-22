"""Build data/clean/ from data/raw/ (raw files are only read, never modified).

Outputs:
  prices_daily.csv        one row per date, one column per EIA spot series (units in names)
  prices_quality.csv      per-series coverage / gap stats
  crudes.csv              Equinor grades: name, api, sulfur_pct, type, region, source (+ extras)
  pooling_instances.csv   MINLPLib pooling instances downloaded, with known objective values
"""
import ast
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
CLEAN.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- EIA prices
EIA_SERIES = {
    "RWTC": "wti_cushing_usd_per_bbl",
    "RBRTE": "brent_europe_usd_per_bbl",
    "EER_EPMRU_PF4_Y35NY_DPG": "gasoline_conv_reg_nyh_usd_per_gal",
    "EER_EPMRU_PF4_RGC_DPG": "gasoline_conv_reg_usgc_usd_per_gal",
    "EER_EPMRR_PF4_Y05LA_DPG": "gasoline_rbob_reg_la_usd_per_gal",
    "EER_EPD2F_PF4_Y35NY_DPG": "heating_oil_no2_nyh_usd_per_gal",
    "EER_EPD2DXL0_PF4_Y35NY_DPG": "ulsd_nyh_usd_per_gal",
    "EER_EPD2DXL0_PF4_RGC_DPG": "ulsd_usgc_usd_per_gal",
    "EER_EPD2DC_PF4_Y05LA_DPG": "ulsd_carb_la_usd_per_gal",
    "EER_EPJK_PF4_RGC_DPG": "jet_kero_usgc_usd_per_gal",
    "EER_EPLLPA_PF4_Y44MB_DPG": "propane_mont_belvieu_usd_per_gal",
}


def load_eia(path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    frames = []
    for sheet in [s for s in xl.sheet_names if s.startswith("Data")]:
        raw = pd.read_excel(xl, sheet, header=None)
        keys = raw.iloc[1, 1:].tolist()  # row 1 = Sourcekey, row 2 = long names, data from row 3
        body = raw.iloc[3:].copy()
        body.columns = ["date"] + keys
        body["date"] = pd.to_datetime(body["date"])
        frames.append(body.set_index("date").apply(pd.to_numeric, errors="coerce"))
    df = pd.concat(frames, axis=1).sort_index()
    unknown = set(df.columns) - set(EIA_SERIES)
    assert not unknown, f"unmapped EIA series: {unknown}"
    return df.rename(columns=EIA_SERIES)[list(EIA_SERIES.values())]


def price_quality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        first, last = s.first_valid_index(), s.last_valid_index()
        span = s.loc[first:last]
        obs = span.dropna()
        bdays = pd.bdate_range(first, last)
        gaps = obs.index.to_series().diff().dt.days
        big = gaps[gaps > 5]
        rows.append({
            "series": c,
            "first_date": first.date(),
            "last_date": last.date(),
            "n_obs": len(obs),
            "weekdays_in_span": len(bdays),
            "weekdays_without_price": len(bdays) - len(obs),
            "pct_weekdays_missing": round(100 * (1 - len(obs) / len(bdays)), 2),
            "max_gap_calendar_days": int(gaps.max()),
            "n_gaps_over_5_days": len(big),
            "largest_gap_ends": big.idxmax().date() if len(big) else None,
            "n_nonpositive": int((obs <= 0).sum()),
            "min": obs.min(),
            "min_date": obs.idxmin().date(),
            "max": obs.max(),
            "max_date": obs.idxmax().date(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Equinor assays
REGION = {
    "Norway": "North Sea", "UK": "North Sea", "Angola": "West Africa", "Algeria": "North Africa",
    "Azerbaijan": "Caspian", "Brazil": "South America", "Argentina": "South America",
    "US": "North America", "Canada": "North America",
}
PROPS = {  # label in assay sheet -> output column
    "API Gravity": "api",
    "Total Sulphur (% wt)": "sulfur_pct",
    "Density @ 15": "density_15c_g_cc",
    "Total Acid Number (mgKOH/g)": "tan_mgkoh_g",
    "Viscosity @ 40": "visc_40c_cst",
    "Pour Point": "pour_point_c",
    "Vanadium (ppm)": "vanadium_ppm",
    "Nickel (ppm)": "nickel_ppm",
}


def crude_type(api: float, s: float, desc: str) -> str:
    if "condensate" in desc.lower():
        return "condensate"
    grav = "light" if api > 31.1 else "medium" if api >= 22.3 else "heavy"  # API/NYMEX convention
    sul = "sweet" if s < 0.5 else "sour"
    return f"{grav} {sul}"


def parse_assay(path: Path) -> dict:
    d = pd.read_excel(path, sheet_name="Summary", header=None)
    lab = d.map(lambda v: v.strip() if isinstance(v, str) else v)
    # whole-crude column: the one where Start == IBP and End == FBP
    start_r = lab.index[lab.apply(lambda r: r.astype(str).str.startswith("Start").any(), axis=1)][0]
    end_r = start_r + 1
    wc = [c for c in lab.columns if lab.at[start_r, c] == "IBP" and lab.at[end_r, c] == "FBP"]
    assert len(wc) == 1, f"{path.name}: whole-crude column not found"
    wc = wc[0]
    # property labels live in the same column as "Start (°C)"; other panels reuse some labels
    label_col = [c for c in lab.columns if str(lab.at[start_r, c]).startswith("Start")][0]
    out = {}
    for label, col in PROPS.items():
        hit = [r for r in lab.index[start_r:]
               if isinstance(lab.at[r, label_col], str) and lab.at[r, label_col].startswith(label)]
        out[col] = pd.to_numeric(d.at[hit[0], wc], errors="coerce") if hit else np.nan

    def info(key):
        hit = [(r, c) for r in lab.index for c in lab.columns if lab.at[r, c] == key]
        return lab.at[hit[0][0], hit[0][1] + 1] if hit else None

    out["assay_name"] = info("Name:")
    out["sample_date"] = info("Sample Date:")
    out["n_cut_columns"] = int(sum(isinstance(v, (int, float, str)) and str(v) not in ("nan", "")
                                   for v in d.iloc[start_r, wc + 1:]))
    return out


def load_crudes() -> pd.DataFrame:
    page = (RAW / "equinor" / "equinor_crude_oil_assays.html").read_text(encoding="utf-8")
    table = pd.read_html(io.StringIO(page))[0]
    table.columns = ["issue_date", "name", "description", "country", "api_table"]
    table["name"] = table["name"].str.replace(r"^xlsx", "", regex=True).str.replace("�", "ø").str.strip()
    links = json.loads((RAW / "equinor" / "assay_links.json").read_text(encoding="utf-8"))
    by_grade = {L["grade"].strip().replace("�", "ø"): L for L in links}
    rows = []
    for t in table.itertuples(index=False):
        L = by_grade[t.name]
        fn = re.sub(r"[^A-Za-z0-9._ -]", "_", L["original_filename"])
        a = parse_assay(RAW / "equinor" / "assays" / fn)
        rows.append({
            "name": t.name,
            "api": round(a["api"], 1),
            "sulfur_pct": round(a["sulfur_pct"], 3),
            "type": crude_type(a["api"], a["sulfur_pct"], t.description),
            "region": REGION.get(t.country, t.country),
            "source": "Equinor crude oil assays (equinor.com/energy/crude-oil-assays)",
            "country": t.country,
            "equinor_description": t.description,
            "api_page_table": t.api_table,
            **{k: a[k] for k in ["density_15c_g_cc", "tan_mgkoh_g", "visc_40c_cst",
                                 "pour_point_c", "vanadium_ppm", "nickel_ppm"]},
            "assay_name": a["assay_name"],
            "issue_date": pd.to_datetime(t.issue_date, dayfirst=True).date(),
            "n_cut_columns": a["n_cut_columns"],
            "raw_file": f"data/raw/equinor/assays/{fn}",
            "url": L["url"],
        })
    return pd.DataFrame(rows).sort_values("api").reset_index(drop=True)


# ---------------------------------------------------------------- BP assays
# BP files are a mix of real .xls and .xlsx saved with an .xls name; pandas sniffs the content.
BP_REGION = {
    "North Sea - Norway": ("Norway", "North Sea"), "North Sea - UK": ("UK", "North Sea"),
    "Norway": ("Norway", "North Sea"), "Angola": ("Angola", "West Africa"),
    "Azerbaijan": ("Azerbaijan", "Caspian"), "Iraq": ("Iraq", "Middle East"),
    "Abu Dhabi": ("UAE", "Middle East"), "Oman": ("Oman", "Middle East"),
    "United States of America": ("US", "North America"), "Trinidad and Tobago": ("Trinidad and Tobago", "Caribbean"),
    "Australia": ("Australia", "Asia-Pacific"), "Indonesia": ("Indonesia", "Asia-Pacific"),
    "Mauritania": ("Mauritania", "West Africa"),
}
BP_PROPS = {
    "Density at 15°C (kg/litre)": "density_15c_g_cc",
    "Total Sulphur (% wt)": "sulfur_pct",
    "Acidity (mgKOH/g)": "tan_mgkoh_g",
    "Viscosity at 40°C (cSt)": "visc_40c_cst",
    "Pour Point (°C)": "pour_point_c",
    "Vanadium (ppm wt)": "vanadium_ppm",
    "Nickel (ppm wt)": "nickel_ppm",
}


def api_from_density(rho_15c: float) -> float:
    # SG 60/60°F approximated as density@15°C / water density@15°C (0.99904 g/cc)
    return 141.5 / (rho_15c / 0.99904) - 131.5


def parse_bp(path: Path) -> dict:
    d = pd.read_excel(path, sheet_name="Summary", header=None)
    lab = d.map(lambda v: v.strip() if isinstance(v, str) else v)
    start_r = [r for r in lab.index if str(lab.iat[r, 1]).startswith("Start")][0]
    wc = [c for c in lab.columns if lab.at[start_r, c] == "IBP" and lab.at[start_r + 1, c] == "FBP"]
    assert wc, f"{path.name}: whole-crude column not found"
    # some sheets carry a second, unlabeled copy of the table further right; use the labeled one
    out = {"duplicate_table_in_sheet": len(wc) > 1}
    wc = wc[0]
    for label, col in BP_PROPS.items():
        hit = [r for r in lab.index[start_r:] if lab.iat[r, 1] == label]
        out[col] = pd.to_numeric(d.at[hit[0], wc], errors="coerce") if hit else np.nan

    def info(key):
        hit = [(r, c) for r in lab.index for c in lab.columns[:-1] if lab.at[r, c] == key]
        return lab.at[hit[0][0], hit[0][1] + 1] if hit else None

    out["assay_name"], out["origin"], out["sample_date"] = info("Name:"), info("Origin:"), info("Sample Date:")
    end_c = next((c for c in lab.columns[wc + 1:] if str(lab.iat[start_r, c]) in ("nan", "")), lab.columns[-1] + 1)
    out["n_cut_columns"] = end_c - wc - 1
    return out


def load_bp_crudes() -> pd.DataFrame:
    rows = []
    for f in sorted((RAW / "bp" / "assays").glob("*.xls")):
        a = parse_bp(f)
        country, region = BP_REGION[a["origin"]]
        api = api_from_density(a["density_15c_g_cc"])
        name = re.sub(r"\s*\(.*\)$", "", a["assay_name"])  # drop "(Mid Range)" etc.; kept in assay_name
        sd = pd.to_datetime(a["sample_date"], errors="coerce")
        rows.append({
            "name": name,
            "api": round(api, 1),
            "sulfur_pct": round(a["sulfur_pct"], 3),
            "type": crude_type(api, a["sulfur_pct"], name),
            "region": region,
            "source": "BP crude assays (bp.com, technical downloads)",
            "country": country,
            **{k: a[k] for k in ["density_15c_g_cc", "tan_mgkoh_g", "visc_40c_cst",
                                 "pour_point_c", "vanadium_ppm", "nickel_ppm"]},
            "api_derived_from_density": True,
            "duplicate_table_in_sheet": a["duplicate_table_in_sheet"],
            "assay_name": a["assay_name"],
            "sample_date": sd.date() if pd.notna(sd) else None,
            "n_cut_columns": a["n_cut_columns"],
            "raw_file": f"data/raw/bp/assays/{f.name}",
            "url": "https://www.bp.com/content/dam/bp/business-sites/en/global/bp-supply-trading-and-shipping/"
                   f"documents/technical-documents-and-downloads/crudes/{f.name}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- MINLPLib
def load_pooling() -> pd.DataFrame:
    meta = pd.read_csv(RAW / "minlplib" / "instancedata.csv", sep=";")
    have = sorted(p.stem for p in (RAW / "minlplib" / "gms").glob("*.gms"))
    m = meta[meta["name"].isin(have)].copy()
    m["references"] = m["references"].map(lambda s: ", ".join(ast.literal_eval(s)) if isinstance(s, str) else s)
    cols = ["name", "probtype", "objsense", "primalbound", "dualbound", "gap",
            "nvars", "ncontvars", "ncons", "nquadcons", "references"]
    m = m[cols].rename(columns={"primalbound": "best_known_obj"})
    m["files"] = m["name"].map(lambda n: ";".join(f"data/raw/minlplib/{e}/{n}.{e}" for e in ("gms", "osil", "py")))
    all_pool = meta[meta["application"].astype(str).str.contains("Pooling", case=False)]
    print(f"MINLPLib: {len(all_pool)} instances tagged 'Pooling problem'; downloaded {len(m)}")
    return m


if __name__ == "__main__":
    prices = load_eia(RAW / "eia" / "PET_PRI_SPT_S1_D.xls")
    prices.index.name = "date"
    prices.to_csv(CLEAN / "prices_daily.csv", date_format="%Y-%m-%d")
    q = price_quality(prices)
    q.to_csv(CLEAN / "prices_quality.csv", index=False)
    print(f"prices: {prices.shape}, {prices.index.min().date()} .. {prices.index.max().date()}")
    print(q.to_string())

    eq = load_crudes().assign(api_derived_from_density=False)
    bp = load_bp_crudes()
    crudes = pd.concat([eq, bp], ignore_index=True).sort_values(["api", "source"]).reset_index(drop=True)
    crudes.to_csv(CLEAN / "crudes.csv", index=False)
    print(crudes[["name", "api", "sulfur_pct", "type", "region", "source", "n_cut_columns"]].to_string())

    pool = load_pooling()
    pool.to_csv(CLEAN / "pooling_instances.csv", index=False)
    print(pool.drop(columns=["files"]).to_string())
