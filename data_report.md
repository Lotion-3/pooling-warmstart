# Data access audit — crude blending / pooling project

Audit run 2026-10-22 from this machine (Windows 11, Python 3.12, `requests` 2.34). Every HTTP
request is logged with status, final URL, content type and size in
[data/raw/access_log.jsonl](data/raw/access_log.jsonl). Raw files are stored as downloaded under
`data/raw/` and are never overwritten (re-downloads get a timestamp suffix).
To rebuild the clean files from raw: `python -P scripts/clean.py`.

> **Public repository note:** the Equinor and BP assay files (and the GAMS `pool` files added later for
> validation) are not redistributed. Re-download them with `scripts/get_equinor.py`, `scripts/get_bp.py` and
> `scripts/get_gamslib.py`; paths below refer to where those scripts put them.

## Summary

| Source | Reachable | What was obtained | Date range | Issues |
|---|---|---|---|---|
| **EIA spot prices**, "Download Series History" Excel (`pet_pri_spt_s1_d.htm` → `xls/PET_PRI_SPT_S1_D.xls`) | ✅ HTTP 200 (page 18.8 KB, xls 3.4 MB) | 11 daily series across 7 sheets: WTI, Brent, conv. gasoline (NYH, USGC), RBOB LA, No.2 heating oil NYH, ULSD (NYH, USGC, LA CARB), jet/kero USGC, propane Mont Belvieu | 1986-01-02 → 2026-10-20 (release 2026-10-21, next 2026-10-28) | Legacy `.xls` needs `xlrd` (installed). Mixed units ($/bbl vs $/gal). Patchy before ~2006. One negative WTI price (−36.98 on 2020-04-20). |
| DataHub mirror (`datahub.io/core/oil-prices`), fallback A | ✅ HTTP 200 (WTI 171 KB, Brent 164 KB CSV; also on GitHub raw) | Probed only, not saved: crude only, so less than EIA provides | — | Not needed |
| EIA v2 API, fallback B | Not tried | — | — | Not needed, so no key required |
| **ExxonMobil crude assays** (`corporate.exxonmobil.com`) | ❌ **HTTP 403 "Access Denied"** from AkamaiGHost on every URL, including `/` and `/robots.txt` | Nothing | — | Bot protection blocks the whole domain for scripted clients. I did not try to get around it. |
| **Equinor crude oil assays** (`equinor.com/energy/crude-oil-assays`) | ✅ HTTP 200 (765 KB, Next.js page with server-rendered table) | Summary table (46 grades: issue date, name, description, country, API) plus **all 46 per-grade XLSX assays** (2.8 MB, `cdn.sanity.io`, all HTTP 200) | Assay issue dates 2015-08 → 2026-10 | The page table has **no numeric sulfur**, so sulfur comes from the XLSX files. The XLSX links sit behind a JS dialog button but are in the embedded page JSON, so no JS rendering was needed. |
| **BP crude assays** (`bp.com/.../technical-downloads/crudes-assays.html`) | ✅ HTTP 200 (old `.../bp-global-energy-trading/...crude-assays.html` URL now redirects to the bp homepage) | **36 per-grade assay files** (1.9 MB), including Middle East sour grades | Sample dates 2012-02 → 2026-01 | Mixed real `.xls` and `.xlsx` saved as `.xls` (pandas detects which). No API row, so API is computed from density. Two sheets (Azeri, Thunder Horse) have a stray second copy of the table; the labeled one is used. |
| **MINLPLib** (`minlplib.org`) | ✅ HTTP 200 (instances.html 1.4 MB, `instancedata.csv`) | 89 instances tagged "Pooling problem". Downloaded 8 in `.gms`, `.osil` and `.py` (Pyomo) formats, plus metadata with best-known objective values | — | None |
| **EIA monthly import / acquisition costs** (7 tables) | ✅ HTTP 200 | Page + Excel for each: landed and F.O.B. costs by area, by crude stream (22 and 14 named streams) and **by API gravity**, plus refiner acquisition cost | 1973/1983 → 2026-07 (RAC → 2026-08) | Raw only, not cleaned yet. Monthly, $/bbl. |
| **eCFR 40 CFR Part 1090** (US fuel rules) | ✅ HTTP 200 via official eCFR API | Full Part 1090 XML (589 KB, 191 sections), current to 2026-10-20 | — | Specs to be read by hand: gasoline S ≤ 10 ppm average / 80 ppm per gallon at refinery gate (§1090.205); ULSD S ≤ 15 ppm, cetane index ≥ 40 or aromatics ≤ 35 vol% (§1090.305) |
| ASTM D975 / D4814 / D1655 (diesel, gasoline, jet specs) | ✅ pages load (`store.astm.org`) | Not downloaded | — | Paywalled standards |

**Fallbacks used:** none for prices (the primary EIA Excel worked) or benchmarks. For assays,
ExxonMobil failed. Equinor and BP both succeeded; for each I used the per-grade XLS files to get sulfur.

## 1. EIA daily spot prices → `data/clean/prices_daily.csv`

10,428 rows (one per date with at least one price) × 11 series. Columns are mapped by EIA source
key, not by position:

| Column | EIA key | Unit | First date |
|---|---|---|---|
| `wti_cushing_usd_per_bbl` | RWTC | $/bbl | 1986-01-02 |
| `brent_europe_usd_per_bbl` | RBRTE | $/bbl | 1987-05-20 |
| `gasoline_conv_reg_nyh_usd_per_gal` | EER_EPMRU_PF4_Y35NY_DPG | $/gal | 1986-06-02 |
| `gasoline_conv_reg_usgc_usd_per_gal` | EER_EPMRU_PF4_RGC_DPG | $/gal | 1986-06-02 |
| `gasoline_rbob_reg_la_usd_per_gal` | EER_EPMRR_PF4_Y05LA_DPG | $/gal | 2003-03-17 |
| `heating_oil_no2_nyh_usd_per_gal` | EER_EPD2F_PF4_Y35NY_DPG | $/gal | 1986-06-02 |
| `ulsd_nyh_usd_per_gal` | EER_EPD2DXL0_PF4_Y35NY_DPG | $/gal | 2006-06-14 |
| `ulsd_usgc_usd_per_gal` | EER_EPD2DXL0_PF4_RGC_DPG | $/gal | 2006-06-14 |
| `ulsd_carb_la_usd_per_gal` | EER_EPD2DC_PF4_Y05LA_DPG | $/gal | 1996-04-17 |
| `jet_kero_usgc_usd_per_gal` | EER_EPJK_PF4_RGC_DPG | $/gal | 1990-04-02 |
| `propane_mont_belvieu_usd_per_gal` | EER_EPLLPA_PF4_Y44MB_DPG | $/gal | 1992-07-09 |

All series end 2026-10-20. Products are **$/gallon** and crude is **$/barrel** (1 bbl = 42 gal), so
convert before computing crack spreads or margins.

**Data quality** (full stats per series in [data/clean/prices_quality.csv](data/clean/prices_quality.csv)):

- **Share of weekdays with no price, by era:**
  - 1986–2005: 16–44%, with multi-week gaps (WTI's longest gap is 14 days, ending 1999-09-15; NYH heating oil has 140 gaps longer than 5 days).
  - 2006-06 to 2015: 3.5–6.3%.
  - 2016–2026: 2.7–4.3%, which is roughly the US market-holiday rate.
- **Market holidays:** since 2006-06, 117 of 5,310 weekdays have no price in any series. Those are market holidays, not missing data.
- **Complete core panel:** 4,894 days (2006-06-14 → 2026-10-20) have all of WTI, Brent, both conventional gasolines, both ULSDs and jet. This is the practical window for the model.
- **Calendar mismatch:** Brent follows the London calendar. Since 2010 there are 42 days with WTI but no Brent (and vice versa on US holidays).
- **Outliers:** WTI −36.98 $/bbl on 2020-04-20 is genuine (the contract-expiry squeeze), not an error. It needs explicit handling (drop, winsorize, or use Brent) in any price model.
- **Missing days are left blank:** no forward-fill or interpolation in the clean file.

**Terms:** EIA content is U.S. government work in the public domain (saved
[eia_copyrights_reuse.htm](data/raw/eia/eia_copyrights_reuse.htm)). EIA asks for attribution with the
publication date, e.g. "Source: U.S. Energy Information Administration (Oct 2026)". The page notes that
weekly/monthly/annual values are unweighted averages of daily closing spot prices.

## 2. Crude assays → `data/clean/crudes.csv`

82 rows: 46 Equinor and 36 BP assays. Grades that both companies publish (17 of them) appear once per
source; `source` tells them apart. Required columns are `name, api, sulfur_pct, type, region, source`.
Extra columns: `country, equinor_description, api_page_table, density_15c_g_cc, tan_mgkoh_g, visc_40c_cst,
pour_point_c, vanadium_ppm, nickel_ppm, assay_name, issue_date (Equinor), sample_date (BP), n_cut_columns,
api_derived_from_density, duplicate_table_in_sheet, raw_file, url`.

Overall coverage: 36 light sweet, 6 light sour, 11 medium sweet, 20 medium sour, 3 heavy sour, 6 condensates.
API runs 14.3–68.6 and sulfur 0.002–4.12%.

### Equinor (46 grades)

- `api` and `sulfur_pct` are **whole-crude** values from each XLSX "Summary" sheet (the IBP→FBP column).
  API from the XLSX matches the page table to 0.1° for all 46 grades.
- `type` is derived with the usual cutoffs: light > 31.1° API, medium 22.3–31.1, heavy < 22.3; sweet < 0.5 wt% S.
  The three condensates are labelled `condensate`. Equinor's own wording is kept in `equinor_description`.
- `region` is grouped by country: North Sea, West Africa, Caspian, South America, North America, North Africa.
- Coverage: 23 light sweet, 1 light sour, 5 medium sweet, 11 medium sour, 3 heavy sour, 3 condensates.
  - API range is 14.3 (Peregrino) to 68.6 (Algerian Condensate).
  - Sulfur range is 0.002% to 2.52% (Southern Green Canyon).
  - Good sour/heavy anchors: Peregrino (14.3°, 1.63% S), Mariner (15.3°, 1.12%), Mars (28.5°, 2.21%), SGC (27.8°, 2.52%), Poseidon (29.9°, 1.92%).
- Quality: TAN is missing for 3 grades. Assay ages vary; some date from 2015–2019 (Ekofisk 2015, Girassol 2015, Pazflor 2017).
- Each XLSX also has **cut data for 15 distillation fractions** (yield, density, sulfur, etc. per cut).
  This is useful later if the pooling model works at the level of cuts rather than whole crudes. I haven't parsed it yet.

**Terms:** the page (and the XLSX footer, checked on Alvheim) says *"The contents of the assays in this section are for guidance only
and Equinor accepts no liability for any loss occurring from the use of these assays and errors that they
may contain."* The site footer links to general Terms and conditions (`equinor.com/about-us/terms-and-conditions`),
which I didn't review in detail.

### BP (36 grades)

- Whole-crude values come from the IBP→FBP column of the "Summary" sheet, with 13 cut columns per file.
  BP gives density at 15 °C but no API, so `api = 141.5 / (ρ15 / 0.99904) − 131.5` (`api_derived_from_density = True`).
- **Cross-check against Equinor** on the 17 shared grades: API differs by at most 1.4° (Ekofisk, Kissanje),
  and sulfur by at most 0.34 points (Southern Green Canyon, 2.52 vs 2.18). That's consistent with different
  sample dates, and it confirms the density→API conversion.
- New sour grades: **Basrah Heavy** (23.7°, 4.12% S), **Basrah Medium** (28.1°, 3.54%), **Kirkuk** (34.1°, 2.24%),
  **Oman Export** (33.1°, 1.29%), Murban (39.9°, 0.77%), Thunder Horse (34.2°, 0.70%).
  Basrah Heavy falls just inside "medium" on the API cutoffs despite its name.
- Quality: vanadium/nickel are blank for 11 light grades (reported as "-"). Alvheim has no sample date.
  Some assays are old (Kirkuk and Basrah Heavy 2015, North West Shelf Condensate 2012).
- **Terms:** the page states the material is "provided for informational purposes only… bp makes no
  representations or warranties… regarding the accuracy… bp accepts no liability in connection with any of such information."

## 3. Pooling benchmarks → `data/raw/minlplib/` and `data/clean/pooling_instances.csv`

Downloaded in `.gms`, `.osil` and `.py` (Pyomo) formats:

| Instance | Best known objective (min) | Vars | Cons | Reference |
|---|---|---|---|---|
| haverly | −400 | 12 | 9 | Haverly 1978 |
| pooling_haverly1pq | −400 | 10 | 13 | Haverly 1978 |
| pooling_haverly2pq | −600 | 10 | 13 | Haverly 1978 |
| pooling_haverly3pq | −750 | 10 | 13 | Haverly 1978 |
| pooling_bental4pq | −450 | 13 | 16 | Ben-Tal et al. 1994 |
| pooling_foulds2pq | −1100 | 36 | 34 | Foulds et al. 1992 |
| pooling_adhya1pq | −549.803 | 33 | 49 | Adhya et al. 1999 |
| pooling_rt2pq | −4391.83 | 34 | 52 | Audet et al. 2000 |

All are QCPs (nonconvex bilinear) and all are solved to proven optimality (gap ≤ 1e-8). The other 81 pooling
instances (tp/stp formulations, the larger `pooling_digabel*`, `pooling_epa*`, `pooling_meyer*` and others) can
be downloaded the same way (`https://www.minlplib.org/{gms|osil|py}/<name>.<ext>`).
**Terms:** MINLPLib is licensed CC BY 4.0, so cite MINLPLib and the original papers.

## 4. Added in the second pass (raw only, not cleaned yet)

- `data/raw/eia_monthly/`: 7 EIA monthly tables (page + Excel each).
  - Landed costs: `land1` by area, `land2` by 22 named crude streams, `land3` by API gravity band.
  - F.O.B. costs: `imc1` by area, `imc2` by 14 named streams, `imc3` by API gravity band.
  - `rac2`: refiner acquisition cost.
  - All are $/bbl monthly, starting 1973/1983. The API-gravity bands are the natural basis for quality price differentials.
- `data/raw/ecfr/40CFR1090_2026-10-20.xml`: full U.S. fuel regulation, public domain.
- `data/raw/bp/`: BP page + 36 assays, cleaned into `crudes.csv` above.

## What's still missing and how to get it

1. **ExxonMobil assays (blocked by 403):**
   https://corporate.exxonmobil.com/what-we-do/energy-supply/crude-trading/assays
   Open it in a normal browser, download the per-grade files (or save the page with Ctrl+S) into
   `data/raw/exxon/`, and I'll parse them. This would add Canadian heavies (e.g. Kearl, Cold Lake) and fill
   the gap below 14° API that neither Equinor nor BP covers.
2. **Daily prices for individual crude grades.** These are paid only:
   Argus https://www.argusmedia.com/en and S&P Global Platts https://www.spglobal.com/commodity-insights/en
   (Platts returns 403 to scripts but works in a browser). The free substitute is Brent/WTI ± a quality
   differential calibrated on the EIA monthly gravity-band and named-stream tables above.
3. **Full product specs beyond the federal limits** (jet freeze point, distillation, octane, RVP by season)
   are in paid ASTM standards:
   - Diesel D975: https://store.astm.org/d0975-23.html
   - Gasoline D4814: https://store.astm.org/d4814-23.html
   - Jet D1655: https://store.astm.org/d1655-24.html

   The key numbers are widely quoted, so for the model we can hard-code a handful of them
   (e.g. Jet A freeze point ≤ −40 °C, ULSD ≤ 15 ppm S) without buying the standards.
