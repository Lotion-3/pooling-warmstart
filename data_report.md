# Data access audit — crude blending / pooling project

Audit run 2026-10-22 from this machine (Windows 11, Python 3.12, `requests` 2.34). Every HTTP
request is logged with status, final URL, content type and size in
[data/raw/access_log.jsonl](data/raw/access_log.jsonl). Raw files are stored as downloaded under
`data/raw/` and are never overwritten (re-downloads get a timestamp suffix).
To rebuild the clean files from raw: `python -P scripts/clean.py`.

## Summary

| Source | Reachable | What was obtained | Date range | Issues |
|---|---|---|---|---|
| **EIA spot prices**, "Download Series History" Excel (`pet_pri_spt_s1_d.htm` → `xls/PET_PRI_SPT_S1_D.xls`) | ✅ HTTP 200 (page 18.8 KB, xls 3.4 MB) | 11 daily series across 7 sheets: WTI, Brent, conv. gasoline (NYH, USGC), RBOB LA, No.2 heating oil NYH, ULSD (NYH, USGC, LA CARB), jet/kero USGC, propane Mont Belvieu | 1986-01-02 → 2026-10-20 (release 2026-10-21, next 2026-10-28) | Legacy `.xls` needs `xlrd` (installed). Mixed units ($/bbl vs $/gal). Patchy before ~2006. One negative WTI price (−36.98 on 2020-04-20). |
| DataHub mirror (`datahub.io/core/oil-prices`), fallback A | ✅ HTTP 200 (WTI 171 KB, Brent 164 KB CSV; also on GitHub raw) | Probed only, not saved: crude only, so less than EIA provides | — | Not needed |
| EIA v2 API, fallback B | Not tried | — | — | Not needed, so no key required |
| **ExxonMobil crude assays** (`corporate.exxonmobil.com`) | ❌ **HTTP 403 "Access Denied"** from AkamaiGHost on every URL, including `/` and `/robots.txt` | Nothing | — | Bot protection blocks the whole domain for scripted clients. I did not try to get around it. |
| **Equinor crude oil assays** (`equinor.com/energy/crude-oil-assays`) | ✅ HTTP 200 (765 KB, Next.js page with server-rendered table) | Summary table (46 grades: issue date, name, description, country, API) plus **all 46 per-grade XLSX assays** (2.8 MB, `cdn.sanity.io`, all HTTP 200) | Assay issue dates 2015-08 → 2026-10 | The page table has **no numeric sulfur**, so sulfur comes from the XLSX files. The XLSX links sit behind a JS dialog button but are in the embedded page JSON, so no JS rendering was needed. |
| **MINLPLib** (`minlplib.org`) | ✅ HTTP 200 (instances.html 1.4 MB, `instancedata.csv`) | 89 instances tagged "Pooling problem". Downloaded 8 in `.gms`, `.osil` and `.py` (Pyomo) formats, plus metadata with best-known objective values | — | None |

**Fallbacks used:** none for prices (the primary EIA Excel worked) or benchmarks. For assays,
ExxonMobil failed and Equinor succeeded; within Equinor I used the "per-grade XLS" route to get sulfur.

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

46 Equinor grades. Required columns are `name, api, sulfur_pct, type, region, source`. Extra columns:
`country, equinor_description, api_page_table, density_15c_g_cc, tan_mgkoh_g, visc_40c_cst, pour_point_c,
vanadium_ppm, nickel_ppm, assay_name, issue_date, n_cut_columns, raw_file, url`.

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

## What's still missing and how to get it

1. **ExxonMobil assays (blocked by 403).** Open
   https://corporate.exxonmobil.com/what-we-do/energy-supply/crude-trading/assays in a normal browser.
   Either save the page (Ctrl+S, "Webpage, complete") or download a few per-grade XLS files, and drop them
   in `data/raw/exxon/`. I can then parse them. This would add the heavy-sour grades Equinor lacks
   (e.g. Canadian heavies such as Kearl and Cold Lake).
2. **Very heavy / very sour benchmarks.** Equinor has only 3 heavy sour grades, and nothing above 2.6% S
   or below 14° API. For a wider quality range: ExxonMobil (above), or BP's assay page
   (bp.com, "crude assays"), which I didn't test.
3. **Crude-specific prices.** EIA spot data covers WTI and Brent only. There are no free daily prices for
   individual grades (Mars, Johan Sverdrup, etc.); those come from paid services (Argus, Platts).
   A simple stand-in is Brent/WTI ± a fixed quality differential per °API and %S. EIA's *monthly* "Landed
   costs of imported crude by API gravity / sulfur" tables would let us calibrate those differentials.
   They aren't downloaded yet but sit on the same EIA site.
4. **Product quality specs** (e.g. ULSD ≤ 15 ppm S, jet freeze point) for the pooling constraints come
   from regulations and standards, not datasets. They are easy to write down by hand when we build the model.
