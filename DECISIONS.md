# Decisions log (overnight run, 2026-10-22)

Every assumption, deviation and adjustment made during the unattended run. Times are local (EDT).

## Phase 0 / 0.5 — setup (02:16–02:25)

- **Preflight:** git identity set (Laksh Agarwal <lakshagarwal65@gmail.com>). `poolingOptim/` held only
  `pooling_real/`, no root `.git`. 14 physical cores / 20 logical. OneDrive process not running.
- **Global solver: SCIP 9 via `pyscipopt` 6.3.0.** `gurobipy` is not installed, so the license check failed;
  per instructions, SCIP is used and Gurobi is not installed.
- **Local solver: IPOPT 3.13.2 (MUMPS) from the IDAES binary bundle** (`idaes get-extensions`, v3.4.2, includes MUMPS/HSL
  as shipped). `pip install cyipopt` failed: it tries to build from source and no IPOPT library is present.
  Pyomo 6.10.1 drives `ipopt.exe`.
- **Repo flattened:** everything (including `.git` and `.gitattributes`) moved from `pooling_real/` to
  `poolingOptim/`, and `pooling_real/` removed. `git status` was clean afterwards, and `python -P scripts/clean.py`
  rebuilt `data/clean/` byte-identically (`git status` still clean).
- **History rewrite:** tag `backup-before-rewrite` points at the ORIGINAL commits (6b02ca4, 6ef0113).
  `git filter-branch --msg-filter` removed the `Co-Authored-By` lines. My first run used `-- --all`, which also
  rewrote the tag; I re-pointed the tag at the original commit from `refs/original/`. Trees, authors and dates
  are unchanged (`git diff backup-before-rewrite master` is empty).
- **Untracked by accident earlier:** `scripts/__pycache__/fetch.cpython-312.pyc` had been committed in the
  first audit. It is now untracked, and a `.gitignore` covers `__pycache__/` and `*.pyc`.
- **No `pooling_real/` path references** in `scripts/` or `data_report.md` (only inside that `.pyc`).

## Phase 1 — validation data

- **MINLPLib files cannot be used directly to validate my formulation.** The `.gms/.py` files are flattened by
  GAMS Convert (anonymous `x1..xn`, with auxiliary variables for each bilinear product), so the instance data
  (qualities, prices, arcs) isn't recoverable cleanly.
- **Downloaded the original data instead:** GAMS Model Library model `pool` (SEQ=254, by M. Tawarmalani), saved
  raw as `data/raw/gamslib/pool.254` and its formulation `data/raw/gamslib/poolmod.inc`. It contains the
  literature data for haverly1–3, foulds2–5, bental4–5, rt2 and adhya1–4, with their global optima. These are the
  same instances MINLPLib's `pooling_*pq` models were generated from, and `best_known_obj` matches for all
  overlapping cases. This is a new download (validation data only, not Phase 2 data).
- **IPOPT binaries relocated.** The Microsoft Store Python silently redirects writes under `%LOCALAPPDATA%` into
  its package sandbox (`...\Packages\PythonSoftwareFoundation.Python.3.12_...\LocalCache\Local\idaes\bin`).
  Python could see `ipopt.exe` there, but Windows couldn't load its DLLs when launching it (exit 0xC0000135).
  Copied the bundle to `%USERPROFILE%\solvers\idaes_bin` (override with env var `IDAES_BIN`). The bundled IPOPT
  is **3.13.2** (MUMPS) and doesn't accept the `sb` option, so that option is removed.
- **Validation tolerance:** "match" = |obj − best_known| / max(1, |best_known|) ≤ 1e-4. The same measure defines
  an IPOPT "hit" for the rest of the project. An IPOPT point only counts if its max violation ≤ 1e-5 (my own
  evaluator, `pooling.evaluate`).
- **Spec-violation scaling:** spec residuals are divided by the flow into the product, floored at 1% of the
  product's capacity, so products with ~0 flow don't produce huge ratios.

## Phase 1 — formulation (02:30–02:50)

- **`src/pooling.py`** holds the general model. Specs are written as Σᵢ (α·λᵢ + β)·Fᵢⱼ ≤ 0, so one form covers
  volume-blended bounds and mass-blended (S·SG / SG) specs. Builders: Pyomo pq (optionally with RLT rows), Pyomo p,
  and a direct pyscipopt pq model with explicit wᵢₗⱼ = qᵢₗ·yₗⱼ variables (SCIP).
- **Local solves use pq WITHOUT the RLT rows** (`rlt=False`). With `pszrlt` + `extensions` included, IPOPT often
  stopped with "internal solver error" or "locally infeasible": those rows are linearly dependent on Σq = 1 and
  break LICQ. Without them, every one of 60 test runs ended "optimal". Hit rates ≥ the p-formulation on 4 of 6
  test cases, and the variable space (q, y, z) is the one the networks predict. The p-formulation hit rate is
  still reported in `results/validation.csv` for reference.
- **Global solves keep the RLT rows** (they tighten SCIP's relaxation). SCIP settings: `limits/gap` 1e-6,
  1 thread per process.
- **Result:** all 7 cases (haverly1–3, bental4, foulds2, adhya1, rt2) match MINLPLib to ≤ 1e-4 relative, with
  SCIP ≤ 0.4 s each. IPOPT hit rates from 20 random starts: 15%–95% (`results/validation.csv`).

## Phase 2 — real-data instances (02:26–02:31)

**Crude prices are MODELED, not observed.** No free daily prices exist for individual grades.

- **Benchmark:** Brent (daily). WTI is not used, because of the −$37 day.
- **Calibration data:** EIA monthly *landed* cost by API band (`land3`; 7 bands, midpoints 17.5…47.5).
  Landed has better coverage than F.O.B.: the 40–45° F.O.B. band ends 2025-09 and the 45°+ band in 2017.
  The difference from monthly-mean Brent is regressed with **month fixed effects**, which absorb freight and the
  Brent basis; that intercept is not used in pricing.
- **First attempt — joint API + sulfur fit — rejected.** Sulfur was taken from proxies: each band's median assay
  sulfur from our 82 grades, plus EIA's Basrah Light stream with the BP Basrah Medium assay (28.1°, 3.54% S) as
  the proxy for its 2021 successor. The fit gave **b = +1.29 $/bbl per %S** (SE 0.11), the wrong sign, and
  +9…+18 in 2022–26. The likely cause is the Basrah proxy: historical Basrah Light was lighter than Basrah Medium,
  so its price residual loads onto sulfur. Band-median sulfur is also non-monotone in API.
  This fit is saved under `joint_api_sulfur_fit_rejected` in `results/price_model.json`.
- **Used — API-only, by year:** price = Brent + a_y·(API − 35), with **b = 0**.
  - a_y ranges 0.25–0.82 $/bbl per °API (all positive, SE ≤ 0.06). Yearly within-R² is 0.71 vs 0.66 pooled, so the
    yearly coefficients are used.
  - Days after the last EIA month (2026-07) use the 2026 coefficient.
  - Consequence: the sulfur discount is only present to the extent sulfur correlates with API in imported crude.
    Sour crudes are therefore probably somewhat over-priced relative to reality.
  - Plot: `results/plots/price_model.png`.
- **Crude worth from assay yields** (volume % on crude):
  - LPG (≤C4; for BP, the remainder of the C5+ chain) → Mont Belvieu propane.
  - Naphtha (≤150 °C) → USGC conventional gasoline.
  - Kero (150–250 °C; BP 149–232) → USGC jet.
  - Gas oil (250–370 °C; BP 232–369) → USGC ULSD.
  - Atmospheric residue (370 °C+ / 369 °C+) → **0.85 × Brent per bbl** (assumption).
  - Products: $/gal × 42. Volume gain/loss and processing cost are ignored.
  - Propane has gaps inside the core panel; those are forward-filled.
  - Missing TAN for 3 grades is set to the median of the 82 grades.
- **Core panel:** 4,949 dates (2006-06-14 → 2026-10-20) where Brent, both gasolines, both ULSDs and jet are all
  present. This is 55 more than the 4,894 in the data report, which also required WTI.
- **Network** (per instance, fixed seed 20261022, `SeedSequence.spawn` per (date, sample)):
  - **Crudes:** 6–8 from the 82, at least one with S > 1% and one light sweet, no two assays of the same grade.
    Ordered by API.
  - **3 tanks**, capacity 60 kbd ±10%. Each crude is linked to 1–2 random tanks, and each tank gets ≥ 2 crudes.
    This breaks tank symmetry, so the optimal structure is learnable.
  - **Both tanks feed both CDUs.** Direct crude→CDU pipelines exist with probability 0.3 per pair.
  - **Availability:** U(15, 50) kbd per crude. **Handling cost:** U(0.10, 0.30) $/bbl per tank.
  - **CDU-A** (sour-tolerant): 100 kbd, min 30% of capacity, S ≤ 2.0 wt%, API 26–36, TAN ≤ 0.8.
    **CDU-B** (low-S): 80 kbd, min 30%, S ≤ 0.8, API 30–40, TAN ≤ 0.8.
    Every number is jittered ×U(0.9, 1.1) per instance; the API window is kept ≥ 4° wide.
  - Product (gasoline/diesel ppm) specs are deliberately NOT used.
- **Tracked qualities** λ = (S·SG, SG, TAN·SG). Sulfur and TAN are mass-blended specs, SG is volume-blended
  (API limits converted to SG). Objective is minimised: Σᵢ (costᵢ − worthᵢ)·Fᵢ + Σₗ hₗ·throughputₗ, in k$/day.
- **9,898 instances** = 4,949 dates × 2 samples → `data/instances/instances.parquet`.

## Phase 3 — smoke test (02:31–02:33)

- 50 sample-0 instances evenly spaced over the panel (1–3 per year).
- **SCIP:** 43 optimal + 7 stopped at the 1e-6 gap limit, so all 50 certified. Mean 0.75 s, max 3.1 s.
- **IPOPT** (20 random starts each): every run ended "optimal", 0.11 s mean per start.
  Hit rate 0.865, **trap rate 0.135**, infeasible endings 0. 15/50 instances had ≥ 1 trapped start, and no IPOPT
  point beat SCIP. Full summary: `results/phase3/summary_round0.txt`.
- **Trap rate ≥ 10%, so no difficulty adjustments were made** (round 0 only).
- **Sizing:** about 240 instances/min on 19 workers, so all 9,898 take ~45 min. **N = all 9,898 instances**
  (both samples, every date), far inside the 06:45 limit.
- Trap is defined as: IPOPT ends at a feasible point (max violation ≤ 1e-5) whose objective is > 1e-4 (relative)
  worse than SCIP's.

## Phase 4 — overnight solve (02:32–03:04)

- **All 9,898 instances** solved in 31.5 min on 19 worker processes (≈314/min).
- **SCIP status:** 8,219 optimal; 1,647 stopped at the 1e-6 gap limit (certified); **32 proven infeasible**.
  Infeasible means the CDU minimum throughputs can't be met within the specs and availabilities. Those 32 are
  kept in `data/solved/` but have no labels and are excluded from Phase 5.
- **Time limit:** no instance hit 60 s (max 11.6 s), so none needed the "uncertified" exclusion.
- **IPOPT** (20 random starts × 9,866 certified instances): hit 0.878, **trap 0.122**, 3,110 instances with
  ≥ 1 trapped start, 0 points better than SCIP. Summary: `results/phase4_summary.txt`.
- **Fixed-network dataset added** (an interpretation decision):
  - Plot 3 asks whether failures cluster "at dates where the optimal structure changed from the previous day".
    That needs the same network on consecutive days, but the main set draws a new crude set and network for
    every (date, sample).
  - So a second set, `instances_fixed.parquet`, reuses **2 fixed networks** (sampled once, seed 20261022+1;
    see `fixed_networks.json`) on all 4,949 dates. Only prices vary.
  - Same solver settings. It is used for plot 3 and as a second learning setting.
  - The main benchmark (`varied`) remains the primary result.
- **Results storage:** batched part files in `data/solved/parts*/`, consolidated into
  `data/solved/solved.parquet` / `solved_fixed.parquet` by `scripts/consolidate.py`.
- **Timing caveat:** all timings were measured with 19 concurrent processes on 14 physical cores, so absolute
  times are inflated relative to a single idle core. That applies equally to SCIP and IPOPT.
