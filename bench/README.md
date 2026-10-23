# Hard linear equality constraints in neural networks, across domains

**Question:** many physical and accounting systems obey exact linear balances: mass, charge, energy, carbon, or
assets = liabilities + equity. Can a neural-network surrogate be forced to satisfy them *exactly*, and what does
that do to its accuracy compared with a standard neural network?

The method tested is the KKT projection layer (KKT-hPINN) of **Chen, Constante-Flores & Li (2024), "Physics-informed
neural networks with hard linear equality constraints"**. The paper demonstrated it on Aspen surrogates of a CSTR,
an extractive distillation subsystem and a chemical plant. This suite applies it to 14 other domains.

## Results at a glance

| test case | field | conserved quantity | standard MLP | KKT-hPINN, as specified | KKT-hPINN, scaled projection | vs standard MLP (as specified / scaled) |
|---|---|---|---|---|---|---|
| [process_flowsheet](#process_flowsheet) | chemical process (IDAES) | component mass per unit | 0.02467 | 0.02501 | 0.02612 | no clear difference / worse |
| [heat_exchanger_network](#heat_exchanger_network) | heat integration | energy per exchanger side | 0.01792 | 0.01841 | 0.01531 | worse / **better** |
| [combustion](#combustion) | combustion chemistry | element mass | 0.5678 | 738.6 | 0.5814 | worse / no clear difference |
| [fermentation](#fermentation) | bioprocess | carbon | 0.08327 | 0.1569 | 0.08281 | worse / no clear difference |
| [metabolic](#metabolic) | systems biology | metabolite mass (S·v = 0) | 0.09272 | 1.003 | 0.08945 | worse / **better** |
| [pharmacokinetics](#pharmacokinetics) | pharmacology | drug mass vs dose given | 0.02519 | 0.02581 | 0.02408 | worse / **better** |
| [epidemic](#epidemic) | epidemiology | population | 0.04426 | 0.1923 | 0.04371 | worse / no clear difference |
| [dc_power_flow](#dc_power_flow) | power systems | power at every bus | 0.02549 | 0.01198 | 0.01157 | **better** / **better** |
| [water_network](#water_network) | water distribution | flow at every junction | 0.02431 | 0.01493 | 0.01155 | **better** / **better** |
| [circuit](#circuit) | electrical circuits | current (Kirchhoff) | 0.2958 | 0.4109 | 0.4033 | worse / worse |
| [truss](#truss) | structural mechanics | force equilibrium | 0.1423 | 0.2407 | 0.2343 | worse / worse |
| [building_end_use](#building_end_use) | building energy (real data) | end uses sum to totals | 0.5468 | 1.732 | 0.5693 | worse / worse |
| [bank_balance_sheets](#bank_balance_sheets) | banking (real data) | assets = liabilities + equity | 0.241 | 0.7124 | 0.2432 | worse / no clear difference |
| [county_population](#county_population) | demography (real data) | components of population change | 0.4807 | 7.503 | 0.4237 | worse / **better** |

**How to read the table:**

- **Metric: normalised RMSE (nRMSE), lower is better.** For each output, the test-set RMSE is divided by that
  output's test-set standard deviation. These ratios are averaged over outputs, then over seeds (3 per domain, 5 for
  fermentation). 1.0 is roughly "no better than predicting the mean".
- **Verdict.** "Better" or "worse" means the 95% bootstrap confidence interval (over test points) of the difference
  from the standard MLP excludes zero; otherwise "no clear difference".
- **Constraint satisfaction.** Both KKT-hPINN variants satisfy the equalities exactly in every domain: 0% of test
  points above a 1e-5 relative violation, and at most 3.9e-12 for the specified variant. The standard MLP violates them
  on 100% of test points in every domain.
- **"As specified" vs "scaled".** "As specified" projects onto the constraints in original units, as the protocol
  requires. "Scaled" uses the same layer, but finds the nearest feasible point after dividing each output by its
  training standard deviation. This was added during the run, beyond the original protocol (see
  [Test bench](#test-bench)).
- **Tally.**
  - As specified, KKT-hPINN is better in 2 domains, shows no clear difference in 1, and is worse in 11.
  - Scaled, it is better in 6, shows no clear difference in 4, and is worse in 4.
  - Where outputs span very different magnitudes, the specified variant's projection error becomes large (combustion
    738.6, county_population 7.503).
- **Two more domains were attempted and failed the data gate,** so no results are reported for them:
  - **FAOSTAT food balance sheets:** values are rounded to 1,000 t, so the identities close only to ±1–2 kt.
  - **SEC EDGAR company financials:** a few filings report totals that don't add up as filed.

All tables, including the soft-penalty model, post-hoc projection, bound violations and timings, are in
[results/summary.md](results/summary.md). Every judgement call is in [DECISIONS.md](DECISIONS.md), and the plots are in
[results/plots/](results/plots/).

## Test bench

Every domain is a module in `bench/domains/` that returns inputs X, outputs Y and a fixed linear equality
A·y + B·x = b. One harness, `bench/run.py`, treats all domains identically.

**1. Gate (before any training).** The equality must hold in the data itself:

- **Pass:** the max relative residual (against the typical magnitude of each row's terms) is ≤ 1e-6.
- **Project:** between 1e-6 and 1e-4 (solver tolerance or rounding), the labels are projected onto the constraints and
  this is recorded.
- **Fail:** above 1e-4, the domain is dropped.

**2. Degeneracy check.** Redundant constraint rows are removed. If rank(A) ≥ 0.8 × the number of outputs, the
constraints nearly determine the answer, so the domain is dropped as trivial. None were.

**3. Models.** All use the same 3 × 256 ReLU MLP, Adam (lr 1e-3, batch 256), early stopping on validation and
standardised inputs and outputs.

| model | what it is |
|---|---|
| (a) standard MLP | no constraint handling |
| (b) soft penalty | MLP + λ·‖relative constraint residual‖², λ ∈ {0.1, 1, 10} tuned on validation |
| (c) post-hoc projection | model (a), with its predictions projected onto the constraints after training |
| (d) KKT-hPINN | projection layer y = ŷ − Aᵀ(AAᵀ)⁻¹(Aŷ + Bx − b) inside the network, trained end to end |
| c_w, d_w (scaled) | as (c) and (d), but the projection is weighted by each output's training variance |

- **Precision and units.** The projection always acts on de-standardised outputs (original units), in float64.
- **Splits.** By time or group where the data has one (no leakage), otherwise random 70 / 15 / 15. At most 50,000
  samples per domain.
- **Metrics on the test set:**
  - nRMSE (above), plus a pooled nRMSE in original units
  - equality violation (max, mean, % of points above 1e-5)
  - bound violations (bounds are reported, never enforced)
  - training time
  - per-seed paired differences and bootstrap 95% CIs

**Why the scaled variant was added.** The specified projection moves outputs in original units, so its correction
falls mostly on outputs with large magnitudes and swamps small ones (trace species, small balance-sheet items,
births vs. population). The scaled variant reaches the same feasible set but distributes the correction relative to
each output's spread. It is reported alongside the specified variant, not instead of it.

**Design notes.**

- Tier 1 domains are simulated, so the constraint is exact by construction. Tier 2 domains are real data, where the
  constraint is an accounting identity and the gate decides. Tier 3 domains are extra simulated domains, added
  because time allowed.
- In circuit and truss, KKT-hPINN reached a lower training loss than the standard MLP but a higher validation loss
  (training curves in `results/<domain>/curves.json`), i.e. it overfit; no regularisation was tuned.

## Test cases

Each section gives the question, how the problem is solved conventionally (the "current method" that generates the
labels and that a surrogate would stand in for), and how it was evaluated. "Rank" is the number of independent
constraints; q is the number of outputs.

### process_flowsheet
*Closest to the original paper's case studies.*

- **Question:** can a surrogate of a separation flowsheet predict every stream's component flows while closing every
  unit's mass balance?
- **Current method:** an equation-oriented flowsheet solved with IDAES-PSE 2.13 + IPOPT. Two feeds enter a mixer, then a
  flash, then a splitter. Benzene/toluene/o-xylene, ideal VLE.
- **Evaluation:**
  - 20,000 converged solves (0 failures).
  - Inputs (13): feed component flows, feed T/P, flash T/P, split fraction.
  - Outputs (15): component flows of the 5 internal/product streams.
  - Constraints: per-component balances on mixer, flash and splitter (rank 9 / q 15).
  - Random split. Gate residual 3.25e-7 (IPOPT tolerance; no projection needed).
- **Data and license:** generated here; IDAES-PSE is BSD-style with a US DOE notice.

### heat_exchanger_network
- **Question:** can a surrogate of a heat-exchanger network predict duties and outlet temperatures that satisfy every
  exchanger's energy balance?
- **Current method:** effectiveness-NTU rating of counter-current exchangers, solved exchanger by exchanger. Two hot and
  two cold streams with fixed heat-capacity flows, four exchangers in a grid.
- **Evaluation:**
  - 20,000 samples.
  - Inputs: 4 inlet temperatures, 4 UA values. Outputs: duty and both outlet temperatures per exchanger (12).
  - Constraints: hot- and cold-side energy balances (rank 8 / q 12).
  - Random split.
- **Data and license:** generated here (MIT).

### combustion
- **Question:** can a surrogate of methane/air chemistry predict all species mass fractions while conserving each
  element exactly?
- **Current method:** stiff integration of detailed kinetics (Cantera 3.2, GRI-Mech 3.0, 53 species) in an adiabatic
  constant-pressure reactor for a fixed 1 ms.
- **Evaluation:**
  - 20,000 samples: 11,421 unignited, 395 igniting, 8,184 burnt.
  - Inputs: T0, P, equivalence ratio and the fresh-mixture mass fractions (so the element balance stays linear).
  - Outputs: 53 mass fractions.
  - Constraints: C, H, O, N and Ar mass, plus ΣY = 1 (rank 5 / q 53).
  - Random split.
  - Trace species make this the most multi-scale domain, which is where the specified projection struggles most.
- **Data and license:** Cantera BSD-3; GRI-Mech 3.0 is freely available from UC Berkeley.

### fermentation
- **Question:** can a surrogate of a batch fermentation predict glucose, biomass, ethanol, glycerol and CO2 over time
  while conserving carbon?
- **Current method:** a kinetic ODE model (Monod growth, product-inhibited ethanol, growth-coupled glycerol) in C-mol,
  integrated with LSODA.
- **Evaluation:**
  - 20,000 samples.
  - Inputs: 6 kinetic parameters, initial glucose and biomass. Outputs: 5 carbon pools × 20 times (100).
  - Constraints: total carbon = initial carbon at each time (rank 20 / q 100).
  - Random split, 5 seeds.
- **Data and license:** generated here (MIT).

### metabolic
- **Question:** can a surrogate predict every reaction flux of *E. coli* central metabolism for a given growth medium
  while keeping every metabolite balanced?
- **Current method:** constraint-based modelling. Parsimonious flux balance analysis (pFBA, a sequence of linear
  programs) with COBRApy on the e_coli_core model, plus a fixed tie-break LP so the answer is unique.
- **Evaluation:**
  - 20,000 media (10 infeasible draws dropped).
  - Inputs: 7 uptake / maintenance bounds. Outputs: all 95 fluxes.
  - Constraints: S·v = 0 (rank 67 / q 95).
  - Random split.
- **Data and license:** COBRApy (LGPL/GPL); model from BiGG (Orth et al. 2010).

### pharmacokinetics
- **Question:** can a surrogate predict drug amounts in each body compartment over a day of dosing while accounting
  for every milligram given?
- **Current method:** a linear three-compartment model with elimination, solved exactly with the matrix exponential.
- **Evaluation:**
  - 20,000 samples.
  - Inputs: 4 bolus doses, 5 rate constants. Outputs: 3 compartments + cumulative eliminated at 20 times (80).
  - Constraints: total = cumulative dose at each time (rank 20 / q 80).
  - Random split.
- **Data and license:** generated here (MIT).

### epidemic
- **Question:** can a surrogate of an epidemic predict S, E, I, R and D trajectories while keeping the population
  fixed?
- **Current method:** the SEIRD compartmental ODE model, integrated with DOP853.
- **Evaluation:**
  - 20,000 samples.
  - Inputs: 4 rates, population, initial E/I/R. Outputs: 5 compartments × 30 days (150).
  - Constraints: S + E + I + R + D = N at each time (rank 30 / q 150).
  - Random split.
- **Data and license:** generated here (MIT).

### dc_power_flow
- **Question:** can a surrogate predict line flows on a 118-bus grid that balance power exactly at every bus?
- **Current method:** DC power flow, a linear solve of the bus susceptance system, on the IEEE 118-bus topology from
  pandapower.
- **Evaluation:**
  - 20,000 samples.
  - Inputs: 99 bus loads and 18 generator outputs, balanced by a slack bus. Outputs: 186 branch flows + slack
    injection.
  - Constraints: power balance at all 118 buses (rank 118 / q 187).
  - Random split.
  - One of the two domains where KKT-hPINN beat the standard MLP even as specified.
- **Data and license:** pandapower BSD-3; the IEEE 118 test case is public test data.

### water_network
- **Question:** can a surrogate of a city water network predict every pipe and pump flow while conserving flow at
  every junction?
- **Current method:** a steady-state hydraulic simulation (WNTR's Newton solver; EPANET Net3: 92 junctions, 3 tanks,
  2 reservoirs, 119 links).
- **Evaluation:**
  - 20,000 demand scenarios.
  - Inputs: demands at 59 junctions. Outputs: 119 link flows.
  - Constraints: conservation at all 92 junctions (rank 92 / q 119, the most tightly constrained domain).
  - Random split.
  - KKT-hPINN beat the standard MLP in both variants.
- **Data and license:** WNTR revised BSD; EPANET example networks are US EPA, public domain.

### circuit
- **Question:** can a surrogate predict branch currents in a meshed resistor network while obeying Kirchhoff's current
  law?
- **Current method:** DC nodal analysis (a linear solve). 15 nodes, 32 branches, 18 independent loops.
- **Evaluation:**
  - 20,000 samples.
  - Inputs: 14 current injections and 32 resistances. Outputs: 32 branch currents.
  - Constraints: KCL at 14 nodes, with the incidence matrix fixed although the resistances vary (rank 14 / q 32).
  - Random split.
  - KKT-hPINN overfit here (see Test bench).
- **Data and license:** generated here (MIT).

### truss
- **Question:** can a surrogate predict member forces in a statically indeterminate truss while satisfying
  equilibrium at every joint?
- **Current method:** stiffness-method finite element analysis of a 2D bar truss (15 joints, 38 members,
  12× indeterminate). The forces depend on the member stiffnesses, not just on equilibrium.
- **Evaluation:**
  - 20,000 samples.
  - Inputs: 26 joint loads and 38 member stiffnesses. Outputs: 38 member forces.
  - Constraints: equilibrium at 26 free DOFs (rank 26 / q 38).
  - Random split.
  - KKT-hPINN overfit here (see Test bench).
- **Data and license:** generated here (MIT).

### building_end_use
- **Question:** can a surrogate predict a home's annual energy use by end use and fuel such that the end uses add up
  to the reported totals?
- **Current method:** building energy simulation of the U.S. housing stock (NREL ResStock 2025 release 1, EnergyPlus
  based), published as annual results per building.
- **Evaluation:**
  - 50,000 buildings from 20 states. Inputs: 119 building characteristics. Outputs: 50 end-use, fuel-total and
    site-total columns (kWh).
  - Constraints: end uses → fuel totals → site total, plus net-of-PV identities (rank 7 / q 50).
  - Group split by state (test: CO, NJ, TX).
  - **Caveat:** published values are rounded, so labels were projected (gate 9.73e-5, just under the 1e-4 limit). The
    state set was chosen after an earlier draw failed the gate on the propane row; see DECISIONS.md.
- **Data and license:** NREL ResStock via OEDI, CC BY 4.0; download script `scripts/get_building_end_use.py`.

### bank_balance_sheets
- **Question:** can a model forecast a bank's next-quarter balance sheet so that assets equal liabilities plus equity
  and the subtotals add up?
- **Current method:** call-report data as filed; next-quarter balance sheets are usually forecast with
  statistical/econometric models that do not enforce the identity.
- **Evaluation:**
  - 50,000 (bank, quarter) pairs, 2015–2024. Inputs: 18 items this quarter. Outputs: the same 18 next quarter.
  - Constraints: asset components → total assets, liability components → total liabilities, assets = liabilities +
    equity (rank 3 / q 18).
  - Time split (test from 2023Q2).
- **Data and license:** FDIC BankFind API (U.S. government data, public); `scripts/get_bank_balance_sheets.py`.

### county_population
- **Question:** can a model forecast a county's births, deaths, migration and population change for next year so
  that the components add up to the change?
- **Current method:** the Census Bureau's components-of-change population estimates; forecasts of the components
  usually do not enforce the accounting identity.
- **Evaluation:**
  - 40,863 county-years (Vintages 2020 and 2025). Inputs: last year's population and 8 components. Outputs: this
    year's 9.
  - Constraints: natural change, net migration, population change, and population_t = population_{t−1} + change (rank
    4 / q 9).
  - Time split (test 2024–2025).
  - Here the specified variant had a worse nRMSE but a much lower original-units error (pooled nRMSE 0.03282 vs
    0.1891). Large population counts dominate the original-units metric.
- **Data and license:** U.S. Census Bureau, public domain; `scripts/get_county_population.py`.

### Attempted, failed the gate
- **food_balance** (FAOSTAT Food Balance Sheets): supply = utilisation closes only to the 1,000 t rounding (max
  relative residual 3.27e-2; 29.7% of samples above 1e-6).
- **company_financials** (SEC EDGAR financial statement data sets): assets = liabilities + stockholders' equity fails
  in 6 of 42,672 filings as filed (max 5.33e-4).

## Credit and scope

KKT-hPINN is the work of Chen, Constante-Flores & Li (2024). This suite only applies their layer to new domains. The
layer is designed for linear equality constraints; inequalities such as nonnegative flows are outside its scope. Bound
violations are reported in [results/summary.md](results/summary.md) to show where inequality-aware methods would
apply, such as the decision-rule approach of Constante-Flores, Chen & Li (NeurIPS 2025, not implemented here).

## Reproduce

Requirements: Python 3.12 with `numpy scipy pandas pyarrow torch matplotlib cobra wntr pandapower cantera
idaes-pse`. process_flowsheet also needs IPOPT from the IDAES binaries (`idaes get-extensions`), on the path given by
`IDAES_BIN` (default `~/solvers/idaes_bin`).

```
python -P bench/tests/test_harness.py              # harness tests on a toy domain with a known answer
python -P bench/scripts/get_<domain>.py            # real-data domains only; files go to bench/data/raw/ (gitignored)
python -P bench/run.py <domain> --gate-only        # gate + rank check
python -P bench/run.py <domain>                    # full run -> results/<domain>/metrics.json, results/summary.csv
python -P bench/ablation.py <domain>               # scaled-projection variant (needs the full run first)
python -P bench/report.py                          # results/summary.md and plots
```

Simulated domains generate their samples on first use (fixed seeds) and cache them in `bench/data/cache/`
(gitignored). Raw downloads are never committed.

## Add a domain

Create `bench/domains/<name>.py` with a function `load()` that returns a `bench.domain.Domain`:

- `X` (n × p) and `Y` (n × q), in original units.
- The equality as `A` (m × q, fixed, not sample-dependent), `B` (m × p) and `b` (m).
- Optionally, `split` with `split_kind` (`"time"` or `"group"`).
- Optionally, `y_lo` and `y_hi` (fixed bounds, reporting only).
- `meta` with source, license, generation, units and tier.

Then run `python -P bench/run.py <name> --gate-only` and, if the gate passes, the full run.
