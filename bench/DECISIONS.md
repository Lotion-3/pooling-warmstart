# Benchmark suite: decisions log

Every judgement call made while building the suite, in the order it was made.

## Environment

- **Packages.** `cobra 0.32.1`, `wntr 1.5.0`, `pandapower 3.5.6` and `cantera 3.2.0` were installed with
  `python -m pip` into the project's Python 3.12. This upgraded numpy 2.2.1 → 2.4.6 and pandas 2.2.3 → 2.3.3
  (pip's resolver). The pooling results were produced before this and were not re-run. A first `pip install`
  went to a different interpreter on PATH (Anaconda); it did not affect this project.
- **Hardware.** 20 logical cores; at most 19 training processes run at once.

## Harness (bench/run.py)

- **Gate scale.** For each constraint row i, `scale_i` = mean over all samples of
  Σ_j |A_ij y_j| + Σ_k |B_ik x_k| + |b_i| (the typical magnitude of the terms in that row). The relative residual of
  a sample is max_i |A y + B x − b|_i / (scale_i + 1e-12). The gate uses the max over all samples. The same scale
  is used for every violation number reported later.
- **Label projection.** When 1e-6 < gate residual ≤ 1e-4, the labels of ALL samples (train/val/test) are
  projected with the same KKT formula as the models, and the gate is re-checked. This is recorded as
  `labels_projected` in metrics.json and in the summary table.
- **Redundant rows.** The rank uses SVD with numpy's default tolerance (σ > σ_max · max(m, q) · eps). Independent
  rows are chosen by QR with column pivoting on Aᵀ. The projection and the KKT layer use only the reduced rows.
  Violation metrics and the soft penalty use ALL original rows.
- **Degeneracy.** Rank(A) ≥ 0.8·q → "degenerate", dropped (no training).
- **Cap.** n > 50,000 → a fixed-seed (12345) uniform random subsample of 50,000 before splitting.
- **Splits.** Random 70/15/15 with seed 12345 when the domain has no key. Time keys: sorted, and whole keys are
  assigned to train until the cumulative share reaches 70%, then val to 85%, then test. Group keys: groups are
  shuffled (seed 12345) and assigned the same way. No key is shared between parts. If a coarse key leaves val or
  test empty, the last key(s) in order go to them.
- **Standardisation.** X and Y are standardised with training mean/std (a std below 1e-12 is set to 1). The network
  output is de-standardised to original units in float64 BEFORE the projection. The training loss is MSE in
  standardised units for every model, including (d), whose projected original-units output is re-standardised for
  the loss.
- **Backbone.** 3 hidden layers × 256, ReLU; Adam lr 1e-3; batch 256; at most 300 epochs; early stopping with
  patience 20 on the model's own validation objective (MSE, plus the penalty for (b)), restoring the best weights.
  Network weights are float32. De-standardisation, projection, penalty and loss run in float64.
- **Soft penalty (b).** λ · mean over rows and batch of ((A y + B x − b)_i / scale_i)², i.e. the relative residual
  ("scaled units"). λ ∈ {0.1, 1, 10} is tuned ONCE with seed 0 by validation nRMSE. Later seeds reuse the chosen λ,
  and the seed-0 model for the chosen λ is reused as seed 0 of (b).
- **Post-hoc (c).** The model (a) of the same seed, projected at test time. Its training time is that of (a) plus
  the projection.
- **Seeds.** Seed 0 is timed (a + b + d, including the two extra λ-tuning fits). < 300 s → 5 seeds, else 3. The data
  split is the same for all seeds; seeds change initialisation and batch order.
- **nRMSE.** Per output RMSE on the test set ÷ that output's TEST-set std (ddof 0), averaged over outputs. Outputs
  with a test std below 1e-12·max(1, |mean|) (constant on the test set) are excluded from the average; their number
  is recorded as `n_outputs_scored`.
- **Equality violation.** Per test point, the max over all original rows of the relative residual (as in the gate).
  Reported as max and mean over points, and % of points above 1e-5.
- **Bounds.** Per point, max over outputs of the bound excess divided by that output's test std. Reported as % of
  points above 1e-5, plus the max in original units and relative to std. The labels' own bound violation is
  recorded too (`label_bounds_test`) as a sanity check.
- **Bootstrap.** 1,000 resamples of test points (seed 0). Each resample recomputes the per-output RMSEs with fixed
  (full-test) std denominators, averages them over outputs and over seeds, and takes the difference between two
  models. Intervals are reported for d−a, d−c and c−a. Paired per-seed differences are reported too.
- **summary.csv.** Each run replaces its own domain's row; a lock file guards concurrent writers.

## Harness tests (bench/tests/test_harness.py)

- **Toy domain.** Y = X W + c with 2 exact equalities, outputs spanning 1e-2 to 1e3. The tests check: projection
  correctness (feasible, idempotent, orthogonal move, redundant row removed); gate pass, label projection (1e-5
  noise) and fail (1e-2 noise); degeneracy flag; time split without leakage; and that the end-to-end (c)/(d)
  violation is < 1e-9.
- **Observation from the tests.** Euclidean projection in original units can only shrink the per-sample error in
  original units. It can still RAISE nRMSE, which weights every output equally after dividing by its std. On the
  toy domain (outputs spanning 5 orders of magnitude) (c) had a higher nRMSE than (a) in seed 0. This is a property of
  the projection as specified (original units, identity metric), not a bug. The test now checks the true guarantee
  instead.

## Run-time choices

- **Simulated sample size.** Every simulated (tier-1) domain uses n = 20,000 samples, for comparability and to fit
  the overnight budget. Real-data domains use up to the 50,000 cap.
- **Caches.** Generated samples are cached in `bench/data/cache/` (gitignored). They are regenerated
  deterministically by the loaders.
- **Multiprocessing.** `concurrent.futures.ProcessPoolExecutor` crashed on this machine (BrokenProcessPool,
  WinError 6). Loaders use `multiprocessing.get_context("spawn").Pool`, as the pooling runner does.
- **Training threads.** Each domain trains in one process. truss and circuit run with 2 torch threads, later domains
  with 1, so the number of busy cores stays near the core count while data generation runs at the same time. One
  training fit took about 500 s under this load. Thread count changes only wall time, not results (up to
  floating-point reduction order).

## Domains

### truss (tier 1)
- **Geometry.** 5 × 3 joint grid at 1 m spacing. Horizontal, vertical and both diagonal members: 38 members. Pins
  at the two bottom corners leave 26 free DOFs, so the truss is 12× indeterminate.
- **Inputs.** Loads U(−50, 50) kN per free DOF. EA = 200 MN × logU(0.25, 4) per member, per sample.
- **Outputs.** Member forces from the stiffness method, float64.
- **Equality.** A = G_free (geometry only), B = [−I, 0].

### circuit (tier 1)
- **Network.** 15 nodes, node 0 ground. A ring plus random chords gives 32 branches (topology seed 202).
- **Inputs.** Current injections U(−1, 1) A at the 14 non-ground nodes; resistances logU(1, 100) Ω per branch per
  sample.
- **Equality.** KCL at the 14 non-ground nodes: A = reduced incidence, B = [−I, 0]. The ground row is omitted
  because it is the negative sum of the others.

### pharmacokinetics (tier 1)
- **Model.** Linear 3-compartment model (central, shallow and deep peripheral), elimination from the central
  compartment into a cumulative state E.
- **Dosing.** IV boluses at 0, 6, 12 and 18 h, amounts U(10, 100) mg.
- **Rates.** k12, k21 logU(0.05, 2); k13 logU(0.01, 0.5); k31 logU(0.005, 0.2); kel logU(0.05, 1) 1/h.
- **Propagation.** Exact, with scipy's matrix exponential between events, instead of a numerical ODE solver.
- **Outputs.** At 20 times linspace(0.5, 23.5) h, chosen so that none coincides with a dose time.
- **Equality.** A1 + A2 + A3 + E = cumulative dose given before t, which is linear in the dose inputs.

### epidemic (tier 1)
- **Model.** SEIRD solved with solve_ivp (DOP853, rtol 1e-10, atol 1e-8). Runge–Kutta methods preserve linear
  invariants up to rounding.
- **Sampling.** beta U(0.1, 0.6); sigma = 1/U(2, 7); gamma = 1/U(4, 14); mu U(0.001, 0.02); N logU(1e4, 1e6);
  E0, I0 = N·U(1e-4, 1e-2); R0 = N·U(0, 0.3).
- **Outputs.** At days 5, 10, …, 150.
- **Equality.** S + E + I + R + D − N = 0 at each time (N is an input).

### metabolic (tier 1; loader written and gate-checked by a parallel worker, reviewed here)
- **Model.** cobra `textbook` (e_coli_core; 72 metabolites × 95 reactions), GLPK with tolerance 1e-9.
- **Inputs.** 7 lower bounds: glucose [−20, −1], O2 [−30, 0], fructose [−10, 0], acetate [−10, 0], NH4 [−10, −1],
  Pi [−10, −1], ATPM [0, 15].
- **Infeasible draws.** 10 of 20,010 candidate media had an infeasible first FBA step and were dropped.
- **pFBA is not always unique.** FVA at the pFBA optimum showed spreads up to ~0.6 in 3 of 100 test media. A third
  LP therefore fixes Σ|v| (relative slack 1e-9) and minimises Σ w_i|v_i| with fixed generic weights w ~ U(1, 2).
  This makes the input → output map a function. It matched cobra's `pfba` to 1e-6 in a spot check.
- **Constant outputs.** 26 of 95 fluxes are always 0 (unused pathways). They are kept as labels and excluded from
  nRMSE only if constant on the test set.
- **Bounds.** Model default bounds; for the 7 sampled reactions, the loosest value of the sampling range;
  ±1000 → ±inf.

### dc_power_flow (tier 1; parallel worker, reviewed here)
- **Network.** pandapower case118: 173 lines + 13 transformers = 186 branches.
- **DC approximation.** b = 1/(x·tap) on a 100 MVA base. Resistance, charging and shunts are ignored; phase shifts
  are all 0 (asserted).
- **Transformers.** Four transformers are stored with negative i0 by pandapower; their magnetising branch is
  ignored, which reproduces the IEEE reactances. The numpy solver matches `pandapower.rundcpp` to 5e-12 MW on equal
  reactances.
- **Inputs.** 99 bus loads = base × U(0.7, 1.3). The 18 non-slack generators with non-zero base dispatch are set to
  base × U(0.7, 1.3); 35 zero-dispatch units (mostly synchronous condensers) are held at 0 and left out of X.
  Sampling around dispatch rather than U(0, pmax) is deliberate: pandapower's pmax is dispatch + 100 MW for every
  unit, and U(0, pmax) would make the slack mostly negative.
- **Slack.** Bus 69 (IEEE numbering).
- **Equality.** Balance at all 118 buses: A = [−C | e_slack], B = [−L | G]. Rank 118 (117 from the connected
  incidence + the slack column).

### combustion (tier 1; parallel worker, reviewed here)
- **Setup.** Methane/air with air = O2 0.21 / N2 0.78 / Ar 0.01 by mole, GRI-Mech 3.0 (53 species).
- **Sampling.** T0 U(1000, 1600) K, P U(1, 20) atm, phi U(0.5, 2.0) (seed 20240611).
- **Reactor.** Adiabatic constant-pressure (IdealGasConstPressureReactor) advanced to a fixed 1.0 ms, rtol 1e-10,
  atol 1e-20. 0 failed integrations.
- **Why 1 ms.** Ignition delays at the corners of the range run from ~3.6e-5 s to ~1.6 s. At 1 ms the samples are
  11,421 unignited, 395 igniting and 8,184 burnt (by temperature progress).
- **Inputs.** The fresh-mixture mass fractions Y0 of CH4, O2, N2 and Ar are inputs alongside T0, P and phi. This
  keeps the element balance linear in X with a fixed A, since the initial element fractions are nonlinear in phi.
- **Equality.** 5 element rows (O, H, C, N, Ar) with A_ek = n_ek·W_e/W_k and B = −A_e on the Y0 columns, plus
  1ᵀY = 1. The sum row is the sum of the element rows (rank 5 of 6 rows); the harness drops it.
- **Bounds.** 0 ≤ Y ≤ 1. 16 label entries are at −1e-31-scale round-off and were left unchanged.
- **Thermo range.** 5 samples end slightly above 3000 K (max 3004.4 K), just outside the GRI thermo fit range.
  They were kept.

### county_population (tier 2; parallel worker, reviewed here)
- **Data.** Census county totals, Vintage 2020 (co-est2020-alldata.csv) and Vintage 2025 (co-est2025-alldata.csv),
  SUMLEV 50 only. A sample is (county, year t), always within a single vintage.
- **Years.** Partial base periods (April–July 2010 and April–July 2020) are excluded as targets and as lags, so the
  targets are 2012–2020 and 2022–2025. The V2020 year 2019→2020 is kept as target 2020.
- **Inputs.** Year t−1 population and its 8 components; no IDs.
- **Outputs.** Year t BIRTHS, DEATHS, NATURALCHG, INTERNATIONALMIG, DOMESTICMIG, NETMIG, RESIDUAL, NPOPCHG and
  POPESTIMATE.
- **Equalities.** Four rows: natural change, net migration, population change, and
  pop_t − npopchg_t − pop_{t−1} = 0. All four hold exactly (residual 0) in every county-year.
- **Split.** By time: train ≤ 2020, val 2022–2023, test 2024–2025. Connecticut appears as its 8 counties in V2020 and
  as its 9 planning regions in V2025, both as published.

### water_network (tier 1; parallel worker, reviewed here)
- **Network.** EPANET Net3, bundled with the wntr package (92 junctions, 3 tanks, 2 reservoirs, 117 pipes, 2 pumps).
- **Simulator.** WNTRSimulator, double-precision flows, Newton TOL 1e-10. A failed solve would drop the sample; 0
  were dropped.
- **Operating state.** One demand-driven steady-state snapshot per sample. Tanks are held at fixed heads at their
  initial levels. All 18 controls are removed and link states fixed (pump 10 open, pump 335 open, pipe 330 closed:
  the state the original controls give while tank 1 fills), so that no pump is a structurally zero column.
- **Demands.** Base × g × u_j with g ~ U(0.6, 1.4) per sample and u_j ~ U(0.5, 1.5) per junction per sample;
  patterns removed. X = the actual demands (L/s) at the 59 junctions with non-zero base demand.
- **Equality.** Conservation at all 92 junctions (tanks and reservoirs excluded): A = signed incidence on junctions,
  B = −E. Rank 92 of q = 119 (0.773 < 0.8).
- **Zero outputs.** Pipe 330 (closed) and pipe 333 (dead end behind it) are always 0. They are kept and excluded from
  nRMSE as constant.
- **Bounds.** Pumps ≥ 0; pipe 330 fixed at 0.
