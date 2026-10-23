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
