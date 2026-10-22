# Results

All numbers below are computed by `scripts/report.py` from `results/<dataset>/test_eval.parquet`, `results/<dataset>/classifier.json` and `data/solved/`. Test set = certified instances dated 2020–2026 (train 2006–2017, validation 2018–2019).

- **hit**: final point feasible (max violation ≤ 1e-5) and objective within 1e-4 (relative, |ref| floored at 1) of SCIP's certified optimum.
- **gap** = (obj − obj*) / max(1, |obj*|), signed, minimisation, so infeasible raw predictions can show negative gaps.
- **time**: solver call only (Pyomo model build excluded for every method); NN inference time is added to (b)–(e). (a20) time is the sum of 20 starts.
- **violation**: share of runs whose max violation (bounds, balance, specs) exceeds 1e-5.

## Main benchmark — a new random network per (date, sample) (`varied`)

Test instances: 3333 (train 5544, val 989).

| method | global-opt hit rate | mean gap | p95 gap | mean time (s) | constraint violation |
|---|---|---|---|---|---|
| (a) IPOPT, random start (per start) | 89.5% | 0.0021 | 0.0021 | 0.1305 | 0.0% > 1e-5; median 1.2e-06 |
| (a20) IPOPT, best of 20 random starts | 99.7% | 0.0000 | 0.0000 | 2.6102 | — (best feasible start) |
| (b) MLP flows, raw prediction | 0.0% | 0.1367 | 0.4379 | 0.0005 | 96.5% > 1e-5; median 9.4 |
| (c) MLP + KKT projection, raw | 0.0% | 0.1292 | 0.4104 | 0.0005 | 99.1% > 1e-5; median 8.3 |
| (d) MLP warm start -> IPOPT | 91.0% | 0.0014 | 0.0012 | 0.1248 | 0.0% > 1e-5; median 1.2e-06 |
| (e) structure-first (classifier + reduced IPOPT) | 2.9% | 0.1169 | 0.4871 | 0.1257 | 9.3% > 1e-5; median 8.6e-07 |
| (f) SCIP global (reference) | 100.0% | 0.0000 | 0.0000 | 0.9177 | feasible (SCIP) |

Raw-prediction constraint violations:

| raw prediction | mass-balance viol (kbd) mean / p95 | quality-spec viol (spec units) mean / p95 | negative flow (kbd) max | other bound viol (kbd) mean |
|---|---|---|---|---|
| (b) MLP | 15.8 / 30.7 | 0.185 / 0.73 | 0 | 11.9 |
| (c) MLP + KKT projection | 3.05e-06 / 7.63e-06 | 0.175 / 0.677 | 11.1 | 10.2 |

Structure classifier on the test set (threshold 0.5, existing arcs + all specs): element accuracy 79.6% (arcs 76.3%, binding specs 88.3%); exact full-pattern match 0.5%.

(e) reduced-problem IPOPT terminations: {'optimal': 2950, 'infeasible': 379, 'maxIterations': 3, 'internalSolverError': 1}

### By year (test)

|   year |   n | hit (a)   | hit (a20)   | hit (b)   | hit (c)   | hit (d)   | hit (e)   |   mean gap (d) |   mean gap (e) |
|-------:|----:|:----------|:------------|:----------|:----------|:----------|:----------|---------------:|---------------:|
|   2020 | 497 | 83.2%     | 99.2%       | 0.0%      | 0.0%      | 85.3%     | 1.6%      |         0.0044 |         0.2724 |
|   2021 | 492 | 88.5%     | 99.8%       | 0.0%      | 0.0%      | 90.9%     | 1.8%      |         0.001  |         0.0954 |
|   2022 | 486 | 91.6%     | 99.8%       | 0.0%      | 0.0%      | 92.6%     | 3.7%      |         0.0008 |         0.0893 |
|   2023 | 487 | 90.7%     | 100.0%      | 0.0%      | 0.0%      | 92.8%     | 3.7%      |         0.0006 |         0.0866 |
|   2024 | 490 | 88.3%     | 99.0%       | 0.0%      | 0.0%      | 90.4%     | 2.0%      |         0.0004 |         0.1135 |
|   2025 | 487 | 91.8%     | 100.0%      | 0.0%      | 0.0%      | 91.8%     | 4.9%      |         0.0015 |         0.0651 |
|   2026 | 394 | 92.8%     | 100.0%      | 0.0%      | 0.0%      | 94.2%     | 2.8%      |         0.001  |         0.0877 |

### Paired comparison and failure analysis (test)

- (d) − (a) hit rate, paired over 3333 instances: +1.57 pp (95% CI +0.77 to +2.37 pp).
- Instances where every random start hits: 72.8%; where (d) fails: 299; of those, instances where the average random start also fails ≥ 50%: 138.
- (d) is faster than one random start on 77.0% of instances (mean 0.1242 s vs 0.1305 s).
- Arc-structure Hamming distance of the MLP-implied active set (flow > 1 kbd) to the optimum: mean 6.48 when (d) hits vs 7.35 when it fails; classifier distance 5.78 vs 7.13.

## Fixed-network benchmark — 2 networks reused on every date, prices vary (`fixed`)

Test instances: 3340 (train 5568, val 990).

| method | global-opt hit rate | mean gap | p95 gap | mean time (s) | constraint violation |
|---|---|---|---|---|---|
| (a) IPOPT, random start (per start) | 94.9% | 0.0002 | 0.0001 | 0.1265 | 0.0% > 1e-5; median 1.3e-06 |
| (a20) IPOPT, best of 20 random starts | 100.0% | 0.0000 | 0.0000 | 2.5307 | — (best feasible start) |
| (b) MLP flows, raw prediction | 0.0% | 0.0091 | 0.0400 | 0.0001 | 88.5% > 1e-5; median 0.78 |
| (c) MLP + KKT projection, raw | 0.0% | 0.0040 | 0.0387 | 0.0003 | 98.7% > 1e-5; median 1.2 |
| (d) MLP warm start -> IPOPT | 99.5% | 0.0000 | 0.0000 | 0.1199 | 0.0% > 1e-5; median 1.4e-06 |
| (e) structure-first (classifier + reduced IPOPT) | 80.4% | 0.0012 | 0.0052 | 0.1116 | 0.0% > 1e-5; median 7.4e-07 |
| (f) SCIP global (reference) | 100.0% | 0.0000 | 0.0000 | 0.6059 | feasible (SCIP) |

Raw-prediction constraint violations:

| raw prediction | mass-balance viol (kbd) mean / p95 | quality-spec viol (spec units) mean / p95 | negative flow (kbd) max | other bound viol (kbd) mean |
|---|---|---|---|---|
| (b) MLP | 1.85 / 5.6 | 0.00537 / 0.0296 | 0 | 1.42 |
| (c) MLP + KKT projection | 3.52e-06 / 7.63e-06 | 0.00618 / 0.0431 | 4.34 | 2.01 |

Structure classifier on the test set (threshold 0.5, existing arcs + all specs): element accuracy 98.4% (arcs 98.7%, binding specs 97.8%); exact full-pattern match 76.6%.

(e) reduced-problem IPOPT terminations: {'optimal': 3308, 'infeasible': 32}

### By year (test)

|   year |   n | hit (a)   | hit (a20)   | hit (b)   | hit (c)   | hit (d)   | hit (e)   |   mean gap (d) |   mean gap (e) |
|-------:|----:|:----------|:------------|:----------|:----------|:----------|:----------|---------------:|---------------:|
|   2020 | 498 | 78.1%     | 100.0%      | 0.0%      | 0.0%      | 98.2%     | 37.1%     |         0.0001 |         0.0056 |
|   2021 | 492 | 95.9%     | 99.8%       | 0.0%      | 0.0%      | 99.6%     | 89.8%     |         0      |         0.0005 |
|   2022 | 486 | 99.1%     | 100.0%      | 0.0%      | 0.0%      | 100.0%    | 85.0%     |         0      |         0.0005 |
|   2023 | 488 | 98.0%     | 100.0%      | 0.0%      | 0.0%      | 99.6%     | 90.4%     |         0      |         0.0002 |
|   2024 | 492 | 95.5%     | 100.0%      | 0.0%      | 0.0%      | 99.6%     | 89.4%     |         0      |         0.0002 |
|   2025 | 488 | 99.1%     | 100.0%      | 0.0%      | 0.0%      | 99.8%     | 95.1%     |         0      |         0.0002 |
|   2026 | 396 | 99.7%     | 100.0%      | 0.0%      | 0.0%      | 100.0%    | 76.0%     |         0      |         0.0011 |

### Paired comparison and failure analysis (test)

- (d) − (a) hit rate, paired over 3340 instances: +4.65 pp (95% CI +4.06 to +5.25 pp).
- Instances where every random start hits: 88.7%; where (d) fails: 16; of those, instances where the average random start also fails ≥ 50%: 8.
- (d) is faster than one random start on 81.7% of instances (mean 0.1198 s vs 0.1265 s).
- Arc-structure Hamming distance of the MLP-implied active set (flow > 1 kbd) to the optimum: mean 0.92 when (d) hits vs 2.94 when it fails; classifier distance 0.42 vs 2.81.
- Optimal structure unchanged vs previous test day (same network): n = 2455, (d) failures = 3 (0.12%), (e) failures = 390 (15.89%), random-start trap rate 2.79%.
- Optimal structure changed vs previous test day (same network): n = 883, (d) failures = 13 (1.47%), (e) failures = 263 (29.78%), random-start trap rate 11.50%.
- Distinct optimal structures in the test period: {0: 35, 1: 27} (per fixed network).
