# Findings (overnight run, 2026-10-22)

Plain-language summary. Every number comes from `results/results.md`, `results/summary_stats.json`,
`results/validation.csv` or `results/phase4_summary.txt`. Test set = 2020–2026, models trained on 2006–2017.

1. **The formulation is correct on the literature instances.** All 7 (haverly1–3, bental4, foulds2, adhya1, rt2)
   match MINLPLib's best-known objective to within 1e-4.
2. **The real-data instances are easy for a global solver.** SCIP certified 9,866 of 9,898 instances (the other 32
   are proven infeasible). None came close to the 60 s limit: mean 0.87 s, max 11.6 s.
3. **They are still hard enough for a local solver to get stuck.** 12.2% of random-start IPOPT runs end at a worse
   local optimum, and 31.5% of instances have at least one such trap.
4. **Multistart is not a shortcut at this size.** The best of 20 random starts finds the global optimum on 99.7% of
   test instances, but takes 2.61 s on average versus 0.92 s for SCIP, which also proves optimality.
5. **Raw network predictions are never usable as solutions.** 0% of MLP outputs are feasible. Pool mass balance is
   off by 15.8 kbd on average, and quality specs by 0.185 (spec units, e.g. wt% S).
6. **The KKT projection layer only fixes what it is designed to fix.** It removes the mass-balance error (residual
   p95 7.6e-6 kbd, float32 rounding), but quality specs are still violated (mean 0.175). Because bounds aren't
   enforced, it also creates negative flows (up to 11.1 kbd). Still 0% feasible.
7. **On the main benchmark the MLP warm start helps, but only a little.** IPOPT from the MLP prediction reaches
   the global optimum on 91.0% of test instances, versus 89.5% for a single random start: a paired difference of
   +1.57 pp (95% CI +0.77 to +2.37). It is far below best-of-20 (99.7%) and SCIP (100%).
8. **Speed:** the warm start takes 0.125 s versus SCIP's 0.92 s, roughly 7× faster, but without any optimality
   guarantee. It fails on 299 of 3,333 test instances. All times were measured with 19 concurrent processes.
9. **Structure-first (method e) failed on the main benchmark.** Global hit rate was 2.9%. The classifier gets
   79.6% of individual arcs and specs right, but the full pattern on only 0.5% of instances. Fixing a wrong
   pattern makes the reduced problem infeasible (379 cases) or leaves it at a poor solution.
10. **On fixed networks (same 2 networks every day, only prices change) learning works much better.**
    - Warm start: 99.5% vs 94.9% for a random start (paired +4.65 pp, CI +4.06 to +5.25).
    - In the 2020 crash year: 98.2% vs 78.1%.
    - Structure-first reaches 80.4% (exact-pattern accuracy 76.6%), still below the plain warm start.
11. **Warm-start failures cluster where the optimal structure changes.** On the fixed networks, (d) failed on 13 of
    883 test days (1.47%) where the optimal structure differed from the previous test day, versus 3 of 2,455
    (0.12%) where it did not. On the main benchmark, failed warm starts had a slightly larger predicted-vs-optimal
    structure distance (7.35 vs 6.48 arcs).
12. **Difficulty swings over time.** On the fixed networks the monthly trap rate ranges from 0% to 40.5%
    (March 2012); on the main benchmark it ranges from 2.4% to 29.8% (plot 4). Random starts are weakest in 2020: 78.1% hit rate on the fixed networks, 83.2% on the main benchmark.
13. **Caveats:**
    - Crude prices are modeled. Brent plus an API differential only; the sulfur term was dropped because its fitted
      sign was wrong. So sour crudes are likely over-priced relative to reality,
      which weakens the incentive to blend them.
    - The fixed-network dataset is my addition beyond the original spec.
    - One seed, one training run each, no hyperparameter tuning.
    - The bulk of the result rests on an instance generator I designed tonight. It is not an industrial dataset.
