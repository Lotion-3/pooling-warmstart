# poolingOptim — real-data crude-blending pooling benchmark

A pooling-problem benchmark built from public data:
- **crude assays:** Equinor and BP
- **prices:** EIA daily spot prices and monthly import costs
- **design:** a refinery crude-blending network — crudes → storage tanks → two crude distillation units

Every instance is solved to certified global optimality with SCIP. The project then tests whether neural networks
can predict warm starts (or the optimal active structure) that lead IPOPT, a fast local solver, to the global optimum.

- Results: [results/results.md](results/results.md)
- Plain-language findings: [findings.md](findings.md)
- Every assumption and adjustment: [DECISIONS.md](DECISIONS.md)
- Data provenance and access audit: [data_report.md](data_report.md)

## Layout

| path | content |
|---|---|
| `data/raw/` | downloads exactly as fetched; never modified (`access_log.jsonl` records every request) |
| `data/clean/` | `prices_daily.csv`, `crudes.csv`, `pooling_instances.csv` (built by `scripts/clean.py`) |
| `data/instances/` | generated instances: `instances.parquet` (new network per date/sample), `instances_fixed.parquet` (2 networks reused on every date), `design.json`, `fixed_networks.json` |
| `data/solved/` | SCIP + IPOPT-multistart results (`parts/`, `parts_fixed/` batch files; `solved.parquet`, `solved_fixed.parquet` consolidated) |
| `src/pooling.py` | general pooling model: pq / p formulations (Pyomo), SCIP global model, evaluator |
| `src/gamslib_pool.py` | reader for the GAMS-library literature instances (validation) |
| `src/realdata.py` | yields, price model, network sampling, instance (de)serialisation |
| `src/runner.py` | parallel, batched, resumable solver driver (Phase 3/4) |
| `src/learn.py`, `src/evalwork.py` | features, models (MLP, KKT projection, classifier), per-instance evaluation |
| `scripts/` | one script per phase; `report.py` builds `results/results.md` and `results/plots/` |

## Reproduce from raw data

Requirements: Python 3.12 with `pandas numpy openpyxl xlrd pyarrow pyomo pyscipopt torch scikit-learn matplotlib
tabulate`. IPOPT comes from the IDAES binaries:

1. `pip install idaes-pse`
2. `idaes get-extensions`
3. Copy the `idaes/bin` folder somewhere the OS loader can see it, and point the `IDAES_BIN` env var at it
   (default `~/solvers/idaes_bin`; see DECISIONS.md for why this is needed with the Microsoft Store Python).

Then run the steps in order. Steps 5–7 take about 1 h total on 19 worker processes.

1. `python -P scripts/clean.py`: raw → `data/clean/`.
2. `python -P scripts/phase1_validate.py`: literature instances → `results/validation.csv`.
3. `python -P scripts/phase2_build.py`: price model and 9,898 instances (seed 20261022).
4. `python -P scripts/phase2_build.py --fixed-networks`: fixed-network variant (9,898 instances).
5. `python -P src/runner.py --out results/phase3/round0 --ids results/phase3/smoke_ids_round0.txt`: smoke test.
6. `python -P src/runner.py --out data/solved/parts --log logs/phase4.log`: Phase 4, resumable.
7. `python -P src/runner.py --instances data/instances/instances_fixed.parquet --out data/solved/parts_fixed --log logs/phase4_fixed.log`
8. `python -P scripts/consolidate.py`: batch files → `data/solved/solved*.parquet`.
9. `python -P scripts/phase5.py --dataset varied`, then the same with `--dataset fixed`.
10. `python -P scripts/report.py`: writes `results/results.md` and the plots.

Steps 7–10 are chained in `scripts/pipeline_after_phase4.sh`, which waits for step 6 to finish.

Parallel timings depend on the machine: 14 cores / 20 threads, 19 workers.

## Data sources and licences

- **U.S. Energy Information Administration (EIA):** daily spot prices (`pet_pri_spt_s1_d`) and monthly
  landed / F.O.B. import costs. U.S. government work in the public domain.
  Cite as *"Source: U.S. Energy Information Administration (Oct 2026)"*: spot-price release 2026-10-21,
  monthly tables through 2026-07.
- **Equinor crude oil assays** (equinor.com/energy/crude-oil-assays): *"for guidance only and Equinor accepts no
  liability for any loss occurring from the use of these assays and errors that they may contain."*
- **BP crude assays** (bp.com, technical downloads): *"provided for informational purposes only… bp makes no
  representations or warranties… regarding the accuracy… bp accepts no liability in connection with any of such
  information."*
- **MINLPLib** (minlplib.org): CC BY 4.0. Cite MINLPLib and the original instance papers (Haverly 1978; Ben-Tal
  et al. 1994; Foulds et al. 1992; Adhya et al. 1999; Audet et al. 2000).
- **GAMS Model Library, model `pool`** (SEQ=254, M. Tawarmalani): literature instance data used only to validate
  the formulation.
- **eCFR 40 CFR Part 1090:** public domain. Downloaded during the access audit; finished-fuel specs are deliberately
  not used as crude-feed constraints.

**Important caveat:** individual crude prices are **modeled** (Brent + an API-gravity differential fitted to EIA
monthly landed costs), not observed. Crude values come from assay yields priced at USGC product spot prices, with
residue at 0.85 × Brent. See DECISIONS.md.
