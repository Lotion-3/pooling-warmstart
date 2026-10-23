# Hard linear equality constraints in neural networks, across domains

A benchmark suite for neural-network surrogates whose outputs must satisfy known linear equalities

    A · y + B · x = b        (A fixed, x = inputs, y = outputs)

such as mass, charge, energy or accounting balances. Every domain plugs into one harness (`bench/run.py`), so all
domains are compared with the same backbone, splits, metrics and statistics.

**Results:** [results/summary.md](results/summary.md) (all tables), [findings.md](findings.md) (plain-language
findings), [DECISIONS.md](DECISIONS.md) (every judgement call), plots in [results/plots/](results/plots/).

## Credit and scope

The method under test is the KKT projection layer of **Chen, Constante-Flores & Li (2024), "Physics-informed neural
networks with hard linear equality constraints"** (KKT-hPINN). The layer maps a network's prediction to the
closest point that satisfies the linear equalities exactly, using the closed-form KKT solution of that projection,
and the network is trained end to end through it. That paper demonstrated the layer on Aspen-based surrogates of a
CSTR, an extractive distillation subsystem and a chemical plant. This suite applies the same layer to domains
outside those case studies, from metabolism and power grids to economic accounting identities. The aim is to
describe where exact satisfaction of linear equalities comes with gains or costs in accuracy.

The layer is designed for linear equality constraints. Inequality constraints (for example nonnegative flows)
are outside its scope. Bound violations are reported here only to show where inequality-aware methods, such as the
decision-rule approach of Constante-Flores, Chen & Li (NeurIPS 2025, not implemented here), would apply.

## What is compared

All four models share the same MLP backbone (3 × 256, ReLU, Adam, early stopping on validation):

| model | description |
|---|---|
| (a) | plain MLP |
| (b) | MLP + soft penalty λ‖(A y + B x − b)/scale‖², λ ∈ {0.1, 1, 10} tuned on validation |
| (c) | model (a) with its test predictions projected by the KKT formula (no end-to-end training) |
| (d) | KKT-hPINN: projection layer y = ŷ − Aᵀ(AAᵀ)⁻¹(Aŷ + Bx − b), trained end to end |

Training runs in float32. De-standardisation, the projection and all metrics run in float64, and the projection
always acts on outputs in original units.

**Before any training, the harness checks every domain:**

- **Gate.** The identity must hold on the data: max relative residual ≤ 1e-6, measured against the typical magnitude
  of each row's terms. Between 1e-6 and 1e-4, the labels are projected and the projection is recorded; above 1e-4,
  the domain is dropped.
- **Degeneracy check.** Redundant rows are removed. If rank(A) ≥ 0.8·q, the constraints nearly determine the
  outputs and the domain is dropped as degenerate.

**Metrics** (test set):

- normalised RMSE (per-output RMSE / std, averaged)
- equality violation (max, mean, % of points above 1e-5)
- bound violations
- training time
- paired per-seed differences and bootstrap 95% CIs for (d) − (a), (d) − (c) and (c) − (a)

## Domains

All 14 domains below were attempted. Whether each one passed is listed in [results/summary.md](results/summary.md).

| domain | tier | data | license |
|---|---|---|---|
| metabolic | 1 (simulated) | COBRApy e_coli_core (bundled), pFBA fluxes; S·v = 0 | COBRApy LGPL/GPL; model from BiGG (Orth et al. 2010), free for research |
| water_network | 1 | WNTR + EPANET example network; flow conservation at junctions | WNTR revised BSD; EPANET example networks public domain (US EPA) |
| dc_power_flow | 1 | pandapower IEEE 118-bus case; DC power flow; nodal balance | pandapower BSD-3; IEEE 118 test case public test data |
| combustion | 1 | Cantera + GRI-Mech 3.0, constant-pressure reactor; element conservation | Cantera BSD-3; GRI-Mech 3.0 freely available (UC Berkeley) |
| truss | 1 | numpy FEM, 12× statically indeterminate 2D truss; joint equilibrium | generated here (MIT) |
| circuit | 1 | numpy nodal analysis, meshed resistor network; KCL | generated here (MIT) |
| process_flowsheet | 1 | IDAES mixer–flash–splitter, no reactions; component balances (closest to the original paper) | IDAES-PSE BSD-style (US DOE notice) |
| pharmacokinetics | 1 | scipy 3-compartment PK model; mass balance vs cumulative dose | generated here (MIT) |
| epidemic | 1 | scipy SEIRD model; S + E + I + R + D = N | generated here (MIT) |
| building_end_use | 2 (real) | NREL ResStock annual results (OEDI); end uses sum to totals | NREL / OEDI open data, see DECISIONS.md |
| bank_balance_sheets | 2 | FDIC BankFind financials; assets = liabilities + equity | U.S. government data, public domain |
| food_balance | 2 | FAOSTAT Food Balance Sheets; supply = utilisation | FAO, CC BY 4.0 |
| company_financials | 2 | SEC EDGAR financial statement data sets; balance-sheet identity | U.S. government data, public domain |
| county_population | 2 | Census county population estimates; components of change | U.S. government data, public domain |

Real-data files are **not** stored in this repository. Each tier-2 domain has a download script in
`bench/scripts/get_<domain>.py` that writes to `bench/data/raw/<domain>/` (gitignored).

## Reproduce

Requirements: Python 3.12 with `numpy scipy pandas pyarrow torch matplotlib cobra wntr pandapower cantera
idaes-pse`. For process_flowsheet you also need IPOPT from the IDAES binaries (`idaes get-extensions`), on the path
given by `IDAES_BIN` (default `~/solvers/idaes_bin`).

```
python -P bench/tests/test_harness.py              # harness tests on a toy domain with a known answer
python -P bench/scripts/get_county_population.py   # one per real-data domain
python -P bench/run.py <domain> --gate-only        # gate + rank check only
python -P bench/run.py <domain>                    # full run -> bench/results/<domain>/metrics.json, summary.csv
python -P bench/report.py                          # summary.md and plots
```

Simulated domains generate their samples on first use (fixed seeds) and cache them in `bench/data/cache/`
(gitignored).

## Add a domain

Create `bench/domains/<name>.py` with a function `load()` that returns a `bench.domain.Domain`:

- `X` (n × p) and `Y` (n × q), in original units.
- The equality as `A` (m × q, fixed, it must not depend on the sample), `B` (m × p) and `b` (m).
- Optionally, `split` and `split_kind` (`"time"` or `"group"`); without them the harness uses a random 70/15/15
  split.
- Optionally, `y_lo` and `y_hi` (fixed bounds, used for reporting only).
- `meta` with source, license, generation, units and tier.

Then run `python -P bench/run.py <name> --gate-only` and, if the gate passes, the full run.
