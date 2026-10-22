# Decisions log (overnight run, 2026-10-22)

Every assumption, deviation and adjustment made during the unattended run. Times are local (EDT).

## Phase 0 / 0.5 — setup (02:16–02:25)

- **Preflight:** git identity set (Laksh Agarwal <lakshagarwal65@gmail.com>). `poolingOptim/` held only
  `pooling_real/`, no root `.git`. 14 physical cores / 20 logical. OneDrive process not running.
- **Global solver: SCIP 9 via `pyscipopt` 6.3.0.** `gurobipy` is not installed, so the license check failed;
  per instructions, SCIP is used and Gurobi is not installed.
- **Local solver: IPOPT 3.14 from the IDAES binary bundle** (`idaes get-extensions`, v3.4.2, includes MUMPS/HSL
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
