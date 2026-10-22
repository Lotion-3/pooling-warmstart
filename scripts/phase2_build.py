"""Phase 2: price model + real-data instances.

  python -P scripts/phase2_build.py [--design data/instances/design.json] [--samples 2] [--seed 20261022]

Writes results/price_model.json, results/plots/price_model.png,
data/instances/instances.parquet (one row per (date, sample)), data/instances/design.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import realdata as R  # noqa: E402
from io_utils import write_parquet  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--design", default=None)
ap.add_argument("--samples", type=int, default=2)
ap.add_argument("--seed", type=int, default=20261022)
ap.add_argument("--fixed-networks", action="store_true",
                help="sample each network (crudes, topology, specs) ONCE per sample index and reuse it on every date")
args = ap.parse_args()

design = R.DEFAULT_DESIGN if args.design is None else json.loads(Path(args.design).read_text())
crudes = R.load_crudes()
prices = R.load_prices()
model, obs = R.fit_price_model(crudes, prices)
(ROOT / "results" / "plots").mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "price_model.json").write_text(json.dumps(model, indent=1))

# plot: yearly API slope with 2 SE, and band differentials vs fit
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
yrs = sorted(model["yearly"], key=int)
a = [model["yearly"][y]["a"] for y in yrs]; se = [model["yearly"][y]["se_a"] for y in yrs]
ax[0].errorbar([int(y) for y in yrs], a, yerr=2 * np.array(se), fmt="o-", capsize=3)
ax[0].axhline(model["pooled"]["a"], ls="--", c="gray", label="pooled")
ax[0].set(title="API differential a_y ($/bbl per °API), ±2 SE", xlabel="year"); ax[0].legend()
mean_diff = obs.groupby("api")["diff"].mean()
ax[1].plot(mean_diff.index, mean_diff.values - mean_diff.mean(), "o", label="EIA landed − Brent (demeaned, 2006–2026 avg)")
xs = np.linspace(15, 50, 10)
ax[1].plot(xs, model["pooled"]["a"] * (xs - R.API_REF) - model["pooled"]["a"] * (mean_diff.index.values - R.API_REF).mean(),
           label="pooled fit")
ax[1].set(title="Differential by API band", xlabel="band midpoint API", ylabel="$/bbl"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(ROOT / "results" / "plots" / "price_model.png", dpi=120)

# instances
t0 = time.time()
rows = []
master = np.random.SeedSequence(args.seed)
dates = prices.index
children = master.spawn(len(dates) * args.samples)
fixed_nets = None
if args.fixed_networks:
    fixed_rngs = [np.random.default_rng(c) for c in np.random.SeedSequence(args.seed + 1).spawn(args.samples)]
    fixed_nets = [R.sample_instance(crudes, r, design) for r in fixed_rngs]
prefix = "F" if args.fixed_networks else ""
for di, (date, prow) in enumerate(prices.iterrows()):
    for s in range(args.samples):
        if fixed_nets is None:
            net = R.sample_instance(crudes, np.random.default_rng(children[di * args.samples + s]), design)
        else:
            net = fixed_nets[s]
        iid = f"{prefix}{date.date()}_s{s}"
        inst = R.build_instance(crudes, net, prow, date, model, name=iid)
        row = {"inst_id": iid, "date": date, "sample": s,
               "crudes": inst.meta["crudes"], "crude_cost": inst.meta["cost"].tolist(),
               "crude_worth": inst.meta["worth"].tolist(),
               "crude_api": crudes.api[net["crude_idx"]].tolist(), "crude_s": crudes.sulfur_pct[net["crude_idx"]].tolist(),
               "crude_tan": crudes.tan_mgkoh_g[net["crude_idx"]].tolist(),
               "cdu_specs": json.dumps(net["cdus"])}
        for k in R.CORE + ["propane_mont_belvieu_usd_per_gal"]:
            row[k] = float(prow[k])
        row.update(R.to_row(inst))
        rows.append(row)
df = pd.DataFrame(rows)
out = ROOT / "data" / "instances"
out.mkdir(parents=True, exist_ok=True)
write_parquet(df, out / ("instances_fixed.parquet" if args.fixed_networks else "instances.parquet"))
(out / "design.json").write_text(json.dumps(design, indent=1))
if fixed_nets is not None:
    (out / "fixed_networks.json").write_text(json.dumps(
        [{"crudes": crudes.crude_id[n["crude_idx"]].tolist(), **{k: (v.tolist() if hasattr(v, "tolist") else v)
          for k, v in n.items() if k != "crude_idx"}} for n in fixed_nets], indent=1))
print(f"{len(df)} instances from {len(dates)} dates x {args.samples} samples in {time.time()-t0:.0f}s; "
      f"price model fallback={model['fallback']} yearly={model['use_yearly']}")
