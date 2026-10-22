"""Build results/results.md and results/plots/*.png from the Phase 4/5 result files only."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from io_utils import read_parquet_dir  # noqa: E402

FEAS, HIT = 1e-5, 1e-4
PLOTS = ROOT / "results" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def rg(o, r):
    return (np.asarray(o, float) - r) / np.maximum(1.0, abs(r))


def method_rows(E: pd.DataFrame, clf: dict):
    inf = clf.get("inference_s_per_instance", {})
    R = {}
    # (a) single random start: every one of the 20 starts is a sample
    g = np.concatenate([rg(o, r) for o, r in zip(E.ip_obj, E.glob_obj)])
    v = np.concatenate(E.ip_viol.values)
    R["(a) IPOPT, random start (per start)"] = dict(hit=np.concatenate(E.ip_hit.values).mean(), gap=g, time=np.concatenate(E.ip_time.values),
                                                    viol=v, n=len(g))
    best = []
    for o, vv, r in zip(E.ip_obj, E.ip_viol, E.glob_obj):
        f = [x for x, y in zip(o, vv) if y <= FEAS]
        best.append(min(rg(f, r)) if f else np.nan)
    R["(a20) IPOPT, best of 20 random starts"] = dict(hit=np.mean([any(h) for h in E.ip_hit]), gap=np.array(best),
                                                      time=np.array([sum(t) for t in E.ip_time]), viol=None, n=len(E))
    for tag, name in [("b", "(b) MLP flows, raw prediction"), ("c", "(c) MLP + KKT projection, raw")]:
        if f"{tag}_gap" in E:
            R[name] = dict(hit=((E[f"{tag}_max_viol"] <= FEAS) & (E[f"{tag}_gap"] <= HIT)).mean(), gap=E[f"{tag}_gap"].values,
                           time=np.full(len(E), inf.get(tag, np.nan)), viol=E[f"{tag}_max_viol"].values, n=len(E))
    R["(d) MLP warm start -> IPOPT"] = dict(hit=E.d_hit.mean(), gap=E.d_gap.values, time=E.d_time.values + inf.get("b", 0),
                                            viol=E.d_viol.values, n=len(E))
    if "e_hit" in E:
        R["(e) structure-first (classifier + reduced IPOPT)"] = dict(
            hit=E.e_hit.mean(), gap=E.e_gap.values, time=E.e_solver_time.values + inf.get("e", 0) + inf.get("b", 0),
            viol=E.e_viol.values, n=len(E))
    R["(f) SCIP global (reference)"] = dict(hit=1.0, gap=np.zeros(len(E)), time=E.glob_time.values, viol=None, n=len(E))
    return R


def table(R):
    lines = ["| method | global-opt hit rate | mean gap | p95 gap | mean time (s) | constraint violation |",
             "|---|---|---|---|---|---|"]
    for k, r in R.items():
        g = r["gap"][np.isfinite(r["gap"])]
        if r["viol"] is None:
            vtxt = "feasible (SCIP)" if k.startswith("(f)") else "— (best feasible start)"
        else:
            vv = np.asarray(r["viol"])
            vtxt = f"{100*np.mean(vv > FEAS):.1f}% > 1e-5; median {np.median(vv):.2g}"
        lines.append(f"| {k} | {100*r['hit']:.1f}% | {np.mean(g):.4f} | {np.quantile(g, .95):.4f} | "
                     f"{np.mean(r['time']):.4f} | {vtxt} |")
    return "\n".join(lines)


def by_year(E, clf):
    rows = []
    for y, G in E.groupby("year"):
        R = method_rows(G, clf)
        row = {"year": y, "n": len(G)}
        for k, r in R.items():
            short = k.split(")")[0] + ")"
            row[f"hit {short}"] = f"{100*r['hit']:.1f}%"
        for k in ["(d) MLP warm start -> IPOPT", "(e) structure-first (classifier + reduced IPOPT)"]:
            if k in R:
                row[f"mean gap {k.split(')')[0]})"] = f"{np.nanmean(R[k]['gap']):.4f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df[[c for c in df.columns if not c.startswith("hit (f)")]]
    return df.to_markdown(index=False)


def viol_detail(E):
    out = ["| raw prediction | mass-balance viol (kbd) mean / p95 | quality-spec viol (spec units) mean / p95 | negative flow (kbd) max | other bound viol (kbd) mean |",
           "|---|---|---|---|---|"]
    for tag, name in [("b", "(b) MLP"), ("c", "(c) MLP + KKT projection")]:
        if f"{tag}_gap" not in E:
            continue
        mb, sv = E[f"{tag}_mass_balance_viol"], E[f"{tag}_spec_viol"]
        out.append(f"| {name} | {mb.mean():.3g} / {mb.quantile(.95):.3g} | {sv.mean():.3g} / {sv.quantile(.95):.3g} | "
                   f"{E[f'{tag}_neg_flow'].max():.3g} | {E[f'{tag}_other_viol'].mean():.3g} |")
    return "\n".join(out)


def main():
    md = ["# Results", "",
          "All numbers below are computed by `scripts/report.py` from `results/<dataset>/test_eval.parquet`, "
          "`results/<dataset>/classifier.json` and `data/solved/`. Test set = certified instances dated 2020–2026 "
          "(train 2006–2017, validation 2018–2019).", "",
          "- **hit**: final point feasible (max violation ≤ 1e-5) and objective within 1e-4 (relative, "
          "|ref| floored at 1) of SCIP's certified optimum.",
          "- **gap** = (obj − obj*) / max(1, |obj*|), signed, minimisation, so infeasible raw predictions can show "
          "negative gaps.",
          "- **time**: solver call only (Pyomo model build excluded for every method); NN inference time is added "
          "to (b)–(e). (a20) time is the sum of 20 starts.",
          "- **violation**: share of runs whose max violation (bounds, balance, specs) exceeds 1e-5.", ""]
    datasets = [d for d in ["varied", "fixed"] if (ROOT / "results" / d / "test_eval.parquet").exists()]
    allE = {}
    for ds in datasets:
        E = pd.read_parquet(ROOT / "results" / ds / "test_eval.parquet")
        clf = json.loads((ROOT / "results" / ds / "classifier.json").read_text())
        allE[ds] = (E, clf)
        R = method_rows(E, clf)
        title = {"varied": "Main benchmark — a new random network per (date, sample)",
                 "fixed": "Fixed-network benchmark — 2 networks reused on every date, prices vary"}[ds]
        md += [f"## {title} (`{ds}`)", "",
               f"Test instances: {len(E)} (train {clf['n']['train']}, val {clf['n']['val']}).", "", table(R), "",
               "Raw-prediction constraint violations:", "", viol_detail(E), "",
               f"Structure classifier on the test set (threshold 0.5, existing arcs + all specs): "
               f"element accuracy {100*clf['acc_all']:.1f}% (arcs {100*clf['acc_arcs']:.1f}%, binding specs "
               f"{100*clf['acc_specs']:.1f}%); exact full-pattern match {100*clf['exact_pattern']:.1f}%.", "",
               f"(e) reduced-problem IPOPT terminations: {E.e_term.value_counts().to_dict()}", "",
               "### By year (test)", "", by_year(E, clf), ""]
    (ROOT / "results" / "results.md").write_text("\n".join(md), encoding="utf-8")

    # ---- plot 1: hit rate by year
    fig, axs = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4), squeeze=False)
    for ax, ds in zip(axs[0], datasets):
        E, clf = allE[ds]
        yrs = sorted(E.year.unique())
        for key, lab, st in [("a", "(a) random start", "o-"), ("a20", "(a20) best of 20 random", "s--"),
                             ("d", "(d) MLP warm start", "^-"), ("e", "(e) structure-first", "v-")]:
            vals = []
            for y in yrs:
                R = method_rows(E[E.year == y], clf)
                k = [n for n in R if n.startswith(f"({key})")]
                vals.append(100 * R[k[0]]["hit"] if k else np.nan)
            ax.plot(yrs, vals, st, label=lab)
        ax.set(title=f"Global-optimum hit rate by year ({ds})", xlabel="year", ylabel="%", ylim=(0, 102))
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(PLOTS / "1_hit_rate_by_year.png", dpi=120); plt.close(fig)

    # ---- plot 2: time per instance (log)
    fig, axs = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4), squeeze=False)
    for ax, ds in zip(axs[0], datasets):
        E, clf = allE[ds]
        R = method_rows(E, clf)
        keys = [k for k in R if k[:3] in ("(a)", "(d)", "(e)", "(f)")] + [k for k in R if k.startswith("(a20)")]
        ax.boxplot([R[k]["time"] for k in keys], showfliers=False)
        ax.set_xticks(range(1, len(keys) + 1), [k.split(" ")[0] for k in keys])
        ax.set_yscale("log"); ax.set(title=f"Time per instance ({ds})", ylabel="seconds (log)")
    fig.tight_layout(); fig.savefig(PLOTS / "2_time_per_instance.png", dpi=120); plt.close(fig)

    # ---- plot 3: failure analysis
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for ds in datasets:
        E, _ = allE[ds]
        for ok, ls in [(True, "-"), (False, "--")]:
            h = E.loc[E.d_hit == ok, "ham_b_arcs"]
            if len(h):
                axs[0].hist(h, bins=range(0, 25), histtype="step", density=True, ls=ls,
                            label=f"{ds}: warm start {'hit' if ok else 'failed'} (n={len(h)})")
    axs[0].set(title="Arc-structure distance: MLP-implied (flow>1 kbd) vs optimal", xlabel="Hamming distance (arcs)",
               ylabel="density"); axs[0].legend(fontsize=8)
    txt = []
    if "fixed" in allE:
        E = allE["fixed"][0].sort_values(["sample", "date"]).copy()
        E["changed"] = E.groupby("sample")["true_struct"].transform(lambda s: s.ne(s.shift()))
        E.loc[E.groupby("sample").head(1).index, "changed"] = np.nan
        Z = E.dropna(subset=["changed"])
        grp = Z.groupby("changed")["d_hit"].agg(["mean", "count"])
        labels = ["structure same as previous test day", "structure changed vs previous test day"]
        vals = [100 * (1 - grp.loc[False, "mean"]) if False in grp.index else np.nan,
                100 * (1 - grp.loc[True, "mean"]) if True in grp.index else np.nan]
        cnt = [int(grp.loc[k, "count"]) if k in grp.index else 0 for k in (False, True)]
        axs[1].bar(labels, vals)
        for xx, (vv, cc) in enumerate(zip(vals, cnt)):
            axs[1].text(xx, (vv if np.isfinite(vv) else 0) + 0.5, f"n={cc}", ha="center")
        axs[1].set(title="Fixed networks: warm-start (d) failure rate", ylabel="% failed")
        axs[1].tick_params(axis="x", labelsize=8)
        txt.append(grp.to_dict())
    else:
        axs[1].text(0.1, 0.5, "fixed-network dataset not available", transform=axs[1].transAxes)
    fig.tight_layout(); fig.savefig(PLOTS / "3_failure_analysis.png", dpi=120); plt.close(fig)

    # ---- plot 4: trap rate over time (all solved instances, monthly)
    fig, ax = plt.subplots(figsize=(10, 4))
    trap_stats = {}
    for ds, parts, inst in [("varied", "parts", "instances.parquet"), ("fixed", "parts_fixed", "instances_fixed.parquet")]:
        S = read_parquet_dir(ROOT / "data" / "solved" / parts)
        if not len(S):
            continue
        I = pd.read_parquet(ROOT / "data" / "instances" / inst, columns=["inst_id", "date"])
        S = S.merge(I, on="inst_id")
        S = S[S.certified]
        mth = S.groupby(pd.to_datetime(S.date).dt.to_period("M"))["ip_trap_rate"].mean()
        ax.plot(mth.index.to_timestamp(), 100 * mth.values, label=f"{ds} (monthly mean, n={len(S)})", lw=1)
        trap_stats[ds] = {"n": int(len(S)), "mean_trap": float(S.ip_trap_rate.mean()),
                          "share_instances_trap>0": float((S.ip_trap_rate > 0).mean())}
    ax.set(title="Trap rate over time: share of random-start IPOPT runs ending at a worse feasible local optimum",
           ylabel="%"); ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS / "4_trap_rate_over_time.png", dpi=120); plt.close(fig)
    (ROOT / "results" / "summary_stats.json").write_text(json.dumps({"trap": trap_stats, "fixed_change": [str(t) for t in txt]}, indent=1))
    print("results.md and plots written")


if __name__ == "__main__":
    main()
