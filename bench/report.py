"""Build bench/results/summary.md and bench/results/plots/ from the per-domain metrics.json files.

    python -P bench/report.py

Domains that never reached the gate (blocked loaders) are listed from bench/results/blocked.json.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from bench.domain import RESULTS  # noqa: E402

MODELS = "abcd"
NAMES = {"a": "(a) MLP", "b": "(b) soft penalty", "c": "(c) post-hoc projection", "d": "(d) KKT-hPINN"}
COLORS = {"a": "#2a78d6", "b": "#eb6834", "c": "#1baf7a", "d": "#eda100"}
NOTES = {"process_flowsheet": " (closest to the original paper)",
         "building_end_use": " (caveat: state set chosen after a failed draw; see DECISIONS.md)"}
SHORT = {"metabolic": "COBRApy e_coli_core, pFBA", "water_network": "WNTR, EPANET Net3",
         "dc_power_flow": "pandapower IEEE 118-bus, DC flow", "combustion": "Cantera GRI-Mech 3.0 reactor",
         "truss": "numpy truss FEM", "circuit": "numpy resistor network", "process_flowsheet": "IDAES flowsheet",
         "pharmacokinetics": "scipy 3-compartment PK", "epidemic": "scipy SEIRD",
         "building_end_use": "NREL ResStock 2025.1 (OEDI)", "bank_balance_sheets": "FDIC BankFind API",
         "food_balance": "FAOSTAT FBS", "company_financials": "SEC EDGAR FS data sets",
         "county_population": "Census county estimates", "heat_exchanger_network": "numpy HEN, eps-NTU"}
DW_COLOR = "#6250d6"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def g(x, d=4):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}g}"


def load_all():
    out = {}
    for p in sorted(RESULTS.glob("*/metrics.json")):
        m = json.loads(p.read_text())
        out[m["domain"]] = m
    blocked = json.loads((RESULTS / "blocked.json").read_text()) if (RESULTS / "blocked.json").exists() else {}
    return out, blocked


def bound_pct_from_preds(name, thr):
    """% of test points (mean over seeds) with a bound excess above thr x test std, from saved predictions."""
    f = RESULTS / name / "test_predictions.npz"
    if not f.exists():
        return None
    from bench.run import load_domain
    dom = load_domain(name)
    z = np.load(f)
    Y = z["Y"]
    sd = Y.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    lo = dom.y_lo if dom.y_lo is not None else np.full(Y.shape[1], -np.inf)
    hi = dom.y_hi if dom.y_hi is not None else np.full(Y.shape[1], np.inf)
    v = np.maximum(np.maximum(lo - Y, Y - hi), 0.0) / sd
    out = {"labels": float(100 * (v.max(1) > thr).mean())}
    for mdl in MODELS:
        pct = []
        for P in z[mdl].astype(np.float64):
            v = np.maximum(np.maximum(lo - P, P - hi), 0.0) / sd
            pct.append(100 * (v.max(1) > thr).mean())
        out[mdl] = float(np.mean(pct))
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelcolor=INK)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plots(ok):
    pdir = RESULTS / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    names = sorted(ok, key=lambda k: ok[k]["rank"]["ratio"])

    # 1. nRMSE(d)/nRMSE(a): 1 + (d - a)/a with the bootstrap CI of (d - a) scaled by a
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(names) + 1.4), facecolor=SURFACE)
    style(ax)
    abl = {k: json.loads((RESULTS / k / "ablation_weighted.json").read_text())
           for k in names if (RESULTS / k / "ablation_weighted.json").exists()}
    for i, k in enumerate(names):
        a = ok[k]["aggregate"]["a"]["nrmse_mean"]
        for key, src, col, dy, lab in (("d_minus_a", ok[k]["bootstrap"], COLORS["d"], -0.15, "(d) KKT-hPINN"),
                                       ("d_w_minus_a", abl.get(k, {}).get("bootstrap", {}), DW_COLOR, 0.15,
                                        "d_w: variance-weighted projection (ablation)")):
            if key not in src:
                continue
            bs = src[key]
            r, lo, hi = 1 + bs["diff"] / a, 1 + bs["ci_lo"] / a, 1 + bs["ci_hi"] / a
            ax.plot([max(lo, 1e-3), hi], [i + dy, i + dy], color=col, lw=2, solid_capstyle="round")
            ax.plot(r, i + dy, "o", ms=7, color=col, mec=SURFACE, mew=1.5, label=lab if i == 0 else None)
            ax.annotate(f"{r:.3g}", (hi, i + dy), xytext=(5, 0), textcoords="offset points", va="center",
                        color=INK2, fontsize=7)
    ax.axvline(1.0, color=INK2, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("nRMSE / nRMSE (a), log scale   (< 1: more accurate than the plain MLP)", color=INK)
    ax.set_title("Accuracy of the hard-constrained models relative to the plain MLP\n"
                 "(mean over seeds; bars: 95% bootstrap CI over test points)", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.4, -0.08 - 0.5 / len(names)),
              ncol=2, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(pdir / "1_rmse_ratio_d_over_a.png", dpi=150)
    plt.close(fig)

    # 2. equality violation per model per domain (log scale), mean relative violation
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(names) + 1.6), facecolor=SURFACE)
    style(ax)
    h = 0.19
    for j, mdl in enumerate(MODELS):
        vals = [max(ok[k]["aggregate"][mdl]["eq_viol_mean_mean"], 1e-17) for k in names]
        ax.barh(np.arange(len(names)) + (j - 1.5) * (h + 0.01), vals, height=h, color=COLORS[mdl],
                label=NAMES[mdl], left=1e-17)
    ax.axvline(1e-5, color=INK2, lw=1, ls="--")
    ax.text(1e-5, 1.0, " 1e-5", color=INK2, fontsize=8, va="bottom", transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_xlim(1e-17, None)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlabel("mean relative equality violation on the test set (log; floored at 1e-17)", color=INK)
    ax.set_title("Equality violation by model", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.12 - 0.6 / len(names)),
              ncol=4, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(pdir / "2_violation_by_model.png", dpi=150)
    plt.close(fig)

    # 3. (d) vs (c): relative difference vs constraint density rank/q
    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for j, k in enumerate(names):
        c = ok[k]["aggregate"]["c"]["nrmse_mean"]
        bs = ok[k]["bootstrap"]["d_minus_c"]
        x = ok[k]["rank"]["ratio"]
        y, lo, hi = 100 * bs["diff"] / c, 100 * bs["ci_lo"] / c, 100 * bs["ci_hi"] / c
        ax.plot([x, x], [lo, hi], color=COLORS["d"], lw=2, solid_capstyle="round")
        ax.plot(x, y, "o", ms=8, color=COLORS["d"], mec=SURFACE, mew=2)
        ax.annotate(k, (x, y), xytext=(6, 4 if j % 2 else -11), textcoords="offset points", fontsize=8, color=INK)
    ax.axhline(0, color=INK2, lw=1, ls="--")
    ax.set_xlabel("rank(A) / q   (share of output directions fixed by the constraints)", color=INK)
    ax.set_ylabel("(nRMSE(d) − nRMSE(c)) / nRMSE(c)  [%]", color=INK)
    ax.set_title("End-to-end training (d) vs projecting after training (c)\n"
                 "below 0: end-to-end more accurate; bars: 95% bootstrap CI", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(pdir / "3_d_vs_c.png", dpi=150)
    plt.close(fig)


def main():
    allm, blocked = load_all()
    allm.pop("toy", None)
    ok = {k: m for k, m in allm.items() if m["status"] == "ok"}
    failed = {k: m for k, m in allm.items() if m["status"] == "failed_gate"}
    degen = {k: m for k, m in allm.items() if m["status"] == "degenerate"}
    L = ["# Benchmark results: hard linear equality constraints across domains", "",
         f"Domains attempted: {len(allm) + len(blocked)}. Passed the gate and trained: {len(ok)}. "
         f"Failed the gate: {len(failed)}. Degenerate (rank(A) >= 0.8 q): {len(degen)}. "
         f"Blocked before the gate: {len(blocked)}.", "",
         "nRMSE = per-output test RMSE / test std, averaged over outputs; mean ± std over seeds. "
         "Violation = % of test points whose max relative equality residual exceeds 1e-5. "
         "Bounds = % of test points violating an output bound by more than 1e-5 x output std "
         "(— where the domain has no bounds). All numbers are generated by `bench/report.py` from "
         "`bench/results/<domain>/metrics.json`.", "",
         "| domain | source | n | rank(A)/q | gate residual | nRMSE (a) | nRMSE (b) | nRMSE (c) | nRMSE (d) "
         "| viol% a / b / c / d | bounds% a / b / c / d | seeds |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in sorted(ok, key=lambda k: (str(ok[k]["meta"].get("tier")), k)):
        m = ok[k]
        ag = m["aggregate"]
        gate = g(m["gate"]["resid_max"], 3) + (" (labels projected)" if m["gate"]["labels_projected"] else "")
        nr = [f"{ag[x]['nrmse_mean']:.4f} ± {ag[x]['nrmse_std']:.4f}" for x in MODELS]
        vi = " / ".join(g(ag[x]["eq_viol_pct_above_1e-5_mean"], 4) for x in MODELS)
        bd = (" / ".join(g(ag[x]["bound_pct_points_mean"], 4) for x in MODELS)
              if "bound_pct_points_mean" in ag["a"] else "—")
        src = SHORT.get(k, m["meta"].get("source", "")[:40])
        name = k + NOTES.get(k, "")
        L.append(f"| {name} | {src} | {m['n']} | {m['rank']['rank']}/{m['q']} = {m['rank']['ratio']:.3f} | {gate} | "
                 + " | ".join(nr) + f" | {vi} | {bd} | {len(m['per_seed'])} |")
    L += ["", "## Paired differences (bootstrap 95% CI over test points, mean over seeds)", "",
          "| domain | nRMSE(d) − nRMSE(a) [CI] | seeds with d < a | nRMSE(d) − nRMSE(c) [CI] | seeds with d < c "
          "| nRMSE(c) − nRMSE(a) [CI] | λ (b) | split |",
          "|---|---|---|---|---|---|---|---|"]
    for k in sorted(ok):
        m = ok[k]
        b = m["bootstrap"]
        fmt = lambda x: f"{x['diff']:+.4f} [{x['ci_lo']:+.4f}, {x['ci_hi']:+.4f}]"
        S = len(m["per_seed"])
        L.append(f"| {k} | {fmt(b['d_minus_a'])} | {m['paired']['d_minus_a']['n_seeds_negative']}/{S} | "
                 f"{fmt(b['d_minus_c'])} | {m['paired']['d_minus_c']['n_seeds_negative']}/{S} | {fmt(b['c_minus_a'])} | "
                 f"{m['lambda']['chosen']} | {m['split']['desc']} |")
    L += ["", "## Pooled nRMSE in original units (||P - Y||_F / ||Y - mean||_F; mean ± std over seeds)", "",
          "nRMSE above weights every output equally after dividing by its std. The pooled version weights outputs "
          "by their size in original units, which is the metric the Euclidean projection is guaranteed not to "
          "worsen for (c).", "",
          "| domain | (a) | (b) | (c) | (d) |", "|---|---|---|---|---|"]
    for k in sorted(ok):
        ag = ok[k]["aggregate"]
        L.append(f"| {k} | " + " | ".join(f"{ag[x]['pooled_nrmse_mean']:.4g} ± {ag[x]['pooled_nrmse_std']:.2g}"
                                          for x in MODELS) + " |")
    L += ["", "## Bound violations", "",
          "% of test points with any output outside its bounds by more than 1e-5 x std (as in the main table) and "
          "by more than 1e-2 x std (computed here from the saved test predictions), and the largest excess / std "
          "(max over seeds). Bounds are never enforced by any model.", "",
          "| domain | labels % (> 1e-5 / > 1e-2 std) | % > 1e-2 std: a / b / c / d | max excess / std: a / b / c / d |",
          "|---|---|---|---|"]
    for k in sorted(ok):
        ag = ok[k]["aggregate"]
        if "bound_max_rel_std" not in ag["a"]:
            continue
        lb = ok[k]["label_bounds_test"]
        coarse = bound_pct_from_preds(k, 1e-2)
        L.append(f"| {k} | {g(lb['pct_points'] if lb else None)} / {g(coarse['labels']) if coarse else 'n/a'} | "
                 + (" / ".join(g(coarse[x], 4) for x in MODELS) if coarse else "n/a") + " | "
                 + " / ".join(g(ag[x]["bound_max_rel_std"], 3) for x in MODELS) + " |")
    L += ["", "## Max equality violation on the test set (max over seeds, relative)", "",
          "| domain | a | b | c | d |", "|---|---|---|---|---|"]
    for k in sorted(ok):
        ag = ok[k]["aggregate"]
        L.append(f"| {k} | " + " | ".join(g(ag[x]["eq_viol_max_over_seeds"], 3) for x in MODELS) + " |")
    L += ["", "## Training time (s per fit, mean over seeds)", "", "| domain | a | b | c (a + projection) | d |",
          "|---|---|---|---|---|"]
    for k in sorted(ok):
        ag = ok[k]["aggregate"]
        L.append(f"| {k} | " + " | ".join(f"{ag[x]['train_seconds_mean']:.1f}" for x in MODELS) + " |")
    abl = {k: json.loads((RESULTS / k / "ablation_weighted.json").read_text())
           for k in ok if (RESULTS / k / "ablation_weighted.json").exists()}
    if abl:
        L += ["", "## Ablation beyond the main protocol: variance-weighted projection", "",
              "c_w / d_w use the same KKT projection, but find the closest feasible point in STANDARDISED units "
              "(weights = training variance of each output) instead of original units. Constraint satisfaction is "
              "identical. Same seeds, split and backbone as (c) / (d). See DECISIONS.md.", "",
              "| domain | nRMSE (a) | (c) | c_w | (d) | d_w | d_w − a [CI] | d_w − d [CI] | pooled (a) / c_w / d_w "
              "| viol% d_w |", "|---|---|---|---|---|---|---|---|---|---|"]
        fmt = lambda x: f"{x['diff']:+.4f} [{x['ci_lo']:+.4f}, {x['ci_hi']:+.4f}]"
        for k in sorted(abl):
            ag, aw = ok[k]["aggregate"], abl[k]["aggregate"]
            L.append(f"| {k} | {ag['a']['nrmse_mean']:.4g} | {ag['c']['nrmse_mean']:.4g} | {aw['c_w']['nrmse_mean']:.4g} "
                     f"| {ag['d']['nrmse_mean']:.4g} | {aw['d_w']['nrmse_mean']:.4g} ± {aw['d_w']['nrmse_std']:.2g} | "
                     f"{fmt(abl[k]['bootstrap']['d_w_minus_a'])} | {fmt(abl[k]['bootstrap']['d_w_minus_d'])} | "
                     f"{ag['a']['pooled_nrmse_mean']:.4g} / {aw['c_w']['pooled_nrmse_mean']:.4g} / "
                     f"{aw['d_w']['pooled_nrmse_mean']:.4g} | {g(aw['d_w']['eq_viol_pct_above_1e-5_mean'])} |")
        missing = sorted(set(ok) - set(abl))
        if missing:
            L.append(f"\nAblation not run (time): {', '.join(missing)}.")
    L += ["", "## Domains dropped", ""]
    if not (failed or degen or blocked):
        L.append("None.")
    for k, m in sorted(failed.items()):
        gt = m["gate"]
        L.append(f"- **{k}**: failed the gate. Max relative residual {g(gt['resid_max'], 3)}, mean {g(gt['resid_mean'], 3)}, "
                 f"p99 {g(gt['resid_p99'], 3)}, {100 * gt['frac_above_1e-6']:.2f}% of samples above 1e-6"
                 + (f" (after label projection {g(gt.get('resid_max_after'), 3)})" if gt["labels_projected"] else "")
                 + ". No results reported.")
    for k, m in sorted(degen.items()):
        L.append(f"- **{k}**: degenerate, rank(A) = {m['rank']['rank']} of q = {m['q']} outputs "
                 f"({m['rank']['ratio']:.3f} >= 0.8). No results reported.")
    for k, why in sorted(blocked.items()):
        L.append(f"- **{k}**: not run. {why}")
    (RESULTS / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    if ok:
        plots(ok)
    print(f"wrote {RESULTS / 'summary.md'}: {len(ok)} trained, {len(failed)} failed gate, "
          f"{len(degen)} degenerate, {len(blocked)} blocked")


if __name__ == "__main__":
    main()
