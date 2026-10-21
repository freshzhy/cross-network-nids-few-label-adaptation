#!/usr/bin/env python3
"""
plot_figures_rq4.py — Fig. 7 (target-label budget curves) and Fig. 8 (variant-held-out stress test).

Reads  results/rq4_budget/{curve_by_config,zero_shot_baseline}.csv and results/rq4_variant/summary_by_test.csv
Writes results/figures/Fig7_label_budget.{pdf,png}, Fig8_variant_holdout.{pdf,png}

  python code/plot_figures_rq4.py [--results results] [--out results/figures]
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--results", default=str(ROOT / "results"))
ap.add_argument("--out", default=None)
a = ap.parse_args()
R = Path(a.results); OUTD = Path(a.out) if a.out else R / "figures"; OUTD.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})

# ───────────────────────── Fig. 7: label budget ─────────────────────────
co = pd.read_csv(R / "rq4_budget" / "curve_by_config.csv", dtype={"budget": str})
zo = pd.read_csv(R / "rq4_budget" / "zero_shot_baseline.csv")
order = ["1", "2", "5", "10", "25", "50", "100", "250", "500", "1000", "All"]
ph = {"1", "2", "5"}
x = np.arange(len(order))
sty = {"Full-40": ("black", "-"), "Source-SHAP-10": ("tab:orange", "-"),
       "Target-SHAP-10": ("tab:blue", "--"), "Random-10": ("grey", ":")}
M = [("f1", "std", "F1 (1:10 test set)"), ("pr_auc", "nat", "PR-AUC (prevalence-matched)")]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), sharex=True)
for j, sem in enumerate(["DoS", "Reconnaissance"]):
    for i, (met, ts, lab) in enumerate(M):
        ax = axes[i][j]
        ax.axvspan(-0.5, 2.5, color="0.93", zorder=0, lw=0)
        ax.axvline(2.5, color="0.65", lw=0.6, ls="-", zorder=1)
        if i == 0:
            ax.text(1.0, 0.97, "post hoc*", ha="center", va="top", fontsize=6, color="0.4")
        for c, (col, ls) in sty.items():
            y = co[(co.semantic_class == sem) & (co.condition == c) & (co.testset == ts)].groupby("budget")[met].mean().reindex(order)
            ax.plot(x, y.values, color=col, ls=ls, marker="o", ms=2.5, lw=1.2, label=c, zorder=3)
        z = zo[(zo.semantic_class == sem) & (zo.condition == "SHAP-10-Direct") & (zo.testset == ts)][met].mean()
        ax.axhline(z, color="purple", ls="-.", lw=1, label="Zero-shot (SHAP-10-Direct)", zorder=2)
        ax.set_xticks(x); ax.set_xticklabels([b + ("*" if b in ph else "") for b in order])
        ax.set_ylim(0, 1); ax.set_title(f"{sem}", fontsize=8) if i == 0 else None
        if j == 0: ax.set_ylabel(lab)
        if i == 1: ax.set_xlabel("Labeled target attack flows")
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=5, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.2)
fig.tight_layout(rect=(0, 0, 1, 0.95))
for e in ("pdf", "png"):
    fig.savefig(OUTD / f"Fig7_label_budget.{e}", dpi=300, bbox_inches="tight")
plt.close(fig)

# ───────────────────────── Fig. 8: variant held-out ─────────────────────────
s = pd.read_csv(R / "rq4_variant" / "summary_by_test.csv", dtype={"budget": str})
ob = ["1", "2", "5", "10", "25", "50", "100", "All"]
xb = np.arange(len(ob))
xz0 = len(ob) + 0.15          # zero-shot strip to the right of the budget axis
groups = [("A_variant", "A. Leave-one-tool-out (CIC-IDS2018 DoS)"), ("B_family", "B. DoS↔DDoS family swap")]
MET = [("pr_auc", "PR-AUC on unseen variant (1:10)"), ("tpr_at_1fpr", "TPR at 1% FPR on unseen variant")]
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
legends = {}
for j, (kind, title) in enumerate(groups):
    ids = list(dict.fromkeys(s[s.kind == kind].test_id))
    if kind == "A_variant":
        cols = dict(zip(ids, plt.cm.tab10(np.arange(len(ids)))))
    else:  # CIC in blues/oranges, IoT distinguished by colour and dashes
        pal = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
        cols = dict(zip(ids, pal))
    for i, (met, lab) in enumerate(MET):
        ax = axes[i][j]
        ax.axvspan(len(ob) - 0.45, xz0 + 0.12 * len(ids) + 0.15, color="0.95", lw=0, zorder=0)
        for k, tid in enumerate(ids):
            g = s[(s.test_id == tid) & (s.metric == met)].set_index("budget").reindex(ob)
            iot = kind == "B_family" and not tid.startswith("CIC")
            name = tid.replace("LOVO:", "held-out ").replace(":", " ")
            ax.plot(xb, g["unseen"].values, color=cols[tid], ls="--" if iot else "-", marker="o", ms=2.5, lw=1.2, label=name, zorder=3)
            zv = g["zero_shot_best"].dropna().iloc[0]
            ax.plot([xz0 + 0.12 * k], [zv], marker="_", ms=8, mew=2.0, color=cols[tid], zorder=3)
        ax.set_xticks(xb); ax.set_xticklabels(ob)
        ax.set_xlim(-0.4, xz0 + 0.12 * len(ids) + 0.15)
        ax.text(xz0 + 0.06 * len(ids), 1.0, "zero-\nshot", ha="center", va="top", fontsize=5.5, color="0.35")
        ax.set_ylim(0, 1.04)
        if i == 0: ax.set_title(title, fontsize=8)
        if j == 0: ax.set_ylabel(lab)
        if i == 1: ax.set_xlabel("Labeled flows from seen variants")
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    hh, ll = axes[0][j].get_legend_handles_labels()
    if kind == "B_family":
        cic = [(h, l) for h, l in zip(hh, ll) if l.startswith("CIC")]
        iot = [(h, l) for h, l in zip(hh, ll) if not l.startswith("CIC")]
        hd = [Line2D([], [], color="none")]
        pad = (len(iot) + 1) - (len(cic) + 1)
        H = hd + [h for h, _ in cic] + hd * pad + hd + [h for h, _ in iot]
        L = ["CIC-IDS2018 (solid)"] + [l for _, l in cic] + [""] * pad + ["IoT networks (dashed)"] + [l for _, l in iot]
        axes[1][j].legend(H, L, fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=3)
    else:
        axes[1][j].legend(hh, ll, fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=3)
fig.tight_layout()
for e in ("pdf", "png"):
    fig.savefig(OUTD / f"Fig8_variant_holdout.{e}", dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"→ {OUTD}/Fig7_label_budget.*, Fig8_variant_holdout.*")
