#!/usr/bin/env python3
"""Generate Figure 2 (exp1_results.pdf) for the paper.
Run from the repo root:  python paper/generate_figures.py
Requires: matplotlib, numpy, pathlib
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent / "exp1_prospective"
OUT   = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────────────
report = json.loads((_ROOT / "data/results/consistency_report_2026-06-08.json").read_text())

MODELS   = ["gpt-5.4", "claude-opus-4-8"]
M_LABELS = {"gpt-5.4": "GPT-5.4", "claude-opus-4-8": "Claude Opus 4.8"}
M_COLOR  = {"gpt-5.4": "#2563eb", "claude-opus-4-8": "#7c3aed"}

DIRECTIONS = ["pro_H1", "anti_H1", "orthogonal"]
D_LABELS   = ["pro-$H_1$", "anti-$H_1$", "orthogonal"]

# market table for scatter
MARKETS = [
    {"id": "pm_1971905_2026-06-05", "label": "Hormuz\n(June)",    "mkt": 0.17},
    {"id": "pm_957019_2026-06-05",  "label": "Iran nuclear\ndeal","mkt": 0.28},
    {"id": "pm_2270338_2026-06-05", "label": "Iran peace\ndeal",  "mkt": 0.41},
    {"id": "pm_601825_2026-06-05",  "label": "Brazil\n(Santos)",  "mkt": 0.16},
    {"id": "pm_628955_2026-06-05",  "label": "California\n(Steyer)","mkt": 0.22},
]

# agent medians from the report
AGENT_P = {
    "gpt-5.4":        {"pm_1971905_2026-06-05": 0.10, "pm_957019_2026-06-05": 0.22,
                       "pm_2270338_2026-06-05": 0.07, "pm_601825_2026-06-05": 0.03,
                       "pm_628955_2026-06-05":  0.01},
    "claude-opus-4-8":{"pm_1971905_2026-06-05": 0.10, "pm_957019_2026-06-05": 0.11,
                       "pm_2270338_2026-06-05": 0.05, "pm_601825_2026-06-05": 0.01,
                       "pm_628955_2026-06-05":  0.10},
}

# ── figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13.5, 4.5))
# Three panels: pipeline (A), bar chart (B), scatter (C)
gs  = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.6, 1.4],
                       left=0.04, right=0.97, wspace=0.32, bottom=0.14, top=0.88)

# ── Panel A: pipeline schematic ────────────────────────────────────────────────
ax_pipe = fig.add_subplot(gs[0])
ax_pipe.set_xlim(0, 10); ax_pipe.set_ylim(0, 10)
ax_pipe.axis("off")

STAGES = [
    ("Stage 1", "Structured\nForecast\n(k=5 runs)"),
    ("Stage 2", "Counterfactual\nPacket\nConstruction\n(9 per market)"),
    ("Stage 3", "Evidence\nInjection\n& Update\n(45 per model)"),
    ("Stage 4", "Consistency\nEvaluation\n(EHC/HFC/ICS)"),
]
ys = [8.2, 6.0, 3.8, 1.6]
box_w, box_h = 3.8, 1.5
x_center = 5.0
COLORS = ["#dbeafe", "#ede9fe", "#dcfce7", "#fef9c3"]
BORDER  = ["#2563eb", "#7c3aed", "#15803d", "#ca8a04"]

for i, ((stage, label), yc) in enumerate(zip(STAGES, ys)):
    fc, ec = COLORS[i], BORDER[i]
    rect = mpatches.FancyBboxPatch(
        (x_center - box_w/2, yc - box_h/2), box_w, box_h,
        boxstyle="round,pad=0.15", linewidth=1.5,
        facecolor=fc, edgecolor=ec,
    )
    ax_pipe.add_patch(rect)
    ax_pipe.text(x_center, yc + 0.52, stage, ha="center", va="center",
                 fontsize=7.5, fontweight="bold", color=ec)
    ax_pipe.text(x_center, yc - 0.10, label, ha="center", va="center",
                 fontsize=6.5, color="#374151", linespacing=1.35)
    # arrows
    if i < len(STAGES) - 1:
        ax_pipe.annotate("", xy=(x_center, ys[i+1] + box_h/2 + 0.06),
                         xytext=(x_center, yc - box_h/2 - 0.06),
                         arrowprops=dict(arrowstyle="->", color="#6b7280", lw=1.2))

ax_pipe.set_title("(A) Pipeline", fontsize=9, fontweight="bold", pad=6, loc="left")

# ── Panel B: bar chart of mean |Δ| by direction ────────────────────────────────
ax_bar = fig.add_subplot(gs[1])

n_dirs  = len(DIRECTIONS)
n_mod   = len(MODELS)
bar_w   = 0.32
x       = np.arange(n_dirs)

for i, model in enumerate(MODELS):
    anc = report["per_model"][model]["anchoring"]
    means = [anc[d]["mean_abs_delta"] * 100 for d in DIRECTIONS]  # convert to pp
    ses   = [(anc[d]["se"] or 0)    * 100 for d in DIRECTIONS]
    offset = (i - (n_mod - 1) / 2) * (bar_w + 0.04)
    bars = ax_bar.bar(
        x + offset, means, bar_w,
        color=M_COLOR[model], alpha=0.85,
        label=M_LABELS[model],
        zorder=3,
    )
    ax_bar.errorbar(
        x + offset, means, yerr=ses,
        fmt="none", color="black", capsize=3, linewidth=1.1, zorder=4,
    )

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(D_LABELS, fontsize=8.5)
ax_bar.set_ylabel(r"Mean $|\Delta\hat{p}|$ (pp)", fontsize=8.5)
ax_bar.set_ylim(0, 16)
ax_bar.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}")
ax_bar.tick_params(labelsize=8)
ax_bar.legend(fontsize=7.5, loc="upper right", framealpha=0.85)
ax_bar.set_title("(B) Anchoring baseline", fontsize=9, fontweight="bold", pad=6, loc="left")
ax_bar.axhline(0, color="#d1d5db", linewidth=0.7)
ax_bar.grid(axis="y", color="#e5e7eb", linewidth=0.6, zorder=0)

# annotate sensitivity ratios
for i, model in enumerate(MODELS):
    sens = report["per_model"][model]["anchoring"]["sensitivity_ratio"]
    color = M_COLOR[model]
    ax_bar.text(0.98, 0.97 - i * 0.12,
                f"{M_LABELS[model]}: {sens:.1f}×",
                transform=ax_bar.transAxes,
                ha="right", va="top", fontsize=7, color=color,
                bbox=dict(fc="white", ec=color, lw=0.7, pad=2, alpha=0.9))

# ── Panel C: scatter agent vs market ──────────────────────────────────────────
ax_sc = fig.add_subplot(gs[2])

lo, hi = 0.0, 0.55
ax_sc.plot([lo, hi], [lo, hi], color="#9ca3af", linewidth=1, linestyle="--", zorder=1,
           label="Agent = Market")

for model in MODELS:
    xs = [mkt["mkt"] for mkt in MARKETS]
    ys = [AGENT_P[model][mkt["id"]] for mkt in MARKETS]
    ax_sc.scatter(xs, ys, color=M_COLOR[model], s=55, zorder=3,
                  label=M_LABELS[model], alpha=0.9)

# label points (use first model's label positions)
for mkt in MARKETS:
    x_m = mkt["mkt"]
    y_a = AGENT_P["gpt-5.4"][mkt["id"]]
    ax_sc.annotate(
        mkt["label"],
        xy=(x_m, y_a),
        xytext=(4, 5), textcoords="offset points",
        fontsize=5.5, color="#374151", linespacing=1.2,
    )

ax_sc.set_xlabel("Polymarket mid-price", fontsize=8.5)
ax_sc.set_ylabel(r"Agent median $\hat{p}$", fontsize=8.5)
ax_sc.set_xlim(lo, hi); ax_sc.set_ylim(lo, hi)
ax_sc.xaxis.set_major_formatter(lambda v, _: f"{v:.1f}")
ax_sc.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}")
ax_sc.tick_params(labelsize=8)
ax_sc.legend(fontsize=7.5, loc="upper left", framealpha=0.85)
ax_sc.set_title("(C) Agent vs. Market price", fontsize=9, fontweight="bold", pad=6, loc="left")
ax_sc.grid(color="#e5e7eb", linewidth=0.6, zorder=0)

# ── save ───────────────────────────────────────────────────────────────────────
out_pdf = OUT / "exp1_results.pdf"
out_png = OUT / "exp1_results.png"
fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
fig.savefig(out_png, bbox_inches="tight", dpi=150)
print(f"Saved {out_pdf}")
print(f"Saved {out_png}")
