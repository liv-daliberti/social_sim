#!/usr/bin/env python3
"""Generate Figure 2 (exp1_results.pdf) for the paper.
Run from the repo root:  python paper/generate_figures.py
Requires: matplotlib, numpy, pathlib
"""
import json
import math
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent / "exp1_prospective"
OUT   = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# ── colour / style constants ───────────────────────────────────────────────────
C_PRO   = "#16a34a"   # green
C_ANTI  = "#dc2626"   # red
C_ORTHO = "#7c3aed"   # purple
C_GPT   = "#2563eb"
C_CLAU  = "#9333ea"

DIR_COLOR  = {"pro_H1": C_PRO, "anti_H1": C_ANTI, "orthogonal": C_ORTHO}
DIR_LABEL  = {"pro_H1": "pro-$H_1$", "anti_H1": "anti-$H_1$", "orthogonal": "orthogonal"}
DIRECTIONS = ["pro_H1", "anti_H1", "orthogonal"]

MODELS   = ["gpt-5.4", "claude-opus-4-8"]
M_LABEL  = {"gpt-5.4": "GPT-5.4", "claude-opus-4-8": "Claude Opus 4.8"}
M_MARKER = {"gpt-5.4": "o", "claude-opus-4-8": "D"}
M_COLOR  = {"gpt-5.4": C_GPT, "claude-opus-4-8": C_CLAU}

# ── load all update records ────────────────────────────────────────────────────
uf_dir  = _ROOT / "data" / "updated_forecasts"
seen    = set()
records = []
for f in sorted(uf_dir.glob("*.jsonl"), reverse=True):
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        uid = r.get("update_id", "")
        if uid and uid not in seen:
            seen.add(uid)
            records.append(r)

# ── market metadata for Panel C ───────────────────────────────────────────────
MARKETS = [
    {"id": "pm_1971905_2026-06-05", "short": "Hormuz",        "mkt": 0.17},
    {"id": "pm_957019_2026-06-05",  "short": "Iran nuclear",   "mkt": 0.28},
    {"id": "pm_2270338_2026-06-05", "short": "Iran peace",     "mkt": 0.41},
    {"id": "pm_601825_2026-06-05",  "short": "Brazil (Santos)","mkt": 0.16},
    {"id": "pm_628955_2026-06-05",  "short": "CA (Steyer)",    "mkt": 0.22},
]
AGENT_P = {
    "gpt-5.4":        {"pm_1971905_2026-06-05": 0.10, "pm_957019_2026-06-05": 0.22,
                       "pm_2270338_2026-06-05": 0.07, "pm_601825_2026-06-05": 0.03,
                       "pm_628955_2026-06-05":  0.01},
    "claude-opus-4-8":{"pm_1971905_2026-06-05": 0.10, "pm_957019_2026-06-05": 0.11,
                       "pm_2270338_2026-06-05": 0.05, "pm_601825_2026-06-05": 0.01,
                       "pm_628955_2026-06-05":  0.10},
}

# ── figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 4.8))
gs  = fig.add_gridspec(1, 3, width_ratios=[1.4, 1.4, 1.0],
                       left=0.05, right=0.97, wspace=0.34,
                       bottom=0.13, top=0.88)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL A — Signed Δyes_prob by CF direction (the "smoking gun")
# ══════════════════════════════════════════════════════════════════════════════
ax_A = fig.add_subplot(gs[0])

rng = np.random.default_rng(42)
x_positions = {"pro_H1": 0, "anti_H1": 1, "orthogonal": 2}

# collect per (model, direction) deltas
for model in MODELS:
    for direction in DIRECTIONS:
        recs = [r for r in records
                if r.get("forecast_model") == model
                and r.get("direction") == direction
                and r.get("delta_yes_prob") is not None]
        if not recs:
            continue
        deltas = [r["delta_yes_prob"] for r in recs]
        x_base = x_positions[direction]

        # model offset: GPT slightly left, Claude slightly right
        offset = -0.14 if model == "gpt-5.4" else +0.14
        jitter = rng.uniform(-0.07, 0.07, len(deltas))
        xs = x_base + offset + jitter

        alpha = 0.75 if direction == "orthogonal" else 0.55
        size  = 22  if direction == "orthogonal" else 18
        ax_A.scatter(xs, deltas,
                     color=DIR_COLOR[direction],
                     marker=M_MARKER[model],
                     s=size, alpha=alpha, linewidths=0.3,
                     edgecolors="white", zorder=3)

        # mean line
        mean_d = np.mean(deltas)
        ax_A.plot([x_base + offset - 0.11, x_base + offset + 0.11],
                  [mean_d, mean_d],
                  color=DIR_COLOR[direction], linewidth=2.0,
                  solid_capstyle="round", zorder=4)

# zero line
ax_A.axhline(0, color="#9ca3af", linewidth=1.0, linestyle="--", zorder=2)

# direction zone shading (subtle)
for xi, direction in enumerate(DIRECTIONS):
    ax_A.axvspan(xi - 0.45, xi + 0.45,
                 color=DIR_COLOR[direction], alpha=0.04, zorder=0)

ax_A.set_xticks([0, 1, 2])
ax_A.set_xticklabels([DIR_LABEL[d] for d in DIRECTIONS], fontsize=9)
ax_A.set_ylabel(r"Forecast revision $\Delta\hat{p}$", fontsize=9)
ax_A.set_ylim(-0.32, 0.58)
ax_A.yaxis.set_major_formatter(lambda v, _: f"{v:+.2f}" if v != 0 else "0")
ax_A.tick_params(labelsize=8)
ax_A.set_xlim(-0.5, 2.5)

# annotation: "EHC = HFC = 100%"
ax_A.text(0.97, 0.97,
          "EHC = HFC = 100%\nacross all updates",
          transform=ax_A.transAxes, ha="right", va="top",
          fontsize=7, color="#374151",
          bbox=dict(fc="white", ec="#d1d5db", lw=0.7, pad=3, alpha=0.92))

# model legend
h = [mlines.Line2D([], [], color="#666", marker=M_MARKER[m],
                   linestyle="none", markersize=5, label=M_LABEL[m])
     for m in MODELS]
ax_A.legend(handles=h, fontsize=7, loc="upper left",
            framealpha=0.85, handletextpad=0.4)

ax_A.set_title("(A)  Direction-selective updating", fontsize=9,
               fontweight="bold", pad=6, loc="left")
ax_A.grid(axis="y", color="#e5e7eb", linewidth=0.6, zorder=0)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL B — Initial → Updated scatter (before/after)
# ══════════════════════════════════════════════════════════════════════════════
ax_B = fig.add_subplot(gs[1])

lo, hi = 0.0, 0.60

# diagonal (no change)
ax_B.plot([lo, hi], [lo, hi], color="#9ca3af", linewidth=0.9,
          linestyle="--", zorder=1, label="No change")

for r in records:
    iyp = r.get("initial_yes_prob")
    uyp = r.get("updated_yes_prob")
    d   = r.get("direction", "")
    if iyp is None or uyp is None or d not in DIR_COLOR:
        continue
    ax_B.scatter(iyp, uyp,
                 color=DIR_COLOR[d], s=15, alpha=0.45,
                 linewidths=0.2, edgecolors="white", zorder=2)

# direction legend patches
patches = [mpatches.Patch(color=DIR_COLOR[d], label=DIR_LABEL[d], alpha=0.85)
           for d in DIRECTIONS]
ax_B.legend(handles=patches, fontsize=7.5, loc="upper left",
            framealpha=0.88, handlelength=0.9)

ax_B.set_xlabel("Initial forecast $\\hat{p}_0$",  fontsize=9)
ax_B.set_ylabel("Updated forecast $\\hat{p}'$", fontsize=9)
ax_B.set_xlim(lo, hi); ax_B.set_ylim(lo, hi)
ax_B.xaxis.set_major_formatter(lambda v, _: f"{v:.1f}")
ax_B.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}")
ax_B.tick_params(labelsize=8)
ax_B.grid(color="#e5e7eb", linewidth=0.6, zorder=0)

# region labels
ax_B.text(0.72, 0.12, "anti-$H_1$\nupdates", transform=ax_B.transAxes,
          ha="center", va="center", fontsize=7, color=C_ANTI, alpha=0.8,
          style="italic")
ax_B.text(0.20, 0.82, "pro-$H_1$\nupdates", transform=ax_B.transAxes,
          ha="center", va="center", fontsize=7, color=C_PRO, alpha=0.8,
          style="italic")

ax_B.set_title("(B)  Before / after forecast revision", fontsize=9,
               fontweight="bold", pad=6, loc="left")

# ══════════════════════════════════════════════════════════════════════════════
# PANEL C — Agent forecast vs. Polymarket price (rules out market imitation)
# ══════════════════════════════════════════════════════════════════════════════
ax_C = fig.add_subplot(gs[2])

lo2, hi2 = 0.0, 0.55
ax_C.plot([lo2, hi2], [lo2, hi2], color="#9ca3af", linewidth=1.0,
          linestyle="--", zorder=1)

for model in MODELS:
    xs = [mkt["mkt"] for mkt in MARKETS]
    ys = [AGENT_P[model][mkt["id"]] for mkt in MARKETS]
    ax_C.scatter(xs, ys, color=M_COLOR[model], s=52, zorder=3,
                 marker=M_MARKER[model], label=M_LABEL[model], alpha=0.9)

# draw vertical drop lines from each market price to show gap
for mkt in MARKETS:
    x_m = mkt["mkt"]
    y_gpt  = AGENT_P["gpt-5.4"][mkt["id"]]
    y_clau = AGENT_P["claude-opus-4-8"][mkt["id"]]
    ax_C.plot([x_m, x_m], [min(y_gpt, y_clau) - 0.004, x_m],
              color="#d1d5db", linewidth=1.0, linestyle=":", zorder=0)

# label all markets (use midpoint between models for placement)
label_offsets = {
    "pm_1971905_2026-06-05": ( 5,  4),
    "pm_957019_2026-06-05":  ( 5, -11),
    "pm_2270338_2026-06-05": ( 5,  4),
    "pm_601825_2026-06-05":  (-8, -11),
    "pm_628955_2026-06-05":  ( 5,  4),
}
for mkt in MARKETS:
    y_mid = (AGENT_P["gpt-5.4"][mkt["id"]] + AGENT_P["claude-opus-4-8"][mkt["id"]]) / 2
    dx, dy = label_offsets.get(mkt["id"], (5, 4))
    ax_C.annotate(mkt["short"],
                  xy=(mkt["mkt"], y_mid),
                  xytext=(dx, dy), textcoords="offset points",
                  fontsize=6.2, color="#374151")

# shaded "below market" region
ax_C.fill_between([lo2, hi2], [lo2, hi2], [lo2, lo2],
                  color="#fee2e2", alpha=0.18, zorder=0,
                  label="Below market")
ax_C.text(0.62, 0.12, "agents below\nmarket price",
          transform=ax_C.transAxes, ha="center", va="center",
          fontsize=7, color="#b91c1c", alpha=0.85, style="italic")

ax_C.set_xlabel("Polymarket mid-price", fontsize=9)
ax_C.set_ylabel("Agent median $\\hat{p}$", fontsize=9)
ax_C.set_xlim(lo2, hi2); ax_C.set_ylim(lo2, hi2)
ax_C.xaxis.set_major_formatter(lambda v, _: f"{v:.1f}")
ax_C.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}")
ax_C.tick_params(labelsize=8)
ax_C.legend(fontsize=7.5, loc="upper left", framealpha=0.88, handletextpad=0.4)
ax_C.grid(color="#e5e7eb", linewidth=0.6, zorder=0)
ax_C.set_title("(C)  Not imitating the market", fontsize=9,
               fontweight="bold", pad=6, loc="left")

# ── save ───────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    path = OUT / f"exp1_results.{ext}"
    fig.savefig(path, bbox_inches="tight", dpi=300 if ext == "pdf" else 150)
    print(f"Saved {path}")
