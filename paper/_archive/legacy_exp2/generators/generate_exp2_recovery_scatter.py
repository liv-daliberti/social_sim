#!/usr/bin/env python3
"""Clean, compact Experiment 2 recovery figure — two panels STACKED in one column.

Top:    recovery — inferred ĝ vs true g (Bayes hugs the diagonal; shortcuts are
        flat reference lines).
Bottom: cost     — next-poll MAE vs g (Bayes is the flat low envelope; each
        shortcut blows up on the half of worlds where it is wrong).

Both share the x-axis (true g), so they stack cleanly and the pair occupies a
single column. Styled to match the Experiment 1 figures (Nimbus Sans, larger
text, no in-figure title — caption lives in LaTeX). Vector PDF + high-DPI PNG.

Run from repo root:
    python paper/generate_exp2_recovery_scatter.py
"""
from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

_BN = Path(__file__).resolve().parent.parent / "exp2_simulated_worlds" / "biased_news"
_OUT = Path(__file__).resolve().parent / "figures"
sys.path.insert(0, str(_BN))
from engine.news_response import generate_episode, compute_bayes, predict_fixed_gain

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Nimbus Sans", "DejaVu Sans", "Helvetica", "Arial"],
    "font.size":         14,
    "axes.labelsize":    16,
    "xtick.labelsize":   13,
    "ytick.labelsize":   13,
    "legend.fontsize":   11.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.axisbelow":    True,
    "grid.color":        "#eef2f7",
    "grid.linewidth":    0.8,
    "axes.linewidth":    1.0,
    "savefig.bbox":      "tight",
})

C_DIAG = "#111827"; C_FACE = "#dc2626"; C_IGN = "#2563eb"; C_BAYES = "#f59e0b"

# data (fixed seed → reproducible): recovery cloud + MAE-vs-g curves
rng = random.Random(7)
tg, gh = [], []
mae = {"bayes": [], "face": [], "ignore": []}
gs = []
for _ in range(2500):
    ep = generate_episode(seed=rng.randint(0, 2**31))
    b = compute_bayes(ep["news"], ep["polls"], ep["test_news"])
    tg.append(ep["g"]); gh.append(b["g_hat"]); gs.append(ep["g"])
    gold, lp, tn = ep["gold_poll"], ep["last_poll"], ep["test_news"]
    mae["bayes"].append(abs(b["predicted_poll"] - gold))
    mae["face"].append(abs(predict_fixed_gain(lp, tn, 1.0) - gold))
    mae["ignore"].append(abs(predict_fixed_gain(lp, tn, 0.0) - gold))


def binned(g, y, nb=10, lo=0.1, hi=1.0):
    e = np.linspace(lo, hi, nb + 1); cx, cy = [], []
    for k in range(nb):
        v = [yy for gg, yy in zip(g, y) if e[k] <= gg < e[k + 1] or (k == nb - 1 and gg == hi)]
        if v:
            cx.append((e[k] + e[k + 1]) / 2); cy.append(statistics.mean(v))
    return cx, cy


fig, (axT, axB) = plt.subplots(2, 1, figsize=(6.2, 6.0), sharex=True,
                               gridspec_kw={"height_ratios": [1.6, 1], "hspace": 0.12})

# ── top: recovery ──
axT.plot([0, 1.05], [0, 1.05], color=C_DIAG, lw=1.8, ls=(0, (1, 2)), zorder=2,
         label=r"perfect recovery ($\hat{g}=g$)")
axT.axhline(1.0, color=C_FACE, lw=2.3, ls=(0, (6, 2)), zorder=2, label=r"face value ($\hat{g}=1$)")
axT.axhline(0.0, color=C_IGN,  lw=2.3, ls=(0, (6, 2)), zorder=2, label=r"ignore news ($\hat{g}=0$)")
# subsample the dots so the cloud reads cleanly
sub = list(range(0, len(tg), 5))
axT.scatter([tg[i] for i in sub], [gh[i] for i in sub], s=20, color=C_BAYES, alpha=0.65,
            edgecolor="white", linewidth=0.25, zorder=3, label="Bayes oracle (infers $g$)")
axT.set_ylabel(r"Inferred  $\hat{g}$")
axT.set_ylim(-0.18, 1.30)
axT.set_yticks([0, 0.5, 1.0])

# single shared legend for BOTH panels (colors are consistent across them),
# placed in the top panel's upper-left corner
shared = [
    mlines.Line2D([], [], color=C_BAYES, lw=2.2, marker="o", ms=6, label="Bayes oracle (infers $g$)"),
    mlines.Line2D([], [], color=C_FACE, lw=2.3, ls=(0, (6, 2)), label=r"face value ($\hat{g}=1$)"),
    mlines.Line2D([], [], color=C_IGN,  lw=2.3, ls=(0, (6, 2)), label=r"ignore news ($\hat{g}=0$)"),
    mlines.Line2D([], [], color=C_DIAG, lw=1.8, ls=(0, (1, 2)), label=r"perfect recovery ($\hat{g}=g$)"),
]
axT.legend(handles=shared, loc="upper left", framealpha=0.96, edgecolor="#d1d5db",
           handlelength=2.2, labelspacing=0.4, borderpad=0.6)

# ── bottom: forecast cost ──
for key, col, lab in [("ignore", C_IGN, "ignore news"), ("face", C_FACE, "face value"),
                      ("bayes", C_BAYES, "Bayes oracle")]:
    cx, cy = binned(gs, mae[key])
    axB.plot(cx, cy, color=col, lw=2.4, marker="o", ms=5, zorder=3, label=lab)
axB.set_ylabel("Next-poll MAE")
axB.set_xlabel(r"True news-responsiveness  $g$   (low = biased)")
axB.set_ylim(0, None)
axB.set_xlim(0, 1.05)
axB.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

for ax in (axT, axB):
    ax.tick_params(length=5, width=1.0)

fig.tight_layout()
for ext, dpi in (("pdf", 300), ("png", 600)):
    p = _OUT / f"exp2_recovery_stacked.{ext}"
    fig.savefig(p, dpi=dpi)
    print(f"  Saved {p}")
plt.close(fig)
