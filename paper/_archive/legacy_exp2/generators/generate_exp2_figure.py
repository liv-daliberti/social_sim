#!/usr/bin/env python3
"""Generate the Experiment 2 figure for the paper.

exp2_latent_inference.pdf — a 3×2 grid:
  rows  = context variant   V0 (no context) / V1 (vague hint) / V2 (named biased pollster)
  cols  = city condition     Neutral-media / Biased-media

Each panel shows, against three fixed reference strategies, where each agent's
next-survey forecast error (MAE, lower = better) falls:
  • Naïve    — ignore the news, predict ≈ the last poll      (gray)
  • Reactive — react to ALL news at full strength, never infer the bias (red)
  • Bayes    — infer the hidden media bias and react accordingly (amber, optimal)

The point: the *optimal fixed strategy flips by condition* — react in neutral
cities, ignore the news in biased cities — so neither Naïve nor Reactive wins
everywhere, but Bayes wins both because it INFERS the latent. An agent that
matches Bayes has inferred the bias; one stuck between Naïve and Reactive has not.

The V1/V2 rows test whether *leaking* information about the bias (a vague hint /
a named sticky pollster) helps the agent infer it. Panels with no data yet render
as "run pending" and fill in automatically when those variants are evaluated.

Reads exp2_simulated_worlds/biased_news/data/batch/results_*.jsonl
(output of eval/run_batch.py). Run from repo root:
    python paper/generate_exp2_figure.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# ── paths / engine (for the recomputed Reactive baseline) ─────────────────────
_BN    = Path(__file__).resolve().parent.parent / "exp2_simulated_worlds" / "biased_news"
_BATCH = _BN / "data" / "batch"
_OUT   = Path(__file__).resolve().parent / "figures"
_OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(_BN))
from engine.temporal_dag import compute_blind_forecast   # noqa: E402

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Nimbus Sans", "DejaVu Sans", "Helvetica", "Arial"],
    "font.size":         12,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.axisbelow":    True,
    "grid.color":        "#e5e7eb",
    "grid.linewidth":    0.6,
    "axes.linewidth":    0.8,
    "legend.fontsize":   10,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

C_QWEN  = "#615CED"
C_LLAMA = "#1877F2"
C_FRONT = "#ea580c"
C_NAIVE = "#6b7280"   # ignore news  (floor in neutral, optimal in biased)
C_REACT = "#dc2626"   # react to all news (optimal in neutral, fails in biased)
C_BAYES = "#d97706"   # infer the bias (optimal in both)

MODELS = [
    ("qwen2.5-7b",     "Qwen 7B",        "qwen"),
    ("llama3.1-8b",    "Llama 3.1-8B",   "llama"),
    ("qwen2.5-14b",    "Qwen 14B",       "qwen"),
    ("qwen2.5-32b",    "Qwen 32B",       "qwen"),
    ("llama3.3-70b",   "Llama 3.3-70B",  "llama"),
    ("llama3.1-70b",   "Llama 3.1-70B",  "llama"),
    ("qwen2.5-72b",    "Qwen 72B",       "qwen"),
    ("DeepSeek-V4-Pro","DeepSeek V4",    "frontier"),
    ("gpt-5.4",        "GPT-5.4",        "frontier"),
    ("claude-opus-4-8","Claude Opus 4.8","frontier"),
]
FAMILY_COLOR = {"qwen": C_QWEN, "llama": C_LLAMA, "frontier": C_FRONT}
LABEL  = {m[0]: m[1] for m in MODELS}
FAMILY = {m[0]: m[2] for m in MODELS}
ORDER  = [m[0] for m in MODELS]

VARIANTS = [
    ("V0", "V0 · no context"),
    ("V1", "V1 · vague hint"),
    ("V2", "V2 · named biased pollster"),
]
CONDITIONS = [("neutral", "Neutral-media city"), ("biased", "Biased-media city")]


# ── data ──────────────────────────────────────────────────────────────────────
def load_records() -> list[dict]:
    latest: dict[tuple, dict] = {}
    for path in sorted(_BATCH.glob("results_*.jsonl")):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                latest[(rec["episode_id"], rec["variant"], rec["model"])] = rec
    return list(latest.values())


def _stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    m = float(np.mean(xs))
    se = float(np.std(xs) / math.sqrt(len(xs))) if len(xs) > 1 else 0.0
    return m, se, len(xs)


def aggregate(records: list[dict]):
    """Per (model, variant, bias): MAE for lm/naive/reactive/bayes.
    Reactive is recomputed offline from the stored episode (assume full neutral
    gain → always react, never infer the bias)."""
    by = defaultdict(lambda: {"lm": [], "naive": [], "reactive": [], "bayes": []})
    for rec in records:
        key = (rec["model"], rec["variant"], rec["bias"])
        ep  = rec.get("episode", {})
        ev, sv = ep.get("events"), ep.get("surveys")
        for s in rec["steps"]:
            by[key]["lm"].append(s.get("lm_abs_err"))
            by[key]["naive"].append(s["naive_abs_err"])
            by[key]["bayes"].append(s["bayes_abs_err"])
            if ev and sv:
                t = s["T_obs"]
                react = compute_blind_forecast(ev[:t], sv[:t], s["next_event_polarity"],
                                               assumed_gain=1.0)
                by[key]["reactive"].append(abs(react["predicted_survey"] - s["gold_survey"]))
    return {k: {f: _stats(v) for f, v in d.items()} for k, d in by.items()}


# ── figure ────────────────────────────────────────────────────────────────────
def make_figure(agg):
    present = [m for m in ORDER if any((m, "V0", b) in agg or (m, "V1", b) in agg or (m, "V2", b) in agg
                                       for b in ("neutral", "biased"))]
    if not present:
        print("No model records found in", _BATCH)
        return None

    # shared y-range across all panels
    ymax = 0.0
    for (m, v, b), d in agg.items():
        for f in ("lm", "naive", "reactive", "bayes"):
            mean = d[f][0]
            if mean is not None:
                ymax = max(ymax, mean)
    ymax = math.ceil((ymax + 0.6))

    nrow, ncol = len(VARIANTS), len(CONDITIONS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * len(present) + 3.0, 2.7 * nrow),
                             sharey=True, squeeze=False)

    for r, (vkey, vlabel) in enumerate(VARIANTS):
        for c, (bkey, blabel) in enumerate(CONDITIONS):
            ax = axes[r][c]
            cells = {m: agg.get((m, vkey, bkey)) for m in present}
            has_data = any(cells[m] and cells[m]["lm"][0] is not None for m in present)

            if not has_data:
                ax.text(0.5, 0.5, f"{vkey} — run pending", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11, color="#9ca3af", style="italic")
                ax.set_xticks([])
                ax.set_ylim(0, ymax)
                if r == 0:
                    ax.set_title(blabel, fontweight="bold")
                if c == 0:
                    ax.set_ylabel(f"{vlabel}\n\nMAE", fontsize=10)
                continue

            xs = np.arange(len(present))
            for i, m in enumerate(present):
                st = cells[m]
                if not st or st["lm"][0] is None:
                    continue
                mean, se, _ = st["lm"]
                ax.bar(i, mean, width=0.62, color=FAMILY_COLOR[FAMILY[m]],
                       edgecolor="white", linewidth=0.6, zorder=3)
                if se:
                    ax.errorbar(i, mean, yerr=se, fmt="none", ecolor="#374151",
                                elinewidth=1.0, capsize=2.5, zorder=4)
                ax.text(i, mean + 0.1, f"{mean:.1f}", ha="center", va="bottom",
                        fontsize=9, color="#374151", zorder=5)

            # reference lines (pooled over models present), label staggered in x to avoid overlap
            refs = []
            for f in ("naive", "reactive", "bayes"):
                vals = [cells[m][f][0] for m in present if cells[m] and cells[m][f][0] is not None]
                refs.append(float(np.mean(vals)) if vals else None)
            for (f, col, xfrac), val in zip(
                    [("Naïve", C_NAIVE, 0.02), ("Reactive", C_REACT, 0.40), ("Bayes", C_BAYES, 0.74)], refs):
                if val is None:
                    continue
                dash = (0, (2, 2)) if f == "Naïve" else (0, (5, 2)) if f == "Reactive" else (0, (4, 1.5))
                ax.axhline(val, ls=dash, lw=1.7, color=col, zorder=2)
                ax.text(-0.55 + xfrac * (len(present) + 0.1), val + 0.08, f"{val:.1f}",
                        fontsize=8.5, color=col, fontweight="bold")

            ax.set_xticks(xs)
            ax.set_xticklabels([LABEL[m] for m in present], rotation=30, ha="right")
            ax.set_ylim(0, ymax)
            ax.set_xlim(-0.7, len(present) - 0.3)
            if r == 0:
                ax.set_title(blabel, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{vlabel}\n\nMAE", fontsize=10)

    handles = [mlines.Line2D([], [], marker="s", linestyle="", markersize=10, color=c, label=l)
               for l, c in [("Qwen", C_QWEN), ("Llama", C_LLAMA), ("Frontier", C_FRONT)]]
    handles += [
        mlines.Line2D([], [], ls=(0, (2, 2)),   color=C_NAIVE, label="Naïve (ignore news)"),
        mlines.Line2D([], [], ls=(0, (5, 2)),   color=C_REACT, label="Reactive (react to all news)"),
        mlines.Line2D([], [], ls=(0, (4, 1.5)), color=C_BAYES, label="Bayes (infers bias → optimal)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=True, framealpha=0.95,
               edgecolor="#d1d5db", bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Forecasting the next poll — only inferring the hidden media bias wins in both conditions",
                 fontsize=13, y=1.06)
    fig.tight_layout(rect=[0, 0, 1, 1.0])
    return fig


def main() -> None:
    records = load_records()
    print(f"Loaded {len(records)} records from {_BATCH}")
    have = sorted({(r['variant']) for r in records})
    print(f"Variants with data: {have or '(none)'}")
    fig = make_figure(aggregate(records))
    if fig is None:
        print("Run the eval first (eval/launch_local.sh / launch_frontier.sh).")
        return
    for ext in ("pdf", "png"):
        path = _OUT / f"exp2_latent_inference.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=900 if ext == "png" else 300)
        print(f"  Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
