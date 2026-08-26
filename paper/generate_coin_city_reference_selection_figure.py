#!/usr/bin/env python3
"""Draw the four-arm reference-selection figure for the Experiment 2 main text.

The claim is a pattern across arms, not a single number: with no reference cases
neither reference loads, with references but no cue both load equally, the correct
cue concentrates loading on the reference it names, and an inverted cue moves that
loading onto the other one. Reading four paired numbers per deployment out of a
table is work; the shape is the argument, so it is drawn.

Reads the frozen authority written by
exp2_v2/biased_news/analysis/analyze_coin_city_reference_selection.py and fails
closed if an arm or deployment is missing.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RESULTS = (
    ROOT
    / "exp2_v2"
    / "biased_news"
    / "data"
    / "coin_city_stable_relationship_claude_n250_v4"
    / "analysis"
    / "reference_selection_results.json"
)
OUTPUT = ROOT / "paper" / "figures" / "exp2_reference_selection.pdf"

MODELS = (
    ("claude-opus-4-8", "Opus 4.8"),
    ("claude-opus-5", "Opus 5"),
    ("gpt-5.6-sol", "GPT-5.6"),
    ("DeepSeek-V4-Pro", "V4-Pro"),
    ("FW-Kimi-K3", "Kimi K3"),
    ("gemini-3.6-flash", "3.6 Flash"),
)
# arm key, panel title, key for the reference the cue names, key for the other
ARMS = (
    ("baseline", "Target only\n(no references shown)",
     "regime_matched_reference", "regime_mismatched_reference"),
    ("abc_no_context", "References, no cue",
     "regime_matched_reference", "regime_mismatched_reference"),
    ("abc_context", "References $+$ correct cue", "cue_reference", "other_reference"),
    ("abc_wrong_context", "References $+$ inverted cue", "cue_reference", "other_reference"),
)

NAMED_COLOR = "#D55E00"
OTHER_COLOR = "#A8ADB4"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load() -> dict:
    results = json.loads(RESULTS.read_text())
    for model, _ in MODELS:
        arms = results["primary"].get(model, {}).get("arms", {})
        for arm, _title, named, other in ARMS:
            cell = arms.get(arm, {}).get("0")
            if cell is None or named not in cell or other not in cell:
                raise SystemExit(f"missing {model}/{arm} in {RESULTS.name}")
    return results


def main() -> None:
    results = load()
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.5), sharey=True)
    positions = np.arange(len(MODELS))
    width = 0.38

    for axis, (arm, title, named_key, other_key) in zip(axes, ARMS):
        named, named_err, other, other_err = [], [[], []], [], [[], []]
        for model, _ in MODELS:
            cell = results["primary"][model]["arms"][arm]["0"]
            for key, values, err in (
                (named_key, named, named_err),
                (other_key, other, other_err),
            ):
                metric = cell[key]
                point = metric["partial_r"] or 0.0
                low, high = metric["ci_95"]
                values.append(point)
                err[0].append(max(0.0, point - (low if low is not None else point)))
                err[1].append(max(0.0, (high if high is not None else point) - point))

        axis.axhline(0.0, color="#666666", linewidth=0.7, zorder=1)
        axis.bar(
            positions - width / 2, named, width, yerr=named_err, capsize=1.8,
            color=NAMED_COLOR, edgecolor="none", zorder=2,
            error_kw={"linewidth": 0.7, "ecolor": "#333333"},
            label="reference the cue names",
        )
        axis.bar(
            positions + width / 2, other, width, yerr=other_err, capsize=1.8,
            color=OTHER_COLOR, edgecolor="none", zorder=2,
            error_kw={"linewidth": 0.7, "ecolor": "#333333"},
            label="the other reference",
        )
        axis.set_title(title, pad=6)
        axis.set_xticks(positions)
        axis.set_xticklabels([label for _, label in MODELS], rotation=38, ha="right")
        axis.set_ylim(-0.15, 1.0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].set_ylabel(r"partial $r$ | regime")
    axes[0].legend(loc="upper left", frameon=False, handlelength=1.1)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, facecolor="white")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
