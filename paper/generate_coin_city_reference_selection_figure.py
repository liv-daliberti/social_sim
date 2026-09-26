#!/usr/bin/env python3
"""Plot forecast/reference partial correlations across the four Coin City arms.

Read saved analysis results and display every estimate and confidence interval.
The orange series uses the cue-named reference when a cue is present and the
true-regime-matched reference otherwise.
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
    ("DeepSeek-V4-Pro", "DeepSeek"),
    ("FW-Kimi-K3", "Kimi K3"),
    ("gemini-3.6-flash", "Gemini"),
)
# arm key, panel title, key for the reference the cue names, key for the other
ARMS = (
    ("baseline", "Target only\n(no references shown)",
     "regime_matched_reference", "regime_mismatched_reference"),
    ("abc_no_context", "References, no cue",
     "regime_matched_reference", "regime_mismatched_reference"),
    ("abc_context", "References\nCorrect cue", "cue_reference", "other_reference"),
    ("abc_wrong_context", "References\nMisleading cue", "cue_reference", "other_reference"),
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
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.0), sharey=True)
    positions = np.arange(len(MODELS))
    offset = 0.14

    for axis, (arm, title, named_key, other_key) in zip(axes, ARMS):
        named, named_err, other, other_err = [], [[], []], [], [[], []]
        for model, _ in MODELS:
            cell = results["primary"][model]["arms"][arm]["0"]
            for key, values, err in (
                (named_key, named, named_err),
                (other_key, other, other_err),
            ):
                metric = cell[key]
                point = metric["partial_r"]
                if point is None or any(v is None for v in metric["ci_95"]):
                    raise ValueError(f"undefined estimate or interval: {model}/{arm}/{key}")
                low, high = metric["ci_95"]
                values.append(point)
                err[0].append(max(0.0, point - (low if low is not None else point)))
                err[1].append(max(0.0, (high if high is not None else point) - point))

        axis.axvline(0.0, color="#777777", linewidth=0.7, zorder=1)
        axis.errorbar(
            named, positions - offset, xerr=named_err, fmt="o", markersize=3.5,
            capsize=2, color=NAMED_COLOR, elinewidth=1, zorder=3,
            label="Named / regime-matched reference",
        )
        axis.errorbar(
            other, positions + offset, xerr=other_err, fmt="o", markersize=3.5,
            capsize=2, color=OTHER_COLOR, elinewidth=1, zorder=2,
            label="Other reference",
        )
        axis.set_title(title, pad=7)
        axis.set_yticks(positions)
        axis.set_yticklabels([label for _, label in MODELS])
        axis.set_xlim(-0.3, 1.05)
        axis.set_xticks([0, 0.5, 1])
        axis.grid(axis="x", color="#E6E6E6", linewidth=0.5)
        axis.tick_params(axis="y", length=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)

    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.55, 1.01))
    fig.supxlabel("Partial correlation controlling for regime", fontsize=9, y=0.02)
    fig.tight_layout(rect=(0, 0.04, 1, 0.91), w_pad=1.6)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, facecolor="white")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
