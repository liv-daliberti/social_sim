#!/usr/bin/env python3
"""Draw arbitrary-symbol regime decoding for the Experiment 2 main text.

This is the induced-mapping counterpart to the semantic-arm slope figure. Under
arbitrary KIV/ZOR labels the mapping has to be inferred from each episode's own
examples, so a probe that follows the cue there cannot be reading a word it
already knows.

Two panels because the two evidence depths say different things. With no target
cases the decoded regime follows whichever label the prompt carries, true or
false. With four target cases the same probe recovers the true regime from an
inverted prompt instead, which is revision in representational form. Llama-3.1-8B
is drawn as the negative control: its forecasts do not recover the mapping, and
neither does the probe.

Reads the frozen probe results and fails closed on a missing run or condition.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PROBE_ROOT = (
    ROOT
    / "exp2_v2"
    / "biased_news"
    / "data"
    / "coin_city_stable_relationship_claude_n250_v4"
    / "mechanistic_probe"
)
OUTPUT = ROOT / "paper" / "figures" / "exp2_symbol_decoding.pdf"
TARGET = "regime"  # the induced mapping decodes as a response family, not a coefficient

# label, run directory, colour, marker, whether the model recovers the mapping
RUNS = (
    ("Qwen3-14B", "qwen3_14b_symbol_probe_v2", "#D55E00", "o", True),
    ("Qwen2.5-32B", "qwen2_5_32b_symbol_probe_v2", "#0072B2", "s", True),
    ("Llama-3.1-8B", "llama3_1_8b_symbol_probe_v2", "#A8ADB4", "^", False),
)
CONDITIONS = (
    ("abc_context", "true_target", "Correct\nlabel"),
    ("abc_wrong_context", "true_target", "Inverted label,\ntrue regime"),
    ("abc_wrong_context", "cue_target", "Inverted label,\nlabelled regime"),
)
EXPECTED_TEST_EPISODES = 88

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


def load(directory: str) -> dict:
    run = PROBE_ROOT / directory
    results = json.loads((run / "probe_results.json").read_text())
    manifest = json.loads((run / "task_manifest.json").read_text())
    if results.get("status") != "probe_complete":
        raise SystemExit(f"{directory}: status is {results.get('status')!r}")
    if manifest.get("episode_counts", {}).get("test") != EXPECTED_TEST_EPISODES:
        raise SystemExit(f"{directory}: sealed test is not {EXPECTED_TEST_EPISODES}")
    if results.get("tasks_sha256") != manifest.get("tasks_sha256"):
        raise SystemExit(f"{directory}: results do not match the frozen task file")
    return results


def series(results: dict, depth: int) -> tuple[list[float], bool]:
    """Held-out scores for one model-depth cell, and whether it rejects its null."""
    condition = results["primary"]["conditions"][f"k{depth}:{TARGET}"]
    selected = condition["selected_test"]
    key = "roc_auc" if TARGET == "regime" else "r2"
    values = []
    for arm, target, _ in CONDITIONS:
        probe = selected.get(arm, {}).get("probe", {}).get(target)
        if probe is None or probe.get(key) is None:
            raise SystemExit(f"missing {arm}/{target} at k={depth}")
        values.append(float(probe[key]))
    rejects = (
        condition["permutation_null_on_correct_context_test"]["one_sided_p"] < 0.05
    )
    return values, rejects


def main() -> None:
    loaded = [(label, load(directory), colour, marker, recovers)
              for label, directory, colour, marker, recovers in RUNS]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    positions = range(len(CONDITIONS))

    for axis, depth in zip(axes, (0, 4)):
        reference = 0.5 if TARGET == "regime" else 0.0
        axis.axhline(
            reference, color="#888888", linewidth=0.7, linestyle=(0, (4, 3)), zorder=1
        )
        for label, results, colour, marker, _recovers in loaded:
            values, rejects = series(results, depth)
            # An open marker and a dashed line mark a cell that does not reject its
            # held-out permutation null, so a shape there is not read as a result.
            axis.plot(
                list(positions),
                values,
                color=colour,
                marker=marker,
                markersize=4.5,
                markerfacecolor=colour if rejects else "white",
                linewidth=2.0 if rejects else 1.3,
                linestyle="-" if rejects else (0, (3, 2)),
                label=label if depth == 0 else None,
                zorder=3 if rejects else 2,
            )
        axis.set_title(
            "No target cases ($k{=}0$)" if depth == 0 else "Four target cases ($k{=}4$)",
            pad=6,
        )
        axis.set_xticks(list(positions))
        axis.set_xticklabels([name for _, _, name in CONDITIONS])
        axis.set_xlim(-0.25, len(CONDITIONS) - 0.75)
        axis.set_ylim(-1.25, 0.55) if TARGET == "slope" else axis.set_ylim(0.0, 1.0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].set_ylabel(
        "Held-out $R^2$" if TARGET == "slope" else "Held-out regime AUC"
    )
    axes[0].legend(loc="lower left", frameon=False, handlelength=1.8)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, facecolor="white")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
