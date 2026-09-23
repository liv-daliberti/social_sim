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

# Seven checkpoints is past what per-model colours can carry in a wrapped column,
# so the encoding is the split the figure is about: solid warm lines for the
# checkpoints whose forecasts recover the mapping, gray dashed for those that do
# not. The behavioural verdict is read from the released comparison, not asserted.
RUNS = (
    ("Qwen3-4B", "qwen3_4b_symbol_probe_v2", "Qwen3-4B-Instruct-2507", "v"),
    ("Qwen3-8B", "qwen3_8b_symbol_probe_v2", "Qwen3-8B", "^"),
    ("Qwen3-14B", "qwen3_14b_symbol_probe_v2", "Qwen3-14B", "o"),
    ("Qwen2.5-32B", "qwen2_5_32b_symbol_probe_v2", "Qwen2.5-32B-Instruct", "s"),
    ("Qwen2.5-72B", "qwen2_5_72b_symbol_probe_v2", "Qwen2.5-72B-Instruct", "D"),
    ("Llama-3.1-8B", "llama3_1_8b_symbol_probe_v2", "Llama-3.1-8B-Instruct", "<"),
    ("Llama-3.1-70B", "llama3_1_70b_symbol_probe_v2", "Llama-3.1-70B-Instruct", ">"),
)
COMPARISON = (
    ROOT
    / "exp2_v2"
    / "biased_news"
    / "data"
    / "coin_city_stable_relationship_claude_n250_v4"
    / "analysis"
    / "symbol_context_model_comparison_20260824.json"
)
RECOVERS_COLOURS = ("#0072B2", "#D55E00", "#009E73", "#6A3D9A")
ABSTAINS_COLOUR = "#A8ADB4"
# Short tick labels: the figure is set narrow enough to wrap beside the text.
CONDITIONS = (
    ("abc_context", "true_target", "Correct\nlabel"),
    ("abc_wrong_context", "true_target", "Inverted,\ntrue"),
    ("abc_wrong_context", "cue_target", "Inverted,\nlabeled"),
)
EXPECTED_TEST_EPISODES = 88

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def recovers(model_key: str) -> bool:
    """Does this checkpoint's forecast recover the mapping behaviourally?"""
    curves = json.loads(COMPARISON.read_text())["models"][model_key]["curves"]["0"]
    low, _high = curves["paired_discrimination"]["symbol_minus_no_context_rho"]["ci_95"]
    return low > 0


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
    loaded, omitted = [], []
    warm = iter(RECOVERS_COLOURS)
    for label, directory, model_key, marker in RUNS:
        if not (PROBE_ROOT / directory / "probe_results.json").exists():
            omitted.append(label)
            continue
        does_recover = recovers(model_key)
        colour = next(warm) if does_recover else ABSTAINS_COLOUR
        loaded.append((label, load(directory), colour, marker, does_recover))
    if not loaded:
        raise SystemExit("no completed symbol probe runs")
    # Draw the abstaining checkpoints first so the positives sit on top.
    loaded.sort(key=lambda row: row[4])
    if omitted:
        print("omitted (run not complete): " + ", ".join(omitted))

    # Two groups, not seven lines. Every positive traces the same path and every
    # null hugs chance, so per-model curves were redundant ink and forced a legend
    # that ate a quarter of the panel. Per-checkpoint values and permutation
    # p-values live in the power-matched appendix table instead.
    fig, axes = plt.subplots(2, 1, figsize=(2.55, 2.90), sharex=True, sharey=True)
    positions = list(range(len(CONDITIONS)))

    for axis, depth in zip(axes, (0, 4)):
        axis.axhline(0.5, color="#888888", linewidth=0.7, linestyle=(0, (4, 3)), zorder=1)
        for does_recover, colour, label in (
            (True, RECOVERS_COLOURS[0], "Qwen + Llama, 14–72B: recover"),
            (False, ABSTAINS_COLOUR, "Qwen + Llama, 4–8B: do not"),
        ):
            group = [series(res, depth)[0] for _, res, _, _, rec in loaded if rec is does_recover]
            if not group:
                continue
            lo = [min(v[i] for v in group) for i in positions]
            hi = [max(v[i] for v in group) for i in positions]
            mid = [sorted(v[i] for v in group)[len(group) // 2] for i in positions]
            axis.fill_between(positions, lo, hi, color=colour, alpha=0.22,
                              linewidth=0, zorder=2)
            axis.plot(positions, mid, color=colour, linewidth=2.0,
                      linestyle="-" if does_recover else (0, (3, 2)),
                      label=f"{label} ($n{{=}}{len(group)}$)" if depth == 4 else None,
                      zorder=3)
        axis.set_title(
            "No target cases ($k{=}0$)" if depth == 0 else "Four target cases ($k{=}4$)",
            pad=4, fontsize=8.5,
        )
        axis.set_xticks(positions)
        axis.set_xticklabels([name for _, _, name in CONDITIONS])
        axis.set_xlim(-0.15, len(CONDITIONS) - 0.85)
        axis.set_ylim(0.0, 1.05)
        axis.set_yticks([0.0, 0.5, 1.0])
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.supylabel("Held-out regime AUC", fontsize=8, x=0.035)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", frameon=False, handlelength=1.6,
               borderpad=0.0, columnspacing=1.1, handletextpad=0.5, ncol=1, fontsize=7)
    fig.tight_layout(pad=0.4, h_pad=0.7, rect=(0, 0.115, 1, 1))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, facecolor="white")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
