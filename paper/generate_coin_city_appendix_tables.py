#!/usr/bin/env python3
"""Emit the Coin City appendix table bodies from the frozen multimodel results.

The per-model diagnostics tables and the paired-difference table in
experiment2_appendix.tex are transcriptions of
exp2_v2/biased_news/data/.../analysis/multimodel_results.json. Regenerating them
here keeps the appendix numbers tied to the analysis output instead of hand
copying.

    python paper/generate_coin_city_appendix_tables.py --model gemini-3.6-flash
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = (
    ROOT
    / "exp2_v2"
    / "biased_news"
    / "data"
    / "coin_city_stable_relationship_claude_n250_v4"
    / "analysis"
    / "multimodel_results.json"
)
ARM_LABELS = (
    ("baseline", "Target only"),
    ("abc_no_context", "ABC, no C context"),
    ("abc_context", "ABC + C context"),
)


def _dot(value: float, places: int = 3) -> str:
    """Format .353 / -.027 in the appendix's leading-dot style."""
    text = f"{value:.{places}f}"
    return text.replace("0.", ".", 1) if text.startswith(("0.", "-0.")) else text


def _bold(text: str, is_best: bool) -> str:
    return f"\\textbf{{{text}}}" if is_best else text


def diagnostics_rows(curves: dict) -> str:
    lines = []
    for k in range(5):
        curve = curves[str(k)]
        arms = curve["arms"]
        best_mae = min(arms[arm]["forecast_mae"] for arm, _ in ARM_LABELS)
        best_slope = min(arms[arm]["slope_mae"] for arm, _ in ARM_LABELS)
        best_rho = max(arms[arm]["rho"] for arm, _ in ARM_LABELS)
        if k:
            lines.append("\\midrule")
        for index, (arm, label) in enumerate(ARM_LABELS):
            metrics = arms[arm]
            lead = f"{k} &" if index == 0 else "  &"
            lines.append(
                f"{lead} {label} & {curve['n']} & "
                + _bold(
                    f"{metrics['forecast_mae']:.3f}",
                    metrics["forecast_mae"] == best_mae,
                )
                + " & "
                + _bold(_dot(metrics["slope_mae"]), metrics["slope_mae"] == best_slope)
                + " & "
                + _bold(_dot(metrics["rho"]), metrics["rho"] == best_rho)
                + " \\\\"
            )
    return "\n".join(lines)


def paired_rows(curves: dict, label: str) -> str:
    lines = []
    for k in range(5):
        paired = curves[str(k)]["paired_forecast_mae"]["context_minus_no_context"]
        low, high = paired["ci_95"]
        lead = label if k == 0 else " " * len(label)
        lines.append(
            f"{lead} & {k} & {_dot(paired['mean'])} & "
            f"$[{_dot(low)},{_dot(high)}]$ \\\\"
        )
    return "\n".join(lines)


MODEL_LABELS = (
    ("claude-opus-4-8", "Claude Opus~4.8"),
    ("gpt-5.6-sol", "GPT-5.6"),
    ("DeepSeek-V4-Pro", "DeepSeek-V4-Pro"),
    ("FW-Kimi-K3", "Kimi K3"),
    ("gemini-3.6-flash", "Gemini 3.6 Flash"),
    ("claude-opus-5", "Claude Opus~5"),
)


def ols_rows() -> str:
    """Paired contrast between the correct-context arm and the ABC OLS benchmark.

    The benchmark is the analyst's through-origin fit on every displayed A, B and
    available C row; it is context-blind and never appears in a model prompt.
    Negative values favour the model. Bold marks intervals excluding zero.
    """
    results = json.loads(RESULTS.read_text())
    lines = []
    for model, label in MODEL_LABELS:
        curves = results["models"][model]["curves"]
        cells = []
        for k in range(5):
            paired = curves[str(k)]["paired_forecast_mae"]["context_minus_abc_ols"]
            low, high = paired["ci_95"]
            text = f"${_dot(paired['mean'], 2)}$"
            cells.append(_bold(text, low * high > 0))
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def pairwise_rows(arm: str = "abc_context", k: int = 0, draws: int = 4000) -> str:
    """Paired model-vs-model contrasts on identical episodes.

    Every deployment answers the same episodes, so overlapping marginal
    intervals do not establish that two models are indistinguishable; the paired
    per-episode difference is the correct comparison and is far tighter.
    """
    import itertools
    import sys

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import generate_coin_city_split_figures as figure

    scores = {
        row["task_id"]: row
        for row in figure.read_jsonl(figure.DESIGN / "scoring_key.jsonl")
    }
    rng = np.random.default_rng(20260812)
    errors = {}
    for model in figure.FIGURE_MODELS:
        responses = figure.load_latest_responses(model)
        ids = [
            task_id
            for task_id, score in scores.items()
            if int(score["c_cases"]) == k
            and all(
                responses[a].get(task_id, {}).get("predicted_poll") is not None
                for a in figure.ARMS
            )
        ]
        errors[model] = {
            task_id: abs(
                responses[arm][task_id]["predicted_poll"]
                - scores[task_id]["gold_expected_poll"]
            )
            for task_id in ids
        }

    order = sorted(
        figure.FIGURE_MODELS, key=lambda m: np.mean(list(errors[m].values()))
    )
    lines, resolved = [], 0
    for a, b in itertools.combinations(order, 2):
        ids = sorted(set(errors[a]) & set(errors[b]))
        diff = np.array([errors[a][t] - errors[b][t] for t in ids])
        boot = diff[rng.integers(0, len(diff), size=(draws, len(diff)))].mean(axis=1)
        low, high = np.quantile(boot, [0.025, 0.975])
        mark = ""
        if low > 0 or high < 0:
            resolved += 1
            mark = "$^{\\dagger}$"
        lines.append(
            f"{figure.MODEL_LABELS[a]} $-$ {figure.MODEL_LABELS[b]} & "
            f"{_dot(float(diff.mean()))}{mark} & "
            f"$[{_dot(float(low))},{_dot(float(high))}]$ \\\\"
        )
    lines.append(f"% resolved {resolved} of {len(lines)}")
    return "\n".join(lines)


def analogical_rows() -> str:
    """Rows for the deterministic label-conditioned regression table."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import generate_coin_city_split_figures as figure

    metrics = figure.compute_metrics()
    reference = metrics["DeepSeek-V4-Pro"]
    lines = []
    for k in range(5):
        correct_llm = [
            metrics[model][str(k)]["mae"]["abc_context"]
            for model in figure.FIGURE_MODELS
        ]
        wrong_llm = [
            metrics[model][str(k)]["mae"][figure.WRONG_ARM]
            for model in figure.FIGURE_MODELS
        ]
        correct = reference[str(k)]["analogical_regression"]["correct"]
        inverted = reference[str(k)]["analogical_regression"]["inverted"]
        lines.append(
            f"{k} & {correct['forecast_mae']:.3f} & {_dot(correct['slope_mae'])} & "
            f"{_dot(correct['rho'])} & {min(correct_llm):.3f}--{max(correct_llm):.3f} & "
            f"{inverted['forecast_mae']:.3f} & {_dot(inverted['slope_mae'])} & "
            f"{_dot(inverted['rho'])} & {min(wrong_llm):.3f}--{max(wrong_llm):.3f} \\\\"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--label", default=None, help="row label in the paired table")
    parser.add_argument("--pairwise", action="store_true")
    parser.add_argument("--analogical", action="store_true")
    parser.add_argument("--ols", action="store_true")
    args = parser.parse_args()

    if args.pairwise:
        print(pairwise_rows())
        return
    if args.analogical:
        print(analogical_rows())
        return
    if args.ols:
        print(ols_rows())
        return

    if not args.model:
        raise SystemExit(
            "--model is required unless --pairwise, --analogical or --ols is given"
        )
    results = json.loads(RESULTS.read_text())
    if args.model not in results["models"]:
        raise SystemExit(f"{args.model} not in {sorted(results['models'])}")
    curves = results["models"][args.model]["curves"]
    parsed = results["models"][args.model]["parsed_counts"]

    print(f"% parsed per arm: {parsed}")
    print(f"% --- diagnostics table body for {args.model} ---")
    print(diagnostics_rows(curves))
    print()
    print(f"% --- paired-difference rows for {args.model} ---")
    print(paired_rows(curves, args.label or args.model))


if __name__ == "__main__":
    main()
