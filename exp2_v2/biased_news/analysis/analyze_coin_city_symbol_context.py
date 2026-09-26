#!/usr/bin/env python3
"""Analyze the arbitrary-symbol Experiment 2 control against frozen arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data" / "coin_city_stable_relationship_claude_n250_v4"
DESIGN = RUN / "design"
RESPONSES = RUN / "responses"
MATCHED_RESPONSES = RESPONSES / "symbol_control_matched_20260813"
DEFAULT_OUTPUT = RUN / "analysis" / "symbol_context_results.json"
OPEN_MODEL_RESPONSES = RESPONSES / "symbol_control_open_qwen3_4b_20260824"
CONTROL_MODELS = (
    "DeepSeek-V4-Pro",
    "Qwen3-4B-Instruct-2507",
)
ARMS = ("abc_no_context", "abc_context", "abc_symbol_context")
EXPECTED_TASKS = 1_250
DISPLAY = {
    "claude-opus-4-8": "Claude Opus 4.8",
    "gpt-5.6-sol": "GPT-5.6",
    "DeepSeek-V4-Pro": "DeepSeek V4-Pro",
    "FW-Kimi-K3": "Kimi K3",
    "Qwen3-4B-Instruct-2507": "Qwen3-4B",
    "gemini-3.6-flash": "Gemini 3.6 Flash",
    "claude-opus-5": "Claude Opus 5",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def response_path(model: str, arm: str) -> Path:
    if model == "Qwen3-4B-Instruct-2507":
        return OPEN_MODEL_RESPONSES / f"responses_{model}_{arm}.jsonl"
    if model == "DeepSeek-V4-Pro" and arm in {"abc_no_context", "abc_context"}:
        matched = MATCHED_RESPONSES / f"responses_{model}_{arm}.jsonl"
        if matched.exists():
            return matched
    return RESPONSES / f"responses_{model}_{arm}.jsonl"


def latest_responses(model: str, arm: str) -> dict[str, dict]:
    path = response_path(model, arm)
    latest: dict[str, dict] = {}
    if not path.exists():
        return latest
    for row in read_jsonl(path):
        old = latest.get(row["task_id"])
        if old is None or row.get("predicted_poll") is not None:
            latest[row["task_id"]] = row
    return latest


def correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_mean_ci(values: np.ndarray, *, seed: int, draws: int) -> dict:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = np.mean(values[indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean": float(np.mean(values)),
        "ci_95": [float(low), float(high)],
    }


def bootstrap_metric_difference(
    left: np.ndarray,
    right: np.ndarray,
    truths: np.ndarray,
    *,
    metric: str,
    seed: int,
    draws: int,
) -> dict:
    def evaluate(predictions: np.ndarray, target: np.ndarray) -> float:
        if metric == "correlation":
            value = correlation(predictions, target)
            return float(value) if value is not None else float("nan")
        if metric == "regime_accuracy":
            return float(np.mean((predictions >= 0.575) == (target >= 0.575)))
        raise ValueError(metric)

    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(draws):
        indices = rng.integers(0, len(truths), size=len(truths))
        value = evaluate(left[indices], truths[indices]) - evaluate(
            right[indices], truths[indices]
        )
        if np.isfinite(value):
            differences.append(value)
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "difference": evaluate(left, truths) - evaluate(right, truths),
        "ci_95": [float(low), float(high)],
    }


def analyze_model(
    model: str,
    *,
    scores: dict[str, dict],
    episodes: dict[int, dict],
    bootstrap_draws: int,
) -> dict:
    responses = {arm: latest_responses(model, arm) for arm in ARMS}
    record_counts = {}
    parsed_counts = {}
    for arm in ARMS:
        path = response_path(model, arm)
        record_counts[arm] = len(read_jsonl(path)) if path.exists() else 0
        parsed_counts[arm] = sum(
            row.get("predicted_poll") is not None for row in responses[arm].values()
        )

    curves: dict[str, dict] = {}
    for k in range(5):
        task_ids = {
            task_id
            for task_id, score in scores.items()
            if int(score["c_cases"]) == k
        }
        common = sorted(
            task_id
            for task_id in task_ids
            if all(
                responses[arm].get(task_id, {}).get("predicted_poll") is not None
                for arm in ARMS
            )
        )
        if not common:
            curves[str(k)] = {"n": 0}
            continue

        truths = np.asarray(
            [scores[task_id]["gold_expected_poll"] for task_id in common], dtype=float
        )
        target_slopes = np.asarray(
            [episodes[scores[task_id]["episode"]]["target_slope"] for task_id in common],
            dtype=float,
        )
        arm_errors: dict[str, np.ndarray] = {}
        arm_slopes: dict[str, np.ndarray] = {}
        arm_metrics: dict[str, dict] = {}
        for arm in ARMS:
            predictions = np.asarray(
                [responses[arm][task_id]["predicted_poll"] for task_id in common],
                dtype=float,
            )
            implied = np.asarray(
                [
                    (
                        prediction
                        - episodes[scores[task_id]["episode"]]["query_starting_poll"]
                    )
                    / episodes[scores[task_id]["episode"]]["query_net_news"]
                    for task_id, prediction in zip(common, predictions)
                ],
                dtype=float,
            )
            arm_slopes[arm] = implied
            arm_errors[arm] = np.abs(predictions - truths)
            arm_metrics[arm] = {
                "forecast_mae": float(np.mean(arm_errors[arm])),
                "slope_mae": float(np.mean(np.abs(implied - target_slopes))),
                "rho": correlation(implied, target_slopes),
                "regime_accuracy": float(
                    np.mean((implied >= 0.575) == (target_slopes >= 0.575))
                ),
            }

        seed = int(hashlib.sha256(f"{model}:{k}".encode()).hexdigest()[:8], 16)
        curves[str(k)] = {
            "n": len(common),
            "arms": arm_metrics,
            "paired_forecast_mae": {
                "semantic_minus_no_context": bootstrap_mean_ci(
                    arm_errors["abc_context"] - arm_errors["abc_no_context"],
                    seed=seed + 1,
                    draws=bootstrap_draws,
                ),
                "symbol_minus_no_context": bootstrap_mean_ci(
                    arm_errors["abc_symbol_context"]
                    - arm_errors["abc_no_context"],
                    seed=seed + 2,
                    draws=bootstrap_draws,
                ),
                "symbol_minus_semantic": bootstrap_mean_ci(
                    arm_errors["abc_symbol_context"] - arm_errors["abc_context"],
                    seed=seed + 3,
                    draws=bootstrap_draws,
                ),
            },
            "paired_discrimination": {
                "symbol_minus_no_context_rho": bootstrap_metric_difference(
                    arm_slopes["abc_symbol_context"],
                    arm_slopes["abc_no_context"],
                    target_slopes,
                    metric="correlation",
                    seed=seed + 4,
                    draws=bootstrap_draws,
                ),
                "symbol_minus_no_context_regime_accuracy": bootstrap_metric_difference(
                    arm_slopes["abc_symbol_context"],
                    arm_slopes["abc_no_context"],
                    target_slopes,
                    metric="regime_accuracy",
                    seed=seed + 5,
                    draws=bootstrap_draws,
                ),
            },
        }
    return {
        "record_counts": record_counts,
        "parsed_counts": parsed_counts,
        "curves": curves,
    }


def fmt(value: float) -> str:
    rounded = round(value, 2)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:.2f}"


def paired_cell(record: dict) -> str:
    low, high = record["ci_95"]
    return f"{fmt(record['mean'])} [{fmt(low)}, {fmt(high)}]"


def difference_cell(record: dict) -> str:
    low, high = record["ci_95"]
    return f"{fmt(record['difference'])} [{fmt(low)}, {fmt(high)}]"


def write_tex(result: dict, path: Path) -> None:
    lines = []
    for model in CONTROL_MODELS:
        model_lines = []
        for k in range(5):
            curve = result["models"][model]["curves"][str(k)]
            if not curve.get("n"):
                continue
            arms = curve["arms"]
            paired = curve["paired_forecast_mae"]
            discrimination = curve["paired_discrimination"]
            model_lines.append(
                " & ".join(
                    [
                        DISPLAY[model] if not model_lines else "",
                        str(k),
                        str(curve["n"]),
                        fmt(arms["abc_no_context"]["forecast_mae"]),
                        fmt(arms["abc_no_context"]["rho"]),
                        fmt(arms["abc_context"]["forecast_mae"]),
                        fmt(arms["abc_context"]["rho"]),
                        fmt(arms["abc_symbol_context"]["forecast_mae"]),
                        fmt(arms["abc_symbol_context"]["rho"]),
                        fmt(arms["abc_symbol_context"]["regime_accuracy"]),
                        paired_cell(paired["symbol_minus_no_context"]),
                        difference_cell(discrimination["symbol_minus_no_context_rho"]),
                        difference_cell(
                            discrimination["symbol_minus_no_context_regime_accuracy"]
                        ),
                    ]
                )
                + r" \\"
            )
        if model_lines:
            if lines:
                lines.append(r"\midrule")
            lines.extend(model_lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = [
        r"\begin{tabular}{l cc rr rr rrr ccc}",
        r"\toprule",
        r"& & & \multicolumn{2}{c}{No C context} & \multicolumn{2}{c}{Semantic context}",
        r"& \multicolumn{3}{c}{Arbitrary symbol} & \multicolumn{3}{c}{Symbol $-$ no context} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}",
        r"Model & $k$ & $n$ & MAE & $\rho$ & MAE & $\rho$ & MAE & $\rho$ & Acc.",
        r"& $\Delta\mathrm{MAE}$ [CI] & $\Delta\rho$ [CI] & $\Delta\mathrm{Acc.}$ [CI] \\",
        r"\midrule",
        *lines,
        r"\bottomrule",
        r"\end{tabular}",
    ]
    path.write_text("\n".join(table) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tex-output", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=2_000)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    scores = {
        row["task_id"]: row for row in read_jsonl(DESIGN / "scoring_key.jsonl")
    }
    episodes = {row["episode"]: row for row in read_jsonl(DESIGN / "episodes.jsonl")}
    result = {
        "experiment": RUN.name,
        "control": "episode-randomized arbitrary KIV/ZOR labels",
        "primary_stratum": "zero City C examples (k=0)",
        "bootstrap_unit": "episode (one task per episode within each k)",
        "bootstrap_draws": args.bootstrap_draws,
        "response_sources": {
            "symbol": str(RESPONSES),
            "deepseek_comparison_arms": str(MATCHED_RESPONSES),
            "qwen3_4b_open_model_arms": str(OPEN_MODEL_RESPONSES),
        },
        "models": {
            model: analyze_model(
                model,
                scores=scores,
                episodes=episodes,
                bootstrap_draws=args.bootstrap_draws,
            )
            for model in CONTROL_MODELS
        },
    }
    complete_models = [
        model
        for model in CONTROL_MODELS
        if all(result["models"][model]["record_counts"][arm] == EXPECTED_TASKS for arm in ARMS)
    ]
    result["complete_models"] = complete_models
    result["complete"] = len(complete_models) == len(CONTROL_MODELS)
    if not result["complete"] and not args.allow_partial:
        raise SystemExit("symbol-control response set is incomplete")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.tex_output:
        write_tex(result, args.tex_output)
    print(f"Wrote {args.output}; complete models: {', '.join(complete_models) or 'none'}")


if __name__ == "__main__":
    main()
