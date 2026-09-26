#!/usr/bin/env python3
"""Analyze the natural continuous-profile C2 v9 experiment."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from engine.three_city_c2_v9 import CASES_BY_PREFIX, PREFIX_LADDER, PROMPT_ARMS

_RUN = (
    _ROOT
    / "data"
    / "three_city_c2_v9"
    / "full_k1_5_clean_information"
)
_RESPONSES = _RUN / "responses"
_ANSWER_KEY = _RUN / "design" / "answer_key_c2_v9.jsonl"
_OUTDIR = _RUN / "analysis"
_EXPECTED_PER_CELL = 120 * len(PREFIX_LADDER)
_BOOTSTRAP = 10_000

_DISPLAY = {
    "claude-opus-4-8": "Claude Opus 4.8",
    "DeepSeek-V4-Pro": "DeepSeek V4 Pro",
    "gpt-5.4": "GPT-5.4",
}
_MODEL_ORDER = tuple(_DISPLAY)
_MODEL_STYLE = {
    "claude-opus-4-8": ("#4C78A8", "o"),
    "DeepSeek-V4-Pro": ("#7A5195", "s"),
    "gpt-5.4": ("#3F3F3F", "^"),
}
_ARM_LABEL = {
    "c_only": "City C evidence only",
    "abc": "A/B/C information",
    "abc_relevance": "A/B/C + relevance hint",
}
_ARM_COLOR = {
    "c_only": "#E68613",
    "abc": "#168A72",
    "abc_relevance": "#267CB5",
}
_TARGET = "#E68613"
_ABC = "#4C9A4A"
_GRID = "#D8D8D8"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _latest_responses(
    paths: Iterable[Path],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for record in _read_jsonl(path):
            key = (record["model"], record["arm"], record["task_id"])
            priority = 2 if record.get("predicted_poll") is not None else 1
            old = latest.get(key)
            old_priority = -1 if old is None else (
                2 if old.get("predicted_poll") is not None else 1
            )
            if priority >= old_priority:
                latest[key] = record
    return latest


def _score_rows(
    responses: Mapping[tuple[str, str, str], Mapping[str, Any]],
    keys: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for (model, arm, task_id), response in responses.items():
        if task_id not in keys or response.get("predicted_poll") is None:
            continue
        key = keys[task_id]
        prediction = float(response["predicted_poll"])
        gold = float(key["gold"]["expected_poll"])
        rows.append(
            {
                "model": model,
                "arm": arm,
                "task_id": task_id,
                "episode_id": key["episode_id"],
                "k": int(key["k"]),
                "prediction": prediction,
                "gold": gold,
                "absolute_error": abs(prediction - gold),
                "target_only": float(
                    key["baselines"]["target_only"]["predicted_poll"]
                ),
                "abc_shrinkage": float(
                    key["baselines"]["abc_shrinkage"]["predicted_poll"]
                ),
            }
        )
    return rows


def _paired(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str,
    k: int,
) -> list[dict[str, Any]]:
    selected = {
        (row["arm"], row["episode_id"]): row
        for row in rows
        if row["model"] == model
        and row["k"] == k
        and row["arm"] in PROMPT_ARMS
    }
    episodes = sorted(
        episode
        for episode in {episode for _, episode in selected}
        if all((arm, episode) in selected for arm in PROMPT_ARMS)
    )
    return [
        {
            "episode_id": episode,
            **{arm: selected[(arm, episode)] for arm in PROMPT_ARMS},
        }
        for episode in episodes
    ]


def _mean(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.mean(clean) if clean else float("nan")


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(probability * len(ordered))))
    return ordered[index]


def _ci(values: Sequence[float]) -> list[float]:
    return [_percentile(values, 0.025), _percentile(values, 0.975)]


def _pooling_slope(rows: Sequence[Mapping[str, Any]]) -> float:
    x = [
        float(row["abc_shrinkage"]) - float(row["target_only"])
        for row in rows
    ]
    y = [
        float(row["prediction"]) - float(row["target_only"])
        for row in rows
    ]
    if len(x) < 3:
        return float("nan")
    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)
    denominator = sum((value - mean_x) ** 2 for value in x)
    if denominator < 1e-12:
        return float("nan")
    return sum(
        (xv - mean_x) * (yv - mean_y)
        for xv, yv in zip(x, y)
    ) / denominator


def _statistics(
    paired: Sequence[Mapping[str, Any]],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {"n": len(paired), "arms": {}}
    for arm in PROMPT_ARMS:
        rows = [pair[arm] for pair in paired]
        result["arms"][arm] = {
            "mae": _mean(row["absolute_error"] for row in rows),
            "pooling_slope": _pooling_slope(rows),
        }
    if len(paired) < 3:
        for arm in PROMPT_ARMS:
            result["arms"][arm]["mae_ci"] = [float("nan")] * 2
            result["arms"][arm]["pooling_slope_ci"] = [float("nan")] * 2
        result["contrasts"] = {}
        return result

    samples = {
        arm: {"mae": [], "pooling_slope": []}
        for arm in PROMPT_ARMS
    }
    contrasts = {
        "abc_minus_c_only": [],
        "relevance_minus_abc": [],
    }
    rng = random.Random(seed)
    for _ in range(draws):
        sample = [paired[rng.randrange(len(paired))] for _ in paired]
        estimates = {}
        for arm in PROMPT_ARMS:
            rows = [pair[arm] for pair in sample]
            estimates[arm] = _mean(row["absolute_error"] for row in rows)
            samples[arm]["mae"].append(estimates[arm])
            slope = _pooling_slope(rows)
            if math.isfinite(slope):
                samples[arm]["pooling_slope"].append(slope)
        contrasts["abc_minus_c_only"].append(
            estimates["c_only"] - estimates["abc"]
        )
        contrasts["relevance_minus_abc"].append(
            estimates["abc"] - estimates["abc_relevance"]
        )
    for arm in PROMPT_ARMS:
        result["arms"][arm]["mae_ci"] = _ci(samples[arm]["mae"])
        result["arms"][arm]["pooling_slope_ci"] = _ci(
            samples[arm]["pooling_slope"]
        )
    result["contrasts"] = {
        name: {"mae_improvement": _mean(values), "mae_improvement_ci": _ci(values)}
        for name, values in contrasts.items()
    }
    return result


def _baseline_mae(
    keys: Mapping[str, Mapping[str, Any]],
    *,
    method: str,
    k: int,
) -> float:
    selected = [row for row in keys.values() if int(row["k"]) == k]
    return _mean(
        abs(
            float(row["baselines"][method]["predicted_poll"])
            - float(row["gold"]["expected_poll"])
        )
        for row in selected
    )


def _make_structure_figure(
    path: Path,
    *,
    curves: Mapping[str, Any],
    keys: Mapping[str, Mapping[str, Any]],
    models: Sequence[str],
    interim: bool,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.15), sharex=True, sharey=True)
    ks = list(PREFIX_LADDER)
    target = [_baseline_mae(keys, method="target_only", k=k) for k in ks]
    abc = [_baseline_mae(keys, method="abc_shrinkage", k=k) for k in ks]
    for axis, arm, panel in zip(axes, PROMPT_ARMS, ("A", "B", "C")):
        for model in models:
            color, marker = _MODEL_STYLE[model]
            estimates = [curves[model][str(k)]["arms"][arm]["mae"] for k in ks]
            lows = [curves[model][str(k)]["arms"][arm]["mae_ci"][0] for k in ks]
            highs = [curves[model][str(k)]["arms"][arm]["mae_ci"][1] for k in ks]
            axis.plot(
                ks,
                estimates,
                color=color,
                marker=marker,
                markerfacecolor="white",
                markeredgewidth=1.1,
                linewidth=1.7,
                markersize=4.5,
                zorder=4,
            )
            axis.fill_between(ks, lows, highs, color=color, alpha=0.10, linewidth=0)
        axis.plot(ks, target, color=_TARGET, linestyle=(0, (3, 2)), linewidth=1.35)
        axis.plot(ks, abc, color=_ABC, linestyle=(0, (3, 2)), linewidth=1.35)
        axis.set_title(
            f"({panel}) {_ARM_LABEL[arm]}",
            fontsize=9.2,
            fontweight="bold",
            color=_ARM_COLOR[arm],
        )
        axis.set_xticks(ks)
        axis.set_xlabel("Evidence round $k$", fontsize=8)
        axis.grid(color=_GRID, linewidth=0.55)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="both", labelsize=7.5)
    axes[0].set_ylabel("Forecast MAE (poll points)", fontsize=8)
    axes[0].set_ylim(bottom=0.0)
    handles = [
        Line2D(
            [0],
            [0],
            color=_MODEL_STYLE[model][0],
            marker=_MODEL_STYLE[model][1],
            linewidth=1.7,
            markersize=4.5,
            label=_DISPLAY[model],
        )
        for model in models
    ]
    handles.extend(
        [
            Line2D([0], [0], color=_TARGET, linestyle=(0, (3, 2)), label="City C estimator"),
            Line2D([0], [0], color=_ABC, linestyle=(0, (3, 2)), label="Matched A/B/C estimator"),
        ]
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=len(handles),
        frameon=False,
        fontsize=6.8,
    )
    figure.text(
        0.5,
        -0.005,
        "Rounds 1–5 contain 1, 2, 4, 8, and 16 completed City C cases. Scores and response slopes are continuous and noisy.",
        ha="center",
        fontsize=7,
        color="#555555",
    )
    if interim:
        figure.text(
            0.5,
            1.105,
            "INTERIM — INCOMPLETE COLLECTION — NOT CONFIRMATORY",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#A61B1B",
        )
    figure.tight_layout(rect=(0, 0.04, 1, 0.91))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, default=_RESPONSES)
    parser.add_argument("--answer-key", type=Path, default=_ANSWER_KEY)
    parser.add_argument("--outdir", type=Path, default=_OUTDIR)
    parser.add_argument("--draws", type=int, default=_BOOTSTRAP)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    latest = _latest_responses(sorted(args.responses.glob("responses_*.jsonl")))
    keys = {row["task_id"]: row for row in _read_jsonl(args.answer_key)}
    rows = _score_rows(latest, keys)
    if not args.allow_incomplete:
        expected = _EXPECTED_PER_CELL * len(PROMPT_ARMS) * len(_MODEL_ORDER)
        if len(rows) != expected:
            raise SystemExit(f"collection incomplete: {len(rows)}/{expected} parsed")
    curves = {}
    for model_index, model in enumerate(_MODEL_ORDER):
        curves[model] = {}
        for k in PREFIX_LADDER:
            curves[model][str(k)] = _statistics(
                _paired(rows, model=model, k=k),
                draws=args.draws,
                seed=20260804 + 1000 * model_index + k,
            )
    args.outdir.mkdir(parents=True, exist_ok=True)
    figure_path = args.outdir / "three_city_c2_v9_structure.png"
    _make_structure_figure(
        figure_path,
        curves=curves,
        keys=keys,
        models=_MODEL_ORDER,
        interim=args.allow_incomplete,
    )
    result = {
        "experiment": "three_city_c2_v9_clean_information_relevance",
        "status": "interim" if args.allow_incomplete else "complete",
        "parsed_rows": len(rows),
        "cases_by_round": CASES_BY_PREFIX,
        "curves": curves,
        "baseline_mae": {
            method: {
                str(k): _baseline_mae(keys, method=method, k=k)
                for k in PREFIX_LADDER
            }
            for method in ("target_only", "abc_shrinkage")
        },
    }
    (args.outdir / "three_city_c2_v9_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {figure_path} and {figure_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
