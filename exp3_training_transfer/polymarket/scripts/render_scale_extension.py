#!/usr/bin/env python3
"""Validate and aggregate the completed Exp4 Qwen scale evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import evaluate_locked_test as registered


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
PROTOCOL_VERSION = "exp4_qwen_scale_v1"
MODEL_KEYS = ("qwen3_4b", "qwen3_8b", "qwen3_14b")
SEEDS = (42, 43, 44)
EXPECTED_DECODING = {
    "mode": "five_draw_non_thinking_stochastic",
    "draws": 5,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "sampling_seed": 20_260_824,
    "max_tokens": 128,
    "incomplete_draw_policy": "endpoint_task_brier_loss_one",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_evaluation_specs(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in value.split(";"):
        model_key, job_id = item.split(":", 1)
        if not job_id.isdigit():
            raise AssertionError(f"evaluation job is not numeric: {item}")
        if model_key in result:
            raise AssertionError(f"duplicate evaluation model: {model_key}")
        result[model_key] = job_id
    if tuple(sorted(result)) != tuple(sorted(MODEL_KEYS)):
        raise AssertionError(f"scale finalizer requires {MODEL_KEYS}")
    return result


def resolve_evaluation(model_key: str, job_id: str) -> tuple[Path, Path]:
    prefix = REPORTS / f"exp4_scale_{model_key}_stochastic_j{job_id}"
    summary = prefix.with_suffix(".summary.json")
    raw = prefix.with_suffix(".jsonl")
    if not summary.is_file() or not raw.is_file():
        raise AssertionError(
            f"incomplete scale evaluation for {model_key} job {job_id}"
        )
    return summary, raw


def row_identity(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["task_id"],
        row["event_id"],
        row["series_id"],
        row["template_key"],
        bool(row["settlement_yes"]),
        float(row["market_yes_prob"]),
    )


def endpoint_loss_vector(rows: list[dict[str, Any]], endpoint: str) -> list[float]:
    return [
        registered.losses(row["forecasts"][endpoint], bool(row["settlement_yes"]))[0]
        for row in rows
    ]


def trained_loss_vector(rows: list[dict[str, Any]]) -> list[float]:
    by_seed = [endpoint_loss_vector(rows, f"seed_{seed}") for seed in SEEDS]
    return [
        sum(losses[index] for losses in by_seed) / len(SEEDS)
        for index in range(len(rows))
    ]


def validate_summary(
    model_key: str,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    if summary.get("protocol_version") != PROTOCOL_VERSION:
        raise AssertionError(f"{model_key} protocol drifted")
    if summary.get("model_key") != model_key:
        raise AssertionError(f"{model_key} summary identity drifted")
    if summary.get("decoding") != EXPECTED_DECODING:
        raise AssertionError(f"{model_key} stochastic decoding drifted")
    if summary.get("test_was_used_during_training") is not False:
        raise AssertionError(f"{model_key} holdout-use declaration drifted")
    expected_endpoints = {"base", *(f"seed_{seed}" for seed in SEEDS)}
    if set(summary.get("models", {})) != expected_endpoints:
        raise AssertionError(f"{model_key} endpoint roster drifted")
    if int(summary["holdout"]["n"]) != len(rows):
        raise AssertionError(f"{model_key} holdout count drifted")
    raw_brier = sum(trained_loss_vector(rows)) / len(rows)
    summary_brier = statistics.fmean(
        float(summary["models"][f"seed_{seed}"]["brier"]) for seed in SEEDS
    )
    if not math.isclose(raw_brier, summary_brier, abs_tol=1e-12):
        raise AssertionError(f"{model_key} raw and summary Brier disagree")


def validate_alignment(rows_by_model: dict[str, list[dict[str, Any]]]) -> None:
    reference = [row_identity(row) for row in rows_by_model[MODEL_KEYS[0]]]
    if not reference:
        raise AssertionError("scale evaluation is empty")
    if len({identity[0] for identity in reference}) != len(reference):
        raise AssertionError("scale evaluation contains duplicate task IDs")
    for model_key in MODEL_KEYS[1:]:
        observed = [row_identity(row) for row in rows_by_model[model_key]]
        if observed != reference:
            raise AssertionError(
                f"{model_key} does not match the registered task order"
            )


def mean_seed_metric(summary: dict[str, Any], name: str) -> float:
    return statistics.fmean(
        float(summary["models"][f"seed_{seed}"][name]) for seed in SEEDS
    )


def model_result(
    summary: dict[str, Any], trained_losses: list[float]
) -> dict[str, Any]:
    seed_briers = {
        str(seed): float(summary["models"][f"seed_{seed}"]["brier"]) for seed in SEEDS
    }
    return {
        "model": summary["model"],
        "base_brier": float(summary["models"]["base"]["brier"]),
        "base_complete_task_parse_coverage": float(
            summary["models"]["base"]["complete_task_parse_coverage"]
        ),
        "base_draw_parse_coverage": float(
            summary["models"]["base"]["draw_parse_coverage"]
        ),
        "trained_seed_mean_brier": statistics.fmean(seed_briers.values()),
        "trained_seed_brier_sd": statistics.stdev(seed_briers.values()),
        "trained_seed_brier": seed_briers,
        "trained_seed_mean_complete_task_parse_coverage": mean_seed_metric(
            summary, "complete_task_parse_coverage"
        ),
        "trained_seed_mean_draw_parse_coverage": mean_seed_metric(
            summary, "draw_parse_coverage"
        ),
        "trained_seed_mean_brier_from_raw": sum(trained_losses) / len(trained_losses),
        "trained_minus_base_brier": summary["comparisons_brier"][
            "trained_seed_mean_minus_base"
        ],
    }


def render_tex(report: dict[str, Any]) -> str:
    labels = {
        "qwen3_4b": r"Qwen3-4B-Instruct-2507$^\dagger$",
        "qwen3_8b": "Qwen3-8B",
        "qwen3_14b": "Qwen3-14B",
    }
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Checkpoint & Base & RL mean & RL $-$ base & Parse \\",
        r"\midrule",
    ]
    for model_key in MODEL_KEYS:
        result = report["models"][model_key]
        delta = result["trained_minus_base_brier"]["estimate"]
        parse = result["trained_seed_mean_complete_task_parse_coverage"]
        lines.append(
            f"{labels[model_key]} & {result['base_brier']:.4f} & "
            f"{result['trained_seed_mean_brier']:.4f} & {delta:+.4f} & "
            f"{100.0 * parse:.1f}\\% \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "",
            (
                r"\(\dagger\) Later Instruct-2507 revision; the controlled "
                r"same-release scale contrast is 8B versus 14B."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def aggregate(specs: dict[str, str]) -> dict[str, Any]:
    summaries: dict[str, dict[str, Any]] = {}
    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, dict[str, str]] = {}
    for model_key in MODEL_KEYS:
        summary_path, raw_path = resolve_evaluation(model_key, specs[model_key])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = registered.read_jsonl(raw_path)
        validate_summary(model_key, summary, rows)
        summaries[model_key] = summary
        rows_by_model[model_key] = rows
        sources[model_key] = {
            "job_id": specs[model_key],
            "summary": str(summary_path),
            "summary_sha256": sha256(summary_path),
            "raw": str(raw_path),
            "raw_sha256": sha256(raw_path),
        }

    validate_alignment(rows_by_model)
    holdout_hashes = {
        summary["holdout"]["test_tasks_sha256"] for summary in summaries.values()
    }
    if len(holdout_hashes) != 1:
        raise AssertionError("scale summaries use different holdout hashes")
    baseline_briers = {
        (
            float(summary["baselines"]["market"]["brier"]),
            float(summary["baselines"]["platt_market_train_only"]["brier"]),
        )
        for summary in summaries.values()
    }
    if len(baseline_briers) != 1:
        raise AssertionError("scale summaries disagree on frozen baselines")

    trained_vectors = {
        model_key: trained_loss_vector(rows_by_model[model_key])
        for model_key in MODEL_KEYS
    }
    rows = rows_by_model[MODEL_KEYS[0]]
    cross_model = {
        "qwen3_14b_minus_qwen3_8b": registered.cluster_bootstrap_delta(
            rows, trained_vectors["qwen3_14b"], trained_vectors["qwen3_8b"]
        ),
        "qwen3_14b_minus_qwen3_4b_descriptive": (
            registered.cluster_bootstrap_delta(
                rows,
                trained_vectors["qwen3_14b"],
                trained_vectors["qwen3_4b"],
            )
        ),
    }
    primary = summaries["qwen3_14b"]["comparisons_brier"][
        "trained_seed_mean_minus_base"
    ]
    market = summaries["qwen3_14b"]["comparisons_brier"][
        "trained_seed_mean_minus_market"
    ]
    platt = summaries["qwen3_14b"]["comparisons_brier"][
        "trained_seed_mean_minus_platt_market"
    ]
    return {
        "status": "pass",
        "protocol_version": PROTOCOL_VERSION,
        "holdout_tasks_sha256": next(iter(holdout_hashes)),
        "n": len(rows),
        "decoding": EXPECTED_DECODING,
        "sources": sources,
        "models": {
            model_key: model_result(summaries[model_key], trained_vectors[model_key])
            for model_key in MODEL_KEYS
        },
        "baselines": summaries["qwen3_14b"]["baselines"],
        "primary_qwen3_14b_trained_minus_base": primary,
        "qwen3_14b_trained_minus_market": market,
        "qwen3_14b_trained_minus_platt_market": platt,
        "cross_model_brier": cross_model,
        "decision_flags": {
            "qwen3_14b_base_relative_gain_conclusive": (
                float(primary["ci95_high"]) < 0.0
            ),
            "qwen3_14b_beats_qwen3_8b_conclusive": (
                float(cross_model["qwen3_14b_minus_qwen3_8b"]["ci95_high"]) < 0.0
            ),
            "qwen3_14b_beats_market_conclusive": (float(market["ci95_high"]) < 0.0),
            "qwen3_14b_beats_train_only_platt_conclusive": (
                float(platt["ci95_high"]) < 0.0
            ),
        },
        "interpretation_lock": (
            "The 8B-to-14B contrast is the controlled same-release scale test. "
            "The 4B-Instruct-2507 point is descriptive."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluations", required=True)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=REPORTS / "exp4_scale_latest",
    )
    args = parser.parse_args()

    report = aggregate(parse_evaluation_specs(args.evaluations))
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".summary.json")
    tex_path = args.output_prefix.with_suffix(".tex")
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tex_path.write_text(render_tex(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"TeX table -> {tex_path}")


if __name__ == "__main__":
    main()
