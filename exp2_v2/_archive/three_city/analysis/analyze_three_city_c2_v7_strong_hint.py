#!/usr/bin/env python3
"""Analyze the post-specified v7 strong-structure positive control.

This script never changes the frozen two-arm confirmatory analysis. It pairs
the later strong-hint arm to both original arms by episode and reports it as a
post-specified descriptive positive control.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))

import analyze_three_city_c2_v7_confirmatory as frozen
import plot_three_city_c2_v7_paper as paper

_DATA = _ROOT / "data" / "three_city_c2_v7"
_STRONG_DATA = _ROOT / "data" / "three_city_c2_v7_strong_hint"
_ARMS = ("blind", "hint", "strong_hint")
_ARM_LABEL = {
    "blind": "Structure-blind",
    "hint": "Relevance hint",
    "strong_hint": "Strong structure hint",
}


def _mean(values: Sequence[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.mean(clean) if clean else float("nan")


def _metric(rows: Sequence[Mapping[str, Any]], name: str) -> float:
    if name == "mae":
        return _mean([float(row["absolute_error"]) for row in rows])
    if name == "beta":
        return frozen._beta(rows)
    raise ValueError(name)


def _bootstrap_three_arm(
    paired: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    arm_rows = {
        arm: [pair[arm] for pair in paired]
        for arm in _ARMS
    }
    result: dict[str, Any] = {"n": len(paired), "arms": {}, "contrasts": {}}
    for arm in _ARMS:
        result["arms"][arm] = {
            "mae": _metric(arm_rows[arm], "mae"),
            "beta": _metric(arm_rows[arm], "beta"),
        }
    if len(paired) < 3:
        for arm in _ARMS:
            result["arms"][arm]["mae_ci"] = [float("nan"), float("nan")]
            result["arms"][arm]["beta_ci"] = [float("nan"), float("nan")]
        return result

    rng = random.Random(seed)
    samples = {
        arm: {"mae": [], "beta": []}
        for arm in _ARMS
    }
    contrast_samples = {
        "strong_over_blind": {
            "mae_improvement": [],
            "beta_increase": [],
        },
        "strong_over_hint": {
            "mae_improvement": [],
            "beta_increase": [],
        },
    }
    for _ in range(draws):
        draw = [paired[rng.randrange(len(paired))] for _ in paired]
        estimates: dict[str, dict[str, float]] = {}
        for arm in _ARMS:
            rows = [pair[arm] for pair in draw]
            estimates[arm] = {
                "mae": _metric(rows, "mae"),
                "beta": _metric(rows, "beta"),
            }
            for metric in ("mae", "beta"):
                value = estimates[arm][metric]
                if math.isfinite(value):
                    samples[arm][metric].append(value)
        for comparison, reference in (
            ("strong_over_blind", "blind"),
            ("strong_over_hint", "hint"),
        ):
            mae_improvement = (
                estimates[reference]["mae"]
                - estimates["strong_hint"]["mae"]
            )
            beta_increase = (
                estimates["strong_hint"]["beta"]
                - estimates[reference]["beta"]
            )
            if math.isfinite(mae_improvement):
                contrast_samples[comparison]["mae_improvement"].append(
                    mae_improvement
                )
            if math.isfinite(beta_increase):
                contrast_samples[comparison]["beta_increase"].append(
                    beta_increase
                )

    for arm in _ARMS:
        result["arms"][arm]["mae_ci"] = list(
            frozen._ci(samples[arm]["mae"])
        )
        result["arms"][arm]["beta_ci"] = list(
            frozen._ci(samples[arm]["beta"])
        )
    for comparison, reference in (
        ("strong_over_blind", "blind"),
        ("strong_over_hint", "hint"),
    ):
        result["contrasts"][comparison] = {
            "mae_improvement": (
                result["arms"][reference]["mae"]
                - result["arms"]["strong_hint"]["mae"]
            ),
            "mae_improvement_ci": list(
                frozen._ci(
                    contrast_samples[comparison]["mae_improvement"]
                )
            ),
            "beta_increase": (
                result["arms"]["strong_hint"]["beta"]
                - result["arms"][reference]["beta"]
            ),
            "beta_increase_ci": list(
                frozen._ci(
                    contrast_samples[comparison]["beta_increase"]
                )
            ),
        }
    return result


def _descriptive_grid(
    rows: Sequence[dict[str, Any]],
    models: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model in models:
        result[model] = {}
        for condition in ("relevant", "none", "orthogonal"):
            result[model][condition] = {}
            for k in frozen.PREFIX_LADDER:
                paired = paper._paired_arm_rows(
                    rows,
                    model=model,
                    condition=condition,
                    k=k,
                    arms=_ARMS,
                )
                arm_metrics = {
                    arm: {
                        "mae": _metric(
                            [pair[arm] for pair in paired],
                            "mae",
                        ),
                        "beta": _metric(
                            [pair[arm] for pair in paired],
                            "beta",
                        ),
                    }
                    for arm in _ARMS
                }
                result[model][condition][str(k)] = {
                    "n": len(paired),
                    **arm_metrics,
                }
    return result


def _read_manifest_audit(
    directory: Path,
    *,
    expected_calls: int,
    models: Sequence[str],
) -> dict[str, Any]:
    manifests = []
    for path in sorted(directory.glob("responses_*.manifest.json")):
        record = json.loads(path.read_text())
        record["_path"] = str(path)
        manifests.append(record)
    by_model = {record.get("model"): record for record in manifests}
    failures = []
    for model in models:
        record = by_model.get(model)
        if record is None:
            failures.append(f"missing manifest for {model}")
            continue
        if record.get("status") != "complete":
            failures.append(f"{model} status={record.get('status')}")
        if record.get("selected_model_calls") != expected_calls:
            failures.append(
                f"{model} selected_model_calls="
                f"{record.get('selected_model_calls')}"
            )
        if record.get("responses_received") != expected_calls:
            failures.append(
                f"{model} responses_received="
                f"{record.get('responses_received')}"
            )
    return {
        "directory": str(directory),
        "expected_calls_per_model": expected_calls,
        "models": {
            model: {
                key: value
                for key, value in by_model.get(model, {}).items()
                if key
                in {
                    "status",
                    "selected_model_calls",
                    "responses_received",
                    "successfully_parsed",
                    "new_errors_this_run",
                    "endpoint",
                    "updated_at",
                    "_path",
                }
            }
            for model in models
        },
        "failures": failures,
        "valid": not failures,
    }


def _format(value: float) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.2f}"


def _write_summary(
    path: Path,
    *,
    primary: Mapping[str, Any],
    models: Sequence[str],
) -> None:
    lines = [
        "# Three-city C2 v7: strong-structure positive control",
        "",
        (
            "**Status:** post-specified descriptive positive control. This arm "
            "was designed after partial results from the frozen blind-versus-"
            "relevance-hint run had been viewed; it is not a third "
            "confirmatory arm."
        ),
        "",
        "## Truthful-background results at k=2",
        "",
        "| Model | Arm | n | MAE | 95% CI | beta | 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        result = primary[model]
        for arm in _ARMS:
            metric = result["arms"][arm]
            lines.append(
                "| "
                + " | ".join(
                    (
                        frozen._DISPLAY[model],
                        _ARM_LABEL[arm],
                        str(result["n"]),
                        _format(metric["mae"]),
                        (
                            f"[{_format(metric['mae_ci'][0])}, "
                            f"{_format(metric['mae_ci'][1])}]"
                        ),
                        _format(metric["beta"]),
                        (
                            f"[{_format(metric['beta_ci'][0])}, "
                            f"{_format(metric['beta_ci'][1])}]"
                        ),
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            (
                "The strong arm explicitly discloses the recurring two-pattern "
                "structure but not City C's type or the answer. All three arms "
                "are compared on the same parsed episodes within each model."
            ),
            "",
            (
                "Here beta=0 tracks forecasts based on City C cases alone, "
                "whereas beta=1 tracks the prompt-visible regression-style "
                "combination of Cities A, B, and C."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _write_csv(
    path: Path,
    *,
    primary: Mapping[str, Any],
    models: Sequence[str],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "model",
                "arm",
                "n",
                "mae",
                "mae_ci_low",
                "mae_ci_high",
                "beta",
                "beta_ci_low",
                "beta_ci_high",
                "status",
            ),
        )
        writer.writeheader()
        for model in models:
            result = primary[model]
            for arm in _ARMS:
                metric = result["arms"][arm]
                writer.writerow(
                    {
                        "model": model,
                        "arm": arm,
                        "n": result["n"],
                        "mae": metric["mae"],
                        "mae_ci_low": metric["mae_ci"][0],
                        "mae_ci_high": metric["mae_ci"][1],
                        "beta": metric["beta"],
                        "beta_ci_low": metric["beta_ci"][0],
                        "beta_ci_high": metric["beta_ci"][1],
                        "status": "postspecified_positive_control",
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--answer-key",
        type=Path,
        default=_DATA / "answer_key_c2_v7.jsonl",
    )
    parser.add_argument(
        "--responses-dir",
        type=Path,
        default=_DATA / "confirmatory",
    )
    parser.add_argument(
        "--strong-responses-dir",
        type=Path,
        default=_STRONG_DATA / "confirmatory",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=_STRONG_DATA / "analysis",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    keys = {
        row["task_id"]: row
        for row in frozen._read_jsonl(args.answer_key)
    }
    responses = frozen._latest_responses(
        sorted(args.responses_dir.glob("responses_*.jsonl"))
    )
    responses.update(
        frozen._latest_responses(
            sorted(args.strong_responses_dir.glob("responses_*.jsonl"))
        )
    )
    rows = frozen._score_rows(responses, keys)
    models = [
        model
        for model in frozen._MODEL_ORDER
        if all(
            any(
                row["model"] == model and row["arm"] == arm
                for row in rows
            )
            for arm in _ARMS
        )
    ]
    if not models:
        raise SystemExit("no model has parsed responses in all three arms")

    original_audit = _read_manifest_audit(
        args.responses_dir,
        expected_calls=2_880,
        models=models,
    )
    strong_audit = _read_manifest_audit(
        args.strong_responses_dir,
        expected_calls=1_440,
        models=models,
    )
    if (
        not args.allow_incomplete
        and (not original_audit["valid"] or not strong_audit["valid"])
    ):
        raise SystemExit(
            "response collection is incomplete; use --allow-incomplete only "
            "for pipeline tests"
        )

    primary = {}
    for model_index, model in enumerate(models):
        paired = paper._paired_arm_rows(
            rows,
            model=model,
            condition=frozen._PRIMARY_CONDITION,
            k=frozen._PRIMARY_K,
            arms=_ARMS,
        )
        primary[model] = _bootstrap_three_arm(
            paired,
            seed=frozen._BOOTSTRAP_SEED + 70_000 + 1_000 * model_index,
            draws=args.bootstrap_draws,
        )

    result = {
        "experiment": "three_city_c2_v7_strong_hint",
        "status": "postspecified_positive_control",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "disclosure": (
            "Designed after partial two-arm v7 results were viewed; not a "
            "third confirmatory arm."
        ),
        "focal_condition": frozen._PRIMARY_CONDITION,
        "focal_k": frozen._PRIMARY_K,
        "bootstrap": {
            "unit": "episode",
            "pairing": "common parsed episodes across all three arms",
            "draws": args.bootstrap_draws,
            "interval": "percentile_95",
        },
        "models": models,
        "collection_audit": {
            "original_two_arm": original_audit,
            "strong_hint": strong_audit,
        },
        "primary_descriptive": primary,
        "secondary_by_condition_and_k": _descriptive_grid(rows, models),
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    json_path = args.outdir / "strong_hint_positive_control_results.json"
    csv_path = args.outdir / "strong_hint_positive_control_primary.csv"
    summary_path = args.outdir / "strong_hint_positive_control_summary.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _write_csv(csv_path, primary=primary, models=models)
    _write_summary(summary_path, primary=primary, models=models)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
