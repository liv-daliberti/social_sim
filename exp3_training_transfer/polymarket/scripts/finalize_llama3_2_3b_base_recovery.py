#!/usr/bin/env python3
"""Finalize the disclosed Llama-3.2-3B base format recovery."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import evaluate_locked_test as scoring
import evaluate_scale_holdout as original


POLY = Path(__file__).resolve().parents[1]
REPORTS = POLY / "reports"
RUNS = POLY / "runs"
HOLDOUT = POLY / "data" / "exp4_scale_registered" / "test.tasks.jsonl"
PARENT_EXTENSION = REPORTS / "exp4_architecture_extension_latest.summary.json"
AMENDMENT = RUNS / "exp4_llama3_2_3b_base_structured_recovery_registration_v2.json"
OUTPUT = REPORTS / "exp4_architecture_extension_recovered_latest.summary.json"
MODEL_KEY = "llama3_2_3b"
SEEDS = ("42", "43", "44")
METADATA_KEYS = (
    "task_id",
    "event_id",
    "series_id",
    "template_key",
    "settlement_yes",
    "market_yes_prob",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return original.registered.read_jsonl(path)


def validate_recovery(
    recovery_summary_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    recovery = load_json(recovery_summary_path)
    if recovery.get("status") != "pass":
        raise RuntimeError("Llama-3B base recovery did not pass complete coverage")
    if (
        recovery.get("protocol_version")
        != "exp4_llama3_2_3b_base_structured_recovery_v2"
    ):
        raise RuntimeError("Llama-3B base recovery protocol drifted")
    if recovery.get("classification") != ("disclosed_holdout_informed_syntax_recovery"):
        raise RuntimeError("Llama-3B base recovery classification drifted")
    if recovery.get("model_key") != MODEL_KEY:
        raise RuntimeError("Llama-3B base recovery identity drifted")
    if recovery.get("amendment_sha256") != original.sha256(AMENDMENT):
        raise RuntimeError("Llama-3B base recovery amendment drifted")
    correction = Path(recovery["implementation_correction"])
    if not correction.is_file() or recovery.get(
        "implementation_correction_sha256"
    ) != original.sha256(correction):
        raise RuntimeError("Llama-3B backend correction drifted")
    analysis_code = Path(recovery["analysis_code"])
    if not analysis_code.is_file() or recovery.get(
        "analysis_code_sha256"
    ) != original.sha256(analysis_code):
        raise RuntimeError("Llama-3B base recovery evaluator drifted")
    raw_path = Path(recovery["raw"])
    if not raw_path.is_file() or recovery.get("raw_sha256") != original.sha256(
        raw_path
    ):
        raise RuntimeError("Llama-3B base recovery raw artifact drifted")
    holdout = recovery.get("holdout", {})
    if holdout.get("n") != 318:
        raise RuntimeError("Llama-3B recovery holdout count drifted")
    if holdout.get("test_tasks_sha256") != original.sha256(HOLDOUT):
        raise RuntimeError("Llama-3B recovery holdout hash drifted")
    decoding = recovery.get("decoding", {})
    expected_decoding = {
        "mode": "five_draw_syntax_constrained_stochastic",
        "draws": 5,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "sampling_seed": 20_260_824,
        "max_tokens": 64,
        "guided_backend": "guidance",
        "json_schema": {
            "type": "object",
            "properties": {
                "yes_prob": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                }
            },
            "required": ["yes_prob"],
            "additionalProperties": False,
        },
        "semantic_value_constraint": "number_in_[0,1]_only",
    }
    if decoding != expected_decoding:
        raise RuntimeError("Llama-3B base recovery decoder drifted")
    endpoint = recovery.get("base", {})
    if endpoint.get("draw_parse_coverage") != 1.0:
        raise RuntimeError("Llama-3B recovery is missing parsed draws")
    if endpoint.get("complete_task_parse_coverage") != 1.0:
        raise RuntimeError("Llama-3B recovery is missing complete tasks")
    if not 0.0 <= float(endpoint.get("brier", -1)) <= 1.0:
        raise RuntimeError("Llama-3B recovered Brier score is invalid")
    rows = read_jsonl(raw_path)
    if len(rows) != 318:
        raise RuntimeError("Llama-3B recovery raw row count drifted")
    return recovery, rows


def combine_rows(
    parent_rows: list[dict[str, Any]],
    recovery_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(parent_rows) != 318 or len(recovery_rows) != 318:
        raise RuntimeError("Llama-3B combined inputs must contain 318 rows")
    combined: list[dict[str, Any]] = []
    for parent_row, recovery_row in zip(parent_rows, recovery_rows, strict=True):
        for key in METADATA_KEYS:
            if parent_row.get(key) != recovery_row.get(key):
                raise RuntimeError(f"Llama-3B recovery row drifted on {key}")
        draws = recovery_row.get("draw_forecasts")
        if not isinstance(draws, list) or len(draws) != 5:
            raise RuntimeError("Llama-3B recovery draw roster drifted")
        if any(
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
            for value in draws
        ):
            raise RuntimeError("Llama-3B recovery contains an invalid draw")
        forecast = recovery_row.get("forecast")
        if (
            not isinstance(forecast, (int, float))
            or abs(float(forecast) - sum(float(value) for value in draws) / 5) > 1e-12
        ):
            raise RuntimeError("Llama-3B recovery aggregate is inconsistent")
        row = copy.deepcopy(parent_row)
        row["forecasts"]["base"] = float(forecast)
        row["draw_forecasts"]["base"] = [float(value) for value in draws]
        row["responses"]["base"] = recovery_row["responses"]
        combined.append(row)
    return combined


def recompute_base_comparison(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    trained_names = [f"seed_{seed}" for seed in SEEDS]
    trained_losses: list[float] = []
    base_losses: list[float] = []
    for row in rows:
        label = bool(row["settlement_yes"])
        base_losses.append(scoring.losses(row["forecasts"]["base"], label)[0])
        trained_losses.append(
            sum(
                scoring.losses(row["forecasts"][name], label)[0]
                for name in trained_names
            )
            / len(trained_names)
        )
    return scoring.cluster_bootstrap_delta(rows, trained_losses, base_losses)


def finalize(
    recovery_summary_path: Path,
    output_path: Path = OUTPUT,
) -> dict[str, Any]:
    parent = load_json(PARENT_EXTENSION)
    if parent.get("status") != "pass":
        raise RuntimeError("parent architecture extension is not a pass")
    model = parent["models"][MODEL_KEY]
    parent_summary_path = Path(model["summary"])
    parent_raw_path = Path(model["raw"])
    if model.get("summary_sha256") != original.sha256(parent_summary_path):
        raise RuntimeError("parent Llama-3B summary drifted")
    if model.get("raw_sha256") != original.sha256(parent_raw_path):
        raise RuntimeError("parent Llama-3B raw artifact drifted")

    recovery, recovery_rows = validate_recovery(recovery_summary_path)
    parent_summary = load_json(parent_summary_path)
    parent_rows = read_jsonl(parent_raw_path)
    combined_rows = combine_rows(parent_rows, recovery_rows)

    recovery_job = recovery_summary_path.stem.removeprefix(
        "exp4_llama3_2_3b_base_recovery_j"
    ).removesuffix(".summary")
    combined_prefix = REPORTS / (
        f"exp4_scale_llama3_2_3b_stochastic_recovered_j{recovery_job}"
    )
    combined_raw_path = combined_prefix.with_suffix(".jsonl")
    with combined_raw_path.open("w", encoding="utf-8") as handle:
        for row in combined_rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")

    combined_summary = copy.deepcopy(parent_summary)
    combined_summary[
        "protocol_version"
    ] = "exp4_llama3_2_3b_base_structured_recovery_v2"
    combined_summary["models"]["base"] = recovery["base"]
    combined_summary["comparisons_brier"][
        "trained_seed_mean_minus_base"
    ] = recompute_base_comparison(combined_rows)
    combined_summary["decoding_by_endpoint"] = {
        "base": recovery["decoding"],
        "trained": parent_summary["decoding"],
    }
    combined_summary["base_recovery"] = {
        "classification": recovery["classification"],
        "amendment": recovery["amendment"],
        "amendment_sha256": recovery["amendment_sha256"],
        "implementation_correction": recovery["implementation_correction"],
        "implementation_correction_sha256": recovery[
            "implementation_correction_sha256"
        ],
        "recovery_summary": str(recovery_summary_path),
        "recovery_summary_sha256": original.sha256(recovery_summary_path),
        "recovery_raw": recovery["raw"],
        "recovery_raw_sha256": recovery["raw_sha256"],
    }
    combined_summary["assembly_code"] = str(Path(__file__))
    combined_summary["assembly_code_sha256"] = original.sha256(Path(__file__))
    combined_summary["combined_raw"] = str(combined_raw_path)
    combined_summary["combined_raw_sha256"] = original.sha256(combined_raw_path)
    combined_summary_path = combined_prefix.with_suffix(".summary.json")
    combined_summary_path.write_text(
        json.dumps(combined_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = copy.deepcopy(parent)
    result["protocol_version"] = parent["protocol_version"]
    result["parent_extension"] = str(PARENT_EXTENSION)
    result["parent_extension_sha256"] = original.sha256(PARENT_EXTENSION)
    result["base_recovery"] = combined_summary["base_recovery"]
    result["finalizer"] = str(Path(__file__))
    result["finalizer_sha256"] = original.sha256(Path(__file__))
    result_model = result["models"][MODEL_KEY]
    result_model["base_brier"] = float(recovery["base"]["brier"])
    result_model["models"]["base"] = recovery["base"]
    result_model["comparisons_brier"] = combined_summary["comparisons_brier"]
    result_model["summary"] = str(combined_summary_path)
    result_model["summary_sha256"] = original.sha256(combined_summary_path)
    result_model["raw"] = str(combined_raw_path)
    result_model["raw_sha256"] = original.sha256(combined_raw_path)
    result_model["recovery_job_id"] = recovery_job
    result_model["base_recovery"] = combined_summary["base_recovery"]

    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = finalize(args.recovery_summary, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "llama3_2_3b_base_brier": result["models"][MODEL_KEY]["base_brier"],
                "coverage": result["models"][MODEL_KEY]["models"]["base"][
                    "complete_task_parse_coverage"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
