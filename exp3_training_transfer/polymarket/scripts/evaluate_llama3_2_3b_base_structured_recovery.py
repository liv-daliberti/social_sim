#!/usr/bin/env python3
"""Syntax-constrained recovery for the Llama-3.2-3B Exp4 base endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import evaluate_scale_holdout as original


POLY = Path(__file__).resolve().parents[1]
RUNS = POLY / "runs"
REPORTS = POLY / "reports"
AMENDMENT = RUNS / "exp4_llama3_2_3b_base_structured_recovery_registration_v2.json"
CORRECTION = RUNS / "exp4_llama3_2_3b_structured_backend_correction_v3.json"
PRIOR_PREFIX = REPORTS / "exp4_llama3_2_3b_base_recovery_j30911810"
EXPECTED_MODEL = (
    "/n/fs/similarity/social_sim/.runtime/hf_home/hub/"
    "models--unsloth--Llama-3.2-3B-Instruct/snapshots/"
    "006f5dcd1393c3add266de40994ba96225e9689d"
)
MODEL_KEY = "llama3_2_3b"
MAX_TOKENS = 64
MAX_MODEL_LEN = 1920
JSON_SCHEMA = {
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
}


def load_amendment() -> dict[str, Any]:
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    if amendment.get("status") != "registered_before_recovery_execution":
        raise RuntimeError("structured Llama-3B recovery is not registered")
    if amendment.get("model_snapshot") != EXPECTED_MODEL:
        raise RuntimeError("structured Llama-3B model snapshot drifted")
    if amendment.get("holdout_tasks_sha256") != original.sha256(
        original.DATA / "test.tasks.jsonl"
    ):
        raise RuntimeError("structured Llama-3B holdout drifted")
    if amendment.get("prior_raw_sha256") != original.sha256(
        PRIOR_PREFIX.with_suffix(".jsonl")
    ):
        raise RuntimeError("prior Llama-3B recovery raw artifact drifted")
    if amendment.get("prior_summary_sha256") != original.sha256(
        PRIOR_PREFIX.with_suffix(".summary.json")
    ):
        raise RuntimeError("prior Llama-3B recovery summary drifted")
    if amendment.get("json_schema") != JSON_SCHEMA:
        raise RuntimeError("structured Llama-3B JSON schema drifted")
    correction = json.loads(CORRECTION.read_text(encoding="utf-8"))
    if correction.get("status") != "registered_before_corrected_execution":
        raise RuntimeError("structured backend correction is not registered")
    if correction.get("parent_amendment_sha256") != original.sha256(AMENDMENT):
        raise RuntimeError("structured backend correction parent drifted")
    if correction.get("json_schema") != JSON_SCHEMA:
        raise RuntimeError("structured backend correction schema drifted")
    return amendment


def main() -> None:
    import vllm
    from vllm.sampling_params import GuidedDecodingParams

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=EXPECTED_MODEL)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    if args.model != EXPECTED_MODEL:
        raise RuntimeError("unregistered structured Llama-3B model")

    load_amendment()
    rows, holdout_manifest = original.validate_holdout()
    guided = GuidedDecodingParams(json=JSON_SCHEMA, backend="guidance")
    sampling = vllm.SamplingParams(
        n=original.DRAWS,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        seed=original.SAMPLING_SEED,
        max_tokens=MAX_TOKENS,
        guided_decoding=guided,
    )
    llm = vllm.LLM(
        model=args.model,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.90,
        guided_decoding_backend="guidance",
        trust_remote_code=True,
    )
    prompts = [
        original.format_prompt(llm.get_tokenizer(), row["input"], "auto")
        for row in rows
    ]
    forecasts, draw_forecasts, responses = original.generate_draws(
        llm, prompts, sampling
    )
    endpoint = original.endpoint_summary(forecasts, draw_forecasts, rows)
    status = (
        "pass"
        if endpoint["complete_task_parse_coverage"] == 1.0
        and endpoint["draw_parse_coverage"] == 1.0
        else "incomplete"
    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_prefix.with_suffix(".jsonl")
    with raw_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            handle.write(
                json.dumps(
                    {
                        "task_id": row["task_id"],
                        "event_id": row["event_id"],
                        "series_id": row["series_id"],
                        "template_key": row["template_key"],
                        "settlement_yes": row["settlement_yes"],
                        "market_yes_prob": row["market_yes_prob"],
                        "forecast": forecasts[index],
                        "draw_forecasts": draw_forecasts[index],
                        "responses": responses[index],
                    },
                    sort_keys=True,
                    ensure_ascii=True,
                )
                + "\n"
            )

    summary = {
        "status": status,
        "protocol_version": "exp4_llama3_2_3b_base_structured_recovery_v2",
        "classification": "disclosed_holdout_informed_syntax_recovery",
        "model_key": MODEL_KEY,
        "model": args.model,
        "template": "auto",
        "amendment": str(AMENDMENT),
        "amendment_sha256": original.sha256(AMENDMENT),
        "implementation_correction": str(CORRECTION),
        "implementation_correction_sha256": original.sha256(CORRECTION),
        "prior_raw": str(PRIOR_PREFIX.with_suffix(".jsonl")),
        "prior_raw_sha256": original.sha256(PRIOR_PREFIX.with_suffix(".jsonl")),
        "prior_summary": str(PRIOR_PREFIX.with_suffix(".summary.json")),
        "prior_summary_sha256": original.sha256(
            PRIOR_PREFIX.with_suffix(".summary.json")
        ),
        "holdout": {
            "n": len(rows),
            "manifest_sha256": original.sha256(original.DATA / "manifest.json"),
            "test_tasks_sha256": holdout_manifest["hashes"]["test_tasks_sha256"],
        },
        "decoding": {
            "mode": "five_draw_syntax_constrained_stochastic",
            "draws": original.DRAWS,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "sampling_seed": original.SAMPLING_SEED,
            "max_tokens": MAX_TOKENS,
            "guided_backend": "guidance",
            "json_schema": JSON_SCHEMA,
            "semantic_value_constraint": "number_in_[0,1]_only",
        },
        "base": endpoint,
        "raw": str(raw_path),
        "raw_sha256": original.sha256(raw_path),
        "analysis_code": str(Path(__file__)),
        "analysis_code_sha256": original.sha256(Path(__file__)),
    }
    summary_path = args.output_prefix.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if status != "pass":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
