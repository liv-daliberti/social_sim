#!/usr/bin/env python3
"""Validate the post-hoc Kimi output-budget repair on truncated holdout cases."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import evaluate_hosted_scale_holdout as hosted


MODEL = "FW-Kimi-K3"
MAX_OUTPUT_TOKENS = 1024
REPORTS = hosted.REPORTS / "exp4_hosted_kimi_repair"
OUTPUT = REPORTS / "kimi_max1024_truncation_diagnostic.jsonl"


def select_truncated_cases(
    limit: int = 2,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows, _ = hosted.validate_inputs("holdout", hosted.REGISTRATION)
    rows_by_id = {row["task_id"]: row for row in rows}
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen: set[str] = set()
    for draw_index in range(hosted.DRAWS):
        raw_path, _ = hosted.output_paths(
            hosted.REPORTS / "exp4_hosted", MODEL, "holdout", draw_index
        )
        for record in hosted.read_rows(raw_path):
            task_id = record["task_id"]
            if (
                task_id not in seen
                and record["yes_prob"] is None
                and record["finish_reason"] == "length"
                and record["error"] == "unparseable response"
            ):
                selected.append((rows_by_id[task_id], record))
                seen.add(task_id)
                if len(selected) == limit:
                    return selected
    raise RuntimeError(f"found only {len(selected)} unique truncated cases")


def main() -> None:
    config = hosted.PROVIDERS[MODEL]
    client = hosted.make_client(config, timeout=1200.0)
    results = []
    for row, source in select_truncated_cases():
        (
            raw,
            response_id,
            finish,
            controls,
            retries,
        ) = hosted.issue_with_transport_retries(
            client,
            MODEL,
            config,
            row["input"],
            MAX_OUTPUT_TOKENS,
        )
        probability = hosted.registered.parse_yes_probability(raw)
        results.append(
            {
                "task_id": row["task_id"],
                "prompt_sha256": hashlib.sha256(row["input"].encode()).hexdigest(),
                "source_draw_index": source["draw_index"],
                "source_finish_reason": source["finish_reason"],
                "source_max_output_tokens": source["max_output_tokens"],
                "repair_max_output_tokens": MAX_OUTPUT_TOKENS,
                "yes_prob": probability,
                "raw": raw[:8000],
                "response_id": response_id,
                "finish_reason": finish,
                "effective_controls": controls,
                "transport_retries": retries,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    REPORTS.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
    parsed = sum(result["yes_prob"] is not None for result in results)
    print(
        json.dumps(
            {
                "selected": len(results),
                "parsed": parsed,
                "finish_reasons": [result["finish_reason"] for result in results],
                "output": str(OUTPUT),
                "output_sha256": hosted.sha256(OUTPUT),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if parsed != len(results):
        raise RuntimeError("Kimi max-token repair failed its targeted diagnostic")


if __name__ == "__main__":
    main()
