#!/usr/bin/env python3
"""Validate DeepSeek responses against the frozen Coin City task contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from engine.coin_city_stable_relationship_claude_n250 import (
    C_CASE_LEVELS,
    EPISODES,
    EXPERIMENT,
    PROMPT_ARMS,
)


MODEL = "DeepSeek-V4-Pro"
ENDPOINT = "https://liv.services.ai.azure.com/openai/v1"
RUN = ROOT / "data" / EXPERIMENT
DESIGN = RUN / "design"
RESPONSES = RUN / "responses"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    expected_per_arm = EPISODES * len(C_CASE_LEVELS)
    report = {
        "experiment": EXPERIMENT,
        "model": MODEL,
        "expected_per_arm": expected_per_arm,
        "expected_total": expected_per_arm * len(PROMPT_ARMS),
        "arms": {},
    }
    all_passed = True
    for arm in PROMPT_ARMS:
        tasks = _read_jsonl(DESIGN / f"tasks_{arm}.jsonl")
        task_hash = {row["task_id"]: row["prompt_sha256"] for row in tasks}
        response_path = RESPONSES / f"responses_{MODEL}_{arm}.jsonl"
        manifest_path = RESPONSES / f"responses_{MODEL}_{arm}.manifest.json"
        if not response_path.exists() or not manifest_path.exists():
            report["arms"][arm] = {"status": "missing"}
            all_passed = False
            continue
        responses = _read_jsonl(response_path)
        manifest = json.loads(manifest_path.read_text())
        ids = [row.get("task_id") for row in responses]
        unique_ids = set(ids)
        parsed = sum(row.get("predicted_poll") is not None for row in responses)
        received = sum(row.get("response_received") is True for row in responses)
        checks = {
            "record_count": len(responses) == expected_per_arm,
            "unique_task_ids": len(unique_ids) == expected_per_arm,
            "exact_task_ids": unique_ids == set(task_hash),
            "prompt_hashes": all(
                row.get("prompt_sha256") == task_hash.get(row.get("task_id"))
                for row in responses
            ),
            "model": all(row.get("model") == MODEL for row in responses),
            "one_attempt": all(row.get("attempts") == 1 for row in responses),
            "no_secret_fields": all(
                not (set(row) & {"api_key", "token", "authorization"})
                for row in responses
            ),
            "manifest_complete": manifest.get("status") == "complete",
            "manifest_endpoint": manifest.get("endpoint") == ENDPOINT,
            "parseable_at_least_99_9_percent": parsed >= expected_per_arm - 1,
        }
        status = "passed" if all(checks.values()) else "failed"
        if status != "passed":
            all_passed = False
        report["arms"][arm] = {
            "status": status,
            "checks": checks,
            "records": len(responses),
            "responses_received": received,
            "successfully_parsed": parsed,
            "missing_or_failed": expected_per_arm - parsed,
        }
    report["status"] = "passed" if all_passed else "failed"
    print(json.dumps(report, indent=2, sort_keys=True))
    if not all_passed:
        raise SystemExit(f"{MODEL} Coin City validation failed")


if __name__ == "__main__":
    main()
