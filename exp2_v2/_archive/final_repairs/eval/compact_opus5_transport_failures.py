#!/usr/bin/env python3
"""Drop placeholder records for requests the service rejected before the model ran.

The Coin City harness writes one terminal record per task, so a request rejected
at the gateway (HTTP 429/5xx, ``response_received`` false, empty ``raw``) would
retire that task without the model ever seeing it. This maintenance pass removes
exactly those placeholders so the shards re-issue the tasks. It refuses to touch
any record that carries a model response, which is what keeps the single-model-
attempt invariant intact: no task is ever asked twice.

Run with no shards active. Writes an audit summary next to the response files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from engine.coin_city_stable_relationship_claude_n250 import (  # noqa: E402
    EXPERIMENT,
    PROMPT_ARMS,
)


MODEL = "claude-opus-5"
RUN = ROOT / "data" / EXPERIMENT
RESPONSES = RUN / "responses"
RETRYABLE_STATUS = ("429", "500", "502", "503", "504")


def _is_transport_placeholder(record: dict) -> bool:
    if record.get("predicted_poll") is not None:
        return False
    if record.get("response_received") is True:
        return False
    if (record.get("raw") or "").strip():
        return False
    error = record.get("error") or ""
    return any(f"Error code: {status}" in error for status in RETRYABLE_STATUS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="rewrite the files")
    args = parser.parse_args()

    summary = {"model": MODEL, "applied": args.apply, "arms": {}}
    for arm in PROMPT_ARMS:
        path = RESPONSES / f"responses_{MODEL}_{arm}.jsonl"
        if not path.exists():
            summary["arms"][arm] = {"status": "missing"}
            continue
        records = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        kept = [row for row in records if not _is_transport_placeholder(row)]
        dropped = len(records) - len(kept)
        with_response = sum(row.get("response_received") is True for row in kept)
        unparsed_with_response = sum(
            row.get("response_received") is True and row.get("predicted_poll") is None
            for row in kept
        )
        summary["arms"][arm] = {
            "records_before": len(records),
            "records_after": len(kept),
            "dropped_transport_placeholders": dropped,
            "kept_with_model_response": with_response,
            "kept_unparseable_model_responses": unparsed_with_response,
        }
        if args.apply and dropped:
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in kept)
            )
    summary["total_dropped"] = sum(
        arm.get("dropped_transport_placeholders", 0)
        for arm in summary["arms"].values()
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.apply:
        audit = RESPONSES / f"responses_{MODEL}_transport_compaction.json"
        history = []
        if audit.exists():
            history = json.loads(audit.read_text()).get("passes", [])
        history.append(summary)
        audit.write_text(
            json.dumps({"passes": history}, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
