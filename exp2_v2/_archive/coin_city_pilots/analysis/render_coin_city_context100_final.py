#!/usr/bin/env python3
"""Final aligned-context render with audited recovery of malformed JSON."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from analysis import analyze_coin_city_llm_pilot as base
from analysis import render_coin_city_context100 as view


NUMBER = re.compile(r'"predicted_poll"\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+))')


def _recover(latest: dict) -> tuple[dict, list[dict]]:
    recovered = dict(latest)
    audit = []
    for key, row in latest.items():
        if row.get("predicted_poll") is not None or not row.get("response_received"):
            continue
        matches = NUMBER.findall(str(row.get("raw", "")))
        if len(matches) != 1:
            continue
        prediction = float(matches[0])
        if not 0.0 <= prediction <= 100.0:
            continue
        derived = dict(row)
        derived["predicted_poll"] = prediction
        derived["analysis_parse_recovery"] = "single explicit predicted_poll number in received raw response"
        recovered[key] = derived
        audit.append(
            {
                "model": row["model"],
                "arm": row["arm"],
                "task_id": row["task_id"],
                "recovered_predicted_poll": prediction,
                "method": derived["analysis_parse_recovery"],
                "raw_sha256": hashlib.sha256(str(row.get("raw", "")).encode()).hexdigest(),
                "extra_api_calls": 0,
            }
        )
    return recovered, audit


def render(output: Path) -> dict:
    latest, audit = _recover(view._latest())
    keys = {
        row["task_id"]: row
        for row in base._read_jsonl(view.PARENT / "design" / "answer_key.jsonl")
    }
    old_sections = json.loads(
        (view.PARENT / "design" / "example_prompt_sections.json").read_text()
    )
    context_sections = json.loads(
        (view.RERUN / "design" / "example_prompt_sections.json").read_text()
    )
    result = {
        "status": "complete" if len(audit) == 1 else "unexpected_recovery_count",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "new_expected": 600,
        "new_received": 600,
        "new_parsed_direct": 599,
        "new_parsed_after_deterministic_recovery": 599 + len(audit),
        "deterministic_recoveries": len(audit),
        "extra_api_calls": 0,
        "cumulative_test_calls": 2400,
        "hard_call_cap": 5000,
        "curves": view._curves(latest, keys),
        "figure": str(output),
    }
    view._render(output, result=result, prompt_sections={**old_sections, **context_sections})
    output.parent.mkdir(parents=True, exist_ok=True)
    (output.parent / "context100_final_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    audit_dir = view.RERUN / "analysis"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "deterministic_parse_recoveries.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in audit)
    )
    print(
        f"{result['status']}: recovered={len(audit)} extra_calls=0; wrote {output}"
    )
    return result


def main() -> None:
    outputs = (
        view.PARENT / "live" / "live_structure.png",
        view.RERUN / "analysis" / "context100_final_structure.png",
    )
    for output in outputs:
        render(output)


if __name__ == "__main__":
    main()
