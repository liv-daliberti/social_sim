#!/usr/bin/env python3
"""Freeze a post-specified strong-structure positive-control arm for v7."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_BASE = _ROOT / "data" / "three_city_c2_v7"
_OUTDIR = _ROOT / "data" / "three_city_c2_v7_strong_hint"
_PREREG = _ROOT / "PREREGISTRATION_THREE_CITY_C2_V7_STRONG_HINT.md"

STRONG_HINT_BLOCK = (
    "Cities in this task follow one of two recurring response patterns, "
    "demonstrated by Cities A and B. City C follows the same response pattern "
    "as one of those cities. Use the background descriptions and City C's "
    "completed cases to infer which pattern applies, then combine that "
    "structure with City C's evidence when forecasting."
)
_INSERTION_MARKER = (
    "Think carefully about the expected poll, but do not provide "
    "step-by-step working."
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def strip_strong_hint(prompt: str) -> str:
    return prompt.replace(STRONG_HINT_BLOCK + "\n\n", "")


def validate_strong_prompt(prompt: str) -> None:
    if prompt.count(STRONG_HINT_BLOCK) != 1:
        raise ValueError("strong structure block must appear exactly once")
    required = (
        "separate polling case",
        "not a time series",
        "vary from case to case",
        "do not provide step-by-step working",
        '"rationale": "one short sentence"',
        '"predicted_poll": <number from 0 to 100>',
    )
    lowered = prompt.lower()
    missing = [phrase for phrase in required if phrase.lower() not in lowered]
    if missing:
        raise ValueError(
            "strong prompt is missing common material: " + ", ".join(missing)
        )


def build_strong_prompt(blind_prompt: str) -> str:
    if STRONG_HINT_BLOCK in blind_prompt:
        raise ValueError("blind prompt already contains strong structure block")
    if blind_prompt.count(_INSERTION_MARKER) != 1:
        raise ValueError("common reasoning instruction not found exactly once")
    strong = blind_prompt.replace(
        _INSERTION_MARKER,
        STRONG_HINT_BLOCK + "\n\n" + _INSERTION_MARKER,
        1,
    )
    validate_strong_prompt(strong)
    if strip_strong_hint(strong) != blind_prompt:
        raise ValueError("strong prompt differs beyond the inserted block")
    return strong


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--blind-tasks",
        type=Path,
        default=_BASE / "tasks_c2_v7_blind.jsonl",
    )
    parser.add_argument(
        "--answer-key",
        type=Path,
        default=_BASE / "answer_key_c2_v7.jsonl",
    )
    parser.add_argument("--outdir", type=Path, default=_OUTDIR)
    parser.add_argument("--preregistration", type=Path, default=_PREREG)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    blind = _read_jsonl(args.blind_tasks)
    answer_key = _read_jsonl(args.answer_key)
    records = []
    for source in blind:
        if set(source) != {"task_id", "prompt", "prompt_sha256"}:
            raise ValueError("blind task contains evaluator material")
        prompt = build_strong_prompt(source["prompt"])
        records.append(
            {
                "task_id": source["task_id"],
                "prompt": prompt,
                "prompt_sha256": _prompt_sha256(prompt),
            }
        )
    if {row["task_id"] for row in records} != {
        row["task_id"] for row in answer_key
    }:
        raise ValueError("strong task IDs do not match the shared answer key")
    if args.dry_run:
        print(records[0]["prompt"])
        return

    task_path = args.outdir / "tasks_c2_v7_strong_hint.jsonl"
    manifest_path = args.outdir / "tasks_c2_v7_strong_hint.manifest.json"
    if (task_path.exists() or manifest_path.exists()) and not args.overwrite:
        parser.error("strong-arm output exists; pass --overwrite intentionally")
    args.outdir.mkdir(parents=True, exist_ok=True)
    with task_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    source_paths = {
        "builder_sha256": Path(__file__),
        "validator_sha256": (
            _HERE / "validate_three_city_c2_v7_strong_hint_tasks.py"
        ),
        "runner_sha256": (
            _HERE / "run_three_city_c2_v7_strong_hint.py"
        ),
        "launcher_sha256": (
            _HERE / "slurm_three_city_c2_v7_strong_hint.sh"
        ),
        "analyzer_sha256": (
            _ROOT / "analysis" / "analyze_three_city_c2_v7_strong_hint.py"
        ),
        "preregistration_sha256": args.preregistration,
    }
    manifest = {
        "experiment": "three_city_c2_v7_strong_hint",
        "status": "frozen_postspecified_positive_control_no_model_runs",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "arm": "strong_hint",
        "postspecified_after_interim_view": True,
        "tasks": len(records),
        "strong_hint_block": STRONG_HINT_BLOCK,
        "source_blind_task_file": str(args.blind_tasks),
        "source_blind_task_file_sha256": _file_sha256(args.blind_tasks),
        "shared_answer_key": str(args.answer_key),
        "shared_answer_key_sha256": _file_sha256(args.answer_key),
        "model_task_file": task_path.name,
        "model_task_file_sha256": _file_sha256(task_path),
        **{
            field: _file_sha256(path)
            for field, path in source_paths.items()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {len(records)} strong-hint tasks to {task_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
