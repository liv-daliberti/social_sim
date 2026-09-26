#!/usr/bin/env python3
"""Validate the frozen v7 strong-structure positive-control task file."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from build_three_city_c2_v7_strong_hint_tasks import (
    STRONG_HINT_BLOCK,
    strip_strong_hint,
    validate_strong_prompt,
)

_BASE = _ROOT / "data" / "three_city_c2_v7"
_DATA = _ROOT / "data" / "three_city_c2_v7_strong_hint"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--blind",
        type=Path,
        default=_BASE / "tasks_c2_v7_blind.jsonl",
    )
    parser.add_argument(
        "--strong",
        type=Path,
        default=_DATA / "tasks_c2_v7_strong_hint.jsonl",
    )
    parser.add_argument(
        "--answer-key",
        type=Path,
        default=_BASE / "answer_key_c2_v7.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_DATA / "tasks_c2_v7_strong_hint.manifest.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DATA / "validation_c2_v7_strong_hint.json",
    )
    args = parser.parse_args()

    blind = _read_jsonl(args.blind)
    strong = _read_jsonl(args.strong)
    keys = _read_jsonl(args.answer_key)
    manifest = json.loads(args.manifest.read_text())
    failures: list[str] = []
    checks: dict[str, bool] = {}

    blind_by_id = {row["task_id"]: row for row in blind}
    strong_by_id = {row["task_id"]: row for row in strong}
    key_ids = {row["task_id"] for row in keys}
    ids_valid = (
        set(blind_by_id) == set(strong_by_id) == key_ids
        and len(blind_by_id) == len(blind)
        and len(strong_by_id) == len(strong)
        and len(strong) == 1440
    )
    checks["same_unique_1440_task_ids"] = ids_valid
    if not ids_valid:
        failures.append("task ID sets/counts are invalid")

    pair_failures = []
    prompt_hash_failures = []
    validation_failures = []
    for task_id in sorted(set(blind_by_id) & set(strong_by_id)):
        source = blind_by_id[task_id]
        control = strong_by_id[task_id]
        if strip_strong_hint(control["prompt"]) != source["prompt"]:
            pair_failures.append(task_id)
        if control["prompt"].count(STRONG_HINT_BLOCK) != 1:
            pair_failures.append(task_id)
        if hashlib.sha256(
            control["prompt"].encode("utf-8")
        ).hexdigest() != control["prompt_sha256"]:
            prompt_hash_failures.append(task_id)
        try:
            validate_strong_prompt(control["prompt"])
        except ValueError as exc:
            validation_failures.append(f"{task_id}:{exc}")
    checks["diff_is_exactly_strong_structure_block"] = not pair_failures
    checks["prompt_hashes_valid"] = not prompt_hash_failures
    checks["common_prompt_material_valid"] = not validation_failures
    if pair_failures:
        failures.append("strong prompts differ beyond the frozen block")
    if prompt_hash_failures:
        failures.append("strong prompt hashes are invalid")
    if validation_failures:
        failures.append("strong prompts fail common-material validation")

    model_records_clean = all(
        set(row) == {"task_id", "prompt", "prompt_sha256"}
        for row in strong
    )
    checks["model_records_contain_only_prompt_material"] = (
        model_records_clean
    )
    if not model_records_clean:
        failures.append("strong model task contains evaluator fields")

    source_paths = {
        "source_blind_task_file_sha256": args.blind,
        "shared_answer_key_sha256": args.answer_key,
        "model_task_file_sha256": args.strong,
        "builder_sha256": (
            _HERE / "build_three_city_c2_v7_strong_hint_tasks.py"
        ),
        "validator_sha256": Path(__file__),
        "runner_sha256": _HERE / "run_three_city_c2_v7_strong_hint.py",
        "launcher_sha256": (
            _HERE / "slurm_three_city_c2_v7_strong_hint.sh"
        ),
        "analyzer_sha256": (
            _ROOT / "analysis" / "analyze_three_city_c2_v7_strong_hint.py"
        ),
        "preregistration_sha256": (
            _ROOT / "PREREGISTRATION_THREE_CITY_C2_V7_STRONG_HINT.md"
        ),
    }
    hash_failures = [
        field
        for field, path in source_paths.items()
        if manifest.get(field) != _file_sha256(path)
    ]
    checks["frozen_manifest_hashes_valid"] = not hash_failures
    if hash_failures:
        failures.append(
            "manifest hash mismatch: " + ", ".join(hash_failures)
        )

    result = {
        "experiment": "three_city_c2_v7_strong_hint",
        "valid": not failures,
        "failures": failures,
        "checks": checks,
        "counts": {"tasks": len(strong)},
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
