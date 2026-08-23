#!/usr/bin/env python3
"""Freeze an auditable exact-n=50 view after the user-directed early stop."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "eval"))

from engine.three_city_c2_v8 import PREFIX_LADDER, PROMPT_ARMS
from eval.run_three_city_c2_v8_confirmatory import (
    MODEL_CONFIG,
    _ORDER_SEED,
    _select_tasks,
    _slug,
)

_DATA = _ROOT / "data" / "three_city_c2_v8"
_RAW = _DATA / "confirmatory"
_OUT = _DATA / "stopped_n50"
_PRIMARY_SUFFIX = "_v0_k2"
_TASK_EPISODE = re.compile(r"^(c2v8_\d{4})_v[0-3]_k(?:0|1|2|3|4|6|8)$")
_TARGET_N = 50


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl_tolerant(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def _latest_by_task(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for source in records:
        record = dict(source)
        task_id = record.get("task_id")
        if not isinstance(task_id, str):
            continue
        priority = (
            2
            if record.get("predicted_poll") is not None
            else (1 if record.get("response_received") is True else 0)
        )
        old = latest.get(task_id)
        old_priority = (
            -1
            if old is None
            else (
                2
                if old.get("predicted_poll") is not None
                else (1 if old.get("response_received") is True else 0)
            )
        )
        if priority >= old_priority:
            latest[task_id] = record
    return latest


def _episode_id(task_id: str) -> str:
    match = _TASK_EPISODE.fullmatch(task_id)
    if match is None:
        raise ValueError(f"unexpected task ID: {task_id!r}")
    return match.group(1)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    task_order = _select_tasks(
        _DATA / "tasks_c2_v8_blind.jsonl",
        arm="blind",
        episodes=range(120),
        conditions=range(4),
        prefixes=PREFIX_LADDER,
    )
    order_index = {
        task["task_id"]: index for index, task in enumerate(task_order)
    }

    output_responses = _OUT / "responses"
    output_responses.mkdir(parents=True, exist_ok=True)
    raw_file_audit = {}
    model_audit = {}
    selections: dict[str, set[str]] = {}

    for model in MODEL_CONFIG:
        slug = _slug(model)
        latest_by_arm = {}
        for arm in PROMPT_ARMS:
            path = _RAW / f"responses_{slug}_{arm}.jsonl"
            records = _read_jsonl_tolerant(path)
            latest = _latest_by_task(records)
            latest_by_arm[arm] = latest
            raw_file_audit[path.name] = {
                "bytes": path.stat().st_size,
                "lines": len(records),
                "latest_tasks": len(latest),
                "received": sum(
                    row.get("response_received") is True
                    or row.get("predicted_poll") is not None
                    for row in latest.values()
                ),
                "parsed": sum(
                    row.get("predicted_poll") is not None
                    for row in latest.values()
                ),
                "sha256": _sha256(path),
            }

        eligible_by_arm = {
            arm: {
                task_id
                for task_id, row in latest_by_arm[arm].items()
                if task_id.endswith(_PRIMARY_SUFFIX)
                and row.get("predicted_poll") is not None
            }
            for arm in PROMPT_ARMS
        }
        common = set.intersection(*eligible_by_arm.values())
        ordered = sorted(common, key=order_index.__getitem__)
        if len(ordered) < _TARGET_N:
            raise SystemExit(
                f"{model} has only {len(ordered)} complete primary pairs"
            )
        selected_tasks = ordered[:_TARGET_N]
        selected_episodes = {_episode_id(task_id) for task_id in selected_tasks}
        if len(selected_episodes) != _TARGET_N:
            raise AssertionError("primary task IDs are not one per episode")
        selections[model] = selected_episodes

        filtered_counts = {}
        for arm in PROMPT_ARMS:
            selected_records = [
                row
                for task_id, row in latest_by_arm[arm].items()
                if _episode_id(task_id) in selected_episodes
            ]
            selected_records.sort(
                key=lambda row: order_index[row["task_id"]]
            )
            destination = (
                output_responses / f"responses_{slug}_{arm}.jsonl"
            )
            temporary = destination.with_suffix(".tmp.jsonl")
            with temporary.open("w") as handle:
                for row in selected_records:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
            os.replace(temporary, destination)
            filtered_counts[arm] = {
                "records": len(selected_records),
                "parsed": sum(
                    row.get("predicted_poll") is not None
                    for row in selected_records
                ),
                "sha256": _sha256(destination),
            }

        model_audit[model] = {
            "raw_complete_paired_primary_n": len(ordered),
            "selected_primary_n": _TARGET_N,
            "selected_primary_task_ids_in_frozen_order": selected_tasks,
            "selected_episode_ids": sorted(selected_episodes),
            "excluded_overrun_primary_task_ids": ordered[_TARGET_N:],
            "eligible_parsed_primary_by_arm": {
                arm: len(values) for arm, values in eligible_by_arm.items()
            },
            "filtered_response_files": filtered_counts,
        }

    audit = {
        "experiment": "three_city_c2_v8",
        "status": "user_directed_early_stop_exact_n50_analysis_view",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "slurm_array_job": "30159738",
        "requested_primary_n_per_model": _TARGET_N,
        "preregistered_primary_n_per_model": 120,
        "primary_condition": "none",
        "primary_k": 2,
        "selection_rule": (
            "For each model separately, intersect successfully parsed k=2 "
            "no-context task IDs across all three prompt arms; order that "
            "intersection by the frozen runner shuffle (seed 20260729); retain "
            "the first 50. Retain all available task cells for those episode "
            "IDs in the stopped-n50 analysis view. Raw response files are "
            "preserved unchanged."
        ),
        "runner_order_seed": _ORDER_SEED,
        "models": model_audit,
        "raw_response_files": raw_file_audit,
        "source_hashes": {
            "tasks_manifest": _sha256(
                _DATA / "tasks_c2_v8.manifest.json"
            ),
            "validation_report": _sha256(
                _DATA / "validation_c2_v8.json"
            ),
            "freezer": _sha256(Path(__file__)),
        },
    }
    _OUT.mkdir(parents=True, exist_ok=True)
    _atomic_json(_OUT / "stopping_audit_n50.json", audit)

    markdown = [
        "# Three-city C2 v8: user-directed early-stop audit",
        "",
        (
            "The preregistered 120-episode collection was stopped at the "
            "user's request after the main paired no-context k=2 sample "
            "reached approximately 50 per model. This is therefore an early-"
            "stopped analysis and is not labelled the preregistered final "
            "confirmatory result."
        ),
        "",
        "| Model | Raw complete paired n | Retained n | Overrun excluded |",
        "|---|---:|---:|---:|",
    ]
    for model, record in model_audit.items():
        markdown.append(
            f"| {model} | {record['raw_complete_paired_primary_n']} | "
            f"{record['selected_primary_n']} | "
            f"{len(record['excluded_overrun_primary_task_ids'])} |"
        )
    markdown.extend(
        [
            "",
            "## Frozen selection rule",
            "",
            audit["selection_rule"],
            "",
            (
                "All raw completions, including in-flight overrun, remain in "
                "`../confirmatory/`. The filtered exact-n=50 view is in "
                "`responses/`; no raw file was modified or deleted."
            ),
            "",
        ]
    )
    markdown_path = _OUT / "STOPPING_AUDIT_N50.md"
    temporary_markdown = markdown_path.with_suffix(".tmp.md")
    temporary_markdown.write_text("\n".join(markdown))
    os.replace(temporary_markdown, markdown_path)
    print(json.dumps(
        {
            model: {
                "raw_n": record["raw_complete_paired_primary_n"],
                "retained_n": record["selected_primary_n"],
                "overrun": len(record["excluded_overrun_primary_task_ids"]),
            }
            for model, record in model_audit.items()
        },
        indent=2,
    ))
    print(f"Wrote {_OUT / 'stopping_audit_n50.json'}")
    print(f"Wrote {markdown_path}")
    print(f"Wrote filtered responses to {output_responses}")


if __name__ == "__main__":
    main()
