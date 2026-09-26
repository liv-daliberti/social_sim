#!/usr/bin/env python3
"""Validate a frozen three-city C2 task set before any model evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from build_three_city_c2_tasks import validate_structure_blind_prompt
from engine.three_city_c2 import HIGH_TYPE


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _p_correct(record, method):
    result = record["baselines"][method]
    truth_a = record["gold"]["target_matches"] == "A"
    p_a = result["p_match_a"]
    return p_a if truth_a else 1.0 - p_a


def _numeric_public(record):
    public = record["public"]
    return {
        "reference_a": {
            "initial_poll": public["reference_a"]["initial_poll"],
            "news": public["reference_a"]["news"],
            "polls": public["reference_a"]["polls"],
        },
        "reference_b": {
            "initial_poll": public["reference_b"]["initial_poll"],
            "news": public["reference_b"]["news"],
            "polls": public["reference_b"]["polls"],
        },
        "target": {
            "initial_poll": public["target"]["initial_poll"],
            "news": public["target"]["news"],
            "polls": public["target"]["polls"],
        },
        "test_news": public["test_news"],
    }


def _default_answer_key_path(tasks_path: Path) -> Path:
    if tasks_path.name.startswith("tasks_"):
        return tasks_path.with_name(
            tasks_path.name.replace("tasks_", "answer_key_", 1)
        )
    return tasks_path.with_suffix(".answer_key.jsonl")


def validate(
    tasks_path: Path,
    answer_key_path: Path,
    manifest_path: Path,
):
    tasks = [
        json.loads(line)
        for line in tasks_path.read_text().splitlines()
        if line.strip()
    ]
    keys = [
        json.loads(line)
        for line in answer_key_path.read_text().splitlines()
        if line.strip()
    ]
    checks = {}
    checks["task_count_1920"] = len(tasks) == 1920
    checks["unique_task_ids"] = len(
        {record["task_id"] for record in tasks}
    ) == len(tasks)
    checks["unique_answer_key_ids"] = len(
        {record["task_id"] for record in keys}
    ) == len(keys)
    checks["task_and_key_ids_match"] = (
        {record["task_id"] for record in tasks}
        == {record["task_id"] for record in keys}
    )
    checks["model_file_contains_only_prompt_material"] = all(
        set(record) == {"task_id", "prompt", "prompt_sha256"}
        and not any(
            token in record["task_id"]
            for token in ("cue", "high", "low", "type", "match")
        )
        for record in tasks
    )
    checks["stored_prompt_hashes_match"] = all(
        record["prompt_sha256"]
        == hashlib.sha256(record["prompt"].encode("utf-8")).hexdigest()
        for record in tasks
    )
    manifest = json.loads(manifest_path.read_text())
    aggregate_prompt_hash = hashlib.sha256()
    for record in tasks:
        aggregate_prompt_hash.update(record["task_id"].encode("utf-8"))
        aggregate_prompt_hash.update(record["prompt"].encode("utf-8"))
    checks["manifest_task_hash_matches"] = (
        manifest["model_task_file_sha256"] == _file_sha256(tasks_path)
    )
    checks["manifest_answer_key_hash_matches"] = (
        manifest["evaluator_answer_key_sha256"]
        == _file_sha256(answer_key_path)
    )
    checks["manifest_aggregate_prompt_hash_matches"] = (
        manifest["aggregate_prompt_sha256"]
        == aggregate_prompt_hash.hexdigest()
    )
    checks["manifest_declares_structure_blind"] = (
        manifest["structure_disclosed_to_model"] is False
    )
    checks["manifest_counts_match"] = (
        manifest["tasks"] == len(tasks)
        and manifest["episodes"] == 120
        and manifest["prefixes"] == [0, 1, 2, 3]
    )
    source_paths = {
        "engine_sha256": _ROOT / "engine" / "three_city_c2.py",
        "builder_sha256": _HERE / "build_three_city_c2_tasks.py",
        "config_sha256": _ROOT / "configs" / "three_city_c2_v1.yml",
        "validator_sha256": Path(__file__),
    }
    checks["manifest_source_hashes_match"] = all(
        manifest.get(field) == _file_sha256(path)
        for field, path in source_paths.items()
    )

    task_by_id = {record["task_id"]: record for record in tasks}
    records = [
        {**key, **task_by_id[key["task_id"]]}
        for key in keys
        if key["task_id"] in task_by_id
    ]
    checks["unique_episode_count_120"] = len(
        {record["episode_id"] for record in records}
    ) == 120

    disclosure_errors = []
    for record in records:
        try:
            validate_structure_blind_prompt(record["prompt"])
        except ValueError as exc:
            disclosure_errors.append((record["task_id"], str(exc)))
    checks["no_structure_disclosures"] = not disclosure_errors

    by_episode_k = defaultdict(list)
    for record in records:
        by_episode_k[(record["episode_id"], record["k"])].append(record)
    checks["four_contexts_per_episode_prefix"] = all(
        len(group) == 4
        and {record["condition"] for record in group}
        == {"none", "orthogonal", "cue_high", "cue_low"}
        for group in by_episode_k.values()
    )
    checks["context_pairs_hold_numbers_fixed"] = all(
        len(
            {
                json.dumps(_numeric_public(record), sort_keys=True)
                for record in group
            }
        )
        == 1
        for group in by_episode_k.values()
    )

    base = [
        record
        for record in records
        if record["condition"] == "none" and record["k"] == 0
    ]
    target_types = Counter(record["gold"]["target_type"] for record in base)
    matches = Counter(record["gold"]["target_matches"] for record in base)
    signs = Counter(
        1 if record["public"]["test_news"] > 0 else -1
        for record in base
    )
    families = Counter(record["context_family"] for record in base)
    checks["balanced_target_type"] = sorted(target_types.values()) == [60, 60]
    checks["balanced_reference_label"] = sorted(matches.values()) == [60, 60]
    checks["balanced_test_news_sign"] = sorted(signs.values()) == [60, 60]
    checks["balanced_context_family"] = sorted(families.values()) == [40, 40, 40]

    none_by_k = {
        k: [
            record
            for record in records
            if record["condition"] == "none" and record["k"] == k
        ]
        for k in range(4)
    }
    target_only_correct = {
        k: statistics.mean(
            _p_correct(record, "target_only_bayes")
            for record in none_by_k[k]
        )
        for k in range(4)
    }
    checks["k1_is_weakly_informative"] = (
        0.50 < target_only_correct[1] < 0.65
    )
    checks["k2_is_diagnostic"] = target_only_correct[2] > 0.94

    orthogonal_pairs = []
    for episode_id, k in by_episode_k:
        group = {
            record["condition"]: record
            for record in by_episode_k[(episode_id, k)]
        }
        orthogonal_pairs.append(
            group["none"]["baselines"]["context_oracle"]["predicted_poll"]
            == group["orthogonal"]["baselines"]["context_oracle"][
                "predicted_poll"
            ]
        )
    checks["orthogonal_context_invariance"] = all(orthogonal_pairs)

    misleading_correct = {}
    for k in (0, 2):
        selected = []
        for record in records:
            if record["k"] != k:
                continue
            misleading = (
                "cue_low"
                if record["gold"]["target_type"] == HIGH_TYPE
                else "cue_high"
            )
            if record["condition"] == misleading:
                selected.append(
                    _p_correct(record, "context_oracle")
                )
        misleading_correct[k] = statistics.mean(selected)
    checks["misleading_cue_is_initially_influential"] = (
        abs(misleading_correct[0] - 0.2) < 1e-9
    )
    checks["diagnostic_evidence_overrides_misleading_cue"] = (
        misleading_correct[2] > 0.93
    )

    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "diagnostics": {
            "task_count": len(tasks),
            "answer_key_count": len(keys),
            "manifest": str(manifest_path),
            "target_type_counts": target_types,
            "target_match_counts": matches,
            "test_news_sign_counts": signs,
            "context_family_counts": families,
            "target_only_p_correct_by_k": target_only_correct,
            "misleading_context_p_correct": misleading_correct,
            "disclosure_errors": disclosure_errors,
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", type=Path)
    parser.add_argument(
        "--answer-key",
        type=Path,
        default=None,
        help="Evaluator-only answer key; defaults beside the task file.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Frozen manifest; defaults beside the task file.",
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    answer_key = args.answer_key or _default_answer_key_path(args.tasks)
    manifest = args.manifest or args.tasks.with_suffix(".manifest.json")
    report = validate(args.tasks, answer_key, manifest)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n")
        print(f"Wrote {args.report}")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
