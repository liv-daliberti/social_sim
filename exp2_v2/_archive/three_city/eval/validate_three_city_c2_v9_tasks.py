#!/usr/bin/env python3
"""Validate the frozen natural continuous-profile C2 v9 design."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from build_three_city_c2_v9_tasks import (
    RELEVANCE_HINT_BLOCK,
    strip_relevance_hint,
    target_suffix,
    validate_arm_prompt,
)
from engine.three_city_c2_v9 import (
    CASES_BY_PREFIX,
    CASE_SIGMA,
    CITY_DEVIATION_SD,
    PREFIX_LADDER,
    PROMPT_ARMS,
    cases_for_prefix,
)

_DATA = (
    _ROOT
    / "data"
    / "three_city_c2_v9"
    / "full_k1_5_clean_information"
    / "design"
)
_EPISODES = 120
_TASKS_PER_ARM = _EPISODES * len(PREFIX_LADDER)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _mean_mae(keys: list[dict[str, Any]], method: str, k: int) -> float:
    rows = [row for row in keys if int(row["k"]) == k]
    return statistics.mean(
        abs(
            float(row["baselines"][method]["predicted_poll"])
            - float(row["gold"]["expected_poll"])
        )
        for row in rows
    )


def validate(
    *,
    task_paths: Mapping[str, Path],
    answer_key_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    tasks = {arm: _read_jsonl(path) for arm, path in task_paths.items()}
    keys = _read_jsonl(answer_key_path)
    manifest = json.loads(manifest_path.read_text())
    checks: dict[str, bool] = {}
    diagnostics: dict[str, Any] = {}

    checks["expected_tasks_per_arm"] = all(
        len(rows) == _TASKS_PER_ARM for rows in tasks.values()
    )
    checks["expected_answer_keys"] = len(keys) == _TASKS_PER_ARM
    id_sets = {
        arm: {row["task_id"] for row in rows}
        for arm, rows in tasks.items()
    }
    key_ids = {row["task_id"] for row in keys}
    checks["unique_ids_within_every_file"] = all(
        len(id_sets[arm]) == len(tasks[arm]) for arm in PROMPT_ARMS
    ) and len(key_ids) == len(keys)
    checks["same_ids_across_arms_and_key"] = all(
        values == key_ids for values in id_sets.values()
    )
    checks["model_records_contain_only_prompt_material"] = all(
        set(row) == {"task_id", "prompt", "prompt_sha256"}
        for rows in tasks.values()
        for row in rows
    )
    checks["stored_prompt_hashes_match"] = all(
        row["prompt_sha256"]
        == hashlib.sha256(row["prompt"].encode()).hexdigest()
        for rows in tasks.values()
        for row in rows
    )

    prompt_failures = []
    by_arm_id = {
        arm: {row["task_id"]: row for row in rows}
        for arm, rows in tasks.items()
    }
    for arm, rows in tasks.items():
        for row in rows:
            try:
                validate_arm_prompt(row["prompt"], arm)
            except ValueError as exc:
                prompt_failures.append(f"{arm}:{row['task_id']}:{exc}")
    checks["all_arm_prompts_validate"] = not prompt_failures
    checks["abc_and_relevance_differ_only_by_hint"] = all(
        strip_relevance_hint(by_arm_id["abc_relevance"][task_id]["prompt"])
        == by_arm_id["abc"][task_id]["prompt"]
        and by_arm_id["abc_relevance"][task_id]["prompt"].count(RELEVANCE_HINT_BLOCK) == 1
        for task_id in key_ids
    )
    checks["city_c_information_identical_across_arms"] = all(
        len(
            {
                target_suffix(by_arm_id[arm][task_id]["prompt"])
                for arm in PROMPT_ARMS
            }
        )
        == 1
        for task_id in key_ids
    )
    checks["c_only_hides_references"] = all(
        "CITY A\n" not in row["prompt"] and "CITY B\n" not in row["prompt"]
        for row in tasks["c_only"]
    )
    checks["abc_arms_show_references"] = all(
        "CITY A\n" in row["prompt"] and "CITY B\n" in row["prompt"]
        for arm in ("abc", "abc_relevance")
        for row in tasks[arm]
    )

    checks["expected_120_episodes"] = len(
        {row["episode_id"] for row in keys}
    ) == _EPISODES
    checks["exact_round_ladder"] = {
        int(row["k"]) for row in keys
    } == set(PREFIX_LADDER)
    checks["exact_case_ladder"] = all(
        int(row["target_cases_seen"]) == cases_for_prefix(int(row["k"]))
        and len(row["public"]["target"]["news"])
        == cases_for_prefix(int(row["k"]))
        for row in keys
    )
    grouped = defaultdict(list)
    for row in keys:
        grouped[row["episode_id"]].append(row)
    checks["five_rounds_per_episode"] = all(
        len(rows) == 5 and {int(row["k"]) for row in rows} == set(PREFIX_LADDER)
        for rows in grouped.values()
    )

    base = [row for row in keys if int(row["k"]) == 1]
    triples = [
        (
            float(row["public"]["reference_a"]["profile_index"]),
            float(row["public"]["reference_b"]["profile_index"]),
            float(row["public"]["target"]["profile_index"]),
        )
        for row in base
    ]
    all_scores = [value for triple in triples for value in triple]
    checks["profiles_are_continuous_nonextreme"] = (
        all(0.0 < score < 100.0 for score in all_scores)
        and 0.0 not in all_scores
        and 100.0 not in all_scores
        and len(set(all_scores)) > 250
    )
    checks["target_profile_is_bracketed"] = all(
        min(a, b) < target < max(a, b)
        for a, b, target in triples
    )
    high_label_counts = Counter(
        "A" if a > b else "B" for a, b, _ in triples
    )
    checks["reference_order_balanced_60_60"] = high_label_counts == {
        "A": 60,
        "B": 60,
    }
    target_responses = [float(row["gold"]["target_response"]) for row in base]
    checks["target_responses_are_continuous"] = (
        len(set(target_responses)) == _EPISODES
        and min(target_responses) > -0.2
        and max(target_responses) < 1.5
    )

    target_mae = {
        k: _mean_mae(keys, "target_only", k) for k in PREFIX_LADDER
    }
    abc_mae = {
        k: _mean_mae(keys, "abc_shrinkage", k) for k in PREFIX_LADDER
    }
    gaps = {k: target_mae[k] - abc_mae[k] for k in PREFIX_LADDER}
    weights = {
        k: statistics.mean(
            float(row["baselines"]["abc_shrinkage"]["reference_weight"])
            for row in keys
            if int(row["k"]) == k
        )
        for k in PREFIX_LADDER
    }
    checks["large_round1_information_value"] = gaps[1] > 1.50
    checks["large_round2_information_value"] = gaps[2] > 0.75
    checks["abc_beats_c_only_rounds_1_to_4"] = all(
        abc_mae[k] < target_mae[k] for k in (1, 2, 3, 4)
    )
    checks["matched_convergence_by_round5"] = (
        abs(gaps[5]) < 0.10
        and target_mae[5] < 0.90
        and abc_mae[5] < 0.90
    )
    checks["reference_weight_strictly_decreases"] = all(
        weights[a] > weights[b] for a, b in zip(PREFIX_LADDER, PREFIX_LADDER[1:])
    )
    checks["reference_weight_moves_from_large_to_small"] = (
        weights[1] > 0.70 and weights[5] < 0.25
    )

    checks["manifest_declares_natural_design"] = (
        manifest.get("experiment") == "three_city_c2_v9_clean_information_relevance"
        and manifest.get("arms") == list(PROMPT_ARMS)
        and manifest.get("cases_by_prefix")
        == {str(k): CASES_BY_PREFIX[k] for k in PREFIX_LADDER}
        and manifest.get("continuous_profiles_no_discrete_types") is True
        and float(manifest.get("case_sigma")) == CASE_SIGMA
        and float(manifest.get("city_deviation_sd")) == CITY_DEVIATION_SD
        and manifest.get("tasks_per_arm") == _TASKS_PER_ARM
        and manifest.get("total_calls_three_models")
        == _TASKS_PER_ARM * len(PROMPT_ARMS) * 3
    )
    checks["manifest_task_and_answer_hashes_match"] = (
        all(
            manifest["model_task_files"][arm]["sha256"]
            == _file_sha256(task_paths[arm])
            for arm in PROMPT_ARMS
        )
        and manifest["answer_key_sha256"] == _file_sha256(answer_key_path)
    )
    aggregate_ok = True
    for arm in PROMPT_ARMS:
        digest = hashlib.sha256()
        for row in tasks[arm]:
            digest.update(row["task_id"].encode())
            digest.update(row["prompt"].encode())
        aggregate_ok = aggregate_ok and (
            digest.hexdigest()
            == manifest["model_task_files"][arm]["aggregate_prompt_sha256"]
        )
    checks["manifest_aggregate_prompt_hashes_match"] = aggregate_ok
    source_paths = {
        "engine_sha256": _ROOT / "engine" / "three_city_c2_v9.py",
        "builder_sha256": _HERE / "build_three_city_c2_v9_tasks.py",
        "config_sha256": _ROOT / "configs" / "three_city_c2_v9.yml",
        "validator_sha256": Path(__file__),
        "runner_sha256": _HERE / "run_three_city_c2_v9_confirmatory.py",
        "launcher_sha256": _HERE / "run_three_city_c2_v9_local.sh",
        "analyzer_sha256": _ROOT / "analysis" / "analyze_three_city_c2_v9_confirmatory.py",
        "live_monitor_sha256": _ROOT / "analysis" / "live_monitor_three_city_c2_v9.py",
        "preregistration_sha256": _ROOT / "PREREGISTRATION_THREE_CITY_C2_V9.md",
    }
    checks["manifest_source_hashes_match"] = all(
        manifest[field] == _file_sha256(path)
        for field, path in source_paths.items()
    )

    diagnostics.update(
        {
            "task_counts": {arm: len(rows) for arm, rows in tasks.items()},
            "answer_key_count": len(keys),
            "high_reference_label_counts": dict(high_label_counts),
            "profile_score_range": [min(all_scores), max(all_scores)],
            "target_response_range": [min(target_responses), max(target_responses)],
            "target_only_mae_by_round": target_mae,
            "abc_shrinkage_mae_by_round": abc_mae,
            "information_value_gap_by_round": gaps,
            "abc_reference_weight_by_round": weights,
            "prompt_failures": prompt_failures,
        }
    )
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "experiment": "three_city_c2_v9_clean_information_relevance",
        "status": "passed" if not failures else "failed",
        "checks": checks,
        "diagnostics": diagnostics,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=_DATA)
    args = parser.parse_args()
    task_paths = {
        arm: args.data / f"tasks_c2_v9_{arm}.jsonl"
        for arm in PROMPT_ARMS
    }
    result = validate(
        task_paths=task_paths,
        answer_key_path=args.data / "answer_key_c2_v9.jsonl",
        manifest_path=args.data / "tasks_c2_v9.manifest.json",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    output = args.data / "validation_c2_v9.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output}")
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
