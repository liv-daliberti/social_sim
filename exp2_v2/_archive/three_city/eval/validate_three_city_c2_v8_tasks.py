#!/usr/bin/env python3
"""Validate the full local five-round similarity-informed C2 v8 design."""

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

from build_three_city_c2_v8_tasks import (
    HINT_BLOCK,
    STRONG_HINT_BLOCK,
    strip_arm_block,
    validate_arm_prompt,
)
from engine.three_city_c2_v8 import (
    CASES_BY_PREFIX,
    CONTEXT_CONDITIONS,
    HIGH_TYPE,
    PREFIX_LADDER,
    PROMPT_ARMS,
    cases_for_prefix,
)

_DATA = _ROOT / "data" / "three_city_c2_v8" / "full_k1_5_similarity" / "design"
_EPISODES = 120
_TASKS_PER_ARM = _EPISODES * len(CONTEXT_CONDITIONS) * len(PREFIX_LADDER)


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


def _numeric_public(record: Mapping[str, Any]) -> dict[str, Any]:
    public = record["public"]
    return {
        "reference_a": {
            "profile_index": public["reference_a"]["profile_index"],
            "starting_poll": public["reference_a"]["starting_poll"],
            "news": public["reference_a"]["news"],
            "ending_polls": public["reference_a"]["ending_polls"],
        },
        "reference_b": {
            "profile_index": public["reference_b"]["profile_index"],
            "starting_poll": public["reference_b"]["starting_poll"],
            "news": public["reference_b"]["news"],
            "ending_polls": public["reference_b"]["ending_polls"],
        },
        "target": {
            "profile_index": public["target"]["profile_index"],
            "starting_poll": public["target"]["starting_poll"],
            "news": public["target"]["news"],
            "ending_polls": public["target"]["ending_polls"],
        },
        "target_cases_seen": record["target_cases_seen"],
        "test_starting_poll": public["test_starting_poll"],
        "test_news": public["test_news"],
    }


def _mean_mae(
    keys: list[dict[str, Any]],
    method: str,
    k: int,
) -> float:
    rows = [
        row
        for row in keys
        if row["condition"] == "none" and row["k"] == k
    ]
    return statistics.mean(
        abs(
            float(row["baselines"][method]["predicted_poll"])
            - float(row["gold"]["expected_poll"])
        )
        for row in rows
    )


def _p_correct(record: Mapping[str, Any], method: str) -> float:
    result = record["baselines"][method]
    truth_a = record["gold"]["target_matches"] == "A"
    p_a = float(result["p_match_a"])
    return p_a if truth_a else 1.0 - p_a


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
        ids == key_ids for ids in id_sets.values()
    )
    checks["model_records_contain_only_prompt_material"] = all(
        set(row) == {"task_id", "prompt", "prompt_sha256"}
        for rows in tasks.values()
        for row in rows
    )
    checks["stored_prompt_hashes_match"] = all(
        row["prompt_sha256"]
        == hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest()
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
                prompt_failures.append(
                    f"{arm}:{row['task_id']}:{exc}"
                )
    checks["all_arm_prompts_validate"] = not prompt_failures
    exact_diff_failures = []
    for task_id in sorted(key_ids):
        blind = by_arm_id["blind"][task_id]["prompt"]
        hint = by_arm_id["hint"][task_id]["prompt"]
        strong = by_arm_id["strong_hint"][task_id]["prompt"]
        if strip_arm_block(hint, "hint") != blind:
            exact_diff_failures.append(f"hint:{task_id}")
        if strip_arm_block(strong, "strong_hint") != blind:
            exact_diff_failures.append(f"strong:{task_id}")
        if hint.count(HINT_BLOCK) != 1 or strong.count(STRONG_HINT_BLOCK) != 1:
            exact_diff_failures.append(f"block:{task_id}")
    checks["arms_differ_only_by_frozen_blocks"] = not exact_diff_failures

    checks["expected_120_episodes"] = len(
        {row["episode_id"] for row in keys}
    ) == _EPISODES
    checks["exact_prefix_ladder"] = {
        int(row["k"]) for row in keys
    } == set(PREFIX_LADDER)
    checks["exact_evidence_round_case_ladder"] = all(
        int(row["target_cases_seen"]) == cases_for_prefix(int(row["k"]))
        and len(row["public"]["target"]["news"]) == cases_for_prefix(int(row["k"]))
        for row in keys
    )
    checks["exact_context_set"] = {
        row["condition"] for row in keys
    } == set(CONTEXT_CONDITIONS)

    by_episode_k = defaultdict(list)
    for row in keys:
        by_episode_k[(row["episode_id"], row["k"])].append(row)
    checks["four_contexts_per_episode_prefix"] = all(
        len(group) == 4
        and {row["condition"] for row in group}
        == set(CONTEXT_CONDITIONS)
        for group in by_episode_k.values()
    )
    checks["contexts_hold_all_numbers_fixed"] = all(
        len(
            {
                json.dumps(_numeric_public(row), sort_keys=True)
                for row in group
            }
        )
        == 1
        for group in by_episode_k.values()
    )

    base = [
        row
        for row in keys
        if row["condition"] == "none" and row["k"] == PREFIX_LADDER[0]
    ]
    target_types = Counter(row["gold"]["target_type"] for row in base)
    target_matches = Counter(row["gold"]["target_matches"] for row in base)
    families = Counter(row["context_family"] for row in base)
    checks["balanced_target_type_60_60"] = sorted(
        target_types.values()
    ) == [60, 60]
    checks["balanced_reference_label_60_60"] = sorted(
        target_matches.values()
    ) == [60, 60]
    checks["balanced_context_families_40_each"] = sorted(
        families.values()
    ) == [40, 40, 40]
    target_profiles = [
        float(row["public"]["target"]["profile_index"]) for row in base
    ]
    profile_midpoint_crossings = sum(
        (row["gold"]["target_type"] == HIGH_TYPE and score < 50.0)
        or (row["gold"]["target_type"] != HIGH_TYPE and score > 50.0)
        for row, score in zip(base, target_profiles)
    )
    checks["profile_indices_are_bounded_and_graded"] = (
        all(0.0 <= score <= 100.0 for score in target_profiles)
        and len(set(target_profiles)) > 80
    )
    checks["profile_signal_is_informative_but_overlapping"] = (
        profile_midpoint_crossings == 14
    )

    cross = Counter()
    for row in keys:
        if row["k"] != PREFIX_LADDER[0] or row["condition"] not in ("cue_high", "cue_low"):
            continue
        aligned = (
            row["condition"] == "cue_high"
            and row["gold"]["target_type"] == HIGH_TYPE
        ) or (
            row["condition"] == "cue_low"
            and row["gold"]["target_type"] != HIGH_TYPE
        )
        cross["aligned" if aligned else "misleading"] += 1
    checks["high_low_cues_crossed_with_truth"] = cross == {
        "aligned": 120,
        "misleading": 120,
    }

    target_mae = {
        str(k): _mean_mae(keys, "target_only", k)
        for k in PREFIX_LADDER
    }
    abc_mae = {
        str(k): _mean_mae(keys, "abc_shrinkage", k)
        for k in PREFIX_LADDER
    }
    checks["large_initial_similarity_headroom_at_k1"] = (
        target_mae["1"] - abc_mae["1"] > 1.25
    )
    checks["city_c_only_improves_by_k5"] = (
        target_mae["5"] < 0.75 and target_mae["5"] < target_mae["1"]
    )
    checks["abc_shrinkage_has_sparse_headroom_at_k2"] = (
        target_mae["2"] - abc_mae["2"] > 0.65
    )
    checks["matched_estimators_are_close_by_k5"] = (
        abs(abc_mae["5"] - target_mae["5"]) < 0.1
        and abc_mae["5"] < 0.75
    )
    checks["abc_beats_city_c_only_at_every_round"] = all(
        abc_mae[str(k)] < target_mae[str(k)] for k in PREFIX_LADDER
    )
    mean_reference_weight = {
        str(k): statistics.mean(
            float(row["baselines"]["abc_shrinkage"]["reference_weight"])
            for row in keys
            if row["condition"] == "none" and row["k"] == k
        )
        for k in PREFIX_LADDER
    }
    checks["abc_reference_weight_strictly_decreases"] = all(
        mean_reference_weight[str(current)]
        < mean_reference_weight[str(previous)]
        for previous, current in zip(PREFIX_LADDER, PREFIX_LADDER[1:])
    )
    checks["abc_reference_weight_moves_from_strong_to_small"] = (
        mean_reference_weight["1"] > 0.70
        and mean_reference_weight["5"] < 0.20
    )

    misleading_correct = {}
    for k in (1, 2, 5):
        selected = []
        for row in keys:
            if row["k"] != k:
                continue
            wrong = (
                "cue_low"
                if row["gold"]["target_type"] == HIGH_TYPE
                else "cue_high"
            )
            if row["condition"] == wrong:
                selected.append(
                    _p_correct(row, "privileged_context_oracle")
                )
        misleading_correct[str(k)] = statistics.mean(selected)
    checks["privileged_wrong_cue_is_overridden_from_k1"] = (
        misleading_correct["5"] > misleading_correct["1"]
    )
    checks["privileged_wrong_cue_is_overridden_by_k5"] = (
        misleading_correct["5"] > 0.9
    )

    orthogonal_equal = []
    for group in by_episode_k.values():
        conditions = {row["condition"]: row for row in group}
        orthogonal_equal.append(
            conditions["none"]["baselines"]["privileged_context_oracle"][
                "predicted_poll"
            ]
            == conditions["orthogonal"]["baselines"][
                "privileged_context_oracle"
            ]["predicted_poll"]
        )
    checks["privileged_oracle_ignores_orthogonal_context"] = all(
        orthogonal_equal
    )

    relevant_lengths = []
    orthogonal_lengths = []
    for row in keys:
        if row["k"] != PREFIX_LADDER[0]:
            continue
        length = len(row["public"]["target"]["background"].split())
        if row["condition"] in ("cue_high", "cue_low"):
            relevant_lengths.append(length)
        elif row["condition"] == "orthogonal":
            orthogonal_lengths.append(length)
    checks["relevant_and_orthogonal_context_lengths_matched"] = (
        abs(
            statistics.mean(relevant_lengths)
            - statistics.mean(orthogonal_lengths)
        )
        < 1.0
        and max(relevant_lengths) - min(relevant_lengths) <= 2
        and max(orthogonal_lengths) - min(orthogonal_lengths) <= 1
    )

    checks["manifest_declares_correct_fixed_design"] = (
        manifest.get("episodes") == _EPISODES
        and manifest.get("tasks_per_arm") == _TASKS_PER_ARM
        and manifest.get("calls_per_model") == _TASKS_PER_ARM * len(PROMPT_ARMS)
        and manifest.get("total_calls_three_models") == _TASKS_PER_ARM * len(PROMPT_ARMS) * 3
        and manifest.get("primary_condition") == "none"
        and manifest.get("primary_k") == 2
        and manifest.get("convergence_k") == 5
        and manifest.get("cases_by_prefix")
        == {str(k): CASES_BY_PREFIX[k] for k in PREFIX_LADDER}
        and manifest.get("profile_index_is_graded_and_overlapping") is True
        and manifest.get("context_cues_crossed_with_truth") is True
    )
    checks["manifest_task_and_answer_hashes_match"] = (
        manifest.get("answer_key_sha256") == _file_sha256(answer_key_path)
        and all(
            manifest["model_task_files"][arm]["sha256"]
            == _file_sha256(task_paths[arm])
            for arm in PROMPT_ARMS
        )
    )
    aggregate_checks = []
    for arm in PROMPT_ARMS:
        digest = hashlib.sha256()
        for row in tasks[arm]:
            digest.update(row["task_id"].encode("utf-8"))
            digest.update(row["prompt"].encode("utf-8"))
        aggregate_checks.append(
            manifest["model_task_files"][arm][
                "aggregate_prompt_sha256"
            ]
            == digest.hexdigest()
        )
    checks["manifest_aggregate_prompt_hashes_match"] = all(aggregate_checks)
    source_paths = {
        "engine_sha256": _ROOT / "engine" / "three_city_c2_v8.py",
        "source_engine_v2_sha256": (
            _ROOT / "engine" / "three_city_c2_v2.py"
        ),
        "builder_sha256": (
            _HERE / "build_three_city_c2_v8_tasks.py"
        ),
        "config_sha256": _ROOT / "configs" / "three_city_c2_v8_k1_5_similarity.yml",
        "validator_sha256": Path(__file__),
        "runner_sha256": (
            _HERE / "run_three_city_c2_v8_confirmatory.py"
        ),
        "launcher_sha256": (
            _HERE / "run_three_city_c2_v8_local.sh"
        ),
        "analyzer_sha256": (
            _ROOT / "analysis" / "analyze_three_city_c2_v8_confirmatory.py"
        ),
        "preregistration_sha256": (
            _ROOT / "PREREGISTRATION_THREE_CITY_C2_V8_K1_5_SIMILARITY.md"
        ),
    }
    checks["manifest_source_hashes_match"] = all(
        manifest.get(field) == _file_sha256(path)
        for field, path in source_paths.items()
    )

    diagnostics.update(
        {
            "task_counts": {
                arm: len(rows) for arm, rows in tasks.items()
            },
            "answer_key_count": len(keys),
            "target_type_counts": target_types,
            "target_match_counts": target_matches,
            "context_family_counts": families,
            "context_truth_cross": cross,
            "evidence_round_cases": {str(k): CASES_BY_PREFIX[k] for k in PREFIX_LADDER},
            "profile_midpoint_crossings": profile_midpoint_crossings,
            "target_profile_indices": target_profiles,
            "target_only_mae_by_k": target_mae,
            "abc_shrinkage_mae_by_k": abc_mae,
            "abc_mean_reference_weight_by_k": mean_reference_weight,
            "privileged_wrong_cue_p_correct": misleading_correct,
            "mean_relevant_context_words": statistics.mean(
                relevant_lengths
            ),
            "mean_orthogonal_context_words": statistics.mean(
                orthogonal_lengths
            ),
            "prompt_failures": prompt_failures[:20],
            "exact_diff_failures": exact_diff_failures[:20],
        }
    )
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "experiment": "three_city_c2_v8_full_k1_5_similarity_matched",
        "status": "passed" if not failures else "failed",
        "checks": checks,
        "failures": failures,
        "diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=_DATA)
    parser.add_argument(
        "--out",
        type=Path,
        default=_DATA / "validation_c2_v8.json",
    )
    args = parser.parse_args()
    task_paths = {
        arm: args.data_dir / f"tasks_c2_v8_{arm}.jsonl"
        for arm in PROMPT_ARMS
    }
    report = validate(
        task_paths=task_paths,
        answer_key_path=args.data_dir / "answer_key_c2_v8.jsonl",
        manifest_path=args.data_dir / "tasks_c2_v8.manifest.json",
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n")
    print(f"Wrote {args.out}")
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
