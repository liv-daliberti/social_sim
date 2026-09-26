#!/usr/bin/env python3
"""Validate frozen minimal three-city C2 v6 tasks before model evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from build_three_city_c2_v6_tasks import validate_structure_blind_prompt
from engine.three_city_c2_v6 import (
    CONTEXT_CONDITIONS,
    HIGH_TYPE,
    LOW_RESPONSE,
    PREFIX_LADDER,
    TARGET_CASES,
)

_PREFIXES = tuple(range(TARGET_CASES + 1))
_LADDER = PREFIX_LADDER
_EPISODES = 120
_EXPECTED_TASKS = _EPISODES * len(CONTEXT_CONDITIONS) * len(_PREFIXES)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_answer_key_path(tasks_path: Path) -> Path:
    if tasks_path.name.startswith("tasks_"):
        return tasks_path.with_name(
            tasks_path.name.replace("tasks_", "answer_key_", 1)
        )
    return tasks_path.with_suffix(".answer_key.jsonl")


def _p_correct(record, method):
    result = record["baselines"][method]
    truth_a = record["gold"]["target_matches"] == "A"
    p_a = result["p_match_a"]
    return p_a if truth_a else 1.0 - p_a


def _numeric_public(record):
    public = record["public"]
    return {
        "reference_a": {
            "starting_poll": public["reference_a"]["starting_poll"],
            "news": public["reference_a"]["news"],
            "ending_polls": public["reference_a"]["ending_polls"],
        },
        "reference_b": {
            "starting_poll": public["reference_b"]["starting_poll"],
            "news": public["reference_b"]["news"],
            "ending_polls": public["reference_b"]["ending_polls"],
        },
        "target": {
            "starting_poll": public["target"]["starting_poll"],
            "news": public["target"]["news"],
            "ending_polls": public["target"]["ending_polls"],
        },
        "test_starting_poll": public["test_starting_poll"],
        "test_news": public["test_news"],
    }


def _mean_mae(records, method, k):
    errors = []
    for record in records:
        if record["condition"] != "none" or record["k"] != k:
            continue
        prediction = record["baselines"][method]["predicted_poll"]
        if prediction is not None:
            errors.append(
                abs(prediction - record["gold"]["expected_poll"])
            )
    return statistics.mean(errors) if errors else None


def validate(tasks_path: Path, answer_key_path: Path, manifest_path: Path):
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
    checks["expected_task_count"] = len(tasks) == _EXPECTED_TASKS
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
    disclosure_errors = []
    sampling_errors = []
    for record in tasks:
        try:
            validate_structure_blind_prompt(record["prompt"])
        except ValueError as exc:
            disclosure_errors.append((record["task_id"], str(exc)))
        lowered = record["prompt"].lower()
        if not all(
            phrase in lowered
            for phrase in (
                "separate polling case",
                "not a time series",
                "began with a poll of 50.0",
                "poll at end of case",
            )
        ):
            sampling_errors.append(record["task_id"])
    checks["no_structure_disclosures"] = not disclosure_errors
    checks["independent_case_sampling_is_explicit"] = not sampling_errors

    task_by_id = {record["task_id"]: record for record in tasks}
    records = [
        {**key, **task_by_id[key["task_id"]]}
        for key in keys
        if key["task_id"] in task_by_id
    ]
    checks["expected_episode_count"] = len(
        {record["episode_id"] for record in records}
    ) == _EPISODES
    checks["every_prefix_present"] = {
        record["k"] for record in records
    } == set(_PREFIXES)

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
    families = Counter(record["context_family"] for record in base)
    checks["balanced_target_type"] = sorted(target_types.values()) == [60, 60]
    checks["balanced_reference_label"] = sorted(matches.values()) == [60, 60]
    checks["balanced_context_family"] = sorted(families.values()) == [40, 40, 40]

    none_by_k = {
        k: [
            record
            for record in records
            if record["condition"] == "none" and record["k"] == k
        ]
        for k in _PREFIXES
    }
    target_only_correct = {
        k: statistics.mean(
            _p_correct(record, "target_only_bayes")
            for record in none_by_k[k]
        )
        for k in _PREFIXES
    }
    checks["target_information_increases_gradually"] = all(
        target_only_correct[a] < target_only_correct[b]
        for a, b in zip(_PREFIXES, _PREFIXES[1:])
    )
    checks["one_case_is_informative_not_decisive"] = (
        0.55 < target_only_correct[1] < 0.75
    )
    # v3's defining property, and the gate v2 would have failed. v2 reached 0.93
    # by k=3 and 0.997 by k=8, so five rungs carried nothing and no model could
    # visibly fall short of the ceiling. The top must be informative without
    # being settled.
    checks["ladder_top_is_informative_but_not_saturated"] = (
        0.85 < target_only_correct[TARGET_CASES] < 0.96
    )
    checks["upper_half_of_ladder_still_carries_information"] = (
        target_only_correct[TARGET_CASES] - target_only_correct[3] >= 0.10
    )

    misleading_correct = {}
    for k in (0, 1, TARGET_CASES):
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
                selected.append(_p_correct(record, "context_oracle"))
        misleading_correct[k] = statistics.mean(selected)
    checks["misleading_context_starts_influential"] = (
        abs(misleading_correct[0] - 0.2) < 1e-9
    )
    checks["ladder_top_overrides_misleading_context"] = (
        misleading_correct[TARGET_CASES] > 0.78
    )

    orthogonal_equal = []
    for group in by_episode_k.values():
        conditions = {record["condition"]: record for record in group}
        orthogonal_equal.append(
            conditions["none"]["baselines"]["context_oracle"][
                "predicted_poll"
            ]
            == conditions["orthogonal"]["baselines"]["context_oracle"][
                "predicted_poll"
            ]
        )
    checks["orthogonal_context_invariance"] = all(orthogonal_equal)

    frequentist_mae = {
        k: _mean_mae(records, "frequentist", k) for k in _PREFIXES
    }
    oracle_mae = {
        k: _mean_mae(records, "target_only_bayes", k) for k in _PREFIXES
    }
    pooled_mae = {
        k: _mean_mae(records, "pooled_exemplar", k) for k in _PREFIXES
    }
    ceiling_gap = {
        k: frequentist_mae[k] - oracle_mae[k]
        for k in _PREFIXES
        if frequentist_mae[k] is not None
    }
    checks["frequentist_unavailable_at_k0"] = frequentist_mae[0] is None
    # Monotone on the reported ladder. Adjacent full-ladder rungs can invert by
    # a few hundredths at 120 episodes, which is below the sampling error.
    checks["frequentist_improves_gradually"] = all(
        frequentist_mae[a] > frequentist_mae[b]
        for a, b in zip(_LADDER[1:], _LADDER[2:])
    )
    checks["frequentist_improves_monotonically_up_to_noise"] = all(
        frequentist_mae[a] > frequentist_mae[b] - 0.1
        for a, b in zip(_PREFIXES[1:], _PREFIXES[2:])
    )
    checks["frequentist_not_nearly_oracular_after_one_case"] = (
        frequentist_mae[1] > 2.0
    )
    # The reference cities must still be worth something at the sparse end and
    # must stop being worth much at the dense end, otherwise the ladder does
    # not span the regime the experiment is about.
    checks["oracle_beats_frequentist_when_sparse"] = all(
        oracle_mae[k] < frequentist_mae[k] for k in (1, 2, 3)
    )
    checks["ceiling_gap_narrows_along_ladder"] = (
        ceiling_gap[1] > ceiling_gap[TARGET_CASES]
    )
    # Recorded, not asserted closed: at TARGET_CASES = 8 the frequentist still
    # trails the ceiling. Reaching a ~0.3-point gap needs about 64 cases.
    checks["ceiling_gap_still_open_at_ladder_top"] = (
        ceiling_gap[TARGET_CASES] > 0.5
    )
    checks["pooled_exemplar_is_flat_in_k"] = (
        max(pooled_mae.values()) - min(pooled_mae.values()) < 1e-9
    )
    checks["target_data_eventually_beats_reference_only_pooling"] = (
        frequentist_mae[TARGET_CASES] < pooled_mae[TARGET_CASES]
    )

    # v4's defining gate. v3 gave every case the same news value, so averaging
    # the end-poll column was a valid estimator of the target's response and
    # arguably the obvious move. v4 varies the news, and this asserts that the
    # shortcut is now genuinely broken rather than merely discouraged.
    column_implied = []
    news_spreads = []
    for record in records:
        if record["condition"] != "none":
            continue
        target = record["public"]["target"]
        if record["k"] == TARGET_CASES:
            mean_poll = statistics.mean(target["ending_polls"])
            column_implied.append(
                (mean_poll - target["starting_poll"])
                / record["public"]["test_news"]
            )
            news_spreads.append(len(set(target["news"])))
    checks["news_takes_both_signs_within_each_city"] = min(news_spreads) == 2
    # v5's defining gates. v4 repeated +8 twice in the target pool while the
    # forecast case was also +8, so models subset to those two cases and
    # averaged them instead of fitting a slope. Every value must now appear at
    # most once, and the forecast value must appear nowhere, so that matching on
    # it returns nothing and matching on anything else returns a single case.
    two_valued = []
    forecast_absent = []
    magnitudes_differ = []
    for record in records:
        if record["condition"] != "none":
            continue
        public = record["public"]
        if public["target"]["news"]:
            # k=0 shows no target cases, so it has no magnitude to compare.
            magnitudes_differ.append(
                len({abs(public[c]["news"][0]) for c in
                     ("reference_a", "reference_b", "target")}) == 3
            )
        for city in ("reference_a", "reference_b", "target"):
            news = public[city]["news"]
            # Only the full schedule is required to be sign-balanced; the ladder
            # deliberately shows prefixes of the target's.
            if not news or (city == "target" and len(news) < TARGET_CASES):
                continue
            two_valued.append(
                len(set(news)) == 2
                and set(news) == {abs(news[0]), -abs(news[0])}
                and sum(news) == 0
            )
            forecast_absent.append(public["test_news"] not in news)
    checks["each_city_uses_exactly_two_signed_values"] = all(two_valued)
    checks["forecast_news_value_never_appears_in_a_displayed_case"] = all(
        forecast_absent
    )
    # v6's point: the three cities use three different magnitudes, so raw poll
    # levels are not comparable across cities and must be normalised.
    checks["all_three_cities_use_different_magnitudes"] = all(magnitudes_differ)
    checks["column_average_implies_almost_no_response"] = (
        abs(statistics.mean(column_implied)) < 0.15
    )
    checks["column_average_falls_outside_demonstrated_band"] = (
        statistics.mean(column_implied) < LOW_RESPONSE
    )

    relevant_lengths = []
    orthogonal_lengths = []
    for record in records:
        if record["k"] != 0:
            continue
        words = len(record["public"]["target"]["background"].split())
        if record["condition"] in ("cue_high", "cue_low"):
            relevant_lengths.append(words)
        elif record["condition"] == "orthogonal":
            orthogonal_lengths.append(words)
    checks["context_lengths_matched"] = (
        abs(statistics.mean(relevant_lengths) - statistics.mean(orthogonal_lengths))
        < 1.0
        and max(relevant_lengths) - min(relevant_lengths) <= 2
        and max(orthogonal_lengths) - min(orthogonal_lengths) <= 1
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
    checks["manifest_declares_no_temporal_state"] = (
        manifest["temporal_state_present"] is False
    )
    source_paths = {
        "engine_sha256": _ROOT / "engine" / "three_city_c2_v6.py",
        "builder_sha256": _HERE / "build_three_city_c2_v6_tasks.py",
        "config_sha256": _ROOT / "configs" / "three_city_c2_v6.yml",
        "validator_sha256": Path(__file__),
    }
    checks["manifest_source_hashes_match"] = all(
        manifest.get(field) == _file_sha256(path)
        for field, path in source_paths.items()
    )

    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "diagnostics": {
            "task_count": len(tasks),
            "answer_key_count": len(keys),
            "target_type_counts": target_types,
            "target_match_counts": matches,
            "context_family_counts": families,
            "target_only_p_correct_by_k": target_only_correct,
            "misleading_context_p_correct": misleading_correct,
            "frequentist_mae_by_k": frequentist_mae,
            "column_average_implied_response": statistics.mean(column_implied),
            "bayes_ceiling_mae_by_k": oracle_mae,
            "pooled_exemplar_mae_by_k": pooled_mae,
            "frequentist_minus_ceiling_by_k": ceiling_gap,
            "mean_relevant_context_words": statistics.mean(relevant_lengths),
            "mean_orthogonal_context_words": statistics.mean(
                orthogonal_lengths
            ),
            "disclosure_errors": disclosure_errors,
            "sampling_errors": sampling_errors,
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", type=Path)
    parser.add_argument("--answer-key", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
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
