#!/usr/bin/env python3
"""Validate frozen v7 prompt pairing, non-disclosure, balance, and baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from engine.three_city_c2_v7 import (
    CONTEXT_CONDITIONS,
    CONTEXT_SENSITIVITY_WEIGHTS,
    HIGH_TYPE,
    ORTHOGONAL_CONTEXTS,
    PREFIX_LADDER,
    TARGET_CONTEXTS,
    infer_context_mechanism,
)
from eval.build_three_city_c2_v7_tasks import (
    HINT_SENTENCE,
    strip_hint,
    validate_prompt,
)

_DATA = _ROOT / "data" / "three_city_c2_v7"


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


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--blind",
        type=Path,
        default=_DATA / "tasks_c2_v7_blind.jsonl",
    )
    parser.add_argument(
        "--hint",
        type=Path,
        default=_DATA / "tasks_c2_v7_hint.jsonl",
    )
    parser.add_argument(
        "--answer-key",
        type=Path,
        default=_DATA / "answer_key_c2_v7.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_DATA / "tasks_c2_v7.manifest.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DATA / "validation_c2_v7.json",
    )
    args = parser.parse_args()

    blind = _read_jsonl(args.blind)
    hint = _read_jsonl(args.hint)
    keys = _read_jsonl(args.answer_key)
    manifest = json.loads(args.manifest.read_text())
    failures: list[str] = []
    checks: dict[str, Any] = {}

    blind_by_id = {row["task_id"]: row for row in blind}
    hint_by_id = {row["task_id"]: row for row in hint}
    key_by_id = {row["task_id"]: row for row in keys}
    id_sets_equal = (
        set(blind_by_id) == set(hint_by_id) == set(key_by_id)
        and len(blind_by_id) == len(blind)
        and len(hint_by_id) == len(hint)
        and len(key_by_id) == len(keys)
    )
    checks["id_sets_equal_and_unique"] = id_sets_equal
    if not id_sets_equal:
        failures.append("task ID sets differ or contain duplicates")

    pair_diff_failures = []
    hash_failures = []
    disclosure_failures = []
    for task_id in sorted(set(blind_by_id) & set(hint_by_id)):
        b = blind_by_id[task_id]
        h = hint_by_id[task_id]
        if strip_hint(h["prompt"]) != b["prompt"]:
            pair_diff_failures.append(task_id)
        if h["prompt"].count(HINT_SENTENCE) != 1:
            pair_diff_failures.append(task_id)
        for arm, row in (("blind", b), ("hint", h)):
            if _sha256_text(row["prompt"]) != row["prompt_sha256"]:
                hash_failures.append(f"{arm}:{task_id}")
            try:
                validate_prompt(row["prompt"], arm=arm)
            except ValueError as exc:
                disclosure_failures.append(f"{arm}:{task_id}:{exc}")
    checks["prompt_pairs_differ_only_by_hint"] = not pair_diff_failures
    checks["prompt_hashes_valid"] = not hash_failures
    checks["prompt_non_disclosure_valid"] = not disclosure_failures
    if pair_diff_failures:
        failures.append(
            f"{len(set(pair_diff_failures))} prompt pairs differ beyond the hint"
        )
    if hash_failures:
        failures.append(f"{len(hash_failures)} prompt hashes fail")
    if disclosure_failures:
        failures.append(f"{len(disclosure_failures)} prompts fail validation")

    expected_tasks = 120 * len(CONTEXT_CONDITIONS) * len(PREFIX_LADDER)
    checks["expected_task_count_per_arm"] = (
        len(blind) == len(hint) == len(keys) == expected_tasks
    )
    if not checks["expected_task_count_per_arm"]:
        failures.append(
            f"expected {expected_tasks} tasks per arm; got "
            f"{len(blind)}/{len(hint)}/{len(keys)}"
        )

    factors = Counter(
        (
            row["condition"],
            int(row["k"]),
            row["gold"]["target_type"],
            row["gold"]["target_matches"],
            int(row["context_family"]),
        )
        for row in keys
    )
    factor_counts = set(factors.values())
    checks["exact_factor_balance"] = factor_counts == {10}
    if not checks["exact_factor_balance"]:
        failures.append(f"factor cell counts are {sorted(factor_counts)}")

    relevant_truth_failures = []
    orthogonal_failures = []
    no_context_failures = []
    for row in keys:
        background = row["public"]["target"]["background"]
        family = int(row["context_family"])
        target_type = row["gold"]["target_type"]
        if row["condition"] == "relevant":
            if background != TARGET_CONTEXTS[target_type][family]:
                relevant_truth_failures.append(row["task_id"])
        elif row["condition"] == "orthogonal":
            if background != ORTHOGONAL_CONTEXTS[family]:
                orthogonal_failures.append(row["task_id"])
        elif background:
            no_context_failures.append(row["task_id"])
    checks["relevant_context_always_truthful"] = not relevant_truth_failures
    checks["orthogonal_context_exact"] = not orthogonal_failures
    checks["none_condition_has_no_context"] = not no_context_failures
    if relevant_truth_failures:
        failures.append("relevant context is not truthful")
    if orthogonal_failures:
        failures.append("orthogonal context mismatch")
    if no_context_failures:
        failures.append("none condition contains context")

    public_mapping_failures = []
    sensitivity_failures = []
    expected_sensitivity_keys = {
        f"{weight:g}" for weight in CONTEXT_SENSITIVITY_WEIGHTS
    }
    for row in keys:
        baselines = row["baselines"]
        sensitivity = baselines.get(
            "empirical_hierarchical_context_sensitivity",
            {},
        )
        if set(sensitivity) != expected_sensitivity_keys:
            sensitivity_failures.append(row["task_id"])
        elif sensitivity["1"] != baselines["empirical_hierarchical"]:
            sensitivity_failures.append(row["task_id"])
        if row["condition"] != "relevant":
            continue
        public = row["public"]
        target_mechanism = infer_context_mechanism(
            public["target"]["background"]
        )
        matching_references = [
            label
            for label, city in (
                ("A", public["reference_a"]),
                ("B", public["reference_b"]),
            )
            if infer_context_mechanism(city["background"])
            == target_mechanism
        ]
        if matching_references != [row["gold"]["target_matches"]]:
            public_mapping_failures.append(row["task_id"])
    checks["public_context_match_uses_visible_text"] = (
        not public_mapping_failures
    )
    checks["context_weight_sensitivity_is_frozen"] = (
        not sensitivity_failures
    )
    if public_mapping_failures:
        failures.append(
            "visible-text codebook does not uniquely recover the target match"
        )
    if sensitivity_failures:
        failures.append("context-weight sensitivity baselines are invalid")

    relevant_lengths = [
        len(text.split())
        for texts in TARGET_CONTEXTS.values()
        for text in texts
    ]
    orthogonal_lengths = [len(text.split()) for text in ORTHOGONAL_CONTEXTS]
    checks["context_length_match"] = (
        max(relevant_lengths + orthogonal_lengths)
        - min(relevant_lengths + orthogonal_lengths)
        <= 2
        and abs(
            statistics.mean(relevant_lengths)
            - statistics.mean(orthogonal_lengths)
        )
        <= 1
    )
    if not checks["context_length_match"]:
        failures.append("relevant and orthogonal context lengths are unmatched")

    sparse_reference_advantage: dict[str, dict[str, float]] = {}
    for condition in CONTEXT_CONDITIONS:
        sparse_reference_advantage[condition] = {}
        for k in (1, 2):
            selected = [
                row
                for row in keys
                if row["condition"] == condition and int(row["k"]) == k
            ]
            target_mae = statistics.mean(
                abs(
                    row["baselines"]["target_only"]["predicted_poll"]
                    - row["gold"]["expected_poll"]
                )
                for row in selected
            )
            hierarchical_mae = statistics.mean(
                abs(
                    row["baselines"]["empirical_hierarchical"]["predicted_poll"]
                    - row["gold"]["expected_poll"]
                )
                for row in selected
            )
            sparse_reference_advantage[condition][str(k)] = round(
                target_mae - hierarchical_mae,
                6,
            )
    checks["sparse_public_hierarchy_beats_target_only"] = all(
        value > 0
        for by_k in sparse_reference_advantage.values()
        for value in by_k.values()
    )
    checks["sparse_reference_advantage_poll_points"] = (
        sparse_reference_advantage
    )
    if not checks["sparse_public_hierarchy_beats_target_only"]:
        failures.append("public hierarchy does not beat target-only at k=1,2")

    model_record_keys_valid = all(
        set(row) == {"task_id", "prompt", "prompt_sha256"}
        for row in blind + hint
    )
    checks["model_records_contain_only_prompt_material"] = (
        model_record_keys_valid
    )
    if not model_record_keys_valid:
        failures.append("model record contains evaluator material")

    hidden_reliability_absent = all(
        "reliability" not in json.dumps(row).lower()
        and "80%" not in row["prompt"]
        for row in blind + hint
    )
    checks["no_hidden_context_reliability_in_prompt"] = (
        hidden_reliability_absent
    )
    if not hidden_reliability_absent:
        failures.append("prompt contains hidden reliability language")

    source_paths = {
        "engine_sha256": _ROOT / "engine" / "three_city_c2_v7.py",
        "builder_sha256": _HERE / "build_three_city_c2_v7_tasks.py",
        "validator_sha256": Path(__file__),
        "runner_sha256": _HERE / "run_three_city_c2_v7_confirmatory.py",
        "launcher_sha256": (
            _HERE / "slurm_three_city_c2_v7_confirmatory.sh"
        ),
        "analysis_sha256": (
            _ROOT
            / "analysis"
            / "analyze_three_city_c2_v7_confirmatory.py"
        ),
        "config_sha256": _ROOT / "configs" / "three_city_c2_v7.yml",
        "preregistration_sha256": (
            _ROOT / "PREREGISTRATION_THREE_CITY_C2_V7.md"
        ),
        "blind_task_file_sha256": args.blind,
        "hint_task_file_sha256": args.hint,
        "answer_key_file_sha256": args.answer_key,
    }
    manifest_hash_failures = [
        field
        for field, path in source_paths.items()
        if manifest.get(field) != _file_sha256(path)
    ]
    checks["frozen_manifest_hashes_valid"] = not manifest_hash_failures
    if manifest_hash_failures:
        failures.append(
            "frozen manifest hash mismatch: "
            + ", ".join(manifest_hash_failures)
        )

    completion = {
        "experiment": "three_city_c2_v7",
        "valid": not failures,
        "failures": failures,
        "checks": checks,
        "counts": {
            "episodes": len({row["episode_id"] for row in keys}),
            "tasks_per_arm": len(blind),
            "target_high": sum(
                row["gold"]["target_type"] == HIGH_TYPE
                for row in keys
                if row["condition"] == "relevant" and row["k"] == 2
            ),
            "target_low": sum(
                row["gold"]["target_type"] != HIGH_TYPE
                for row in keys
                if row["condition"] == "relevant" and row["k"] == 2
            ),
        },
    }
    args.out.write_text(json.dumps(completion, indent=2, sort_keys=True) + "\n")
    print(json.dumps(completion, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
