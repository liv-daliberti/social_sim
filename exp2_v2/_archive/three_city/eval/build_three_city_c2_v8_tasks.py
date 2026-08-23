#!/usr/bin/env python3
"""Freeze the k=1...5 matched-estimator C2 v8 tasks and private key."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from engine.three_city_c2_v8 import (
    CONTEXT_CONDITIONS,
    CASES_BY_PREFIX,
    CONTEXT_RELIABILITY,
    PREFIX_LADDER,
    PROMPT_ARMS,
    SEED_OFFSET,
    STARTING_POLL,
    balanced_triplets,
    cases_for_prefix,
    compute_abc_shrinkage,
    compute_bayes,
    compute_context_oracle,
    compute_reference_midpoint,
    compute_target_only,
    gold_expected_poll,
    target_context,
)

_OUTDIR = _ROOT / "data" / "three_city_c2_v8" / "full_k1_5_similarity" / "design"
_PREREG = _ROOT / "PREREGISTRATION_THREE_CITY_C2_V8_K1_5_SIMILARITY.md"
_CONFIG = _ROOT / "configs" / "three_city_c2_v8_k1_5_similarity.yml"

HINT_BLOCK = (
    "When forecasting City C, consider whether the earlier-city records, "
    "background descriptions, and regional-profile indices are informative."
)
STRONG_HINT_BLOCK = (
    "Cities with similar regional-profile indices tend to have related but "
    "not necessarily identical response patterns. Use Cities A and B to form "
    "a continuous initial estimate for City C, then give the completed City C "
    "cases increasing weight as more evidence is observed."
)
_REASONING_MARKER = (
    "Think carefully about the expected poll, but do not provide step-by-step "
    "working."
)
_FORBIDDEN_BLIND = {
    "type/class/regime": re.compile(
        r"\b(?:types?|classes?|regimes?)\b",
        re.IGNORECASE,
    ),
    "matching a reference": re.compile(
        r"\bmatch(?:es|ed|ing)?\s+(?:city\s+)?[ab]\b",
        re.IGNORECASE,
    ),
    "same response/pattern": re.compile(
        r"\bsame\s+(?:response|pattern|process|behavior)\b",
        re.IGNORECASE,
    ),
    "gain/coefficient": re.compile(
        r"\b(?:gains?|coefficients?)\b",
        re.IGNORECASE,
    ),
    "prior/probability": re.compile(
        r"\b(?:priors?|probabilit(?:y|ies)|equally\s+likely)\b",
        re.IGNORECASE,
    ),
    "noise/sigma": re.compile(r"\b(?:noise|sigma)\b", re.IGNORECASE),
    "cue/reliability": re.compile(
        r"\b(?:cues?|indicators?|reliab(?:le|ility))\b",
        re.IGNORECASE,
    ),
    "50/50": re.compile(r"\b50\s*/\s*50\b", re.IGNORECASE),
    "hidden type label": re.compile(
        r"\b(?:open_information|buffered_information)\b",
        re.IGNORECASE,
    ),
    "points per news": re.compile(
        r"\bpoints?\s+per\s+(?:point\s+of\s+)?news\b",
        re.IGNORECASE,
    ),
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _table(
    city: Mapping[str, Any],
    k: Optional[int] = None,
) -> list[str]:
    if k is None:
        k = len(city["news"])
    rows = [
        "| Case | Starting poll | Net news | Poll at end of case |",
        "|------|--------------:|---------:|--------------------:|",
    ]
    rows.extend(
        f"| {case:>4} | {city['starting_poll']:>13.1f} | "
        f"{dose:>+8d} | {poll:>19.1f} |"
        for case, (dose, poll) in enumerate(
            zip(city["news"][:k], city["ending_polls"][:k]),
            1,
        )
    )
    return rows


def validate_blind_prompt(prompt: str) -> None:
    found = [
        label
        for label, pattern in _FORBIDDEN_BLIND.items()
        if pattern.search(prompt)
    ]
    if found:
        raise ValueError(
            "blind prompt discloses evaluator structure: " + ", ".join(found)
        )
    required = (
        "separate polling case",
        "not a time series",
        "began with a poll of 50.0",
        "vary from case to case",
        "do not provide step-by-step working",
        '"rationale": "one short sentence"',
        '"predicted_poll": <number from 0 to 100>',
    )
    missing = [
        phrase for phrase in required if phrase.lower() not in prompt.lower()
    ]
    if missing:
        raise ValueError("missing common prompt material: " + ", ".join(missing))


def strip_arm_block(prompt: str, arm: str) -> str:
    if arm == "blind":
        return prompt
    block = HINT_BLOCK if arm == "hint" else STRONG_HINT_BLOCK
    return prompt.replace(block + "\n\n", "")


def validate_arm_prompt(prompt: str, arm: str) -> None:
    if arm not in PROMPT_ARMS:
        raise ValueError(f"unknown prompt arm: {arm!r}")
    if arm == "blind":
        if HINT_BLOCK in prompt or STRONG_HINT_BLOCK in prompt:
            raise ValueError("blind prompt contains a hint block")
        validate_blind_prompt(prompt)
        return
    block = HINT_BLOCK if arm == "hint" else STRONG_HINT_BLOCK
    other = STRONG_HINT_BLOCK if arm == "hint" else HINT_BLOCK
    if prompt.count(block) != 1 or other in prompt:
        raise ValueError(f"{arm} prompt does not contain exactly its own block")
    blind = strip_arm_block(prompt, arm)
    validate_blind_prompt(blind)


def build_blind_prompt(
    triplet: Mapping[str, Any],
    *,
    k: int,
    condition: str,
) -> str:
    cases_seen = cases_for_prefix(k)
    profile_a = float(triplet["profile_index_a"])
    profile_b = float(triplet["profile_index_b"])
    profile_target = float(triplet["profile_index_target"])
    lines = [
        "You are forecasting a post-news local-election poll.",
        (
            "Each table row is a separate polling case, not a time series. "
            "Every case began with a poll of 50.0. The signed net-news value "
            "then occurred, and the poll at the end of that case was measured."
        ),
        (
            "End-of-case polls vary from case to case because polling and local "
            "opinion are not perfectly stable."
        ),
        (
            "Positive net news is favorable to the candidate and negative net "
            "news is unfavorable."
        ),
        (
            "Each city also has a preregistered regional-profile index from 0 "
            "to 100. It is a descriptive composite, not an outcome or response "
            "label."
        ),
        "",
        "Below are records from two earlier cities in the same region.",
        "",
        "CITY A",
        f"Regional-profile index: {profile_a:.1f} / 100",
        f"Background: {triplet['reference_context_a']}",
        *_table(triplet["reference_a"]),
        "",
        "CITY B",
        f"Regional-profile index: {profile_b:.1f} / 100",
        f"Background: {triplet['reference_context_b']}",
        *_table(triplet["reference_b"]),
        "",
        "CITY C",
        f"Regional-profile index: {profile_target:.1f} / 100",
    ]
    context = target_context(triplet, condition)
    if context:
        lines.append(f"Background: {context}")
    else:
        lines.append("No additional background information is available.")
    lines.extend(_table(triplet["target"], k=cases_seen))
    lines.extend(
        [
            "",
            (
                f"A new City C case begins with a poll of {STARTING_POLL:.1f} "
                f"and has net news {int(triplet['test_news']):+d}."
            ),
            "What poll do you predict at the end of this new case?",
            "",
            _REASONING_MARKER,
            "Respond with one JSON object only:",
            (
                '{"rationale": "one short sentence", '
                '"predicted_poll": <number from 0 to 100>}'
            ),
        ]
    )
    prompt = "\n".join(lines)
    validate_blind_prompt(prompt)
    return prompt


def build_prompt(
    triplet: Mapping[str, Any],
    *,
    k: int,
    condition: str,
    arm: str,
) -> str:
    blind = build_blind_prompt(triplet, k=k, condition=condition)
    if arm == "blind":
        return blind
    block = HINT_BLOCK if arm == "hint" else STRONG_HINT_BLOCK
    if blind.count(_REASONING_MARKER) != 1:
        raise ValueError("reasoning insertion marker is not unique")
    prompt = blind.replace(
        _REASONING_MARKER,
        block + "\n\n" + _REASONING_MARKER,
        1,
    )
    validate_arm_prompt(prompt, arm)
    if strip_arm_block(prompt, arm) != blind:
        raise ValueError(f"{arm} differs from blind beyond its frozen block")
    return prompt


def build_records(
    triplet: Mapping[str, Any],
    *,
    episode_index: int,
    k: int,
    condition: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    cases_seen = cases_for_prefix(k)
    condition_index = CONTEXT_CONDITIONS.index(condition)
    task_id = f"c2v8_{episode_index:04d}_v{condition_index}_k{k}"
    model_records = {}
    for arm in PROMPT_ARMS:
        prompt = build_prompt(
            triplet,
            k=k,
            condition=condition,
            arm=arm,
        )
        model_records[arm] = {
            "task_id": task_id,
            "prompt": prompt,
            "prompt_sha256": _text_sha256(prompt),
        }
    target = triplet["target"]
    evaluator_record = {
        "task_id": task_id,
        "episode_id": f"c2v8_{episode_index:04d}",
        "episode_seed": triplet["seed"],
        "condition": condition,
        "context_family": triplet["context_family"],
        "k": k,
        "target_cases_seen": cases_seen,
        "public": {
            "reference_a": {
                "profile_index": triplet["profile_index_a"],
                "background": triplet["reference_context_a"],
                "starting_poll": triplet["reference_a"]["starting_poll"],
                "news": triplet["reference_a"]["news"],
                "ending_polls": triplet["reference_a"]["ending_polls"],
            },
            "reference_b": {
                "profile_index": triplet["profile_index_b"],
                "background": triplet["reference_context_b"],
                "starting_poll": triplet["reference_b"]["starting_poll"],
                "news": triplet["reference_b"]["news"],
                "ending_polls": triplet["reference_b"]["ending_polls"],
            },
            "target": {
                "profile_index": triplet["profile_index_target"],
                "background": target_context(triplet, condition),
                "starting_poll": target["starting_poll"],
                "news": target["news"][:cases_seen],
                "ending_polls": target["ending_polls"][:cases_seen],
            },
            "test_starting_poll": STARTING_POLL,
            "test_news": triplet["test_news"],
        },
        "gold": {
            "expected_poll": gold_expected_poll(triplet, cases_seen),
            "target_matches": triplet["target_matches"],
            "target_response": target["response"],
            "target_type": target["city_type"],
        },
        "baselines": {
            "reference_midpoint": compute_reference_midpoint(triplet),
            "target_only": compute_target_only(triplet, k),
            "abc_shrinkage": compute_abc_shrinkage(triplet, k),
            "privileged_structure_ceiling": compute_bayes(
                triplet,
                cases_seen,
                condition="none",
                use_context=False,
            ),
            "privileged_context_oracle": compute_context_oracle(
                triplet,
                k,
                condition=condition,
            ),
        },
    }
    return model_records, evaluator_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--seed-offset", type=int, default=SEED_OFFSET)
    parser.add_argument("--outdir", type=Path, default=_OUTDIR)
    parser.add_argument("--preregistration", type=Path, default=_PREREG)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.n <= 0 or args.n % 12:
        parser.error("--n must be a positive multiple of 12")

    triplets = list(
        balanced_triplets(args.n, seed_offset=args.seed_offset)
    )
    if args.dry_run:
        for arm in PROMPT_ARMS:
            print(f"\n===== {arm} =====\n")
            print(
                build_prompt(
                    triplets[0],
                    k=2,
                    condition="none",
                    arm=arm,
                )
            )
        return

    task_paths = {
        arm: args.outdir / f"tasks_c2_v8_{arm}.jsonl"
        for arm in PROMPT_ARMS
    }
    key_path = args.outdir / "answer_key_c2_v8.jsonl"
    manifest_path = args.outdir / "tasks_c2_v8.manifest.json"
    existing = [
        path
        for path in (*task_paths.values(), key_path, manifest_path)
        if path.exists()
    ]
    if existing and not args.overwrite:
        parser.error("outputs exist; pass --overwrite intentionally")
    args.outdir.mkdir(parents=True, exist_ok=True)

    arm_handles = {
        arm: path.open("w")
        for arm, path in task_paths.items()
    }
    arm_aggregate = {
        arm: hashlib.sha256()
        for arm in PROMPT_ARMS
    }
    task_count = 0
    try:
        with key_path.open("w") as key_handle:
            for episode_index, triplet in enumerate(triplets):
                for condition in CONTEXT_CONDITIONS:
                    for k in PREFIX_LADDER:
                        model_records, evaluator = build_records(
                            triplet,
                            episode_index=episode_index,
                            k=k,
                            condition=condition,
                        )
                        for arm, record in model_records.items():
                            arm_aggregate[arm].update(
                                record["task_id"].encode("utf-8")
                            )
                            arm_aggregate[arm].update(
                                record["prompt"].encode("utf-8")
                            )
                            arm_handles[arm].write(
                                json.dumps(record, sort_keys=True) + "\n"
                            )
                        key_handle.write(
                            json.dumps(evaluator, sort_keys=True) + "\n"
                        )
                        task_count += 1
    finally:
        for handle in arm_handles.values():
            handle.close()

    source_paths = {
        "engine_sha256": _ROOT / "engine" / "three_city_c2_v8.py",
        "source_engine_v2_sha256": (
            _ROOT / "engine" / "three_city_c2_v2.py"
        ),
        "builder_sha256": Path(__file__),
        "config_sha256": _CONFIG,
        "validator_sha256": (
            _HERE / "validate_three_city_c2_v8_tasks.py"
        ),
        "runner_sha256": (
            _HERE / "run_three_city_c2_v8_confirmatory.py"
        ),
        "launcher_sha256": (
            _HERE / "run_three_city_c2_v8_local.sh"
        ),
        "analyzer_sha256": (
            _ROOT / "analysis" / "analyze_three_city_c2_v8_confirmatory.py"
        ),
        "preregistration_sha256": args.preregistration,
    }
    manifest = {
        "experiment": "three_city_c2_v8_full_k1_5_similarity_matched",
        "status": "frozen_local_matched_redesign_no_model_runs",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "episodes": args.n,
        "seed_offset": args.seed_offset,
        "arms": list(PROMPT_ARMS),
        "conditions": list(CONTEXT_CONDITIONS),
        "prefixes": list(PREFIX_LADDER),
        "cases_by_prefix": {str(k): CASES_BY_PREFIX[k] for k in PREFIX_LADDER},
        "profile_index_is_graded_and_overlapping": True,
        "tasks_per_arm": task_count,
        "calls_per_model": task_count * len(PROMPT_ARMS),
        "total_calls_three_models": task_count * len(PROMPT_ARMS) * 3,
        "primary_condition": "none",
        "primary_k": 2,
        "convergence_k": 5,
        "context_cues_crossed_with_truth": True,
        "context_reliability_for_privileged_oracle_only": (
            CONTEXT_RELIABILITY
        ),
        "hint_block": HINT_BLOCK,
        "strong_hint_block": STRONG_HINT_BLOCK,
        "model_task_files": {
            arm: {
                "file": path.name,
                "sha256": _file_sha256(path),
                "aggregate_prompt_sha256": arm_aggregate[arm].hexdigest(),
            }
            for arm, path in task_paths.items()
        },
        "answer_key_file": key_path.name,
        "answer_key_sha256": _file_sha256(key_path),
        **{
            field: _file_sha256(path)
            for field, path in source_paths.items()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"Wrote {task_count} tasks per arm "
        f"({task_count * len(PROMPT_ARMS)} per model)"
    )
    for path in task_paths.values():
        print(f"Wrote {path}")
    print(f"Wrote {key_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
