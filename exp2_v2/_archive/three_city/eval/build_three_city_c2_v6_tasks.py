#!/usr/bin/env python3
"""Freeze independent-case three-city C2 v6 prompts and scoring keys.

Identical construction to v2; only the engine calibration differs. See
engine/three_city_c2_v6.py for why.
"""

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

from engine.three_city_c2_v6 import (
    CONTEXT_CONDITIONS,
    CONTEXT_RELIABILITY,
    STARTING_POLL,
    TARGET_CASES,
    balanced_triplets,
    compute_bayes,
    compute_frequentist,
    compute_pooled_exemplar,
    gold_expected_poll,
    target_context,
)

_OUTDIR = _ROOT / "data" / "three_city_c2_v6"

_FORBIDDEN_DISCLOSURES = {
    "type/class/regime": re.compile(
        r"\b(?:types?|classes?|regimes?)\b",
        re.IGNORECASE,
    ),
    "mechanism": re.compile(r"\bmechanisms?\b", re.IGNORECASE),
    "matching a reference": re.compile(
        r"\bmatch(?:es|ed|ing)?\s+(?:city\s+)?[ab]\b",
        re.IGNORECASE,
    ),
    "same response/pattern": re.compile(
        r"\bsame\s+(?:response|pattern|process|behavior)\b",
        re.IGNORECASE,
    ),
    "gain": re.compile(r"\bgains?\b", re.IGNORECASE),
    "coefficient": re.compile(r"\bcoefficients?\b", re.IGNORECASE),
    "prior": re.compile(r"\bpriors?\b", re.IGNORECASE),
    "probability": re.compile(
        r"\b(?:probabilit(?:y|ies)|equally\s+likely)\b",
        re.IGNORECASE,
    ),
    "noise": re.compile(r"\bnoise\b", re.IGNORECASE),
    "sigma": re.compile(r"\bsigma\b", re.IGNORECASE),
    "context cue/reliability": re.compile(
        r"\b(?:cues?|indicators?|reliab(?:le|ility))\b",
        re.IGNORECASE,
    ),
    "50/50": re.compile(r"\b50\s*/\s*50\b", re.IGNORECASE),
    "hidden type label": re.compile(
        r"\b(open_information|buffered_information)\b",
        re.IGNORECASE,
    ),
    "points per news": re.compile(
        r"\bpoints?\s+per\s+(?:point\s+of\s+)?news\b",
        re.IGNORECASE,
    ),
}


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


def build_prompt(
    triplet: Mapping[str, Any],
    *,
    k: int,
    condition: str,
) -> str:
    """Render independent cases and one forecast without latent disclosures."""
    lines = [
        "You are forecasting a post-news local-election poll.",
        "Each table row is a separate polling case, not a time series. Every "
        "case began with a poll of 50.0. The signed net-news value then occurred, "
        "and the poll at the end of that case was measured.",
        "Positive net news is favorable to the candidate and negative net news "
        "is unfavorable.",
        "",
        "Below are records from two earlier cities in the same region.",
        "",
        "CITY A",
        f"Background: {triplet['reference_context_a']}",
        *_table(triplet["reference_a"]),
        "",
        "CITY B",
        f"Background: {triplet['reference_context_b']}",
        *_table(triplet["reference_b"]),
        "",
        "CITY C",
    ]
    context = target_context(triplet, condition)
    if context:
        lines.append(f"Background: {context}")
    else:
        lines.append("No additional background information is available.")
    if k:
        lines.extend(_table(triplet["target"], k=k))
    else:
        lines.append("No completed City C cases are available yet.")
    lines += [
        "",
        f"A new City C case begins with a poll of {STARTING_POLL:.1f} and "
        f"has net news {int(triplet['test_news']):+d}.",
        "What poll do you predict at the end of this new case?",
        "",
        "Respond with one JSON object only:",
        '{"rationale": "one short sentence", "predicted_poll": <number from 0 to 100>}',
    ]
    prompt = "\n".join(lines)
    validate_structure_blind_prompt(prompt)
    return prompt


def validate_structure_blind_prompt(prompt: str) -> None:
    found = [
        label
        for label, pattern in _FORBIDDEN_DISCLOSURES.items()
        if pattern.search(prompt)
    ]
    if found:
        raise ValueError(
            "prompt discloses evaluator structure: " + ", ".join(found)
        )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_records(
    triplet: Mapping[str, Any],
    *,
    episode_index: int,
    k: int,
    condition: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    prompt = build_prompt(triplet, k=k, condition=condition)
    condition_index = CONTEXT_CONDITIONS.index(condition)
    task_id = f"c2v6_{episode_index:04d}_v{condition_index}_k{k}"
    target = triplet["target"]
    public = {
        "reference_a": {
            "background": triplet["reference_context_a"],
            "starting_poll": triplet["reference_a"]["starting_poll"],
            "news": triplet["reference_a"]["news"],
            "ending_polls": triplet["reference_a"]["ending_polls"],
        },
        "reference_b": {
            "background": triplet["reference_context_b"],
            "starting_poll": triplet["reference_b"]["starting_poll"],
            "news": triplet["reference_b"]["news"],
            "ending_polls": triplet["reference_b"]["ending_polls"],
        },
        "target": {
            "background": target_context(triplet, condition),
            "starting_poll": target["starting_poll"],
            "news": target["news"][:k],
            "ending_polls": target["ending_polls"][:k],
        },
        "test_starting_poll": STARTING_POLL,
        "test_news": triplet["test_news"],
    }
    model_record = {
        "task_id": task_id,
        "prompt": prompt,
        "prompt_sha256": _sha256_text(prompt),
    }
    evaluator_record = {
        "task_id": task_id,
        "episode_id": f"c2v6_{episode_index:04d}",
        "episode_seed": triplet["seed"],
        "condition": condition,
        "context_family": triplet["context_family"],
        "k": k,
        "public": public,
        "gold": {
            "expected_poll": gold_expected_poll(triplet, k),
            "target_matches": triplet["target_matches"],
            "target_response": target["response"],
            "target_type": target["city_type"],
        },
        "baselines": {
            "frequentist": compute_frequentist(triplet, k),
            "pooled_exemplar": compute_pooled_exemplar(triplet, k),
            "target_only_bayes": compute_bayes(
                triplet,
                k,
                condition=condition,
                use_context=False,
            ),
            "context_oracle": compute_bayes(
                triplet,
                k,
                condition=condition,
                use_context=True,
            ),
        },
    }
    return model_record, evaluator_record


def _default_answer_key_path(tasks_path: Path) -> Path:
    if tasks_path.name.startswith("tasks_"):
        return tasks_path.with_name(
            tasks_path.name.replace("tasks_", "answer_key_", 1)
        )
    return tasks_path.with_suffix(".answer_key.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n", type=int, default=120)
    # Distinct from v2's 10_000 so the noise draws are independent. The factor
    # structure per episode index is unchanged, since balanced_triplets assigns
    # type, label and paraphrase family explicitly rather than from the seed.
    parser.add_argument("--seed-offset", type=int, default=30_000)
    parser.add_argument(
        "--out",
        type=Path,
        default=_OUTDIR / "tasks_c2_v6.jsonl",
    )
    parser.add_argument("--answer-key", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.n <= 0 or args.n % 12:
        parser.error("--n must be a positive multiple of 12 for exact balance")
    triplets = list(
        balanced_triplets(args.n, seed_offset=args.seed_offset)
    )
    if args.dry_run:
        print(
            build_prompt(
                triplets[0],
                k=TARGET_CASES,
                condition="cue_high",
            )
        )
        return

    answer_key_path = args.answer_key or _default_answer_key_path(args.out)
    existing = [path for path in (args.out, answer_key_path) if path.exists()]
    if existing and not args.overwrite:
        joined = ", ".join(str(path) for path in existing)
        parser.error(f"{joined} already exists; pass --overwrite intentionally")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    answer_key_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out.with_suffix(".manifest.json")
    aggregate_prompt_hash = hashlib.sha256()
    task_count = 0
    with args.out.open("w") as task_handle, answer_key_path.open(
        "w"
    ) as key_handle:
        for episode_index, triplet in enumerate(triplets):
            for condition in CONTEXT_CONDITIONS:
                for k in range(len(triplet["target"]["news"]) + 1):
                    model_record, evaluator_record = build_records(
                        triplet,
                        episode_index=episode_index,
                        k=k,
                        condition=condition,
                    )
                    aggregate_prompt_hash.update(
                        model_record["task_id"].encode("utf-8")
                    )
                    aggregate_prompt_hash.update(
                        model_record["prompt"].encode("utf-8")
                    )
                    task_handle.write(
                        json.dumps(model_record, sort_keys=True) + "\n"
                    )
                    key_handle.write(
                        json.dumps(evaluator_record, sort_keys=True) + "\n"
                    )
                    task_count += 1

    source_paths = {
        "engine_sha256": _ROOT / "engine" / "three_city_c2_v6.py",
        "builder_sha256": Path(__file__),
        "config_sha256": _ROOT / "configs" / "three_city_c2_v6.yml",
        "validator_sha256": _HERE / "validate_three_city_c2_v6_tasks.py",
    }
    manifest = {
        "experiment": "three_city_c2_v6",
        "calibration": "d_prime_1.12",
        "status": "frozen_no_model_runs",
        "episodes": args.n,
        "tasks": task_count,
        "seed_offset": args.seed_offset,
        "conditions": list(CONTEXT_CONDITIONS),
        "prefixes": list(range(TARGET_CASES + 1)),
        "context_reliability_for_oracle_only": CONTEXT_RELIABILITY,
        "structure_disclosed_to_model": False,
        "temporal_state_present": False,
        "aggregate_prompt_sha256": aggregate_prompt_hash.hexdigest(),
        "model_task_file": str(args.out.name),
        "model_task_file_sha256": _file_sha256(args.out),
        "evaluator_answer_key_file": str(answer_key_path.name),
        "evaluator_answer_key_sha256": _file_sha256(answer_key_path),
        **{
            field: _file_sha256(path)
            for field, path in source_paths.items()
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {task_count} frozen tasks to {args.out}")
    print(f"Wrote evaluator-only answer key to {answer_key_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
