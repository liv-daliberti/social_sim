#!/usr/bin/env python3
"""Freeze structure-blind three-city C2 prompts and scoring metadata.

This script makes no model/API calls. It writes:

1. a model-facing JSONL containing only opaque IDs and rendered prompts;
2. a separate evaluator-only answer key containing conditions, hidden truth,
   and baseline outputs; and
3. a manifest containing source, file, and aggregate prompt hashes.

Keeping the key separate prevents a future runner from exposing the evaluator
structure by accidentally serializing an entire task record.
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

from engine.three_city_c2 import (
    CONTEXT_CONDITIONS,
    CONTEXT_RELIABILITY,
    compute_bayes,
    compute_clairvoyant,
    compute_frequentist,
    compute_naive,
    compute_pooled_exemplar,
    compute_two_prototype_exemplar,
    gold_expected_poll,
    balanced_triplets,
    target_context,
)

_OUTDIR = _ROOT / "data" / "three_city_c2"

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


def _table(city: Mapping[str, Any], k: Optional[int] = None) -> list[str]:
    if k is None:
        k = len(city["news"])
    rows = [
        "| Week | Net news | Poll |",
        "|------|----------|------|",
        f"| {0:>4} | {'baseline':>8} | {city['initial_poll']:>5.1f} |",
    ]
    rows.extend(
        f"| {week:>4} | {dose:>+8d} | {poll:>5.1f} |"
        for week, (dose, poll) in enumerate(
            zip(city["news"][:k], city["polls"][:k]),
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
    """Render examples and a forecast question without disclosing the DGP."""
    lines = [
        "You are forecasting a weekly local-election tracking poll.",
        "Net news is a signed summary of that week's national campaign news: "
        "positive values are favorable to the candidate and negative values are unfavorable.",
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
    lines.extend(_table(triplet["target"], k=k))
    if k == 0:
        lines.append("No post-baseline City C poll has yet been observed.")
    lines += [
        "",
        f"Next week, City C's net news will be {int(triplet['test_news']):+d}.",
        "What poll result do you predict for City C next week?",
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
    target = triplet["target"]
    condition_index = CONTEXT_CONDITIONS.index(condition)
    task_id = f"c2_{episode_index:04d}_v{condition_index}_k{k}"
    public = {
        "reference_a": {
            "background": triplet["reference_context_a"],
            "initial_poll": triplet["reference_a"]["initial_poll"],
            "news": triplet["reference_a"]["news"],
            "polls": triplet["reference_a"]["polls"],
        },
        "reference_b": {
            "background": triplet["reference_context_b"],
            "initial_poll": triplet["reference_b"]["initial_poll"],
            "news": triplet["reference_b"]["news"],
            "polls": triplet["reference_b"]["polls"],
        },
        "target": {
            "background": target_context(triplet, condition),
            "initial_poll": target["initial_poll"],
            "news": target["news"][:k],
            "polls": target["polls"][:k],
        },
        "test_news": triplet["test_news"],
    }
    model_record = {
        "task_id": task_id,
        "prompt": prompt,
        "prompt_sha256": _sha256_text(prompt),
    }
    evaluator_record = {
        "task_id": task_id,
        "episode_id": f"c2_{episode_index:04d}",
        "episode_seed": triplet["seed"],
        "condition": condition,
        "context_family": triplet["context_family"],
        "k": k,
        "public": public,
        "gold": {
            "expected_poll": gold_expected_poll(triplet, k),
            "target_matches": triplet["target_matches"],
            "target_gain": target["gain"],
            "target_type": target["city_type"],
        },
        "baselines": {
            "naive": compute_naive(triplet, k),
            "frequentist": compute_frequentist(triplet, k),
            "pooled_exemplar": compute_pooled_exemplar(triplet, k),
            "two_prototype_exemplar": compute_two_prototype_exemplar(
                triplet,
                k,
            ),
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
            "clairvoyant": compute_clairvoyant(triplet, k),
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
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=_OUTDIR / "tasks_c2_v1.jsonl",
    )
    parser.add_argument(
        "--answer-key",
        type=Path,
        default=None,
        help="Evaluator-only key; defaults beside --out.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.n <= 0:
        parser.error("--n must be positive")

    triplets = list(balanced_triplets(args.n, seed_offset=args.seed_offset))
    if args.dry_run:
        print(
            build_prompt(
                triplets[0],
                k=1,
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

    engine_path = _ROOT / "engine" / "three_city_c2.py"
    config_path = _ROOT / "configs" / "three_city_c2_v1.yml"
    validator_path = _HERE / "validate_three_city_c2_tasks.py"
    manifest = {
        "experiment": "three_city_c2_v1",
        "status": "frozen",
        "episodes": args.n,
        "tasks": task_count,
        "seed_offset": args.seed_offset,
        "conditions": list(CONTEXT_CONDITIONS),
        "prefixes": [0, 1, 2, 3],
        "context_reliability_for_oracle_only": CONTEXT_RELIABILITY,
        "structure_disclosed_to_model": False,
        "aggregate_prompt_sha256": aggregate_prompt_hash.hexdigest(),
        "model_task_file": str(args.out.name),
        "model_task_file_sha256": _file_sha256(args.out),
        "evaluator_answer_key_file": str(answer_key_path.name),
        "evaluator_answer_key_sha256": _file_sha256(answer_key_path),
        "engine_sha256": _file_sha256(engine_path),
        "builder_sha256": _file_sha256(Path(__file__)),
        "config_sha256": _file_sha256(config_path),
        "validator_sha256": _file_sha256(validator_path),
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
