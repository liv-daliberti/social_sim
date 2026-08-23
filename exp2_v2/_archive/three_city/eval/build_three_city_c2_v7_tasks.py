#!/usr/bin/env python3
"""Freeze paired blind/hint tasks for the paper-ready three-city C2 v7."""

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

from engine.three_city_c2_v7 import (
    CONTEXT_CONDITIONS,
    CONTEXT_PSEUDO_CASES,
    CONTEXT_SENSITIVITY_WEIGHTS,
    PREFIX_LADDER,
    STARTING_POLL,
    balanced_triplets,
    compute_dgp_oracle,
    compute_empirical_hierarchical,
    compute_naive,
    compute_pooled_references,
    compute_target_only,
    gold_expected_poll,
    target_context,
)

_OUTDIR = _ROOT / "data" / "three_city_c2_v7"
_PREREG = _ROOT / "PREREGISTRATION_THREE_CITY_C2_V7.md"
HINT_SENTENCE = (
    "When forecasting City C, consider whether the earlier cities' response "
    "patterns and background descriptions are informative."
)

_FORBIDDEN_DISCLOSURES = {
    "type/class/regime": re.compile(
        r"\b(?:types?|classes?|regimes?)\b",
        re.IGNORECASE,
    ),
    "mechanism label": re.compile(r"\bmechanisms?\b", re.IGNORECASE),
    "matching a reference": re.compile(
        r"\bmatch(?:es|ed|ing)?\s+(?:city\s+)?[ab]\b",
        re.IGNORECASE,
    ),
    "same latent pattern": re.compile(
        r"\bsame\s+(?:response|pattern|process|behavior)\b",
        re.IGNORECASE,
    ),
    "coefficient/gain": re.compile(
        r"\b(?:coefficients?|gains?)\b",
        re.IGNORECASE,
    ),
    "prior/probability": re.compile(
        r"\b(?:priors?|probabilit(?:y|ies)|equally\s+likely)\b",
        re.IGNORECASE,
    ),
    "noise scale": re.compile(
        r"\b(?:sigma|standard deviation|noise scale)\b",
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


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _common_prompt_lines(
    triplet: Mapping[str, Any],
    *,
    k: int,
    condition: str,
) -> list[str]:
    lines = [
        "You are forecasting a post-news local-election poll.",
        "Each table row is a separate polling case, not a time series. Every "
        "case began with a poll of 50.0. The signed net-news value then occurred, "
        "and the poll at the end of that case was measured.",
        "End-of-case polls vary from case to case because polling and local "
        "opinion are not perfectly stable.",
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
    lines.extend(
        [
            "",
            f"A new City C case begins with a poll of {STARTING_POLL:.1f} and "
            f"has net news {int(triplet['test_news']):+d}.",
            "What poll do you predict at the end of this new case?",
            "",
        ]
    )
    return lines


def build_prompt(
    triplet: Mapping[str, Any],
    *,
    k: int,
    condition: str,
    arm: str,
) -> str:
    if arm not in ("blind", "hint"):
        raise ValueError("arm must be 'blind' or 'hint'")
    lines = _common_prompt_lines(triplet, k=k, condition=condition)
    if arm == "hint":
        lines.extend([HINT_SENTENCE, ""])
    lines.extend(
        [
            "Think carefully about the expected poll, but do not provide "
            "step-by-step working.",
            "Respond with one JSON object only:",
            '{"rationale": "one short sentence", '
            '"predicted_poll": <number from 0 to 100>}',
        ]
    )
    prompt = "\n".join(lines)
    validate_prompt(prompt, arm=arm)
    return prompt


def strip_hint(prompt: str) -> str:
    return prompt.replace(HINT_SENTENCE + "\n\n", "")


def validate_prompt(prompt: str, *, arm: str) -> None:
    if arm not in ("blind", "hint"):
        raise ValueError("unknown arm")
    if arm == "blind" and HINT_SENTENCE in prompt:
        raise ValueError("blind prompt contains the hint")
    if arm == "hint" and prompt.count(HINT_SENTENCE) != 1:
        raise ValueError("hint prompt must contain the hint exactly once")
    disclosure_scope = strip_hint(prompt)
    found = [
        label
        for label, pattern in _FORBIDDEN_DISCLOSURES.items()
        if pattern.search(disclosure_scope)
    ]
    if found:
        raise ValueError(
            "prompt discloses evaluator structure: " + ", ".join(found)
        )
    required = (
        "separate polling case",
        "not a time series",
        "vary from case to case",
        "one short sentence",
        "do not provide step-by-step working",
    )
    lowered = prompt.lower()
    missing = [phrase for phrase in required if phrase not in lowered]
    if missing:
        raise ValueError("prompt missing common instruction: " + ", ".join(missing))


def build_records(
    triplet: Mapping[str, Any],
    *,
    episode_index: int,
    k: int,
    condition: str,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    condition_index = CONTEXT_CONDITIONS.index(condition)
    task_id = f"c2v7_{episode_index:04d}_c{condition_index}_k{k}"
    prompts = {
        arm: build_prompt(
            triplet,
            k=k,
            condition=condition,
            arm=arm,
        )
        for arm in ("blind", "hint")
    }
    if strip_hint(prompts["hint"]) != prompts["blind"]:
        raise ValueError(f"{task_id}: arms differ by more than the hint sentence")
    model_records = {
        arm: {
            "task_id": task_id,
            "prompt": prompt,
            "prompt_sha256": _sha256_text(prompt),
        }
        for arm, prompt in prompts.items()
    }
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
    evaluator_record = {
        "task_id": task_id,
        "episode_id": f"c2v7_{episode_index:04d}",
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
            "naive": compute_naive(triplet, k),
            "target_only": compute_target_only(triplet, k),
            "pooled_references": compute_pooled_references(triplet, k),
            "empirical_hierarchical": compute_empirical_hierarchical(
                triplet,
                k,
                condition=condition,
            ),
            "empirical_hierarchical_context_sensitivity": {
                f"{weight:g}": compute_empirical_hierarchical(
                    triplet,
                    k,
                    condition=condition,
                    context_pseudo_cases=weight,
                )
                for weight in CONTEXT_SENSITIVITY_WEIGHTS
            },
            "dgp_oracle": compute_dgp_oracle(triplet, k),
        },
    }
    return model_records, evaluator_record


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--seed-offset", type=int, default=70_000)
    parser.add_argument("--outdir", type=Path, default=_OUTDIR)
    parser.add_argument("--preregistration", type=Path, default=_PREREG)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.n <= 0 or args.n % 12:
        parser.error("--n must be a positive multiple of 12")
    if not args.preregistration.exists():
        parser.error(f"missing preregistration: {args.preregistration}")

    triplets = list(
        balanced_triplets(args.n, seed_offset=args.seed_offset)
    )
    if args.dry_run:
        pair, _ = build_records(
            triplets[0],
            episode_index=0,
            k=2,
            condition="relevant",
        )
        print("===== BLIND =====")
        print(pair["blind"]["prompt"])
        print("\n===== HINT =====")
        print(pair["hint"]["prompt"])
        return

    blind_path = args.outdir / "tasks_c2_v7_blind.jsonl"
    hint_path = args.outdir / "tasks_c2_v7_hint.jsonl"
    key_path = args.outdir / "answer_key_c2_v7.jsonl"
    manifest_path = args.outdir / "tasks_c2_v7.manifest.json"
    outputs = (blind_path, hint_path, key_path, manifest_path)
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        parser.error(
            "output exists; pass --overwrite intentionally: "
            + ", ".join(str(path) for path in existing)
        )
    args.outdir.mkdir(parents=True, exist_ok=True)

    aggregate = {"blind": hashlib.sha256(), "hint": hashlib.sha256()}
    task_count = 0
    with (
        blind_path.open("w") as blind_handle,
        hint_path.open("w") as hint_handle,
        key_path.open("w") as key_handle,
    ):
        handles = {"blind": blind_handle, "hint": hint_handle}
        for episode_index, triplet in enumerate(triplets):
            for condition in CONTEXT_CONDITIONS:
                for k in PREFIX_LADDER:
                    model_records, evaluator_record = build_records(
                        triplet,
                        episode_index=episode_index,
                        k=k,
                        condition=condition,
                    )
                    for arm, record in model_records.items():
                        handles[arm].write(
                            json.dumps(record, sort_keys=True) + "\n"
                        )
                        aggregate[arm].update(
                            record["task_id"].encode("utf-8")
                        )
                        aggregate[arm].update(
                            record["prompt"].encode("utf-8")
                        )
                    key_handle.write(
                        json.dumps(evaluator_record, sort_keys=True) + "\n"
                    )
                    task_count += 1

    source_paths = {
        "engine_sha256": _ROOT / "engine" / "three_city_c2_v7.py",
        "builder_sha256": Path(__file__),
        "validator_sha256": _HERE / "validate_three_city_c2_v7_tasks.py",
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
        "preregistration_sha256": args.preregistration,
    }
    manifest = {
        "experiment": "three_city_c2_v7",
        "status": "frozen_preregistered_no_model_runs",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "episodes": args.n,
        "seed_offset": args.seed_offset,
        "arms": ["blind", "hint"],
        "conditions": list(CONTEXT_CONDITIONS),
        "prefixes": list(PREFIX_LADDER),
        "tasks_per_arm": task_count,
        "hint_sentence": HINT_SENTENCE,
        "arms_differ_by_exactly_one_sentence": True,
        "target_context_always_truthful": True,
        "hidden_context_reliability": None,
        "context_pseudo_cases_for_empirical_reference": CONTEXT_PSEUDO_CASES,
        "context_sensitivity_weights": list(CONTEXT_SENSITIVITY_WEIGHTS),
        "reference_samples_centered_as_representative_exemplars": True,
        "blind_task_file": blind_path.name,
        "blind_task_file_sha256": _file_sha256(blind_path),
        "blind_aggregate_prompt_sha256": aggregate["blind"].hexdigest(),
        "hint_task_file": hint_path.name,
        "hint_task_file_sha256": _file_sha256(hint_path),
        "hint_aggregate_prompt_sha256": aggregate["hint"].hexdigest(),
        "answer_key_file": key_path.name,
        "answer_key_file_sha256": _file_sha256(key_path),
        **{
            field: _file_sha256(path)
            for field, path in source_paths.items()
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {task_count} tasks per arm to {args.outdir}")
    print(f"Wrote evaluator answer key to {key_path}")
    print(f"Wrote frozen manifest to {manifest_path}")


if __name__ == "__main__":
    main()
