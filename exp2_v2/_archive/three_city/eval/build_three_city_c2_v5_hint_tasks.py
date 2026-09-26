#!/usr/bin/env python3
"""Freeze the v3 relevance-hint arm: a labelled NON-BLIND control.

Every number, table, and background line is copied byte-for-byte from the frozen
structure-blind `tasks_c2_v5.jsonl`. Only the closing instruction changes, and
task IDs are deliberately kept identical so the same answer key scores both arms
and the comparison is exactly paired.

WHAT THIS ARM DISCLOSES, and the blind arm does not:

1. that the two earlier cities may be informative about City C;
2. that City C's own measurements are noisy.

It still does NOT disclose how many kinds of city exist, that City C corresponds
to either reference, any coefficient, the generating equation, a prior, or the
contextual-cue reliability.

Because of (1) and (2) this arm is **not** evidence about spontaneous inductive
inference and must never be pooled with the blind arm. Its only job is to
separate "cannot do this" from "does not do this unprompted".

Note: `validate_structure_blind_prompt` does not flag the hint, because its
noise pattern matches `noise` and the hint says `noisy`. That is a gap in the
blindness checker, not a licence to treat this arm as blind. The manifest records
the disclosure explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

_BLIND_TAIL = (
    "Respond with one JSON object only:\n"
    '{"rationale": "one short sentence", "predicted_poll": <number from 0 to 100>}'
)
_HINT_TAIL = (
    "The records from the two earlier cities may be informative about City C, "
    "and City C's own measurements are noisy.\n"
    "Reason through the problem carefully first, showing your working.\n"
    "Then end your reply with one JSON object:\n"
    '{"rationale": "your reasoning", "predicted_poll": <number from 0 to 100>}'
)
_DISCLOSURES = [
    "reference_cities_may_be_informative_about_target",
    "target_measurements_are_noisy",
    "extended_reasoning_permitted",
]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--blind-tasks",
        type=Path,
        default=_ROOT / "data" / "three_city_c2_v5" / "tasks_c2_v5.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_ROOT
        / "data"
        / "three_city_c2_v5_hint"
        / "tasks_c2_v5_hint.jsonl",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    blind = [
        json.loads(line)
        for line in args.blind_tasks.read_text().splitlines()
        if line.strip()
    ]
    rewritten = []
    for record in blind:
        prompt = record["prompt"]
        if _BLIND_TAIL not in prompt:
            raise ValueError(
                f"{record['task_id']}: blind instruction block not found; the "
                "source task file is not the expected frozen v3 build"
            )
        hinted = prompt.replace(_BLIND_TAIL, _HINT_TAIL)
        # The only permitted difference is the instruction tail: strip each
        # arm's own tail and what remains must match byte for byte.
        if prompt.replace(_BLIND_TAIL, "") != hinted.replace(_HINT_TAIL, ""):
            raise ValueError(
                f"{record['task_id']}: the rewrite changed something outside "
                "the instruction block"
            )
        rewritten.append(
            {
                "task_id": record["task_id"],
                "prompt": hinted,
                "prompt_sha256": hashlib.sha256(
                    hinted.encode("utf-8")
                ).hexdigest(),
            }
        )

    if args.dry_run:
        print(rewritten[0]["prompt"])
        return

    if args.out.exists() and not args.overwrite:
        parser.error(f"{args.out} exists; pass --overwrite intentionally")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for record in rewritten:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    aggregate = hashlib.sha256()
    for record in rewritten:
        aggregate.update(record["task_id"].encode("utf-8"))
        aggregate.update(record["prompt"].encode("utf-8"))
    manifest = {
        "experiment": "three_city_c2_v5_hint",
        "arm": "non_blind_relevance_hint_control",
        "structure_disclosed_to_model": False,
        "relevance_disclosed_to_model": True,
        "disclosures_beyond_the_blind_arm": _DISCLOSURES,
        "must_not_be_pooled_with": "three_city_c2_v5",
        "base_design": "v5_varied_news_no_repeats",
        "shares_answer_key_with": "three_city_c2_v5",
        "task_ids_identical_to_blind_arm": True,
        "tasks": len(rewritten),
        "source_blind_task_file": str(args.blind_tasks),
        "source_blind_task_file_sha256": _file_sha256(args.blind_tasks),
        "model_task_file": args.out.name,
        "model_task_file_sha256": _file_sha256(args.out),
        "aggregate_prompt_sha256": aggregate.hexdigest(),
        "builder_sha256": _file_sha256(Path(__file__)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote {len(rewritten)} hint-arm tasks to {args.out}")
    print(f"Wrote manifest to {args.out.with_suffix('.manifest.json')}")


if __name__ == "__main__":
    main()
