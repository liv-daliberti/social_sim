#!/usr/bin/env python3
"""Freeze the v6 solo-city control: City C alone, no reference cities.

WHY THIS ARM EXISTS

The headline metric is how far a model's forecast sits from the ceiling. That
gap has two sources that the main arms cannot separate:

  1. failing to recover City C's own response from City C's own cases;
  2. recovering it, then declining to combine it with the two examples.

Only (2) is the inductive-inference claim. In v3, where the target-only
estimator was a column average, (1) was negligible: 4-14% of the gap. Removing
that shortcut made (1) dominant -- 41-74% in v5, and still 37-53% in v6, even
though v6's estimator is only two group means and a division. Designing the
shortcut away therefore cannot work: any estimator these models compute
imprecisely will swamp the effect.

So (1) has to be measured. This arm shows City C's cases and nothing else, with
the same question. There are no siblings to neglect, so a model's error here is
purely (1). Sibling neglect is then

    (model - ceiling) - (solo - target_only)

measured rather than assumed, and the same subtraction works for any design.

WHAT IS REMOVED

Only the two reference cities and the sentence that introduces them. City C's
background line, its cases, the framing, the forecast question and the response
instruction are byte-identical to the blind arm. Task IDs are preserved so the
same answer key scores both.

This arm is NOT structure-blind evidence and NOT a condition of the experiment.
It is an instrument for decomposing the blind arm's gap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

_INTRO = "Below are records from two earlier cities in the same region.\n\n"
_BLOCK = re.compile(r"CITY [AB]\n.*?\n\n", re.DOTALL)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_reference_cities(prompt: str) -> str:
    """Remove City A, City B, and the sentence that introduces them."""
    if _INTRO not in prompt:
        raise ValueError("reference-city intro sentence not found")
    stripped = prompt.replace(_INTRO, "")
    stripped, count = _BLOCK.subn("", stripped)
    if count != 2:
        raise ValueError(f"expected to remove 2 city blocks, removed {count}")
    if "CITY A" in stripped or "CITY B" in stripped:
        raise ValueError("a reference city survived the strip")
    if "CITY C" not in stripped:
        raise ValueError("City C was removed")
    return stripped


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--blind-tasks",
        type=Path,
        default=_ROOT / "data" / "three_city_c2_v6" / "tasks_c2_v6.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_ROOT
        / "data"
        / "three_city_c2_v6_solo"
        / "tasks_c2_v6_solo.jsonl",
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
        solo = strip_reference_cities(record["prompt"])
        # Everything from CITY C onward must be untouched.
        if record["prompt"].split("CITY C", 1)[1] != solo.split("CITY C", 1)[1]:
            raise ValueError(
                f"{record['task_id']}: content after CITY C changed"
            )
        rewritten.append(
            {
                "task_id": record["task_id"],
                "prompt": solo,
                "prompt_sha256": hashlib.sha256(
                    solo.encode("utf-8")
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
        "experiment": "three_city_c2_v6_solo",
        "arm": "solo_city_control_for_decomposition",
        "purpose": (
            "measure the cost of failing to recover City C's own response, so "
            "it can be subtracted from the blind arm's gap"
        ),
        "reference_cities_shown": False,
        "is_a_condition_of_the_experiment": False,
        "must_not_be_pooled_with": "three_city_c2_v6",
        "shares_answer_key_with": "three_city_c2_v6",
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
    print(f"Wrote {len(rewritten)} solo-arm tasks to {args.out}")
    print(f"Wrote manifest to {args.out.with_suffix('.manifest.json')}")


if __name__ == "__main__":
    main()
