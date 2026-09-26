#!/usr/bin/env python3
"""Audit a six-cell canary ledger after explicitly recorded job replacements."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from audit_canaries import ROOT, audit, audit_base_evaluations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--base-ledger", type=Path, required=True)
    parser.add_argument(
        "--replacement",
        action="append",
        required=True,
        help="Failed and replacement Slurm IDs formatted as OLD=NEW; repeatable.",
    )
    parser.add_argument("--time-limit", default="01:30:00")
    parser.add_argument("--write", type=Path, required=True)
    args = parser.parse_args()

    source_path = args.ledger.resolve()
    source_bytes = source_path.read_bytes()
    ledger = json.loads(source_bytes)
    replacements = []
    seen_old, seen_new = set(), set()
    for specification in args.replacement:
        try:
            old_job_id, new_job_id = specification.split("=", 1)
        except ValueError as exc:
            raise SystemExit("--replacement must have the form OLD=NEW") from exc
        if not (old_job_id.isdigit() and new_job_id.isdigit()):
            raise SystemExit("replacement job IDs must be numeric")
        if old_job_id in seen_old or new_job_id in seen_new:
            raise AssertionError("replacement job IDs must be unique")
        seen_old.add(old_job_id)
        seen_new.add(new_job_id)
        matches = [
            item
            for item in ledger["submissions"]
            if item.get("kind") == "training"
            and str(item.get("job_id")) == old_job_id
        ]
        if len(matches) != 1:
            raise AssertionError(
                f"expected exactly one training job {old_job_id}, found {len(matches)}"
            )
        replaced = matches[0]
        if replaced.get("arm") != "causal_family" or replaced.get("seed") != 42:
            raise AssertionError(
                f"job {old_job_id} is not a registered seed-42 causal canary"
            )
        identity = {
            key: replaced[key] for key in ("disclosure", "model", "arm", "seed")
        }
        replaced["job_id"] = new_job_id
        replaced["replaces_job_id"] = old_job_id
        replacements.append({
            "failed_job_id": old_job_id,
            "replacement_job_id": new_job_id,
            "cell": identity,
            "scientific_configuration": "identical",
            "scheduler_time_limit_override": args.time_limit,
        })
    ledger["recovery"] = {
        "source_ledger": str(source_path),
        "source_ledger_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "replacements": replacements,
    }
    replacement_suffix = "_".join(
        f"j{item['replacement_job_id']}" for item in replacements
    )
    derived_path = source_path.with_name(
        f"{source_path.stem}_recovery_{replacement_suffix}.json"
    )
    derived_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")

    result = audit(derived_path)
    result["base_evaluations"] = audit_base_evaluations(args.base_ledger)
    result["base_ledger"] = str(args.base_ledger.resolve())
    result["source_ledger"] = str(source_path.relative_to(ROOT.parent.parent))
    result["recovery"] = ledger["recovery"]
    args.write.parent.mkdir(parents=True, exist_ok=True)
    args.write.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"PASS -> {args.write}")


if __name__ == "__main__":
    main()
