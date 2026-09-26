#!/usr/bin/env python3
"""Check every training wrapper against the hash its own ledger recorded.

Why this exists
---------------
`coin_city_structural/train.sh` was modified on 2026-08-28, between Coin City
seeds 42--44 and seeds 45--49: `--save_steps` went from 999999 to a 40 default
and `--save_ckpt` was added, both silently because the launcher set neither.
The roster was split across two wrapper versions for three weeks and nobody
noticed, which made the whole eight-seed analysis provisional until a six-job
crossover control could rule the change out.

It went unnoticed for a specific, fixable reason: the Coin City ledger records
`preflight_sha256` and `source_sha256` but **no train-script hash**. The
mechanism-family and polymarket ledgers do record one, and both verify clean.
The one roster that did not record its wrapper hash is the one that drifted.

Git is not an adequate baseline here. This repository's history is coarse
relative to run times, and a file can differ from HEAD while exactly matching
what ran -- `polymarket_rl_model_extension.sh` does precisely that. Only the
hash a ledger recorded at submission time says what actually executed.

Usage
-----
    python exp3_training_transfer/audit_wrapper_integrity.py
    python exp3_training_transfer/audit_wrapper_integrity.py --strict

Exit code 1 under --strict if any roster's wrapper fails to verify, so this can
gate a release or a paper build.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXP3 = REPO / "exp3_training_transfer"

# Ledger keys that name the training wrapper that ran.
SCRIPT_KEYS = ("train_script_sha256", "locked_script_sha256", "base_script_sha256")

WRAPPERS = {
    "coin_city_structural/train.sh": "Coin City train.sh",
    "mechanism_family/mechanism_rl.sh": "mechanism_rl.sh",
    "polymarket/scripts/polymarket_rl.sh": "polymarket_rl.sh",
    "polymarket/scripts/polymarket_rl_model_extension.sh": "polymarket model extension",
}


def sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def ledgers() -> list[Path]:
    return sorted(EXP3.glob("*/runs/*.json")) + sorted(EXP3.glob("*/*/runs/*.json"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero if any wrapper fails to verify")
    args = parser.parse_args()

    current = {rel: sha256(EXP3 / rel) for rel in WRAPPERS}
    known = {h: rel for rel, h in current.items() if h}

    verified: dict[str, list[str]] = {rel: [] for rel in WRAPPERS}
    unmatched: list[tuple[str, str, str]] = []
    no_hash: list[str] = []

    for path in ledgers():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        recorded = {k: v for k, v in payload.items()
                    if k in SCRIPT_KEYS and isinstance(v, str)}
        looks_like_roster = any(k in payload for k in
                                ("submissions", "full_training", "training_jobs"))
        if not recorded:
            if looks_like_roster:
                no_hash.append(str(path.relative_to(REPO)))
            continue
        for key, value in recorded.items():
            if value in known:
                verified[known[value]].append(f"{path.name}:{key}")
            else:
                unmatched.append((path.name, key, value[:16]))

    print("WRAPPER INTEGRITY AUDIT\n")
    print("Wrappers verified against a ledger hash:")
    failures = []
    for rel, label in WRAPPERS.items():
        hits = verified[rel]
        if hits:
            # A ledger written by the same submission that used the wrapper is
            # self-referential: it proves the file has not changed since today,
            # not that it matches the roster being extended. Name the sources so
            # that distinction is visible rather than buried in a count.
            print(f"  OK       {label:<28} verified by {len(hits)} ledger record(s)")
            for h in hits:
                print(f"           via {h}")
        else:
            print(f"  UNVERIFIED {label:<26} no ledger records this file's current hash")
            failures.append(label)

    print(f"\nLedger records naming a script no current file matches: {len(unmatched)}")
    print("  (expected for superseded wrapper versions and retired rosters)")

    if no_hash:
        print(f"\nRoster ledgers recording NO train-script hash -- these cannot be")
        print(f"audited at all, and this is the gap that hid the Coin City drift:")
        for item in sorted(set(no_hash)):
            print(f"  {item}")

    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard",
         "exp3_training_transfer/**/*.sh"],
        cwd=REPO, capture_output=True, text=True).stdout.split()
    if untracked:
        print(f"\nUntracked wrappers ({len(untracked)}) -- never committed, so they have")
        print("no baseline of any kind:")
        for item in untracked:
            print(f"  {item}")

    if args.strict and failures:
        raise SystemExit(f"\nFAILED: {len(failures)} wrapper(s) unverified")
    print("\naudit complete")


if __name__ == "__main__":
    main()
