#!/usr/bin/env python3
"""Build the unopened, family-disjoint holdout for the Exp4 Qwen scale test."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import build_exp3b_dataset as registered


ROOT = Path(__file__).resolve().parents[1]
REGISTERED_DATA = ROOT / "data" / "exp3b_registered"
DEFAULT_OUTPUT = ROOT / "data" / "exp4_scale_registered"
PROTOCOL = ROOT / "SCALE_PROTOCOL.md"
PROTOCOL_VERSION = "exp4_qwen_scale_v1"
MINIMUM_HOLDOUT = 256


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def family_keys(row: dict[str, Any]) -> set[str]:
    return registered.leakage_keys(row)


def union_keys(rows: Iterable[dict[str, Any]]) -> set[str]:
    result: set[str] = set()
    for row in rows:
        result.update(family_keys(row))
    return result


def verify_registered_inputs() -> (
    tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]
):
    manifest_path = REGISTERED_DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass":
        raise AssertionError("registered Exp4 manifest is not a pass")
    if manifest.get("protocol_version") != "exp3b_registered_v1":
        raise AssertionError("registered Exp4 protocol version drifted")
    rows: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "dev", "test"):
        path = REGISTERED_DATA / f"{split}.tasks.jsonl"
        if sha256(path) != manifest["output_sha256"][path.name]:
            raise AssertionError(f"registered {split} split hash drifted")
        rows[split] = read_jsonl(path)
    return manifest, rows


def family_clean_test_candidates(
    candidates: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Reproduce the registered all-candidate train/dev family exclusions."""
    train_keys = union_keys(candidates["train"])
    dev = [row for row in candidates["dev"] if not (family_keys(row) & train_keys)]
    earlier_keys = train_keys | union_keys(dev)
    return [row for row in candidates["test"] if not (family_keys(row) & earlier_keys)]


def select_scale_holdout(
    candidates: dict[str, list[dict[str, Any]]],
    registered_rows: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    clean_test = family_clean_test_candidates(candidates)
    exposed = [
        row for split in ("train", "dev", "test") for row in registered_rows[split]
    ]
    exposed_keys = union_keys(exposed)
    exposed_ids = {row["task_id"] for row in exposed}
    holdout = [
        row
        for row in clean_test
        if row["task_id"] not in exposed_ids and not (family_keys(row) & exposed_keys)
    ]
    holdout.sort(
        key=lambda row: hashlib.sha256(
            f"{PROTOCOL_VERSION}|holdout|{row['task_id']}".encode()
        ).hexdigest()
    )
    task_ids = [row["task_id"] for row in holdout]
    if len(task_ids) != len(set(task_ids)):
        raise AssertionError("scale holdout contains duplicate task IDs")
    if union_keys(holdout) & exposed_keys:
        raise AssertionError("scale holdout overlaps an exposed family")
    return holdout


def scan_candidates(
    snapshot: Path, max_rows: int = 0
) -> tuple[dict[str, list[dict[str, Any]]], str, Counter[str], int]:
    candidates: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "dev": [],
        "test": [],
    }
    skipped: Counter[str] = Counter()
    digest = hashlib.sha256()
    parsed = 0
    for row, fast_skip in registered.iter_archive(snapshot, digest, max_rows):
        parsed += 1
        if parsed % 100_000 == 0:
            print(f"parsed {parsed:,} archive rows", flush=True)
        if fast_skip:
            skipped[fast_skip] += 1
            continue
        assert row is not None
        task = registered.task_from_row(row, skipped)
        if task is None:
            continue
        split = registered.split_name(task)
        if split is not None:
            candidates[split].append(task)
    return candidates, digest.hexdigest(), skipped, parsed


def public_description(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dates = sorted(row["decision_ts"] for row in rows)
    return {
        "count": len(rows),
        "first_decision_ts": dates[0] if dates else None,
        "last_decision_ts": dates[-1] if dates else None,
        "distinct_events": len({row["event_id"] for row in rows}),
        "distinct_series": len({row["series_id"] for row in rows if row["series_id"]}),
        "distinct_templates": len({row["template_key"] for row in rows}),
        "outcomes_summarized": False,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def build(args: argparse.Namespace) -> dict[str, Any]:
    registered_manifest, registered_rows = verify_registered_inputs()
    candidates, snapshot_hash, skipped, parsed = scan_candidates(
        args.snapshot, args.max_rows
    )
    if not args.max_rows and snapshot_hash != registered.EXPECTED_SNAPSHOT_SHA256:
        raise AssertionError("frozen Polymarket archive hash drifted")
    holdout = select_scale_holdout(candidates, registered_rows)
    if not args.max_rows and len(holdout) < MINIMUM_HOLDOUT:
        raise AssertionError(
            f"only {len(holdout)} family-disjoint holdout tasks; "
            f"minimum is {MINIMUM_HOLDOUT}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "test.tasks.jsonl"
    write_jsonl(output_path, holdout)
    manifest = {
        "status": "pass" if not args.max_rows else "audit_only",
        "protocol_version": PROTOCOL_VERSION,
        "test_was_used_for_protocol_or_hyperparameter_selection": False,
        "selection_uses_outcomes_or_market_probabilities": False,
        "source": {
            "path": str(args.snapshot),
            "sha256": snapshot_hash,
            "expected_sha256": registered.EXPECTED_SNAPSHOT_SHA256,
            "fully_scanned": not bool(args.max_rows),
            "parsed_rows": parsed,
        },
        "registered_parent": {
            "protocol_version": registered_manifest["protocol_version"],
            "manifest_sha256": sha256(REGISTERED_DATA / "manifest.json"),
            "split_sha256": {
                split: sha256(REGISTERED_DATA / f"{split}.tasks.jsonl")
                for split in ("train", "dev", "test")
            },
        },
        "selection": {
            "eligible_test_before_opened_test_exclusion": len(
                family_clean_test_candidates(candidates)
            ),
            "included_all_remaining_family_disjoint_tasks": True,
            "ordering": "sha256(protocol_version|holdout|task_id)",
            "minimum_required": MINIMUM_HOLDOUT,
        },
        "holdout": public_description(holdout),
        "candidate_counts_before_family_filter": {
            split: len(rows) for split, rows in candidates.items()
        },
        "skipped_counts": dict(sorted(skipped.items())),
        "hashes": {
            "protocol_sha256": sha256(PROTOCOL),
            "builder_sha256": sha256(Path(__file__)),
            "test_tasks_sha256": sha256(output_path),
        },
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--snapshot", type=Path, default=registered.DEFAULT_SNAPSHOT)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Smoke-test prefix; disables full hash and minimum-size checks",
    )
    return result


if __name__ == "__main__":
    build(parser().parse_args())
