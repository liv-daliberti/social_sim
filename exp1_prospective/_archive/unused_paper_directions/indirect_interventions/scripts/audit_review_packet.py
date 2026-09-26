#!/usr/bin/env python3
"""Fail-closed structural and blinding audit for a human-review packet."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_CANDIDATES = ROOT / "data/development/pilot_candidates.jsonl"
DEFAULT_PACKET = ROOT / "data/development/pilot_v2_review_packet.json"
DEFAULT_REVIEWERS = ROOT / "data/development/pilot_v2_reviewers.json"
DEFAULT_MANIFEST = ROOT / "data/development/pilot_v2_review_manifest.json"

PRIVATE_KEYS = {
    "candidate_id",
    "market_id",
    "intended_label",
    "chain_nodes",
    "chain_edges",
    "generation",
    "bridge_frame",
    "initial_event_model",
    "role",
    "family",
}
EXPECTED_LABELS = {
    "positive": "increase_yes",
    "negative": "decrease_yes",
    "broken": "no_material_effect",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keys_recursive(value: Any) -> set[str]:
    if isinstance(value, dict):
        result = set(value)
        for child in value.values():
            result.update(keys_recursive(child))
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for child in value:
            result.update(keys_recursive(child))
        return result
    return set()


def audit(
    candidates_path: Path,
    packet_path: Path,
    reviewers_path: Path,
    manifest_path: Path,
) -> list[str]:
    errors: list[str] = []
    candidates = read_jsonl(candidates_path)
    packet = read_json(packet_path)
    reviewers_file = read_json(reviewers_path)
    manifest = read_json(manifest_path)

    if len(candidates) != 3:
        errors.append(f"expected 3 pilot families, found {len(candidates)}")
    candidate_ids = [row.get("candidate_id") for row in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        errors.append("candidate IDs are not unique")

    for candidate in candidates:
        if candidate.get("development_only") is not True:
            errors.append(f"{candidate.get('candidate_id')}: development_only must be true")
        contexts = candidate.get("contexts", {})
        if set(contexts) != set(EXPECTED_LABELS):
            errors.append(f"{candidate.get('candidate_id')}: expected positive/negative/broken contexts")
            continue
        for role, label in EXPECTED_LABELS.items():
            context = contexts[role]
            if context.get("intended_label") != label:
                errors.append(f"{candidate.get('candidate_id')}/{role}: wrong private label")
            nodes = context.get("chain_nodes", [])
            edges = context.get("chain_edges", [])
            if len(edges) < 2 or len(nodes) != len(edges) + 1:
                errors.append(f"{candidate.get('candidate_id')}/{role}: malformed causal chain")
        if contexts["positive"].get("bridge_text") == contexts["negative"].get("bridge_text"):
            errors.append(f"{candidate.get('candidate_id')}: directional bridges are identical")

    if packet.get("development_only") is not True:
        errors.append("browser packet is not marked development_only")
    panels = packet.get("panels", {})
    full = panels.get("full_context", [])
    shortcut = panels.get("shortcut", [])
    if len(full) != 9 or len(shortcut) != 3:
        errors.append(f"expected 9 full and 3 shortcut tasks, found {len(full)} and {len(shortcut)}")
    all_tasks = full + shortcut
    task_ids = [row.get("task_id") for row in all_tasks]
    if len(task_ids) != len(set(task_ids)):
        errors.append("browser task IDs are not unique")
    leaked = keys_recursive(packet) & PRIVATE_KEYS
    if leaked:
        errors.append("private keys leaked into browser packet: " + ", ".join(sorted(leaked)))

    for task in full:
        required = {"task_id", "panel", "market", "evidence", "bridge_context"}
        if set(task) != required or task.get("panel") != "full_context":
            errors.append(f"{task.get('task_id')}: malformed full-context task")
    for task in shortcut:
        required = {"task_id", "panel", "market", "evidence"}
        if set(task) != required or task.get("panel") != "shortcut":
            errors.append(f"{task.get('task_id')}: shortcut task exposes extra or missing fields")

    full_evidence = {}
    for task in full:
        key = json.dumps(task.get("evidence"), sort_keys=True)
        full_evidence[key] = full_evidence.get(key, 0) + 1
    if sorted(full_evidence.values()) != [3, 3, 3]:
        errors.append("full-context evidence is not repeated byte-identically three times per family")
    shortcut_evidence = {json.dumps(task.get("evidence"), sort_keys=True) for task in shortcut}
    if shortcut_evidence != set(full_evidence):
        errors.append("shortcut panel does not contain exactly the three full-panel evidence items")

    reviewers = reviewers_file.get("reviewers", [])
    if len(reviewers) != 12:
        errors.append(f"expected 12 reviewer codes, found {len(reviewers)}")
    codes = [row.get("access_code") for row in reviewers]
    reviewer_ids = [row.get("reviewer_id") for row in reviewers]
    if len(codes) != len(set(codes)) or len(reviewer_ids) != len(set(reviewer_ids)):
        errors.append("reviewer IDs or access codes are not unique")
    full_ids = {row["task_id"] for row in full}
    shortcut_ids = {row["task_id"] for row in shortcut}
    task_groups = {
        row["task_id"]: json.dumps(row.get("market", {}), sort_keys=True)
        for row in all_tasks
    }
    ownership: Counter[str] = Counter()
    for reviewer in reviewers:
        assigned = reviewer.get("task_ids", [])
        expected = full_ids if reviewer.get("panel") == "full_context" else shortcut_ids
        if not set(assigned) <= expected or len(assigned) != len(set(assigned)):
            errors.append(
                f"{reviewer.get('reviewer_id')}: assignment includes a duplicate "
                "or task from the wrong panel"
            )
        ownership.update(assigned)
        assigned_groups = [
            task_groups[task_id] for task_id in assigned if task_id in task_groups
        ]
        if len(assigned_groups) != len(set(assigned_groups)):
            errors.append(
                f"{reviewer.get('reviewer_id')}: repeated market exposure reveals "
                "within-family variants"
            )
    ratings_per_task = int(reviewers_file.get("ratings_per_task", 0))
    if ratings_per_task != 3:
        errors.append(f"expected three ratings per task, found {ratings_per_task}")
    if set(ownership) != set(task_ids) or set(ownership.values()) != {3}:
        errors.append("tasks do not each have exactly three assigned reviewers")
    if reviewers_file.get("assignment_constraint") != (
        "at_most_one_task_per_market_per_reviewer"
    ):
        errors.append("reviewer file does not register the market-disjoint constraint")

    expected_hashes = {
        "packet_sha256": sha256(packet_path),
        "reviewers_sha256": sha256(reviewers_path),
        "private_candidates_sha256": sha256(candidates_path),
    }
    for key, actual in expected_hashes.items():
        if manifest.get(key) != actual:
            errors.append(f"manifest {key} does not match")

    serialized = packet_path.read_text(encoding="utf-8")
    for forbidden in ("increase_yes", "decrease_yes", "no_material_effect", "draft_for_human_validation"):
        if forbidden in serialized:
            errors.append(f"private value leaked into browser packet: {forbidden}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--reviewers", type=Path, default=DEFAULT_REVIEWERS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    errors = audit(args.candidates, args.packet, args.reviewers, args.manifest)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        raise SystemExit(1)
    print("PASS: 3 families; 9 full-context tasks; 3 shortcut tasks; 12 reviewer codes")
    print("PASS: every task has three raters; no reviewer sees a repeated market")
    print("PASS: private roles, labels, chains, and generator metadata are absent from the browser packet")
    print("PASS: packet and assignment hashes match the manifest")


if __name__ == "__main__":
    main()
