#!/usr/bin/env python3
"""Build blinded core-review tasks and balanced three-rater assignments."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import random
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_CANDIDATES = ROOT / "data/candidates/core_candidates.jsonl"
DEFAULT_CONTROLS = ROOT / "data/candidates/core_controls.jsonl"
DEFAULT_PACKET = ROOT / "data/review/core_review_packet.json"
DEFAULT_REVIEWERS = ROOT / "data/review/core_reviewers.json"
DEFAULT_KEY = ROOT / "data/review/core_review_key.jsonl"
DEFAULT_MANIFEST = ROOT / "data/review/core_review_manifest.json"
PROTOCOL_VERSION = "indirect-core-v1"
EVIDENCE_WORDS = (20, 45)
DIRECTIONAL_BRIDGE_WORDS = (20, 40)
BROKEN_BRIDGE_WORDS = (15, 28)
DECISIVE_CLAUSE_WORDS = (3, 12)
BRIDGE_ACRONYM = re.compile(r"\b[A-Z]{2,}\b")
MAX_BRIDGE_SENTENCES = 2
MAX_BRIDGE_SENTENCE_WORDS = 24


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def opaque_id(panel: str, source_id: str, role: str, version: str = PROTOCOL_VERSION) -> str:
    raw = f"{version}|{panel}|{source_id}|{role}".encode()
    return "review_" + hashlib.sha256(raw).hexdigest()[:22]


def public_market(row: dict[str, Any]) -> dict[str, str]:
    market = row["market"]
    return {
        "question": market["question"],
        "resolution_criteria": market["resolution_criteria"],
    }


def candidate_readability_errors(candidate: dict[str, Any]) -> list[str]:
    """Return reviewer-facing word-budget violations for one family."""
    cid = candidate.get("candidate_id", "<missing>")
    errors: list[str] = []
    evidence_words = len(
        (candidate["evidence"]["headline"] + " " + candidate["evidence"]["text"]).split()
    )
    if not EVIDENCE_WORDS[0] <= evidence_words <= EVIDENCE_WORDS[1]:
        errors.append(f"{cid}: evidence={evidence_words} words")
    for role in ("positive", "negative"):
        context = candidate["contexts"][role]
        bridge_words = len(context["bridge_text"].split())
        clause_words = len(context.get("decisive_clause", "").split())
        if not DIRECTIONAL_BRIDGE_WORDS[0] <= bridge_words <= DIRECTIONAL_BRIDGE_WORDS[1]:
            errors.append(f"{cid}/{role}: bridge={bridge_words} words")
        if not DECISIVE_CLAUSE_WORDS[0] <= clause_words <= DECISIVE_CLAUSE_WORDS[1]:
            errors.append(f"{cid}/{role}: clause={clause_words} words")
    broken_words = len(candidate["contexts"]["broken"]["bridge_text"].split())
    if not BROKEN_BRIDGE_WORDS[0] <= broken_words <= BROKEN_BRIDGE_WORDS[1]:
        errors.append(f"{cid}/broken: bridge={broken_words} words")
    for role, context in candidate["contexts"].items():
        bridge_text = context["bridge_text"]
        acronyms = sorted(set(BRIDGE_ACRONYM.findall(bridge_text)))
        acronyms = [value for value in acronyms if value not in {"YES", "NO"}]
        if acronyms:
            errors.append(
                f"{cid}/{role}: all-caps abbreviation(s)={','.join(acronyms)}"
            )
        sentences = [
            sentence.strip() for sentence in re.split(r"[.!?]+", bridge_text)
            if sentence.strip()
        ]
        if len(sentences) > MAX_BRIDGE_SENTENCES:
            errors.append(f"{cid}/{role}: sentences={len(sentences)}")
        if any(len(sentence.split()) > MAX_BRIDGE_SENTENCE_WORDS for sentence in sentences):
            errors.append(
                f"{cid}/{role}: sentence>{MAX_BRIDGE_SENTENCE_WORDS} words"
            )
        if ";" in bridge_text:
            errors.append(f"{cid}/{role}: semicolon")
    return errors


def assignments(
    task_ids: list[str],
    *,
    panel: str,
    reviewer_count: int,
    ratings_per_task: int,
    seed: int,
    group_ids: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    if reviewer_count < ratings_per_task:
        raise ValueError("reviewer count must be >= ratings per task")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task IDs must be unique")
    task_groups = group_ids or {task_id: task_id for task_id in task_ids}
    if set(task_groups) != set(task_ids):
        raise ValueError("group IDs must cover exactly the assigned tasks")
    grouped: dict[str, list[str]] = defaultdict(list)
    for task_id in task_ids:
        grouped[task_groups[task_id]].append(task_id)
    maximum_group_assignments = max(
        (len(tasks) * ratings_per_task for tasks in grouped.values()), default=0
    )
    if maximum_group_assignments > reviewer_count:
        raise ValueError(
            f"no-repeat assignment requires at least {maximum_group_assignments} "
            f"{panel} reviewers; got {reviewer_count}"
        )

    buckets: list[list[str]] = [[] for _ in range(reviewer_count)]
    loads = [0] * reviewer_count
    rng = random.Random(f"{seed}|{panel}|market-disjoint-v1")
    group_order = sorted(grouped)
    rng.shuffle(group_order)
    for group_id in group_order:
        tasks = sorted(grouped[group_id])
        rng.shuffle(tasks)
        needed = len(tasks) * ratings_per_task
        tie_order = list(range(reviewer_count))
        rng.shuffle(tie_order)
        selected = sorted(tie_order, key=lambda index: loads[index])[:needed]
        rng.shuffle(selected)
        for task_index, task_id in enumerate(tasks):
            owners = selected[
                task_index * ratings_per_task:(task_index + 1) * ratings_per_task
            ]
            if len(owners) != ratings_per_task or len(set(owners)) != ratings_per_task:
                raise AssertionError("assignment did not produce independent reviewers")
            for reviewer_index in owners:
                buckets[reviewer_index].append(task_id)
                loads[reviewer_index] += 1
    reviewers = []
    for index, bucket in enumerate(buckets):
        rng = random.Random(f"{seed}|{panel}|{index}")
        rng.shuffle(bucket)
        reviewers.append({
            "reviewer_id": f"{panel}_{index + 1:03d}",
            "access_code": secrets.token_urlsafe(12),
            "panel": panel,
            "task_ids": bucket,
        })
    counts = {task_id: 0 for task_id in task_ids}
    for reviewer in reviewers:
        for task_id in reviewer["task_ids"]:
            counts[task_id] += 1
    if set(counts.values()) != {ratings_per_task}:
        raise AssertionError("tasks do not have exactly the registered number of ratings")
    for reviewer in reviewers:
        assigned_groups = [task_groups[task_id] for task_id in reviewer["task_ids"]]
        if len(assigned_groups) != len(set(assigned_groups)):
            raise AssertionError("reviewer received repeated variants from one market")
    reviewer_loads = [len(reviewer["task_ids"]) for reviewer in reviewers]
    if reviewer_loads and max(reviewer_loads) - min(reviewer_loads) > 1:
        raise AssertionError("reviewer loads differ by more than one task")
    return reviewers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--controls", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--reviewers", type=Path, default=DEFAULT_REVIEWERS)
    parser.add_argument("--private-key", type=Path, default=DEFAULT_KEY)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--full-reviewers", type=int)
    parser.add_argument("--shortcut-reviewers", type=int)
    parser.add_argument("--ratings-per-task", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    full_reviewer_count = args.full_reviewers or (27 if args.development else 90)
    shortcut_reviewer_count = args.shortcut_reviewers or (9 if args.development else 30)

    candidates = read_jsonl(args.candidates)
    controls = [] if args.development else read_jsonl(args.controls)
    expected_candidates = 60 if args.development else 300
    expected_controls = 0 if args.development else 400
    if len(candidates) != expected_candidates or len(controls) != expected_controls:
        raise SystemExit(
            f"refusing incomplete candidate pool: {len(candidates)} candidates, "
            f"{len(controls)} controls"
        )
    unedited = [
        row.get("candidate_id", "<missing>")
        for row in candidates
        if row.get("generation", {}).get("status") != "human_edited"
        or not row.get("generation", {}).get("editor_id")
        or not row.get("generation", {}).get("edited_at")
    ]
    if unedited:
        raise SystemExit(
            f"refusing blinded ratings before independent human editing: "
            f"{len(unedited)} candidates lack complete editor provenance"
        )
    readability_errors: list[str] = []
    for candidate in candidates:
        readability_errors.extend(candidate_readability_errors(candidate))
    if readability_errors:
        raise SystemExit(
            f"refusing reviewer packet: {len(readability_errors)} readability "
            f"violations; first: {readability_errors[0]}"
        )
    version = "indirect-development-v1" if args.development else PROTOCOL_VERSION

    full_tasks: list[dict[str, Any]] = []
    shortcut_tasks: list[dict[str, Any]] = []
    private_key: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda row: row["candidate_id"]):
        for role in ("positive", "negative", "broken"):
            task_id = opaque_id("full_context", candidate["candidate_id"], role, version)
            full_tasks.append({
                "task_id": task_id,
                "panel": "full_context",
                "market": public_market(candidate),
                "evidence": {
                    "headline": candidate["evidence"]["headline"],
                    "text": candidate["evidence"]["text"],
                },
                "bridge_context": candidate["contexts"][role]["bridge_text"],
            })
            private_key.append({
                "task_id": task_id,
                "panel": "full_context",
                "source_type": "family",
                "source_id": candidate["candidate_id"],
                "market_task_id": candidate["market"]["task_id"],
                "role": role,
                "expected_label": candidate["contexts"][role]["intended_label"],
            })
        task_id = opaque_id("shortcut", candidate["candidate_id"], "masked", version)
        shortcut_tasks.append({
            "task_id": task_id,
            "panel": "shortcut",
            "market": public_market(candidate),
            "evidence": {
                "headline": candidate["evidence"]["headline"],
                "text": candidate["evidence"]["text"],
            },
        })
        private_key.append({
            "task_id": task_id,
            "panel": "shortcut",
            "source_type": "family",
            "source_id": candidate["candidate_id"],
            "market_task_id": candidate["market"]["task_id"],
            "role": "masked",
            "expected_label": None,
        })

    for control in sorted(controls, key=lambda row: row["control_id"]):
        task_id = opaque_id("full_context", control["control_id"], control["condition"], version)
        full_tasks.append({
            "task_id": task_id,
            "panel": "full_context",
            "market": public_market(control),
            "evidence": control["evidence"],
        })
        private_key.append({
            "task_id": task_id,
            "panel": "full_context",
            "source_type": "control",
            "source_id": control["control_id"],
            "market_task_id": control["market"]["task_id"],
            "role": control["condition"],
            "expected_label": control["expected_label"],
        })

    packet = {
        "protocol_version": version,
        "packet_name": (
            "development candidate human review"
            if args.development else "frozen-core candidate human review"
        ),
        "development_only": args.development,
        "panels": {
            "full_context": full_tasks,
            "shortcut": shortcut_tasks,
        },
    }
    full_group_ids = {
        row["task_id"]: row["market_task_id"]
        for row in private_key if row["panel"] == "full_context"
    }
    shortcut_group_ids = {
        row["task_id"]: row["market_task_id"]
        for row in private_key if row["panel"] == "shortcut"
    }
    reviewer_rows = (
        assignments(
            [task["task_id"] for task in full_tasks],
            panel="full_context",
            reviewer_count=full_reviewer_count,
            ratings_per_task=args.ratings_per_task,
            seed=args.seed,
            group_ids=full_group_ids,
        )
        + assignments(
            [task["task_id"] for task in shortcut_tasks],
            panel="shortcut",
            reviewer_count=shortcut_reviewer_count,
            ratings_per_task=args.ratings_per_task,
            seed=args.seed,
            group_ids=shortcut_group_ids,
        )
    )
    reviewers = {
        "protocol_version": version,
        "ratings_per_task": args.ratings_per_task,
        "assignment_constraint": "at_most_one_task_per_market_per_reviewer",
        "reviewers": reviewer_rows,
    }

    packet_bytes = (json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    reviewer_bytes = (json.dumps(reviewers, indent=2, sort_keys=True) + "\n").encode()
    key_bytes = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in private_key
    ).encode()
    for path in (args.packet, args.reviewers, args.private_key, args.manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.packet.write_bytes(packet_bytes)
    args.reviewers.write_bytes(reviewer_bytes)
    args.private_key.write_bytes(key_bytes)
    manifest = {
        "protocol_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "ratings_per_task": args.ratings_per_task,
        "candidate_count": len(candidates),
        "control_count": len(controls),
        "full_context_task_count": len(full_tasks),
        "shortcut_task_count": len(shortcut_tasks),
        "reviewer_count": len(reviewer_rows),
        "full_reviewer_count": full_reviewer_count,
        "shortcut_reviewer_count": shortcut_reviewer_count,
        "assignment_constraint": "at_most_one_task_per_market_per_reviewer",
        "packet_sha256": sha256_bytes(packet_bytes),
        "reviewers_sha256": sha256_bytes(reviewer_bytes),
        "private_key_sha256": sha256_bytes(key_bytes),
        "candidates_sha256": sha256_bytes(args.candidates.read_bytes()),
        "controls_sha256": (
            sha256_bytes(args.controls.read_bytes()) if not args.development else None
        ),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    loads = [len(row["task_ids"]) for row in reviewer_rows]
    print(
        f"wrote {len(full_tasks)} full-context + {len(shortcut_tasks)} shortcut tasks; "
        f"{len(reviewer_rows)} reviewers; load {min(loads)}--{max(loads)} cases"
    )


if __name__ == "__main__":
    main()
