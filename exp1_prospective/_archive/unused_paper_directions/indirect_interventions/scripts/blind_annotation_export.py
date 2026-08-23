#!/usr/bin/env python3
"""Build a market-disjoint v2 UI-pilot packet and reviewer assignments."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_CANDIDATES = ROOT / "data/development/pilot_candidates.jsonl"
DEFAULT_PACKET = ROOT / "data/development/pilot_v2_review_packet.json"
DEFAULT_REVIEWERS = ROOT / "data/development/pilot_v2_reviewers.json"
DEFAULT_MANIFEST = ROOT / "data/development/pilot_v2_review_manifest.json"
PROTOCOL_VERSION = "indirect-pilot-v2"

REVIEWERS = [
    ("pilot_full_01", "maple-7-harbor-92", "full_context"),
    ("pilot_full_02", "cobalt-4-meadow-61", "full_context"),
    ("pilot_full_03", "cedar-8-comet-35", "full_context"),
    ("pilot_full_04", "granite-2-lantern-47", "full_context"),
    ("pilot_full_05", "willow-5-compass-83", "full_context"),
    ("pilot_full_06", "saffron-1-forest-64", "full_context"),
    ("pilot_full_07", "indigo-8-summit-26", "full_context"),
    ("pilot_full_08", "copper-3-valley-75", "full_context"),
    ("pilot_full_09", "juniper-6-aster-41", "full_context"),
    ("pilot_shortcut_01", "amber-3-river-74", "shortcut"),
    ("pilot_shortcut_02", "violet-6-orbit-28", "shortcut"),
    ("pilot_shortcut_03", "silver-9-garden-43", "shortcut"),
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def opaque_id(panel: str, candidate_id: str, role: str) -> str:
    raw = f"{PROTOCOL_VERSION}|{panel}|{candidate_id}|{role}".encode()
    return "review_" + hashlib.sha256(raw).hexdigest()[:18]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def public_market(candidate: dict) -> dict:
    market = candidate["market"]
    return {
        "question": market["question"],
        "resolution_criteria": market["resolution_criteria"],
    }


def build_orders(full_by_family: list[dict[str, str]], shortcut_ids: list[str]) -> dict[str, list[str]]:
    orders: dict[str, list[str]] = {}
    roles = ("positive", "negative", "broken")
    full_reviewers = [row for row in REVIEWERS if row[2] == "full_context"]
    for reviewer_index, (reviewer_id, _, _) in enumerate(full_reviewers):
        order = []
        for family_index, family in enumerate(full_by_family):
            role = roles[(reviewer_index + family_index) % len(roles)]
            order.append(family[role])
        assert len(order) == len(set(order)) == len(full_by_family)
        orders[reviewer_id] = order
    shortcut_reviewers = [row for row in REVIEWERS if row[2] == "shortcut"]
    for reviewer_index, (reviewer_id, _, _) in enumerate(shortcut_reviewers):
        shift = reviewer_index % len(shortcut_ids)
        orders[reviewer_id] = shortcut_ids[shift:] + shortcut_ids[:shift]
    return orders


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--reviewers", type=Path, default=DEFAULT_REVIEWERS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    candidates = read_jsonl(args.candidates)
    full_tasks = []
    shortcut_tasks = []
    full_by_family = []
    shortcut_ids = []
    for candidate in candidates:
        family_tasks: dict[str, str] = {}
        for role in ("positive", "negative", "broken"):
            task_id = opaque_id("full_context", candidate["candidate_id"], role)
            family_tasks[role] = task_id
            full_tasks.append({
                "task_id": task_id,
                "panel": "full_context",
                "market": public_market(candidate),
                "evidence": candidate["evidence"],
                "bridge_context": candidate["contexts"][role]["bridge_text"],
            })
        full_by_family.append(family_tasks)
        task_id = opaque_id("shortcut", candidate["candidate_id"], "masked")
        shortcut_ids.append(task_id)
        shortcut_tasks.append({
            "task_id": task_id,
            "panel": "shortcut",
            "market": public_market(candidate),
            "evidence": candidate["evidence"],
        })

    packet = {
        "protocol_version": PROTOCOL_VERSION,
        "packet_name": "market-disjoint development UI pilot",
        "development_only": True,
        "panels": {"full_context": full_tasks, "shortcut": shortcut_tasks},
    }
    packet_bytes = (json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    args.packet.parent.mkdir(parents=True, exist_ok=True)
    args.packet.write_bytes(packet_bytes)

    orders = build_orders(full_by_family, shortcut_ids)
    reviewers = {
        "protocol_version": packet["protocol_version"],
        "ratings_per_task": 3,
        "assignment_constraint": "at_most_one_task_per_market_per_reviewer",
        "reviewers": [
            {"reviewer_id": rid, "access_code": code, "panel": panel, "task_ids": orders[rid]}
            for rid, code, panel in REVIEWERS
        ],
    }
    reviewer_bytes = (json.dumps(reviewers, indent=2, sort_keys=True) + "\n").encode()
    args.reviewers.write_bytes(reviewer_bytes)

    manifest = {
        "protocol_version": packet["protocol_version"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "development_only": True,
        "candidate_count": len(candidates),
        "full_context_task_count": len(full_tasks),
        "shortcut_task_count": len(shortcut_tasks),
        "reviewer_count": len(REVIEWERS),
        "full_reviewer_count": 9,
        "shortcut_reviewer_count": 3,
        "ratings_per_task": 3,
        "assignment_constraint": "at_most_one_task_per_market_per_reviewer",
        "packet_sha256": sha256_bytes(packet_bytes),
        "reviewers_sha256": sha256_bytes(reviewer_bytes),
        "private_candidates_sha256": sha256_bytes(args.candidates.read_bytes()),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(full_tasks)} full-context and {len(shortcut_tasks)} shortcut tasks")


if __name__ == "__main__":
    main()
