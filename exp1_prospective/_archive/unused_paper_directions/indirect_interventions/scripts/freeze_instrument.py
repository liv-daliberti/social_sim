#!/usr/bin/env python3
"""Freeze only a fully audited, human-validated indirect instrument."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
DEFAULT_CANDIDATES = ROOT / "data/candidates/core_candidates.jsonl"
DEFAULT_CONTROLS = ROOT / "data/candidates/core_controls.jsonl"
DEFAULT_AUDIT = ROOT / "reports/core_candidate_audit.json"
DEFAULT_VALIDATION = ROOT / "reports/core_human_validation.json"
DEFAULT_BASELINES = ROOT / "reports/shortcut_baselines.json"
DEFAULT_ANNOTATIONS = ROOT / "data/annotations/core_annotations.json"
DEFAULT_REVIEW_MANIFEST = ROOT / "data/review/core_review_manifest.json"
DEFAULT_OUTPUT = ROOT / "data/frozen/instrument.jsonl"
DEFAULT_MANIFEST = ROOT / "data/frozen/manifest.json"
CANONICAL_FORECASTS = REPO / "exp1_prospective/data/initial_forecasts/forecasts_DeepSeek-V4-Pro_2026-06-10.jsonl"
PROTOCOL_VERSION = "indirect-core-v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family_packets(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    common = {
        "protocol_version": PROTOCOL_VERSION,
        "task_id": candidate["market"]["task_id"],
        "market_id": candidate["market"]["market_id"],
        "question": candidate["market"]["question"],
        "resolution_criteria": candidate["market"]["resolution_criteria"],
        "initial_event_model": candidate["initial_event_model"],
        "candidate_id": candidate["candidate_id"],
        "evidence": {
            "headline": candidate["evidence"]["headline"],
            "text": candidate["evidence"]["text"],
        },
    }
    rows = []
    for condition in ("positive", "negative", "broken"):
        rows.append({
            **common,
            "packet_id": f"{candidate['candidate_id']}_{condition}",
            "condition": condition,
            "bridge_context": candidate["contexts"][condition]["bridge_text"],
            "expected_label": candidate["contexts"][condition]["intended_label"],
        })
    rows.append({
        **common,
        "packet_id": f"{candidate['candidate_id']}_masked",
        "condition": "masked",
        "bridge_context": None,
        "expected_label": None,
    })
    return rows


def control_packet(control: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "packet_id": control["control_id"],
        "task_id": control["market"]["task_id"],
        "market_id": control["market"]["market_id"],
        "question": control["market"]["question"],
        "resolution_criteria": control["market"]["resolution_criteria"],
        "initial_event_model": control["initial_event_model"],
        "candidate_id": control["control_id"],
        "condition": control["condition"],
        "bridge_context": None,
        "evidence": control["evidence"],
        "expected_label": control["expected_label"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--controls", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--review-manifest", type=Path, default=DEFAULT_REVIEW_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--tau", type=float, default=0.03)
    args = parser.parse_args()
    required = (
        args.candidates, args.controls, args.audit, args.validation, args.baselines,
        args.annotations, args.review_manifest, CANONICAL_FORECASTS,
    )
    for path in required:
        if not path.exists():
            raise SystemExit(f"refusing freeze: required artifact missing: {path}")

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    baselines = json.loads(args.baselines.read_text(encoding="utf-8"))
    if not audit.get("pass") or not audit.get("complete_required"):
        raise SystemExit("refusing freeze: candidate audit did not pass as a complete core")
    if not validation.get("ready_to_freeze"):
        raise SystemExit("refusing freeze: human validation gates did not pass")
    if not baselines.get("ready_for_target_freeze"):
        raise SystemExit("refusing freeze: required bridge-masked baselines did not pass")
    retained = validation["retention"]
    market_ids = set(retained["retained_market_task_ids"])
    candidate_ids = set(retained["retained_candidate_ids"])
    control_ids = set(retained["retained_control_ids"])
    candidates = [
        row for row in read_jsonl(args.candidates)
        if row["candidate_id"] in candidate_ids
        and row["market"]["task_id"] in market_ids
    ]
    controls = [
        row for row in read_jsonl(args.controls)
        if row["control_id"] in control_ids
        and row["market"]["task_id"] in market_ids
    ]
    if len(market_ids) < 80 or len(candidates) != 3 * len(market_ids):
        raise SystemExit("refusing freeze: retained sample is below registered 80 x 3 minimum")
    if len(controls) != 4 * len(market_ids):
        raise SystemExit("refusing freeze: retained controls are not four per market")

    packets = [
        packet
        for candidate in candidates
        for packet in family_packets(candidate)
    ] + [control_packet(control) for control in controls]
    expected_count = 16 * len(market_ids)
    if len(packets) != expected_count:
        raise SystemExit(f"refusing freeze: {len(packets)} packets, expected {expected_count}")
    rng = random.Random(args.seed)
    rng.shuffle(packets)
    instrument_bytes = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in packets
    ).encode()
    if args.output.exists() and args.output.read_bytes() != instrument_bytes:
        raise SystemExit("refusing to overwrite an existing, different frozen instrument")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(instrument_bytes)

    tracked = {
        "candidates": args.candidates,
        "controls": args.controls,
        "candidate_audit": args.audit,
        "human_validation": args.validation,
        "shortcut_baselines": args.baselines,
        "annotations": args.annotations,
        "review_manifest": args.review_manifest,
        "canonical_forecasts": CANONICAL_FORECASTS,
        "generation_prompt": ROOT / "prompts/generate_family.txt",
        "update_prompt": ROOT / "prompts/update.txt",
        "candidate_schema": ROOT / "schema/candidate.schema.json",
        "annotation_schema": ROOT / "schema/annotation.schema.json",
        "evaluation_schema": ROOT / "schema/evaluation.schema.json",
    }
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "tau": args.tau,
        "market_count": len(market_ids),
        "family_count": len(candidates),
        "packet_count": len(packets),
        "instrument_sha256": hashlib.sha256(instrument_bytes).hexdigest(),
        "artifact_sha256": {name: digest(path) for name, path in tracked.items()},
        "target_outputs_opened_before_freeze": False,
        "status": "FROZEN_FOR_TARGET_EVALUATION",
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if args.manifest.exists():
        old = json.loads(args.manifest.read_text(encoding="utf-8"))
        stable_keys = ("seed", "tau", "market_count", "family_count", "packet_count", "instrument_sha256", "artifact_sha256")
        if any(old.get(key) != manifest.get(key) for key in stable_keys):
            raise SystemExit("refusing to overwrite an existing, different freeze manifest")
        print("frozen artifacts already match manifest")
        return
    args.manifest.write_bytes(manifest_bytes)
    print(
        f"FROZEN: {len(market_ids)} markets, {len(candidates)} families, "
        f"{len(packets)} packets; sha256={manifest['instrument_sha256']}"
    )


if __name__ == "__main__":
    main()
