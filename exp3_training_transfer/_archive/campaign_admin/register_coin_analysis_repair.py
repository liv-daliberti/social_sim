#!/usr/bin/env python3
"""Pin the corrected Coin City paper renderer and replace its pending finalizer.

The frozen data use ``none`` as the no-hint cue key.  A pre-execution ingestion
audit found that the post-run renderer expected ``absent``.  This registrar makes
that analysis-only correction auditable without altering any scientific job,
dataset, prompt, target, checkpoint, or estimand.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
COIN = ROOT / "coin_city_structural"
OLD_RENDERER_SHA256 = "fbd2464a4de0f19047818d2d85109fd7d94e13b6ab24390fa4bd7e90cc5de67e"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_pending(job_id: str) -> None:
    output = subprocess.check_output(
        ["squeue", "-h", "-j", job_id, "-o", "%i|%T"], text=True
    ).strip()
    if output != f"{job_id}|PENDING":
        raise SystemExit(f"refusing to replace non-pending finalizer {job_id}: {output!r}")


def roster_ids(ledger: dict) -> list[str]:
    confirmatory = [row for row in ledger.get("full_training", [])
                    if row.get("kind") == "confirmatory"]
    diagnostic = [row for row in ledger.get("full_training", [])
                  if row.get("kind") == "diagnostic"]
    bases = ledger.get("base_evaluations", [])
    if (len(confirmatory), len(diagnostic), len(bases)) != (18, 3, 3):
        raise AssertionError("Coin City ledger is not the exact 18+3+3 roster")
    ids = [str(row["job_id"]) for row in confirmatory + diagnostic + bases]
    if len(set(ids)) != 24 or any(not value.isdigit() for value in ids):
        raise AssertionError("Coin City scientific endpoints are not 24 distinct numeric IDs")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin-ledger", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--old-finalizer", default="30731849")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ledger_path = args.coin_ledger.resolve()
    handoff_path = args.handoff.resolve()
    ledger = load(ledger_path)
    handoff = load(handoff_path)
    ids = roster_ids(ledger)
    assert_pending(args.old_finalizer)

    renderer = COIN / "make_paper_outputs.py"
    finalizer_script = COIN / "finalize_campaign.sbatch"
    renderer_hash = sha256(renderer)
    if renderer_hash == OLD_RENDERER_SHA256:
        raise AssertionError("renderer still has the pre-repair hash")
    command = [
        "sbatch", "--parsable", "--partition=cs", "--cpus-per-task=4", "--mem=16G",
        "--time=01:00:00", "--dependency=afterany:" + ":".join(ids),
        "--export=ALL," +
        f"CAMPAIGN_LEDGER={ledger_path},EXPECTED_RENDERER_SHA256={renderer_hash}",
        str(finalizer_script),
    ]
    print(shlex.join(command), flush=True)
    if args.dry_run:
        print("DRY RUN: would submit replacement, write a derived handoff, then cancel "
              + args.old_finalizer)
        return

    output = subprocess.check_output(command, cwd=REPO, text=True).strip()
    new_finalizer = output.split(";", 1)[0]
    if not new_finalizer.isdigit():
        raise AssertionError(f"submission did not return a numeric job ID: {output}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {
        "registered_at": timestamp,
        "reason": "paper renderer used `absent` while the frozen no-hint cue key is `none`",
        "scope": "post-run validation and table rendering only",
        "scientific_configuration": "identical",
        "old_renderer_sha256": OLD_RENDERER_SHA256,
        "renderer": str(renderer.relative_to(REPO)),
        "renderer_sha256": renderer_hash,
        "finalizer_script": str(finalizer_script.relative_to(REPO)),
        "finalizer_script_sha256": sha256(finalizer_script),
        "coin_ledger": str(ledger_path.relative_to(REPO)),
        "coin_ledger_sha256": sha256(ledger_path),
        "old_finalizer": args.old_finalizer,
        "replacement_finalizer": new_finalizer,
        "registrar": str(Path(__file__).resolve().relative_to(REPO)),
        "registrar_sha256": sha256(Path(__file__).resolve()),
        "regression_test": "coin_city_structural/tests/test_protocol.py::test_renderer_uses_frozen_no_hint_label",
    }
    revised = copy.deepcopy(handoff)
    revised["parent_handoff"] = str(handoff_path.relative_to(REPO))
    revised["parent_handoff_sha256"] = sha256(handoff_path)
    revised["coin_analysis_repair"] = record
    revised.setdefault("cancelled_superseded_finalizers", []).append(args.old_finalizer)
    revised["replacement_finalizers"]["coin_city"] = new_finalizer
    destination = ROOT / "runs" / f"coin_analysis_repair_{timestamp}.json"
    destination.write_text(json.dumps(revised, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    subprocess.run(["scancel", args.old_finalizer], check=True)
    print(f"handoff -> {destination}")
    print(f"replacement Coin City finalizer -> {new_finalizer}")


if __name__ == "__main__":
    main()
