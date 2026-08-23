#!/usr/bin/env python3
"""Pin the prespecified Coin City inference extension and replace its pending finalizer."""
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pending_status(job_id: str, expected_name: str) -> str:
    output = subprocess.check_output(
        ["scontrol", "show", "job", job_id, "-o"], text=True
    ).strip()
    if f"JobName={expected_name}" not in output or "JobState=PENDING" not in output:
        raise SystemExit(f"refusing to replace non-pending {expected_name} job {job_id}")
    return output


def scientific_ids(ledger: dict) -> list[str]:
    canaries = ledger.get("canaries", [])
    gate = [ledger.get("canary_gate", {})]
    training = ledger.get("full_training", [])
    bases = ledger.get("base_evaluations", [])
    if tuple(map(len, (canaries, gate, training, bases))) != (6, 1, 21, 3):
        raise AssertionError("Coin City ledger is not the exact 6+1+21+3 DAG")
    ids = [str(row.get("job_id", "")) for row in canaries + gate + training + bases]
    if len(set(ids)) != 31 or any(not value.isdigit() for value in ids):
        raise AssertionError("Coin City DAG must contain 31 distinct numeric job IDs")
    return ids


def assert_preexecution(ids: list[str]) -> None:
    output = subprocess.check_output(
        ["squeue", "-h", "-j", ",".join(ids), "-o", "%i|%T"], text=True
    )
    states = dict(line.split("|", 1) for line in output.splitlines() if "|" in line)
    unexpected = {job_id: states.get(job_id, "NOT_QUEUED") for job_id in ids
                  if states.get(job_id) != "PENDING"}
    if unexpected:
        raise SystemExit(f"analysis extension is allowed only before Coin execution: {unexpected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin-ledger", type=Path, required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--old-finalizer", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.old_finalizer.isdigit():
        raise AssertionError("--old-finalizer must be numeric")
    ledger_path = args.coin_ledger.resolve()
    handoff_path = args.handoff.resolve()
    ledger = load(ledger_path)
    handoff = load(handoff_path)
    ids = scientific_ids(ledger)
    assert_preexecution(ids)
    pending_status(args.old_finalizer, "c3_coin_paper")
    registered_old = str((handoff.get("replacement_finalizers") or {}).get("coin_city", ""))
    if registered_old != args.old_finalizer:
        raise AssertionError("handoff and --old-finalizer disagree")

    renderer = COIN / "make_paper_outputs.py"
    finalizer_script = COIN / "finalize_campaign.sbatch"
    prior_analysis = (handoff.get("coin_analysis_extension")
                      or handoff.get("coin_analysis_repair") or {})
    old_hash = str(prior_analysis.get("renderer_sha256", ""))
    renderer_hash = sha256(renderer)
    if len(old_hash) != 64 or renderer_hash == old_hash:
        raise AssertionError("renderer did not change from the registered predecessor")

    command = [
        "sbatch", "--parsable", "--partition=cs", "--cpus-per-task=4", "--mem=16G",
        "--time=01:00:00", "--dependency=afterany:" + ":".join(ids[7:]),
        "--export=ALL," +
        f"CAMPAIGN_LEDGER={ledger_path},EXPECTED_RENDERER_SHA256={renderer_hash}",
        str(finalizer_script),
    ]
    print(shlex.join(command), flush=True)
    if args.dry_run:
        print("DRY RUN: all 31 Coin DAG jobs are pending; would submit the replacement "
              "finalizer and then cancel only the registered predecessor")
        return

    output = subprocess.check_output(command, cwd=REPO, text=True).strip()
    replacement = output.split(";", 1)[0]
    if not replacement.isdigit():
        raise AssertionError(f"submission did not return a numeric job ID: {output}")
    subprocess.check_call(["scancel", args.old_finalizer])

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {
        "registered_at": timestamp,
        "reason": (
            "render every prespecified transfer-cell and cue contrast with seed-first "
            "paired uncertainty, including the k=8-minus-k=0 evidence-override interaction"
        ),
        "scope": "post-run inference, machine-readable estimates, and table rendering only",
        "scientific_configuration": "identical",
        "preexecution_assertion": "all 31 Coin City DAG jobs were PENDING",
        "old_renderer_sha256": old_hash,
        "renderer": str(renderer.relative_to(REPO)),
        "renderer_sha256": renderer_hash,
        "added_outputs": [
            "absent_minus_correct_interval_by_k",
            "all_transfer_cell_population_prior_minus_causal_intervals",
            "k8_minus_k0_misleading_penalty_interaction",
            "machine_readable_registered_estimates",
        ],
        "variance_unit": "training seed; paired latent episode within seed; draws averaged",
        "finalizer_script": str(finalizer_script.relative_to(REPO)),
        "finalizer_script_sha256": sha256(finalizer_script),
        "coin_ledger": str(ledger_path.relative_to(REPO)),
        "coin_ledger_sha256": sha256(ledger_path),
        "old_finalizer": args.old_finalizer,
        "replacement_finalizer": replacement,
        "cancellation_command": ["scancel", args.old_finalizer],
        "registrar": str(Path(__file__).resolve().relative_to(REPO)),
        "registrar_sha256": sha256(Path(__file__).resolve()),
        "regression_tests": [
            "coin_city_structural/tests/test_protocol.py::test_renderer_uses_frozen_no_hint_label",
            "coin_city_structural/tests/test_protocol.py::test_all_registered_estimands_render_with_seed_first_intervals",
        ],
    }
    revised = copy.deepcopy(handoff)
    revised["parent_handoff"] = str(handoff_path.relative_to(REPO))
    revised["parent_handoff_sha256"] = sha256(handoff_path)
    revised["coin_analysis_extension"] = record
    revised.setdefault("cancelled_superseded_finalizers", []).append(args.old_finalizer)
    revised["replacement_finalizers"]["coin_city"] = replacement
    destination = ROOT / "runs" / f"coin_analysis_extension_{timestamp}.json"
    destination.write_text(json.dumps(revised, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"handoff -> {destination}")
    print(f"replacement Coin City finalizer -> {replacement}")


if __name__ == "__main__":
    main()
