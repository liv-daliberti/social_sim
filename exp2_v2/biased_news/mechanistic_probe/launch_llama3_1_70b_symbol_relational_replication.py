#!/usr/bin/env python3
"""Preflight and submit the frozen Llama-3.1-70B patching replication."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from symbol_relational_common import file_sha256


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN_DIR = (
    ROOT
    / "exp2_v2"
    / "biased_news"
    / "data"
    / "coin_city_stable_relationship_claude_n250_v4"
    / "mechanistic_probe"
    / "llama3_1_70b_symbol_relational_v1"
)
MODEL_COMMIT = "1605565b47bb9346c5515c34102e054115b4f98b"
MODEL_SNAPSHOT = (
    ROOT
    / ".runtime"
    / "hf_home"
    / "hub"
    / "models--meta-llama--Llama-3.1-70B-Instruct"
    / "snapshots"
    / MODEL_COMMIT
)
SCRIPTS = {
    "extract_select": HERE / "run_llama3_1_70b_symbol_relational_extract.sbatch",
    "patch_k0": HERE / "run_llama3_1_70b_symbol_relational_patch.sbatch",
    "patch_k4": HERE / "run_llama3_1_70b_symbol_relational_patch.sbatch",
    "finalize": HERE / "run_llama3_1_70b_symbol_relational_finalize.sbatch",
}


def submit(command: list[str]) -> str:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout.strip().split(";")[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument(
        "--extract-job",
        help="attach downstream stages to an already-submitted extraction job",
    )
    args = parser.parse_args()

    if not MODEL_SNAPSHOT.is_dir():
        raise FileNotFoundError(f"missing pinned model snapshot: {MODEL_SNAPSHOT}")
    protocol_path = RUN_DIR / "relational_protocol_manifest.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_label_activation_extraction":
        raise ValueError("replication protocol is not frozen")
    replication = protocol.get("replication") or {}
    if replication.get("role") != "cross_family_confirmation":
        raise ValueError("protocol is not the authorized cross-family replication")
    if any((RUN_DIR / name).exists() for name in (
        "label_extraction_manifest.json",
        "activation_patch_generations.jsonl",
        "relational_probe_results.json",
    )):
        raise FileExistsError("replication outputs already exist; refusing a fresh submit")
    for path in SCRIPTS.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    commands: dict[str, list[str]] = {
        "extract_select": [
            "sbatch", "--parsable", "--account=allcs",
            str(SCRIPTS["extract_select"]),
        ],
    }
    jobs: dict[str, str] = {}
    if args.submit:
        jobs["extract_select"] = (
            str(args.extract_job)
            if args.extract_job
            else submit(commands["extract_select"])
        )
        commands["patch_k0"] = [
            "sbatch", "--parsable", "--account=allcs",
            f"--dependency=afterok:{jobs['extract_select']}",
            "--export=ALL,DEPTH=0", str(SCRIPTS["patch_k0"]),
        ]
        jobs["patch_k0"] = submit(commands["patch_k0"])
        commands["patch_k4"] = [
            "sbatch", "--parsable", "--account=allcs",
            f"--dependency=afterok:{jobs['patch_k0']}",
            "--export=ALL,DEPTH=4", str(SCRIPTS["patch_k4"]),
        ]
        jobs["patch_k4"] = submit(commands["patch_k4"])
        commands["finalize"] = [
            "sbatch", "--parsable", "--account=allcs",
            f"--dependency=afterok:{jobs['patch_k4']}",
            str(SCRIPTS["finalize"]),
        ]
        jobs["finalize"] = submit(commands["finalize"])
        receipt = {
            "study": protocol["study"],
            "status": "submitted_after_protocol_freeze",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "protocol_sha256": file_sha256(protocol_path),
            "jobs": jobs,
            "commands": commands,
            "script_sha256": {
                name: file_sha256(path) for name, path in SCRIPTS.items()
            },
            "scheduler_amendment": (
                "allcs account used because the scheduler rejected the otherwise "
                "identical mltheory/A6000 request as unavailable"
            ),
        }
        receipt_path = RUN_DIR / "submission_receipt.json"
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"submitted": args.submit, "jobs": jobs, "commands": commands},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
