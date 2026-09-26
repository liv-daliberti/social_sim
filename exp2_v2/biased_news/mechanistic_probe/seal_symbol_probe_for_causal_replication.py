#!/usr/bin/env python3
"""Seal a completed arbitrary-symbol probe as a causal-replication source."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-commit", required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    path = run_dir / "causal_source_receipt.json"
    if path.exists():
        raise FileExistsError(f"receipt already exists: {path}")
    extraction_path = run_dir / "extraction_manifest.json"
    results_path = run_dir / "probe_results.json"
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if extraction.get("status") != "complete":
        raise ValueError("source extraction is incomplete")
    if results.get("status") != "probe_complete":
        raise ValueError("source probe analysis is incomplete")
    if extraction.get("model") != args.model_id:
        raise ValueError("source model id drifted")
    if extraction.get("model_commit") != args.model_commit:
        raise ValueError("source model commit drifted")
    if extraction.get("study") != run_dir.name or results.get("study") != run_dir.name:
        raise ValueError("source study name drifted")
    receipt = {
        "study": run_dir.name,
        "status": "sealed_for_relational_replication",
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "model": {"id": args.model_id, "commit": args.model_commit},
        "artifacts": {
            "tasks.jsonl": file_sha256(run_dir / "tasks.jsonl"),
            "task_manifest.json": file_sha256(run_dir / "task_manifest.json"),
            "hidden_states.npz": extraction["features_sha256"],
            "extraction_manifest.json": file_sha256(extraction_path),
            "probe_results.json": file_sha256(results_path),
        },
        "scope": (
            "hash seal created after the frozen probe completed and before any "
            "checkpoint-specific label-token activation or causal patch output"
        ),
    }
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"receipt": str(path), "sha256": file_sha256(path)}))


if __name__ == "__main__":
    main()
