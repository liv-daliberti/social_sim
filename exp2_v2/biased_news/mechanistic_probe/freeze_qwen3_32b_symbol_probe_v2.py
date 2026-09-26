#!/usr/bin/env python3
"""Freeze the Qwen3-32B arbitrary-symbol probe before extraction."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE_DIR = (
    ROOT
    / "exp2_v2"
    / "biased_news"
    / "data"
    / "coin_city_stable_relationship_claude_n250_v4"
    / "mechanistic_probe"
    / "qwen3_14b_symbol_probe_v2"
)
RUN_DIR = SOURCE_DIR.with_name("qwen3_32b_symbol_probe_v2")
TASKS_SHA256 = "ee6735e9c2ec7b1c9bdae76fd5f52ce19bd6cfec3f96eee1808b52d9832e5a01"
TASK_MANIFEST_SHA256 = "eef19c61692942af998c9cc38b4a9f148424accb291dc6d01b6e03da8a515b56"
MODEL_ID = "Qwen/Qwen3-32B"
MODEL_COMMIT = "9216db5781bf21249d130ec9da846c4624c16137"
STUDY = "qwen3_32b_symbol_probe_v2"
CODE_FILES = (
    "extract_qwen_hidden_states.py",
    "analyze_probe.py",
    "run_qwen3_32b_symbol_probe_v2.sbatch",
    Path(__file__).name,
)
FORBIDDEN_OUTPUTS = (
    "hidden_states.npz",
    "hidden_states.npz.partial.npz",
    "behavior_test.jsonl",
    "behavior_test.jsonl.partial",
    "extraction_manifest.json",
    "probe_results.json",
    "layerwise_results.csv",
    "selected_predictions.jsonl",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_exact(name: str, expected: str) -> None:
    source = SOURCE_DIR / name
    destination = RUN_DIR / name
    if file_sha256(source) != expected:
        raise ValueError(f"source {name} hash drifted")
    if destination.exists():
        if file_sha256(destination) != expected:
            raise FileExistsError(f"destination {name} differs")
        return
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)
    if file_sha256(destination) != expected:
        raise RuntimeError(f"copied {name} hash drifted")


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    existing = [name for name in FORBIDDEN_OUTPUTS if (RUN_DIR / name).exists()]
    if existing:
        raise FileExistsError(
            f"cannot freeze after probe output exists: {', '.join(existing)}"
        )
    copy_exact("tasks.jsonl", TASKS_SHA256)
    copy_exact("task_manifest.json", TASK_MANIFEST_SHA256)
    task_manifest = json.loads((RUN_DIR / "task_manifest.json").read_text())
    protocol = {
        "study": STUDY,
        "status": "frozen_before_hidden_state_extraction",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "model": {"id": MODEL_ID, "commit": MODEL_COMMIT},
        "source_task_run": SOURCE_DIR.name,
        "tasks_sha256": TASKS_SHA256,
        "task_manifest_sha256": TASK_MANIFEST_SHA256,
        "task_design": {
            "episode_counts": task_manifest["episode_counts"],
            "record_count": task_manifest["record_count"],
            "development_folds": 4,
            "sealed_test_reused_without_modification": True,
        },
        "analysis": {
            "targets": ["regime", "slope", "residual_slope"],
            "ridge_alphas": [0.0001, 0.003, 0.1, 3.0, 100.0],
            "selection": "four-fold development CV; sealed test unread",
            "permutation_repeats": 10000,
            "permutation_seed": 20260825,
            "required_reporting": "all frozen arms, depths, and targets",
        },
        "next_stage": (
            "freeze and run the identical bidirectional City C label-state "
            "causal patch regardless of the sealed probe result"
        ),
        "code_sha256": {name: file_sha256(HERE / name) for name in CODE_FILES},
    }
    content = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    path = RUN_DIR / "probe_protocol_manifest.json"
    if path.exists() and path.read_text() != content:
        raise FileExistsError("probe protocol already exists with different content")
    path.write_text(content)
    print(json.dumps({"run_dir": str(RUN_DIR), "protocol_sha256": file_sha256(path)}))


if __name__ == "__main__":
    main()
