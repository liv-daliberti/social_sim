#!/usr/bin/env python3
"""Render or submit the frozen cross-architecture Experiment 2 probe extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "exp2_v2" / "biased_news" / "mechanistic_probe" / "run_open_model_probe.sbatch"
RESULT_ROOT = (
    ROOT
    / "exp2_v2"
    / "biased_news"
    / "data"
    / "coin_city_stable_relationship_claude_n250_v4"
    / "mechanistic_probe"
)
TASK_SOURCE = RESULT_ROOT / "qwen3_8b_toy_v1"
HF_CACHE = ROOT / ".runtime" / "hf_home" / "hub"
EXPECTED_TASKS_SHA256 = "2e0f65d6801b96616fb53b764470960a96b4c08c73483006ca626477de30ccb8"
EXPECTED_TASK_MANIFEST_SHA256 = "4b73ac0e9634d144feead8a35668f4ddd85c6c6ac8ab0207ef2d17f01d4c4ecb"

MODELS = {
    "qwen3_4b": {
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "model_commit": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "model_label": "Qwen3-4B-Instruct-2507",
        "study": "qwen3_4b_probe_v1",
        "run_dir": "qwen3_4b_probe_v1",
    },
    "llama3_1_8b": {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "model_commit": "0e9e39f249a16976918f6564b8830bc894c89659",
        "model_label": "Llama-3.1-8B-Instruct",
        "study": "llama3_1_8b_probe_v1",
        "run_dir": "llama3_1_8b_probe_v1",
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight(model_key: str) -> Path:
    if file_sha256(TASK_SOURCE / "tasks.jsonl") != EXPECTED_TASKS_SHA256:
        raise ValueError("unexpected Experiment 2 probe task hash")
    if file_sha256(TASK_SOURCE / "task_manifest.json") != EXPECTED_TASK_MANIFEST_SHA256:
        raise ValueError("unexpected Experiment 2 probe task-manifest hash")
    if not SCRIPT.is_file():
        raise FileNotFoundError(SCRIPT)
    item = MODELS[model_key]
    slug = "models--" + item["model"].replace("/", "--")
    snapshot = HF_CACHE / slug / "snapshots" / item["model_commit"]
    if not snapshot.is_dir():
        raise FileNotFoundError(f"missing pinned model snapshot: {snapshot}")
    return snapshot


def command(model_key: str) -> list[str]:
    item = MODELS[model_key]
    exports = {
        "MODEL": item["model"],
        "MODEL_COMMIT": item["model_commit"],
        "MODEL_LABEL": item["model_label"],
        "STUDY": item["study"],
        "RUN_DIR": str(RESULT_ROOT / item["run_dir"]),
    }
    export_arg = "ALL," + ",".join(f"{key}={value}" for key, value in exports.items())
    return ["sbatch", "--parsable", f"--export={export_arg}", str(SCRIPT)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODELS),
        default=list(MODELS),
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit jobs. Without this flag, only print the exact commands.",
    )
    args = parser.parse_args()

    records = []
    for model_key in args.models:
        snapshot = preflight(model_key)
        cmd = command(model_key)
        record = {
            "model_key": model_key,
            "model_snapshot": str(snapshot),
            "command": cmd,
            "submitted": args.submit,
        }
        if args.submit:
            completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
            record["job_id"] = completed.stdout.strip().split(";")[0]
        records.append(record)
    print(json.dumps(records, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
