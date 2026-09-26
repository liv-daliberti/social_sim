#!/usr/bin/env python3
"""Register and submit the ten-checkpoint Coin City population robustness run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
RUN = ROOT / "data" / "coin_city_population_075_040_sd010_v1"
DESIGN = RUN / "design"
LOGS = RUN / "logs"
RESPONSES = RUN / "responses"
RUNS = RUN / "runs"
SBATCH = HERE / "run_coin_city_robustness_open_model.sbatch"
PROTOCOL = ROOT / "ROBUSTNESS_EXTENSION_PROTOCOL.md"
PROTOCOL_VERSION = "coin_city_robustness_v1"

MODELS = (
    dict(
        slug="qwen3_4b",
        model="Qwen/Qwen3-4B-Instruct-2507",
        label="Qwen3-4B-Instruct-2507",
        gres="gpu:a6000:1",
        cpus=8,
        mem="48G",
        time="00:45:00",
        tp=1,
        template="qwen-no-thinking",
        license="Apache-2.0_Qwen",
    ),
    dict(
        slug="qwen3_8b",
        model="Qwen/Qwen3-8B",
        label="Qwen3-8B",
        gres="gpu:a6000:1",
        cpus=8,
        mem="48G",
        time="00:45:00",
        tp=1,
        template="qwen-no-thinking",
        license="Apache-2.0_Qwen",
    ),
    dict(
        slug="qwen3_14b",
        model="Qwen/Qwen3-14B",
        label="Qwen3-14B",
        gres="gpu:a6000:1",
        cpus=8,
        mem="56G",
        time="01:00:00",
        tp=1,
        template="qwen-no-thinking",
        license="Apache-2.0_Qwen",
    ),
    dict(
        slug="qwen3_32b",
        model="Qwen/Qwen3-32B",
        label="Qwen3-32B",
        gres="gpu:a5000:4",
        cpus=16,
        mem="128G",
        time="01:30:00",
        tp=4,
        template="qwen-no-thinking",
        license="Apache-2.0_Qwen",
    ),
    dict(
        slug="qwen2_5_7b",
        model="Qwen/Qwen2.5-7B-Instruct",
        label="Qwen2.5-7B-Instruct",
        gres="gpu:a5000:1",
        cpus=8,
        mem="48G",
        time="01:00:00",
        tp=1,
        template="qwen-no-thinking",
        license="Apache-2.0_Qwen",
    ),
    dict(
        slug="qwen2_5_14b",
        model="Qwen/Qwen2.5-14B-Instruct",
        label="Qwen2.5-14B-Instruct",
        gres="gpu:a5000:2",
        cpus=8,
        mem="64G",
        time="01:00:00",
        tp=2,
        template="qwen-no-thinking",
        license="Apache-2.0_Qwen",
    ),
    dict(
        slug="qwen2_5_32b",
        model="Qwen/Qwen2.5-32B-Instruct",
        label="Qwen2.5-32B-Instruct",
        gres="gpu:a5000:4",
        cpus=16,
        mem="128G",
        time="01:30:00",
        tp=4,
        template="qwen-no-thinking",
        license="Apache-2.0_Qwen",
    ),
    dict(
        slug="qwen2_5_72b",
        model="Qwen/Qwen2.5-72B-Instruct",
        label="Qwen2.5-72B-Instruct",
        gres="gpu:a5000:8",
        cpus=16,
        mem="192G",
        time="03:00:00",
        tp=8,
        template="qwen-no-thinking",
        license="Apache-2.0_Qwen",
    ),
    dict(
        slug="llama3_1_8b",
        model="meta-llama/Llama-3.1-8B-Instruct",
        label="Llama-3.1-8B-Instruct",
        gres="gpu:a5000:1",
        cpus=8,
        mem="48G",
        time="01:00:00",
        tp=1,
        template="auto",
        license="Llama_3.1_Community_License",
    ),
    dict(
        slug="llama3_1_70b",
        model="meta-llama/Llama-3.1-70B-Instruct",
        label="Llama-3.1-70B-Instruct",
        gres="gpu:a5000:8",
        cpus=16,
        mem="192G",
        time="03:00:00",
        tp=8,
        template="auto",
        license="Llama_3.1_Community_License",
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(spec: dict) -> list[str]:
    exports = ",".join(
        (
            "ALL",
            f"MODEL={spec['model']}",
            f"MODEL_LABEL={spec['label']}",
            f"OUTDIR={RESPONSES / spec['slug']}",
            f"TENSOR_PARALLEL_SIZE={spec['tp']}",
            f"CHAT_TEMPLATE_MODE={spec['template']}",
            f"MODEL_LICENSE_FAMILY={spec['license']}",
        )
    )
    return [
        "sbatch",
        "--parsable",
        "--account=mltheory",
        "--partition=all",
        f"--job-name=cc_r_{spec['slug']}",
        f"--gres={spec['gres']}",
        f"--cpus-per-task={spec['cpus']}",
        f"--mem={spec['mem']}",
        f"--time={spec['time']}",
        "--exclude=node206",
        f"--export={exports}",
        str(SBATCH),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((DESIGN / "manifest.json").read_text())
    if (
        manifest.get("protocol_version") != PROTOCOL_VERSION
        or manifest.get("status") != "frozen_no_model_calls"
    ):
        raise SystemExit("robustness design is not frozen")
    if RESPONSES.exists() and any(RESPONSES.rglob("*.jsonl")):
        raise SystemExit("responses already exist; refusing a second launch")
    LOGS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    registration = {
        "protocol_version": PROTOCOL_VERSION,
        "kind": "generator_population_registration",
        "status": "registered",
        "registered_at": timestamp,
        "classification": "post_hoc_robustness",
        "design_manifest": str((DESIGN / "manifest.json").resolve()),
        "design_manifest_sha256": sha256(DESIGN / "manifest.json"),
        "protocol_sha256": sha256(PROTOCOL),
        "runner_sha256": sha256(HERE / "run_coin_city_robustness_open_model.py"),
        "sbatch_sha256": sha256(SBATCH),
        "launcher_sha256": sha256(Path(__file__).resolve()),
        "models": list(MODELS),
        "planned_responses": 37500,
        "scientific_settings": {
            "temperature": 0.0,
            "max_tokens": 512,
            "arms": ["abc_no_context", "abc_context", "abc_symbol_context"],
            "prefixes": [0, 1, 2, 3, 4],
        },
    }
    registration_path = RUNS / f"generator_population_registration_{timestamp}.json"
    registration_path.write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n"
    )
    commands = {spec["slug"]: command(spec) for spec in MODELS}
    if not args.submit:
        print(
            json.dumps(
                {"registration": str(registration_path), "commands": commands}, indent=2
            )
        )
        return
    jobs = {}
    for spec in MODELS:
        completed = subprocess.run(
            command(spec), check=True, capture_output=True, text=True
        )
        jobs[spec["slug"]] = completed.stdout.strip().split(";")[0]
    launch = {
        "protocol_version": PROTOCOL_VERSION,
        "kind": "generator_population_launch",
        "launched_at": datetime.now(timezone.utc).isoformat(),
        "registration": str(registration_path.resolve()),
        "registration_sha256": sha256(registration_path),
        "jobs": jobs,
        "commands": commands,
    }
    launch_path = RUNS / f"generator_population_launch_{timestamp}.json"
    launch_path.write_text(json.dumps(launch, indent=2, sort_keys=True) + "\n")
    print(json.dumps(launch, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
