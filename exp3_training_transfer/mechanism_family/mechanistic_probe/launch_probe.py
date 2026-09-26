#!/usr/bin/env python3
"""Fail-closed launcher for the Experiment 3 latent-mechanism probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
ROOT = FAMILY.parents[1]
REPORTS = FAMILY / "reports"
SCRIPT = HERE / "run_probe.sbatch"
TASK_DIR = HERE / "data"
RUN_ROOT = HERE / "runs"
HF_CACHE = ROOT / ".runtime" / "hf_home" / "hub"
EXPECTED_TASKS_SHA256 = "5c462b8e78c9e28a3e648f282cc3089d2a1db1e62ffd9f03fd295db7a7cfb5b6"
EXPECTED_TASK_MANIFEST_SHA256 = "364072b8d87cacd283ba776542086471a684ad1d3c6709a789c987370a053eef"

MODELS = {
    "qwen3_4b": {
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "model_label": "Qwen3-4B-Instruct-2507",
        "commit": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "jobs": {
            "disclosed_causal_family_s42": "30639718",
            "disclosed_population_prior_s42": "30639721",
            "undisclosed_causal_family_s42": "30639736",
            "undisclosed_population_prior_s42": "30639739",
        },
    },
    "qwen3_8b": {
        "model": "Qwen/Qwen3-8B",
        "model_label": "Qwen3-8B",
        "commit": "b968826d9c46dd6066d109eabc6255188de91218",
        "jobs": {
            "disclosed_causal_family_s42": "30788373",
            "disclosed_population_prior_s42": "30730351",
            "undisclosed_causal_family_s42": "30730360",
            "undisclosed_population_prior_s42": "30730363",
        },
    },
    "llama3_1_8b": {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "model_label": "Llama-3.1-8B-Instruct",
        "commit": "0e9e39f249a16976918f6564b8830bc894c89659",
        "jobs": {
            "disclosed_causal_family_s42": "30730354",
            "disclosed_population_prior_s42": "30730357",
            "undisclosed_causal_family_s42": "30730366",
            "undisclosed_population_prior_s42": "30730369",
        },
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_snapshot(model: str, commit: str) -> Path:
    slug = "models--" + model.replace("/", "--")
    path = HF_CACHE / slug / "snapshots" / commit
    if not path.is_dir():
        raise FileNotFoundError(f"missing pinned model snapshot: {path}")
    return path


def resolve_adapter(model_key: str, endpoint: str, job_id: str) -> Path:
    disclosure = next(
        (value for value in ("disclosed", "undisclosed") if endpoint.startswith(value + "_")),
        None,
    )
    if disclosure is None or not endpoint.endswith("_s42"):
        raise ValueError(f"bad registered endpoint: {endpoint}")
    arm = endpoint[len(disclosure) + 1 : -len("_s42")]
    if arm not in {"causal_family", "population_prior"}:
        raise ValueError(f"bad registered arm: {arm}")
    pattern = f"{disclosure}_{arm}_{model_key}_s42_*_j{job_id}"
    reports = sorted(path for path in REPORTS.glob(pattern) if path.is_dir())
    if len(reports) != 1:
        raise RuntimeError(f"expected one report for {pattern}, found {len(reports)}")
    adapters = sorted(reports[0].glob("**/saved_models/step_00301"))
    if len(adapters) != 1:
        raise RuntimeError(
            f"expected one final step_00301 under {reports[0]}, found {len(adapters)}"
        )
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapters[0] / filename).is_file():
            raise FileNotFoundError(adapters[0] / filename)
    config = json.loads((adapters[0] / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("base_model_name_or_path") != MODELS[model_key]["model"]:
        raise ValueError(f"wrong adapter base model for {endpoint}")
    return adapters[0]


def preflight() -> None:
    if file_sha256(TASK_DIR / "tasks.jsonl") != EXPECTED_TASKS_SHA256:
        raise ValueError("unexpected exp3 probe task hash")
    if file_sha256(TASK_DIR / "task_manifest.json") != EXPECTED_TASK_MANIFEST_SHA256:
        raise ValueError("unexpected exp3 probe task-manifest hash")
    if not SCRIPT.is_file():
        raise FileNotFoundError(SCRIPT)


def submission_command(
    model_key: str,
    endpoint: str,
    adapter: Path | None,
) -> list[str]:
    item = MODELS[model_key]
    exports: dict[str, Any] = {
        "MODEL": item["model"],
        "MODEL_COMMIT": item["commit"],
        "MODEL_LABEL": item["model_label"],
        "ENDPOINT_LABEL": endpoint,
        "RUN_DIR": str(RUN_ROOT / f"{model_key}_{endpoint}"),
    }
    if adapter is not None:
        exports["ADAPTER"] = str(adapter)
    export_arg = "ALL," + ",".join(f"{key}={value}" for key, value in exports.items())
    return ["sbatch", "--parsable", f"--export={export_arg}", str(SCRIPT)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODELS),
        default=["qwen3_8b"],
        help="Qwen3-8B is the pilot; 4B and Llama are replications.",
    )
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    preflight()
    records = []
    for model_key in args.models:
        item = MODELS[model_key]
        snapshot = model_snapshot(item["model"], item["commit"])
        endpoints: list[tuple[str, Path | None]] = [("base", None)]
        endpoints.extend(
            (endpoint, resolve_adapter(model_key, endpoint, job_id))
            for endpoint, job_id in item["jobs"].items()
        )
        for endpoint, adapter in endpoints:
            cmd = submission_command(model_key, endpoint, adapter)
            record = {
                "model_key": model_key,
                "model": item["model"],
                "model_commit": item["commit"],
                "model_snapshot": str(snapshot),
                "endpoint": endpoint,
                "adapter": None if adapter is None else str(adapter),
                "command": cmd,
                "submitted": args.submit,
            }
            if adapter is not None:
                record["adapter_config_sha256"] = file_sha256(
                    adapter / "adapter_config.json"
                )
                record["adapter_weights_sha256"] = file_sha256(
                    adapter / "adapter_model.safetensors"
                )
            if args.submit:
                completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
                record["job_id"] = completed.stdout.strip().split(";")[0]
            records.append(record)

    ledger = {
        "protocol": "exp3_mechanism_probe_v1",
        "tasks_sha256": EXPECTED_TASKS_SHA256,
        "task_manifest_sha256": EXPECTED_TASK_MANIFEST_SHA256,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "submitted": args.submit,
        "jobs": records,
    }
    if args.submit:
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = RUN_ROOT / f"submission_{timestamp}.json"
        path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        ledger["ledger_path"] = str(path)
    print(json.dumps(ledger, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
