#!/usr/bin/env python3
"""Fail-closed launcher for the replicated Exp3 mechanism-probe v2 roster."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
FAMILY = HERE.parent
ROOT = FAMILY.parents[1]
REPORTS = FAMILY / "reports"
SCRIPT = HERE / "run_probe_v2.sbatch"
TASK_DIR = HERE / "data"
AUXILIARY = TASK_DIR / "auxiliary_targets_v2.jsonl"
RUN_ROOT = HERE / "runs"
HF_CACHE = ROOT / ".runtime" / "hf_home" / "hub"
EXPECTED_TASKS_SHA256 = "5c462b8e78c9e28a3e648f282cc3089d2a1db1e62ffd9f03fd295db7a7cfb5b6"
EXPECTED_TASK_MANIFEST_SHA256 = "364072b8d87cacd283ba776542086471a684ad1d3c6709a789c987370a053eef"


MODELS = {
    "qwen3_4b": {
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "model_label": "Qwen3-4B-Instruct-2507",
        "commit": "cdbee75f17c01a7cc42f958dc650907174af0554",
        "batch_size": 2,
        "jobs": {
            "disclosed_causal_family_s42": "30639718",
            "disclosed_causal_family_s43": "30639719",
            "disclosed_causal_family_s44": "30639720",
            "disclosed_population_prior_s42": "30639721",
            "disclosed_population_prior_s43": "30639722",
            "disclosed_population_prior_s44": "30639723",
            "undisclosed_causal_family_s42": "30639736",
            "undisclosed_causal_family_s43": "30639737",
            "undisclosed_causal_family_s44": "30639738",
            "undisclosed_population_prior_s42": "30639739",
            "undisclosed_population_prior_s43": "30639740",
            "undisclosed_population_prior_s44": "30639741",
        },
    },
    "qwen3_8b": {
        "model": "Qwen/Qwen3-8B",
        "model_label": "Qwen3-8B",
        "commit": "b968826d9c46dd6066d109eabc6255188de91218",
        "batch_size": 1,
        "jobs": {
            "disclosed_causal_family_s42": "30788373",
            "disclosed_causal_family_s43": "30730349",
            "disclosed_causal_family_s44": "30730350",
            "disclosed_population_prior_s42": "30730351",
            "disclosed_population_prior_s43": "30730352",
            "disclosed_population_prior_s44": "30730353",
            "undisclosed_causal_family_s42": "30730360",
            "undisclosed_causal_family_s43": "30730361",
            "undisclosed_causal_family_s44": "30730362",
            "undisclosed_population_prior_s42": "30730363",
            "undisclosed_population_prior_s43": "30730364",
            "undisclosed_population_prior_s44": "30730365",
        },
    },
    "llama3_1_8b": {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "model_label": "Llama-3.1-8B-Instruct",
        "commit": "0e9e39f249a16976918f6564b8830bc894c89659",
        "batch_size": 1,
        "jobs": {
            "disclosed_causal_family_s42": "30730354",
            "disclosed_causal_family_s43": "30730355",
            "disclosed_causal_family_s44": "30730356",
            "disclosed_population_prior_s42": "30730357",
            "disclosed_population_prior_s43": "30730358",
            "disclosed_population_prior_s44": "30730359",
            "undisclosed_causal_family_s42": "30730366",
            "undisclosed_causal_family_s43": "30730367",
            "undisclosed_causal_family_s44": "30730368",
            "undisclosed_population_prior_s42": "30730369",
            "undisclosed_population_prior_s43": "30730370",
            "undisclosed_population_prior_s44": "30730371",
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


def endpoint_parts(endpoint: str) -> tuple[str, str, int]:
    match = re.fullmatch(
        r"(disclosed|undisclosed)_(causal_family|population_prior)_s(42|43|44)",
        endpoint,
    )
    if match is None:
        raise ValueError(f"bad registered endpoint: {endpoint}")
    return match.group(1), match.group(2), int(match.group(3))


def resolve_adapter(model_key: str, endpoint: str, job_id: str) -> Path:
    disclosure, arm, seed = endpoint_parts(endpoint)
    pattern = f"{disclosure}_{arm}_{model_key}_s{seed}_*_j{job_id}"
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


def preflight() -> dict[str, str]:
    paths = {
        "tasks": TASK_DIR / "tasks.jsonl",
        "task_manifest": TASK_DIR / "task_manifest.json",
        "auxiliary_targets": AUXILIARY,
        "launcher": Path(__file__),
        "runner": SCRIPT,
        "extractor": HERE / "extract_hidden_states.py",
        "analysis_v2": HERE / "analyze_probe_v2.py",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if file_sha256(paths["tasks"]) != EXPECTED_TASKS_SHA256:
        raise ValueError("unexpected exp3 probe task hash")
    if file_sha256(paths["task_manifest"]) != EXPECTED_TASK_MANIFEST_SHA256:
        raise ValueError("unexpected exp3 probe task-manifest hash")
    auxiliary_rows = sum(1 for line in AUXILIARY.read_text().splitlines() if line.strip())
    if auxiliary_rows != 1_440:
        raise ValueError(f"auxiliary sidecar has {auxiliary_rows} rows")
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in paths.values()}


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
        "BATCH_SIZE": item["batch_size"],
    }
    if adapter is not None:
        exports["ADAPTER"] = str(adapter)
    export_arg = "ALL," + ",".join(f"{key}={value}" for key, value in exports.items())
    return ["sbatch", "--parsable", f"--export={export_arg}", str(SCRIPT)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODELS), default=list(MODELS)
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, choices=(42, 43, 44), default=[42, 43, 44]
    )
    parser.add_argument("--no-base", action="store_true")
    parser.add_argument("--include-complete", action="store_true")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    code_hashes = preflight()
    records = []
    for model_key in args.models:
        item = MODELS[model_key]
        snapshot = model_snapshot(item["model"], item["commit"])
        endpoints: list[tuple[str, Path | None]] = []
        if not args.no_base:
            endpoints.append(("base", None))
        for endpoint, job_id in item["jobs"].items():
            _, _, seed = endpoint_parts(endpoint)
            if seed in args.seeds:
                endpoints.append((endpoint, resolve_adapter(model_key, endpoint, job_id)))
        for endpoint, adapter in endpoints:
            run_dir = RUN_ROOT / f"{model_key}_{endpoint}"
            complete = (run_dir / "probe_results_v2.json").is_file()
            record = {
                "model_key": model_key,
                "model": item["model"],
                "model_commit": item["commit"],
                "model_snapshot": str(snapshot),
                "endpoint": endpoint,
                "adapter": None if adapter is None else str(adapter),
                "run_dir": str(run_dir),
                "already_complete": complete,
                "submitted": False,
            }
            if adapter is not None:
                record["adapter_config_sha256"] = file_sha256(
                    adapter / "adapter_config.json"
                )
                record["adapter_weights_sha256"] = file_sha256(
                    adapter / "adapter_model.safetensors"
                )
            if complete and not args.include_complete:
                record["action"] = "skip_complete"
            else:
                command = submission_command(model_key, endpoint, adapter)
                record["command"] = command
                record["action"] = "submit" if args.submit else "render"
                if args.submit:
                    completed = subprocess.run(
                        command, check=True, capture_output=True, text=True
                    )
                    record["job_id"] = completed.stdout.strip().split(";")[0]
                    record["submitted"] = True
            records.append(record)

    expected = sum(
        (0 if args.no_base else 1) + 4 * len(args.seeds) for _ in args.models
    )
    if len(records) != expected:
        raise AssertionError(f"roster drift: expected {expected}, found {len(records)}")
    ledger = {
        "protocol": "exp3_mechanism_probe_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": args.models,
        "seeds": args.seeds,
        "include_base": not args.no_base,
        "submitted": args.submit,
        "code_and_data_sha256": code_hashes,
        "jobs": records,
    }
    if args.submit:
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = RUN_ROOT / f"submission_v2_{timestamp}.json"
        path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
        ledger["ledger_path"] = str(path)
    print(json.dumps(ledger, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
