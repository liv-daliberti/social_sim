#!/usr/bin/env python3
"""Render or submit the complete Exp3/Exp4 seed-45/46 extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EXP3 = REPO / "exp3_training_transfer"
MECH = EXP3 / "mechanism_family"
COIN = EXP3 / "coin_city_structural"
POLY = EXP3 / "polymarket"
RUNS = HERE / "runs"
SEEDS = (45, 46)
ALL_SEEDS = (42, 43, 44, 45, 46)
LARGE_GATE_JOB = "30891304"
QWEN32_GATE_JOB = "30892282"

CURRENT_MODELS = {
    "qwen3_4b": ("Qwen/Qwen3-4B-Instruct-2507", "biased_news", "cdbee75f17c01a7cc42f958dc650907174af0554", "Qwen3-4B-Instruct-2507"),
    "qwen3_8b": ("Qwen/Qwen3-8B", "auto_no_think", "b968826d9c46dd6066d109eabc6255188de91218", "Qwen3-8B"),
    "llama3_1_8b": ("meta-llama/Llama-3.1-8B-Instruct", "auto", "0e9e39f249a16976918f6564b8830bc894c89659", "Llama-3.1-8B-Instruct"),
}
LARGE_MODELS = {
    "qwen3_14b": {
        "model": "Qwen/Qwen3-14B", "template": "auto_no_think",
        "commit": "40c069824f4251a91eefaf281ebe4c544efd3e18", "label": "Qwen3-14B",
        "train": (2, 12, "160G", "36:00:00"), "extract": (1, 12, "80G", "12:00:00", "none"),
    },
    "qwen3_32b": {
        "model": "Qwen/Qwen3-32B", "template": "auto_no_think",
        "commit": "9216db5781bf21249d130ec9da846c4624c16137", "label": "Qwen3-32B",
        "train": (4, 16, "260G", "48:00:00"), "extract": (2, 16, "180G", "18:00:00", "balanced"),
    },
    "llama3_1_70b": {
        "model": "meta-llama/Llama-3.1-70B-Instruct", "template": "auto",
        "commit": "1605565b47bb9346c5515c34102e054115b4f98b", "label": "Llama-3.1-70B-Instruct",
        "train": (8, 24, "460G", "72:00:00"), "extract": (4, 24, "320G", "30:00:00", "balanced"),
    },
}

EXP4_MODELS = {
    "qwen3_4b": {"model": "Qwen/Qwen3-4B-Instruct-2507", "template": "biased_news", "parent": {42: "30572571", 43: "30572572", 44: "30572573"}},
    "qwen3_8b": {"model": "Qwen/Qwen3-8B", "template": "auto_no_think", "parent": {42: "30856247", 43: "30856248", 44: "30856249"}},
    "llama3_1_8b": {"model": "meta-llama/Llama-3.1-8B-Instruct", "template": "auto", "parent": {42: "30856251", 43: "30856252", 44: "30856253"}},
    "qwen3_14b": {"model": "Qwen/Qwen3-14B", "template": "auto_no_think", "parent": {42: "30870543", 43: "30870544", 44: "30870545"}},
    "qwen3_1_7b": {"model": "Qwen/Qwen3-1.7B", "template": "auto_no_think", "parent": {42: "30890065", 43: "30890108", 44: "30890109"}},
    "llama3_2_3b": {"model": "unsloth/Llama-3.2-3B-Instruct", "template": "auto", "parent": {42: "30890112", 43: "30890113", 44: "30890114"}},
    "qwen3_32b": {"model": "Qwen/Qwen3-32B", "template": "auto_no_think", "parent": {}},
}


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def export_arg(environment: dict[str, Any]) -> str:
    return "ALL," + ",".join(f"{key}={value}" for key, value in environment.items())


def gpu_command(
    script: Path, environment: dict[str, Any], *, gpus: int, cpus: int,
    memory: str, walltime: str, dependency: str | None = None,
    partition: str = "all", account: str | None = None,
    qos: str | None = None, gpu_type: str = "a6000",
) -> list[str]:
    command = ["sbatch", "--parsable"]
    if account:
        command.append(f"--account={account}")
    command.append(f"--partition={partition}")
    if qos:
        command.append(f"--qos={qos}")
    command.extend([
        f"--gres=gpu:{gpu_type}:{gpus}", f"--cpus-per-task={cpus}",
        f"--mem={memory}", f"--time={walltime}", "--exclude=node206",
    ])
    if dependency:
        command.append(f"--dependency={dependency}")
    command.extend([f"--export={export_arg(environment)}", str(script)])
    return command


def cpu_command(script: Path, environment: dict[str, Any], dependency: str) -> list[str]:
    return [
        "sbatch", "--parsable", f"--dependency={dependency}",
        f"--export={export_arg(environment)}", str(script),
    ]


def submit(command: list[str], enabled: bool) -> str:
    print(shlex.join(command), flush=True)
    if not enabled:
        return "DRYRUN"
    output = subprocess.check_output(command, cwd=REPO, text=True).strip()
    job_id = output.split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"unexpected sbatch output: {output!r}")
    return job_id


def mechanism_env(model_key: str, disclosure: str, arm: str, seed: int) -> dict[str, str]:
    model, template, _, _ = CURRENT_MODELS[model_key]
    return {
        "DISCLOSURE": disclosure, "ARM": arm, "MODEL_KEY": model_key,
        "MODEL": model, "PROMPT_TEMPLATE": template, "SEED": str(seed),
        "MAX_TRAIN": "4800", "EVAL_STEPS": "25", "COLLOCATE": "0",
        "GPUS": "2", "BATCH": "16", "ROLLOUT_PER_PROMPT": "8",
        "LORA_RANK": "32", "LORA_ALPHA": "64", "VLLM_RATIO": "0.38",
        "MAX_MODEL_LEN": "3072", "PROMPT_MAX_LENGTH": "2304",
        "GEN_LEN": "192", "LR": "0.000001", "TEMP": "1.3",
        "RESPONSE_W": "0.60", "EVAL_BATCH": "120", "ZERO_STAGE": "2",
        "STOCHASTIC_N": "5", "STOCHASTIC_TEMP": "0.7",
        "STRUCTURED_OUTPUT": "forecast_array",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }


def coin_env(model_key: str, arm: str, seed: int, *, scale: bool = False) -> dict[str, str]:
    if scale:
        model, template, ratio = "Qwen/Qwen3-14B", "auto_no_think", "0.78"
    else:
        model, template, _, _ = CURRENT_MODELS[model_key]
        ratio = "0.38"
    prefix = "scale_" if scale else ""
    return {
        "ARM": arm, "BATCH": "16", "EVAL_BATCH": "120", "EVAL_STEPS": "50",
        "GEN_LEN": "192", "GPUS": "2", "LORA_ALPHA": "64", "LORA_RANK": "32",
        "LR": "0.000001", "MAX_MODEL_LEN": "3072", "MAX_TRAIN": "4800",
        "MODEL": model, "MODEL_KEY": model_key, "PROMPT_MAX_LENGTH": "2304",
        "PROMPT_TEMPLATE": template, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "RESPONSE_W": "0.60", "ROLLOUT_PER_PROMPT": "8", "SEED": str(seed),
        "STOCHASTIC_N": "5", "STOCHASTIC_TEMP": "0.7",
        "TAG": f"{prefix}{arm}_{model_key}_s{seed}", "TEMP": "1.3",
        "VLLM_RATIO": ratio, "ZERO_STAGE": "2",
    }


def add_record(records: list[dict[str, Any]], campaign: str, identity: str,
               seed: int, command: list[str], environment: dict[str, Any]) -> None:
    records.append({"campaign": campaign, "identity": identity, "seed": seed,
                    "command": command, "environment": environment})


def training_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for disclosure in ("disclosed", "undisclosed"):
        for model_key in CURRENT_MODELS:
            for arm in ("causal_family", "population_prior"):
                for seed in SEEDS:
                    env = mechanism_env(model_key, disclosure, arm, seed)
                    cmd = gpu_command(MECH / "mechanism_rl.sh", env, gpus=2, cpus=8,
                                      memory="100G", walltime="30:00:00")
                    add_record(records, "exp3_mechanism_current",
                               f"{disclosure}/{model_key}/{arm}", seed, cmd, env)
        for seed in SEEDS:
            env = mechanism_env("qwen3_4b", disclosure, "structureless", seed)
            cmd = gpu_command(MECH / "mechanism_rl.sh", env, gpus=2, cpus=8,
                              memory="100G", walltime="30:00:00")
            add_record(records, "exp3_mechanism_current_diagnostic",
                       f"{disclosure}/qwen3_4b/structureless", seed, cmd, env)

    for model_key, item in LARGE_MODELS.items():
        gpus, cpus, memory, walltime = item["train"]
        for disclosure in ("disclosed", "undisclosed"):
            for arm in ("causal_family", "population_prior"):
                for seed in SEEDS:
                    env = {"DISCLOSURE": disclosure, "ARM": arm, "MODEL_KEY": model_key,
                           "MODEL": item["model"], "PROMPT_TEMPLATE": item["template"],
                           "SEED": str(seed), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
                    cmd = gpu_command(HERE / "large_mechanism_rl_five_seed.sh", env,
                                      gpus=gpus, cpus=cpus, memory=memory, walltime=walltime,
                                      dependency=f"afterok:{LARGE_GATE_JOB}", partition="cs",
                                      account="allcs", qos="medium")
                    add_record(records, "exp3_mechanism_large",
                               f"{disclosure}/{model_key}/{arm}", seed, cmd, env)

    for model_key in CURRENT_MODELS:
        for arm in ("causal", "population_prior"):
            for seed in SEEDS:
                env = coin_env(model_key, arm, seed)
                cmd = gpu_command(COIN / "train.sh", env, gpus=2, cpus=8, memory="100G",
                                  walltime="30:00:00", partition="cs", account="allcs", qos="medium")
                add_record(records, "exp3_coin_current", f"{model_key}/{arm}", seed, cmd, env)
    for seed in SEEDS:
        env = coin_env("qwen3_4b", "structureless", seed)
        cmd = gpu_command(COIN / "train.sh", env, gpus=2, cpus=8, memory="100G",
                          walltime="30:00:00", partition="cs", account="allcs", qos="medium")
        add_record(records, "exp3_coin_current_diagnostic", "qwen3_4b/structureless", seed, cmd, env)
    for arm in ("causal", "population_prior"):
        for seed in SEEDS:
            env = coin_env("qwen3_14b", arm, seed, scale=True)
            cmd = gpu_command(COIN / "train.sh", env, gpus=2, cpus=8, memory="100G",
                              walltime="30:00:00", partition="cs", account="allcs", qos="medium")
            add_record(records, "exp3_coin_qwen14", f"qwen3_14b/{arm}", seed, cmd, env)

    for model_key, item in EXP4_MODELS.items():
        for seed in SEEDS:
            if model_key == "qwen3_4b":
                env = {"DATA": str(POLY / "data/exp3b_registered"), "TAG": "market",
                       "MODEL": item["model"], "VLLM_RATIO": "0.40", "LORA_RANK": "32",
                       "LORA_ALPHA": "64", "BATCH": "16", "ROLLOUT_PER_PROMPT": "8",
                       "MAX_TRAIN": "4800", "TEMP": "1.3", "LR": "0.000001",
                       "EVAL_STEPS": "25", "GEN_LEN": "128", "MAX_MODEL_LEN": "1920",
                       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "SEED": str(seed)}
                cmd = gpu_command(POLY / "scripts/polymarket_rl.sh", env, gpus=1, cpus=8,
                                  memory="90G", walltime="12:00:00")
            elif model_key in ("qwen3_8b", "llama3_1_8b"):
                env = {"MODEL_KEY": model_key, "MODEL": item["model"],
                       "PROMPT_TEMPLATE": item["template"], "SEED": str(seed),
                       "COLLOCATE": "0", "GPUS": "2",
                       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
                cmd = gpu_command(POLY / "scripts/polymarket_rl_model_extension.sh", env,
                                  gpus=2, cpus=8, memory="100G", walltime="30:00:00")
            elif model_key == "qwen3_14b":
                env = {"MODEL_KEY": model_key, "MODEL": item["model"],
                       "PROMPT_TEMPLATE": item["template"], "SEED": str(seed),
                       "RUN_KIND": "train", "MAX_TRAIN": "4800", "EVAL_STEPS": "25",
                       "VLLM_RATIO": "0.78", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
                cmd = gpu_command(POLY / "scripts/polymarket_rl_scale_extension.sh", env,
                                  gpus=2, cpus=8, memory="100G", walltime="30:00:00")
            elif model_key == "qwen3_32b":
                env = {"MODEL_KEY": model_key, "MODEL": item["model"],
                       "PROMPT_TEMPLATE": item["template"], "SEED": str(seed),
                       "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
                cmd = gpu_command(HERE / "polymarket_qwen32_five_seed.sh", env,
                                  gpus=4, cpus=12, memory="220G", walltime="30:00:00",
                                  dependency=f"afterok:{QWEN32_GATE_JOB}", partition="cs",
                                  account="allcs", qos="medium")
            else:
                env = {"MODEL_KEY": model_key, "MODEL": item["model"],
                       "MODEL_CHECKPOINT": item["model"], "PROMPT_TEMPLATE": item["template"],
                       "SEED": str(seed), "RUN_KIND": "train", "MAX_TRAIN": "4800",
                       "EVAL_STEPS": "25", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
                hours = "10:00:00" if model_key == "llama3_2_3b" else "08:00:00"
                cmd = gpu_command(POLY / "scripts/polymarket_rl_architecture_extension.sh", env,
                                  gpus=1, cpus=8, memory="72G", walltime=hours,
                                  account="mltheory")
            add_record(records, f"exp4_{model_key}", model_key, seed, cmd, env)
    return records


def probe_commands(row: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    campaign = row["campaign"]
    disclosure, model_key, arm = row["identity"].split("/")
    seed = int(row["seed"])
    endpoint = f"{disclosure}_{arm}_s{seed}"
    run_dir = MECH / "mechanistic_probe/runs" / f"{model_key}_{endpoint}"
    if campaign == "exp3_mechanism_current":
        model, _, commit, label = CURRENT_MODELS[model_key]
        pattern = f"{disclosure}_{arm}_{model_key}_s{seed}_*_j{row['job_id']}"
        resources = (1, 12, "48G", "04:00:00", "none")
        route = {"partition": "lowprio", "account": "mltheory", "gpu_type": "a5000"}
        batch = "2" if model_key == "qwen3_4b" else "1"
    else:
        item = LARGE_MODELS[model_key]
        model, commit, label = item["model"], item["commit"], item["label"]
        pattern = f"scale_v2_1_train_{disclosure}_{arm}_{model_key}_s{seed}_*_j{row['job_id']}"
        resources = item["extract"]
        route = {"partition": "cs", "account": "allcs", "qos": "medium", "gpu_type": "a6000"}
        batch = "1"
    gpus, cpus, memory, walltime, device_map = resources
    env = {"REPORT_PATTERN": pattern, "MODEL": model, "MODEL_COMMIT": commit,
           "MODEL_LABEL": label, "ENDPOINT_LABEL": endpoint, "RUN_DIR": str(run_dir),
           "DEVICE_MAP": device_map, "BATCH_SIZE": batch}
    command = gpu_command(HERE / "resolve_extract_probe.sbatch", env, gpus=gpus,
                          cpus=cpus, memory=memory, walltime=walltime,
                          dependency=f"afterok:{row['job_id']}", **route)
    return command, {"endpoint": endpoint, "run_dir": str(run_dir), "environment": env}


def evaluation_command(model_key: str, jobs: dict[int, str]) -> list[str]:
    item = EXP4_MODELS[model_key]
    specs = ";".join(f"{seed}:job={jobs[seed]}" for seed in ALL_SEEDS)
    env = {"MODEL_KEY": model_key, "MODEL": item["model"],
           "PROMPT_TEMPLATE": item["template"], "ADAPTER_SPECS": specs,
           "TENSOR_PARALLEL_SIZE": "2" if model_key == "qwen3_32b" else "1"}
    return gpu_command(HERE / "evaluate_exp4_five_seed.sbatch", env,
                       gpus=2 if model_key == "qwen3_32b" else 1,
                       cpus=10 if model_key == "qwen3_32b" else 8,
                       memory="140G" if model_key == "qwen3_32b" else "100G",
                       walltime="12:00:00" if model_key == "qwen3_32b" else "08:00:00",
                       dependency="afterok:" + ":".join(jobs[seed] for seed in ALL_SEEDS))


def preflight(records: list[dict[str, Any]], submitting: bool) -> dict[str, str]:
    if len(records) != 84:
        raise AssertionError(f"expected 84 training jobs, found {len(records)}")
    keys = {(row["campaign"], row["identity"], row["seed"]) for row in records}
    if len(keys) != len(records) or {row["seed"] for row in records} != set(SEEDS):
        raise AssertionError("duplicate or incomplete five-seed roster")
    required = {Path(row["command"][-1]) for row in records}
    required.update({HERE / "PROTOCOL.md", HERE / "launch.py",
                     HERE / "resolve_extract_probe.sbatch", HERE / "analyze_probe.sbatch",
                     HERE / "evaluate_exp4_five_seed.py", HERE / "evaluate_exp4_five_seed.sbatch",
                     HERE / "schedule_qwen32_eval.sbatch"})
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if submitting:
        prior = sorted(RUNS.glob("five_seed_training_*.json"))
        if prior:
            raise SystemExit(f"five-seed submission ledger already exists: {prior}")
    return {str(path.relative_to(REPO)): sha256(path) for path in sorted(required)}


def schedule_qwen32_eval(ledger_path: Path) -> None:
    ledger = json.loads(ledger_path.read_text())
    new = {int(row["seed"]): str(row["job_id"]) for row in ledger["training_jobs"]
           if row["campaign"] == "exp4_qwen3_32b"}
    if set(new) != set(SEEDS):
        raise SystemExit("central ledger lacks both Qwen3-32B added seeds")
    candidates = []
    for path in (POLY / "runs").glob("exp4_qwen32_full_*.json"):
        payload = json.loads(path.read_text())
        jobs = payload.get("training_jobs", [])
        seeds = {int(row["seed"]): str(row["job_id"]) for row in jobs}
        if payload.get("dry_run") is False and set(seeds) == {42, 43, 44}:
            candidates.append((path, seeds))
    if len(candidates) != 1:
        raise SystemExit(f"expected one real Qwen3-32B full ledger, found {candidates}")
    parent_path, parent = candidates[0]
    jobs = {**parent, **new}
    command = evaluation_command("qwen3_32b", jobs)
    job_id = submit(command, True)
    payload = {"protocol": "exp3_exp4_five_seed_extension_v1", "created_at": now(),
               "parent_full_ledger": str(parent_path), "central_ledger": str(ledger_path),
               "adapter_jobs": jobs, "evaluation_job_id": job_id, "command": command}
    atomic_json(RUNS / f"five_seed_qwen32_evaluation_{now()}.json", payload)


def launch(submitting: bool) -> None:
    records = training_records()
    hashes = preflight(records, submitting)
    stamp = now()
    ledger_path = RUNS / f"five_seed_training_{stamp}.json"
    ledger: dict[str, Any] = {
        "protocol": "exp3_exp4_five_seed_extension_v1", "created_at": stamp,
        "submitted": submitting, "status": "submitting" if submitting else "rendered",
        "added_seeds": list(SEEDS), "combined_seed_roster": list(ALL_SEEDS),
        "code_sha256": hashes, "training_jobs": [], "probe_jobs": [],
        "exp4_evaluations": [],
    }
    if submitting:
        atomic_json(ledger_path, ledger)
    for record in records:
        job_id = submit(record["command"], submitting)
        completed = {**record, "job_id": job_id}
        ledger["training_jobs"].append(completed)
        if submitting:
            atomic_json(ledger_path, ledger)

    for row in ledger["training_jobs"]:
        if row["campaign"] not in ("exp3_mechanism_current", "exp3_mechanism_large"):
            continue
        extract_command, metadata = probe_commands(row)
        extract_id = submit(extract_command, submitting)
        analysis_env = {"RUN_DIR": metadata["run_dir"]}
        analysis_command = cpu_command(HERE / "analyze_probe.sbatch", analysis_env,
                                       f"afterok:{extract_id}")
        analysis_id = submit(analysis_command, submitting)
        ledger["probe_jobs"].append({**metadata, "campaign": row["campaign"],
                                     "seed": row["seed"], "training_job_id": row["job_id"],
                                     "extract_job_id": extract_id, "extract_command": extract_command,
                                     "analysis_job_id": analysis_id, "analysis_command": analysis_command})
        if submitting:
            atomic_json(ledger_path, ledger)

    new_exp4 = {(row["campaign"].removeprefix("exp4_"), int(row["seed"])): row["job_id"]
                for row in ledger["training_jobs"] if row["campaign"].startswith("exp4_")}
    for model_key, item in EXP4_MODELS.items():
        if model_key == "qwen3_32b":
            env = {"FIVE_SEED_LEDGER": str(ledger_path)}
            command = cpu_command(HERE / "schedule_qwen32_eval.sbatch", env,
                                  f"afterok:{QWEN32_GATE_JOB}")
            job_id = submit(command, submitting)
            ledger["exp4_evaluations"].append({"model_key": model_key,
                                               "kind": "dynamic_scheduler",
                                               "job_id": job_id, "command": command})
            continue
        jobs = {**item["parent"], **{seed: new_exp4[(model_key, seed)] for seed in SEEDS}}
        command = evaluation_command(model_key, jobs)
        job_id = submit(command, submitting)
        ledger["exp4_evaluations"].append({"model_key": model_key, "kind": "evaluation",
                                           "adapter_jobs": jobs, "job_id": job_id,
                                           "command": command})
        if submitting:
            atomic_json(ledger_path, ledger)

    ledger["counts"] = {"training": len(ledger["training_jobs"]),
                        "probe_extraction": len(ledger["probe_jobs"]),
                        "probe_analysis": len(ledger["probe_jobs"]),
                        "exp4_evaluation_or_scheduler": len(ledger["exp4_evaluations"])}
    ledger["status"] = "submitted" if submitting else "rendered"
    if submitting:
        atomic_json(ledger_path, ledger)
        print(f"ledger -> {ledger_path}")
    else:
        print(json.dumps({"counts": ledger["counts"], "training_roster": 84}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--schedule-qwen32-eval", type=Path)
    args = parser.parse_args()
    if args.schedule_qwen32_eval:
        if args.submit:
            parser.error("--schedule-qwen32-eval is already a real scheduler action")
        schedule_qwen32_eval(args.schedule_qwen32_eval.resolve())
    else:
        launch(args.submit)


if __name__ == "__main__":
    main()
