#!/usr/bin/env python3
"""Registered launcher for the paired disclosed/undisclosed C3 factorial.

Confirmatory: 2 disclosures x 3 models x 2 arms x 3 seeds = 36.
Diagnostic: 2 disclosures x Qwen3-4B x structureless x 3 seeds = 6.
Base: 2 disclosures x 3 models = 6 evaluation-only jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
TRAIN_SCRIPT = ROOT / "mechanism_rl.sh"
BASE_SCRIPT = ROOT / "base_eval.sh"
TRAINER = ROOT / "run_mechanism_rl.py"
PREFLIGHT = ROOT / "preflight.py"
CANARY_AUDIT = ROOT / "audit_canaries.py"
MANIFEST = ROOT / "protocol" / "c3_mechanism_manifest.json"
RUNS = ROOT / "runs"
HF_HUB = REPO / ".runtime" / "hf_home" / "hub"

DISCLOSURES = ("disclosed", "undisclosed")
CONFIRMATORY_ARMS = ("causal_family", "population_prior")
SEEDS = (42, 43, 44)
MODELS = {
    "qwen3_4b": {"id": "Qwen/Qwen3-4B-Instruct-2507", "template": "biased_news"},
    "qwen3_8b": {"id": "Qwen/Qwen3-8B", "template": "auto_no_think"},
    "llama3_1_8b": {"id": "meta-llama/Llama-3.1-8B-Instruct", "template": "auto"},
}
PROTOCOL = "c3_mechanism_disclosure_v4"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_path(model_id: str) -> Path:
    return HF_HUB / ("models--" + model_id.replace("/", "--"))


def git_revision() -> str | None:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          text=True, capture_output=True, check=False)
    return proc.stdout.strip() or None


def run_preflight(require_full: bool) -> None:
    build = json.loads((ROOT / "data" / "build_manifest.json").read_text())
    if require_full and (build["n_train_per_world"], build["n_eval_per_world"]) != (600, 60):
        raise SystemExit("full launch requires registered 600/world train and 60/world eval data")
    oracle_rows = min(3, int(build["n_eval_per_world"]))
    proc = subprocess.run(
        [sys.executable, str(PREFLIGHT), "--oracle-per-world", str(oracle_rows),
         "--write-manifest", str(MANIFEST)], cwd=ROOT,
        text=True, capture_output=True, check=False,
    )
    if proc.returncode:
        print(proc.stdout, end="")
        print(proc.stderr, end="", file=sys.stderr)
        raise SystemExit("C3 preflight failed; no jobs submitted")
    print(proc.stdout.strip())


def sbatch_command(script: Path, env: dict[str, str], *, time: str, gpus: int,
                   memory: str, partition: str, exclude: str | None) -> list[str]:
    export = "ALL," + ",".join(f"{key}={value}" for key, value in env.items())
    command = ["sbatch", "--parsable", f"--partition={partition}",
               f"--gres=gpu:a6000:{gpus}", "--cpus-per-task=8",
               f"--mem={memory}", f"--time={time}"]
    if exclude:
        command.append(f"--exclude={exclude}")
    command.extend([f"--export={export}", str(script)])
    return command


def submit(command: list[str], dry_run: bool) -> str | None:
    print(shlex.join(command))
    if dry_run:
        return None
    job_id = subprocess.check_output(command, cwd=REPO, text=True).strip().split(";")[0]
    print(f"submitted job {job_id}")
    return job_id


def training_specs(model_keys: list[str], disclosures: list[str], seeds: list[int],
                   full: bool, diagnostics: bool) -> list[tuple[str, str, str, int]]:
    arms = CONFIRMATORY_ARMS if full else ("causal_family",)
    specs = [(disclosure, model, arm, seed)
             for disclosure in disclosures for model in model_keys
             for arm in arms for seed in seeds]
    if full and diagnostics and "qwen3_4b" in model_keys:
        specs.extend((disclosure, "qwen3_4b", "structureless", seed)
                     for disclosure in disclosures for seed in seeds)
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--canary", action="store_true",
                       help="six causal canaries: both disclosures x three models, seed 42")
    group.add_argument("--full", action="store_true",
                       help="36 confirmatory + 6 Qwen3-4B structureless jobs")
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument("--disclosures", nargs="+", choices=DISCLOSURES,
                        default=list(DISCLOSURES))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--no-diagnostics", action="store_true")
    parser.add_argument("--no-base-evals", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-missing-model", action="store_true",
                        help="dry-run only: render commands when an offline model is absent")
    parser.add_argument("--partition", default="all")
    parser.add_argument("--exclude", default="node206")
    parser.add_argument("--collocate", action="store_true",
                        help="one GPU/job; default separates actor and learner for reliability")
    parser.add_argument("--time")
    args = parser.parse_args()

    seeds = args.seeds or ([42] if args.canary else list(SEEDS))
    if args.full and sorted(set(seeds)) != list(SEEDS):
        parser.error("the registered full protocol requires exactly seeds 42 43 44")
    missing = [MODELS[key]["id"] for key in args.models
               if not cache_path(MODELS[key]["id"]).is_dir()]
    if missing and not (args.dry_run and args.allow_missing_model):
        raise SystemExit("offline model cache missing: " + ", ".join(missing))

    if args.full and not args.dry_run:
        gate = subprocess.run([sys.executable, str(CANARY_AUDIT)], cwd=ROOT,
                              text=True, capture_output=True, check=False)
        if gate.returncode:
            print(gate.stdout, end="")
            print(gate.stderr, end="", file=sys.stderr)
            raise SystemExit("synthetic canary gate failed; full grid remains locked")
        print(gate.stdout.strip())
    run_preflight(require_full=args.full and not args.dry_run)
    specs = training_specs(args.models, args.disclosures, seeds, args.full,
                           diagnostics=not args.no_diagnostics)
    gpus = 1 if args.collocate else 2
    max_train = "160" if args.canary else "4800"
    walltime = args.time or ("03:00:00" if args.canary else "20:00:00")
    submissions = []
    for disclosure, model_key, arm, seed in specs:
        model = MODELS[model_key]
        env = {
            "DISCLOSURE": disclosure, "ARM": arm, "MODEL_KEY": model_key,
            "MODEL": model["id"], "PROMPT_TEMPLATE": model["template"],
            "SEED": str(seed), "MAX_TRAIN": max_train,
            "EVAL_STEPS": "10" if args.canary else "25",
            "COLLOCATE": "1" if args.collocate else "0", "GPUS": str(gpus),
            "BATCH": "16", "ROLLOUT_PER_PROMPT": "8",
            "LORA_RANK": "32", "LORA_ALPHA": "64", "VLLM_RATIO": "0.38",
            "MAX_MODEL_LEN": "3072", "PROMPT_MAX_LENGTH": "2304",
            "GEN_LEN": "192", "LR": "0.000001", "TEMP": "1.3",
            "RESPONSE_W": "0.60", "EVAL_BATCH": "120", "ZERO_STAGE": "2",
            "STOCHASTIC_N": "5", "STOCHASTIC_TEMP": "0.7",
            "STRUCTURED_OUTPUT": "forecast_array",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
        command = sbatch_command(TRAIN_SCRIPT, env, time=walltime, gpus=gpus,
                                 memory="100G", partition=args.partition,
                                 exclude=args.exclude)
        job_id = submit(command, args.dry_run)
        submissions.append({"kind": "training", "disclosure": disclosure,
                            "model": model_key, "arm": arm, "seed": seed,
                            "job_id": job_id, "command": command, "environment": env})

    if not args.no_base_evals:
        for disclosure in args.disclosures:
            for model_key in args.models:
                model = MODELS[model_key]
                env = {"DISCLOSURE": disclosure, "MODEL_KEY": model_key,
                       "MODEL": model["id"], "PROMPT_TEMPLATE": model["template"],
                       "MAX_TOKENS": "192", "MAX_MODEL_LEN": "3072",
                       "STOCHASTIC_N": "5", "STOCHASTIC_TEMP": "0.7",
                       "STRUCTURED_OUTPUT": "forecast_array"}
                command = sbatch_command(BASE_SCRIPT, env, time="00:30:00", gpus=1,
                                         memory="24G", partition=args.partition,
                                         exclude=args.exclude)
                job_id = submit(command, args.dry_run)
                submissions.append({"kind": "base_evaluation", "disclosure": disclosure,
                                    "model": model_key, "job_id": job_id,
                                    "command": command, "environment": env})

    full_roster = set(args.models) == set(MODELS) and set(args.disclosures) == set(DISCLOSURES)
    expected = (36 + (0 if args.no_diagnostics else 6) +
                (0 if args.no_base_evals else 6)) if args.full and full_roster else None
    print(f"\nrendered {len(submissions)} jobs" +
          (f" (registered total {expected})" if expected is not None else ""))
    if args.dry_run:
        print("dry run: nothing submitted")
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ledger = {
        "protocol": PROTOCOL, "submitted_at": timestamp,
        "structured_decoding": {
            "kind": "gbnf", "backend": "xgrammar",
            "contract": "exactly one forecasts array containing ten JSON numbers",
            "value_constraints": "none",
        },
        "git_revision": git_revision(), "preflight_manifest": str(MANIFEST.relative_to(REPO)),
        "preflight_sha256": sha256(MANIFEST), "trainer_sha256": sha256(TRAINER),
        "train_script_sha256": sha256(TRAIN_SCRIPT), "base_script_sha256": sha256(BASE_SCRIPT),
        "evaluator_sha256": sha256(ROOT / "evaluate_endpoint.py"),
        "report_sha256": sha256(ROOT / "report.py"),
        "world_catalog_sha256": sha256(ROOT / "worlds.py"),
        "prompt_renderer_sha256": sha256(ROOT / "prompt.py"),
        "output_contract_sha256": sha256(ROOT / "output_contract.py"),
        "preflight_code_sha256": sha256(PREFLIGHT),
        "canary_audit_sha256": sha256(CANARY_AUDIT),
        "submissions": submissions,
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"c3_mechanism_{timestamp}.json"
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"ledger -> {path}")


if __name__ == "__main__":
    main()
