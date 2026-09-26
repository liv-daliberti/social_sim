#!/usr/bin/env python3
"""Submit the Coin City structure-selection pilot.

Protocol `coin_city_structure_selection_v1`: Qwen3-8B, two arms, three seeds,
six training jobs.

Wrapper. This uses the parent's `coin_city_structural/train.sh` unmodified --
no copied script, so there is nothing to drift. `DATA` is pointed at this
experiment's dataset and `TAG` carries a `sel_` prefix, so the run directories
land in the parent's `reports/` but cannot be matched by any parent glob, all
of which are anchored at `{arm}_{model}_s{seed}_*`.

Checkpointing flags are passed **explicitly** rather than left to `train.sh`
defaults. The parent protocol was silently split across two wrapper versions
when those defaults changed on 2026-08-28; recording them here means this
protocol's runs can never be ambiguous about which behaviour they used.
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
PARENT = REPO / "exp3_training_transfer/coin_city_structural"
# The jobs run under this interpreter; the preflights must too, or they
# validate a different environment from the one that trains.
OAT_PYTHON = str(REPO / ".runtime/oat_conda/bin/python")
RUNS = ROOT / "runs"

PROTOCOL = "coin_city_structure_selection_v1"
MODEL_KEY = "qwen3_8b"
MODEL = "Qwen/Qwen3-8B"
MODEL_COMMIT = "b968826d9c46dd6066d109eabc6255188de91218"
TEMPLATE = "auto_no_think"
ARMS = ("causal", "population_prior")
SEEDS = (42, 43, 44)
TAG_PREFIX = "sel_"

# Explicit, not inherited from train.sh defaults. See module docstring.
CHECKPOINT_FLAGS = {"SAVE_STEPS": "999999", "SAVE_CKPT": "0"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment(arm: str, seed: int) -> dict[str, str]:
    return {
        "ARM": arm,
        "DATA": str(ROOT / "data" / arm),
        "MODEL": MODEL,
        "MODEL_KEY": MODEL_KEY,
        "PROMPT_TEMPLATE": TEMPLATE,
        "SEED": str(seed),
        "TAG": f"{TAG_PREFIX}{arm}_{MODEL_KEY}_s{seed}",
        "BATCH": "16",
        "EVAL_BATCH": "120",
        "EVAL_STEPS": "50",
        "GEN_LEN": "192",
        "GPUS": "2",
        "LORA_ALPHA": "64",
        "LORA_RANK": "32",
        "LR": "0.000001",
        "MAX_MODEL_LEN": "3072",
        "MAX_TRAIN": "4800",
        "PROMPT_MAX_LENGTH": "2304",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "RESPONSE_W": "0.60",
        "ROLLOUT_PER_PROMPT": "8",
        "STOCHASTIC_N": "5",
        "STOCHASTIC_TEMP": "0.7",
        "TEMP": "1.3",
        "VLLM_RATIO": "0.38",
        "ZERO_STAGE": "2",
        **CHECKPOINT_FLAGS,
    }


def command(env: dict[str, str]) -> list[str]:
    export = "ALL," + ",".join(f"{k}={v}" for k, v in sorted(env.items()))
    return [
        "sbatch", "--parsable", "--account=allcs", "--partition=cs,all",
        "--qos=medium", "--gres=gpu:a6000:2", "--cpus-per-task=8", "--mem=100G",
        "--time=30:00:00", "--exclude=node206", f"--export={export}",
        str(PARENT / "train.sh"),
    ]


def preflight() -> None:
    for arm in ARMS:
        for split in ("train", "heldout"):
            path = ROOT / "data" / arm / split
            if not (path / "dataset_dict.json").is_file():
                raise SystemExit(
                    f"missing or non-Arrow dataset: {path}  (run make_dataset.py "
                    f"with the oat interpreter)")
    snapshot = (REPO / ".runtime/hf_home/hub/models--Qwen--Qwen3-8B/snapshots"
                / MODEL_COMMIT)
    if not snapshot.is_dir():
        raise SystemExit(f"missing pinned offline snapshot: {snapshot}")
    existing = sorted((PARENT / "reports").glob(f"{TAG_PREFIX}*_{MODEL_KEY}_s*"))
    if existing:
        raise SystemExit(
            "pilot run directories already exist; move the failed attempts under "
            "reports/failed_attempts/ first:\n  "
            + "\n  ".join(p.name for p in existing))
    # Two gates, both run with the interpreter the jobs themselves use.
    #
    #   validate_dataset.py     the dataset says what the protocol claims
    #   preflight_trainer_load  the trainer can actually open it
    #
    # The second exists because the first passed on a raw .jsonl that oat's
    # loader cannot read: six jobs reached the cluster, died 80 seconds in and
    # hung for 7.5 hours holding two A6000s each. Content validity and
    # loadability are different properties and both are checked here.
    for script in ("validate_dataset.py", "preflight_trainer_load.py"):
        result = subprocess.run([OAT_PYTHON, str(ROOT / script)],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(
                f"{script} failed, refusing to submit:\n"
                + (result.stdout or "")[-2000:] + (result.stderr or "")[-1000:])
        print(f"{script} passed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    preflight()
    roster = [{"arm": arm, "seed": seed, "environment": environment(arm, seed)}
              for arm in ARMS for seed in SEEDS]
    for record in roster:
        record["command"] = command(record["environment"])
        print(shlex.join(record["command"]), flush=True)
    print(f"\n{len(roster)} pilot job(s); submit={args.submit}", flush=True)
    if not args.submit:
        return

    for record in roster:
        out = subprocess.check_output(record["command"], cwd=REPO, text=True).strip()
        job_id = out.split(";", 1)[0]
        if not job_id.isdigit():
            raise RuntimeError(f"unexpected sbatch output: {out!r}")
        record["job_id"] = job_id
        print(f"submitted {job_id}  {record['arm']} s{record['seed']}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RUNS.mkdir(parents=True, exist_ok=True)
    ledger = {
        "created_at": stamp,
        "kind": "structure_selection_pilot",
        "protocol": PROTOCOL,
        "status": "pilot",
        "model_key": MODEL_KEY,
        "model": MODEL,
        "model_commit": MODEL_COMMIT,
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "checkpoint_flags": CHECKPOINT_FLAGS,
        "tag_prefix": TAG_PREFIX,
        "report_root": str((PARENT / "reports").relative_to(REPO)),
        "fine_contrast_requires_eight_seeds": True,
        "training_jobs": roster,
        "code_sha256": {
            str(p.relative_to(REPO)): sha256(p) for p in (
                ROOT / "make_dataset.py", ROOT / "validate_dataset.py",
                ROOT / "PROTOCOL.md", ROOT / "submit_pilot.py",
                PARENT / "train.sh")
        },
        "dataset_manifest": json.loads((ROOT / "data/manifest.json").read_text()),
    }
    path = RUNS / f"structure_selection_pilot_{stamp}.json"
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    print(f"\nledger -> {path}", flush=True)


if __name__ == "__main__":
    main()
