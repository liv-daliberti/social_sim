#!/usr/bin/env python3
"""Run the sealed Coin City behavior screen without extracting hidden states."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from extract_qwen_hidden_states import (
    DECODE_SEED,
    file_sha256,
    generate_behavior,
    load_tasks,
    read_jsonl,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_TASKS_DIR = (
    ROOT
    / "data"
    / "coin_city_stable_relationship_claude_n250_v4"
    / "mechanistic_probe"
    / "qwen3_8b_toy_v1"
)


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def verify_existing(path: Path, expected_ids: list[str]) -> bool:
    if not path.exists():
        return False
    rows = read_jsonl(path)
    return (
        len(rows) == len(expected_ids)
        and [row["sample_id"] for row in rows] == expected_ids
        and len(set(expected_ids)) == len(expected_ids)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument(
        "--device-map",
        choices=("auto", "balanced", "balanced_low_0", "sequential"),
        default="auto",
    )
    parser.add_argument(
        "--gpu-memory-gib",
        type=int,
        default=44,
        help="Per-GPU placement limit; leaves space for generation KV cache.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("at least one CUDA GPU is required")
    torch.manual_seed(DECODE_SEED)
    torch.cuda.manual_seed_all(DECODE_SEED)
    torch.backends.cuda.matmul.allow_tf32 = True

    tasks, task_manifest = load_tasks(args.tasks_dir)
    test_ids = [row["sample_id"] for row in tasks if row["split"] == "test"]
    if len(test_ids) != 96:
        raise RuntimeError(f"sealed behavior set has {len(test_ids)} prompts, expected 96")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    behavior_path = args.out_dir / "behavior_test.jsonl"
    manifest_path = args.out_dir / "behavior_manifest.json"
    if not args.force and verify_existing(behavior_path, test_ids):
        print(f"verified existing behavior: {behavior_path}", flush=True)
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    max_memory = {
        index: f"{args.gpu_memory_gib}GiB" for index in range(torch.cuda.device_count())
    }
    load_started = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map=args.device_map,
        max_memory=max_memory,
    )
    model.eval()
    load_seconds = time.time() - load_started
    print(
        f"loaded {args.model}: parameters={model.num_parameters():,}, "
        f"device_map={model.hf_device_map}, seconds={load_seconds:.1f}",
        flush=True,
    )

    behavior_summary = generate_behavior(
        model=model,
        tokenizer=tokenizer,
        tasks=tasks,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        output_path=behavior_path,
    )
    if behavior_summary["record_count"] != 96:
        raise RuntimeError("behavior screen did not produce all sealed prompts")

    manifest = {
        "study": "qwen_behavioral_size_screen_v1",
        "status": "complete",
        "model": args.model,
        "model_label": args.model_label,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "parameter_count": int(model.num_parameters()),
        "decode_seed": DECODE_SEED,
        "decoding": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "chat_template": (
                "single user message; add_generation_prompt=True; "
                "enable_thinking=False"
            ),
        },
        "task_manifest_sha256": file_sha256(args.tasks_dir / "task_manifest.json"),
        "tasks_sha256": task_manifest["tasks_sha256"],
        "behavior_sha256": file_sha256(behavior_path),
        "behavior_summary": behavior_summary,
        "load_seconds": load_seconds,
        "device_map_request": args.device_map,
        "gpu_memory_gib": args.gpu_memory_gib,
        "resolved_device_map": {str(k): str(v) for k, v in model.hf_device_map.items()},
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "accelerate": package_version("accelerate"),
            "safetensors": package_version("safetensors"),
        },
        "cuda": {
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "max_memory_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(index)
                    ),
                }
                for index in range(torch.cuda.device_count())
            ],
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
