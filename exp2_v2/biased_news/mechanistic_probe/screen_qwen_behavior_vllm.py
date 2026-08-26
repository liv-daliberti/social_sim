#!/usr/bin/env python3
"""Run the sealed Coin City behavior screen with tensor-parallel vLLM."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from extract_qwen_hidden_states import (
    DECODE_SEED,
    _parse_reply,
    _strip_reasoning,
    file_sha256,
    formatted_prompt,
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


def verify_existing(
    behavior_path: Path,
    manifest_path: Path,
    expected_ids: list[str],
    tasks_sha256: str,
) -> bool:
    if not behavior_path.exists() or not manifest_path.exists():
        return False
    rows = read_jsonl(behavior_path)
    manifest = json.loads(manifest_path.read_text())
    observed_ids = [row["sample_id"] for row in rows]
    return (
        manifest.get("status") == "complete"
        and manifest.get("tasks_sha256") == tasks_sha256
        and manifest.get("behavior_sha256") == file_sha256(behavior_path)
        and observed_ids == expected_ids
        and len(observed_ids) == len(set(observed_ids)) == 96
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, default=2304)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")

    import torch
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM, SamplingParams

    if torch.cuda.device_count() != args.tensor_parallel_size:
        raise RuntimeError(
            f"visible GPUs={torch.cuda.device_count()}, "
            f"tensor_parallel_size={args.tensor_parallel_size}"
        )
    tasks, task_manifest = load_tasks(args.tasks_dir)
    test_tasks = [row for row in tasks if row["split"] == "test"]
    if len(test_tasks) != 96:
        raise RuntimeError(f"sealed behavior set has {len(test_tasks)} prompts")
    expected_ids = [row["sample_id"] for row in test_tasks]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    behavior_path = args.out_dir / "behavior_test.jsonl"
    manifest_path = args.out_dir / "behavior_manifest.json"
    if verify_existing(
        behavior_path,
        manifest_path,
        expected_ids,
        task_manifest["tasks_sha256"],
    ):
        print(f"verified existing behavior: {behavior_path}", flush=True)
        return
    if behavior_path.exists() or manifest_path.exists():
        raise RuntimeError(
            f"refusing to overwrite incomplete or invalid screen in {args.out_dir}"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = [formatted_prompt(tokenizer, row["prompt"]) for row in test_tasks]
    token_counts = [len(tokenizer.encode(text, add_special_tokens=False)) for text in prompts]
    if max(token_counts) > args.max_model_len - args.max_new_tokens:
        raise RuntimeError("prompt plus generation allowance exceeds max_model_len")
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)

    load_started = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        seed=DECODE_SEED,
        trust_remote_code=False,
        disable_log_stats=True,
    )
    load_seconds = time.time() - load_started
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        seed=DECODE_SEED,
    )
    generation_started = time.time()
    outputs = llm.generate(prompts, sampling, use_tqdm=True)
    generation_seconds = time.time() - generation_started
    if len(outputs) != len(test_tasks):
        raise RuntimeError("vLLM did not return every sealed prompt")

    records = []
    for task, output in zip(test_tasks, outputs):
        text = output.outputs[0].text
        raw = _strip_reasoning(text.strip())
        parsed = _parse_reply(raw)
        prediction = parsed["predicted_poll"]
        implied_slope = None
        absolute_error = None
        if prediction is not None:
            implied_slope = (
                float(prediction) - float(task["query_starting_poll"])
            ) / float(task["query_net_news"])
            absolute_error = abs(float(prediction) - float(task["gold_expected_poll"]))
        records.append(
            {
                "sample_id": task["sample_id"],
                "task_id": task["task_id"],
                "episode": task["episode"],
                "arm": task["arm"],
                "c_cases": task["c_cases"],
                "predicted_poll": prediction,
                "implied_slope": implied_slope,
                "absolute_error": absolute_error,
                "rationale": parsed["rationale"],
                "raw": raw[:8000],
                "parsed": prediction is not None,
            }
        )

    temporary = behavior_path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    temporary.replace(behavior_path)

    manifest = {
        "study": "qwen_behavioral_size_screen_v1",
        "status": "complete",
        "backend": "vllm_tensor_parallel",
        "model": os.environ.get("MODEL_ID", args.model),
        "model_label": args.model_label,
        "model_commit": os.environ.get("MODEL_COMMIT")
        or getattr(config, "_commit_hash", None),
        "parameter_count": None,
        "decode_seed": DECODE_SEED,
        "decoding": {
            "temperature": 0.0,
            "max_new_tokens": args.max_new_tokens,
            "chat_template": (
                "single user message; add_generation_prompt=True; "
                "enable_thinking=False"
            ),
        },
        "task_manifest_sha256": file_sha256(args.tasks_dir / "task_manifest.json"),
        "tasks_sha256": task_manifest["tasks_sha256"],
        "behavior_sha256": file_sha256(behavior_path),
        "behavior_summary": {
            "record_count": len(records),
            "parsed_count": sum(row["parsed"] for row in records),
            "token_count_min": min(token_counts),
            "token_count_max": max(token_counts),
            "load_seconds": load_seconds,
            "generation_seconds": generation_seconds,
        },
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "vllm": package_version("vllm"),
        },
        "cuda": {
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
