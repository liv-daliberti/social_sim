#!/usr/bin/env python3
"""Extract all-layer states at the six frozen prompt anchors for one seed pair."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    ANCHORS,
    MAX_INPUT_TOKENS,
    MODEL,
    MODEL_COMMIT,
    RUNS_DIR,
    STUDY,
    TRAINING_SEEDS,
    adapter_freeze,
    atomic_json,
    file_sha256,
    load_frozen_tasks,
    resolve_adapter,
)


def formatted_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def token_layout(tokenizer, tasks: list[dict[str, Any]]) -> tuple[list[str], np.ndarray, np.ndarray]:
    formatted = []
    token_counts = np.empty(len(tasks), dtype=np.int32)
    positions = np.empty((len(tasks), len(ANCHORS)), dtype=np.int32)
    for index, task in enumerate(tasks):
        rendered = formatted_prompt(tokenizer, task["prompt"])
        raw_start = rendered.find(task["prompt"])
        if raw_start < 0 or rendered.find(task["prompt"], raw_start + 1) >= 0:
            raise ValueError(f"raw prompt is not uniquely embedded for {task['sample_id']}")
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded["offset_mapping"]
        token_counts[index] = len(encoded["input_ids"])
        for anchor_index, anchor in enumerate(ANCHORS):
            if anchor == "final_prompt":
                positions[index, anchor_index] = len(offsets) - 1
                continue
            boundary = raw_start + int(task["anchor_char_ends"][anchor])
            eligible = [
                token_index
                for token_index, (start, end) in enumerate(offsets)
                if end > start and end <= boundary
            ]
            if not eligible:
                raise ValueError(f"no token before {anchor} for {task['sample_id']}")
            token_index = eligible[-1]
            if offsets[token_index][1] < boundary - 16:
                raise ValueError(f"anchor mapping is unexpectedly distant for {task['sample_id']}")
            positions[index, anchor_index] = token_index
        formatted.append(rendered)
    if int(token_counts.max()) > MAX_INPUT_TOKENS:
        worst = int(token_counts.argmax())
        raise ValueError(
            f"{tasks[worst]['sample_id']} has {token_counts[worst]} tokens; "
            f"frozen limit is {MAX_INPUT_TOKENS}"
        )
    return formatted, token_counts, positions


def input_device(model):
    return model.get_input_embeddings().weight.device


def verified_state_file(path: Path, shape: tuple[int, ...]) -> bool:
    if not path.is_file():
        return False
    observed = np.load(path, mmap_mode="r", allow_pickle=False)
    return observed.shape == shape and observed.dtype == np.float16


def extract_endpoint(
    *,
    model,
    tokenizer,
    tasks: list[dict[str, Any]],
    formatted: list[str],
    token_counts: np.ndarray,
    positions: np.ndarray,
    output_path: Path,
    batch_size: int,
    endpoint: str,
    disable_adapter: bool = False,
) -> dict[str, Any]:
    import torch

    layer_count = int(model.config.num_hidden_layers) + 1
    hidden_size = int(model.config.hidden_size)
    shape = (len(tasks), layer_count, len(ANCHORS), hidden_size)
    if verified_state_file(output_path, shape):
        return {
            "endpoint": endpoint,
            "path": str(output_path),
            "shape": list(shape),
            "status": "verified_existing",
            "sha256": file_sha256(output_path),
        }
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite invalid partial state file: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    states_out = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float16, shape=shape
    )
    order = np.argsort(token_counts, kind="stable")
    started = time.time()
    context = model.disable_adapter() if disable_adapter else nullcontext()
    with context:
        for batch_start in range(0, len(order), batch_size):
            indices = order[batch_start : batch_start + batch_size].tolist()
            encoded = tokenizer(
                [formatted[index] for index in indices],
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
            padded_length = int(encoded["input_ids"].shape[1])
            encoded = {key: value.to(input_device(model)) for key, value in encoded.items()}
            with torch.inference_mode():
                output = model(
                    **encoded,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            if len(output.hidden_states) != layer_count:
                raise RuntimeError("unexpected hidden-state layer count")
            for layer_index, layer_states in enumerate(output.hidden_states):
                for batch_index, task_index in enumerate(indices):
                    left_padding = padded_length - int(token_counts[task_index])
                    selected = positions[task_index] + left_padding
                    values = layer_states[batch_index, selected, :]
                    states_out[task_index, layer_index, :, :] = (
                        values.float().cpu().numpy().astype(np.float16)
                    )
            states_out.flush()
            completed = min(batch_start + batch_size, len(order))
            print(
                f"{endpoint}: {completed}/{len(order)} prompts "
                f"({time.time() - started:.1f}s)",
                flush=True,
            )
            del output, encoded
            torch.cuda.empty_cache()
    del states_out
    temporary.replace(output_path)
    return {
        "endpoint": endpoint,
        "path": str(output_path),
        "shape": list(shape),
        "dtype": "float16",
        "elapsed_seconds": time.time() - started,
        "status": "complete",
        "sha256": file_sha256(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=TRAINING_SEEDS)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, help="smoke-test prefix only; never accepted as complete")
    parser.add_argument("--include-base", action="store_true")
    parser.add_argument(
        "--arms", nargs="*", choices=("matched", "prior"),
        help="endpoint shard; default extracts both trained arms",
    )
    args = parser.parse_args()
    if args.include_base and args.seed != 42:
        raise ValueError("the frozen base diagnostic is assigned only to seed 42")
    requested_arms = ["matched", "prior"] if args.arms is None else list(args.arms)
    if not requested_arms and not args.include_base:
        raise ValueError("no extraction endpoint requested")

    protocol_path = HERE / "protocol" / "frozen_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("study") != STUDY:
        raise ValueError("missing frozen protocol")
    tasks, task_manifest = load_frozen_tasks()
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        tasks = tasks[: args.limit]
    run_dir = args.run_dir or RUNS_DIR / f"qwen3_8b_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(20260825 + args.seed)
    torch.cuda.manual_seed_all(20260825 + args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    formatted, token_counts, positions = token_layout(tokenizer, tasks)
    np.save(run_dir / "anchor_token_positions.npy", positions, allow_pickle=False)
    np.save(run_dir / "token_counts.npy", token_counts, allow_pickle=False)
    (run_dir / "sample_ids.json").write_text(
        json.dumps([row["sample_id"] for row in tasks]) + "\n", encoding="utf-8"
    )
    print(
        f"token layout: min={token_counts.min()} median={np.median(token_counts):.0f} "
        f"max={token_counts.max()}",
        flush=True,
    )

    base = AutoModelForCausalLM.from_pretrained(
        MODEL,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    resolved_commit = getattr(base.config, "_commit_hash", None)
    if resolved_commit not in (None, MODEL_COMMIT):
        raise ValueError(f"model commit mismatch: {resolved_commit}")
    matched_path = resolve_adapter(args.seed, "matched")
    prior_path = resolve_adapter(args.seed, "prior")
    model = PeftModel.from_pretrained(
        base, str(matched_path), adapter_name="matched", is_trainable=False
    )
    model.load_adapter(str(prior_path), adapter_name="prior", is_trainable=False)
    model.eval().to("cuda")
    outputs = []
    if args.include_base:
        outputs.append(
            extract_endpoint(
                model=model,
                tokenizer=tokenizer,
                tasks=tasks,
                formatted=formatted,
                token_counts=token_counts,
                positions=positions,
                output_path=run_dir / "states_base.npy",
                batch_size=args.batch_size,
                endpoint="base",
                disable_adapter=True,
            )
        )
    for arm in requested_arms:
        model.set_adapter(arm)
        outputs.append(
            extract_endpoint(
                model=model,
                tokenizer=tokenizer,
                tasks=tasks,
                formatted=formatted,
                token_counts=token_counts,
                positions=positions,
                output_path=run_dir / f"states_{arm}.npy",
                batch_size=args.batch_size,
                endpoint=arm,
            )
        )
    manifest = {
        "study": STUDY,
        "status": "smoke_complete" if args.limit else "endpoint_extraction_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "model": MODEL,
        "model_commit": MODEL_COMMIT,
        "resolved_model_commit": resolved_commit,
        "task_manifest_sha256": file_sha256(HERE / "data" / "task_manifest.json"),
        "tasks_sha256": task_manifest["tasks_sha256"],
        "protocol_sha256": file_sha256(protocol_path),
        "record_count": len(tasks),
        "full_frozen_record_count": task_manifest["record_count"],
        "anchors": list(ANCHORS),
        "requested_arms": requested_arms,
        "include_base": args.include_base,
        "token_count_min": int(token_counts.min()),
        "token_count_median": float(np.median(token_counts)),
        "token_count_max": int(token_counts.max()),
        "adapters": [adapter_freeze(args.seed, arm) for arm in ("matched", "prior")],
        "outputs": outputs,
    }
    atomic_json(run_dir / "extraction_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
