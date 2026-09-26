#!/usr/bin/env python3
"""Extract open-model hidden states and held-out behavior for the Coin City probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_RUN_DIR = (
    ROOT
    / "data"
    / "coin_city_stable_relationship_claude_n250_v4"
    / "mechanistic_probe"
    / "qwen3_8b_toy_v1"
)
DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_MODEL_LABEL = "Qwen3-8B"
DECODE_SEED = 20260824

EVAL_DIR = ROOT / "eval"
sys.path.insert(0, str(EVAL_DIR))
from run_frozen_task_shard import _parse_reply, _strip_reasoning  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_tasks(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks_path = run_dir / "tasks.jsonl"
    manifest_path = run_dir / "task_manifest.json"
    tasks = read_jsonl(tasks_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(tasks) != manifest["record_count"]:
        raise ValueError("task count does not match task manifest")
    if file_sha256(tasks_path) != manifest["tasks_sha256"]:
        raise ValueError("tasks.jsonl hash does not match task manifest")
    sample_ids: set[str] = set()
    for row in tasks:
        if row["sample_id"] in sample_ids:
            raise ValueError(f"duplicate sample ID {row['sample_id']}")
        sample_ids.add(row["sample_id"])
        if hashlib.sha256(row["prompt"].encode()).hexdigest() != row["prompt_sha256"]:
            raise ValueError(f"bad prompt hash for {row['sample_id']}")
    return tasks, manifest


def formatted_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)


def verified_existing_behavior(path: Path, tasks: list[dict[str, Any]]) -> dict | None:
    """Summary of an existing complete generation file, or None if unusable."""
    if not path.exists():
        return None
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = [row["sample_id"] for row in tasks if row["split"] == "test"]
    if [row["sample_id"] for row in rows] != expected:
        return None
    return {
        "status": "verified_existing",
        "record_count": len(rows),
        "parsed_count": sum(bool(row.get("parsed")) for row in rows),
    }


def verified_existing_features(path: Path, tasks: list[dict[str, Any]]) -> bool:
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as payload:
        observed = payload["sample_ids"].astype(str).tolist()
        features = payload["last_token"]
    return observed == [row["sample_id"] for row in tasks] and features.shape[0] == len(tasks)


def model_input_device(model):
    """Return the device that owns token embeddings under either load strategy."""
    return model.get_input_embeddings().weight.device


def load_feature_checkpoint(
    path: Path, tasks: list[dict[str, Any]], layer_count: int, hidden_size: int
):
    """Return (last_token, mean_control, completed) from a partial run, if usable."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            if payload["sample_ids"].astype(str).tolist() != [
                row["sample_id"] for row in tasks
            ]:
                return None
            last_token = payload["last_token"]
            mean_control = payload["mean_control"]
            completed = payload["completed"].astype(bool)
    except Exception:
        return None
    if last_token.shape != (len(tasks), layer_count, hidden_size):
        return None
    return last_token, mean_control, completed


def extract_features(
    *,
    model,
    tokenizer,
    tasks: list[dict[str, Any]],
    batch_size: int,
    max_input_tokens: int,
    output_path: Path,
    checkpoint_every: int = 200,
) -> dict[str, Any]:
    import torch

    formatted = [formatted_prompt(tokenizer, row["prompt"]) for row in tasks]
    token_counts = np.asarray(
        [len(tokenizer.encode(text, add_special_tokens=False)) for text in formatted],
        dtype=np.int32,
    )
    if int(token_counts.max()) > max_input_tokens:
        index = int(token_counts.argmax())
        raise ValueError(
            f"{tasks[index]['sample_id']} has {token_counts[index]} tokens, "
            f"above max_input_tokens={max_input_tokens}"
        )

    layer_count = int(model.config.num_hidden_layers) + 1
    hidden_size = int(model.config.hidden_size)
    partial_path = output_path.with_name(output_path.name + ".partial.npz")
    resumed = load_feature_checkpoint(partial_path, tasks, layer_count, hidden_size)
    if resumed is None:
        last_token = np.empty((len(tasks), layer_count, hidden_size), dtype=np.float16)
        mean_control = np.empty((len(tasks), 2, hidden_size), dtype=np.float16)
        completed_mask = np.zeros(len(tasks), dtype=bool)
    else:
        last_token, mean_control, completed_mask = resumed
        print(
            f"resuming feature extraction: {int(completed_mask.sum())}/{len(tasks)} "
            "records already present",
            flush=True,
        )

    def checkpoint() -> None:
        atomic_savez(
            partial_path,
            last_token=last_token,
            mean_control=mean_control,
            completed=completed_mask,
            sample_ids=np.asarray([row["sample_id"] for row in tasks]),
        )

    order = np.argsort(token_counts, kind="stable")
    started = time.time()
    since_checkpoint = 0
    for batch_start in range(0, len(order), batch_size):
        indices = order[batch_start : batch_start + batch_size].tolist()
        indices = [index for index in indices if not completed_mask[index]]
        if not indices:
            continue
        encoded = tokenizer(
            [formatted[index] for index in indices],
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(model_input_device(model)) for key, value in encoded.items()
        }
        with torch.inference_mode():
            output = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden_states = output.hidden_states
        if len(hidden_states) != layer_count:
            raise RuntimeError(
                f"model returned {len(hidden_states)} states; expected {layer_count}"
            )
        mask = encoded["attention_mask"].unsqueeze(-1)
        for layer_index, state in enumerate(hidden_states):
            final = state[:, -1, :].to(dtype=torch.float32).cpu().numpy()
            last_token[indices, layer_index, :] = final.astype(np.float16)
        for control_index, layer_index in enumerate((0, layer_count - 1)):
            state = hidden_states[layer_index].to(dtype=torch.float32)
            layer_mask = mask.to(device=state.device, dtype=torch.float32)
            denominator = layer_mask.sum(dim=1).clamp_min(1.0)
            pooled = ((state * layer_mask).sum(dim=1) / denominator).cpu().numpy()
            mean_control[indices, control_index, :] = pooled.astype(np.float16)

        del output, hidden_states, encoded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        completed_mask[indices] = True
        since_checkpoint += len(indices)
        if since_checkpoint >= checkpoint_every:
            checkpoint()
            since_checkpoint = 0
        elapsed = time.time() - started
        print(
            f"features {int(completed_mask.sum())}/{len(order)} "
            f"({elapsed:.1f}s, max_tokens={max(token_counts[i] for i in indices)})",
            flush=True,
        )

    if not completed_mask.all():
        checkpoint()
        raise RuntimeError(
            f"feature extraction incomplete: {int(completed_mask.sum())}/{len(tasks)}"
        )
    atomic_savez(
        output_path,
        last_token=last_token,
        mean_control=mean_control,
        mean_control_layer_indices=np.asarray((0, layer_count - 1), dtype=np.int32),
        sample_ids=np.asarray([row["sample_id"] for row in tasks]),
        token_counts=token_counts,
    )
    if partial_path.exists():
        partial_path.unlink()
    return {
        "record_count": len(tasks),
        "layer_count_including_embedding": layer_count,
        "transformer_layer_count": layer_count - 1,
        "hidden_size": hidden_size,
        "token_count_min": int(token_counts.min()),
        "token_count_median": float(np.median(token_counts)),
        "token_count_max": int(token_counts.max()),
        "feature_dtype": "float16",
        "feature_shape": list(last_token.shape),
        "mean_control_shape": list(mean_control.shape),
        "elapsed_seconds": time.time() - started,
    }


def generate_behavior(
    *,
    model,
    tokenizer,
    tasks: list[dict[str, Any]],
    batch_size: int,
    max_new_tokens: int,
    output_path: Path,
) -> dict[str, Any]:
    import torch

    test_tasks = [row for row in tasks if row["split"] == "test"]
    formatted = [formatted_prompt(tokenizer, row["prompt"]) for row in test_tasks]
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    records: list[dict[str, Any]] = []
    if partial_path.exists():
        records = [
            json.loads(line)
            for line in partial_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        print(
            f"resuming generation: {len(records)}/{len(test_tasks)} already present",
            flush=True,
        )
    done_ids = {row["sample_id"] for row in records}
    pending = [
        (task, text)
        for task, text in zip(test_tasks, formatted)
        if task["sample_id"] not in done_ids
    ]
    started = time.time()
    for batch_start in range(0, len(pending), batch_size):
        batch_tasks = [item[0] for item in pending[batch_start : batch_start + batch_size]]
        batch_prompts = [item[1] for item in pending[batch_start : batch_start + batch_size]]
        encoded = tokenizer(
            batch_prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(model_input_device(model)) for key, value in encoded.items()
        }
        input_width = int(encoded["input_ids"].shape[1])
        with torch.inference_mode():
            sequences = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        generated = sequences[:, input_width:]
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for task, text in zip(batch_tasks, texts):
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
        print(
            f"behavior {len(records)}/{len(test_tasks)}",
            flush=True,
        )
        partial_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
            encoding="utf-8",
        )
        del encoded, sequences, generated
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(records) != len(test_tasks):
        raise RuntimeError(
            f"generation incomplete: {len(records)}/{len(test_tasks)}"
        )
    order = {row["sample_id"]: index for index, row in enumerate(test_tasks)}
    records.sort(key=lambda row: order[row["sample_id"]])
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(output_path)
    if partial_path.exists():
        partial_path.unlink()
    return {
        "record_count": len(records),
        "parsed_count": sum(row["parsed"] for row in records),
        "max_new_tokens": max_new_tokens,
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-id",
        default=None,
        help="Canonical model ID to record when --model is a staged local snapshot.",
    )
    parser.add_argument(
        "--model-commit",
        default=None,
        help="Pinned checkpoint commit to record for a staged local snapshot.",
    )
    parser.add_argument("--model-label", default=DEFAULT_MODEL_LABEL)
    parser.add_argument(
        "--study",
        default=None,
        help="Run identity recorded in the extraction manifest (defaults to task study).",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--generation-batch-size", type=int, default=2)
    parser.add_argument("--max-input-tokens", type=int, default=2304)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument(
        "--device-map",
        choices=("none", "auto", "balanced", "balanced_low_0", "sequential"),
        default="none",
        help="Use 'none' for the original one-GPU load or an Accelerate map for larger models.",
    )
    parser.add_argument(
        "--gpu-memory-gib",
        type=int,
        default=44,
        help="Per-GPU placement limit when --device-map is not 'none'.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # The CUDA requirement belongs to the branch that loads weights; a run whose
    # artifacts are already complete only records provenance and needs no GPU.
    if torch.cuda.is_available():
        torch.manual_seed(DECODE_SEED)
        torch.cuda.manual_seed_all(DECODE_SEED)
        torch.backends.cuda.matmul.allow_tf32 = True

    tasks, task_manifest = load_tasks(args.run_dir)
    features_path = args.run_dir / "hidden_states.npz"
    behavior_path = args.run_dir / "behavior_test.jsonl"
    existing_behavior = (
        None if args.force else verified_existing_behavior(behavior_path, tasks)
    )
    behavior_ready = args.skip_generation or existing_behavior is not None
    if (
        not args.force
        and behavior_ready
        and verified_existing_features(features_path, tasks)
    ):
        # Both artifacts are already complete, so this run only records provenance.
        print(f"verified existing features: {features_path}", flush=True)
        loaded_model = None
        feature_summary = {
            "status": "verified_existing",
            "record_count": len(tasks),
        }
        behavior_summary = existing_behavior or {"status": "skipped"}
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("a CUDA GPU is required for hidden-state extraction")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            local_files_only=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        load_kwargs = {
            "local_files_only": True,
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        }
        if args.device_map != "none":
            load_kwargs.update(
                {
                    "low_cpu_mem_usage": True,
                    "device_map": args.device_map,
                    "max_memory": {
                        index: f"{args.gpu_memory_gib}GiB"
                        for index in range(torch.cuda.device_count())
                    },
                }
            )
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        model.eval()
        if args.device_map == "none":
            model.to("cuda")
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"loaded {args.model}: layers={model.config.num_hidden_layers}, "
            f"hidden={model.config.hidden_size}, parameters={parameter_count}, "
            f"device_map={getattr(model, 'hf_device_map', None)}",
            flush=True,
        )
        feature_summary = extract_features(
            model=model,
            tokenizer=tokenizer,
            tasks=tasks,
            batch_size=args.batch_size,
            max_input_tokens=args.max_input_tokens,
            output_path=features_path,
            checkpoint_every=args.checkpoint_every,
        )
        if args.skip_generation:
            behavior_summary = {"status": "skipped"}
        else:
            behavior_summary = generate_behavior(
                model=model,
                tokenizer=tokenizer,
                tasks=tasks,
                batch_size=args.generation_batch_size,
                max_new_tokens=args.max_new_tokens,
                output_path=behavior_path,
            )
        loaded_model = model
    if loaded_model is None:
        # Nothing was loaded because both artifacts were already verified, so the
        # commit is the asserted one rather than one read back from weights.
        model_commit = args.model_commit
        commit_source = "asserted by --model-commit; no weights loaded this run"
        resolved_device_map = {}
        parameter_count = None
    else:
        model_commit = args.model_commit or getattr(
            loaded_model.config, "_commit_hash", None
        )
        commit_source = "read from the loaded checkpoint config"
        resolved_device_map = {
            str(key): str(value)
            for key, value in getattr(loaded_model, "hf_device_map", {}).items()
        }
        del loaded_model
        torch.cuda.empty_cache()

    manifest = {
        "study": args.study or task_manifest["study"],
        "status": "complete",
        "model": args.model_id or args.model,
        "model_label": args.model_label,
        "model_commit": model_commit,
        "model_commit_source": commit_source,
        "device_map_request": args.device_map,
        "gpu_memory_gib": (
            None if args.device_map == "none" else args.gpu_memory_gib
        ),
        "resolved_device_map": resolved_device_map,
        "parameter_count": parameter_count,
        "decode_seed": DECODE_SEED,
        "chat_template": "single user message; add_generation_prompt=True; enable_thinking=False",
        "primary_anchor": "final input token",
        "control_pooling": ["mean embedding output", "mean final layer"],
        "task_manifest_sha256": file_sha256(args.run_dir / "task_manifest.json"),
        "tasks_sha256": file_sha256(args.run_dir / "tasks.jsonl"),
        "features_sha256": file_sha256(features_path),
        "behavior_sha256": (
            None if args.skip_generation else file_sha256(behavior_path)
        ),
        "feature_summary": feature_summary,
        "behavior_summary": behavior_summary,
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
    manifest_path = args.run_dir / "extraction_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

