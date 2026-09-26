#!/usr/bin/env python3
"""Extract final-token layer states for base or LoRA Experiment 3 endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_TASK_DIR = HERE / "data"
DEFAULT_RUN_DIR = HERE / "runs" / "qwen3_8b_base"


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


def load_tasks(task_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks_path = task_dir / "tasks.jsonl"
    manifest_path = task_dir / "task_manifest.json"
    tasks = read_jsonl(tasks_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(tasks) != manifest["record_count"]:
        raise ValueError("task count does not match manifest")
    if file_sha256(tasks_path) != manifest["tasks_sha256"]:
        raise ValueError("tasks.jsonl hash does not match manifest")
    seen = set()
    for row in tasks:
        if row["sample_id"] in seen:
            raise ValueError(f"duplicate sample ID {row['sample_id']}")
        seen.add(row["sample_id"])
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


def verified_existing_features(path: Path, tasks: list[dict[str, Any]]) -> bool:
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as payload:
        observed = payload["sample_ids"].astype(str).tolist()
        features = payload["last_token"]
    return observed == [row["sample_id"] for row in tasks] and features.shape[0] == len(tasks)


def model_input_device(model):
    return model.get_input_embeddings().weight.device


def extract_features(
    *,
    model,
    tokenizer,
    tasks: list[dict[str, Any]],
    batch_size: int,
    max_input_tokens: int,
    output_path: Path,
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
            f"{tasks[index]['sample_id']} has {token_counts[index]} tokens; "
            f"limit is {max_input_tokens}"
        )

    layer_count = int(model.config.num_hidden_layers) + 1
    hidden_size = int(model.config.hidden_size)
    last_token = np.empty((len(tasks), layer_count, hidden_size), dtype=np.float16)
    mean_control = np.empty((len(tasks), 2, hidden_size), dtype=np.float16)
    order = np.argsort(token_counts, kind="stable")
    started = time.time()

    for batch_start in range(0, len(order), batch_size):
        indices = order[batch_start : batch_start + batch_size].tolist()
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
        torch.cuda.empty_cache()
        completed = min(batch_start + len(indices), len(order))
        print(
            f"features {completed}/{len(order)} "
            f"({time.time() - started:.1f}s, max_tokens={max(token_counts[i] for i in indices)})",
            flush=True,
        )

    atomic_savez(
        output_path,
        last_token=last_token,
        mean_control=mean_control,
        mean_control_layer_indices=np.asarray((0, layer_count - 1), dtype=np.int32),
        sample_ids=np.asarray([row["sample_id"] for row in tasks]),
        token_counts=token_counts,
    )
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


def adapter_manifest(adapter: Path | None) -> dict[str, Any] | None:
    if adapter is None:
        return None
    config_path = adapter / "adapter_config.json"
    weights_path = adapter / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"incomplete adapter: {adapter}")
    return {
        "path": str(adapter.resolve()),
        "config_sha256": file_sha256(config_path),
        "weights_sha256": file_sha256(weights_path),
        "config": json.loads(config_path.read_text(encoding="utf-8")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, default=DEFAULT_TASK_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--model-commit", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--endpoint-label", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-input-tokens", type=int, default=3072)
    parser.add_argument(
        "--device-map",
        choices=("none", "auto", "balanced", "balanced_low_0", "sequential"),
        default="none",
    )
    parser.add_argument("--gpu-memory-gib", type=int, default=44)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required for hidden-state extraction")
    torch.manual_seed(20260824)
    torch.cuda.manual_seed_all(20260824)
    torch.backends.cuda.matmul.allow_tf32 = True

    tasks, task_manifest = load_tasks(args.task_dir)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    features_path = args.run_dir / "hidden_states.npz"
    if not args.force and verified_existing_features(features_path, tasks):
        manifest_path = args.run_dir / "extraction_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("verified features exist without an extraction manifest")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if file_sha256(features_path) != existing["features_sha256"]:
            raise ValueError("existing hidden-state hash does not match extraction manifest")
        if existing["endpoint_label"] != args.endpoint_label:
            raise ValueError("existing endpoint label does not match request")
        print(f"verified existing features: {features_path}", flush=True)
        return

    adapter_info = adapter_manifest(args.adapter)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    load_kwargs: dict[str, Any] = {
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
    resolved_commit = getattr(model.config, "_commit_hash", None)
    if resolved_commit is not None and resolved_commit != args.model_commit:
        raise ValueError(
            f"model commit mismatch: expected {args.model_commit}, observed {resolved_commit}"
        )
    if args.adapter is not None:
        from peft import PeftModel

        base_name = adapter_info["config"].get("base_model_name_or_path")
        if base_name != args.model:
            raise ValueError(f"adapter base {base_name!r} does not match {args.model!r}")
        model = PeftModel.from_pretrained(model, str(args.adapter), is_trainable=False)
    model.eval()
    if args.device_map == "none":
        model.to("cuda")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        f"loaded {args.model_label}/{args.endpoint_label}: "
        f"layers={model.config.num_hidden_layers}, hidden={model.config.hidden_size}",
        flush=True,
    )
    feature_summary = extract_features(
        model=model,
        tokenizer=tokenizer,
        tasks=tasks,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
        output_path=features_path,
    )
    resolved_device_map = {
        str(key): str(value) for key, value in getattr(model, "hf_device_map", {}).items()
    }
    manifest = {
        "study": task_manifest["study"],
        "status": "features_complete",
        "model": args.model,
        "model_label": args.model_label,
        "model_commit": args.model_commit,
        "resolved_model_commit": resolved_commit,
        "endpoint_label": args.endpoint_label,
        "adapter": adapter_info,
        "parameter_count": parameter_count,
        "trainable_parameter_count_at_extraction": trainable_count,
        "chat_template": "native single-user template; thinking disabled when supported",
        "primary_anchor": "final input token",
        "control_pooling": ["mean embedding output", "mean final layer"],
        "device_map_request": args.device_map,
        "resolved_device_map": resolved_device_map,
        "task_manifest_sha256": file_sha256(args.task_dir / "task_manifest.json"),
        "tasks_sha256": file_sha256(args.task_dir / "tasks.jsonl"),
        "features_sha256": file_sha256(features_path),
        "feature_summary": feature_summary,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": package_version("peft"),
            "accelerate": package_version("accelerate"),
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.run_dir / "extraction_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
