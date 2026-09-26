#!/usr/bin/env python3
"""Extract all-layer states at the exact A/B/C arbitrary-label subtokens."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from symbol_relational_common import (
    ANCHORS,
    checkpoint_spec,
    DEFAULT_RUN_DIR,
    run_spec,
    file_sha256,
    find_label_anchors,
    formatted_prompt,
    label_tasks,
    load_tasks,
)


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)


def model_input_device(model):
    return model.get_input_embeddings().weight.device


def validate_protocol(run_dir: Path) -> dict[str, Any]:
    checkpoint = checkpoint_spec(run_dir)
    path = run_dir / "relational_protocol_manifest.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("study") != run_spec(run_dir)["study"]:
        raise ValueError("protocol study mismatch")
    if protocol.get("status") != "frozen_before_label_activation_extraction":
        raise ValueError("protocol was not frozen before label-state extraction")
    if protocol.get("model", {}).get("id") != checkpoint["model_id"]:
        raise ValueError("protocol model mismatch")
    if protocol.get("model", {}).get("commit") != checkpoint["model_commit"]:
        raise ValueError("protocol checkpoint mismatch")
    return protocol


def extract(
    *,
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    output_path: Path,
    max_input_tokens: int,
) -> dict[str, Any]:
    import torch

    rendered = [formatted_prompt(tokenizer, row["prompt"]) for row in rows]
    anchors = [find_label_anchors(tokenizer, text) for text in rendered]
    token_counts = np.asarray(
        [len(tokenizer.encode(text, add_special_tokens=False)) for text in rendered],
        dtype=np.int32,
    )
    if int(token_counts.max()) > max_input_tokens:
        bad = int(token_counts.argmax())
        raise ValueError(
            f"{rows[bad]['sample_id']} has {token_counts[bad]} tokens; "
            f"limit is {max_input_tokens}"
        )

    layer_count = int(model.config.num_hidden_layers) + 1
    hidden_size = int(model.config.hidden_size)
    states = np.empty(
        (len(rows), layer_count, len(ANCHORS), 2, hidden_size),
        dtype=np.float16,
    )
    positions = np.empty((len(rows), len(ANCHORS), 2), dtype=np.int32)
    token_ids = np.empty((len(rows), len(ANCHORS), 2), dtype=np.int32)
    symbols = np.empty((len(rows), len(ANCHORS)), dtype="<U3")

    started = time.time()
    for index, (row, text, anchor_map) in enumerate(zip(rows, rendered, anchors)):
        encoded = tokenizer(
            text,
            add_special_tokens=False,
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
        if len(output.hidden_states) != layer_count:
            raise RuntimeError("model returned an unexpected hidden-state count")
        for anchor_index, anchor in enumerate(ANCHORS):
            span = anchor_map[anchor]["token_positions"]
            positions[index, anchor_index, :] = span
            token_ids[index, anchor_index, :] = anchor_map[anchor]["token_ids"]
            symbols[index, anchor_index] = anchor_map[anchor]["symbol"]
            for layer_index, hidden in enumerate(output.hidden_states):
                selected = hidden[0, span, :].to(dtype=torch.float32)
                if not torch.isfinite(selected).all():
                    raise RuntimeError(
                        f"nonfinite state: {row['sample_id']} layer={layer_index} "
                        f"anchor={anchor}"
                    )
                states[index, layer_index, anchor_index, :, :] = (
                    selected.cpu().numpy().astype(np.float16)
                )
        del output, encoded
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"label states {index + 1}/{len(rows)} "
            f"({time.time() - started:.1f}s, tokens={token_counts[index]})",
            flush=True,
        )

    atomic_savez(
        output_path,
        label_token_states=states,
        anchor_names=np.asarray(ANCHORS),
        sample_ids=np.asarray([row["sample_id"] for row in rows]),
        token_positions=positions,
        token_ids=token_ids,
        symbols=symbols,
        token_counts=token_counts,
    )
    with np.load(output_path, allow_pickle=False) as payload:
        if payload["label_token_states"].shape != states.shape:
            raise RuntimeError("written label-state tensor shape changed")
        if not np.isfinite(payload["label_token_states"]).all():
            raise RuntimeError("written label-state tensor contains nonfinite values")
    return {
        "record_count": len(rows),
        "feature_shape": list(states.shape),
        "feature_dtype": "float16",
        "layer_count_including_embedding": layer_count,
        "transformer_layer_count": layer_count - 1,
        "hidden_size": hidden_size,
        "anchors": list(ANCHORS),
        "subtokens_per_anchor": 2,
        "token_count_min": int(token_counts.min()),
        "token_count_median": float(np.median(token_counts)),
        "token_count_max": int(token_counts.max()),
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--model")
    parser.add_argument(
        "--device-map",
        help="optional Accelerate device map, e.g. balanced for multi-GPU models",
    )
    parser.add_argument(
        "--gpu-memory-gib",
        type=int,
        help="per-GPU memory ceiling used with --device-map",
    )
    parser.add_argument("--max-input-tokens", type=int, default=2304)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(20260825)
    torch.cuda.manual_seed_all(20260825)
    torch.backends.cuda.matmul.allow_tf32 = True

    checkpoint = checkpoint_spec(args.run_dir)
    model_id = str(checkpoint["model_id"])
    expected_commit = str(checkpoint["model_commit"])
    if args.model is not None and args.model != model_id:
        raise ValueError(f"--model must equal the frozen checkpoint {model_id}")
    protocol = validate_protocol(args.run_dir)
    tasks, _ = load_tasks(args.run_dir)
    rows = label_tasks(tasks)
    output_path = args.run_dir / "label_token_states.npz"
    manifest_path = args.run_dir / "label_extraction_manifest.json"
    if output_path.exists() and not args.force:
        raise FileExistsError(f"{output_path} already exists; refusing to overwrite")

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs = {
        "local_files_only": True,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
    }
    if args.device_map:
        load_kwargs["device_map"] = args.device_map
        if args.gpu_memory_gib is not None:
            load_kwargs["max_memory"] = {
                index: f"{args.gpu_memory_gib}GiB"
                for index in range(torch.cuda.device_count())
            }
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    if not args.device_map:
        model.to("cuda")
    observed_commit = getattr(model.config, "_commit_hash", None)
    if observed_commit != expected_commit:
        raise RuntimeError(
            f"checkpoint commit changed: observed={observed_commit}, "
            f"expected={expected_commit}"
        )
    if int(model.config.num_hidden_layers) != int(checkpoint["transformer_layers"]):
        raise RuntimeError("model depth differs from the frozen protocol")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"loaded {model_id}: layers={model.config.num_hidden_layers}, "
        f"hidden={model.config.hidden_size}, parameters={parameter_count}",
        flush=True,
    )
    summary = extract(
        model=model,
        tokenizer=tokenizer,
        rows=rows,
        output_path=output_path,
        max_input_tokens=args.max_input_tokens,
    )
    manifest = {
        "study": run_spec(args.run_dir)["study"],
        "status": "complete",
        "model": model_id,
        "model_commit": observed_commit,
        "parameter_count": parameter_count,
        "chat_template": (
            "single user message; add_generation_prompt=True; enable_thinking=False"
        ),
        "anchor_definition": (
            "mean is computed downstream over both exact symbol subtokens; "
            "artifact preserves both subtoken states separately"
        ),
        "protocol_sha256": file_sha256(
            args.run_dir / "relational_protocol_manifest.json"
        ),
        "tasks_sha256": file_sha256(args.run_dir / "tasks.jsonl"),
        "label_states_sha256": file_sha256(output_path),
        "feature_summary": summary,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "accelerate": package_version("accelerate"),
            "safetensors": package_version("safetensors"),
        },
        "cuda": {
            "name": torch.cuda.get_device_name(0),
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "device_map_request": args.device_map,
            "resolved_device_map": getattr(model, "hf_device_map", None),
        },
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    del protocol, model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
