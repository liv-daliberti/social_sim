#!/usr/bin/env python3
"""Offline, resumable local-vLLM inference for paired context reversal.

Each input unit gets one baseline and three independent updates using that same
baseline. Input metadata never enters model prompts. Importing this module or
using --dry-run does not import vLLM/torch or require a GPU.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import sys
import socket
import tempfile
from typing import Any

CONDITIONS = ("new_news", "no_news", "repeated_news")
CONTEXTS = {"positive", "negative", "broken", "masked"}
PRIOR_TOKEN = "__PRIOR_PROBABILITY__"
# Bounded decimal form enforces the numeric range even with vLLM 0.8.4,
# whose xgrammar JSON-schema backend does not support numeric ranges.
PROBABILITY_REGEX = r'\{"probability": (?:0(?:\.[0-9]{1,6})?|1(?:\.0{1,6})?)\}'
SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def read_units(path: Path) -> list[dict]:
    units = []
    seen = set()
    seen_cells = set()
    required = {"trial_id", "family_id", "domain", "context_id", "repeat", "baseline_prompt", "update_templates", "material_status"}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError(f"Input line {line_number}: missing required unit fields")
        for key in ("trial_id", "family_id", "domain", "baseline_prompt", "material_status"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"Input line {line_number}: invalid {key}")
        if row["context_id"] not in CONTEXTS:
            raise ValueError(f"{row['trial_id']}: unknown context")
        if type(row["repeat"]) is not int or row["repeat"] < 0:
            raise ValueError(f"{row['trial_id']}: repeat must be a nonnegative integer")
        cell = (row["family_id"], row["context_id"], row["repeat"])
        if row["trial_id"] in seen or cell in seen_cells:
            raise ValueError(f"{row['trial_id']}: duplicate trial or family/context/repeat")
        seen.add(row["trial_id"])
        seen_cells.add(cell)
        if PRIOR_TOKEN in row["baseline_prompt"]:
            raise ValueError(f"{row['trial_id']}: baseline cannot contain a prior placeholder")
        updates = row["update_templates"]
        if not isinstance(updates, dict) or set(updates) != set(CONDITIONS):
            raise ValueError(f"{row['trial_id']}: update_templates must contain exactly {CONDITIONS}")
        for condition, prompt in updates.items():
            if not isinstance(prompt, str) or prompt.count(PRIOR_TOKEN) != 1:
                raise ValueError(f"{row['trial_id']}/{condition}: expected exactly one prior placeholder")
        # Check optional frozen hashes when supplied by a caller.
        if "baseline_prompt_sha256" in row and row["baseline_prompt_sha256"] != text_sha256(row["baseline_prompt"]):
            raise ValueError(f"{row['trial_id']}: baseline prompt hash mismatch")
        if "update_template_sha256" in row:
            expected = {key: text_sha256(value) for key, value in updates.items()}
            if row["update_template_sha256"] != expected:
                raise ValueError(f"{row['trial_id']}: update-template hash mismatch")
        units.append(row)
    if not units:
        raise ValueError("Input contains no units")
    return units


def parse_probability(raw: str) -> float:
    # Parsing independently checks the contract rather than trusting the decoder.
    if not re.fullmatch(PROBABILITY_REGEX, raw):
        raise ValueError("Completion does not match the constrained probability JSON format")
    value = json.loads(raw)
    if set(value) != {"probability"} or type(value["probability"]) not in (int, float):
        raise ValueError("Expected exactly one numeric probability field")
    probability = float(value["probability"])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Probability must be finite and in [0, 1]")
    return probability


def render_update(unit: dict, condition: str, probability: float) -> str:
    return unit["update_templates"][condition].replace(PRIOR_TOKEN, json.dumps(probability, allow_nan=False))


def request_seed(base_seed: int, trial_id: str, stage: str, condition: str) -> int:
    key = canonical_json([base_seed, trial_id, stage, condition])
    return int(text_sha256(key)[:16], 16) % (2**31 - 1)


def record_key(record: dict) -> tuple[str, str, str]:
    return record["trial_id"], record["stage"], record["condition"]


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def output_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process owns {path}") from exc
        yield


def inspect_local_model(model_path: Path) -> dict:
    model_path = model_path.expanduser().resolve()
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        raise ValueError(f"--model-path must point to an existing local model snapshot: {model_path}")
    config = json.loads((model_path / "config.json").read_text())
    index = model_path / "model.safetensors.index.json"
    if index.exists():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    elif (model_path / "model.safetensors").exists():
        shards = ["model.safetensors"]
    else:
        raise ValueError(f"Local model has no safetensors checkpoint: {model_path}")
    missing = [name for name in shards if not (model_path / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete local model snapshot: {missing}")
    if not (model_path / "tokenizer_config.json").exists():
        raise ValueError("Local model tokenizer_config.json is missing")
    metadata = ["config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json", "model.safetensors.index.json"]
    return {
        "path": str(model_path),
        "snapshot_revision": model_path.name if model_path.parent.name == "snapshots" else None,
        "model_type": config.get("model_type"),
        "metadata_sha256": {name: file_sha256(model_path / name) for name in metadata if (model_path / name).exists()},
        "weight_shards": len(shards),
        "weight_bytes": sum((model_path / name).stat().st_size for name in shards),
    }


def make_config(args: argparse.Namespace, units: list[dict]) -> dict:
    model = inspect_local_model(args.model_path)
    template_mode = args.chat_template_mode
    if template_mode == "auto" and model["model_type"] == "qwen3":
        template_mode = "qwen-no-thinking"
    return {
        "schema_version": SCHEMA_VERSION,
        "input_path": str(args.input.resolve()),
        "input_sha256": file_sha256(args.input),
        "unit_count": len(units),
        "model_key": args.model_key,
        "local_model": model,
        "runner_sha256": file_sha256(Path(__file__)),
        "decode": {
            "n": 1, "seed": args.seed, "temperature": args.temperature,
            "top_p": 1.0, "max_tokens": args.max_tokens,
            "per_request_seed": "SHA256(canonical JSON [base_seed, trial_id, stage, condition]) first 64 bits modulo 2**31-1",
            "guided_backend": "outlines:no-fallback", "guided_regex": PROBABILITY_REGEX,
            "chat_template_mode": template_mode,
        },
        "engine": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "dtype": args.dtype, "batch_size": args.batch_size,
            "enforce_eager": args.enforce_eager,
            "enable_prefix_caching": True, "disable_custom_all_reduce": True,
            "guided_decoding_backend": "outlines:no-fallback", "vllm_use_v1": False,
        },
        "software": {"python": platform.python_version(), **{name: package_version(name) for name in ("vllm", "transformers", "torch", "outlines")}},
    }


def load_existing(output_path: Path, manifest_path: Path, config: dict, units: list[dict]) -> dict:
    signature = text_sha256(canonical_json(config))
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("run_signature") != signature:
            raise ValueError("Existing run manifest differs from this input/model/configuration/code. Use a new output path.")
    elif output_path.exists() and output_path.stat().st_size:
        raise ValueError("Existing nonempty output has no run manifest; refusing unsafe resume")
    records = {}
    by_id = {unit["trial_id"]: unit for unit in units}
    if not output_path.exists():
        return records
    for line in output_path.read_text().splitlines():
        row = json.loads(line)
        key = record_key(row)
        if key in records or key[0] not in by_id:
            raise ValueError(f"Existing output has duplicate or unknown key {key}")
        if row.get("input_sha256") != config["input_sha256"] or row.get("run_signature") != signature:
            raise ValueError(f"Existing output provenance mismatch for {key}")
        if row.get("model_key") != config["model_key"]:
            raise ValueError(f"Existing output model mismatch for {key}")
        unit = by_id[key[0]]
        for field in ("family_id", "domain", "context_id", "repeat", "material_status"):
            if row.get(field) != unit[field]:
                raise ValueError(f"Existing output metadata mismatch for {key}/{field}")
        if key[1:] != ("baseline", "baseline") and not (key[1] == "update" and key[2] in CONDITIONS):
            raise ValueError(f"Existing output invalid stage/condition {key}")
        if row.get("status") not in {"ok", "parse_error", "generation_error", "blocked_baseline"}:
            raise ValueError(f"Existing output invalid status for {key}")
        if row["status"] == "ok":
            if parse_probability(row["raw"]) != row["probability"]:
                raise ValueError(f"Existing output parsed probability mismatch for {key}")
        elif row.get("probability") is not None:
            raise ValueError(f"Failed output has nonnull probability for {key}")
        records[key] = row
    for key, row in records.items():
        unit = by_id[key[0]]
        if key[1] == "baseline":
            if row["status"] == "blocked_baseline":
                raise ValueError("A baseline cannot be blocked on itself")
            expected_prompt = unit["baseline_prompt"]
        else:
            baseline = records.get((key[0], "baseline", "baseline"))
            if baseline is None:
                raise ValueError(f"Update lacks its baseline: {key}")
            valid_baseline = baseline["status"] == "ok"
            if (row["status"] == "blocked_baseline") == valid_baseline:
                raise ValueError(f"Update/baseline status mismatch for {key}")
            expected_prompt = render_update(unit, key[2], baseline["probability"]) if valid_baseline else None
            if row.get("prior_probability") != baseline["probability"]:
                raise ValueError(f"Update prior does not match its baseline: {key}")
        expected_hash = text_sha256(expected_prompt) if expected_prompt is not None else None
        if row.get("prompt") != expected_prompt or row.get("prompt_sha256") != expected_hash:
            raise ValueError(f"Existing output prompt mismatch for {key}")
    return records


def base_record(unit: dict, stage: str, condition: str, prompt: str | None, config: dict, prior: float | None = None) -> dict:
    return {
        **{key: unit[key] for key in ("trial_id", "family_id", "domain", "context_id", "repeat", "material_status")},
        "model_key": config["model_key"], "stage": stage, "condition": condition,
        "probability": None, "prior_probability": prior,
        "status": "generation_error", "raw": "", "error": None,
        "prompt": prompt, "prompt_sha256": text_sha256(prompt) if prompt is not None else None,
        "input_sha256": config["input_sha256"],
        "run_signature": text_sha256(canonical_json(config)),
        "seed": request_seed(config["decode"]["seed"], unit["trial_id"], stage, condition),
        "created_at": utc_now(),
    }


def describe_hardware() -> dict:
    import torch
    return {"hostname": socket.gethostname(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_gpus": [{"ordinal": index, "name": torch.cuda.get_device_name(index),
                              "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                              "capability": list(torch.cuda.get_device_capability(index))}
                             for index in range(torch.cuda.device_count())]}


def execute(args: argparse.Namespace, units: list[dict], config: dict) -> None:
    manifest_path = Path(str(args.output) + ".manifest.json")
    with output_lock(Path(str(args.output) + ".lock")):
        records = load_existing(args.output, manifest_path, config, units)
        previous_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        manifest = {
            "run_signature": text_sha256(canonical_json(config)), "config": config,
            "started_at": previous_manifest.get("started_at", utc_now()),
            "slurm_job_ids": sorted(set(previous_manifest.get("slurm_job_ids", [])) | ({os.environ["SLURM_JOB_ID"]} if os.environ.get("SLURM_JOB_ID") else set())),
            "status": "running", "expected_records": len(units) * 4,
        }

        def checkpoint(status: str = "running", error: str | None = None) -> None:
            manifest.update({"status": status, "updated_at": utc_now(), "records": len(records),
                             "counts": dict(Counter(row["status"] for row in records.values()))})
            if error:
                manifest["error"] = error
            atomic_write(args.output, "".join(canonical_json(row) + "\n" for row in records.values()))
            atomic_write(manifest_path, json.dumps(manifest, sort_keys=True, indent=2) + "\n")

        # Establish provenance before writing the first response checkpoint.
        atomic_write(manifest_path, json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        if len(records) == len(units) * 4:
            checkpoint("complete" if all(row["status"] == "ok" for row in records.values()) else "complete_with_errors")
            print(f"Verified complete output: {args.output}", flush=True)
            return

        # Force offline operation regardless of inherited shell settings.
        os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false", "VLLM_USE_V1": "0"})
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import GuidedDecodingParams

        try:
            manifest["hardware"] = describe_hardware()
            engine = LLM(model=config["local_model"]["path"], tokenizer=config["local_model"]["path"],
                         trust_remote_code=False, tensor_parallel_size=args.tensor_parallel_size,
                         max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
                         dtype=args.dtype, seed=args.seed, enable_prefix_caching=True,
                         enforce_eager=args.enforce_eager, max_num_seqs=args.batch_size, disable_custom_all_reduce=True,
                         guided_decoding_backend="outlines:no-fallback")
            tokenizer = engine.get_tokenizer()
            manifest["tokenizer_chat_template_sha256"] = text_sha256(canonical_json(tokenizer.chat_template))
            template_kwargs = {"enable_thinking": False} if config["decode"]["chat_template_mode"] == "qwen-no-thinking" else {}

            def generate_batch(requests: list[dict]) -> None:
                prompts, sampling = [], []
                for request in requests:
                    chat = tokenizer.apply_chat_template([{"role": "user", "content": request["prompt"]}], tokenize=False, add_generation_prompt=True, **template_kwargs)
                    tokens = tokenizer.encode(chat, add_special_tokens=False)
                    if len(tokens) + args.max_tokens > args.max_model_len:
                        raise ValueError(f"{record_key(request)}: {len(tokens)} input tokens plus {args.max_tokens} output tokens exceed --max-model-len")
                    request["chat_prompt_sha256"] = text_sha256(chat)
                    request["prompt_token_count"] = len(tokens)
                    prompts.append({"prompt_token_ids": tokens})
                    sampling.append(SamplingParams(n=1, temperature=args.temperature, top_p=1.0,
                                                   max_tokens=args.max_tokens, seed=request["seed"],
                                                   guided_decoding=GuidedDecodingParams(regex=PROBABILITY_REGEX, backend="outlines:no-fallback")))
                generated = engine.generate(prompts, sampling_params=sampling, use_tqdm=False)
                if len(generated) != len(requests):
                    raise RuntimeError("vLLM returned a different number of completions than requests")
                for request, result in zip(requests, generated):
                    request["created_at"] = utc_now()
                    request["response_received"] = bool(result.outputs)
                    if result.outputs:
                        completion = result.outputs[0]
                        request.update({"raw": completion.text, "finish_reason": completion.finish_reason,
                                        "stop_reason": completion.stop_reason, "output_token_count": len(completion.token_ids)})
                        try:
                            request["probability"] = parse_probability(completion.text)
                            request["status"] = "ok"
                        except (ValueError, TypeError, json.JSONDecodeError) as exc:
                            request["status"] = "parse_error"
                            request["error"] = str(exc)
                    else:
                        request["error"] = "vLLM returned no completion"
                    records[record_key(request)] = request
                checkpoint()
                print(f"Checkpoint: {len(records)}/{len(units) * 4} records; {manifest['counts']}", flush=True)

            baselines = [base_record(unit, "baseline", "baseline", unit["baseline_prompt"], config) for unit in units]
            random.Random(args.seed).shuffle(baselines)
            baselines = [row for row in baselines if record_key(row) not in records]
            for start in range(0, len(baselines), args.batch_size):
                generate_batch(baselines[start:start + args.batch_size])

            updates = []
            for unit in units:
                baseline = records[(unit["trial_id"], "baseline", "baseline")]
                for condition in CONDITIONS:
                    prior = baseline["probability"]
                    prompt = render_update(unit, condition, prior) if baseline["status"] == "ok" else None
                    request = base_record(unit, "update", condition, prompt, config, prior)
                    if baseline["status"] != "ok":
                        if record_key(request) in records:
                            continue
                        request.update({"status": "blocked_baseline", "error": "Context-specific baseline has no valid probability", "response_received": False,
                                        "prompt_template": unit["update_templates"][condition],
                                        "prompt_template_sha256": text_sha256(unit["update_templates"][condition])})
                        records[record_key(request)] = request
                    else:
                        updates.append(request)
            checkpoint()
            random.Random(args.seed + 1).shuffle(updates)
            updates = [row for row in updates if record_key(row) not in records]
            for start in range(0, len(updates), args.batch_size):
                generate_batch(updates[start:start + args.batch_size])
            if len(records) != len(units) * 4:
                raise RuntimeError("Missing final stage records")
            status = "complete" if all(row["status"] == "ok" for row in records.values()) else "complete_with_errors"
            checkpoint(status)
            print(f"{status}: {args.output}", flush=True)
        except BaseException as exc:
            checkpoint("interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed", f"{type(exc).__name__}: {exc}")
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path", type=Path, required=True, help="Complete existing local snapshot; repository IDs are rejected")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--chat-template-mode", choices=("auto", "qwen-no-thinking"), default="auto")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate input/model/resume contract on CPU; no model imports or writes")
    args = parser.parse_args()
    if min(args.tensor_parallel_size, args.batch_size, args.max_tokens, args.max_model_len) < 1:
        parser.error("Parallelism, batch size and token limits must be positive")
    if not 0 < args.gpu_memory_utilization < 1 or args.temperature < 0 or not math.isfinite(args.temperature):
        parser.error("Invalid memory utilization or temperature")
    if args.output.resolve() == args.input.resolve():
        parser.error("Input and output must be different files")
    units = read_units(args.input)
    config = make_config(args, units)
    if args.dry_run:
        existing = load_existing(args.output, Path(str(args.output) + ".manifest.json"), config, units)
        print(json.dumps({"status": "dry_run_valid", "config": config,
                          "run_signature": text_sha256(canonical_json(config)),
                          "families": len({u['family_id'] for u in units}),
                          "contexts": dict(Counter(u['context_id'] for u in units)),
                          "expected_records": len(units) * 4, "existing_records": len(existing)}, sort_keys=True, indent=2))
        return
    execute(args, units, config)


if __name__ == "__main__":
    main()
