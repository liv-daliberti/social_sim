#!/usr/bin/env python3
"""Offline resumable vLLM direction classification; three independent chats per unit."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import re

from exp1_prospective.context_reversal import run_local as common
from exp1_prospective.context_reversal import direction_design as design

DIRECTION_REGEX = r'\{"direction": "(?:increase|decrease|unchanged|unclear)"\}'
STATUSES = {"ok", "parse_error", "generation_error"}


def parse_direction(raw: str) -> str:
    if not isinstance(raw, str) or not re.fullmatch(DIRECTION_REGEX, raw):
        raise ValueError("Completion does not match the constrained direction JSON format")
    return json.loads(raw)["direction"]


def make_config(args: argparse.Namespace, units: list[dict]) -> dict:
    config = common.make_config(args, units)
    config.update({"schema_version": "direction_companion_run_v1", "protocol": design.PROTOCOL,
                   "runner_sha256": common.file_sha256(Path(__file__)),
                   "helper_sha256": {path.name: common.file_sha256(path) for path in (Path(common.__file__), Path(design.__file__))},
                   "parent_plan_sha256": units[0]["parent_plan_sha256"]})
    config["decode"]["guided_regex"] = DIRECTION_REGEX
    return config


def base_record(unit: dict, condition: str, config: dict) -> dict:
    prompt = unit["direction_prompts"][condition]
    return {
        **{key: unit[key] for key in design.METADATA},
        "model_key": config["model_key"], "stage": "direction", "condition": condition,
        "direction": None, "status": "generation_error", "raw": "", "error": None,
        "prompt": prompt, "prompt_sha256": common.text_sha256(prompt),
        "input_sha256": config["input_sha256"], "parent_plan_sha256": unit["parent_plan_sha256"],
        "run_signature": common.text_sha256(common.canonical_json(config)),
        "seed": common.request_seed(config["decode"]["seed"], unit["trial_id"], "direction", condition),
        "created_at": common.utc_now(),
    }


def load_existing(output: Path, manifest_path: Path, config: dict, units: list[dict]) -> dict:
    signature = common.text_sha256(common.canonical_json(config))
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("run_signature") != signature or manifest.get("config") != config:
            raise ValueError("Existing direction manifest differs from input/model/configuration/code; use a new output")
    elif output.exists() and output.stat().st_size:
        raise ValueError("Existing nonempty direction output has no manifest; refusing unsafe resume")
    records, by_id = {}, {u["trial_id"]: u for u in units}
    if not output.exists():
        return records
    for line in output.read_text().splitlines():
        row = json.loads(line)
        if not isinstance(row, dict) or not {"trial_id", "stage", "condition"} <= row.keys():
            raise ValueError("Malformed direction response")
        key = common.record_key(row)
        if key in records or key[0] not in by_id or key[1] != "direction" or key[2] not in design.CONDITIONS:
            raise ValueError(f"Duplicate or unknown direction response key: {key}")
        expected = base_record(by_id[key[0]], key[2], config)
        for field in (*design.METADATA, "model_key", "input_sha256", "parent_plan_sha256", "run_signature", "seed", "prompt", "prompt_sha256"):
            if row.get(field) != expected[field]:
                raise ValueError(f"Existing direction response {field} mismatch for {key}")
        if row.get("status") not in STATUSES:
            raise ValueError(f"Unknown direction status for {key}")
        if row["status"] == "ok":
            if parse_direction(row.get("raw")) != row.get("direction"):
                raise ValueError(f"Parsed direction mismatch for {key}")
        elif row.get("direction") is not None:
            raise ValueError(f"Failed response has a nonnull direction for {key}")
        if "probability" in row or "prior_probability" in row:
            raise ValueError("Numeric forecast field in direction response")
        records[key] = row
    return records


def execute(args: argparse.Namespace, units: list[dict], config: dict) -> None:
    manifest_path = Path(str(args.output) + ".manifest.json")
    with common.output_lock(Path(str(args.output) + ".lock")):
        records = load_existing(args.output, manifest_path, config, units)
        previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        manifest = {
            **previous, "run_signature": common.text_sha256(common.canonical_json(config)), "config": config,
            "started_at": previous.get("started_at", common.utc_now()), "status": "running",
            "expected_records": len(units) * 3,
            "slurm_job_ids": sorted(set(previous.get("slurm_job_ids", [])) | ({os.environ["SLURM_JOB_ID"]} if os.environ.get("SLURM_JOB_ID") else set())),
        }

        def checkpoint(status: str = "running", error: str | None = None) -> None:
            manifest.update({"status": status, "updated_at": common.utc_now(), "records": len(records),
                             "counts": dict(Counter(row["status"] for row in records.values()))})
            manifest.pop("error", None)
            if error is not None:
                manifest["error"] = error
            common.atomic_write(args.output, "".join(common.canonical_json(row) + "\n" for row in records.values()))
            common.atomic_write(manifest_path, json.dumps(manifest, sort_keys=True, indent=2) + "\n")

        common.atomic_write(manifest_path, json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        if len(records) == len(units) * 3:
            checkpoint("complete" if all(r["status"] == "ok" for r in records.values()) else "complete_with_errors")
            print(f"Verified complete direction output: {args.output}", flush=True)
            return
        os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                           "TOKENIZERS_PARALLELISM": "false", "VLLM_USE_V1": "0"})
        try:
            from vllm import LLM, SamplingParams
            from vllm.sampling_params import GuidedDecodingParams
            manifest["hardware"] = common.describe_hardware()
            engine = LLM(model=config["local_model"]["path"], tokenizer=config["local_model"]["path"],
                         trust_remote_code=False, tensor_parallel_size=args.tensor_parallel_size,
                         max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
                         dtype=args.dtype, seed=args.seed, enable_prefix_caching=True,
                         enforce_eager=args.enforce_eager, max_num_seqs=args.batch_size,
                         disable_custom_all_reduce=True, guided_decoding_backend="outlines:no-fallback")
            tokenizer = engine.get_tokenizer()
            template_hash = common.text_sha256(common.canonical_json(tokenizer.chat_template))
            if previous.get("tokenizer_chat_template_sha256", template_hash) != template_hash:
                raise ValueError("Tokenizer chat template changed during resume")
            manifest["tokenizer_chat_template_sha256"] = template_hash
            template_kwargs = {"enable_thinking": False} if config["decode"]["chat_template_mode"] == "qwen-no-thinking" else {}
            requests = [base_record(unit, condition, config) for unit in units for condition in design.CONDITIONS]
            random.Random(args.seed).shuffle(requests)
            requests = [row for row in requests if common.record_key(row) not in records]
            for start in range(0, len(requests), args.batch_size):
                batch, prompts, sampling = requests[start:start + args.batch_size], [], []
                for request in batch:
                    chat = tokenizer.apply_chat_template([{"role": "user", "content": request["prompt"]}], tokenize=False,
                                                         add_generation_prompt=True, **template_kwargs)
                    tokens = tokenizer.encode(chat, add_special_tokens=False)
                    if len(tokens) + args.max_tokens > args.max_model_len:
                        raise ValueError(f"{common.record_key(request)} exceeds --max-model-len")
                    request.update({"chat_prompt_sha256": common.text_sha256(chat), "prompt_token_count": len(tokens)})
                    prompts.append({"prompt_token_ids": tokens})
                    sampling.append(SamplingParams(n=1, temperature=args.temperature, top_p=1.0,
                                                   max_tokens=args.max_tokens, seed=request["seed"],
                                                   guided_decoding=GuidedDecodingParams(regex=DIRECTION_REGEX, backend="outlines:no-fallback")))
                generated = engine.generate(prompts, sampling_params=sampling, use_tqdm=False)
                if len(generated) != len(batch):
                    raise RuntimeError("vLLM returned a different number of completions than requests")
                for request, result in zip(batch, generated):
                    request.update({"created_at": common.utc_now(), "response_received": bool(result.outputs)})
                    if result.outputs:
                        completion = result.outputs[0]
                        request.update({"raw": completion.text, "finish_reason": completion.finish_reason,
                                        "stop_reason": completion.stop_reason, "output_token_count": len(completion.token_ids)})
                        try:
                            request["direction"] = parse_direction(completion.text)
                            request["status"] = "ok"
                        except ValueError as exc:
                            request.update({"status": "parse_error", "error": str(exc)})
                    else:
                        request["error"] = "vLLM returned no completion"
                    records[common.record_key(request)] = request
                checkpoint()
                print(f"Direction checkpoint: {len(records)}/{len(units) * 3}; {manifest['counts']}", flush=True)
            if len(records) != len(units) * 3:
                raise RuntimeError("Missing final direction records")
            checkpoint("complete" if all(r["status"] == "ok" for r in records.values()) else "complete_with_errors")
        except BaseException as exc:
            checkpoint("interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed", f"{type(exc).__name__}: {exc}")
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.90)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--temperature", type=float, default=.7)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--chat-template-mode", choices=("auto", "qwen-no-thinking"), default="auto")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate on CPU without importing vLLM/torch or writing files")
    args = parser.parse_args()
    if min(args.tensor_parallel_size, args.batch_size, args.max_tokens, args.max_model_len) < 1:
        parser.error("Parallelism, batch size and token limits must be positive")
    if not 0 < args.gpu_memory_utilization < 1 or args.temperature < 0 or not math.isfinite(args.temperature):
        parser.error("Invalid memory utilization or temperature")
    if args.output.resolve() == args.input.resolve():
        parser.error("Input and output must be different files")
    units = design.read_units(args.input)
    config = make_config(args, units)
    if args.dry_run:
        existing = load_existing(args.output, Path(str(args.output) + ".manifest.json"), config, units)
        print(json.dumps({"status": "dry_run_valid", "config": config,
                          "run_signature": common.text_sha256(common.canonical_json(config)),
                          "families": len({u['family_id'] for u in units}),
                          "contexts": dict(Counter(u["context_id"] for u in units)),
                          "expected_records": len(units) * 3, "existing_records": len(existing)}, indent=2, sort_keys=True))
        return
    execute(args, units, config)


if __name__ == "__main__":
    main()
