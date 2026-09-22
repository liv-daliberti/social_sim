#!/usr/bin/env python3
"""Separate, offline Qwen3 thinking/unconstrained-control follow-up.

Reuses the original compiled prompts, context-specific baselines, three independent
update branches, and request seeds. Both modes decode without a token grammar.
Enabled thinking requires a closing </think>, followed by
one strict probability JSON object. Every attempted completion, including an
invalid or truncated completion, is terminal and is never automatically retried.
Import and --dry-run are CPU-only; neither imports vLLM/torch nor writes output.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import platform
import random
import re
from typing import Any

try:
    from . import run_local as common
except ImportError:  # Direct script invocation, including an archived code copy.
    import run_local as common

CONDITIONS = common.CONDITIONS
STATUSES = {"ok", "parse_error", "generation_error", "blocked_baseline"}
SCHEMA_VERSION = 1
PARSER_VERSION = "strict-json-after-thinking-close-v1"
DIAGNOSTIC_FIELDS = (
    "status", "probability", "failure_kind", "error", "reasoning_present",
    "reasoning_closed", "reasoning_empty", "reasoning_open_in_completion", "reasoning_char_count",
    "final_output", "final_output_char_count", "raw_output_char_count", "truncated",
)


def strict_probability(text: str) -> float:
    """Accept only one JSON object, with one finite numeric probability field."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Non-JSON numeric constant: {value}")

    value = json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)
    if not isinstance(value, dict) or set(value) != {"probability"}:
        raise ValueError("Expected exactly one probability field in a JSON object")
    if type(value["probability"]) not in (int, float):
        raise ValueError("Probability must be numeric, not a boolean or string")
    probability = float(value["probability"])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Probability must be finite and in [0, 1]")
    return probability


def inspect_completion(raw: str, thinking: str, finish_reason: str | None,
                       *, reasoning_open_in_prompt: bool = False) -> dict:
    """Split at the reasoning boundary; never search thoughts for a probability.

    The enabled chat template may provide <think> itself. When it does not, the
    completion must supply it. A length-terminated response is always a failure,
    even if its last sampled token happens to complete valid JSON.
    """
    if thinking not in {"enabled", "disabled"}:
        raise ValueError("Unknown thinking mode")
    reasoning = ""
    final = raw.strip() if thinking == "disabled" else None
    framing_error = None
    has_open = raw.lstrip().startswith("<think>")
    closed = "</think>" in raw
    if thinking == "enabled":
        before, separator, after = raw.partition("</think>")
        reasoning = before.strip()
        if reasoning.startswith("<think>"):
            reasoning = reasoning[len("<think>"):].strip()
        if separator:
            final = after.strip()
        if not separator:
            framing_error = ("missing_reasoning_close", "Enabled thinking did not produce </think>")
        elif not has_open and not reasoning_open_in_prompt:
            framing_error = ("missing_reasoning_open", "Neither prompt nor completion opens <think>")
        elif "<think>" in reasoning or "</think>" in after or "<think>" in after:
            framing_error = ("invalid_reasoning_frame", "Unexpected repeated or nested thinking marker")
    result = {
        "status": "parse_error", "probability": None, "failure_kind": None, "error": None,
        "reasoning_present": bool(reasoning), "reasoning_closed": closed,
        "reasoning_empty": thinking == "enabled" and closed and not reasoning,
        "reasoning_open_in_completion": has_open, "reasoning_char_count": len(reasoning),
        "final_output": final, "final_output_char_count": len(final) if final is not None else 0,
        "raw_output_char_count": len(raw), "truncated": finish_reason == "length",
    }
    if result["truncated"]:
        result.update(failure_kind="truncated", error="vLLM exhausted the configured output token budget")
    elif framing_error:
        result.update(failure_kind=framing_error[0], error=framing_error[1])
    elif finish_reason != "stop":
        result.update(status="generation_error", failure_kind="unexpected_finish_reason",
                      error=f"Unexpected vLLM finish reason: {finish_reason!r}")
    else:
        try:
            result.update(probability=strict_probability(final), status="ok")
        except (ValueError, TypeError, OverflowError) as exc:
            result.update(failure_kind="invalid_final_json", error=str(exc))
    return result


def make_config(args: argparse.Namespace, units: list[dict]) -> dict:
    model = common.inspect_local_model(args.model_path)
    if model["model_type"] != "qwen3":
        raise ValueError("This follow-up requires a local Qwen3 model snapshot")
    return {
        "schema_version": SCHEMA_VERSION, "experiment": "qwen3_unconstrained_thinking_followup",
        "input_path": str(args.input.resolve()), "input_sha256": common.file_sha256(args.input),
        "unit_count": len(units), "model_key": args.model_key, "local_model": model,
        "runner_sha256": common.file_sha256(Path(__file__)),
        "helper_runner_sha256": common.file_sha256(Path(common.__file__)),
        "parser_version": PARSER_VERSION,
        "decode": {"n": 1, "seed": args.seed, "temperature": args.temperature, "top_p": 1.0,
                   "max_tokens": args.max_tokens, "thinking": args.thinking,
                   "enable_thinking": args.thinking == "enabled", "guided_decoding": None,
                   "skip_special_tokens": False,
                   "per_request_seed": "SHA256(canonical JSON [base_seed, trial_id, stage, condition]) first 64 bits modulo 2**31-1"},
        "engine": {"tensor_parallel_size": args.tensor_parallel_size, "max_model_len": args.max_model_len,
                   "gpu_memory_utilization": args.gpu_memory_utilization, "dtype": args.dtype,
                   "batch_size": args.batch_size, "enforce_eager": args.enforce_eager,
                   "enable_prefix_caching": True, "disable_custom_all_reduce": True, "vllm_use_v1": False},
        "software": {"python": platform.python_version(), **{name: common.package_version(name)
                     for name in ("vllm", "transformers", "torch")}},
    }


def base_record(unit: dict, stage: str, condition: str, prompt: str | None,
                config: dict, prior: float | None = None) -> dict:
    row = common.base_record(unit, stage, condition, prompt, config, prior)
    row.update(thinking=config["decode"]["thinking"], enable_thinking=config["decode"]["enable_thinking"],
               guided_decoding=None, max_tokens=config["decode"]["max_tokens"],
               max_model_len=config["engine"]["max_model_len"], failure_kind=None,
               response_received=False, final_output=None, final_output_char_count=0,
               raw_output_char_count=0, reasoning_present=False, reasoning_closed=False, reasoning_empty=False,
               reasoning_open_in_completion=False, reasoning_open_in_prompt=False,
               reasoning_char_count=0, truncated=False, prompt_token_count=None,
               output_token_count=0, total_token_count=None, finish_reason=None, stop_reason=None)
    return row


def load_existing(output_path: Path, manifest_path: Path, config: dict, units: list[dict]) -> dict:
    signature = common.text_sha256(common.canonical_json(config))
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("run_signature") != signature or previous.get("config") != config:
            raise ValueError("Existing run manifest differs from input/model/configuration/code; use a new output path")
    elif output_path.exists() and output_path.stat().st_size:
        raise ValueError("Existing nonempty output has no run manifest; refusing unsafe resume")
    if not output_path.exists():
        return {}
    records, by_id = {}, {u["trial_id"]: u for u in units}
    for line in output_path.read_text().splitlines():
        row = json.loads(line)
        key = common.record_key(row)
        if key in records or key[0] not in by_id:
            raise ValueError(f"Existing output has duplicate or unknown key {key}")
        if key[1:] != ("baseline", "baseline") and not (key[1] == "update" and key[2] in CONDITIONS):
            raise ValueError(f"Existing output invalid stage/condition {key}")
        for field, expected in {"run_signature": signature, "input_sha256": config["input_sha256"],
                                "model_key": config["model_key"], "thinking": config["decode"]["thinking"],
                                "enable_thinking": config["decode"]["enable_thinking"], "guided_decoding": None,
                                "max_tokens": config["decode"]["max_tokens"], "max_model_len": config["engine"]["max_model_len"],
                                "seed": common.request_seed(config["decode"]["seed"], *key)}.items():
            if field not in row or row[field] != expected:
                raise ValueError(f"Existing output provenance mismatch for {key}/{field}")
        for field in ("family_id", "domain", "context_id", "repeat", "material_status"):
            if row.get(field) != by_id[key[0]][field]:
                raise ValueError(f"Existing output metadata mismatch for {key}/{field}")
        if row.get("status") not in STATUSES:
            raise ValueError(f"Existing output invalid status for {key}")
        if row.get("response_received"):
            expected = inspect_completion(row["raw"], config["decode"]["thinking"], row["finish_reason"],
                                          reasoning_open_in_prompt=row["reasoning_open_in_prompt"])
            for field in DIAGNOSTIC_FIELDS:
                if row.get(field) != expected[field]:
                    raise ValueError(f"Existing output completion mismatch for {key}/{field}")
            if (type(row.get("output_token_count")) is not int or not 0 <= row["output_token_count"] <= config["decode"]["max_tokens"]
                    or type(row.get("prompt_token_count")) is not int or row["prompt_token_count"] < 1
                    or row.get("total_token_count") != row["prompt_token_count"] + row["output_token_count"]):
                raise ValueError(f"Existing output token usage mismatch for {key}")
        elif row["status"] not in {"generation_error", "blocked_baseline"} or row.get("probability") is not None:
            raise ValueError(f"Existing output without a completion has invalid outcome for {key}")
        records[key] = row
    for key, row in records.items():
        unit = by_id[key[0]]
        if key[1] == "baseline":
            if row["status"] == "blocked_baseline" or row.get("prior_probability") is not None:
                raise ValueError("Baseline has invalid prior or blocked status")
            expected_prompt = unit["baseline_prompt"]
        else:
            baseline = records.get((key[0], "baseline", "baseline"))
            if baseline is None:
                raise ValueError(f"Update lacks its own context baseline: {key}")
            valid = baseline["status"] == "ok"
            if (row["status"] == "blocked_baseline") == valid:
                raise ValueError(f"Update/baseline status mismatch for {key}")
            if row.get("prior_probability") != baseline["probability"]:
                raise ValueError(f"Update prior does not match its own context baseline: {key}")
            expected_prompt = common.render_update(unit, key[2], baseline["probability"]) if valid else None
            if not valid and (row.get("prompt_template") != unit["update_templates"][key[2]] or
                              row.get("prompt_template_sha256") != common.text_sha256(unit["update_templates"][key[2]])):
                raise ValueError(f"Blocked update template mismatch for {key}")
        if row.get("prompt") != expected_prompt or row.get("prompt_sha256") != (common.text_sha256(expected_prompt) if expected_prompt is not None else None):
            raise ValueError(f"Existing output prompt mismatch for {key}")
    return records


def render_chat(tokenizer: Any, prompt: str, thinking: str) -> tuple[str, bool]:
    messages = [{"role": "user", "content": prompt}]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=thinking == "enabled")
    if "<|im_start|>assistant\n" not in chat:
        raise ValueError("Qwen3 chat template lacks an assistant generation prefix")
    suffix = chat.rsplit("<|im_start|>assistant\n", 1)[1]
    if thinking == "enabled":
        if suffix.strip() not in {"", "<think>"}:
            raise ValueError("Enabled thinking template unexpectedly suppresses or prefills reasoning")
        opposite = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
        if opposite == chat or not re.fullmatch(r"\s*<think>\s*</think>\s*", opposite.rsplit("<|im_start|>assistant\n", 1)[-1]):
            raise ValueError("Tokenizer does not demonstrate that enable_thinking changes the Qwen3 prefix")
    elif not re.fullmatch(r"\s*<think>\s*</think>\s*", suffix):
        raise ValueError("Disabled thinking template does not contain the empty Qwen3 thinking prefix")
    return chat, suffix.strip() == "<think>"


def execute(args: argparse.Namespace, units: list[dict], config: dict) -> None:
    manifest_path = Path(str(args.output) + ".manifest.json")
    with common.output_lock(Path(str(args.output) + ".lock")):
        records = load_existing(args.output, manifest_path, config, units)
        previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        manifest = {"run_signature": common.text_sha256(common.canonical_json(config)), "config": config,
                    "started_at": previous.get("started_at", common.utc_now()), "status": "running",
                    "slurm_job_ids": sorted(set(previous.get("slurm_job_ids", [])) | ({os.environ["SLURM_JOB_ID"]} if os.environ.get("SLURM_JOB_ID") else set())),
                    "expected_records": len(units) * 4}
        for field in ("hardware", "tokenizer_chat_template_sha256"):
            if field in previous:
                manifest[field] = previous[field]

        def checkpoint(status="running", error=None):
            manifest.update(status=status, updated_at=common.utc_now(), records=len(records),
                            counts=dict(Counter(r["status"] for r in records.values())),
                            failure_counts=dict(Counter(r["failure_kind"] for r in records.values() if r.get("failure_kind"))),
                            reasoning_present_count=sum(r["reasoning_present"] for r in records.values()),
                            reasoning_closed_count=sum(r["reasoning_closed"] for r in records.values()),
                            reasoning_empty_count=sum(r["reasoning_empty"] for r in records.values()),
                            total_output_tokens=sum(r["output_token_count"] for r in records.values()))
            if error:
                manifest["error"] = error
            common.atomic_write(args.output, "".join(common.canonical_json(r) + "\n" for r in records.values()))
            common.atomic_write(manifest_path, json.dumps(manifest, sort_keys=True, indent=2) + "\n")

        common.atomic_write(manifest_path, json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        if len(records) == len(units) * 4:
            checkpoint("complete" if all(r["status"] == "ok" for r in records.values()) else "complete_with_errors")
            print(f"Verified complete output: {args.output}", flush=True)
            return
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                          TOKENIZERS_PARALLELISM="false", VLLM_USE_V1="0")
        try:
            from vllm import LLM, SamplingParams
            manifest["hardware"] = common.describe_hardware()
            engine = LLM(model=config["local_model"]["path"], tokenizer=config["local_model"]["path"],
                         trust_remote_code=False, tensor_parallel_size=args.tensor_parallel_size,
                         max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
                         dtype=args.dtype, seed=args.seed, enable_prefix_caching=True,
                         enforce_eager=args.enforce_eager, max_num_seqs=args.batch_size, disable_custom_all_reduce=True)
            tokenizer = engine.get_tokenizer()
            template_hash = common.text_sha256(common.canonical_json(tokenizer.chat_template))
            if previous.get("tokenizer_chat_template_sha256", template_hash) != template_hash:
                raise ValueError("Tokenizer chat template changed since the earlier run")
            manifest["tokenizer_chat_template_sha256"] = template_hash
            # Validate the actual effective chat prefix before any generation.
            render_chat(tokenizer, units[0]["baseline_prompt"], args.thinking)

            def generate_batch(requests):
                prompts, sampling = [], []
                for request in requests:
                    chat, open_in_prompt = render_chat(tokenizer, request["prompt"], args.thinking)
                    tokens = tokenizer.encode(chat, add_special_tokens=False)
                    if len(tokens) + args.max_tokens > args.max_model_len:
                        raise ValueError(f"{common.record_key(request)}: {len(tokens)} input tokens plus {args.max_tokens} output tokens exceed --max-model-len")
                    request.update(chat_prompt_sha256=common.text_sha256(chat), prompt_token_count=len(tokens),
                                   reasoning_open_in_prompt=open_in_prompt)
                    prompts.append({"prompt_token_ids": tokens})
                    # Retain thinking delimiters. EOS is not part of the decoded text
                    # unless explicitly requested via include_stop_str_in_output.
                    sampling.append(SamplingParams(n=1, temperature=args.temperature, top_p=1.0,
                                                   max_tokens=args.max_tokens, seed=request["seed"],
                                                   skip_special_tokens=False))
                generated = engine.generate(prompts, sampling_params=sampling, use_tqdm=False)
                if len(generated) != len(requests):
                    raise RuntimeError("vLLM returned a different number of completions than requests")
                for request, result in zip(requests, generated):
                    request["created_at"] = common.utc_now()
                    request["response_received"] = bool(result.outputs)
                    if len(result.outputs) > 1:
                        raise RuntimeError("vLLM returned multiple completions for n=1")
                    if result.outputs:
                        completion = result.outputs[0]
                        request.update(raw=completion.text, finish_reason=completion.finish_reason,
                                       stop_reason=completion.stop_reason, output_token_count=len(completion.token_ids),
                                       total_token_count=request["prompt_token_count"] + len(completion.token_ids))
                        request.update(inspect_completion(completion.text, args.thinking, completion.finish_reason,
                                       reasoning_open_in_prompt=request["reasoning_open_in_prompt"]))
                    else:
                        request.update(error="vLLM returned no completion", failure_kind="no_completion")
                    records[common.record_key(request)] = request
                checkpoint()
                print(f"Checkpoint: {len(records)}/{len(units) * 4} records; {manifest['counts']}; {manifest['failure_counts']}", flush=True)

            baselines = [base_record(u, "baseline", "baseline", u["baseline_prompt"], config) for u in units]
            random.Random(args.seed).shuffle(baselines)
            baselines = [r for r in baselines if common.record_key(r) not in records]
            for start in range(0, len(baselines), args.batch_size):
                generate_batch(baselines[start:start + args.batch_size])
            updates = []
            for unit in units:
                baseline = records[(unit["trial_id"], "baseline", "baseline")]
                for condition in CONDITIONS:
                    prior = baseline["probability"]
                    prompt = common.render_update(unit, condition, prior) if baseline["status"] == "ok" else None
                    request = base_record(unit, "update", condition, prompt, config, prior)
                    if common.record_key(request) in records:
                        continue
                    if baseline["status"] != "ok":
                        request.update(status="blocked_baseline", error="Context-specific baseline has no valid probability",
                                       failure_kind="blocked_baseline", prompt_template=unit["update_templates"][condition],
                                       prompt_template_sha256=common.text_sha256(unit["update_templates"][condition]))
                        records[common.record_key(request)] = request
                    else:
                        updates.append(request)
            checkpoint()
            random.Random(args.seed + 1).shuffle(updates)
            for start in range(0, len(updates), args.batch_size):
                generate_batch(updates[start:start + args.batch_size])
            if len(records) != len(units) * 4:
                raise RuntimeError("Missing final stage records")
            status = "complete" if all(r["status"] == "ok" for r in records.values()) else "complete_with_errors"
            checkpoint(status)
            print(f"{status}: {args.output}", flush=True)
        except BaseException as exc:
            checkpoint("interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed", f"{type(exc).__name__}: {exc}")
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thinking", choices=("enabled", "disabled"), required=True)
    parser.add_argument("--tensor-parallel-size", "--tp", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.90)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--temperature", type=float, default=.7)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="CPU validation of inputs/model/resume; no model imports or writes")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if min(args.tensor_parallel_size, args.batch_size, args.max_tokens, args.max_model_len) < 1:
        parser.error("Parallelism, batch size and token limits must be positive")
    if not 0 < args.gpu_memory_utilization < 1 or args.temperature < 0 or not math.isfinite(args.temperature):
        parser.error("Invalid memory utilization or temperature")
    if args.max_tokens >= args.max_model_len:
        parser.error("Output token budget must leave room for the input prompt")
    if args.output.resolve() == args.input.resolve():
        parser.error("Input and output must be different files")
    units = common.read_units(args.input)
    config = make_config(args, units)
    if args.dry_run:
        records = load_existing(args.output, Path(str(args.output) + ".manifest.json"), config, units)
        print(json.dumps({"status": "dry_run_valid", "config": config,
                          "run_signature": common.text_sha256(common.canonical_json(config)),
                          "families": len({u["family_id"] for u in units}),
                          "contexts": dict(Counter(u["context_id"] for u in units)),
                          "expected_records": len(units) * 4, "existing_records": len(records)}, sort_keys=True, indent=2))
    else:
        execute(args, units, config)


if __name__ == "__main__":
    main()
