#!/usr/bin/env python3
"""Greedy locked-test evaluation for the preregistered 8B model-roster extension."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import vllm
from vllm.lora.request import LoRARequest

import evaluate_locked_test as registered

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "exp3b_registered"
BASELINES = ROOT / "reports" / "exp3b_baselines.json"


def format_prompt(tokenizer, text: str, template: str) -> str:
    if template == "biased_news":
        return registered.prompt(text)
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if template == "auto_no_think":
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template([{"role": "user", "content": text}], **kwargs)


def resolve_adapter(model_key: str, job_id: str) -> Path:
    roots = sorted((ROOT / "reports").glob(f"train_{model_key}_market_*_j{job_id}"))
    if len(roots) != 1:
        raise AssertionError(f"expected one {model_key} report for job {job_id}, got {roots}")
    adapters = sorted(roots[0].glob("debug_*/saved_models/step_*"))
    if not adapters or not (adapters[-1] / "adapter_config.json").is_file():
        raise AssertionError(f"no valid final adapter for job {job_id}")
    return adapters[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapters", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--template", choices=("biased_news", "auto", "auto_no_think"),
                        required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    seed_jobs = dict((int(seed), job) for seed, job in
                     (value.split(":", 1) for value in args.adapters.split(";")))
    if sorted(seed_jobs) != [42, 43, 44]:
        raise AssertionError("locked evaluation requires seeds 42, 43, and 44")

    rows = registered.read_jsonl(DATA / "test.tasks.jsonl")
    sampling = vllm.SamplingParams(temperature=0.0, max_tokens=128)
    llm = vllm.LLM(model=args.model, enable_lora=True, max_lora_rank=32,
                   max_model_len=1920, gpu_memory_utilization=0.90,
                   trust_remote_code=True)
    prompts = [format_prompt(llm.get_tokenizer(), row["input"], args.template) for row in rows]
    forecasts, responses = {}, {}
    forecasts["base"], responses["base"] = registered.generate(llm, prompts, sampling)
    adapter_paths = {}
    for adapter_id, seed in enumerate(sorted(seed_jobs), start=1):
        path = resolve_adapter(args.model_key, seed_jobs[seed])
        adapter_paths[str(seed)] = str(path)
        request = LoRARequest(f"exp3b_{args.model_key}_seed{seed}", adapter_id, str(path))
        forecasts[f"seed_{seed}"], responses[f"seed_{seed}"] = registered.generate(
            llm, prompts, sampling, request
        )

    baseline_report = json.loads(BASELINES.read_text())
    platt = baseline_report["platt_parameters"]
    market = [float(row["market_yes_prob"]) for row in rows]
    calibrated = [registered.sigmoid(float(platt["intercept"]) +
                    float(platt["market_logit_slope"]) * registered.logit(value))
                  for value in market]
    labels = [bool(row["settlement_yes"]) for row in rows]
    model_losses = {name: [registered.losses(p, y)[0] for p, y in zip(values, labels)]
                    for name, values in forecasts.items()}
    market_losses = [registered.losses(p, y)[0] for p, y in zip(market, labels)]
    calibrated_losses = [registered.losses(p, y)[0] for p, y in zip(calibrated, labels)]
    trained_names = [f"seed_{seed}" for seed in sorted(seed_jobs)]
    trained_mean = [sum(model_losses[name][i] for name in trained_names) / 3
                    for i in range(len(rows))]
    summary = {
        "protocol_version": "exp3b_model_extension_v1",
        "frozen_parent_protocol": "exp3b_registered_v1",
        "test_was_used_during_training": False,
        "model": args.model, "model_key": args.model_key,
        "decoding": {"temperature": 0.0, "max_tokens": 128},
        "adapter_paths": adapter_paths,
        "models": {name: registered.summarize(values, rows)
                   for name, values in forecasts.items()},
        "baselines": {"market": registered.summarize(market, rows),
                      "platt_market_train_only": registered.summarize(calibrated, rows)},
        "comparisons_brier": {
            "trained_seed_mean_minus_base": registered.cluster_bootstrap_delta(
                rows, trained_mean, model_losses["base"]),
            "trained_seed_mean_minus_market": registered.cluster_bootstrap_delta(
                rows, trained_mean, market_losses),
            "trained_seed_mean_minus_platt_market": registered.cluster_bootstrap_delta(
                rows, trained_mean, calibrated_losses),
        },
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_prefix.with_suffix(".jsonl")
    with raw_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            handle.write(json.dumps({
                "task_id": row["task_id"], "event_id": row["event_id"],
                "settlement_yes": row["settlement_yes"],
                "market_yes_prob": row["market_yes_prob"],
                "forecasts": {name: values[index] for name, values in forecasts.items()},
                "responses": {name: values[index] for name, values in responses.items()},
            }, sort_keys=True) + "\n")
    args.output_prefix.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
