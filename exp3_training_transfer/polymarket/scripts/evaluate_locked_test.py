#!/usr/bin/env python3
"""Evaluate base and all registered 3B adapters once on the locked test split."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import vllm
from vllm.lora.request import LoRARequest

from family_clusters import connected_index_groups


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
DATA = ROOT / "data" / "exp3b_registered"
BASELINES = ROOT / "reports" / "exp3b_baselines.json"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
EPS = 1e-6

import sys

sys.path.insert(0, str(REPO / "exp3_training_transfer" / "biased_news"))
from forecast_scoring import parse_yes_probability  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prompt(text: str) -> str:
    return "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n"


def clamp(value: float) -> float:
    return max(EPS, min(1.0 - EPS, value))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def logit(value: float) -> float:
    p = clamp(value)
    return math.log(p / (1.0 - p))


def losses(probability: float | None, label: bool) -> tuple[float, float]:
    if probability is None:
        return 1.0, -math.log(EPS)
    p = clamp(probability)
    y = float(label)
    return (p - y) ** 2, -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))


def summarize(probabilities: list[float | None], rows: list[dict[str, Any]]) -> dict[str, float]:
    scored = [losses(p, bool(row["settlement_yes"])) for p, row in zip(probabilities, rows)]
    return {
        "n": len(rows),
        "parse_coverage": sum(p is not None for p in probabilities) / len(rows),
        "brier": sum(item[0] for item in scored) / len(scored),
        "log_loss_nats": sum(item[1] for item in scored) / len(scored),
    }


def cluster_bootstrap_delta(
    rows: list[dict[str, Any]], first: list[float], second: list[float], repetitions: int = 5_000
) -> dict[str, float]:
    groups = connected_index_groups(rows)
    rng = random.Random(30_032_026)
    samples = []
    for _ in range(repetitions):
        indices = [index for _ in groups for index in rng.choice(groups)]
        samples.append(sum(first[i] - second[i] for i in indices) / len(indices))
    samples.sort()
    return {
        "estimate": sum(a - b for a, b in zip(first, second)) / len(first),
        "ci95_low": samples[int(0.025 * repetitions)],
        "ci95_high": samples[int(0.975 * repetitions)],
        "family_clusters": len(groups),
        "bootstrap_repetitions": repetitions,
    }


def resolve_adapter(job_id: str) -> Path:
    roots = sorted((ROOT / "reports").glob(f"train_market_*_j{job_id}"))
    if len(roots) != 1:
        raise AssertionError(f"expected one report directory for training job {job_id}, got {roots}")
    adapters = sorted(roots[0].glob("debug_*/saved_models/step_*"))
    if not adapters:
        raise AssertionError(f"no saved adapter for training job {job_id}")
    adapter = adapters[-1]
    if not (adapter / "adapter_config.json").is_file():
        raise AssertionError(f"not a PEFT adapter: {adapter}")
    return adapter


def generate(
    llm: vllm.LLM,
    prompts: list[str],
    sampling: vllm.SamplingParams,
    adapter: LoRARequest | None = None,
) -> tuple[list[float | None], list[str]]:
    outputs = llm.generate(prompts, sampling, lora_request=adapter)
    texts = [output.outputs[0].text for output in outputs]
    return [parse_yes_probability(text) for text in texts], texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapters",
        required=True,
        help="Semicolon-separated seed:training_job_id pairs, e.g. 42:1;43:2;44:3",
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    seed_jobs = {}
    for value in args.adapters.split(";"):
        seed, job = value.split(":", 1)
        seed_jobs[int(seed)] = job
    if sorted(seed_jobs) != [42, 43, 44]:
        raise AssertionError("locked evaluation requires seeds 42, 43, and 44")

    rows = read_jsonl(DATA / "test.tasks.jsonl")
    prompts = [prompt(row["input"]) for row in rows]
    sampling = vllm.SamplingParams(temperature=0.0, max_tokens=128)
    llm = vllm.LLM(
        model=MODEL,
        enable_lora=True,
        max_lora_rank=32,
        max_model_len=1920,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )

    forecasts: dict[str, list[float | None]] = {}
    responses: dict[str, list[str]] = {}
    forecasts["base"], responses["base"] = generate(llm, prompts, sampling)
    adapter_paths = {}
    for adapter_id, seed in enumerate(sorted(seed_jobs), start=1):
        path = resolve_adapter(seed_jobs[seed])
        adapter_paths[str(seed)] = str(path)
        request = LoRARequest(f"exp3b_seed{seed}", adapter_id, str(path))
        forecasts[f"seed_{seed}"], responses[f"seed_{seed}"] = generate(
            llm, prompts, sampling, request
        )

    baseline_report = json.loads(BASELINES.read_text(encoding="utf-8"))
    platt = baseline_report["platt_parameters"]
    market = [float(row["market_yes_prob"]) for row in rows]
    calibrated = [
        sigmoid(float(platt["intercept"]) + float(platt["market_logit_slope"]) * logit(value))
        for value in market
    ]
    labels = [bool(row["settlement_yes"]) for row in rows]
    model_losses = {
        name: [losses(p, y)[0] for p, y in zip(values, labels)]
        for name, values in forecasts.items()
    }
    market_losses = [losses(p, y)[0] for p, y in zip(market, labels)]
    calibrated_losses = [losses(p, y)[0] for p, y in zip(calibrated, labels)]
    trained_names = [f"seed_{seed}" for seed in sorted(seed_jobs)]
    trained_mean_losses = [
        sum(model_losses[name][index] for name in trained_names) / len(trained_names)
        for index in range(len(rows))
    ]

    summary = {
        "protocol_version": "exp3b_registered_v1",
        "test_was_used_during_training": False,
        "model": MODEL,
        "decoding": {"temperature": 0.0, "max_tokens": 128},
        "adapter_paths": adapter_paths,
        "models": {name: summarize(values, rows) for name, values in forecasts.items()},
        "baselines": {
            "market": summarize(market, rows),
            "platt_market_train_only": summarize(calibrated, rows),
        },
        "comparisons_brier": {
            "trained_seed_mean_minus_base": cluster_bootstrap_delta(
                rows, trained_mean_losses, model_losses["base"]
            ),
            "trained_seed_mean_minus_market": cluster_bootstrap_delta(
                rows, trained_mean_losses, market_losses
            ),
            "trained_seed_mean_minus_platt_market": cluster_bootstrap_delta(
                rows, trained_mean_losses, calibrated_losses
            ),
            "base_minus_market": cluster_bootstrap_delta(rows, model_losses["base"], market_losses),
        },
    }

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_prefix.with_suffix(".jsonl")
    with raw_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            handle.write(
                json.dumps(
                    {
                        "task_id": row["task_id"],
                        "event_id": row["event_id"],
                        "settlement_yes": row["settlement_yes"],
                        "market_yes_prob": row["market_yes_prob"],
                        "forecasts": {name: values[index] for name, values in forecasts.items()},
                        "responses": {name: values[index] for name, values in responses.items()},
                    },
                    sort_keys=True,
                    ensure_ascii=True,
                )
                + "\n"
            )
    summary_path = args.output_prefix.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
