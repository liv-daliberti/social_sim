#!/usr/bin/env python3
"""Five-draw locked evaluation for the Exp4 Qwen scale extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import evaluate_locked_test as registered

if TYPE_CHECKING:
    import vllm
    from vllm.lora.request import LoRARequest


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "exp4_scale_registered"
BASELINES = ROOT / "reports" / "exp3b_baselines.json"
PROTOCOL_VERSION = "exp4_qwen_scale_v1"
SEEDS = (42, 43, 44)
DRAWS = 5
SAMPLING_SEED = 20_260_824


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def format_prompt(tokenizer: Any, text: str, template: str) -> str:
    if template == "biased_news":
        return registered.prompt(text)
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if template == "auto_no_think":
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template([{"role": "user", "content": text}], **kwargs)


def parse_adapter_specs(value: str) -> dict[int, str]:
    result: dict[int, str] = {}
    for item in value.split(";"):
        seed, spec = item.split(":", 1)
        result[int(seed)] = spec
    if tuple(sorted(result)) != SEEDS:
        raise AssertionError("scale evaluation requires seeds 42, 43, and 44")
    return result


def resolve_job_adapter(model_key: str, job_id: str) -> Path:
    roots = sorted(
        (ROOT / "reports").glob(f"scale_train_{model_key}_market_*_j{job_id}")
    )
    if len(roots) != 1:
        raise AssertionError(
            f"expected one {model_key} scale report for job {job_id}, got {roots}"
        )
    adapters = sorted(roots[0].glob("debug_*/saved_models/step_*"))
    if not adapters:
        raise AssertionError(f"no final adapter for scale job {job_id}")
    return adapters[-1]


def resolve_adapter(model_key: str, spec: str) -> Path:
    if spec.startswith("job="):
        path = resolve_job_adapter(model_key, spec.removeprefix("job="))
    elif spec.startswith("path="):
        path = Path(spec.removeprefix("path="))
    else:
        raise AssertionError(f"invalid adapter spec: {spec}")
    if not path.is_absolute():
        raise AssertionError(f"adapter path must be absolute: {path}")
    if not (path / "adapter_config.json").is_file():
        raise AssertionError(f"not a PEFT adapter: {path}")
    if not (path / "adapter_model.safetensors").is_file():
        raise AssertionError(f"adapter weights missing: {path}")
    return path


def aggregate_draws(draws: list[float | None]) -> float | None:
    if len(draws) != DRAWS:
        raise AssertionError(f"expected {DRAWS} draws, got {len(draws)}")
    if any(value is None for value in draws):
        return None
    values = [float(value) for value in draws if value is not None]
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise AssertionError("parsed probability is outside [0, 1]")
    return sum(values) / DRAWS


def generate_draws(
    llm: vllm.LLM,
    prompts: list[str],
    sampling: vllm.SamplingParams,
    adapter: LoRARequest | None = None,
) -> tuple[list[float | None], list[list[float | None]], list[list[str]]]:
    outputs = llm.generate(prompts, sampling, lora_request=adapter)
    if len(outputs) != len(prompts):
        raise AssertionError("vLLM returned the wrong number of task outputs")
    task_draws: list[list[float | None]] = []
    task_responses: list[list[str]] = []
    aggregate: list[float | None] = []
    for output in outputs:
        if len(output.outputs) != DRAWS:
            raise AssertionError(
                f"vLLM returned {len(output.outputs)} draws; expected {DRAWS}"
            )
        texts = [candidate.text for candidate in output.outputs]
        draws = [registered.parse_yes_probability(text) for text in texts]
        task_draws.append(draws)
        task_responses.append(texts)
        aggregate.append(aggregate_draws(draws))
    return aggregate, task_draws, task_responses


def validate_holdout() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = DATA / "manifest.json"
    task_path = DATA / "test.tasks.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "pass":
        raise AssertionError("scale holdout manifest is not a pass")
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise AssertionError("scale holdout protocol version drifted")
    if sha256(task_path) != manifest["hashes"]["test_tasks_sha256"]:
        raise AssertionError("scale holdout task hash drifted")
    rows = registered.read_jsonl(task_path)
    if len(rows) != manifest["holdout"]["count"] or len(rows) < 256:
        raise AssertionError("scale holdout task count drifted")
    if len({row["task_id"] for row in rows}) != len(rows):
        raise AssertionError("scale holdout contains duplicate task IDs")
    return rows, manifest


def endpoint_summary(
    aggregate: list[float | None],
    draws: list[list[float | None]],
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    summary = registered.summarize(aggregate, rows)
    total_draws = sum(len(values) for values in draws)
    valid_draws = sum(value is not None for values in draws for value in values)
    summary["draw_parse_coverage"] = valid_draws / total_draws
    summary["complete_task_parse_coverage"] = summary["parse_coverage"]
    summary["draws_per_task"] = float(DRAWS)
    return summary


def main() -> None:
    import vllm
    from vllm.lora.request import LoRARequest

    parser = argparse.ArgumentParser()
    parser.add_argument("--adapters", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--template",
        choices=("biased_news", "auto", "auto_no_think"),
        required=True,
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    adapter_specs = parse_adapter_specs(args.adapters)
    rows, holdout_manifest = validate_holdout()
    sampling = vllm.SamplingParams(
        n=DRAWS,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        seed=SAMPLING_SEED,
        max_tokens=128,
    )
    llm = vllm.LLM(
        model=args.model,
        enable_lora=True,
        max_lora_rank=32,
        max_model_len=1920,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )
    prompts = [
        format_prompt(llm.get_tokenizer(), row["input"], args.template) for row in rows
    ]

    forecasts: dict[str, list[float | None]] = {}
    draw_forecasts: dict[str, list[list[float | None]]] = {}
    responses: dict[str, list[list[str]]] = {}
    forecasts["base"], draw_forecasts["base"], responses["base"] = generate_draws(
        llm, prompts, sampling
    )
    adapter_paths: dict[str, str] = {}
    adapter_hashes: dict[str, str] = {}
    for adapter_id, seed in enumerate(SEEDS, start=1):
        path = resolve_adapter(args.model_key, adapter_specs[seed])
        adapter_paths[str(seed)] = str(path)
        adapter_hashes[str(seed)] = sha256(path / "adapter_model.safetensors")
        request = LoRARequest(
            f"exp4scale_{args.model_key}_seed{seed}", adapter_id, str(path)
        )
        name = f"seed_{seed}"
        forecasts[name], draw_forecasts[name], responses[name] = generate_draws(
            llm, prompts, sampling, request
        )

    baseline_report = json.loads(BASELINES.read_text(encoding="utf-8"))
    if baseline_report.get("protocol_version") != "exp3b_registered_v1":
        raise AssertionError("registered Platt baseline drifted")
    platt = baseline_report["platt_parameters"]
    market = [float(row["market_yes_prob"]) for row in rows]
    calibrated = [
        registered.sigmoid(
            float(platt["intercept"])
            + float(platt["market_logit_slope"]) * registered.logit(value)
        )
        for value in market
    ]
    labels = [bool(row["settlement_yes"]) for row in rows]
    model_losses = {
        name: [
            registered.losses(value, label)[0] for value, label in zip(values, labels)
        ]
        for name, values in forecasts.items()
    }
    market_losses = [
        registered.losses(value, label)[0] for value, label in zip(market, labels)
    ]
    platt_losses = [
        registered.losses(value, label)[0] for value, label in zip(calibrated, labels)
    ]
    trained_names = [f"seed_{seed}" for seed in SEEDS]
    trained_mean_losses = [
        sum(model_losses[name][index] for name in trained_names) / len(SEEDS)
        for index in range(len(rows))
    ]

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "test_was_used_during_training": False,
        "model": args.model,
        "model_key": args.model_key,
        "decoding": {
            "mode": "five_draw_non_thinking_stochastic",
            "draws": DRAWS,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "sampling_seed": SAMPLING_SEED,
            "max_tokens": 128,
            "incomplete_draw_policy": "endpoint_task_brier_loss_one",
        },
        "holdout": {
            "manifest_sha256": sha256(DATA / "manifest.json"),
            "test_tasks_sha256": holdout_manifest["hashes"]["test_tasks_sha256"],
            "n": len(rows),
        },
        "analysis_code_sha256": sha256(Path(__file__)),
        "adapter_paths": adapter_paths,
        "adapter_sha256": adapter_hashes,
        "models": {
            name: endpoint_summary(values, draw_forecasts[name], rows)
            for name, values in forecasts.items()
        },
        "baselines": {
            "market": registered.summarize(market, rows),
            "platt_market_train_only": registered.summarize(calibrated, rows),
        },
        "comparisons_brier": {
            "trained_seed_mean_minus_base": registered.cluster_bootstrap_delta(
                rows, trained_mean_losses, model_losses["base"]
            ),
            "trained_seed_mean_minus_market": registered.cluster_bootstrap_delta(
                rows, trained_mean_losses, market_losses
            ),
            "trained_seed_mean_minus_platt_market": (
                registered.cluster_bootstrap_delta(
                    rows, trained_mean_losses, platt_losses
                )
            ),
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
                        "series_id": row["series_id"],
                        "template_key": row["template_key"],
                        "settlement_yes": row["settlement_yes"],
                        "market_yes_prob": row["market_yes_prob"],
                        "forecasts": {
                            name: values[index] for name, values in forecasts.items()
                        },
                        "draw_forecasts": {
                            name: values[index]
                            for name, values in draw_forecasts.items()
                        },
                        "responses": {
                            name: values[index] for name, values in responses.items()
                        },
                    },
                    sort_keys=True,
                    ensure_ascii=True,
                )
                + "\n"
            )
    summary_path = args.output_prefix.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
