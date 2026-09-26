#!/usr/bin/env python3
"""Price-ablation evaluation for the Exp4 historical-market models.

Re-evaluates an untrained base and its three trained adapters with the
pre-cutoff YES price block removed from every prompt. All other prompt text,
chat formatting, decoding, scoring, and the family-clustered bootstrap are the
ones used for the reported results. ``--condition with_price`` reruns the
unmodified prompt under the same code path as a reproduction check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import evaluate_locked_test as registered
import evaluate_scale_holdout as scale

ROOT = Path(__file__).resolve().parents[1]
HOLDOUTS = {
    "scale318": ROOT / "data" / "exp4_scale_registered" / "test.tasks.jsonl",
    "test1024": ROOT / "data" / "exp3b_registered" / "test.tasks.jsonl",
}
TRAIN_TASKS = ROOT / "data" / "exp3b_registered" / "train.tasks.jsonl"
PRICE_HEADER = "Genuine pre-cutoff YES closing prices (oldest to newest):\n"
OUTPUT_TAIL = "Return exactly one JSON object"
SEEDS = (42, 43, 44)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strip_prices(row: dict[str, Any]) -> str:
    """Remove the price header, its dated lines, and the blank line after them."""
    text = row["input"]
    if text.count(PRICE_HEADER) != 1:
        raise AssertionError(f"{row['task_id']}: expected one price block")
    head, rest = text.split(PRICE_HEADER)
    block, tail = rest.split("\n\n" + OUTPUT_TAIL, 1)
    lines = block.split("\n")
    expected = [
        f"- {point['ts'][:10]}: {float(point['yes_close']):.4f}"
        for point in row["price_history"]
    ]
    if lines != expected:
        raise AssertionError(f"{row['task_id']}: price block does not match history")
    stripped = head + OUTPUT_TAIL + tail
    if PRICE_HEADER.strip() in stripped or any(line in stripped for line in expected):
        raise AssertionError(f"{row['task_id']}: price block survived")
    if stripped != text.replace(PRICE_HEADER + block + "\n\n", ""):
        raise AssertionError(f"{row['task_id']}: removal changed other text")
    return stripped


def resolve_adapter(spec: str) -> Path:
    path = Path(spec)
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (path / name).is_file():
            raise AssertionError(f"{path} is missing {name}")
    return path


def main() -> None:
    import vllm
    from vllm.lora.request import LoRARequest

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument(
        "--template", choices=("biased_news", "auto", "auto_no_think"), required=True
    )
    parser.add_argument("--adapters", required=True, help="42:/abs/path;43:...;44:...")
    parser.add_argument("--holdout", choices=sorted(HOLDOUTS), required=True)
    parser.add_argument("--decoder", choices=("five_draw", "greedy"), required=True)
    parser.add_argument(
        "--condition", choices=("no_price", "with_price"), default="no_price"
    )
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    specs = dict(item.split(":", 1) for item in args.adapters.split(";"))
    if sorted(int(seed) for seed in specs) != list(SEEDS):
        raise AssertionError("price ablation requires seeds 42, 43, and 44")
    adapters = {int(seed): resolve_adapter(path) for seed, path in specs.items()}

    task_path = HOLDOUTS[args.holdout]
    rows = registered.read_jsonl(task_path)
    texts = [
        strip_prices(row) if args.condition == "no_price" else row["input"]
        for row in rows
    ]

    if args.decoder == "five_draw":
        sampling = vllm.SamplingParams(
            n=scale.DRAWS,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            seed=scale.SAMPLING_SEED,
            max_tokens=128,
        )
    else:
        sampling = vllm.SamplingParams(temperature=0.0, max_tokens=128)
    llm = vllm.LLM(
        model=args.model,
        enable_lora=True,
        max_lora_rank=32,
        max_model_len=1920,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel,
    )
    prompts = [scale.format_prompt(llm.get_tokenizer(), text, args.template) for text in texts]

    def run(request: LoRARequest | None) -> tuple[list, list, list]:
        if args.decoder == "five_draw":
            return scale.generate_draws(llm, prompts, sampling, request)
        values, responses = registered.generate(llm, prompts, sampling, request)
        return values, [[v] for v in values], [[r] for r in responses]

    forecasts, draws, responses = {}, {}, {}
    forecasts["base"], draws["base"], responses["base"] = run(None)
    for adapter_id, seed in enumerate(SEEDS, start=1):
        request = LoRARequest(
            f"noprice_{args.model_key}_seed{seed}", adapter_id, str(adapters[seed])
        )
        name = f"seed_{seed}"
        forecasts[name], draws[name], responses[name] = run(request)

    labels = [bool(row["settlement_yes"]) for row in rows]
    train_rate = sum(bool(r["settlement_yes"]) for r in registered.read_jsonl(TRAIN_TASKS))
    train_rate /= len(registered.read_jsonl(TRAIN_TASKS))
    market = [float(row["market_yes_prob"]) for row in rows]
    climatology = [train_rate] * len(rows)

    def loss_list(values: list[float | None]) -> list[float]:
        return [registered.losses(v, y)[0] for v, y in zip(values, labels)]

    model_losses = {name: loss_list(values) for name, values in forecasts.items()}
    trained = [
        sum(model_losses[f"seed_{seed}"][i] for seed in SEEDS) / len(SEEDS)
        for i in range(len(rows))
    ]
    base = model_losses["base"]
    delta = registered.cluster_bootstrap_delta
    summary = {
        "analysis": "exp4_price_ablation_v1",
        "condition": args.condition,
        "holdout": args.holdout,
        "holdout_sha256": sha256(task_path),
        "n": len(rows),
        "model": args.model,
        "model_key": args.model_key,
        "template": args.template,
        "decoder": args.decoder,
        "adapter_paths": {str(k): str(v) for k, v in adapters.items()},
        "analysis_code_sha256": sha256(Path(__file__)),
        "train_yes_rate": train_rate,
        "models": {
            name: scale.endpoint_summary(values, draws[name], rows)
            if args.decoder == "five_draw"
            else registered.summarize(values, rows)
            for name, values in forecasts.items()
        },
        "baselines": {
            "market_last_price": registered.summarize(market, rows),
            "train_climatology": registered.summarize(climatology, rows),
        },
        "comparisons_brier": {
            "trained_seed_mean_minus_base": delta(rows, trained, base),
            "trained_seed_mean_minus_climatology": delta(
                rows, trained, loss_list(climatology)
            ),
            "base_minus_climatology": delta(rows, base, loss_list(climatology)),
            "trained_seed_mean_minus_market": delta(rows, trained, loss_list(market)),
        },
    }

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with args.output_prefix.with_suffix(".jsonl").open("w", encoding="utf-8") as out:
        for i, row in enumerate(rows):
            out.write(
                json.dumps(
                    {
                        "task_id": row["task_id"],
                        "event_id": row["event_id"],
                        "template_key": row["template_key"],
                        "settlement_yes": row["settlement_yes"],
                        "market_yes_prob": row["market_yes_prob"],
                        "forecasts": {k: v[i] for k, v in forecasts.items()},
                        "draw_forecasts": {k: v[i] for k, v in draws.items()},
                        "responses": {k: v[i] for k, v in responses.items()},
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    args.output_prefix.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["comparisons_brier"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
