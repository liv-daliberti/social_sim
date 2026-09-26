#!/usr/bin/env python3
"""Analyze the frozen harder-generator Coin City population across open models."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import analyze_coin_city_symbol_context as shared

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data" / "coin_city_population_075_040_sd010_v1"
DESIGN = RUN / "design"
OUTPUT = RUN / "analysis" / "population_robustness_results_v1.json"
ORIGINAL = (
    ROOT
    / "local_results"
    / "symbol_context_model_comparison_20260824"
    / "exploratory_results.json"
)
ARMS = ("abc_no_context", "abc_context", "abc_symbol_context")
BOOTSTRAP_DRAWS = 5_000
MODELS = (
    ("qwen3_4b", "Qwen3-4B-Instruct-2507"),
    ("qwen3_8b", "Qwen3-8B"),
    ("qwen3_14b", "Qwen3-14B"),
    ("qwen3_32b", "Qwen3-32B"),
    ("qwen2_5_7b", "Qwen2.5-7B-Instruct"),
    ("qwen2_5_14b", "Qwen2.5-14B-Instruct"),
    ("qwen2_5_32b", "Qwen2.5-32B-Instruct"),
    ("qwen2_5_72b", "Qwen2.5-72B-Instruct"),
    ("llama3_1_8b", "Llama-3.1-8B-Instruct"),
    ("llama3_1_70b", "Llama-3.1-70B-Instruct"),
)
SLUG = {model: slug for slug, model in MODELS}


def response_path(model: str, arm: str) -> Path:
    return RUN / "responses" / SLUG[model] / f"responses_{model}_{arm}.jsonl"


def main() -> None:
    scores = {
        row["task_id"]: row for row in shared.read_jsonl(DESIGN / "scoring_key.jsonl")
    }
    episodes = {
        int(row["episode"]): row for row in shared.read_jsonl(DESIGN / "episodes.jsonl")
    }
    missing = [
        str(response_path(model, arm))
        for _, model in MODELS
        for arm in ARMS
        if not response_path(model, arm).exists()
    ]
    if missing:
        raise SystemExit(
            f"harder-population response set incomplete: {len(missing)} files missing"
        )

    # Reuse the paper's metric implementation, changing only the registered
    # design and response-file resolver.
    shared.response_path = response_path
    results = {
        model: shared.analyze_model(
            model,
            scores=scores,
            episodes=episodes,
            bootstrap_draws=BOOTSTRAP_DRAWS,
        )
        for _, model in MODELS
    }
    for model, result in results.items():
        for arm in ARMS:
            if result["record_counts"][arm] != 1_250:
                raise SystemExit(f"{model}:{arm} does not have exactly 1,250 records")

    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    point_comparison = {}
    for _, model in MODELS:
        old = original["models"][model]["curves"]["0"]
        new = results[model]["curves"]["0"]
        point_comparison[model] = {
            "original_population": {
                "realized_generator": {
                    "strong_mean": 0.902,
                    "weak_mean": 0.250,
                    "slope_sd": 0.03,
                },
                "symbol_rho": old["arms"]["abc_symbol_context"]["rho"],
                "no_context_rho": old["arms"]["abc_no_context"]["rho"],
                "symbol_minus_no_context_rho": old["paired_discrimination"][
                    "symbol_minus_no_context_rho"
                ],
                "symbol_regime_accuracy": old["arms"]["abc_symbol_context"][
                    "regime_accuracy"
                ],
            },
            "harder_population": {
                "target_generator": {
                    "strong_mean": 0.75,
                    "weak_mean": 0.40,
                    "slope_sd": 0.10,
                },
                "symbol_rho": new["arms"]["abc_symbol_context"]["rho"],
                "no_context_rho": new["arms"]["abc_no_context"]["rho"],
                "symbol_minus_no_context_rho": new["paired_discrimination"][
                    "symbol_minus_no_context_rho"
                ],
                "symbol_regime_accuracy": new["arms"]["abc_symbol_context"][
                    "regime_accuracy"
                ],
            },
        }

    manifest = json.loads((DESIGN / "manifest.json").read_text(encoding="utf-8"))
    report = {
        "protocol_version": "coin_city_robustness_v1",
        "classification": "post_hoc_generator_population_robustness",
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bootstrap_unit": "episode",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "generator_population": {
            "data_generating_process": manifest["data_generating_process"],
            "realized": manifest["realized"],
        },
        "models": results,
        "k0_point_comparison": point_comparison,
        "sources": {
            "design": str(DESIGN.resolve()),
            "original_results": str(ORIGINAL.resolve()),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"output": str(OUTPUT), "models": len(results), "records": 37_500}, indent=2
        )
    )


if __name__ == "__main__":
    main()
