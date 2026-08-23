#!/usr/bin/env python3
"""Score Coin City structural-transfer endpoints and estimate nested contrasts."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_forecasts(text: str, count: int, clip: list[float]):
    try:
        payload = json.loads(text.strip())
    except (AttributeError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"forecasts"}:
        return None
    values = payload["forecasts"]
    if not isinstance(values, list) or len(values) != count:
        return None
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(float(value)) for value in values):
        return None
    lo, hi = map(float, clip)
    return np.asarray([max(lo, min(hi, float(value))) for value in values])


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0


def response_vector(values: np.ndarray, pairs: list[list[int]]) -> np.ndarray:
    return np.asarray([values[int(left)] - values[int(right)] for left, right in pairs])


def score_records(records: list[dict], metadata: dict) -> list[dict]:
    scored = []
    for record_index, record in enumerate(records):
        ref = json.loads(record["reference"])
        outputs = record.get("output", [])
        outputs = outputs if isinstance(outputs, list) else [outputs]
        decode = "greedy" if float(record.get("temperature", 0.0)) == 0 else "stochastic"
        truth = np.asarray(ref["truth_targets"], dtype=float)
        truth_response = np.asarray(ref["truth_response"], dtype=float)
        prior_response = np.asarray(ref["prior_response"], dtype=float)
        prior_error = float(np.mean(np.abs(prior_response - truth_response)))
        templates = {
            "direct_a": np.asarray(ref["direct_template_response"], dtype=float),
            "mediated_b": np.asarray(ref["mediated_template_response"], dtype=float),
        }
        for draw, output in enumerate(outputs):
            predicted = parse_forecasts(output, len(truth), ref["clip"])
            row = {
                **metadata,
                "record": record_index,
                "task_id": ref["task_id"],
                "pair_id": ref["pair_id"],
                "domain": ref["domain"],
                "target_structure": ref["target_structure"],
                "cue": ref["cue"],
                "k": int(ref["k"]),
                "draw": draw,
                "decode": decode,
                "parsed": predicted is not None,
                "prior_response_mae": prior_error,
            }
            if predicted is None:
                penalty = float(ref["clip"][1] - ref["clip"][0])
                row.update({
                    "level_mae": penalty,
                    "response_mae": penalty,
                    "normalized_response_error": penalty / max(prior_error, 1e-8),
                    "response_cosine": None,
                    "predicted_structure": None,
                    "structure_correct": False,
                    "predicted_response": None,
                    "truth_response": truth_response.round(6).tolist(),
                })
            else:
                predicted_response = response_vector(predicted, ref["response_pairs"])
                predicted_structure = max(
                    templates, key=lambda key: cosine(predicted_response, templates[key])
                )
                response_mae = float(np.mean(np.abs(predicted_response - truth_response)))
                row.update({
                    "level_mae": float(np.mean(np.abs(predicted - truth))),
                    "response_mae": response_mae,
                    "normalized_response_error": response_mae / max(prior_error, 1e-8),
                    "response_cosine": cosine(predicted_response, truth_response),
                    "predicted_structure": predicted_structure,
                    "structure_correct": predicted_structure == ref["target_structure"],
                    "predicted_response": predicted_response.round(6).tolist(),
                    "truth_response": truth_response.round(6).tolist(),
                })
            scored.append(row)
    return scored


def summarize(rows: list[dict]) -> list[dict]:
    keys = ("model", "arm", "seed", "decode", "domain", "target_structure", "cue", "k")
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        parsed = [row for row in values if row["parsed"]]
        result = dict(zip(keys, key))
        result.update({
            "n_tasks": len({row["task_id"] for row in values}),
            "n_draws": len(values),
            "parse_rate": float(np.mean([row["parsed"] for row in values])),
            "level_mae": float(np.mean([row["level_mae"] for row in values])),
            "response_mae": float(np.mean([row["response_mae"] for row in values])),
            "normalized_response_error": float(
                np.sum([row["response_mae"] for row in values])
                / max(np.sum([row["prior_response_mae"] for row in values]), 1e-8)
            ),
            "response_correlation": None,
            "structure_accuracy": float(np.mean([row["structure_correct"] for row in values])),
        })
        if parsed:
            predicted = np.concatenate([row["predicted_response"] for row in parsed])
            truth = np.concatenate([row["truth_response"] for row in parsed])
            if np.std(predicted) > 1e-12 and np.std(truth) > 1e-12:
                result["response_correlation"] = float(np.corrcoef(predicted, truth)[0, 1])
        output.append(result)
    return output


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8")


def command_score(args) -> None:
    records = json.loads(args.dump.read_text(encoding="utf-8"))
    scored = score_records(records, {"model": args.model, "arm": args.arm, "seed": args.seed})
    output = args.output or args.dump.with_suffix(".scores.jsonl")
    write_jsonl(output, scored)
    output.with_suffix(".summary.json").write_text(
        json.dumps(summarize(scored), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"scored {len(scored)} row-draws -> {output}")


def read_scores(paths: list[Path]) -> list[dict]:
    output = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            output.extend(json.loads(line) for line in handle if line.strip())
    return output


def episode_means(rows: list[dict], *, model: str, arm: str, seed: int, decode: str,
                  domain: str, structure: str, cue: str) -> dict[str, float]:
    grouped = defaultdict(list)
    for row in rows:
        if (row["model"], row["arm"], int(row["seed"]), row["decode"], row["domain"],
                row["target_structure"], row["cue"]) != (
                model, arm, seed, decode, domain, structure, cue):
            continue
        grouped[row["task_id"]].append(float(row["response_mae"]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def hierarchical_contrasts(rows: list[dict], repetitions: int, bootstrap_seed: int) -> list[dict]:
    rng = np.random.default_rng(bootstrap_seed)
    dimensions = sorted({
        (row["model"], row["decode"], row["domain"], row["target_structure"], row["cue"])
        for row in rows if row["arm"] in {"causal", "population_prior"}
    })
    results = []
    for model, decode, domain, structure, cue in dimensions:
        paired = {}
        for seed in (42, 43, 44):
            causal = episode_means(rows, model=model, arm="causal", seed=seed,
                                   decode=decode, domain=domain, structure=structure, cue=cue)
            prior = episode_means(rows, model=model, arm="population_prior", seed=seed,
                                  decode=decode, domain=domain, structure=structure, cue=cue)
            task_ids = sorted(set(causal) & set(prior))
            if task_ids:
                paired[seed] = np.asarray([prior[key] - causal[key] for key in task_ids])
        if len(paired) != 3:
            continue
        estimate = float(np.mean([np.mean(values) for values in paired.values()]))
        draws = []
        seed_values = np.asarray(sorted(paired))
        for _ in range(repetitions):
            selected = rng.choice(seed_values, len(seed_values), replace=True)
            sampled_seeds = []
            for seed in selected:
                values = paired[int(seed)]
                sampled_seeds.append(float(np.mean(rng.choice(values, len(values), replace=True))))
            draws.append(float(np.mean(sampled_seeds)))
        low, high = np.quantile(draws, [0.025, 0.975])
        results.append({
            "model": model, "decode": decode, "domain": domain,
            "target_structure": structure, "cue": cue,
            "contrast": "population_prior_minus_causal_response_mae",
            "estimate": estimate, "ci95_low": float(low), "ci95_high": float(high),
            "training_seeds": [42, 43, 44], "episodes_per_seed": len(next(iter(paired.values()))),
        })
    return results


def command_aggregate(args) -> None:
    rows = read_scores(args.scores)
    result = {
        "summaries": summarize(rows),
        "hierarchical_contrasts": hierarchical_contrasts(
            rows, args.bootstrap_repetitions, args.bootstrap_seed
        ),
        "variance_unit": "training seed; paired episode within seed; decode draws averaged within episode",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"aggregated {len(rows)} row-draws -> {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score")
    score.add_argument("dump", type=Path)
    score.add_argument("--model", required=True)
    score.add_argument("--arm", required=True)
    score.add_argument("--seed", type=int, required=True)
    score.add_argument("--output", type=Path)
    score.set_defaults(func=command_score)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("scores", nargs="+", type=Path)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--bootstrap-repetitions", type=int, default=5000)
    aggregate.add_argument("--bootstrap-seed", type=int, default=20260818)
    aggregate.set_defaults(func=command_aggregate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
