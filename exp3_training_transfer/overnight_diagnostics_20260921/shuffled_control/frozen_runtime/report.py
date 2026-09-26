#!/usr/bin/env python3
"""Score C3 mechanism dumps and estimate seed/episode-clustered contrasts."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from contrasts import _bootstrap_draws, _task_means
from prompt import parse_forecasts
from worlds import BY_NAME, TEST, forecast_scenarios, response_vector

TEST_NAMES = tuple(world.name for world in TEST)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-10 else 0.0


def mechanism_prediction(predicted_response: np.ndarray, reference: dict) -> str:
    """Nearest response-shape template; level and response magnitude are normalized away."""
    surface = BY_NAME[reference["world"]]
    history = np.asarray(reference["target_inputs"], dtype=float)
    gain = float(reference["g"])
    scores = {}
    for candidate in TEST:
        candidate = candidate.on_surface_of(surface)
        template = response_vector(forecast_scenarios(candidate, history, gain))
        scores[candidate.name] = cosine(predicted_response, template)
    return max(scores, key=scores.get)


def score_records(records: list[dict], metadata: dict) -> list[dict]:
    scored = []
    for record_index, record in enumerate(records):
        reference = json.loads(record["reference"])
        outputs = record.get("output", [])
        outputs = outputs if isinstance(outputs, list) else [outputs]
        decode = "greedy" if float(record.get("temperature", 0.0)) == 0 else "stochastic"
        truth = np.asarray(reference["truth_targets"], dtype=float)
        truth_response = np.asarray(reference["truth_response"], dtype=float)
        prior_response = np.asarray(reference["prior_response"], dtype=float)
        prior_error = float(np.mean(np.abs(prior_response - truth_response)))
        for draw, output in enumerate(outputs):
            predicted = parse_forecasts(
                output, reference["scenario_labels"], reference["clip"]
            )
            row = {
                **metadata,
                "record": record_index,
                "task_id": reference["task_id"],
                "world": reference["world"],
                "block": reference["block"],
                "k": int(reference["k"]),
                "draw": draw,
                "decode": decode,
                "parsed": predicted is not None,
                "prior_response_mae": prior_error,
            }
            if predicted is not None:
                predicted = np.asarray(predicted, dtype=float)
                predicted_response = response_vector(predicted)
                response_mae = float(np.mean(np.abs(predicted_response - truth_response)))
                predicted_world = mechanism_prediction(predicted_response, reference)
                row.update({
                    "level_mae": float(np.mean(np.abs(predicted - truth))),
                    "response_mae": response_mae,
                    "normalized_response_error": response_mae / max(prior_error, 1e-8),
                    "response_cosine": cosine(predicted_response, truth_response),
                    "mechanism_prediction": predicted_world,
                    "mechanism_correct": predicted_world == reference["world"],
                    "predicted_response": predicted_response.round(6).tolist(),
                    "truth_response": truth_response.round(6).tolist(),
                })
            else:
                # Parse failures are failed forecasts, not missing-at-random data. Assign the
                # full observable range as the preregistered strict error penalty.
                penalty = float(reference["clip"][1] - reference["clip"][0])
                row.update({"level_mae": penalty, "response_mae": penalty,
                            "normalized_response_error": penalty / max(prior_error, 1e-8),
                            "response_cosine": None,
                            "mechanism_prediction": None, "mechanism_correct": False,
                            "predicted_response": None,
                            "truth_response": truth_response.round(6).tolist()})
            scored.append(row)
    return scored


def correlation(scored: list[dict]) -> float | None:
    parsed = [row for row in scored if row["parsed"]]
    if not parsed:
        return None
    predicted = np.concatenate([row["predicted_response"] for row in parsed])
    truth = np.concatenate([row["truth_response"] for row in parsed])
    if np.std(predicted) < 1e-10 or np.std(truth) < 1e-10:
        return None
    return float(np.corrcoef(predicted, truth)[0, 1])


def summarize(scored: list[dict]) -> list[dict]:
    keys = ("model", "disclosure", "arm", "seed", "decode", "block", "k")
    groups = defaultdict(list)
    for row in scored:
        groups[tuple(row[key] for key in keys)].append(row)
    summaries = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        result = dict(zip(keys, key))
        result.update({
            "n_tasks": len({row["task_id"] for row in rows}),
            "n_draws": len(rows),
            "parse_rate": float(np.mean([row["parsed"] for row in rows])),
            "level_mae": float(np.mean([row["level_mae"] for row in rows])),
            "response_mae": float(np.mean([row["response_mae"] for row in rows])),
            "normalized_response_error": (
                float(np.sum([row["response_mae"] for row in rows]) /
                      np.sum([row["prior_response_mae"] for row in rows]))
                if np.sum([row["prior_response_mae"] for row in rows]) > 0 else None
            ),
            "response_correlation": correlation(rows),
            "mechanism_accuracy": float(np.mean([row["mechanism_correct"] for row in rows])),
        })
        summaries.append(result)
    return summaries


def read_scores(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def task_arm_means(rows: list[dict], model: str, disclosure: str, decode: str,
                   block: str, seed: int, arm: str) -> dict[str, float]:
    return _task_means(rows, {
        "model": model, "disclosure": disclosure, "decode": decode,
        "block": block, "seed": seed, "arm": arm,
    })


def hierarchical_arm_contrasts(rows: list[dict], repetitions: int, bootstrap_seed: int) -> list[dict]:
    """Causal benefit = prior response error - causal response error.

    Training seeds are sampled at the top level; paired evaluation episodes are sampled within
    each selected seed. Decode draws are averaged inside an episode and are never counted as
    independent experimental replicates.
    """
    rng = np.random.default_rng(bootstrap_seed)
    dimensions = sorted({(row["model"], row["disclosure"], row["decode"], row["block"])
                         for row in rows if row["arm"] in
                         {"causal_family", "population_prior"}})
    results = []
    for model, disclosure, decode, block in dimensions:
        seeds = sorted({int(row["seed"]) for row in rows
                        if (row["model"], row["disclosure"], row["decode"], row["block"])
                        == (model, disclosure, decode, block)
                        and row["arm"] == "causal_family"})
        paired = {}
        for seed in seeds:
            causal = task_arm_means(rows, model, disclosure, decode, block, seed,
                                    "causal_family")
            prior = task_arm_means(rows, model, disclosure, decode, block, seed,
                                   "population_prior")
            task_ids = sorted(set(causal) & set(prior))
            if task_ids:
                paired[seed] = np.asarray([prior[item] - causal[item] for item in task_ids])
        if len(paired) < 2:
            continue
        seed_values = sorted(paired)
        estimate = float(np.mean([np.mean(paired[seed]) for seed in seed_values]))
        draws = _bootstrap_draws(paired, repetitions, rng)
        low, high = np.quantile(draws, [0.025, 0.975])
        results.append({
            "model": model, "disclosure": disclosure, "decode": decode, "block": block,
            "contrast": "population_prior_minus_causal_response_mae",
            "estimate": estimate, "ci95_low": float(low), "ci95_high": float(high),
            "training_seeds": seed_values,
            "episodes_per_seed": {str(seed): len(paired[seed]) for seed in seed_values},
            "bootstrap_repetitions": repetitions,
        })
    return results


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8")


def command_score(args) -> None:
    records = json.loads(args.dump.read_text(encoding="utf-8"))
    metadata = {"model": args.model, "disclosure": args.disclosure,
                "arm": args.arm, "seed": args.seed}
    scored = score_records(records, metadata)
    output = args.output or args.dump.with_suffix(".scores.jsonl")
    write_jsonl(output, scored)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summarize(scored), indent=2, sort_keys=True) + "\n")
    print(f"scored {len(scored)} row-draws -> {output}")


def command_aggregate(args) -> None:
    rows = read_scores(args.scores)
    result = {
        "summaries": summarize(rows),
        "hierarchical_arm_contrasts": hierarchical_arm_contrasts(
            rows, args.bootstrap_repetitions, args.bootstrap_seed
        ),
        "variance_unit": "training seed; paired episode resampling nested within seed; draws averaged",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"aggregated {len(rows)} row-draws -> {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score")
    score.add_argument("dump", type=Path)
    score.add_argument("--model", required=True)
    score.add_argument("--disclosure", choices=("disclosed", "undisclosed"), required=True)
    score.add_argument("--arm", required=True)
    score.add_argument("--seed", type=int, required=True)
    score.add_argument("--output", type=Path)
    score.set_defaults(func=command_score)

    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("scores", nargs="+", type=Path)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--bootstrap-repetitions", type=int, default=5000)
    aggregate.add_argument("--bootstrap-seed", type=int, default=20260814)
    aggregate.set_defaults(func=command_aggregate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
