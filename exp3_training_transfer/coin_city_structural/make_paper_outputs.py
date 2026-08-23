#!/usr/bin/env python3
"""Validate the complete registered Coin City grid and render paper tables.

This script is deliberately fail closed: no table is written unless the exact
18 confirmatory, three structureless, and three base endpoints are complete and
share the frozen 1,440-task universe under both decoding regimes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
MODELS = ("qwen3_4b", "qwen3_8b", "llama3_1_8b")
ARMS = ("causal", "population_prior")
SEEDS = (42, 43, 44)
MODEL_LABELS = {
    "qwen3_4b": "Qwen3-4B",
    "qwen3_8b": "Qwen3-8B",
    "llama3_1_8b": "Llama-3.1-8B",
}
DOMAIN_LABELS = {"coin_city": "City", "coin_harbor": "Harbor"}
STRUCTURE_LABELS = {"direct_a": "A", "mediated_b": "B"}
EXPECTED_GROUPS = {
    (domain, structure, cue, k)
    for domain in ("coin_city", "coin_harbor")
    for structure in ("direct_a", "mediated_b")
    for cue in ("correct", "none", "misleading")
    for k in (0, 2, 4, 8)
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latest_ledger() -> Path:
    paths = sorted(RUNS.glob("coin_city_structural_*.json"))
    if not paths:
        raise SystemExit("no Coin City structural-transfer ledger found")
    return paths[-1]


def validate_ledger(ledger: dict) -> list[dict]:
    if ledger.get("protocol") != "coin_city_structural_transfer_v1":
        raise AssertionError("unexpected protocol")
    confirmatory = [row for row in ledger.get("full_training", [])
                    if row.get("kind") == "confirmatory"]
    diagnostic = [row for row in ledger.get("full_training", [])
                  if row.get("kind") == "diagnostic"]
    bases = ledger.get("base_evaluations", [])
    expected_confirmatory = {
        (model, arm, seed) for model in MODELS for arm in ARMS for seed in SEEDS
    }
    actual_confirmatory = {
        (row.get("model"), row.get("arm"), row.get("seed")) for row in confirmatory
    }
    if len(confirmatory) != 18 or actual_confirmatory != expected_confirmatory:
        raise AssertionError("ledger is not the exact 3-model x 2-arm x 3-seed grid")
    expected_diagnostic = {("qwen3_4b", "structureless", seed) for seed in SEEDS}
    actual_diagnostic = {
        (row.get("model"), row.get("arm"), row.get("seed")) for row in diagnostic
    }
    if len(diagnostic) != 3 or actual_diagnostic != expected_diagnostic:
        raise AssertionError("ledger is not the exact three-seed structureless diagnostic")
    if len(bases) != 3 or {row.get("model") for row in bases} != set(MODELS):
        raise AssertionError("ledger is not the exact three-model base roster")
    rows = confirmatory + diagnostic + [dict(row, kind="base") for row in bases]
    ids = [str(row.get("job_id", "")) for row in rows]
    if any(not job_id.isdigit() for job_id in ids) or len(set(ids)) != 24:
        raise AssertionError("scientific roster must contain 24 distinct numeric job IDs")
    return rows


def report_root(item: dict) -> Path:
    job_id = str(item["job_id"])
    if item["kind"] == "base":
        pattern = f"base_{item['model']}_j{job_id}"
    else:
        pattern = f"{item['arm']}_{item['model']}_s{item['seed']}_*_j{job_id}"
    roots = sorted(REPORTS.glob(pattern))
    if len(roots) != 1:
        raise AssertionError(f"job {job_id}: expected one report root, found {roots}")
    return roots[0]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_rows(rows: list[dict], item: dict, decode: str) -> frozenset[tuple]:
    draws = 1 if decode == "greedy" else 5
    arm = "base" if item["kind"] == "base" else item["arm"]
    seed = 0 if item["kind"] == "base" else int(item["seed"])
    identity = {
        (row.get("model"), row.get("arm"), int(row.get("seed", -1)), row.get("decode"))
        for row in rows
    }
    expected_identity = {(item["model"], arm, seed, decode)}
    if identity != expected_identity:
        raise AssertionError(
            f"job {item['job_id']} {decode}: identity {identity}, expected {expected_identity}"
        )
    if len(rows) != 1440 * draws:
        raise AssertionError(
            f"job {item['job_id']} {decode}: expected {1440 * draws} rows, found {len(rows)}"
        )
    by_task: dict[str, list[int]] = defaultdict(list)
    metadata: dict[str, tuple] = {}
    groups: dict[tuple, set[str]] = defaultdict(set)
    for row in rows:
        task_id = str(row["task_id"])
        meta = (
            str(row["domain"]), str(row["target_structure"]), str(row["cue"]),
            int(row["k"]), str(row["pair_id"]),
        )
        if task_id in metadata and metadata[task_id] != meta:
            raise AssertionError(f"job {item['job_id']}: inconsistent task metadata")
        metadata[task_id] = meta
        by_task[task_id].append(int(row["draw"]))
        groups[meta[:4]].add(task_id)
    if set(groups) != EXPECTED_GROUPS or any(len(tasks) != 30 for tasks in groups.values()):
        raise AssertionError(f"job {item['job_id']} {decode}: incomplete 48-cell grid")
    expected_draws = list(range(draws))
    if len(by_task) != 1440 or any(sorted(values) != expected_draws for values in by_task.values()):
        raise AssertionError(f"job {item['job_id']} {decode}: missing/duplicate task draws")
    return frozenset((task_id, *meta) for task_id, meta in metadata.items())


def collect(ledger_rows: list[dict]) -> tuple[list[dict], list[Path]]:
    all_rows: list[dict] = []
    paths: list[Path] = []
    reference: frozenset[tuple] | None = None
    problems = []
    for item in ledger_rows:
        try:
            root = report_root(item)
            greedy = sorted(root.glob("greedy.scores.jsonl"))
            stochastic = sorted(root.glob("stochastic_n5.scores.jsonl"))
            if len(greedy) != 1 or len(stochastic) != 1:
                raise AssertionError(
                    f"job {item['job_id']}: greedy={greedy}, stochastic={stochastic}"
                )
            for decode, path in (("greedy", greedy[0]), ("stochastic", stochastic[0])):
                rows = read_jsonl(path)
                universe = validate_rows(rows, item, decode)
                if reference is None:
                    reference = universe
                elif universe != reference:
                    raise AssertionError(
                        f"job {item['job_id']} {decode}: endpoint task universe differs"
                    )
                all_rows.extend(rows)
                paths.append(path)
        except (AssertionError, OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            problems.append(str(error))
    if problems:
        raise SystemExit("registered Coin City grid is incomplete:\n" + "\n".join(problems))
    if len(paths) != 48 or len(all_rows) != 24 * (1440 + 7200):
        raise AssertionError("complete roster did not yield exactly 48 endpoint files")
    return all_rows, paths


def episode_values(rows: list[dict], selectors: dict, key: str = "task_id",
                   key_transform=None) -> dict[int, dict[str, float]]:
    grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if all(row.get(name) == value for name, value in selectors.items()):
            episode = str(row[key])
            if key_transform is not None:
                episode = key_transform(episode)
            grouped[int(row["seed"])][episode].append(float(row["response_mae"]))
    return {
        seed: {episode: float(np.mean(values)) for episode, values in episodes.items()}
        for seed, episodes in grouped.items()
    }


def mean_cell(rows: list[dict], selectors: dict, key: str = "task_id") -> float:
    values = episode_values(rows, selectors, key)
    means = [np.mean(list(episodes.values())) for episodes in values.values() if episodes]
    return float(np.mean(means)) if means else float("nan")


def paired_interval(rows: list[dict], left: dict, right: dict, *, key: str,
                    repetitions: int, seed: int) -> tuple[float, float, float]:
    """Return left-minus-right with seed-first, paired-episode interval."""
    left_values = episode_values(rows, left, key)
    right_values = episode_values(rows, right, key)
    seed_differences: dict[int, np.ndarray] = {}
    for training_seed in SEEDS:
        common = sorted(set(left_values.get(training_seed, {})) &
                        set(right_values.get(training_seed, {})))
        if not common:
            raise AssertionError(f"no paired episodes for seed {training_seed}")
        seed_differences[training_seed] = np.asarray([
            left_values[training_seed][episode] - right_values[training_seed][episode]
            for episode in common
        ])
    estimate = float(np.mean([np.mean(values) for values in seed_differences.values()]))
    rng = np.random.default_rng(seed)
    draws = []
    seed_array = np.asarray(SEEDS)
    for _ in range(repetitions):
        sampled = rng.choice(seed_array, len(seed_array), replace=True)
        means = []
        for selected in sampled:
            values = seed_differences[int(selected)]
            means.append(float(np.mean(rng.choice(values, len(values), replace=True))))
        draws.append(float(np.mean(means)))
    low, high = np.quantile(draws, (0.025, 0.975))
    return estimate, float(low), float(high)


def pair_family(pair_id: str) -> str:
    """Remove only the terminal evidence-depth component from a frozen pair ID."""
    family = re.sub(r":k(?:0|2|4|8)$", "", pair_id)
    if family == pair_id:
        raise AssertionError(f"pair ID lacks a registered terminal k component: {pair_id}")
    return family


def paired_override_interval(rows: list[dict], common: dict, *, repetitions: int,
                             seed: int) -> tuple[float, float, float]:
    """Estimate (misleading-correct at k=8) minus the same penalty at k=0.

    Pairing follows the frozen latent episode family across evidence depths. A
    negative estimate means observations reduced reliance on the misleading hint.
    """
    differences: dict[int, np.ndarray] = {}
    for training_seed in SEEDS:
        penalties = {}
        for k in (0, 8):
            misleading = episode_values(
                rows, {**common, "cue": "misleading", "k": k}, "pair_id", pair_family
            ).get(training_seed, {})
            correct = episode_values(
                rows, {**common, "cue": "correct", "k": k}, "pair_id", pair_family
            ).get(training_seed, {})
            shared = sorted(set(misleading) & set(correct))
            if not shared:
                raise AssertionError(f"no paired cue episodes for seed {training_seed}, k={k}")
            penalties[k] = {
                episode: misleading[episode] - correct[episode] for episode in shared
            }
        families = sorted(set(penalties[0]) & set(penalties[8]))
        if not families:
            raise AssertionError(f"no cross-k episode families for seed {training_seed}")
        differences[training_seed] = np.asarray([
            penalties[8][episode] - penalties[0][episode] for episode in families
        ])
    estimate = float(np.mean([np.mean(values) for values in differences.values()]))
    rng = np.random.default_rng(seed)
    draws = []
    seed_array = np.asarray(SEEDS)
    for _ in range(repetitions):
        sampled = rng.choice(seed_array, len(seed_array), replace=True)
        means = []
        for selected in sampled:
            values = differences[int(selected)]
            means.append(float(np.mean(rng.choice(values, len(values), replace=True))))
        draws.append(float(np.mean(means)))
    low, high = np.quantile(draws, (0.025, 0.975))
    return estimate, float(low), float(high)


def fmt(value: float) -> str:
    return "--" if not np.isfinite(value) else f"{value:.2f}"


def fmt_interval(values: tuple[float, float, float]) -> str:
    estimate, low, high = values
    return f"{estimate:.2f} [{low:.2f},{high:.2f}]"


def render_primary(rows: list[dict], repetitions: int, bootstrap_seed: int) -> str:
    lines = [
        "% generated by coin_city_structural/make_paper_outputs.py -- do not edit",
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Model & Decode & Base & Prior & Causal & $\Delta_{\rm causal}$ [95\% CI] \\",
        r"\midrule",
    ]
    for model in MODELS:
        for decode in ("greedy", "stochastic"):
            common = {"model": model, "decode": decode, "domain": "coin_harbor",
                      "target_structure": "mediated_b", "cue": "correct"}
            causal = mean_cell(rows, {**common, "arm": "causal"})
            prior = mean_cell(rows, {**common, "arm": "population_prior"})
            base = mean_cell(rows, {**common, "arm": "base"})
            contrast = paired_interval(
                rows, {**common, "arm": "population_prior"},
                {**common, "arm": "causal"}, key="task_id",
                repetitions=repetitions, seed=bootstrap_seed,
            )
            lines.append(
                f"{MODEL_LABELS[model]} & {decode.capitalize()} & {fmt(base)} & "
                f"{fmt(prior)} & {fmt(causal)} & {fmt_interval(contrast)} " + r"\\"
            )
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def render_cues(rows: list[dict], repetitions: int, bootstrap_seed: int) -> str:
    lines = [
        "% generated by coin_city_structural/make_paper_outputs.py -- do not edit",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Model & $k$ & Correct & Absent & Misleading & Absent $-$ correct [95\% CI] & Mislead. $-$ correct [95\% CI] \\",
        r"\midrule",
    ]
    for model in MODELS:
        for k in (0, 2, 4, 8):
            common = {"model": model, "arm": "causal", "decode": "stochastic",
                      "domain": "coin_harbor", "target_structure": "mediated_b", "k": k}
            correct = mean_cell(rows, {**common, "cue": "correct"}, "pair_id")
            absent = mean_cell(rows, {**common, "cue": "none"}, "pair_id")
            misleading = mean_cell(rows, {**common, "cue": "misleading"}, "pair_id")
            absent_contrast = paired_interval(
                rows, {**common, "cue": "none"}, {**common, "cue": "correct"},
                key="pair_id", repetitions=repetitions,
                seed=bootstrap_seed + 100 + k,
            )
            misleading_contrast = paired_interval(
                rows, {**common, "cue": "misleading"}, {**common, "cue": "correct"},
                key="pair_id", repetitions=repetitions, seed=bootstrap_seed + k,
            )
            lines.append(
                f"{MODEL_LABELS[model]} & {k} & {fmt(correct)} & {fmt(absent)} & "
                f"{fmt(misleading)} & {fmt_interval(absent_contrast)} & "
                f"{fmt_interval(misleading_contrast)} " + r"\\"
            )
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def render_diagnostics(rows: list[dict], repetitions: int, bootstrap_seed: int) -> str:
    lines = [
        "% generated by coin_city_structural/make_paper_outputs.py -- do not edit",
        r"\begin{tabular}{lllrrrrr}", r"\toprule",
        r"Model & Domain & Structure & Base & Prior & Causal & $\Delta_{\rm causal}$ [95\% CI] & Structureless \\",
        r"\midrule",
    ]
    for model in MODELS:
        for domain in ("coin_city", "coin_harbor"):
            for structure in ("direct_a", "mediated_b"):
                common = {"model": model, "decode": "stochastic", "domain": domain,
                          "target_structure": structure, "cue": "correct"}
                values = {
                    arm: mean_cell(rows, {**common, "arm": arm})
                    for arm in ("base", "population_prior", "causal", "structureless")
                }
                contrast = paired_interval(
                    rows, {**common, "arm": "population_prior"},
                    {**common, "arm": "causal"}, key="task_id",
                    repetitions=repetitions,
                    seed=bootstrap_seed + 1000 * (1 + MODELS.index(model))
                    + 100 * (domain == "coin_harbor")
                    + 10 * (structure == "mediated_b"),
                )
                lines.append(
                    f"{MODEL_LABELS[model]} & {DOMAIN_LABELS[domain]} & "
                    f"{STRUCTURE_LABELS[structure]} & {fmt(values['base'])} & "
                    f"{fmt(values['population_prior'])} & {fmt(values['causal'])} & "
                    f"{fmt_interval(contrast)} & {fmt(values['structureless'])} " + r"\\"
                )
            lines.append(r"\addlinespace")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def render_summary(rows: list[dict], repetitions: int, bootstrap_seed: int) -> str:
    pieces = []
    favorable = {"greedy": 0, "stochastic": 0}
    for decode in ("greedy", "stochastic"):
        effects = []
        for model in MODELS:
            common = {"model": model, "decode": decode, "domain": "coin_harbor",
                      "target_structure": "mediated_b", "cue": "correct"}
            contrast = paired_interval(
                rows, {**common, "arm": "population_prior"},
                {**common, "arm": "causal"}, key="task_id",
                repetitions=repetitions, seed=bootstrap_seed,
            )
            favorable[decode] += int(contrast[0] > 0)
            effects.append(f"{MODEL_LABELS[model]} {fmt_interval(contrast)}")
        label = "Greedy" if decode == "greedy" else "Five-draw stochastic"
        pieces.append(f"{label}: " + "; ".join(effects) + ".")
    cue_pieces = []
    for model in MODELS:
        common = {"model": model, "arm": "causal", "decode": "stochastic",
                  "domain": "coin_harbor", "target_structure": "mediated_b"}
        absent = paired_interval(
            rows, {**common, "cue": "none"}, {**common, "cue": "correct"},
            key="pair_id", repetitions=repetitions, seed=bootstrap_seed + 3000,
        )
        misleading = paired_interval(
            rows, {**common, "cue": "misleading"}, {**common, "cue": "correct"},
            key="pair_id", repetitions=repetitions, seed=bootstrap_seed + 4000,
        )
        override = paired_override_interval(
            rows, common, repetitions=repetitions, seed=bootstrap_seed + 5000,
        )
        cue_pieces.append(
            f"{MODEL_LABELS[model]}: absent-minus-correct {fmt_interval(absent)}, "
            f"misleading-minus-correct {fmt_interval(misleading)}, and the "
            f"$k=8$ minus $k=0$ change in misleading penalty {fmt_interval(override)}"
        )
    return (
        r"\paragraph{Registered endpoint result.} "
        r"In the prespecified joint domain-plus-structure transfer cell, "
        r"population-prior minus causal response-MAE estimates (positive favors "
        r"causal training) were " + " ".join(pieces) + " "
        f"The point estimate favored causal training for {favorable['greedy']}/3 "
        f"models under greedy decoding and {favorable['stochastic']}/3 under the "
        r"repeated stochastic endpoint; the table reports the registered "
        r"seed-first, paired-episode intervals. In the same joint-transfer cell, "
        r"the prespecified stochastic cue contrasts were " + "; ".join(cue_pieces) +
        r". Negative values for the final contrast mean that target observations "
        r"reduced the misleading-hint penalty."
        "\n"
    )


def registered_estimates(rows: list[dict], repetitions: int,
                         bootstrap_seed: int) -> dict:
    """Machine-readable counterpart of every inferential table entry."""
    primary_rows = []
    transfer_cells = []
    cue_rows = []
    cue_overall_rows = []
    override_rows = []
    for model in MODELS:
        for decode in ("greedy", "stochastic"):
            common = {"model": model, "decode": decode, "domain": "coin_harbor",
                      "target_structure": "mediated_b", "cue": "correct"}
            contrast = paired_interval(
                rows, {**common, "arm": "population_prior"},
                {**common, "arm": "causal"}, key="task_id",
                repetitions=repetitions, seed=bootstrap_seed,
            )
            primary_rows.append({
                "model": model, "decode": decode,
                "mean_response_mae": {
                    arm: mean_cell(rows, {**common, "arm": arm})
                    for arm in ("base", "population_prior", "causal")
                },
                "population_prior_minus_causal": {
                    "estimate": contrast[0], "ci95_low": contrast[1],
                    "ci95_high": contrast[2]
                },
            })
        for domain in ("coin_city", "coin_harbor"):
            for structure in ("direct_a", "mediated_b"):
                common = {"model": model, "decode": "stochastic", "domain": domain,
                          "target_structure": structure, "cue": "correct"}
                estimate, low, high = paired_interval(
                    rows, {**common, "arm": "population_prior"},
                    {**common, "arm": "causal"}, key="task_id",
                    repetitions=repetitions,
                    seed=bootstrap_seed + 1000 * (1 + MODELS.index(model))
                    + 100 * (domain == "coin_harbor")
                    + 10 * (structure == "mediated_b"),
                )
                transfer_cells.append({
                    "model": model, "decode": "stochastic", "domain": domain,
                    "target_structure": structure,
                    "population_prior_minus_causal": estimate,
                    "ci95_low": low, "ci95_high": high,
                })
        common = {"model": model, "arm": "causal", "decode": "stochastic",
                  "domain": "coin_harbor", "target_structure": "mediated_b"}
        for k in (0, 2, 4, 8):
            values = {
                cue: mean_cell(rows, {**common, "cue": cue, "k": k}, "pair_id")
                for cue in ("correct", "none", "misleading")
            }
            absent = paired_interval(
                rows, {**common, "cue": "none", "k": k},
                {**common, "cue": "correct", "k": k}, key="pair_id",
                repetitions=repetitions, seed=bootstrap_seed + 100 + k,
            )
            misleading = paired_interval(
                rows, {**common, "cue": "misleading", "k": k},
                {**common, "cue": "correct", "k": k}, key="pair_id",
                repetitions=repetitions, seed=bootstrap_seed + k,
            )
            cue_rows.append({
                "model": model, "k": k, "mean_response_mae": values,
                "absent_minus_correct": {
                    "estimate": absent[0], "ci95_low": absent[1], "ci95_high": absent[2]
                },
                "misleading_minus_correct": {
                    "estimate": misleading[0], "ci95_low": misleading[1],
                    "ci95_high": misleading[2]
                },
            })
        override = paired_override_interval(
            rows, common, repetitions=repetitions, seed=bootstrap_seed + 5000,
        )
        absent_overall = paired_interval(
            rows, {**common, "cue": "none"}, {**common, "cue": "correct"},
            key="pair_id", repetitions=repetitions, seed=bootstrap_seed + 3000,
        )
        misleading_overall = paired_interval(
            rows, {**common, "cue": "misleading"}, {**common, "cue": "correct"},
            key="pair_id", repetitions=repetitions, seed=bootstrap_seed + 4000,
        )
        cue_overall_rows.append({
            "model": model,
            "absent_minus_correct": {
                "estimate": absent_overall[0], "ci95_low": absent_overall[1],
                "ci95_high": absent_overall[2]
            },
            "misleading_minus_correct": {
                "estimate": misleading_overall[0], "ci95_low": misleading_overall[1],
                "ci95_high": misleading_overall[2]
            },
        })
        override_rows.append({
            "model": model,
            "contrast": "(misleading_minus_correct_at_k8)_minus_(at_k0)",
            "estimate": override[0], "ci95_low": override[1], "ci95_high": override[2],
        })
    return {
        "primary": primary_rows,
        "transfer_cells": transfer_cells,
        "cue_by_k": cue_rows,
        "cue_overall": cue_overall_rows,
        "evidence_override": override_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--output", type=Path, default=REPORTS / "registered_results.json")
    parser.add_argument("--paper-root", type=Path, action="append", default=[])
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260818)
    args = parser.parse_args()
    ledger_path = args.ledger or latest_ledger()
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    roster = validate_ledger(ledger)
    rows, paths = collect(roster)
    primary = render_primary(rows, args.bootstrap_repetitions, args.bootstrap_seed)
    cues = render_cues(rows, args.bootstrap_repetitions, args.bootstrap_seed)
    diagnostics = render_diagnostics(rows, args.bootstrap_repetitions, args.bootstrap_seed)
    summary = render_summary(rows, args.bootstrap_repetitions, args.bootstrap_seed)
    estimates = registered_estimates(rows, args.bootstrap_repetitions, args.bootstrap_seed)
    result = {
        "protocol": ledger["protocol"],
        "ledger": str(ledger_path.relative_to(REPO)),
        "ledger_sha256": sha256(ledger_path),
        "score_files": [str(path.relative_to(REPO)) for path in paths],
        "score_file_sha256": {str(path.relative_to(REPO)): sha256(path) for path in paths},
        "analysis_code_sha256": sha256(Path(__file__).resolve()),
        "row_draws": len(rows),
        "validated_jobs": len(roster),
        "validated_tasks_per_job": 1440,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.bootstrap_seed,
        "variance_unit": "training seed; paired episode within seed; draws averaged within episode",
        "registered_estimates": estimates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paper_roots = args.paper_root or [REPO / "paper", REPO / "paper" / "ICLR"]
    for paper_root in paper_roots:
        tables = paper_root / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        (tables / "exp3_coin_structural_primary.tex").write_text(primary, encoding="utf-8")
        (tables / "exp3_coin_structural_cues.tex").write_text(cues, encoding="utf-8")
        (tables / "exp3_coin_structural_diagnostics.tex").write_text(
            diagnostics, encoding="utf-8"
        )
        (tables / "exp3_coin_structural_summary.tex").write_text(summary, encoding="utf-8")
    print(f"validated {len(roster)} jobs and {len(paths)} endpoint files")
    print(f"result -> {args.output}")


if __name__ == "__main__":
    main()
