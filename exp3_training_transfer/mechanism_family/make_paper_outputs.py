#!/usr/bin/env python3
"""Aggregate one complete registered C3 grid and render its paper tables."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from contrasts import (
    hierarchical_base_contrasts,
    hierarchical_disclosure_benefit_contrasts,
    hierarchical_model_benefit_contrasts,
    hierarchical_structureless_contrasts,
)
from report import hierarchical_arm_contrasts, read_scores, summarize

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
MODEL_LABELS = {
    "qwen3_4b": "Qwen3-4B",
    "qwen3_8b": "Qwen3-8B",
    "llama3_1_8b": "Llama-3.1-8B",
}
BLOCK_LABELS = {
    "topology_composition": "topology",
    "nonlinear_composition": "nonlinear",
    "mixed_composition": "mixed",
    "parameter_extrapolation": "extrapolation",
}
DISCLOSURES = ("disclosed", "undisclosed")
MODELS = ("qwen3_4b", "qwen3_8b", "llama3_1_8b")
CONFIRMATORY_ARMS = ("causal_family", "population_prior")
TRAINING_SEEDS = (42, 43, 44)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_full_ledger(payload: dict) -> bool:
    training = [item for item in payload.get("submissions", [])
                if item.get("kind") == "training"]
    bases = [item for item in payload.get("submissions", [])
             if item.get("kind") == "base_evaluation"]
    return len(training) == 42 and len(bases) == 6


def validate_full_ledger(payload: dict) -> None:
    if payload.get("protocol") != "c3_mechanism_disclosure_v4":
        raise AssertionError("full ledger must use the structured-decoding v4 protocol")
    decoding = payload.get("structured_decoding", {})
    if (decoding.get("kind"), decoding.get("backend"), decoding.get("value_constraints")) != (
            "gbnf", "xgrammar", "none"):
        raise AssertionError("full ledger must register syntax-only GBNF decoding")

    training = [item for item in payload.get("submissions", [])
                if item.get("kind") == "training"]
    expected_training = {
        (disclosure, model, arm, seed)
        for disclosure in DISCLOSURES
        for model in MODELS
        for arm in CONFIRMATORY_ARMS
        for seed in TRAINING_SEEDS
    } | {
        (disclosure, "qwen3_4b", "structureless", seed)
        for disclosure in DISCLOSURES
        for seed in TRAINING_SEEDS
    }
    actual_training = [
        (item.get("disclosure"), item.get("model"), item.get("arm"), item.get("seed"))
        for item in training
    ]
    if len(actual_training) != 42 or set(actual_training) != expected_training:
        raise AssertionError("full ledger does not contain the exact registered training roster")
    if len(set(actual_training)) != len(actual_training):
        raise AssertionError("full ledger contains duplicate training cells")

    bases = [item for item in payload.get("submissions", [])
             if item.get("kind") == "base_evaluation"]
    expected_bases = {(disclosure, model) for disclosure in DISCLOSURES for model in MODELS}
    actual_bases = [(item.get("disclosure"), item.get("model")) for item in bases]
    if len(actual_bases) != 6 or set(actual_bases) != expected_bases:
        raise AssertionError("full ledger does not contain the exact registered base roster")
    if len(set(actual_bases)) != len(actual_bases):
        raise AssertionError("full ledger contains duplicate base cells")

    job_ids = [str(item.get("job_id")) for item in training + bases]
    if len(set(job_ids)) != 48 or any(not job_id.isdigit() for job_id in job_ids):
        raise AssertionError("full ledger must contain 48 distinct numeric Slurm job IDs")


def latest_full_ledger() -> Path:
    for path in sorted(RUNS.glob("c3_mechanism_*.json"), reverse=True):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if is_full_ledger(payload):
            validate_full_ledger(payload)
            return path
    raise SystemExit("no complete 42-training/6-base C3 ledger found")


def report_root(item: dict) -> Path:
    job_id = str(item["job_id"])
    if item["kind"] == "training":
        pattern = (
            f"{item['disclosure']}_{item['arm']}_{item['model']}_"
            f"s{item['seed']}_*_j{job_id}"
        )
    elif item["kind"] == "base_evaluation":
        pattern = f"base_{item['disclosure']}_{item['model']}_j{job_id}"
    else:
        raise AssertionError(f"unknown submission kind: {item['kind']}")
    roots = sorted(REPORTS.glob(pattern))
    if len(roots) != 1:
        raise AssertionError(f"job {job_id}: expected one report root, found {roots}")
    return roots[0]


def validate_score_rows(rows: list[dict], item: dict, decode: str) -> frozenset[tuple]:
    """Require one complete, correctly identified copy of the paired endpoint grid."""
    draws = 1 if decode == "greedy" else 5
    expected_arm = "base" if item["kind"] == "base_evaluation" else item["arm"]
    expected_seed = 0 if item["kind"] == "base_evaluation" else int(item["seed"])
    expected_identity = {
        (
            item["model"],
            item["disclosure"],
            expected_arm,
            expected_seed,
            decode,
        )
    }
    try:
        observed_identity = {
            (
                row["model"],
                row["disclosure"],
                row["arm"],
                int(row["seed"]),
                row["decode"],
            )
            for row in rows
        }
    except (KeyError, TypeError, ValueError) as error:
        raise AssertionError(
            f"job {item['job_id']} {decode}: malformed score-row identity"
        ) from error
    if observed_identity != expected_identity:
        raise AssertionError(
            f"job {item['job_id']} {decode}: expected identity "
            f"{expected_identity}, found {observed_identity}"
        )

    expected_rows = 720 * draws
    if len(rows) != expected_rows:
        raise AssertionError(
            f"job {item['job_id']} {decode}: expected {expected_rows} rows, "
            f"found {len(rows)}"
        )

    draws_by_task: dict[str, list[int]] = defaultdict(list)
    task_metadata: dict[str, tuple] = {}
    tasks_by_block_horizon: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        try:
            task_id = str(row["task_id"])
            block = str(row["block"])
            horizon = int(row["k"])
            world = str(row["world"])
            draw = int(row["draw"])
        except (KeyError, TypeError, ValueError) as error:
            raise AssertionError(
                f"job {item['job_id']} {decode}: malformed task/draw metadata"
            ) from error
        metadata = (block, horizon, world)
        previous = task_metadata.setdefault(task_id, metadata)
        if previous != metadata:
            raise AssertionError(
                f"job {item['job_id']} {decode}: task {task_id} has "
                f"inconsistent metadata {previous} and {metadata}"
            )
        draws_by_task[task_id].append(draw)
        tasks_by_block_horizon[(block, horizon)].add(task_id)

    expected_groups = {
        (block, horizon)
        for block in BLOCK_LABELS
        for horizon in (3, 6, 9)
    }
    if set(tasks_by_block_horizon) != expected_groups:
        raise AssertionError(
            f"job {item['job_id']} {decode}: incomplete block/horizon grid"
        )
    incorrect_group_sizes = {
        group: len(task_ids)
        for group, task_ids in tasks_by_block_horizon.items()
        if len(task_ids) != 60
    }
    if incorrect_group_sizes:
        raise AssertionError(
            f"job {item['job_id']} {decode}: expected 60 tasks per "
            f"block/horizon, found {incorrect_group_sizes}"
        )
    if len(draws_by_task) != 720:
        raise AssertionError(
            f"job {item['job_id']} {decode}: expected 720 unique tasks, "
            f"found {len(draws_by_task)}"
        )
    expected_draws = list(range(draws))
    incorrect_draws = {
        task_id: sorted(task_draws)
        for task_id, task_draws in draws_by_task.items()
        if sorted(task_draws) != expected_draws
    }
    if incorrect_draws:
        examples = dict(list(incorrect_draws.items())[:5])
        raise AssertionError(
            f"job {item['job_id']} {decode}: duplicate or missing draws; "
            f"examples {examples}"
        )
    return frozenset(
        (task_id, *metadata) for task_id, metadata in task_metadata.items()
    )


def score_files(ledger: dict) -> list[Path]:
    paths = []
    missing = []
    reference_tasks: frozenset[tuple] | None = None
    for item in ledger["submissions"]:
        if item.get("kind") not in {"training", "base_evaluation"}:
            continue
        try:
            root = report_root(item)
        except AssertionError as error:
            missing.append(str(error))
            continue
        if item["kind"] == "training":
            greedy = sorted(root.glob("debug_*/eval_results/*.scores.jsonl"))
        else:
            greedy = sorted(root.glob("greedy.scores.jsonl"))
        stochastic = sorted(root.glob("stochastic_n*.scores.jsonl"))
        if len(greedy) != 1 or len(stochastic) != 1:
            missing.append(
                f"job {item['job_id']}: greedy={greedy}, stochastic={stochastic}"
            )
            continue
        for decode, path in (("greedy", greedy[0]), ("stochastic", stochastic[0])):
            try:
                task_contract = validate_score_rows(read_scores([path]), item, decode)
            except AssertionError as error:
                missing.append(str(error))
                continue
            if reference_tasks is None:
                reference_tasks = task_contract
            elif task_contract != reference_tasks:
                missing.append(
                    f"job {item['job_id']} {decode}: endpoint task universe "
                    "does not match the paired registered grid"
                )
                continue
            paths.append(path)
    if missing:
        raise SystemExit("registered grid is incomplete:\n" + "\n".join(missing))
    if len(paths) != 96:
        raise AssertionError(f"expected 96 score files, found {len(paths)}")
    return paths


def add_overall_rows(rows: list[dict]) -> list[dict]:
    """Add a pooled structural-OOD block without discarding block-level rows."""
    result = list(rows)
    for row in rows:
        if row.get("block") == "overall":
            continue
        pooled = dict(row)
        pooled["task_id"] = f"{row['block']}::{row['task_id']}"
        pooled["block"] = "overall"
        result.append(pooled)
    return result


def cell_mean(rows: list[dict], selectors: dict, metric: str) -> float:
    """Average draws within task, tasks within seed, and then training seeds."""
    grouped: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if all(row.get(key) == value for key, value in selectors.items()):
            value = row.get(metric)
            if value is not None:
                grouped[int(row["seed"])][str(row["task_id"])].append(float(value))
    seed_means = []
    for tasks in grouped.values():
        task_means = [float(np.mean(values)) for values in tasks.values()]
        if task_means:
            seed_means.append(float(np.mean(task_means)))
    if not seed_means:
        return float("nan")
    return float(np.mean(seed_means))


def fmt(value: float, digits: int = 2) -> str:
    return "--" if not np.isfinite(value) else f"{value:.{digits}f}"


def interval(contrast: dict | None) -> str:
    if contrast is None:
        return "--"
    return (
        f"{contrast['estimate']:.2f} "
        f"[{contrast['ci95_low']:.2f},{contrast['ci95_high']:.2f}]"
    )


def render_primary_table(rows: list[dict], arm_contrasts: list[dict],
                         decode: str) -> str:
    index = {
        (item["model"], item["disclosure"], item["block"]): item
        for item in arm_contrasts if item["decode"] == decode
    }
    lines = [
        "% generated by mechanism_family/make_paper_outputs.py -- do not edit",
        r"\begin{tabular}{lllrrrrr}",
        r"\toprule",
        r"Disclosure & Model & Held-out block & Base & Prior & Causal & "
        r"$\Delta_{\rm causal}$ [95\% CI] & Mech. acc. \\",
        r"\midrule",
    ]
    for disclosure in ("disclosed", "undisclosed"):
        for model in ("qwen3_4b", "qwen3_8b", "llama3_1_8b"):
            for block in BLOCK_LABELS:
                common = {
                    "model": model, "disclosure": disclosure,
                    "decode": decode, "block": block,
                }
                base = cell_mean(rows, {**common, "arm": "base"}, "response_mae")
                prior = cell_mean(
                    rows, {**common, "arm": "population_prior"}, "response_mae"
                )
                causal = cell_mean(
                    rows, {**common, "arm": "causal_family"}, "response_mae"
                )
                accuracy = cell_mean(
                    rows, {**common, "arm": "causal_family"}, "mechanism_correct"
                )
                effect = index.get((model, disclosure, block))
                lines.append(
                    f"{disclosure.capitalize()} & {MODEL_LABELS[model]} & "
                    f"{BLOCK_LABELS[block]} & {fmt(base)} & {fmt(prior)} & "
                    f"{fmt(causal)} & {interval(effect)} & {fmt(accuracy, 3)} "
                    + r"\\"
                )
            lines.append(r"\addlinespace")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def render_overall_table(rows: list[dict], arm_contrasts: list[dict]) -> str:
    index = {
        (item["decode"], item["model"], item["disclosure"]): item
        for item in arm_contrasts if item["block"] == "overall"
    }
    lines = [
        "% generated by mechanism_family/make_paper_outputs.py -- do not edit",
        r"\begin{tabular}{lllrrrrr}",
        r"\toprule",
        r"Decode & Disclosure & Model & Base & Prior & Causal & "
        r"$\Delta_{\rm causal}$ [95\% CI] & Mech. acc. \\",
        r"\midrule",
    ]
    for decode in ("greedy", "stochastic"):
        for disclosure in ("disclosed", "undisclosed"):
            for model in ("qwen3_4b", "qwen3_8b", "llama3_1_8b"):
                common = {
                    "model": model, "disclosure": disclosure,
                    "decode": decode, "block": "overall",
                }
                base = cell_mean(rows, {**common, "arm": "base"}, "response_mae")
                prior = cell_mean(
                    rows, {**common, "arm": "population_prior"}, "response_mae"
                )
                causal = cell_mean(
                    rows, {**common, "arm": "causal_family"}, "response_mae"
                )
                accuracy = cell_mean(
                    rows, {**common, "arm": "causal_family"}, "mechanism_correct"
                )
                lines.append(
                    f"{decode.capitalize()} & {disclosure.capitalize()} & "
                    f"{MODEL_LABELS[model]} & {fmt(base)} & {fmt(prior)} & "
                    f"{fmt(causal)} & "
                    f"{interval(index.get((decode, model, disclosure)))} & "
                    f"{fmt(accuracy, 3)} " + r"\\"
                )
            lines.append(r"\addlinespace")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def render_secondary_table(result: dict) -> str:
    """Render every registered overall secondary contrast for both decoders."""
    base = {
        (item["decode"], item["model"], item["disclosure"]): item
        for item in result["hierarchical_base_contrasts"]
        if item["block"] == "overall"
    }
    structureless = {
        (item["decode"], item["disclosure"]): item
        for item in result["hierarchical_structureless_contrasts"]
        if item["block"] == "overall" and item["model"] == "qwen3_4b"
    }
    model = {
        (item["decode"], item["contrast"], item["disclosure"]): item
        for item in result["hierarchical_model_benefit_contrasts"]
        if item["block"] == "overall"
    }
    disclosure = {
        (item["decode"], item["model"]): item
        for item in result["hierarchical_disclosure_benefit_contrasts"]
        if item["block"] == "overall"
    }

    rows: list[tuple[str, str, dict | None, dict | None]] = []
    for disclosure_key in ("disclosed", "undisclosed"):
        for model_key in ("qwen3_4b", "qwen3_8b", "llama3_1_8b"):
            rows.append((
                r"Base $-$ causal",
                f"{disclosure_key.capitalize()} / {MODEL_LABELS[model_key]}",
                base.get(("greedy", model_key, disclosure_key)),
                base.get(("stochastic", model_key, disclosure_key)),
            ))
    rows.append(("", "", None, None))
    for disclosure_key in ("disclosed", "undisclosed"):
        rows.append((
            r"Structureless $-$ causal",
            f"{disclosure_key.capitalize()} / Qwen3-4B",
            structureless.get(("greedy", disclosure_key)),
            structureless.get(("stochastic", disclosure_key)),
        ))
    rows.append(("", "", None, None))
    model_labels = (
        (
            "qwen3_8b_minus_qwen3_4b_causal_benefit",
            "Qwen3-8B $-$ Qwen3-4B causal benefit",
        ),
        (
            "qwen3_8b_minus_llama3_1_8b_causal_benefit",
            "Qwen3-8B $-$ Llama-3.1-8B causal benefit",
        ),
    )
    for disclosure_key in ("disclosed", "undisclosed"):
        for contrast_key, label in model_labels:
            rows.append((
                label,
                disclosure_key.capitalize(),
                model.get(("greedy", contrast_key, disclosure_key)),
                model.get(("stochastic", contrast_key, disclosure_key)),
            ))
    rows.append(("", "", None, None))
    for model_key in ("qwen3_4b", "qwen3_8b", "llama3_1_8b"):
        rows.append((
            "Disclosed $-$ undisclosed causal benefit",
            MODEL_LABELS[model_key],
            disclosure.get(("greedy", model_key)),
            disclosure.get(("stochastic", model_key)),
        ))

    lines = [
        "% generated by mechanism_family/make_paper_outputs.py -- do not edit",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Contrast & Scope & Greedy [95\% CI] & Stochastic [95\% CI] \\",
        r"\midrule",
    ]
    for label, scope, greedy, stochastic in rows:
        if not label:
            lines.append(r"\addlinespace")
            continue
        lines.append(
            f"{label} & {scope} & {interval(greedy)} & "
            f"{interval(stochastic)} " + r"\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines)


def aggregate(rows: list[dict], repetitions: int, seed: int) -> dict:
    arm = hierarchical_arm_contrasts(rows, repetitions, seed)
    return {
        "summaries": summarize(rows),
        "hierarchical_arm_contrasts": arm,
        "hierarchical_base_contrasts":
            hierarchical_base_contrasts(rows, repetitions, seed),
        "hierarchical_structureless_contrasts":
            hierarchical_structureless_contrasts(rows, repetitions, seed),
        "hierarchical_model_benefit_contrasts":
            hierarchical_model_benefit_contrasts(rows, repetitions, seed),
        "hierarchical_disclosure_benefit_contrasts":
            hierarchical_disclosure_benefit_contrasts(rows, repetitions, seed),
        "variance_unit":
            "training seed; paired episode resampling nested within seed; draws averaged",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports" / "c3_mechanism_full_aggregate.json",
    )
    parser.add_argument("--greedy-latex", type=Path, action="append", default=[])
    parser.add_argument(
        "--stochastic-latex", type=Path, action="append", default=[]
    )
    parser.add_argument("--overall-latex", type=Path, action="append", default=[])
    parser.add_argument("--secondary-latex", type=Path, action="append", default=[])
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260814)
    args = parser.parse_args()

    ledger_path = args.ledger or latest_full_ledger()
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if not is_full_ledger(ledger):
        raise SystemExit("selected ledger is not the registered 42-training/6-base grid")
    validate_full_ledger(ledger)
    score_paths = score_files(ledger)
    rows = read_scores(score_paths)
    analysis_rows = add_overall_rows(rows)
    result = aggregate(analysis_rows, args.bootstrap_repetitions, args.bootstrap_seed)
    result.update({
        "ledger": str(ledger_path.relative_to(ROOT.parent.parent)),
        "ledger_sha256": sha256(ledger_path),
        "score_files": [
            str(path.relative_to(ROOT.parent.parent)) for path in score_paths
        ],
        "score_file_sha256": {
            str(path.relative_to(ROOT.parent.parent)): sha256(path)
            for path in score_paths
        },
        "analysis_code_sha256": {
            str(path.relative_to(ROOT.parent.parent)): sha256(path)
            for path in (Path(__file__).resolve(), ROOT / "contrasts.py", ROOT / "report.py")
        },
        "row_draws": len(rows),
        "analysis_row_draws_including_overall": len(analysis_rows),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for decode, latex_paths in (
        ("greedy", args.greedy_latex),
        ("stochastic", args.stochastic_latex),
    ):
        rendered = render_primary_table(
            analysis_rows, result["hierarchical_arm_contrasts"], decode
        ) + "\n"
        for path in latex_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered, encoding="utf-8")
    rendered_overall = render_overall_table(
        analysis_rows, result["hierarchical_arm_contrasts"]
    ) + "\n"
    for path in args.overall_latex:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered_overall, encoding="utf-8")
    rendered_secondary = render_secondary_table(result) + "\n"
    for path in args.secondary_latex:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered_secondary, encoding="utf-8")
    print(f"aggregated {len(rows)} row-draws from {len(score_paths)} score files")
    print(f"result -> {args.output}")


if __name__ == "__main__":
    main()
