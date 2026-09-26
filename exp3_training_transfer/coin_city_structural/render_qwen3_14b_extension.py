#!/usr/bin/env python3
"""Validate and render the prospectively frozen Qwen3-14B Coin City extension."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    import make_paper_outputs as core
except ModuleNotFoundError:  # Support package imports in the maintained tests.
    from . import make_paper_outputs as core


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
REPORTS = ROOT / "reports"
PARENT_RESULTS = REPORTS / "registered_results.json"
PROTOCOL = ROOT / "QWEN3_14B_SCALE_PROTOCOL.md"
PROTOCOL_VERSION = "coin_city_qwen3_14b_scale_v1"
MODEL = "qwen3_14b"
MODEL_LABEL = "Qwen3-14B"
ARMS = ("causal", "population_prior")
SEEDS = (42, 43, 44)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def interval(values: tuple[float, float, float]) -> dict[str, float]:
    estimate, low, high = values
    return {"estimate": estimate, "ci95_low": low, "ci95_high": high}


def validate_registration(ledger: dict, ledger_path: Path) -> list[dict]:
    if ledger.get("protocol_version") != PROTOCOL_VERSION:
        raise AssertionError("unexpected Qwen3-14B extension protocol")
    if ledger.get("dry_run") is not False:
        raise AssertionError("extension ledger is not a real submission")
    expected_hashes = {
        "protocol_sha256": sha256(PROTOCOL),
        "parent_results_sha256": sha256(PARENT_RESULTS),
        "renderer_sha256": sha256(Path(__file__).resolve()),
        "train_script_sha256": sha256(ROOT / "train.sh"),
        "base_eval_script_sha256": sha256(ROOT / "base_eval.sh"),
        "report_script_sha256": sha256(ROOT / "report.py"),
        "core_analysis_sha256": sha256(ROOT / "make_paper_outputs.py"),
    }
    for key, expected in expected_hashes.items():
        if ledger.get(key) != expected:
            raise AssertionError(f"{key} drifted after extension registration")

    parent = json.loads(PARENT_RESULTS.read_text(encoding="utf-8"))
    parent_ledger = REPO / parent["ledger"]
    if ledger.get("parent_ledger_sha256") != sha256(parent_ledger):
        raise AssertionError("original Experiment 3 ledger drifted")
    manifest = ROOT / "protocol" / "coin_city_structural_manifest.json"
    if ledger.get("parent_manifest_sha256") != sha256(manifest):
        raise AssertionError("original Experiment 3 manifest drifted")

    parent_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    frozen_dataset_sha256 = {
        relative: expected
        for relative, expected in parent_manifest.get("dataset_sha256", {}).items()
        if not Path(relative).name.startswith("cache-")
    }
    if ledger.get("frozen_dataset_sha256") != frozen_dataset_sha256:
        raise AssertionError("registered canonical dataset hash roster drifted")
    for relative, expected in frozen_dataset_sha256.items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise AssertionError(f"canonical dataset shard drifted: {relative}")

    training = ledger.get("training_jobs", [])
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    actual = {(row.get("arm"), row.get("seed")) for row in training}
    if len(training) != 6 or actual != expected:
        raise AssertionError("extension is not the exact two-arm by three-seed grid")
    base = ledger.get("base_evaluation")
    if not isinstance(base, dict) or base.get("model") != MODEL:
        raise AssertionError("extension base endpoint is missing")
    rows = [dict(row, kind="confirmatory", model=MODEL) for row in training]
    rows.append(dict(base, kind="base", model=MODEL))
    job_ids = [str(row.get("job_id", "")) for row in rows]
    if any(not value.isdigit() for value in job_ids) or len(set(job_ids)) != 7:
        raise AssertionError("extension must contain seven distinct numeric job IDs")
    if ledger_path.name != ledger.get("ledger_name"):
        raise AssertionError("extension ledger filename drifted")
    return rows


def report_root(item: dict) -> Path:
    job_id = str(item["job_id"])
    if item["kind"] == "base":
        pattern = f"base_{MODEL}_j{job_id}"
    else:
        pattern = f"scale_{item['arm']}_{MODEL}_s{item['seed']}_*_j{job_id}"
    matches = sorted(REPORTS.glob(pattern))
    if len(matches) != 1:
        raise AssertionError(f"job {job_id}: expected one report root, found {matches}")
    return matches[0]


def collect_extension(roster: list[dict]) -> tuple[list[dict], list[Path], frozenset]:
    all_rows: list[dict] = []
    paths: list[Path] = []
    reference: frozenset | None = None
    problems: list[str] = []
    for item in roster:
        try:
            root = report_root(item)
            for decode in ("greedy", "stochastic"):
                path = core.endpoint_path(root, decode)
                rows = core.read_jsonl(path)
                universe = core.validate_rows(rows, item, decode)
                if reference is None:
                    reference = universe
                elif universe != reference:
                    raise AssertionError(f"job {item['job_id']}: task universe drifted")
                all_rows.extend(rows)
                paths.append(path)
        except (AssertionError, OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            problems.append(str(error))
    if problems:
        raise SystemExit("Qwen3-14B extension is incomplete:\n" + "\n".join(problems))
    if len(paths) != 14 or len(all_rows) != 60_480 or reference is None:
        raise AssertionError("extension did not yield 14 files and 60,480 row-draws")
    return all_rows, paths, reference


def collect_parent_qwen8(ledger: dict, reference: frozenset) -> tuple[list[dict], list[Path]]:
    parent = json.loads(PARENT_RESULTS.read_text(encoding="utf-8"))
    if parent.get("protocol") != "coin_city_structural_transfer_v1":
        raise AssertionError("unexpected original Experiment 3 result")
    parent_hashes = parent.get("score_file_sha256", {})
    selected = [
        REPO / relative
        for relative in parent.get("score_files", [])
        if "qwen3_8b" in relative
    ]
    if len(selected) != 14:
        raise AssertionError("original result lacks the exact Qwen3-8B endpoint roster")
    rows: list[dict] = []
    for path in selected:
        relative = str(path.relative_to(REPO))
        if sha256(path) != parent_hashes.get(relative):
            raise AssertionError(f"frozen Qwen3-8B score file drifted: {relative}")
        rows.extend(core.read_jsonl(path))
    identities = {
        (row.get("model"), row.get("arm"), int(row.get("seed", -1)), row.get("decode"))
        for row in rows
    }
    expected = {
        ("qwen3_8b", arm, seed, decode)
        for arm in ARMS
        for seed in SEEDS
        for decode in ("greedy", "stochastic")
    } | {("qwen3_8b", "base", 0, decode) for decode in ("greedy", "stochastic")}
    if identities != expected or len(rows) != 60_480:
        raise AssertionError("frozen Qwen3-8B rows have an invalid identity roster")
    parent_universe = frozenset(
        (
            str(row["task_id"]),
            str(row["domain"]),
            str(row["target_structure"]),
            str(row["cue"]),
            int(row["k"]),
            str(row["pair_id"]),
        )
        for row in rows
    )
    if parent_universe != reference:
        raise AssertionError("Qwen3-14B and frozen Qwen3-8B task universes differ")
    if ledger.get("parent_qwen8_score_sha256") != {
        str(path.relative_to(REPO)): parent_hashes[str(path.relative_to(REPO))]
        for path in selected
    }:
        raise AssertionError("registered Qwen3-8B parent hash roster drifted")
    return rows, selected


def scale_difference(
    qwen14: list[dict], qwen8: list[dict], *, repetitions: int, seed: int
) -> tuple[float, float, float]:
    common = {
        "decode": "stochastic",
        "domain": "coin_harbor",
        "target_structure": "mediated_b",
        "cue": "correct",
    }
    differences: dict[int, np.ndarray] = {}
    for training_seed in SEEDS:
        values: dict[tuple[str, str], dict[str, float]] = {}
        for model, source in ((MODEL, qwen14), ("qwen3_8b", qwen8)):
            for arm in ARMS:
                values[(model, arm)] = core.episode_values(
                    source,
                    {**common, "model": model, "arm": arm, "seed": training_seed},
                ).get(training_seed, {})
        task_ids = sorted(set.intersection(*(set(item) for item in values.values())))
        if len(task_ids) != 120:
            raise AssertionError(
                f"seed {training_seed}: expected 120 paired primary-cell tasks"
            )
        differences[training_seed] = np.asarray(
            [
                (values[(MODEL, "population_prior")][task]
                 - values[(MODEL, "causal")][task])
                - (values[("qwen3_8b", "population_prior")][task]
                   - values[("qwen3_8b", "causal")][task])
                for task in task_ids
            ]
        )
    estimate = float(np.mean([np.mean(values) for values in differences.values()]))
    rng = np.random.default_rng(seed)
    draws = []
    seed_array = np.asarray(SEEDS)
    for _ in range(repetitions):
        sampled = rng.choice(seed_array, len(seed_array), replace=True)
        draws.append(
            float(
                np.mean(
                    [
                        np.mean(
                            rng.choice(
                                differences[int(selected)],
                                len(differences[int(selected)]),
                                replace=True,
                            )
                        )
                        for selected in sampled
                    ]
                )
            )
        )
    low, high = np.quantile(draws, (0.025, 0.975))
    return estimate, float(low), float(high)


def extension_estimates(
    rows: list[dict], qwen8: list[dict], *, repetitions: int, bootstrap_seed: int
) -> dict:
    primary = []
    for decode in ("greedy", "stochastic"):
        common = {
            "model": MODEL,
            "decode": decode,
            "domain": "coin_harbor",
            "target_structure": "mediated_b",
            "cue": "correct",
        }
        contrast = core.paired_interval(
            rows,
            {**common, "arm": "population_prior"},
            {**common, "arm": "causal"},
            key="task_id",
            repetitions=repetitions,
            seed=core.transfer_bootstrap_seed(
                bootstrap_seed, "coin_harbor", "mediated_b"
            ),
        )
        primary.append(
            {
                "model": MODEL,
                "decode": decode,
                "mean_response_mae": {
                    arm: core.mean_cell(rows, {**common, "arm": arm})
                    for arm in ("base", "population_prior", "causal")
                },
                "population_prior_minus_causal": interval(contrast),
            }
        )

    transfer_cells = []
    for domain, structure in core.TRANSFER_CELLS:
        common = {
            "model": MODEL,
            "decode": "stochastic",
            "domain": domain,
            "target_structure": structure,
            "cue": "correct",
        }
        contrast = core.paired_interval(
            rows,
            {**common, "arm": "population_prior"},
            {**common, "arm": "causal"},
            key="task_id",
            repetitions=repetitions,
            seed=core.transfer_bootstrap_seed(bootstrap_seed, domain, structure),
        )
        transfer_cells.append(
            {"domain": domain, "target_structure": structure, **interval(contrast)}
        )

    cue_by_k = []
    cue_common = {
        "model": MODEL,
        "arm": "causal",
        "decode": "stochastic",
        "domain": "coin_harbor",
        "target_structure": "mediated_b",
    }
    for k in (0, 2, 4, 8):
        values = {
            cue: core.mean_cell(rows, {**cue_common, "cue": cue, "k": k}, "pair_id")
            for cue in ("correct", "none", "misleading")
        }
        absent = core.paired_interval(
            rows,
            {**cue_common, "cue": "none", "k": k},
            {**cue_common, "cue": "correct", "k": k},
            key="pair_id",
            repetitions=repetitions,
            seed=bootstrap_seed + 100 + k,
        )
        misleading = core.paired_interval(
            rows,
            {**cue_common, "cue": "misleading", "k": k},
            {**cue_common, "cue": "correct", "k": k},
            key="pair_id",
            repetitions=repetitions,
            seed=bootstrap_seed + k,
        )
        cue_by_k.append(
            {
                "k": k,
                "mean_response_mae": values,
                "absent_minus_correct": interval(absent),
                "misleading_minus_correct": interval(misleading),
            }
        )
    cue_overall = {
        name: interval(
            core.paired_interval(
                rows,
                {**cue_common, "cue": cue},
                {**cue_common, "cue": "correct"},
                key="pair_id",
                repetitions=repetitions,
                seed=bootstrap_seed + offset,
            )
        )
        for name, cue, offset in (
            ("absent_minus_correct", "none", 3000),
            ("misleading_minus_correct", "misleading", 4000),
        )
    }
    override = interval(
        core.paired_override_interval(
            rows, cue_common, repetitions=repetitions, seed=bootstrap_seed + 5000
        )
    )
    difference = interval(
        scale_difference(
            rows, qwen8, repetitions=repetitions, seed=bootstrap_seed + 6000
        )
    )
    return {
        "primary": primary,
        "transfer_cells": transfer_cells,
        "cue_by_k": cue_by_k,
        "cue_overall": cue_overall,
        "evidence_override": override,
        "qwen3_14b_minus_qwen3_8b_primary_training_contrast": difference,
    }


def fmt(value: float) -> str:
    return f"{value:.2f}"


def fmt_interval(value: dict[str, float]) -> str:
    return (
        f"{value['estimate']:.2f} "
        f"[{value['ci95_low']:.2f},{value['ci95_high']:.2f}]"
    )


def render_table(estimates: dict, parent: dict) -> str:
    parent_rows = {
        row["model"]: row
        for row in parent["registered_estimates"]["primary"]
        if row["decode"] == "stochastic" and row["model"] in {"qwen3_4b", "qwen3_8b"}
    }
    qwen14 = next(row for row in estimates["primary"] if row["decode"] == "stochastic")
    rows = [("Qwen3-4B-Instruct-2507", parent_rows["qwen3_4b"]),
            ("Qwen3-8B", parent_rows["qwen3_8b"]), (MODEL_LABEL, qwen14)]
    lines = [
        "% generated by render_qwen3_14b_extension.py -- do not edit",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Base & Prior & Episode-matched & $\Delta$ [95\% CI] \\",
        r"\midrule",
    ]
    for label, row in rows:
        means = row["mean_response_mae"]
        lines.append(
            f"{label} & {fmt(means['base'])} & {fmt(means['population_prior'])} & "
            f"{fmt(means['causal'])} & {fmt_interval(row['population_prior_minus_causal'])} "
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def render_summary(estimates: dict) -> str:
    row = next(item for item in estimates["primary"] if item["decode"] == "stochastic")
    means = row["mean_response_mae"]
    contrast = row["population_prior_minus_causal"]
    difference = estimates["qwen3_14b_minus_qwen3_8b_primary_training_contrast"]
    return (
        r"\paragraph{Prospectively frozen Qwen3-14B scale extension.} "
        r"On the original shared endpoint, joint-shift response MAE for the "
        f"Qwen3-14B base, population-prior, and episode-matched conditions was "
        f"{fmt(means['base'])}, {fmt(means['population_prior'])}, and "
        f"{fmt(means['causal'])}, respectively. The prior-minus-matched contrast "
        f"was {fmt_interval(contrast)}. The paired Qwen3-14B-minus-Qwen3-8B "
        f"difference in that training contrast was {fmt_interval(difference)}. "
        r"This extension was frozen after the original roster was analyzed and is "
        r"reported regardless of direction."
        "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=REPORTS / "qwen3_14b_scale_results.json")
    parser.add_argument("--paper-root", type=Path, default=REPO / "paper")
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260825)
    args = parser.parse_args()
    ledger_path = args.ledger.resolve()
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    roster = validate_registration(ledger, ledger_path)
    rows, paths, universe = collect_extension(roster)
    qwen8_rows, qwen8_paths = collect_parent_qwen8(ledger, universe)
    estimates = extension_estimates(
        rows,
        qwen8_rows,
        repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    parent = json.loads(PARENT_RESULTS.read_text(encoding="utf-8"))
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": sha256(PROTOCOL),
        "ledger": str(ledger_path.relative_to(REPO)),
        "ledger_sha256": sha256(ledger_path),
        "parent_results_sha256": sha256(PARENT_RESULTS),
        "analysis_code_sha256": sha256(Path(__file__).resolve()),
        "score_files": [str(path.relative_to(REPO)) for path in paths],
        "score_file_sha256": {
            str(path.relative_to(REPO)): sha256(path) for path in paths
        },
        "parent_qwen3_8b_score_files": [str(path.relative_to(REPO)) for path in qwen8_paths],
        "row_draws": len(rows),
        "validated_jobs": len(roster),
        "validated_tasks_per_job": 1440,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.bootstrap_seed,
        "variance_unit": "training seed; paired task within seed; draws averaged within task",
        "registered_estimates": estimates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tables = args.paper_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "exp3_coin_qwen3_14b_scale.tex").write_text(
        render_table(estimates, parent), encoding="utf-8"
    )
    (tables / "exp3_coin_qwen3_14b_scale_summary.tex").write_text(
        render_summary(estimates), encoding="utf-8"
    )
    print(f"validated {len(roster)} jobs, {len(paths)} extension endpoints, and "
          f"{len(qwen8_paths)} frozen Qwen3-8B endpoints")
    print(f"result -> {args.output}")


if __name__ == "__main__":
    main()
