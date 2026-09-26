#!/usr/bin/env python3
"""Validate and render the completed Qwen3-8B Coin City stochastic endpoint.

Slice renderer; fail-closed on the exact six trained Qwen3-8B runs
and the Qwen3-8B base endpoint.  It uses the registered five-draw files and the
same seed-first, paired-task bootstrap as the full renderer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from make_interim_outputs import CELLS, KEYS, _fmt_bounds, fmt, fmt_interval, mean_binary
from make_paper_outputs import (
    mean_cell,
    paired_interval,
    paired_override_interval,
    read_jsonl,
    report_root,
    validate_ledger,
    validate_rows,
)


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
SEEDS = (42, 43, 44)
ARMS = ("causal", "population_prior")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latest_effective_ledger() -> Path:
    candidates = sorted(RUNS.glob("coin_city_structural_*.json"), reverse=True)
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("protocol") == "coin_city_structural_transfer_v1":
            return path
    raise SystemExit("no effective Coin City structural-transfer ledger found")


def collect_qwen3_8b(roster: list[dict]) -> tuple[list[dict], list[Path]]:
    trained = [
        row for row in roster
        if row.get("model") == "qwen3_8b"
        and row.get("kind") in {"confirmatory", "diagnostic"}
    ]
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    actual = {(row.get("arm"), row.get("seed")) for row in trained}
    if len(trained) != 6 or actual != expected:
        raise AssertionError("ledger does not contain the exact six-run Qwen3-8B roster")

    bases = [
        row for row in roster
        if row.get("model") == "qwen3_8b" and row.get("kind") == "base"
    ]
    if len(bases) != 1:
        raise AssertionError("ledger does not contain exactly one Qwen3-8B base")

    rows: list[dict] = []
    paths: list[Path] = []
    reference = None
    for item in [*trained, *bases]:
        path = report_root(item) / "stochastic_n5.scores.jsonl"
        if not path.is_file():
            raise AssertionError(f"job {item['job_id']}: missing {path}")
        scored = read_jsonl(path)
        universe = validate_rows(scored, item, "stochastic")
        if reference is None:
            reference = universe
        elif universe != reference:
            raise AssertionError(f"job {item['job_id']}: endpoint task universe differs")
        rows.extend(scored)
        paths.append(path)

    if len(paths) != 7 or len(rows) != 7 * 1440 * 5:
        raise AssertionError("Qwen3-8B slice must contain seven complete five-draw files")
    return rows, paths


def interval_dict(values: tuple[float, float, float]) -> dict[str, float]:
    estimate, low, high = values
    return {"estimate": estimate, "ci95_low": low, "ci95_high": high}


def analyze(rows: list[dict], repetitions: int, bootstrap_seed: int) -> dict:
    cell_results = []
    for cell_index, (label, domain, structure) in enumerate(CELLS):
        common = {
            "model": "qwen3_8b", "decode": "stochastic", "domain": domain,
            "target_structure": structure, "cue": "correct",
        }
        values = {
            arm: mean_cell(rows, {**common, "arm": arm})
            for arm in ("base", "population_prior", "causal")
        }
        contrast = paired_interval(
            rows,
            {**common, "arm": "population_prior"},
            {**common, "arm": "causal"},
            key="task_id", repetitions=repetitions,
            seed=bootstrap_seed + cell_index,
        )
        cell_results.append({
            "label": label, "domain": domain, "target_structure": structure,
            **values, "prior_minus_causal": list(contrast),
        })

    primary = {
        "model": "qwen3_8b", "decode": "stochastic",
        "domain": "coin_harbor", "target_structure": "mediated_b",
        "cue": "correct",
    }
    cue_means = {
        arm: {
            cue: mean_cell(rows, {**primary, "arm": arm, "cue": cue})
            for cue in ("correct", "none", "misleading")
        }
        for arm in ("base", "population_prior", "causal")
    }
    shape_accuracy = {
        arm: mean_binary(rows, {**primary, "arm": arm}, "structure_correct")
        for arm in ("base", "population_prior", "causal")
    }
    parse_rate = {
        arm: mean_binary(
            rows,
            {"model": "qwen3_8b", "arm": arm, "decode": "stochastic"},
            "parsed",
        )
        for arm in ("base", "population_prior", "causal")
    }

    cue_by_k = []
    for k in (0, 2, 4, 8):
        common = {**primary, "arm": "causal", "k": k}
        absent = paired_interval(
            rows, {**common, "cue": "none"}, {**common, "cue": "correct"},
            key="pair_id", repetitions=repetitions, seed=bootstrap_seed + 100 + k,
        )
        misleading = paired_interval(
            rows, {**common, "cue": "misleading"}, {**common, "cue": "correct"},
            key="pair_id", repetitions=repetitions, seed=bootstrap_seed + k,
        )
        cue_by_k.append({
            "k": k,
            "mean_response_mae": {
                cue: mean_cell(rows, {**common, "cue": cue}, "pair_id")
                for cue in ("correct", "none", "misleading")
            },
            "absent_minus_correct": interval_dict(absent),
            "misleading_minus_correct": interval_dict(misleading),
        })

    cue_common = {
        "model": "qwen3_8b", "arm": "causal", "decode": "stochastic",
        "domain": "coin_harbor", "target_structure": "mediated_b",
    }
    absent_overall = paired_interval(
        rows, {**cue_common, "cue": "none"}, {**cue_common, "cue": "correct"},
        key="pair_id", repetitions=repetitions, seed=bootstrap_seed + 3000,
    )
    misleading_overall = paired_interval(
        rows,
        {**cue_common, "cue": "misleading"},
        {**cue_common, "cue": "correct"},
        key="pair_id", repetitions=repetitions, seed=bootstrap_seed + 4000,
    )
    override = paired_override_interval(
        rows, cue_common, repetitions=repetitions, seed=bootstrap_seed + 5000,
    )
    return {
        "cells_correct_hint": cell_results,
        "primary_cue_response_mae": cue_means,
        "primary_shape_accuracy": shape_accuracy,
        "parse_rate": parse_rate,
        "cue_by_k": cue_by_k,
        "cue_overall": {
            "absent_minus_correct": interval_dict(absent_overall),
            "misleading_minus_correct": interval_dict(misleading_overall),
        },
        "evidence_override": interval_dict(override),
    }




def render_table(cell_results: list[dict]) -> str:
    lines = [
        "% generated by coin_city_structural/make_qwen3_8b_endpoint_outputs.py -- do not edit",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Test cell & Base & Prior & Episode-matched & Prior $-$ episode-matched [95\% interval] \\",
        r"\midrule",
    ]
    for row in cell_results:
        lines.append(
            f"{row['label']} & {fmt(row['base'])} & "
            f"{fmt(row['population_prior'])} & {fmt(row['causal'])} & "
            f"{fmt_interval(tuple(row['prior_minus_causal']))} " + r"\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def render_grid_macros(cell_results: list[dict]) -> str:
    lines = [
        "% generated by coin_city_structural/make_qwen3_8b_endpoint_outputs.py -- do not edit"
    ]
    peak = max(
        max(row["base"], row["population_prior"], row["causal"])
        for row in cell_results
    )
    lines.append("\\newcommand{\\cciPeak}{%.2f}" % peak)
    for row in cell_results:
        key = KEYS[row["label"]]
        estimate, low, high = row["prior_minus_causal"]
        body = "%.2f %s" % (estimate, _fmt_bounds(low, high))
        if low > 0 or high < 0:
            body = "\\textbf{%.2f} %s" % (estimate, _fmt_bounds(low, high))
        for name, value in (
            ("base", row["base"]),
            ("prior", row["population_prior"]),
            ("matched", row["causal"]),
        ):
            lines.append("\\newcommand{\\cci%s%s}{%.2f}" % (key, name, value))
        lines.append("\\newcommand{\\cci%sdelta}{%s}" % (key, body))
    return "\n".join(lines) + "\n"

def _fmt3(value: float) -> str:
    text = f"{value:.3f}"
    if text.startswith("-0."):
        return "-." + text[3:]
    if text.startswith("0."):
        return "." + text[2:]
    return text


def _fmt3_interval(summary: dict[str, float]) -> str:
    return fmt_interval((
        summary["estimate"], summary["ci95_low"], summary["ci95_high"]
    ))


def render_cue_table(estimates: dict) -> str:
    lines = [
        "% generated by coin_city_structural/make_qwen3_8b_endpoint_outputs.py -- do not edit",
        r"\begin{tabular}{crrrrr}",
        r"\toprule",
        r"$k$ & Correct & Absent & Misleading & Absent $-$ correct [95\% CI] & Misleading $-$ correct [95\% CI] \\",
        r"\midrule",
    ]
    for row in estimates["cue_by_k"]:
        means = row["mean_response_mae"]
        lines.append(
            f"{row['k']} & {_fmt3(means['correct'])} & {_fmt3(means['none'])} & "
            f"{_fmt3(means['misleading'])} & "
            f"{_fmt3_interval(row['absent_minus_correct'])} & "
            f"{_fmt3_interval(row['misleading_minus_correct'])} " + r"\\"
        )
    overall_means = estimates["primary_cue_response_mae"]["causal"]
    overall = estimates["cue_overall"]
    lines.extend([
        r"\midrule",
        "All $k$ & "
        f"{_fmt3(overall_means['correct'])} & {_fmt3(overall_means['none'])} & "
        f"{_fmt3(overall_means['misleading'])} & "
        f"{_fmt3_interval(overall['absent_minus_correct'])} & "
        f"{_fmt3_interval(overall['misleading_minus_correct'])} " + r"\\",
        r"\bottomrule",
        r"\end{tabular}",
    ])
    return "\n".join(lines) + "\n"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=latest_effective_ledger())
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260818)
    parser.add_argument(
        "--output", type=Path,
        default=REPORTS / "qwen3_8b_step300_stochastic.json",
    )
    parser.add_argument("--paper-root", type=Path, action="append", default=[])
    args = parser.parse_args()

    args.ledger = args.ledger.resolve()
    args.output = args.output.resolve()

    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    roster = validate_ledger(ledger)
    rows, paths = collect_qwen3_8b(roster)
    estimates = analyze(rows, args.bootstrap_repetitions, args.bootstrap_seed)
    artifact = {
        "analysis": "qwen3_8b_round300_five_draw_endpoint",
        "status": "complete_round300_five_draw_endpoint",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ledger": str(args.ledger.relative_to(REPO)),
        "ledger_sha256": sha256(args.ledger),
        "analysis_code_sha256": sha256(Path(__file__).resolve()),
        "score_file_sha256": {
            str(path.relative_to(REPO)): sha256(path) for path in paths
        },
        "validated_jobs": 7,
        "validated_tasks_per_job": 1440,
        "draws_per_task": 5,
        "row_draws": len(rows),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": args.bootstrap_seed,
        "variance_unit": "training seed; paired task within seed; draws averaged within task",
        "registered_estimates": estimates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    table = render_table(estimates["cells_correct_hint"]).replace(
        "make_interim_outputs.py", "make_qwen3_8b_endpoint_outputs.py"
    )
    macros = render_grid_macros(estimates["cells_correct_hint"]).replace(
        "make_interim_outputs.py", "make_qwen3_8b_endpoint_outputs.py"
    )
    cue_table = render_cue_table(estimates)
    paper_roots = args.paper_root or [REPO / "paper"]
    for paper_root in paper_roots:
        destination = paper_root / "tables" / "exp3_coin_qwen3_8b_stochastic.tex"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(table, encoding="utf-8")
        macro_destination = (
            paper_root / "tables" / "exp3_coin_qwen3_8b_stochastic_data.tex"
        )
        macro_destination.write_text(macros, encoding="utf-8")
        cue_destination = (
            paper_root / "tables" / "exp3_coin_qwen3_8b_cue_by_k.tex"
        )
        cue_destination.write_text(cue_table, encoding="utf-8")
        print(f"wrote table -> {destination}")
        print(f"wrote figure data -> {macro_destination}")
        print(f"wrote cue table -> {cue_destination}")
    print(f"validated {len(paths)} stochastic endpoint files and {len(rows)} draws")
    print(f"result -> {args.output}")


if __name__ == "__main__":
    main()
