#!/usr/bin/env python3
"""Validate and render the completed Qwen3-4B Coin City greedy endpoint.

The full registered renderer remains fail-closed on the three-model roster. This
slice renderer is separately fail-closed on the exact nine trained Qwen3-4B runs
and the Qwen3-4B base endpoint at the requested round.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from make_paper_outputs import (
    mean_cell,
    paired_interval,
    report_root,
    validate_ledger,
    validate_rows,
)
from report import read_scores, score_records


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"

CELLS = (
    ("City / A (ID)", "coin_city", "direct_a"),
    ("Harbor / A (domain)", "coin_harbor", "direct_a"),
    ("City / B (structure)", "coin_city", "mediated_b"),
    ("Harbor / B (joint)", "coin_harbor", "mediated_b"),
)


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


def evaluation_path(item: dict, step: int) -> Path:
    paths = sorted(report_root(item).glob(f"debug_*/eval_results/{step}.json"))
    if len(paths) != 1:
        raise AssertionError(
            f"job {item['job_id']}: expected one round-{step} evaluation, found {paths}"
        )
    return paths[0]


def load_online_evaluation(item: dict, step: int) -> tuple[list[dict], Path]:
    path = evaluation_path(item, step)
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != 1440:
        raise AssertionError(f"job {item['job_id']}: expected 1,440 records at round {step}")
    rows = score_records(
        records,
        {"model": "qwen3_4b", "arm": item["arm"], "seed": int(item["seed"])},
    )
    validate_rows(rows, item, "greedy")
    return rows, path


def mean_binary(rows: list[dict], selectors: dict, key: str) -> float:
    values = [float(row[key]) for row in rows if all(row.get(k) == v for k, v in selectors.items())]
    return float(np.mean(values)) if values else float("nan")


def fmt(value: float) -> str:
    return f"{value:.2f}"


def _interval_places(low: float, high: float) -> int:
    """Two decimals, unless an endpoint would print as a signed zero.

    A bound like -0.002 renders as "-0.00" at two places, which reads as a typo
    and hides that the interval only just covers zero. In that case show three.
    """
    for bound in (low, high):
        if bound != 0.0 and abs(round(bound, 2)) < 0.005:
            return 3
    return 2


def fmt_interval(values: tuple[float, float, float]) -> str:
    estimate, low, high = values
    places = _interval_places(low, high)
    return f"{estimate:.2f} [{low:.{places}f}, {high:.{places}f}]"


KEYS = {
    "City / A (ID)": "ID",
    "Harbor / A (domain)": "Domain",
    "City / B (structure)": "Structure",
    "Harbor / B (joint)": "Joint",
}


def _fmt_bounds(low: float, high: float) -> str:
    places = _interval_places(low, high)
    return f"[{low:.{places}f}, {high:.{places}f}]"


def render_grid_macros(cell_results: list[dict]) -> str:
    """Emit the same numbers as plain macros for the TikZ results grid.

    Formatting decisions -- rounding, and whether an interval excludes zero and
    so earns emphasis -- are made here rather than in TeX, so the figure only
    places values it is handed.
    """
    lines = [
        "% generated from completed round-300 Coin City endpoint artifacts -- do not edit",
    ]
    peak = max(max(row["base"], row["structureless"], row["population_prior"],
                   row["causal"]) for row in cell_results)
    lines.append("\\newcommand{\\cciPeak}{%.2f}" % peak)
    for row in cell_results:
        key = KEYS[row["label"]]
        estimate, low, high = row["prior_minus_causal"]
        excludes_zero = low > 0 or high < 0
        if excludes_zero:
            body = "\\textbf{%.2f} %s" % (estimate, _fmt_bounds(low, high))
        else:
            body = "%.2f %s" % (estimate, _fmt_bounds(low, high))
        for name, value in (("base", row["base"]),
                            ("less", row["structureless"]),
                            ("prior", row["population_prior"]),
                            ("causal", row["causal"])):
            lines.append("\\newcommand{\\cci%s%s}{%.2f}" % (key, name, value))
        lines.append("\\newcommand{\\cci%sdelta}{%s}" % (key, body))
        lines.append("\\newcommand{\\cci%ssig}{%d}" % (key, 1 if excludes_zero else 0))
    return "\n".join(lines) + "\n"


def render_table(cell_results: list[dict]) -> str:
    lines = [
        "% generated from completed round-300 Coin City endpoint artifacts -- do not edit",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Test cell & Base & Structureless & Prior & Causal & Prior $-$ causal [95\% interval] \\",
        r"\midrule",
    ]
    for row in cell_results:
        lines.append(
            f"{row['label']} & {fmt(row['base'])} & {fmt(row['structureless'])} & "
            f"{fmt(row['population_prior'])} & {fmt(row['causal'])} & "
            f"{fmt_interval(tuple(row['prior_minus_causal']))} " + r"\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=latest_effective_ledger())
    parser.add_argument("--checkpoint", type=int, default=300)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260818)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS / "qwen3_4b_step300_greedy.json",
    )
    parser.add_argument("--paper-root", type=Path, action="append", default=[])
    args = parser.parse_args()

    ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
    roster = validate_ledger(ledger)
    trained = [
        row for row in roster
        if row.get("model") == "qwen3_4b" and row.get("kind") in {"confirmatory", "diagnostic"}
    ]
    expected = {
        (arm, seed) for arm in ("causal", "population_prior", "structureless")
        for seed in (42, 43, 44)
    }
    if {(row.get("arm"), row.get("seed")) for row in trained} != expected:
        raise AssertionError("effective ledger does not contain the exact nine-run Qwen3-4B roster")

    rows: list[dict] = []
    inputs: list[Path] = []
    universe = None
    for item in trained:
        scored, path = load_online_evaluation(item, args.checkpoint)
        current = validate_rows(scored, item, "greedy")
        if universe is None:
            universe = current
        elif current != universe:
            raise AssertionError("online checkpoint task universes differ")
        rows.extend(scored)
        inputs.append(path)

    base_items = [row for row in roster if row.get("model") == "qwen3_4b" and row.get("kind") == "base"]
    if len(base_items) != 1:
        raise AssertionError("effective ledger does not contain exactly one Qwen3-4B base")
    base_item = base_items[0]
    base_path = report_root(base_item) / "greedy.scores.jsonl"
    base_rows = read_scores([base_path])
    base_universe = validate_rows(base_rows, base_item, "greedy")
    if universe != base_universe:
        raise AssertionError("base and checkpoint task universes differ")
    rows.extend(base_rows)
    inputs.append(base_path)

    cell_results = []
    for cell_index, (label, domain, structure) in enumerate(CELLS):
        common = {
            "model": "qwen3_4b",
            "decode": "greedy",
            "domain": domain,
            "target_structure": structure,
            "cue": "correct",
        }
        values = {
            arm: mean_cell(rows, {**common, "arm": arm})
            for arm in ("base", "structureless", "population_prior", "causal")
        }
        contrast = paired_interval(
            rows,
            {**common, "arm": "population_prior"},
            {**common, "arm": "causal"},
            key="task_id",
            repetitions=args.bootstrap_repetitions,
            seed=args.bootstrap_seed + cell_index,
        )
        cell_results.append({
            "label": label,
            "domain": domain,
            "target_structure": structure,
            **values,
            "prior_minus_causal": list(contrast),
        })

    primary = {
        "model": "qwen3_4b",
        "decode": "greedy",
        "domain": "coin_harbor",
        "target_structure": "mediated_b",
        "cue": "correct",
    }
    cue_results = {
        arm: {
            cue: mean_cell(rows, {**primary, "arm": arm, "cue": cue})
            for cue in ("correct", "none", "misleading")
        }
        for arm in ("base", "structureless", "population_prior", "causal")
    }
    shape_accuracy = {
        arm: mean_binary(rows, {**primary, "arm": arm}, "structure_correct")
        for arm in ("base", "structureless", "population_prior", "causal")
    }
    parse_rate = {
        arm: mean_binary(rows, {"model": "qwen3_4b", "arm": arm, "decode": "greedy"}, "parsed")
        for arm in ("base", "structureless", "population_prior", "causal")
    }

    causal_300_rows: list[dict] = []
    causal_300_inputs: list[Path] = []
    for item in trained:
        if item.get("arm") != "causal":
            continue
        try:
            scored, path = load_online_evaluation(item, 300)
        except (AssertionError, OSError, json.JSONDecodeError):
            causal_300_rows = []
            causal_300_inputs = []
            break
        causal_300_rows.extend(scored)
        causal_300_inputs.append(path)
    causal_300 = None
    if len(causal_300_rows) == 3 * 1440:
        causal_300 = {
            "primary_response_mae": mean_cell(causal_300_rows, {**primary, "arm": "causal"}),
            "primary_shape_accuracy": mean_binary(
                causal_300_rows, {**primary, "arm": "causal"}, "structure_correct"
            ),
        }
        inputs.extend(causal_300_inputs)

    artifact = {
        "analysis": "qwen3_4b_round300_greedy_endpoint",
        "status": "qwen3_4b_decode_complete_larger_models_pending",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint_round": args.checkpoint,
        "checkpoint_selection": (
            "prespecified online-evaluation round complete for all nine Qwen3-4B "
            "causal, population-prior, and structureless runs"
        ),
        "scope": [
            "Qwen3-4B greedy endpoint complete",
            "Qwen3-8B and Llama-3.1-8B confirmatory endpoints remain pending",
        ],
        "ledger": str(args.ledger.relative_to(REPO)),
        "ledger_sha256": sha256(args.ledger),
        "input_sha256": {str(path.relative_to(REPO)): sha256(path) for path in inputs},
        "bootstrap": {
            "repetitions": args.bootstrap_repetitions,
            "seed": args.bootstrap_seed,
            "variance_unit": "training seed, then paired task within seed",
        },
        "row_counts": {
            "trained_runs": 9,
            "tasks_per_run": 1440,
            "base_tasks": 1440,
            "total_scored_rows": len(rows),
        },
        "parse_rate": parse_rate,
        "cells_correct_hint": cell_results,
        "primary_cue_response_mae": cue_results,
        "primary_shape_accuracy": shape_accuracy,
        "causal_round300_greedy_only": causal_300,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    table = render_table(cell_results)
    paper_roots = args.paper_root or [REPO / "paper"]
    for paper_root in paper_roots:
        destination = paper_root / "tables" / "exp3_coin_qwen3_4b_greedy.tex"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(table, encoding="utf-8")
        (paper_root / "tables" / "exp3_coin_qwen3_4b_greedy_data.tex").write_text(
            render_grid_macros(cell_results), encoding="utf-8")

    print(f"wrote greedy endpoint artifact -> {args.output}")
    for paper_root in paper_roots:
        print(f"wrote table -> {paper_root / 'tables' / 'exp3_coin_qwen3_4b_greedy.tex'}")


if __name__ == "__main__":
    main()
