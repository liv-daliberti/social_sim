#!/usr/bin/env python3
"""Validate and render the three-model locked Polymarket comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
REPORTS = ROOT / "reports"
QWEN4_DEFAULT = REPORTS / "exp3b_locked_test_j30505540.summary.json"
QWEN4_LEDGER_DEFAULT = ROOT / "runs" / "exp3b_20260812T195932Z.json"
DATA_TEST = ROOT / "data" / "exp3b_registered" / "test.tasks.jsonl"
SEEDS = (42, 43, 44)
MODEL_SPECS = (
    ("qwen3_4b", "Qwen3-4B"),
    ("qwen3_8b", "Qwen3-8B"),
    ("llama3_1_8b", "Llama-3.1-8B"),
)
MODEL_NAMES = {
    "qwen3_4b": "Qwen/Qwen3-4B-Instruct-2507",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "llama3_1_8b": "meta-llama/Llama-3.1-8B-Instruct",
}
EXPECTED_OUTPUTS = {"base", *(f"seed_{seed}" for seed in SEEDS)}
SCORE_RECOMPUTE_ATOL = 1e-7


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latest_extension_summary(model_key: str) -> Path:
    paths = sorted(REPORTS.glob(f"exp3b_{model_key}_locked_test_j*.summary.json"))
    if not paths:
        raise SystemExit(f"no locked-test summary found for {model_key}")
    return paths[-1]


def latest_extension_ledger() -> Path:
    paths = sorted((ROOT / "runs").glob("exp3b_model_extension_*.json"))
    if not paths:
        raise SystemExit("no Polymarket model-extension ledger found")
    return paths[-1]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def job_id_from_summary_path(path: Path) -> str:
    match = re.search(r"_j(\d+)\.summary\.json$", path.name)
    if match is None:
        raise AssertionError(f"locked-test summary lacks a job ID: {path}")
    return match.group(1)


def validate_ledgers(qwen4: dict, extension: dict) -> dict[str, dict]:
    if qwen4.get("protocol_version") != "exp3b_registered_v1":
        raise AssertionError("Qwen3-4B ledger has the wrong protocol")
    qwen4_jobs = {
        int(item["seed"]): str(item["job_id"]) for item in qwen4.get("submissions", [])
    }
    if set(qwen4_jobs) != set(SEEDS) or len(qwen4.get("submissions", [])) != 3:
        raise AssertionError("Qwen3-4B ledger must contain exactly seeds 42/43/44")
    qwen4_test = qwen4.get("locked_test_submission", {})
    if qwen4_test.get("adapter_spec") != ";".join(
        f"{seed}:{qwen4_jobs[seed]}" for seed in SEEDS
    ):
        raise AssertionError("Qwen3-4B locked-test adapter specification drifted")

    if extension.get("protocol_version") != "exp3b_model_extension_v1":
        raise AssertionError("8B ledger has the wrong protocol")
    expected_cells = {
        (model, seed) for model in ("qwen3_8b", "llama3_1_8b") for seed in SEEDS
    }
    submissions = extension.get("submissions", [])
    actual_cells = [
        (item.get("model"), int(item.get("seed", -1))) for item in submissions
    ]
    if len(actual_cells) != 6 or set(actual_cells) != expected_cells:
        raise AssertionError("8B ledger does not contain the exact six training cells")
    training_jobs = {
        (str(item["model"]), int(item["seed"])): str(item["job_id"])
        for item in submissions
    }
    evaluations = extension.get("locked_evaluations", [])
    if len(evaluations) != 2 or {item.get("model") for item in evaluations} != {
        "qwen3_8b",
        "llama3_1_8b",
    }:
        raise AssertionError("8B ledger does not contain the exact two locked tests")

    result = {
        "qwen3_4b": {
            "evaluation_job_id": str(qwen4_test["job_id"]),
            "training_job_ids": qwen4_jobs,
        }
    }
    for evaluation in evaluations:
        model = str(evaluation["model"])
        jobs = {seed: training_jobs[(model, seed)] for seed in SEEDS}
        expected_spec = ";".join(f"{seed}:{jobs[seed]}" for seed in SEEDS)
        if evaluation.get("adapter_spec") != expected_spec:
            raise AssertionError(f"{model}: locked-test adapter specification drifted")
        result[model] = {
            "evaluation_job_id": str(evaluation["job_id"]),
            "training_job_ids": jobs,
        }
    return result


def validate_summary(summary: dict, model_key: str) -> None:
    expected_protocol = (
        "exp3b_registered_v1" if model_key == "qwen3_4b" else "exp3b_model_extension_v1"
    )
    if summary.get("protocol_version") != expected_protocol:
        raise AssertionError(f"{model_key}: wrong protocol version")
    if summary.get("test_was_used_during_training") is not False:
        raise AssertionError(f"{model_key}: locked-test isolation is not explicit")
    if summary.get("model") != MODEL_NAMES[model_key]:
        raise AssertionError(f"{model_key}: wrong base model identity")
    if model_key != "qwen3_4b":
        if summary.get("model_key") != model_key:
            raise AssertionError(f"{model_key}: wrong model key")
        if summary.get("frozen_parent_protocol") != "exp3b_registered_v1":
            raise AssertionError(f"{model_key}: frozen parent protocol missing")
    decoding = summary.get("decoding", {})
    if (
        float(decoding.get("temperature", -1)),
        int(decoding.get("max_tokens", -1)),
    ) != (0.0, 128):
        raise AssertionError(f"{model_key}: locked decoding configuration drifted")
    if set(summary.get("adapter_paths", {})) != {str(seed) for seed in SEEDS}:
        raise AssertionError(f"{model_key}: expected three registered adapters")
    if set(summary.get("models", {})) != EXPECTED_OUTPUTS:
        raise AssertionError(f"{model_key}: expected base and seeds 42/43/44")
    for name, metrics in summary["models"].items():
        if int(metrics.get("n", 0)) != 1024:
            raise AssertionError(f"{model_key}/{name}: expected 1,024 test tasks")
        coverage = float(metrics.get("parse_coverage", -1))
        if not 0.0 <= coverage <= 1.0:
            raise AssertionError(f"{model_key}/{name}: invalid parse coverage")
    for name in ("market", "platt_market_train_only"):
        metrics = summary.get("baselines", {}).get(name, {})
        if int(metrics.get("n", 0)) != 1024:
            raise AssertionError(f"{model_key}: missing registered {name} baseline")
    required = {
        "trained_seed_mean_minus_base",
        "trained_seed_mean_minus_market",
        "trained_seed_mean_minus_platt_market",
    }
    if not required <= set(summary.get("comparisons_brier", {})):
        raise AssertionError(f"{model_key}: registered Brier comparisons missing")
    for name in required:
        contrast = summary["comparisons_brier"][name]
        values = [
            float(contrast.get(field, float("nan")))
            for field in ("estimate", "ci95_low", "ci95_high")
        ]
        if not all(math.isfinite(value) for value in values):
            raise AssertionError(f"{model_key}/{name}: non-finite comparison")
        if values[1] > values[2]:
            raise AssertionError(f"{model_key}/{name}: inverted confidence interval")
        if int(contrast.get("bootstrap_repetitions", 0)) != 5000:
            raise AssertionError(f"{model_key}/{name}: wrong bootstrap repetitions")
        if int(contrast.get("family_clusters", 0)) != 466:
            raise AssertionError(f"{model_key}/{name}: wrong connected-family count")


def validate_raw_rows(
    rows: list[dict], test_rows: list[dict], summary: dict, model_key: str
) -> None:
    expected = {str(row["task_id"]): row for row in test_rows}
    if len(test_rows) != 1024 or len(expected) != 1024:
        raise AssertionError("frozen test data must contain 1,024 unique task IDs")
    if len(rows) != 1024:
        raise AssertionError(f"{model_key}: expected 1,024 raw locked-test rows")
    task_ids = [str(row.get("task_id")) for row in rows]
    if len(set(task_ids)) != 1024 or set(task_ids) != set(expected):
        raise AssertionError(f"{model_key}: raw output task universe drifted")
    valid_counts = {name: 0 for name in EXPECTED_OUTPUTS}
    brier_losses = {name: [] for name in EXPECTED_OUTPUTS}
    for row in rows:
        task_id = str(row["task_id"])
        reference = expected[task_id]
        if (
            str(row.get("event_id")) != str(reference["event_id"])
            or bool(row.get("settlement_yes")) != bool(reference["settlement_yes"])
            or abs(
                float(row.get("market_yes_prob")) - float(reference["market_yes_prob"])
            )
            > 1e-12
        ):
            raise AssertionError(f"{model_key}/{task_id}: frozen task metadata drifted")
        if set(row.get("forecasts", {})) != EXPECTED_OUTPUTS:
            raise AssertionError(
                f"{model_key}/{task_id}: forecast roster is incomplete"
            )
        if set(row.get("responses", {})) != EXPECTED_OUTPUTS:
            raise AssertionError(
                f"{model_key}/{task_id}: response roster is incomplete"
            )
        outcome = float(bool(reference["settlement_yes"]))
        for name, raw_value in row["forecasts"].items():
            if raw_value is None:
                # The frozen evaluator assigns an invalid parse Brier loss one.
                brier_losses[name].append(1.0)
                continue
            if isinstance(raw_value, bool):
                raise AssertionError(f"{model_key}/{task_id}/{name}: Boolean forecast")
            try:
                probability = float(raw_value)
            except (TypeError, ValueError) as error:
                raise AssertionError(
                    f"{model_key}/{task_id}/{name}: invalid forecast probability"
                ) from error
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise AssertionError(
                    f"{model_key}/{task_id}/{name}: invalid forecast probability"
                )
            valid_counts[name] += 1
            brier_losses[name].append((probability - outcome) ** 2)

    for name in EXPECTED_OUTPUTS:
        coverage = valid_counts[name] / len(rows)
        brier = sum(brier_losses[name]) / len(rows)
        recorded = summary["models"][name]
        if abs(coverage - float(recorded["parse_coverage"])) > 1e-12:
            raise AssertionError(f"{model_key}/{name}: raw parse coverage drifted")
        # Summaries were accumulated from the parser's in-memory float values;
        # raw JSON stores their short decimal serialization. The largest frozen
        # round-trip discrepancy is below 1e-7 Brier.
        if not math.isclose(
            brier,
            float(recorded["brier"]),
            rel_tol=0.0,
            abs_tol=SCORE_RECOMPUTE_ATOL,
        ):
            raise AssertionError(f"{model_key}/{name}: raw Brier score drifted")


def row_from_summary(summary: dict, model_key: str, label: str) -> dict:
    validate_summary(summary, model_key)
    seeds = {
        str(seed): float(summary["models"][f"seed_{seed}"]["brier"]) for seed in SEEDS
    }
    seed_log_losses = {
        str(seed): float(summary["models"][f"seed_{seed}"]["log_loss_nats"])
        for seed in SEEDS
    }
    trained_log_losses = [
        float(summary["models"][f"seed_{seed}"]["log_loss_nats"]) for seed in SEEDS
    ]
    coverages = [
        float(summary["models"][name]["parse_coverage"])
        for name in ("base", *(f"seed_{seed}" for seed in SEEDS))
    ]
    return {
        "model_key": model_key,
        "label": label,
        "base_brier": float(summary["models"]["base"]["brier"]),
        "base_log_loss_nats": float(summary["models"]["base"]["log_loss_nats"]),
        "seed_brier": seeds,
        "seed_log_loss_nats": seed_log_losses,
        "trained_mean_brier": sum(seeds.values()) / len(seeds),
        "trained_mean_log_loss_nats": sum(trained_log_losses) / len(trained_log_losses),
        "minimum_parse_coverage": min(coverages),
        "trained_minus_base": summary["comparisons_brier"][
            "trained_seed_mean_minus_base"
        ],
        "trained_minus_market": summary["comparisons_brier"][
            "trained_seed_mean_minus_market"
        ],
        "trained_minus_platt_market": summary["comparisons_brier"][
            "trained_seed_mean_minus_platt_market"
        ],
    }


def fmt(value: float, digits: int = 4) -> str:
    if round(float(value), digits) == 0:
        return "." + "0" * digits
    text = f"{float(value):.{digits}f}"
    if text.startswith("-0."):
        return "-" + text[2:]
    if text.startswith("0."):
        return text[1:]
    return text


def interval(contrast: dict) -> str:
    return (
        f"{fmt(contrast['estimate'])} "
        f"[{fmt(contrast['ci95_low'], 5)},{fmt(contrast['ci95_high'], 5)}]"
    )


def render_latex(rows: list[dict]) -> str:
    lines = [
        "% generated by polymarket/scripts/render_model_extension_table.py -- do not edit",
        r"\begin{tabular}{l" + "r" * len(rows) + "}",
        r"\toprule",
        "Measure & " + " & ".join(row["label"] for row in rows) + r" \\",
        r"\midrule",
    ]
    measures = [
        ("Base Brier", lambda row: fmt(row["base_brier"])),
        *((f"Trained Brier, seed {seed}",
           lambda row, seed=seed: fmt(row["seed_brier"][str(seed)])) for seed in SEEDS),
        ("Trained mean Brier", lambda row: fmt(row["trained_mean_brier"])),
        (r"$\Delta$ base [95\% CI]", lambda row: interval(row["trained_minus_base"])),
        (r"$\Delta$ market [95\% CI]", lambda row: interval(row["trained_minus_market"])),
        (r"$\Delta$ Platt [95\% CI]", lambda row: interval(row["trained_minus_platt_market"])),
        ("Minimum parse coverage", lambda row: fmt(row["minimum_parse_coverage"], 3)),
    ]
    for label, value in measures:
        lines.append(label + " & " + " & ".join(value(row) for row in rows) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines)


def render_qwen8_main_latex(row: dict) -> str:
    lines = [
        "% generated by polymarket/scripts/render_model_extension_table.py -- do not edit",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Forecaster & Brier $\downarrow$ & $\Delta$ base & Log loss $\downarrow$ \\",
        r"\midrule",
        f"Untrained Qwen3-8B & {fmt(row['base_brier'])} & --- & "
        f"{fmt(row['base_log_loss_nats'])} " + r"\\",
    ]
    for seed in SEEDS:
        key = str(seed)
        delta = row["seed_brier"][key] - row["base_brier"]
        lines.append(
            f"Trained, seed {seed} & {fmt(row['seed_brier'][key])} & "
            f"{fmt(delta)} & {fmt(row['seed_log_loss_nats'][key])} " + r"\\"
        )
    lines.extend(
        (
            r"\midrule",
            f"Trained mean & {fmt(row['trained_mean_brier'])} & "
            f"{interval(row['trained_minus_base'])} & "
            f"{fmt(row['trained_mean_log_loss_nats'])} " + r"\\",
            r"\bottomrule",
            r"\end{tabular}",
        )
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen4", type=Path, default=QWEN4_DEFAULT)
    parser.add_argument("--qwen8", type=Path)
    parser.add_argument("--llama8", type=Path)
    parser.add_argument("--qwen4-ledger", type=Path, default=QWEN4_LEDGER_DEFAULT)
    parser.add_argument("--extension-ledger", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORTS / "exp3b_model_roster_summary.json",
    )
    parser.add_argument("--latex", type=Path, action="append", default=[])
    parser.add_argument("--qwen8-main-latex", type=Path, action="append", default=[])
    args = parser.parse_args()

    paths = {
        "qwen3_4b": args.qwen4,
        "qwen3_8b": args.qwen8 or latest_extension_summary("qwen3_8b"),
        "llama3_1_8b": args.llama8 or latest_extension_summary("llama3_1_8b"),
    }
    summaries = {
        key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items()
    }
    extension_ledger_path = args.extension_ledger or latest_extension_ledger()
    ledgers = {
        "qwen3_4b": json.loads(args.qwen4_ledger.read_text(encoding="utf-8")),
        "extension": json.loads(extension_ledger_path.read_text(encoding="utf-8")),
    }
    lineage = validate_ledgers(ledgers["qwen3_4b"], ledgers["extension"])
    test_rows = read_jsonl(DATA_TEST)
    raw_paths = {
        key: path.with_name(path.name.removesuffix(".summary.json") + ".jsonl")
        for key, path in paths.items()
    }
    for key in paths:
        if job_id_from_summary_path(paths[key]) != lineage[key]["evaluation_job_id"]:
            raise AssertionError(
                f"{key}: summary does not match the registered test job"
            )
        expected_training = lineage[key]["training_job_ids"]
        for seed in SEEDS:
            adapter_path = str(summaries[key]["adapter_paths"][str(seed)])
            if f"_j{expected_training[seed]}" not in adapter_path:
                raise AssertionError(f"{key}/seed {seed}: adapter job lineage drifted")
        validate_raw_rows(read_jsonl(raw_paths[key]), test_rows, summaries[key], key)
    rows = [row_from_summary(summaries[key], key, label) for key, label in MODEL_SPECS]
    markets = [
        float(summary["baselines"]["market"]["brier"]) for summary in summaries.values()
    ]
    platts = [
        float(summary["baselines"]["platt_market_train_only"]["brier"])
        for summary in summaries.values()
    ]
    if max(markets) - min(markets) > 1e-12 or max(platts) - min(platts) > 1e-12:
        raise AssertionError("model summaries do not share the frozen market baselines")

    result = {
        "protocol_version": "exp3b_three_model_roster_v1",
        "test_tasks": 1024,
        "market_brier": markets[0],
        "platt_market_train_only_brier": platts[0],
        "rows": rows,
        "inputs": {
            key: {
                "summary_path": str(path.resolve().relative_to(REPO)),
                "summary_sha256": sha256(path),
                "raw_path": str(raw_paths[key].resolve().relative_to(REPO)),
                "raw_sha256": sha256(raw_paths[key]),
            }
            for key, path in paths.items()
        },
        "run_ledgers": {
            "qwen3_4b": {
                "path": str(args.qwen4_ledger.resolve().relative_to(REPO)),
                "sha256": sha256(args.qwen4_ledger),
            },
            "extension": {
                "path": str(extension_ledger_path.resolve().relative_to(REPO)),
                "sha256": sha256(extension_ledger_path),
            },
        },
        "test_data": {
            "path": str(DATA_TEST.resolve().relative_to(REPO)),
            "sha256": sha256(DATA_TEST),
        },
        "analysis_code": {
            "path": str(Path(__file__).resolve().relative_to(REPO)),
            "sha256": sha256(Path(__file__).resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    appendix_rows = [row for row in rows if row["model_key"] != "qwen3_8b"]
    rendered = render_latex(appendix_rows) + "\n"
    for path in args.latex:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    qwen8 = next(row for row in rows if row["model_key"] == "qwen3_8b")
    qwen8_rendered = render_qwen8_main_latex(qwen8) + "\n"
    for path in args.qwen8_main_latex:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(qwen8_rendered, encoding="utf-8")
    print(f"result -> {args.output}")


if __name__ == "__main__":
    main()
