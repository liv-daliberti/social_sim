#!/usr/bin/env python3
"""Aggregate the frozen 4B/8B/14B result with prospective Qwen3-32B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import render_scale_extension as parent


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "QWEN32_SCALE_PROTOCOL.md"
EXTENSION_VERSION = "exp4_qwen32_scale_v1"
MODEL_KEYS = ("qwen3_4b", "qwen3_8b", "qwen3_14b", "qwen3_32b")


def parse_evaluation_specs(value: str) -> dict[str, str]:
    parent_keys = parent.MODEL_KEYS
    try:
        parent.MODEL_KEYS = MODEL_KEYS
        return parent.parse_evaluation_specs(value)
    finally:
        parent.MODEL_KEYS = parent_keys


def validate_registration(path: Path) -> dict[str, Any]:
    registration = json.loads(path.read_text(encoding="utf-8"))
    if registration.get("kind") != "registration":
        raise AssertionError("Qwen3-32B registration ledger has the wrong kind")
    if registration.get("protocol_version") != EXTENSION_VERSION:
        raise AssertionError("Qwen3-32B registration protocol drifted")
    if registration.get("protocol_sha256") != parent.sha256(PROTOCOL):
        raise AssertionError("Qwen3-32B protocol changed after registration")
    return registration


def render_tex(report: dict[str, Any]) -> str:
    labels = {
        "qwen3_4b": r"Qwen3-4B-Instruct-2507$^\dagger$",
        "qwen3_8b": "Qwen3-8B",
        "qwen3_14b": "Qwen3-14B",
        "qwen3_32b": "Qwen3-32B",
    }
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Checkpoint & Base & RL mean & RL $-$ base & Parse \\",
        r"\midrule",
    ]
    for model_key in MODEL_KEYS:
        result = report["models"][model_key]
        delta = result["trained_minus_base_brier"]["estimate"]
        coverage = result["trained_seed_mean_complete_task_parse_coverage"]
        lines.append(
            f"{labels[model_key]} & {result['base_brier']:.4f} & "
            f"{result['trained_seed_mean_brier']:.4f} & {delta:+.4f} & "
            f"{100.0 * coverage:.1f}\\% \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "",
            (
                r"$\dagger$ Later Instruct-2507 revision; controlled dense "
                r"same-release contrasts use 8B, 14B, and 32B."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def aggregate(specs: dict[str, str], registration_path: Path) -> dict[str, Any]:
    registration = validate_registration(registration_path)
    parent_keys = parent.MODEL_KEYS
    try:
        parent.MODEL_KEYS = MODEL_KEYS
        report = parent.aggregate(specs)
    finally:
        parent.MODEL_KEYS = parent_keys

    summary_path, raw_path = parent.resolve_evaluation("qwen3_32b", specs["qwen3_32b"])
    summary32 = json.loads(summary_path.read_text(encoding="utf-8"))
    rows32 = parent.registered.read_jsonl(raw_path)
    rows14 = parent.registered.read_jsonl(
        parent.resolve_evaluation("qwen3_14b", specs["qwen3_14b"])[1]
    )
    rows8 = parent.registered.read_jsonl(
        parent.resolve_evaluation("qwen3_8b", specs["qwen3_8b"])[1]
    )
    vector32 = parent.trained_loss_vector(rows32)
    vector14 = parent.trained_loss_vector(rows14)
    vector8 = parent.trained_loss_vector(rows8)
    rows = rows32

    cross32 = {
        "qwen3_32b_minus_qwen3_14b": parent.registered.cluster_bootstrap_delta(
            rows, vector32, vector14
        ),
        "qwen3_32b_minus_qwen3_8b": parent.registered.cluster_bootstrap_delta(
            rows, vector32, vector8
        ),
    }
    primary = summary32["comparisons_brier"]["trained_seed_mean_minus_base"]
    market = summary32["comparisons_brier"]["trained_seed_mean_minus_market"]
    platt = summary32["comparisons_brier"]["trained_seed_mean_minus_platt_market"]

    report["parent_protocol_version"] = report["protocol_version"]
    report["protocol_version"] = EXTENSION_VERSION
    report["registration"] = {
        "path": str(registration_path),
        "sha256": parent.sha256(registration_path),
        "registered_at": registration["registered_at"],
    }
    report["primary_qwen3_32b_trained_minus_base"] = primary
    report["qwen3_32b_trained_minus_market"] = market
    report["qwen3_32b_trained_minus_platt_market"] = platt
    report["cross_model_brier"].update(cross32)
    report["decision_flags"].update(
        {
            "qwen3_32b_base_relative_gain_conclusive": (
                float(primary["ci95_high"]) < 0.0
            ),
            "qwen3_32b_beats_qwen3_14b_conclusive": (
                float(cross32["qwen3_32b_minus_qwen3_14b"]["ci95_high"]) < 0.0
            ),
            "qwen3_32b_beats_qwen3_8b_conclusive": (
                float(cross32["qwen3_32b_minus_qwen3_8b"]["ci95_high"]) < 0.0
            ),
            "qwen3_32b_beats_market_conclusive": float(market["ci95_high"]) < 0.0,
            "qwen3_32b_beats_train_only_platt_conclusive": (
                float(platt["ci95_high"]) < 0.0
            ),
        }
    )
    report["interpretation_lock"] = (
        "The prospective dense same-release scale extension compares Qwen3-8B, "
        "Qwen3-14B, and Qwen3-32B. Qwen3-4B-Instruct-2507 remains descriptive."
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluations", required=True)
    parser.add_argument("--registration-ledger", type=Path, required=True)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ROOT / "reports" / "exp4_qwen32_scale_latest",
    )
    args = parser.parse_args()

    report = aggregate(
        parse_evaluation_specs(args.evaluations), args.registration_ledger
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".summary.json")
    tex_path = args.output_prefix.with_suffix(".tex")
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tex_path.write_text(render_tex(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Qwen3-32B extension table -> {tex_path}")


if __name__ == "__main__":
    main()
