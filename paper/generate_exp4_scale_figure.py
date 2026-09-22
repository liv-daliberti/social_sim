#!/usr/bin/env python3
"""Generate the Experiment 4 Qwen scale/Llama comparator figure and table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
ICONS = ROOT / "paper" / "icons"
SUMMARY = (
    ROOT
    / "exp3_training_transfer"
    / "polymarket"
    / "reports"
    / "exp4_scale_latest.summary.json"
)
LLAMA_SUMMARY = (
    ROOT
    / "exp3_training_transfer"
    / "polymarket"
    / "reports"
    / "exp4_scale_llama3_1_8b_stochastic_j30889436.summary.json"
)
LLAMA_LEDGER = (
    ROOT
    / "exp3_training_transfer"
    / "polymarket"
    / "runs"
    / "exp4_scale_llama_posthoc_20260825T201141Z.json"
)
ARCHITECTURE_EXTENSION = (
    ROOT
    / "exp3_training_transfer"
    / "polymarket"
    / "reports"
    / "exp4_architecture_extension_recovered_latest.summary.json"
)
QWEN32_SUMMARY = (
    ROOT
    / "exp3_training_transfer"
    / "polymarket"
    / "reports"
    / "exp4_scale_qwen3_32b_stochastic_j30913533.summary.json"
)
QWEN32_RAW = QWEN32_SUMMARY.with_suffix("").with_suffix(".jsonl")
QWEN32_REGISTRATION = (
    ROOT
    / "exp3_training_transfer"
    / "polymarket"
    / "runs"
    / "exp4_qwen32_registration_20260827T015925Z.json"
)
QWEN32_PROTOCOL = (
    ROOT
    / "exp3_training_transfer"
    / "polymarket"
    / "QWEN32_SCALE_PROTOCOL.md"
)
HOSTED_REPORTS = (
    {
        "company": "DeepSeek",
        "version": "V4-Pro",
        "icon": "deepseek.png",
        "model": "DeepSeek-V4-Pro",
        "path": ROOT
        / "exp3_training_transfer"
        / "polymarket"
        / "reports"
        / "exp4_hosted"
        / "exp4_hosted_DeepSeek-V4-Pro_stochastic.summary.json",
        "color": "#009E73",
    },
    {
        "company": "Anthropic",
        "version": "Opus 4.8",
        "icon": "claude.png",
        "icon_scale": 1.45,
        "model": "claude-opus-4-8",
        "path": ROOT
        / "exp3_training_transfer"
        / "polymarket"
        / "reports"
        / "exp4_hosted"
        / "exp4_hosted_claude-opus-4-8_stochastic.summary.json",
        "color": "#CC79A7",
    },
    {
        "company": "OpenAI",
        "version": "GPT-5.6",
        "icon": "openai.png",
        "model": "gpt-5.6-sol",
        "path": ROOT
        / "exp3_training_transfer"
        / "polymarket"
        / "reports"
        / "exp4_hosted"
        / "exp4_hosted_gpt-5.summary.json",
        "color": "#E69F00",
    },
    {
        "company": "Moonshot AI",
        "version": "K3",
        "icon": "kimi.png",
        "model": "FW-Kimi-K3",
        "path": ROOT
        / "exp3_training_transfer"
        / "polymarket"
        / "reports"
        / "exp4_hosted_kimi_repair"
        / "exp4_hosted_FW-Kimi-K3_stochastic.summary.json",
        "color": "#0072B2",
    },
)
FIGURES = ROOT / "paper" / "figures"
TABLES = ROOT / "paper" / "tables"
PDF = FIGURES / "exp4_qwen_scale_results.pdf"
PNG = FIGURES / "exp4_qwen_scale_results.png"
LLAMA_PDF = FIGURES / "exp4_llama_scale_results.pdf"
LLAMA_PNG = FIGURES / "exp4_llama_scale_results.png"
TABLE = TABLES / "exp4_qwen_scale_results.tex"

PROTOCOL_VERSION = "exp4_qwen_scale_v1"
HOLDOUT_SHA256 = "ee9bd976402a746195291b4b43e209a084a0f1c6e5bf3f27c9d4ef638f1c7f24"
MODEL_KEYS = ("qwen3_4b", "qwen3_8b", "qwen3_14b")
MODEL_PARAMETERS = (4.0, 8.0, 14.0)
MODEL_LABELS = ("4B", "8B", "14B")
QWEN32_MODEL_KEY = "qwen3_32b"
QWEN32_MODEL = "Qwen/Qwen3-32B"
QWEN32_SUMMARY_SHA256 = (
    "408ef7bb125ecd25b30c55b11a7ce57f5ee4fae00769da9469c0fb4c4fd919ab"
)
QWEN32_RAW_SHA256 = (
    "a6eaf66a2f02dc6524b4e176a3111d34f3a6e5eaebf6629585773cb1c803385b"
)
QWEN32_REGISTRATION_SHA256 = (
    "a3f6b34158b2830e15b4d69a646dea388df9c452900d0d9a773fcb53c157aca5"
)
QWEN32_PROTOCOL_SHA256 = (
    "0b94766fc0109c286526b7723745996ad1ef874e8610e92492ea773f878f7f70"
)
LLAMA_MODEL_KEY = "llama3_1_8b"
LLAMA_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
EXPECTED_DECODING = {
    "mode": "five_draw_non_thinking_stochastic",
    "draws": 5,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "sampling_seed": 20_260_824,
    "max_tokens": 128,
    "incomplete_draw_policy": "endpoint_task_brier_loss_one",
}

QWEN_COLOR = "#0072B2"
QWEN_SEED_COLOR = "#56B4E9"
LLAMA_COLOR = "#7A3E9D"
LLAMA_SEED_COLOR = "#CC79A7"
MARKET_COLOR = "#D55E00"
GRID_COLOR = "#D8DEE6"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 6.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_report(path: Path = SUMMARY) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "pass":
        raise RuntimeError("Exp4 scale report is not a pass")
    if report.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Exp4 scale protocol drifted")
    if report.get("holdout_tasks_sha256") != HOLDOUT_SHA256:
        raise RuntimeError("Exp4 scale holdout drifted")
    if report.get("n") != 318:
        raise RuntimeError("Exp4 scale holdout count drifted")
    if report.get("decoding") != EXPECTED_DECODING:
        raise RuntimeError("Exp4 stochastic decoding drifted")
    if set(report.get("models", {})) != set(MODEL_KEYS):
        raise RuntimeError("Exp4 scale model roster or order drifted")

    expected_flags = {
        "qwen3_14b_base_relative_gain_conclusive": False,
        "qwen3_14b_beats_market_conclusive": False,
        "qwen3_14b_beats_qwen3_8b_conclusive": True,
        "qwen3_14b_beats_train_only_platt_conclusive": False,
    }
    if report.get("decision_flags") != expected_flags:
        raise RuntimeError("Exp4 scale decision flags drifted")

    for model_key in MODEL_KEYS:
        model = report["models"][model_key]
        if model["trained_seed_mean_complete_task_parse_coverage"] != 1:
            raise RuntimeError(f"{model_key} trained parse coverage is incomplete")
        if model["base_complete_task_parse_coverage"] != 1:
            raise RuntimeError(f"{model_key} base parse coverage is incomplete")
        source = report["sources"][model_key]
        for kind in ("summary", "raw"):
            source_path = Path(source[kind])
            if (
                not source_path.is_file()
                or sha256(source_path) != source[f"{kind}_sha256"]
            ):
                raise RuntimeError(f"{model_key} {kind} artifact drifted")
    return report


def load_llama_report(
    path: Path = LLAMA_SUMMARY, ledger_path: Path = LLAMA_LEDGER
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("status") != "complete":
        raise RuntimeError("Llama comparator ledger is not complete")
    if ledger.get("protocol_version") != "exp4_scale_llama_posthoc_v1":
        raise RuntimeError("Llama comparator registration drifted")
    if ledger.get("classification") != "post_hoc_architecture_comparator":
        raise RuntimeError("Llama comparator classification drifted")
    if report.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Llama parent protocol drifted")
    if report.get("model_key") != LLAMA_MODEL_KEY or report.get("model") != LLAMA_MODEL:
        raise RuntimeError("Llama comparator model drifted")
    if report.get("decoding") != EXPECTED_DECODING:
        raise RuntimeError("Llama stochastic decoding drifted")
    holdout = report.get("holdout", {})
    if holdout.get("n") != 318 or holdout.get("test_tasks_sha256") != HOLDOUT_SHA256:
        raise RuntimeError("Llama comparator holdout drifted")
    if holdout.get("manifest_sha256") != ledger["holdout"]["manifest_sha256"]:
        raise RuntimeError("Llama holdout manifest drifted")
    expected_models = {"base", "seed_42", "seed_43", "seed_44"}
    if set(report.get("models", {})) != expected_models:
        raise RuntimeError("Llama comparator endpoint roster drifted")
    for seed in (42, 43, 44):
        endpoint = report["models"][f"seed_{seed}"]
        if endpoint["draw_parse_coverage"] != 1:
            raise RuntimeError(f"Llama seed {seed} draw coverage is incomplete")
        if endpoint["complete_task_parse_coverage"] != 1:
            raise RuntimeError(f"Llama seed {seed} task coverage is incomplete")
        registered = ledger["adapters"][str(seed)]
        if report["adapter_paths"][str(seed)] != registered["path"]:
            raise RuntimeError(f"Llama seed {seed} adapter path drifted")
        if report["adapter_sha256"][str(seed)] != registered["weights_sha256"]:
            raise RuntimeError(f"Llama seed {seed} adapter hash drifted")
    if (
        report.get("analysis_code_sha256")
        != ledger["frozen_sources"]["evaluator_sha256"]
    ):
        raise RuntimeError("Llama evaluation code drifted")
    results = ledger["results"]
    for artifact in ("summary", "raw"):
        artifact_path = ROOT / results[artifact]
        if (
            not artifact_path.is_file()
            or sha256(artifact_path) != results[f"{artifact}_sha256"]
        ):
            raise RuntimeError(f"Llama {artifact} artifact drifted")
    raw_path = ROOT / results["raw"]
    raw_rows = sum(
        1 for line in raw_path.read_text(encoding="utf-8").splitlines() if line
    )
    if raw_rows != 318 or raw_rows != results["raw_rows"]:
        raise RuntimeError("Llama raw row count drifted")
    trained_mean = (
        sum(report["models"][f"seed_{seed}"]["brier"] for seed in (42, 43, 44)) / 3
    )
    if abs(trained_mean - results["trained_seed_mean_brier"]) > 1e-12:
        raise RuntimeError("Llama registered trained mean drifted")
    if abs(report["models"]["base"]["brier"] - results["base_brier"]) > 1e-12:
        raise RuntimeError("Llama registered base score drifted")
    return report


def resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_architecture_extension(path: Path = ARCHITECTURE_EXTENSION) -> dict[str, Any]:
    extension = json.loads(path.read_text(encoding="utf-8"))
    if extension.get("status") != "pass":
        raise RuntimeError("architecture extension is not a pass")
    if extension.get("protocol_version") != "exp4_architecture_provider_sweep_v1":
        raise RuntimeError("architecture extension protocol drifted")
    if extension.get("classification") != "post_hoc_descriptive_extension":
        raise RuntimeError("architecture extension classification drifted")
    if extension.get("holdout_n") != 318:
        raise RuntimeError("architecture extension holdout count drifted")
    if extension.get("holdout_tasks_sha256") != HOLDOUT_SHA256:
        raise RuntimeError("architecture extension holdout drifted")

    registration_path = resolve_recorded_path(extension["registration"])
    if (
        not registration_path.is_file()
        or sha256(registration_path) != extension["registration_sha256"]
    ):
        raise RuntimeError("architecture extension registration drifted")
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if registration.get("status") != "registered":
        raise RuntimeError("architecture extension registration is not frozen")

    finalizer_path = resolve_recorded_path(extension["finalizer"])
    if (
        not finalizer_path.is_file()
        or sha256(finalizer_path) != extension["finalizer_sha256"]
    ):
        raise RuntimeError("architecture extension finalizer drifted")

    expected = {"qwen3_1_7b", "llama3_2_3b"}
    if set(extension.get("models", {})) != expected:
        raise RuntimeError("architecture extension model roster drifted")
    market = float(extension["market_brier"])
    for model_key in sorted(expected):
        model = extension["models"][model_key]
        registered = registration["local_models"][model_key]
        for key in ("checkpoint", "revision", "figure_role"):
            if model.get(key) != registered[key]:
                raise RuntimeError(f"{model_key} registered {key} drifted")
        for artifact in ("summary", "raw"):
            artifact_path = resolve_recorded_path(model[artifact])
            if (
                not artifact_path.is_file()
                or sha256(artifact_path) != model[f"{artifact}_sha256"]
            ):
                raise RuntimeError(f"{model_key} {artifact} artifact drifted")
        if model.get("raw_rows") != 318:
            raise RuntimeError(f"{model_key} raw row count drifted")
        if set(model.get("models", {})) != {
            "base",
            "seed_42",
            "seed_43",
            "seed_44",
        }:
            raise RuntimeError(f"{model_key} endpoint roster drifted")
        if set(model.get("adapter_lineage", {})) != {"42", "43", "44"}:
            raise RuntimeError(f"{model_key} adapter roster drifted")
        if abs(float(model["market_brier"]) - market) > 1e-12:
            raise RuntimeError(f"{model_key} market baseline drifted")
    return extension


def load_qwen32_report(
    summary_path: Path = QWEN32_SUMMARY,
    raw_path: Path = QWEN32_RAW,
    registration_path: Path = QWEN32_REGISTRATION,
    protocol_path: Path = QWEN32_PROTOCOL,
) -> dict[str, Any]:
    """Load the prospectively registered Qwen3-32B three-seed extension."""
    for path, expected, label in (
        (summary_path, QWEN32_SUMMARY_SHA256, "summary"),
        (raw_path, QWEN32_RAW_SHA256, "raw"),
        (registration_path, QWEN32_REGISTRATION_SHA256, "registration"),
        (protocol_path, QWEN32_PROTOCOL_SHA256, "protocol"),
    ):
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Qwen3-32B {label} artifact drifted")

    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if registration.get("kind") != "registration":
        raise RuntimeError("Qwen3-32B registration kind drifted")
    if registration.get("protocol_version") != "exp4_qwen32_scale_v1":
        raise RuntimeError("Qwen3-32B extension protocol drifted")
    if registration.get("protocol_sha256") != QWEN32_PROTOCOL_SHA256:
        raise RuntimeError("Qwen3-32B registered protocol hash drifted")
    if registration.get("holdout_tasks_sha256") != HOLDOUT_SHA256:
        raise RuntimeError("Qwen3-32B registered holdout drifted")
    if registration.get("seeds") != [42, 43, 44]:
        raise RuntimeError("Qwen3-32B registered seed roster drifted")
    if registration.get("scientific_hyperparameters_changed_from_qwen3_14b"):
        raise RuntimeError("Qwen3-32B scientific settings no longer match 14B")
    model_cache = registration.get("model_cache", {})
    if model_cache.get("model") != QWEN32_MODEL:
        raise RuntimeError("Qwen3-32B registered model identity drifted")

    report = json.loads(summary_path.read_text(encoding="utf-8"))
    if report.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("Qwen3-32B parent protocol drifted")
    if (
        report.get("model_key") != QWEN32_MODEL_KEY
        or report.get("model") != QWEN32_MODEL
    ):
        raise RuntimeError("Qwen3-32B evaluated model identity drifted")
    if report.get("decoding") != EXPECTED_DECODING:
        raise RuntimeError("Qwen3-32B stochastic decoding drifted")
    holdout = report.get("holdout", {})
    if holdout.get("n") != 318 or holdout.get("test_tasks_sha256") != HOLDOUT_SHA256:
        raise RuntimeError("Qwen3-32B evaluated holdout drifted")
    if report.get("test_was_used_during_training") is not False:
        raise RuntimeError("Qwen3-32B holdout isolation flag drifted")

    expected_endpoints = {"base", "seed_42", "seed_43", "seed_44"}
    if set(report.get("models", {})) != expected_endpoints:
        raise RuntimeError("Qwen3-32B endpoint roster drifted")
    for endpoint_name, endpoint in report["models"].items():
        if endpoint.get("n") != 318 or endpoint.get("draws_per_task") != 5:
            raise RuntimeError(f"Qwen3-32B {endpoint_name} task contract drifted")
        if endpoint.get("draw_parse_coverage") != 1:
            raise RuntimeError(f"Qwen3-32B {endpoint_name} draw coverage is incomplete")
        if endpoint.get("complete_task_parse_coverage") != 1:
            raise RuntimeError(f"Qwen3-32B {endpoint_name} task coverage is incomplete")
    if set(report.get("adapter_sha256", {})) != {"42", "43", "44"}:
        raise RuntimeError("Qwen3-32B adapter roster drifted")
    for contrast in report.get("comparisons_brier", {}).values():
        if contrast.get("bootstrap_repetitions") != 5000:
            raise RuntimeError("Qwen3-32B bootstrap repetitions drifted")
        if contrast.get("family_clusters") != 223:
            raise RuntimeError("Qwen3-32B family clusters drifted")
    raw_rows = sum(
        1 for line in raw_path.read_text(encoding="utf-8").splitlines() if line
    )
    if raw_rows != 318:
        raise RuntimeError("Qwen3-32B raw row count drifted")
    return report


def load_hosted_reports(
    specs: tuple[dict[str, Any], ...] = HOSTED_REPORTS,
) -> list[dict[str, Any]]:
    """Load comparable hosted scores for the final categorical panel."""
    loaded: list[dict[str, Any]] = []
    market_values: set[float] = set()
    for spec in specs:
        path = Path(spec["path"])
        if not path.is_file():
            raise RuntimeError(f"missing hosted result for {spec['model']}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("protocol_version") != ("exp4_architecture_provider_sweep_v1"):
            raise RuntimeError(f"{spec['model']} hosted protocol drifted")
        if summary.get("classification") != "off_the_shelf_api_reference":
            raise RuntimeError(f"{spec['model']} hosted classification drifted")
        if summary.get("model") != spec["model"]:
            raise RuntimeError(f"{spec['model']} hosted identity drifted")
        holdout = summary.get("holdout", {})
        if (
            holdout.get("n") != 318
            or holdout.get("test_tasks_sha256") != HOLDOUT_SHA256
        ):
            raise RuntimeError(f"{spec['model']} hosted holdout drifted")
        endpoint = summary.get("endpoint_summary", {})
        if endpoint.get("n") != 318:
            raise RuntimeError(f"{spec['model']} hosted count drifted")
        for key in ("brier", "complete_task_parse_coverage"):
            value = endpoint.get(key)
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise RuntimeError(f"{spec['model']} hosted {key} is invalid")
        raw_path = resolve_recorded_path(summary["raw"])
        if not raw_path.is_file() or sha256(raw_path) != summary.get("raw_sha256"):
            raise RuntimeError(f"{spec['model']} hosted raw artifact drifted")
        market = float(summary["market"]["brier"])
        market_values.add(market)
        delta = summary["hosted_minus_market_brier"]
        loaded.append(
            {
                "company": spec["company"],
                "version": spec["version"],
                "icon": str(ICONS / spec["icon"]),
                "icon_scale": float(spec.get("icon_scale", 1.0)),
                "model": spec["model"],
                "brier": float(endpoint["brier"]),
                "ci_low": market + float(delta["ci95_low"]),
                "ci_high": market + float(delta["ci95_high"]),
                "parse_coverage": float(endpoint["complete_task_parse_coverage"]),
                "color": spec["color"],
            }
        )
    if len(market_values) != 1:
        raise RuntimeError("hosted models use different market baselines")
    return loaded


def trained_parse_from_finalized(model: dict[str, Any]) -> float:
    return (
        sum(
            float(model["models"][f"seed_{seed}"]["complete_task_parse_coverage"])
            for seed in (42, 43, 44)
        )
        / 3
    )


def render_table(
    report: dict[str, Any],
    llama: dict[str, Any],
    extension: dict[str, Any] | None = None,
    qwen32: dict[str, Any] | None = None,
) -> str:
    lines = [
        "% Generated by paper/generate_exp4_scale_figure.py; do not edit manually.",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        "Checkpoint & Base & Trained mean & Trained $-$ base & Trained parse " + r"\\",
        r"\midrule",
    ]

    qwen_rows: list[tuple[str, dict[str, Any], float]] = []
    if extension is not None:
        small = extension["models"]["qwen3_1_7b"]
        qwen_rows.append(
            ("Qwen3-1.7B", small, trained_parse_from_finalized(small))
        )
    for model_key, label in zip(
        MODEL_KEYS,
        (
            "Qwen3-4B-Instruct-2507",
            "Qwen3-8B",
            "Qwen3-14B",
        ),
        strict=True,
    ):
        model = report["models"][model_key]
        qwen_rows.append(
            (
                label,
                model,
                float(model["trained_seed_mean_complete_task_parse_coverage"]),
            )
        )
    if qwen32 is not None:
        qwen32_seeds = [
            float(qwen32["models"][f"seed_{seed}"]["brier"])
            for seed in (42, 43, 44)
        ]
        qwen_rows.append(
            (
                "Qwen3-32B",
                {
                    "base_brier": float(qwen32["models"]["base"]["brier"]),
                    "trained_seed_mean_brier": sum(qwen32_seeds) / 3,
                    "trained_minus_base_brier": qwen32["comparisons_brier"][
                        "trained_seed_mean_minus_base"
                    ],
                },
                sum(
                    float(
                        qwen32["models"][f"seed_{seed}"][
                            "complete_task_parse_coverage"
                        ]
                    )
                    for seed in (42, 43, 44)
                )
                / 3,
            )
        )
    for label, model, parse in qwen_rows:
        base_brier = float(model["base_brier"])
        trained_brier = float(model["trained_seed_mean_brier"])
        if "trained_minus_base_brier" in model:
            delta = float(model["trained_minus_base_brier"]["estimate"])
        else:
            delta = trained_brier - base_brier
        lines.append(
            f"{label} & {base_brier:.4f} & "
            f"{trained_brier:.4f} & {delta:+.4f} & "
            rf"{100 * parse:.1f}\% \\"
        )

    llama_rows: list[tuple[str, float, float, float]] = []
    if extension is not None:
        small_llama = extension["models"]["llama3_2_3b"]
        llama_rows.append(
            (
                "Llama-3.2-3B",
                float(small_llama["base_brier"]),
                float(small_llama["trained_seed_mean_brier"]),
                trained_parse_from_finalized(small_llama),
            )
        )
    llama_base = float(llama["models"]["base"]["brier"])
    llama_seeds = [
        float(llama["models"][f"seed_{seed}"]["brier"]) for seed in (42, 43, 44)
    ]
    llama_rows.append(
        (
            "Llama-3.1-8B",
            llama_base,
            sum(llama_seeds) / len(llama_seeds),
            sum(
                float(llama["models"][f"seed_{seed}"]["complete_task_parse_coverage"])
                for seed in (42, 43, 44)
            )
            / 3,
        )
    )
    for label, base_brier, trained_brier, parse in llama_rows:
        lines.append(
            f"{label} & {base_brier:.4f} & "
            f"{trained_brier:.4f} & {trained_brier - base_brier:+.4f} & "
            rf"{100 * parse:.1f}\% \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def finalized_point(
    model: dict[str, Any], label: str, parameters: float
) -> dict[str, Any]:
    base_parse = float(
        model.get("models", {}).get("base", {}).get("complete_task_parse_coverage", 1.0)
    )
    return {
        "label": label,
        "parameters": parameters,
        "base": float(model["base_brier"]),
        "base_parse": base_parse,
        "trained": float(model["trained_seed_mean_brier"]),
        "seeds": np.asarray(
            [float(model["trained_seed_brier"][str(seed)]) for seed in (42, 43, 44)]
        ),
    }


def draw_figure(
    report: dict[str, Any],
    llama: dict[str, Any],
    pdf: Path = PDF,
    png: Path = PNG,
    extension: dict[str, Any] | None = None,
    hosted: list[dict[str, Any]] | None = None,
    qwen32: dict[str, Any] | None = None,
) -> None:
    hosted = [] if hosted is None else hosted
    models = report["models"]
    qwen_points: list[dict[str, Any]] = []
    if extension is not None:
        qwen_points.append(
            finalized_point(extension["models"]["qwen3_1_7b"], "1.7B", 1.7)
        )
    for model_key, label, parameters in zip(
        MODEL_KEYS, MODEL_LABELS, MODEL_PARAMETERS, strict=True
    ):
        qwen_points.append(
            {
                "label": label,
                "parameters": parameters,
                "base": float(models[model_key]["base_brier"]),
                "trained": float(models[model_key]["trained_seed_mean_brier"]),
                "seeds": np.asarray(
                    [
                        float(models[model_key]["trained_seed_brier"][str(seed)])
                        for seed in (42, 43, 44)
                    ]
                ),
            }
        )
    if qwen32 is not None:
        qwen32_seeds = np.asarray(
            [
                float(qwen32["models"][f"seed_{seed}"]["brier"])
                for seed in (42, 43, 44)
            ]
        )
        qwen_points.append(
            {
                "label": "32B",
                "parameters": 32.0,
                "base": float(qwen32["models"]["base"]["brier"]),
                "trained": float(qwen32_seeds.mean()),
                "seeds": qwen32_seeds,
            }
        )

    x = np.asarray([point["parameters"] for point in qwen_points])
    base = np.asarray([point["base"] for point in qwen_points])
    trained = np.asarray([point["trained"] for point in qwen_points])
    seeds = np.asarray([point["seeds"] for point in qwen_points])
    market = float(report["baselines"]["market"]["brier"])
    if abs(float(llama["baselines"]["market"]["brier"]) - market) > 1e-12:
        raise RuntimeError("Qwen and Llama market baselines differ")
    if extension is not None and (
        abs(float(extension["market_brier"]) - market) > 1e-12
    ):
        raise RuntimeError("new local models use a different market baseline")
    if qwen32 is not None and (
        abs(float(qwen32["baselines"]["market"]["brier"]) - market) > 1e-12
    ):
        raise RuntimeError("Qwen3-32B uses a different market baseline")

    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(7.15, 1.48),
        sharey=False,
        gridspec_kw={"width_ratios": (1.55, 1.0), "wspace": 0.10},
    )
    for index in range(len(x)):
        left.plot(
            [x[index], x[index]],
            [trained[index], base[index]],
            color=QWEN_COLOR,
            alpha=0.28,
            linewidth=1.0,
            zorder=1,
        )
    for values, filled, alpha, width in (
        (base, False, 0.55, 1.15),
        (trained, True, 1.0, 1.65),
    ):
        left.plot(
            x,
            values,
            color=QWEN_COLOR,
            alpha=alpha,
            linestyle="-",
            linewidth=width,
        )
        left.scatter(
            x,
            values,
            s=38,
            marker="o",
            facecolor=QWEN_COLOR if filled else "white",
            edgecolor=QWEN_COLOR,
            linewidth=1.3,
            zorder=4,
        )

    jitter = np.asarray([0.975, 1.0, 1.025])
    for index in range(len(x)):
        left.scatter(
            x[index] * jitter,
            seeds[index],
            s=12,
            color=QWEN_SEED_COLOR,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.95,
            zorder=5,
        )

    plotted = [
        market,
        *base,
        *trained,
        *seeds.ravel(),
    ]
    plotted = [value for value in plotted if np.isfinite(value)]
    y_min = float(np.floor((min(plotted) - 0.001) / 0.005) * 0.005)
    y_max = float(np.ceil((max(plotted) + 0.001) / 0.005) * 0.005)
    y_span = y_max - y_min
    y_step = 0.02 if y_span > 0.08 else 0.01 if y_span > 0.04 else 0.005
    y_tick_min = np.ceil(y_min / y_step) * y_step
    y_ticks = np.arange(y_tick_min, y_max + y_step / 2, y_step)

    left.axhline(market, color=MARKET_COLOR, linestyle="--", linewidth=1.05)
    parameter_ticks = sorted(x.tolist())
    left.set_xscale("log", base=2)
    left.set_xticks(
        parameter_ticks,
        [f"{value:g}" for value in parameter_ticks],
    )
    left.set_xlim(min(parameter_ticks) / 1.22, max(parameter_ticks) * 1.16)
    left.set_ylim(y_min, y_max)
    left.set_yticks(y_ticks)
    left.set_xlabel("Parameters (billions; log scale)")
    left.set_ylabel("Brier score (lower is better)")
    left.set_title(
        "Qwen3 checkpoints (log parameter scale)",
        loc="left",
        fontweight="bold",
    )
    left.grid(axis="y", color=GRID_COLOR, linewidth=0.55, alpha=0.85)
    left.spines[["top", "right"]].set_visible(False)

    right.axhline(market, color=MARKET_COLOR, linestyle="--", linewidth=1.05)
    hosted_x = np.arange(len(hosted), dtype=float)
    for point_x, point in zip(hosted_x, hosted, strict=True):
        right.errorbar(
            point_x,
            point["brier"],
            yerr=np.asarray(
                [
                    [point["brier"] - point["ci_low"]],
                    [point["ci_high"] - point["brier"]],
                ]
            ),
            fmt="o",
            markersize=5.5,
            color=point["color"],
            ecolor=point["color"],
            elinewidth=1.1,
            capsize=2.3,
            markeredgecolor="white",
            markeredgewidth=0.55,
            zorder=4,
        )
        right.text(
            point_x,
            point["ci_high"] + 0.0008,
            f"{point['brier']:.4f}",
            ha="center",
            va="bottom",
            color=point["color"],
            fontsize=6.2,
        )
    if hosted:
        hosted_values = [
            market,
            *[point["ci_low"] for point in hosted],
            *[point["ci_high"] for point in hosted],
        ]
        hosted_y_min = float(np.floor((min(hosted_values) - 0.001) / 0.005) * 0.005)
        hosted_y_max = float(np.ceil((max(hosted_values) + 0.001) / 0.005) * 0.005)
        right.set_ylim(hosted_y_min, hosted_y_max)
        hosted_span = hosted_y_max - hosted_y_min
        hosted_step = 0.01 if hosted_span > 0.04 else 0.005
        right.set_yticks(
            np.arange(hosted_y_min, hosted_y_max + hosted_step / 2, hosted_step)
        )
        right.set_xticks(
            hosted_x,
            [point["version"] for point in hosted],
        )
        brand_transform = blended_transform_factory(right.transData, right.transAxes)
        for point_x, point in zip(hosted_x, hosted, strict=True):
            icon_path = Path(point["icon"])
            if not icon_path.is_file():
                raise RuntimeError(f"missing hosted brand icon: {icon_path}")
            icon = plt.imread(icon_path)
            imagebox = OffsetImage(
                icon,
                zoom=(11.0 / icon.shape[0]) * point["icon_scale"],
            )
            right.add_artist(
                AnnotationBbox(
                    imagebox,
                    (point_x, 0.055),
                    xycoords=brand_transform,
                    frameon=False,
                    box_alignment=(0.5, 0.5),
                    zorder=3,
                )
            )
        right.set_xlim(-0.45, len(hosted) - 0.55)
    else:
        right.set_xticks([])
        right.text(
            0.5,
            0.5,
            "No hosted references",
            transform=right.transAxes,
            ha="center",
            va="center",
            color="#6B7280",
        )
    right.set_xlabel("API deployment (categorical axis)")
    right.set_title(
        "Hosted frontier references",
        loc="left",
        fontweight="bold",
    )
    right.grid(axis="y", color=GRID_COLOR, linewidth=0.55, alpha=0.85)
    right.spines[["top", "right"]].set_visible(False)
    right.tick_params(axis="y", left=True, labelleft=True)

    encoding_legend = [
        Line2D(
            [0],
            [0],
            color="#6B7280",
            marker="o",
            markerfacecolor="white",
            linestyle="none",
            label="Untrained base",
        ),
        Line2D(
            [0],
            [0],
            color="#6B7280",
            marker="o",
            linestyle="none",
            label="Trained mean",
        ),
        Line2D(
            [0],
            [0],
            color="#9CA3AF",
            marker="o",
            linestyle="none",
            markersize=3.5,
            label="Trained seeds",
        ),
        Line2D(
            [0],
            [0],
            color=MARKET_COLOR,
            linestyle="--",
            label="Polymarket",
        ),
    ]
    left.legend(
        handles=encoding_legend,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=1,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.45,
    )

    pdf.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Creator": "paper/generate_exp4_scale_figure.py",
        "Title": "Experiment 4 local scale and hosted frontier results",
        "CreationDate": None,
    }
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.025, metadata=metadata)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def draw_llama_figure(
    llama: dict[str, Any],
    extension: dict[str, Any],
    pdf: Path = LLAMA_PDF,
    png: Path = LLAMA_PNG,
) -> None:
    market = float(llama["baselines"]["market"]["brier"])
    if abs(float(extension["market_brier"]) - market) > 1e-12:
        raise RuntimeError("Llama endpoints use different market baselines")

    points = [
        finalized_point(extension["models"]["llama3_2_3b"], "3B", 3.0)
    ]
    seeds_8b = np.asarray(
        [float(llama["models"][f"seed_{seed}"]["brier"]) for seed in (42, 43, 44)]
    )
    points.append(
        {
            "label": "8B",
            "parameters": 8.0,
            "base": float(llama["models"]["base"]["brier"]),
            "trained": float(seeds_8b.mean()),
            "seeds": seeds_8b,
        }
    )

    x = np.asarray([point["parameters"] for point in points])
    base = np.asarray([point["base"] for point in points])
    trained = np.asarray([point["trained"] for point in points])
    seeds = np.asarray([point["seeds"] for point in points])

    fig, axis = plt.subplots(figsize=(3.65, 1.75))
    for point_x, base_value, trained_value in zip(x, base, trained, strict=True):
        axis.plot(
            [point_x, point_x],
            [trained_value, base_value],
            color=LLAMA_COLOR,
            alpha=0.28,
            linewidth=1.0,
            zorder=1,
        )
    for values, filled, alpha, width in (
        (base, False, 0.55, 1.15),
        (trained, True, 1.0, 1.65),
    ):
        axis.plot(x, values, color=LLAMA_COLOR, alpha=alpha, linewidth=width)
        axis.scatter(
            x,
            values,
            s=42,
            marker="s",
            facecolor=LLAMA_COLOR if filled else "white",
            edgecolor=LLAMA_COLOR,
            linewidth=1.3,
            zorder=4,
        )
    jitter = np.asarray([0.975, 1.0, 1.025])
    for index in range(len(x)):
        axis.scatter(
            x[index] * jitter,
            seeds[index],
            s=14,
            marker="s",
            color=LLAMA_SEED_COLOR,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.95,
            zorder=5,
        )
    axis.axhline(market, color=MARKET_COLOR, linestyle="--", linewidth=1.05)
    axis.set_xscale("log", base=2)
    axis.set_xticks(x, [point["label"] for point in points])
    axis.set_xlim(min(x) / 1.18, max(x) * 1.18)
    plotted = [market, *base, *trained, *seeds.ravel()]
    y_min = float(np.floor((min(plotted) - 0.002) / 0.01) * 0.01)
    y_max = float(np.ceil((max(plotted) + 0.002) / 0.01) * 0.01)
    axis.set_ylim(y_min, y_max)
    axis.set_yticks(np.arange(y_min, y_max + 0.001, 0.02))
    axis.set_xlabel("Parameters (billions; log scale)")
    axis.set_ylabel("Brier score (lower is better)")
    axis.set_title("Llama replication", loc="left", fontweight="bold")
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.55, alpha=0.85)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        handles=[
            Line2D(
                [0], [0], color="#6B7280", marker="s", markerfacecolor="white",
                linestyle="none", label="Untrained base"
            ),
            Line2D(
                [0], [0], color="#6B7280", marker="s", linestyle="none",
                label="Trained mean"
            ),
            Line2D(
                [0], [0], color=LLAMA_SEED_COLOR, marker="s", linestyle="none",
                markersize=3.5, label="Trained seeds"
            ),
            Line2D(
                [0], [0], color=MARKET_COLOR, linestyle="--", label="Polymarket"
            ),
        ],
        loc="upper right",
        frameon=False,
        handlelength=1.8,
        handletextpad=0.45,
    )
    pdf.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Creator": "paper/generate_exp4_scale_figure.py",
        "Title": "Experiment 4 Llama scale replication",
        "CreationDate": None,
    }
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.025, metadata=metadata)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--architecture-extension", type=Path)
    args = parser.parse_args()

    report = load_report(args.summary)
    llama = load_llama_report()
    qwen32 = load_qwen32_report()
    extension_path = args.architecture_extension
    if extension_path is None and ARCHITECTURE_EXTENSION.is_file():
        extension_path = ARCHITECTURE_EXTENSION
    extension = (
        load_architecture_extension(extension_path)
        if extension_path is not None
        else None
    )
    hosted = load_hosted_reports()
    draw_figure(
        report,
        llama,
        extension=extension,
        hosted=hosted,
        qwen32=qwen32,
    )
    if extension is not None:
        draw_llama_figure(llama, extension)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(
        render_table(report, llama, extension, qwen32),
        encoding="utf-8",
    )
    print(f"figure -> {PDF}")
    print(f"figure -> {PNG}")
    if extension is not None:
        print(f"figure -> {LLAMA_PDF}")
        print(f"figure -> {LLAMA_PNG}")
    print(f"table -> {TABLE}")


if __name__ == "__main__":
    main()
