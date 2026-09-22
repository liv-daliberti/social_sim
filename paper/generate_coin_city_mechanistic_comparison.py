#!/usr/bin/env python3
"""Generate the cross-model Coin City probe assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT
    / "exp2_v2"
    / "biased_news"
    / "data"
    / "coin_city_stable_relationship_claude_n250_v4"
    / "mechanistic_probe"
)
EXPECTED_TASKS_SHA256 = (
    "2e0f65d6801b96616fb53b764470960a96b4c08c73483006ca626477de30ccb8"
)
EXPECTED_NULL_SCHEME = (
    "held-out target-label permutation with the development-selected "
    "layer, ridge strength, and fitted probe held fixed"
)
BEHAVIOR_SCREEN = BASE / "behavior_screens" / "size_screen_v2.json"
EXPECTED_BEHAVIOR_SHA256 = (
    "d62e50d87f0ea5465a98fbb65efee825ddbdf60a3865a2aec23fa86222e1f8c4"
)
EXPECTED_BEHAVIOR_MODELS = {
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-32B",
    "Qwen2.5-72B-Instruct",
}


@dataclass(frozen=True)
class RunSpec:
    label: str
    directory: str
    status: str
    commit: str
    features_sha256: str
    layers: int
    hidden: int


RUNS = (
    RunSpec(
        label="Qwen3-4B",
        directory="qwen3_4b_probe_v1",
        status="probe_complete",
        commit="cdbee75f17c01a7cc42f958dc650907174af0554",
        features_sha256=(
            "e191385963d639aa5b8574a9abbeb91b3519aed934ac491f863655631b26dbbc"
        ),
        layers=37,
        hidden=2560,
    ),
    RunSpec(
        label="Qwen3-8B",
        directory="qwen3_8b_toy_v1",
        status="toy_complete",
        commit="b968826d9c46dd6066d109eabc6255188de91218",
        features_sha256=(
            "e1d99e5913d1b14e00697276298f5c015f56fbb09eb59731bbabad836bb2078b"
        ),
        layers=37,
        hidden=4096,
    ),
    RunSpec(
        label="Llama-3.1-8B",
        directory="llama3_1_8b_probe_v1",
        status="probe_complete",
        commit="0e9e39f249a16976918f6564b8830bc894c89659",
        features_sha256=(
            "7f65aeec7a91e404cc60068618fedf25c679abde5d48c2b8477f736670fbc54f"
        ),
        layers=33,
        hidden=4096,
    ),
    RunSpec(
        label="Qwen3-14B",
        directory="qwen3_14b_probe_v1",
        status="probe_complete",
        commit="40c069824f4251a91eefaf281ebe4c544efd3e18",
        features_sha256=(
            "0c14369a9eeda0fab0dc28b83e0a03d52663e49a2c68c143880e46bfe944247a"
        ),
        layers=41,
        hidden=5120,
    ),
    RunSpec(
        label="Qwen3-32B",
        directory="qwen3_32b_probe_v1",
        status="probe_complete",
        commit="9216db5781bf21249d130ec9da846c4624c16137",
        features_sha256=(
            "a14d351b59350cc0d1e8c0853c38efe8f739f53d84551cf1f247ec0864c83626"
        ),
        layers=65,
        hidden=5120,
    ),
    RunSpec(
        label="Qwen2.5-72B",
        directory="qwen2_5_72b_probe_v1",
        status="probe_complete",
        commit="495f39366efef23836d0cfae4fbe635880d2be31",
        features_sha256=(
            "0c9a87ffe32bb12296b2028817c489ed52cf9e348c98133f4826fcb3d4374d88"
        ),
        layers=81,
        hidden=8192,
    ),
)
BEHAVIOR_RUN_LABELS = {
    "Qwen3-8B",
    "Qwen3-14B",
    "Qwen3-32B",
    "Qwen2.5-72B",
}
TARGETS = (
    ("regime", "Regime (AUC)", "roc_auc", 0.5),
    ("slope", r"Full slope ($R^2$)", "r2", 0.0),
    ("residual_slope", r"Residual slope ($R^2$)", "r2", 0.0),
)
COLORS = {0: "#2C7FB8", 4: "#D55E00"}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_behavior_screen() -> dict:
    if sha256(BEHAVIOR_SCREEN) != EXPECTED_BEHAVIOR_SHA256:
        raise RuntimeError("Behavior-screen artifact hash drifted")
    screen = json.loads(BEHAVIOR_SCREEN.read_text())
    if screen.get("status") != "complete":
        raise RuntimeError("Behavior screen is not complete")
    if screen.get("tasks_sha256") != EXPECTED_TASKS_SHA256:
        raise RuntimeError("Behavior-screen task hash drifted")
    if set(screen.get("models", {})) != EXPECTED_BEHAVIOR_MODELS:
        raise RuntimeError("Behavior-screen model roster drifted")
    for label, model in screen["models"].items():
        summary = model["manifest"]["behavior_summary"]
        if model["behavior"].get("status") != "complete":
            raise RuntimeError(f"{label}: behavior analysis is incomplete")
        if summary.get("record_count") != 96 or summary.get("parsed_count") != 96:
            raise RuntimeError(f"{label}: behavior screen is incomplete")

    return screen

def load_run(spec: RunSpec) -> dict:
    run = BASE / spec.directory
    results = json.loads((run / "probe_results.json").read_text())
    tasks = json.loads((run / "task_manifest.json").read_text())
    extraction = json.loads((run / "extraction_manifest.json").read_text())
    with (run / "layerwise_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    if results["status"] != spec.status:
        raise RuntimeError(f"{spec.label}: analysis is not complete")
    if tasks["record_count"] != 384 or tasks["test_per_cell"] != 4:
        raise RuntimeError(f"{spec.label}: task dimensions drifted")
    if tasks["tasks_sha256"] != EXPECTED_TASKS_SHA256:
        raise RuntimeError(f"{spec.label}: task manifest hash drifted")
    if sha256(run / "tasks.jsonl") != EXPECTED_TASKS_SHA256:
        raise RuntimeError(f"{spec.label}: tasks.jsonl hash drifted")
    if results["tasks_sha256"] != EXPECTED_TASKS_SHA256:
        raise RuntimeError(f"{spec.label}: analysis task hash drifted")
    if results["features_sha256"] != spec.features_sha256:
        raise RuntimeError(f"{spec.label}: analysis feature hash drifted")
    if sha256(run / "hidden_states.npz") != spec.features_sha256:
        raise RuntimeError(f"{spec.label}: hidden-state file hash drifted")
    if extraction["model_commit"] != spec.commit:
        raise RuntimeError(f"{spec.label}: checkpoint commit drifted")
    expected_shape = [384, spec.layers, spec.hidden]
    if extraction["feature_summary"]["feature_shape"] != expected_shape:
        raise RuntimeError(f"{spec.label}: hidden-state shape drifted")

    primary = [row for row in rows if row["representation"] == "last_input_token"]
    for target, *_ in TARGETS:
        for k in (0, 4):
            observed_layers = sorted(
                int(row["layer"])
                for row in primary
                if row["target"] == target and int(row["c_cases"]) == k
            )
            if observed_layers != list(range(spec.layers)):
                raise RuntimeError(f"{spec.label}: incomplete {target}/k={k} curve")
            null = results["primary"]["conditions"][f"k{k}:{target}"][
                "permutation_null_on_correct_context_test"
            ]
            if null.get("scheme") != EXPECTED_NULL_SCHEME or null["repeats"] != 10_000:
                raise RuntimeError(f"{spec.label}: permutation audit drifted")
    return {"spec": spec, "results": results, "primary": primary}


def number(value: float) -> str:
    text = f"{value:.3f}"
    if text.startswith("0."):
        return text[1:]
    if text.startswith("-0."):
        return "-" + text[2:]
    return text


def p_text(value: float) -> str:
    return r"$<.001$" if value < 0.001 else f"${number(value)}$"


def signed_number(value: float) -> str:
    rendered = number(value)
    return rendered if value < 0 else f"+{rendered}"


def contrast_text(metric: dict) -> str:
    low, high = metric["ci95"]
    return (
        f"${signed_number(metric['mean'])}$ "
        f"$[{signed_number(low)},{signed_number(high)}]$"
    )


def behavior_table(screen: dict) -> str:
    key_by_label = {
        "Qwen3-8B": "Qwen3-8B",
        "Qwen3-14B": "Qwen3-14B",
        "Qwen3-32B": "Qwen3-32B",
        "Qwen2.5-72B": "Qwen2.5-72B-Instruct",
    }
    lines = [
        "% Generated by paper/generate_coin_city_mechanistic_comparison.py; do not edit.",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & Evidence & Selection & Reversal & Forecast penalty \\",
        r"\midrule",
    ]
    for spec in RUNS:
        if spec.label not in BEHAVIOR_RUN_LABELS:
            continue
        contrasts = screen["models"][key_by_label[spec.label]]["paired_contrasts"]
        evidence = contrasts["evidence_k4_minus_k0"]["abc_no_context"][
            "regime_aligned_slope"
        ]
        k0 = contrasts["by_prefix"]["k0"]
        selection = k0["context_minus_no_context"]["regime_aligned_slope"]
        reversal = k0["wrong_context_minus_correct_context"][
            "regime_aligned_slope"
        ]
        penalty = k0["wrong_context_minus_correct_context"]["poll_absolute_error"]
        lines.append(
            f"{spec.label} & {contrast_text(evidence)} & "
            f"{contrast_text(selection)} & {contrast_text(reversal)} & "
            f"{contrast_text(penalty)} \\\\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def table_body(runs: list[dict]) -> str:
    lines = [
        "% Generated by paper/generate_coin_city_mechanistic_comparison.py; do not edit.",
        r"\begin{tabular}{lllr rrrrrr}",
        r"\toprule",
        r"Model & $k$ & Target & Layer & Dev CV & Correct & None & Wrong/true & Wrong/cue & $p$ \\",
        r"\midrule",
    ]
    for run_index, loaded in enumerate(runs):
        conditions = loaded["results"]["primary"]["conditions"]
        for k in (0, 4):
            for target_index, (target, label, metric, _) in enumerate(TARGETS):
                condition = conditions[f"k{k}:{target}"]
                selected = condition["selected_test"]
                correct = selected["abc_context"]["probe"]["true_target"][metric]
                absent = selected["abc_no_context"]["probe"]["true_target"][metric]
                wrong = selected["abc_wrong_context"]["probe"]
                wrong_true = wrong["true_target"][metric]
                wrong_cue = wrong.get("cue_target", {}).get(metric)
                model = loaded["spec"].label if k == 0 and target_index == 0 else ""
                k_text = str(k) if target_index == 0 else ""
                cue_text = "--" if wrong_cue is None else number(wrong_cue)
                p_value = condition["permutation_null_on_correct_context_test"][
                    "one_sided_p"
                ]
                lines.append(
                    f"{model} & {k_text} & {label} & {condition['selected_layer']} & "
                    f"{number(condition['development_cv_score'])} & {number(correct)} & "
                    f"{number(absent)} & {number(wrong_true)} & {cue_text} & "
                    f"{p_text(p_value)} \\\\"
                )
            if k == 0:
                lines.append(r"\cmidrule(lr){2-10}")
        if run_index + 1 < len(runs):
            lines.append(r"\midrule")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def summary_table(runs: list[dict]) -> str:
    lines = [
        "% Generated by paper/generate_coin_city_mechanistic_comparison.py; do not edit.",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Model & Regime AUC & Full-slope $R^2$ & Residual-slope $R^2$ & Residual $p$ \\",
        r"\midrule",
    ]
    for loaded in runs:
        conditions = loaded["results"]["primary"]["conditions"]

        def pair(target: str, metric: str) -> str:
            values = []
            for k in (0, 4):
                selected = conditions[f"k{k}:{target}"]["selected_test"]
                values.append(selected["abc_context"]["probe"]["true_target"][metric])
            return "/".join(number(value) for value in values)

        residual_ps = [
            conditions[f"k{k}:residual_slope"][
                "permutation_null_on_correct_context_test"
            ]["one_sided_p"]
            for k in (0, 4)
        ]
        row = (
            f"{loaded['spec'].label} & {pair('regime', 'roc_auc')} & "
            f"{pair('slope', 'r2')} & {pair('residual_slope', 'r2')} & "
            f"{'/'.join(number(value) for value in residual_ps)}"
        )
        lines.append(row + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return "\n".join(lines) + "\n"


def make_figure(runs: list[dict], output_base: Path) -> None:
    fig, axes = plt.subplots(
        len(runs),
        3,
        figsize=(7.05, 1.25 * len(runs) + 0.5),
        sharex=True,
        squeeze=False,
    )
    limits = ((0.46, 1.04), (-0.04, 0.93), (-0.035, 0.17))
    for row_index, loaded in enumerate(runs):
        spec = loaded["spec"]
        conditions = loaded["results"]["primary"]["conditions"]
        for col_index, ((target, title, _, reference), ylim) in enumerate(
            zip(TARGETS, limits)
        ):
            ax = axes[row_index, col_index]
            for k in (0, 4):
                rows = sorted(
                    (
                        row
                        for row in loaded["primary"]
                        if row["target"] == target and int(row["c_cases"]) == k
                    ),
                    key=lambda row: int(row["layer"]),
                )
                raw_x = [int(row["layer"]) for row in rows]
                x = [value / (spec.layers - 1) for value in raw_x]
                y = [float(row["cv_score"]) for row in rows]
                ax.plot(x, y, color=COLORS[k], linewidth=1.4, label=f"$k={k}$")
                chosen = conditions[f"k{k}:{target}"]["selected_layer"]
                chosen_index = raw_x.index(chosen)
                ax.scatter(
                    [x[chosen_index]],
                    [y[chosen_index]],
                    color=COLORS[k],
                    marker="*",
                    s=38,
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=3,
                )
            ax.axhline(reference, color="#777777", linestyle="--", linewidth=0.8)
            ax.set_ylim(*ylim)
            ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
            ax.spines[["top", "right"]].set_visible(False)
            if row_index == 0:
                ax.set_title(title, pad=4)
            if row_index == len(runs) - 1:
                ax.set_xlabel("Normalized layer depth")
            metric_label = "CV AUC" if target == "regime" else r"CV $R^2$"
            if col_index == 0:
                ax.set_ylabel(f"{spec.label}\n{metric_label}")
            else:
                ax.set_ylabel(metric_label)
            ax.set_xticks((0.0, 0.25, 0.5, 0.75, 1.0))
    axes[0, 0].legend(frameon=False, loc="lower right")
    fig.tight_layout(w_pad=1.1, h_pad=1.2)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_main_figure(runs: list[dict], output_base: Path) -> None:
    """Summarize held-out slope decoding for the main paper."""
    conditions = (
        ("abc_context", "true_target", "Correct\ncue"),
        ("abc_wrong_context", "true_target", "Inverted cue,\ntrue target"),
        ("abc_wrong_context", "cue_target", "Inverted cue,\ncue target"),
    )
    fig, axis = plt.subplots(figsize=(3.35, 2.25))
    x = list(range(len(conditions)))
    all_series: list[list[float]] = []
    for run_index, loaded in enumerate(runs):
        primary = loaded["results"]["primary"]["conditions"]
        for k_index, k in enumerate((0, 4)):
            selected = primary[f"k{k}:slope"]["selected_test"]
            values = [
                selected[arm]["probe"][score_target]["r2"]
                for arm, score_target, _ in conditions
            ]
            all_series.append(values)
            center = (2 * len(runs) - 1) / 2
            offset = ((2 * run_index + k_index) - center) * 0.012
            shifted_x = [value + offset for value in x]
            axis.plot(
                shifted_x,
                values,
                color=COLORS[k],
                alpha=0.30,
                linewidth=0.8,
                zorder=1,
            )
            axis.scatter(
                shifted_x,
                values,
                color=COLORS[k],
                alpha=0.72,
                marker="o" if k == 0 else "s",
                s=15,
                linewidth=0,
                zorder=2,
            )

    medians = [median(series[index] for series in all_series) for index in x]
    axis.plot(x, medians, color="#4B1F6F", linewidth=2.3, zorder=3)
    axis.scatter(x, medians, color="#4B1F6F", s=24, zorder=4)
    axis.axhline(0.0, color="#777777", linestyle="--", linewidth=0.8)
    axis.set_ylabel(r"Held-out $R^2$")
    axis.set_ylim(-3.15, 1.05)
    axis.set_yticks((-3.0, -2.0, -1.0, 0.0, 1.0))
    axis.set_xticks(x)
    axis.set_xticklabels([label for _, _, label in conditions])
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    handles = (
        Line2D([], [], color=COLORS[0], marker="o", linewidth=0.8, label="$k=0$"),
        Line2D([], [], color=COLORS[4], marker="s", linewidth=0.8, label="$k=4$"),
        Line2D([], [], color="#4B1F6F", linewidth=2.3, label="Median"),
    )
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "paper" / "tables" / "exp2_qwen3_mechanistic_probe.tex",
    )
    parser.add_argument(
        "--behavior-table",
        type=Path,
        default=ROOT / "paper" / "tables" / "exp2_qwen_behavior_gate.tex",
    )
    parser.add_argument(
        "--summary-table",
        type=Path,
        default=ROOT / "paper" / "tables" / "exp2_open_model_mechanistic_summary.tex",
    )
    parser.add_argument(
        "--figure-base",
        type=Path,
        default=ROOT / "paper" / "figures" / "exp2_qwen3_mechanistic_probe",
    )
    parser.add_argument(
        "--main-figure-base",
        type=Path,
        default=ROOT / "paper" / "figures" / "exp2_qwen3_mechanistic_main",
    )
    args = parser.parse_args()
    screen = validate_behavior_screen()
    runs = [load_run(spec) for spec in RUNS]
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(table_body(runs), encoding="utf-8")
    args.summary_table.parent.mkdir(parents=True, exist_ok=True)
    args.summary_table.write_text(summary_table(runs), encoding="utf-8")
    args.behavior_table.parent.mkdir(parents=True, exist_ok=True)
    args.behavior_table.write_text(behavior_table(screen), encoding="utf-8")
    make_figure(runs, args.figure_base)
    make_main_figure(runs, args.main_figure_base)
    print(args.table)
    print(args.summary_table)
    print(args.behavior_table)
    print(args.figure_base.with_suffix(".pdf"))
    print(args.figure_base.with_suffix(".png"))
    print(args.main_figure_base.with_suffix(".pdf"))
    print(args.main_figure_base.with_suffix(".png"))


if __name__ == "__main__":
    main()
