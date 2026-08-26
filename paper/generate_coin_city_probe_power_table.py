#!/usr/bin/env python3
"""Emit the power-matched Qwen3-14B semantic-versus-symbol probe table body.

Reads the two 88-episode probe runs and fails closed if either is missing, is not
complete, or does not carry the expected sealed-test size.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE_ROOT = (
    ROOT
    / "exp2_v2"
    / "biased_news"
    / "data"
    / "coin_city_stable_relationship_claude_n250_v4"
    / "mechanistic_probe"
)
OUTPUT = ROOT / "paper" / "tables" / "exp2_probe_power_matched.tex"
# Each row is a run at the same 88-episode sealed test. The behavioural contrast
# is carried alongside so the table shows decodability tracking behaviour rather
# than leaving that to a cross-reference.
RUNS = (
    ("qwen3_14b_probe_v2", "Qwen3-14B, semantic", None),
    ("qwen3_14b_symbol_probe_v2", "Qwen3-14B, symbol", "Qwen3-14B"),
    ("qwen3_8b_symbol_probe_v2", "Qwen3-8B, symbol", "Qwen3-8B"),
    ("qwen3_4b_symbol_probe_v2", "Qwen3-4B, symbol", "Qwen3-4B-Instruct-2507"),
    ("qwen2_5_32b_symbol_probe_v2", "Qwen2.5-32B, symbol", "Qwen2.5-32B-Instruct"),
    ("llama3_1_8b_symbol_probe_v2", "Llama~3.1-8B, symbol", "Llama-3.1-8B-Instruct"),
)
COMPARISON = (
    ROOT
    / "exp2_v2"
    / "biased_news"
    / "data"
    / "coin_city_stable_relationship_claude_n250_v4"
    / "analysis"
    / "symbol_context_model_comparison_20260824.json"
)
EXPECTED_TEST_EPISODES = 88
CONDITIONS = ("k0:regime", "k4:regime", "k0:slope", "k4:slope")


def _dot(value: float, places: int = 3) -> str:
    text = f"{value:.{places}f}"
    for prefix, replacement in (("0.", "."), ("-0.", "-.")):
        if text.startswith(prefix):
            return replacement + text[len(prefix):]
    return text


def _score(condition: dict, arm: str, target: str, key: str) -> float | None:
    probe = condition["selected_test"].get(arm, {}).get("probe", {}).get(target)
    if probe is None:
        return None
    return probe.get(key)


def _p(value: float) -> str:
    return "$<$.001" if value < 0.001 else f"${_dot(value, 4)}$"


def load(run: str) -> dict:
    directory = PROBE_ROOT / run
    results = json.loads((directory / "probe_results.json").read_text())
    manifest = json.loads((directory / "task_manifest.json").read_text())
    if results.get("status") != "probe_complete":
        raise SystemExit(f"{run}: status is {results.get('status')!r}")
    test_episodes = manifest.get("episode_counts", {}).get("test")
    if test_episodes is None:
        # The semantic builder records per-cell splits rather than totals.
        test_episodes = sum(
            cell["test"] for cell in manifest.get("cell_counts", {}).values()
        ) or None
    if test_episodes != EXPECTED_TEST_EPISODES:
        raise SystemExit(f"{run}: sealed test is {test_episodes}, expected {EXPECTED_TEST_EPISODES}")
    if results.get("tasks_sha256") != manifest.get("tasks_sha256"):
        raise SystemExit(f"{run}: probe results do not match the frozen task file")
    return results


def behavioural_delta(model: str | None) -> str:
    """Symbol-minus-no-context slope correlation for the same checkpoint."""
    if model is None:
        return "---"
    curves = json.loads(COMPARISON.read_text())["models"][model]["curves"]["0"]
    metric = curves["paired_discrimination"]["symbol_minus_no_context_rho"]
    low, high = metric["ci_95"]
    text = f"${_dot(metric['difference'])}$"
    return f"\\textbf{{{text}}}" if low * high > 0 else text


def rows() -> str:
    lines = []
    for index, (run, label, behaviour_key) in enumerate(RUNS):
        results = load(run)
        conditions = results["primary"]["conditions"]
        pooled = (
            results["pooling_controls"]["mean_pool_source_layer_0"]["conditions"]
        )
        if index:
            lines.append(r"\midrule")
        for position, name in enumerate(CONDITIONS):
            condition = conditions[name]
            key = "roc_auc" if name.endswith("regime") else "r2"
            prefix, target = name.split(":")
            pretty = f"${prefix[0]}{{=}}{prefix[1:]}$ " + (
                "regime AUC" if target == "regime" else "slope $R^2$"
            )
            embed = _score(pooled[name], "abc_context", "true_target", key)
            math = lambda value: f"${_dot(value)}$"
            cells = [
                math(_score(condition, "abc_context", "true_target", key)),
                math(_score(condition, "abc_no_context", "true_target", key)),
                math(_score(condition, "abc_wrong_context", "true_target", key)),
                math(_score(condition, "abc_wrong_context", "cue_target", key)),
                str(condition["selected_layer"]),
                math(embed) if embed is not None else "---",
                _p(condition["permutation_null_on_correct_context_test"]["one_sided_p"]),
                behavioural_delta(behaviour_key) if position == 0 else "",
            ]
            lead = label if position == 0 else ""
            lines.append(f"{lead} & {pretty} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    preamble = (
        "\\begin{tabular}{l l rrrr r r r r}\n"
        "\\toprule\n"
        "Run & Target & Correct & None & Wrong & Wrong/cue & Layer & Embed. & $p$"
        " & Behav. $\\Delta\\rho$ \\\\\n"
        "\\midrule"
    )
    OUTPUT.write_text(
        "% Generated by paper/generate_coin_city_probe_power_table.py; do not edit.\n"
        + preamble
        + "\n"
        + rows()
        + "\n\\bottomrule\n\\end{tabular}\n"
    )
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
