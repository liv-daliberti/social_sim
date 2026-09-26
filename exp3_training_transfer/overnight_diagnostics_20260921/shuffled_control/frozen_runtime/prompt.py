"""Paired prompt renderers and strict parser for the C3 mechanism family."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from output_contract import gold_forecast_array, parse_forecast_array

from worlds import SCENARIO_LABELS, mechanism_description, scenario_grid

MECHANISM_BEGIN = "[[MECHANISM_DESCRIPTION_BEGIN]]"
MECHANISM_END = "[[MECHANISM_DESCRIPTION_END]]"
OUTPUT_PREFIX = '{"forecasts":['
OUTPUT_SUFFIX = "]}"


def _table(driver_labels: Sequence[str], outcome_label: str, inputs, observed) -> str:
    headers = ["Step", *driver_labels, outcome_label]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for index, (row, outcome) in enumerate(zip(np.asarray(inputs), np.asarray(observed)), start=1):
        values = [str(index), *(f"{float(value):+.1f}" for value in row), f"{float(outcome):.2f}"]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_prompt(world, bundle: dict, disclosure: str) -> str:
    if disclosure not in {"disclosed", "undisclosed"}:
        raise ValueError(f"unknown disclosure: {disclosure}")
    header = [
        f"You are forecasting {world.outcome_label} in {world.domain}.",
        f"Time advances one {world.period} at a time. Signed drivers may be positive, negative, or zero.",
        f"The usual outcome level is near {world.baseline:g}; valid forecasts range from "
        f"{world.clip[0]:g} to {world.clip[1]:g}.",
    ]
    if disclosure == "disclosed":
        header.extend([MECHANISM_BEGIN, mechanism_description(world), MECHANISM_END])
    else:
        header.append(
            "The repeatable response pattern is not described. Infer it from the calibration "
            "trajectories and the target trajectory."
        )

    calibration_parts = []
    for index, calibration in enumerate(bundle["calibrations"], start=1):
        calibration_parts.extend([
            "",
            f"Calibration trajectory {index} (same response system; its own fixed hidden sensitivity):",
            _table(world.driver_labels, world.outcome_label,
                   calibration["inputs"], calibration["observed"]),
        ])

    target_inputs = np.asarray(bundle["target_inputs"])
    target_observed = np.asarray(bundle["target_observed"])
    target = [
        "",
        "Target trajectory (infer its own fixed hidden sensitivity):",
        _table(world.driver_labels, world.outcome_label, target_inputs, target_observed),
        "",
        "Starting immediately after the target trajectory, consider each intervention independently.",
        "The named shock is applied to the first driver for one period; every driver is zero afterward.",
    ]
    for scenario in scenario_grid():
        target.append(
            f"- Scenario {scenario['label']}: shock {scenario['shock']:+g}; forecast horizon "
            f"{scenario['horizon']} {world.period}(s)."
        )

    tail = [
        "",
        "Forecast every scenario using one coherent interpretation of the target trajectory.",
        "Put the ten forecasts in scenario order A through J.",
        "Use exactly the key shown, exactly ten numeric entries, and no prose or extra keys.",
        f"Start your response with {OUTPUT_PREFIX}",
        "Write exactly ten finite decimal numbers separated by commas inside that list.",
        f"After the tenth number, write {OUTPUT_SUFFIX} and stop.",
    ]
    return "\n".join(header + calibration_parts + target + tail)


def parse_forecasts(text: str, labels: Sequence[str] = SCENARIO_LABELS, clip=(0.0, 100.0)):
    """Apply the same strict compact-array contract used by the training reward."""
    return parse_forecast_array(text, len(labels), clip)


def gold_response(targets: Sequence[float]) -> str:
    return gold_forecast_array(targets)
