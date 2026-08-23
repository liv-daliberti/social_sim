"""Prompt renderer for the locked Coin City structural-transfer study."""
from __future__ import annotations

from typing import Sequence

from worlds import Domain


def _number(value: float) -> str:
    return f"{float(value):.1f}"


def _trajectory(domain: Domain, series: dict) -> str:
    rows = [f"  {domain.period:>10s} | {domain.driver:>22s} | {domain.outcome}"]
    for index, (driver, outcome) in enumerate(
        zip(series["inputs"], series["observed"]), start=1
    ):
        rows.append(f"  {index:>10d} | {_number(driver):>22s} | {_number(outcome)}")
    return "\n".join(rows)


def render_prompt(domain: Domain, references: Sequence[dict], target: dict,
                  cue: str, k: int, *, training: bool) -> str:
    sections = [
        f"Forecasting task in {domain.label}.",
        (
            f"The resting level of {domain.outcome} is {domain.baseline:.1f}. "
            f"Values are bounded to [{domain.clip[0]:.1f}, {domain.clip[1]:.1f}]."
        ),
        (
            "The reference systems are complete examples. The target system may share a "
            "stable response pattern with one reference. Infer from the information shown; "
            "no response equation or coefficient is provided."
        ),
    ]
    for index, reference in enumerate(references, start=1):
        sections.extend([
            f"REFERENCE SYSTEM {index}",
            f"Background: {reference['context']}",
            _trajectory(domain, reference),
        ])
    sections.append("TARGET SYSTEM")
    if cue != "none":
        sections.append(f"Background: {target['shown_context']}")
    if k:
        sections.append(_trajectory(domain, {
            "inputs": target["inputs"][:k],
            "observed": target["observed"][:k],
        }))
    else:
        sections.append("No target-system observations are available yet.")

    sections.extend([
        "COUNTERFACTUAL FORECASTS",
        (
            f"For each scenario, apply the stated {domain.driver} in the next "
            f"{domain.period} and then set it to 0 in later periods. Forecast "
            f"{domain.outcome} at the stated horizon."
        ),
    ])
    for scenario in domain.scenarios:
        sections.append(
            f"  {scenario['label']}: shock={_number(scenario['shock'])}; "
            f"horizon={scenario['horizon']}"
        )
    sections.extend([
        "Return only strict JSON with exactly ten finite numbers in scenario order:",
        '{"forecasts":[A,B,C,D,E,F,G,H,I,J]}',
    ])
    return "\n\n".join(sections)
