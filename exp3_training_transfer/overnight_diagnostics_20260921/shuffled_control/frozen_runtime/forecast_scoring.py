"""Small dependency-free scoring helpers shared by training and unit tests."""

from __future__ import annotations

import math
import re
from typing import Any


YES_PROB_RE = re.compile(
    r'"?(?:yes_prob|probability)"?\s*:\s*"?(-?\d+(?:\.\d+)?)',
    re.IGNORECASE,
)


def parse_yes_probability(text: str) -> float | None:
    match = None
    for match in YES_PROB_RE.finditer(text):
        pass
    if match is None:
        return None
    probability = float(match.group(1))
    return probability if 0.0 <= probability <= 1.0 else None


def score_binary_forecast(
    response: str,
    settlement_yes: bool,
    market_yes_prob: float,
) -> tuple[float, dict[str, Any]]:
    probability = parse_yes_probability(response)
    if probability is None:
        return 0.0, {"formatted": False}
    label = float(bool(settlement_yes))
    brier = (probability - label) ** 2
    clipped = max(1e-6, min(1.0 - 1e-6, probability))
    log_loss = -(
        label * math.log(clipped)
        + (1.0 - label) * math.log(1.0 - clipped)
    )
    return 1.0 - brier, {
        "formatted": True,
        "yes_prob": probability,
        "brier": brier,
        "log_loss": log_loss,
        "market_yes_prob": float(market_yes_prob),
    }
