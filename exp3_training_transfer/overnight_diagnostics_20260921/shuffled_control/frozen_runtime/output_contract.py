"""Single strict output contract shared by C3 training and evaluation."""
from __future__ import annotations

import json
import math
from typing import Sequence

OUTPUT_KEY = "forecasts"


def forecast_array_grammar(count: int = 10) -> str:
    """Return a GBNF grammar for the strict forecast-array response contract.

    The grammar constrains syntax only: every item remains an unrestricted finite
    JSON number. In particular, it does not expose targets or restrict forecasts
    to the observable range used by the scorer.
    """
    if count < 1:
        raise ValueError("forecast count must be positive")
    numbers = ' ws "," ws '.join(["number"] * count)
    return (
        'root ::= "{" ws "\\\"forecasts\\\"" ws ":" ws "[" ws '
        f'{numbers} ws "]" ws "}}"\n'
        'number ::= "-"? int frac? exp?\n'
        'int ::= "0" | [1-9] [0-9]*\n'
        'frac ::= "." [0-9]+\n'
        'exp ::= [eE] [+-]? [0-9]+\n'
        'ws ::= [ \\t\\n\\r]*\n'
    )


FORECAST_ARRAY_GBNF = forecast_array_grammar(10)

def _unique_object(pairs):
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def parse_forecast_array(text: str, count: int, clip=(0.0, 100.0)):
    """Parse exactly ``{"forecasts": [x0, ..., xN]}`` or return ``None``.

    The entire response must be the JSON object. Extra keys, prose, missing values,
    booleans, and non-finite numbers are rejected. Values are clipped only after the
    response has passed the format contract.
    """
    if not text:
        return None
    try:
        payload = json.loads(text.strip(), object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or set(payload) != {OUTPUT_KEY}:
        return None
    values = payload[OUTPUT_KEY]
    if not isinstance(values, list) or len(values) != count:
        return None
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in values):
        return None
    lo, hi = (float(value) for value in clip)
    return [max(lo, min(hi, float(value))) for value in values]


def gold_forecast_array(targets: Sequence[float]) -> str:
    """Render a compact canonical response satisfying the strict contract."""
    values = [round(float(value), 3) for value in targets]
    return json.dumps({OUTPUT_KEY: values}, separators=(",", ":"))
