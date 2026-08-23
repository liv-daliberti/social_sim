"""prompt_domain.py -- the domain-skinned forecasting prompt for Experiment 3's transfer arm.

Same contract as dag_family/prompt.py: REVEAL the generative structure and hide only the per-episode
sensitivity g, then ask for the observable under four probe interventions in ONE reply so the slope
of the answers is readable (and therefore rewardable) from a single response.

What changes is that every surface element -- the entity, the period, the driver, the observable,
its units and numeric scale -- comes from a `Domain` rather than being hardcoded to elections. The
underlying arithmetic is identical across domains, so a model that has learned the abstract skill
should transfer; a model that has learned "polls are near 50 and move by about 0.5 per news point"
should not.

The reply key is domain-neutral (`forecast_A..forecast_D`) rather than `poll_A..poll_D`: a fishing
prompt that demanded a "poll" would leak the training domain's vocabulary into the held-out task and
turn a domain-transfer failure into a formatting failure.
"""
from __future__ import annotations

import json
import re
from typing import List, Optional, Sequence

import numpy as np

from domains import Domain

SCENARIO_LABELS = ("A", "B", "C", "D")


def describe_world(dom: Domain) -> str:
    """Verbalize the world in the domain's own vocabulary: the dynamics, the driver's effect, the
    observation noise -- everything except the hidden sensitivity g.

    The structure is the canonical `direct` world for every domain, so this text differs across
    domains ONLY in nouns, units, and the stated numeric ranges. That is the point: the described
    world-model is the same object dressed differently."""
    g0, g1 = dom.g_disp_range
    sigma = f"{dom.sigma_y_disp:.1f}".rstrip("0").rstrip(".")
    return "\n".join([
        f"Here is exactly how this {dom.entity} works, the same every {dom.period}. There is a "
        f"hidden {dom.latent_name} you never observe directly -- you only see the "
        f"{dom.input_name} and the {dom.output_name}:",
        f"- The {dom.latent_name} keeps about 90% of last {dom.period}'s value (otherwise easing "
        f"back toward its normal level).",
        f"- It moves by an UNKNOWN amount g for every +1 of {dom.input_name}. g is fixed for this "
        f"{dom.entity}, somewhere between {g0:.2f} and {g1:.2f} -- inferring it is your job.",
        # phrased without a verb agreeing with the observable, so it reads correctly whether the
        # observable is singular ("poll", "catch") or plural ("points")
        f"- What you observe is the {dom.latent_name} plus measurement noise of about "
        f"{sigma} {dom.output_unit}.",
    ])


def _history_table(dom: Domain, X: np.ndarray, Yd: np.ndarray) -> List[str]:
    """History in DISPLAYED units, one row per observed period."""
    head = f"| {dom.period.capitalize()} | {dom.input_name} | {dom.output_name} |"
    sep = "|--------|--------|--------|"
    rows = [head, sep]
    for t in range(len(X)):
        rows.append(f"| {t + 1:>6} | {dom.fmt_input(float(X[t])):>6} | "
                    f"{dom.fmt_output(float(Yd[t])):>6} |")
    if len(X) == 0:
        rows = [f"(no {dom.period}s observed yet.)"]
    return rows


def build_prompt_domain(dom: Domain, U_can: np.ndarray, Y_can: np.ndarray,
                        shocks_can: Sequence[float]) -> str:
    """One multi-scenario prompt for a history given in CANONICAL units.

    U_can (k,) / Y_can (k,) and `shocks_can` are canonical; everything the model sees is mapped
    through the domain skin here, so the caller never has to think in display units.
    """
    X = dom.show_input(np.asarray(U_can, float).reshape(-1))
    Yd = dom.show_output(np.asarray(Y_can, float).reshape(-1))
    shocks_disp = dom.show_input(np.asarray(shocks_can, float))
    labels = SCENARIO_LABELS[: len(shocks_disp)]

    head = [
        f"You are forecasting {dom.output_desc} for one {dom.entity}, {dom.period} by {dom.period}.",
        f"Each {dom.period} you observe a signed {dom.input_name} value ({dom.input_desc}) and the "
        f"resulting {dom.output_name}"
        # skip the unit clause when the observable IS its own unit ("points, measured in points")
        + ("" if dom.output_unit == dom.output_name else f", measured in {dom.output_unit}")
        + f", normally between {dom.fmt_output(dom.out_lo)} and {dom.fmt_output(dom.out_hi)}.",
        "",
        describe_world(dom),
        "",
        f"Everything above is known EXCEPT g, which is fixed for this {dom.entity}. Use the history "
        "below to pin down g, then forecast.",
        "",
    ]

    scen = ["", f"Consider these scenarios for next {dom.period}'s {dom.input_name}:"]
    for lab, s in zip(labels, shocks_disp):
        scen.append(f"- Scenario {lab}: {dom.input_name} = {dom.fmt_input(float(s))}")

    keys = ", ".join(f'"forecast_{lab}": <number>' for lab in labels)
    tail = [
        "",
        f"For EACH scenario, predict next {dom.period}'s {dom.output_name} in {dom.output_unit}. "
        "Use the SAME inferred g for all of them.",
        "Reason in AT MOST two short sentences. Do NOT write a period-by-period table or long "
        "arithmetic, and do NOT second-guess yourself. Then immediately give the answer.",
        "Output EXACTLY ONE JSON object on its own line and then STOP: "
        '{"rationale": "<=12 words", ' + keys + "}",
    ]
    return "\n".join(head + _history_table(dom, X, Yd) + scen + tail)


_JSON_MULTI_RE = re.compile(r"\{[^{}]*\"forecast_A\"[^{}]*\}", re.DOTALL)


def parse_forecasts(text: str, dom: Optional[Domain] = None,
                    n: int = 4) -> Optional[List[float]]:
    """Extract forecast_A..forecast_D (displayed units). Accepts only a well-formed JSON object
    carrying ALL n keys (last such object wins) -- no partial credit, since a missing scenario would
    corrupt the slope. Clipped to the domain's observable range when a domain is supplied."""
    if not text:
        return None
    labels = SCENARIO_LABELS[:n]
    for m in reversed(_JSON_MULTI_RE.findall(text)):
        try:
            obj = json.loads(m)
        except Exception:
            continue
        vals = [obj.get(f"forecast_{lab}") for lab in labels]
        if all(isinstance(v, (int, float)) for v in vals):
            out = [float(v) for v in vals]
            if dom is not None:
                out = [float(np.clip(v, dom.out_lo, dom.out_hi)) for v in out]
            return out
    return None


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dag_family"))
    from lg_dag import SHOCKS, generate_episode

    from domains import COIN_BASKETBALL, COIN_CITY, COIN_FISHING

    for dom in (COIN_CITY, COIN_FISHING, COIN_BASKETBALL):
        ep = generate_episode(dom.structure(), seed=7, T=4)
        print("=" * 78, f"\n{dom.label}  (g={ep['g']:.3f} canonical -> "
                        f"{dom.show_slope(ep['g']):.3f} displayed)\n", "=" * 78, sep="")
        print(build_prompt_domain(dom, ep["U"][:, 0], ep["Y"][:, 0], SHOCKS))
        print()
