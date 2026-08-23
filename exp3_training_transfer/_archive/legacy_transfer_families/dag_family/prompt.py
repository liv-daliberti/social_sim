"""prompt.py — the unified forecasting prompt for the LG DAG family, and a hardened parser.

The prompt REVEALS the generative structure and hides only the gain g. `describe_structure` (below)
verbalizes each world from its A/B/C matrices -- the graph, the number of latent stages, the delays,
the persistence, and the noise -- so the model is TOLD everything except the one hidden scalar g
(drawn per series), which it must infer from the observed news/poll history. Transfer to a held-out
structure is therefore a test of whether the model can USE a described-but-unseen world-model and pin
down its one free parameter -- not of memorizing a structure it trained on. (Note: the structure-blind
naive-frequentist reference in eval.py sees only the history, so the LLM here is given strictly MORE
information than that baseline.)

Recovery is read behaviourally, exactly as in Experiment 2: query the model at the four held-out
shocks on the probe news channel and take the slope of its forecasts. So the model is only ever asked
for a `predicted_poll`; no "stated gain" is required, and the same prompt works for every structure
(including the ones with no clean one-step gain).
"""
from __future__ import annotations

import json
import re
from typing import List, Optional

import numpy as np


def _news_cols(m: int) -> List[str]:
    return ["news"] if m == 1 else [f"news{i+1}" for i in range(m)]


def _poll_cols(p: int) -> List[str]:
    return ["poll"] if p == 1 else [f"poll{i+1}" for i in range(p)]


def _coef(x: float) -> str:
    return f"{x:+g}"


def describe_structure(struct) -> str:
    """Verbalize the world from A/B/C + the hidden slot: REVEAL the full generative structure
    (dynamics, couplings, delays, noise), hiding ONLY the one gain g (drawn per series). Generalizes
    the Experiment-2 prompt ('opinion moves by g*news + drift; poll = opinion + noise') to any DAG."""
    A, B, C = struct.A, struct.B, struct.C
    d, m, p = struct.d, struct.m, struct.p
    kind, hi, hj = struct.hidden
    ncol = _news_cols(m)

    def hid(kc, i, j):
        return kind == kc and (i, j) == (hi, hj)

    L = [f"Here is exactly how this world works, the same every week. There "
         f"{'is a hidden opinion' if d == 1 else f'are {d} hidden opinions'} you never see directly — "
         f"you only see the news and the poll{'s' if p > 1 else ''}:"]
    for i in range(d):
        oi = "the opinion" if d == 1 else f"opinion {i + 1}"
        cl = []
        if hid("A", i, i):
            cl.append("keeps an UNKNOWN fraction g of last week's value (g is fixed for this series, "
                      "somewhere between 0.1 and 1 — inferring it is your job)")
        elif A[i, i] != 0:
            cl.append(f"keeps about {A[i, i] * 100:.0f}% of last week's value (otherwise easing back toward 50)")
        for j in range(m):
            if B[i, j] != 0:
                nm = "the news" if m == 1 else ncol[j]
                if hid("B", i, j):
                    cl.append(f"moves by an UNKNOWN amount g for every +1 of {nm} (g fixed for this "
                              "series, between 0.1 and 1 — inferring it is your job)")
                else:
                    cl.append(f"moves by {_coef(B[i, j])} for every +1 of {nm}")
        for j in range(d):
            if i != j and A[i, j] != 0:
                oj = "the opinion" if d == 1 else f"opinion {j + 1}"
                if hid("A", i, j):
                    cl.append(f"is pushed by {oj} by an UNKNOWN amount g (fixed, 0.1-1 — infer it)")
                else:
                    cl.append(f"is pushed by {oj} by {_coef(A[i, j])}")
        L.append(f"- {oi[0].upper() + oi[1:]} " + "; ".join(cl) +
                 f"; plus small random drift (about {struct.sigma_s:.0f} point).")
    for k in range(p):
        reads = [i for i in range(d) if C[k, i] != 0]
        pk = "The poll" if p == 1 else f"Poll {k + 1}"
        oi = ("the opinion" if d == 1 else f"opinion {reads[0] + 1}") if len(reads) == 1 \
            else "a blend of opinions " + "+".join(str(i + 1) for i in reads)
        L.append(f"- {pk} reports {oi} plus survey noise (about {struct.sigma_y:.0f} points).")
    return "\n".join(L)


def build_prompt(struct, U: np.ndarray, Y: np.ndarray, next_u: np.ndarray, reason_cap: int = 400) -> str:
    """A forecasting prompt for one structure's history U (k x m) / Y (k x p), asking for the probe
    poll next week given `next_u`. REVEALS the generative structure (like Exp-2), hiding only gain g."""
    m, p = struct.m, struct.p
    ncols, pcols = _news_cols(m), _poll_cols(p)
    target = pcols[struct.probe_output]

    head = [
        "You are forecasting a weekly tracking poll for an election campaign.",
        (f"Each week you observe a signed news value" if m == 1
         else f"Each week you observe {m} signed news values")
        + " (positive = good for the candidate, negative = bad) and the resulting "
        + ("poll" if p == 1 else f"{p} polls") + " (support, 0-100).",
        "",
        describe_structure(struct),
        "",
        "Everything above is known EXCEPT the gain g, which is fixed for this series. Use the history "
        "below to pin down g, then forecast.",
        "",
    ]

    header = "| Week | " + " | ".join(ncols + pcols) + " |"
    sep = "|" + "|".join(["------"] * (1 + m + p)) + "|"
    rows = [header, sep]
    for t in range(len(U)):
        cells = [f"{t+1:>4}"] + [f"{v:>+4.0f}" for v in U[t]] + [f"{v:>4.0f}" for v in Y[t]]
        rows.append("| " + " | ".join(cells) + " |")
    if len(U) == 0:
        rows = ["(no weeks observed yet.)"]

    if m == 1:
        nxt = f"next week's news will be {next_u[0]:+.0f}"
    else:
        nxt = "next week's news will be " + ", ".join(f"{c}={v:+.0f}" for c, v in zip(ncols, next_u))
    tail = [
        "",
        f"Given that {nxt}, predict next week's {target}.",
        "Reason in AT MOST two short sentences. Do NOT write a week-by-week table or long "
        "arithmetic, and do NOT second-guess yourself. Then immediately give the answer.",
        'Output EXACTLY ONE JSON object on its own line and then STOP: '
        '{"rationale": "<=12 words", "predicted_poll": <integer 0-100>}',
    ]
    return "\n".join(head + rows + tail)


SCENARIO_LABELS = ("A", "B", "C", "D")


def build_prompt_multi(struct, U: np.ndarray, Y: np.ndarray, shocks, reason_cap: int = 400,
                       horizon: int = 1) -> str:
    """Multi-shock forecasting prompt: ONE prompt per (episode, prefix) asking for the probed poll
    `horizon` weeks ahead under EACH of the given probe shocks (scenarios A..D), all other news
    channels 0 and NO further news after the pulse. The joint elicitation is what makes the slope of
    the model's answers readable from a single response -- and what lets the reward score the
    RESPONSE (slope) rather than only the level, so a constant 'ignore the news' policy provably
    leaves reward on the table (2026-07-08 audit fix).

    `horizon` is the structure's first-response horizon h* (lg_dag.first_response_horizon). Probing a
    delayed world at h=1 is degenerate: the pulse cannot have reached the poll, every scenario shares
    one target, and the slope carries no information about g."""
    m, p = struct.m, struct.p
    ncols, pcols = _news_cols(m), _poll_cols(p)
    target = pcols[struct.probe_output]
    labels = SCENARIO_LABELS[: len(shocks)]

    head = [
        "You are forecasting a weekly tracking poll for an election campaign.",
        (f"Each week you observe a signed news value" if m == 1
         else f"Each week you observe {m} signed news values")
        + " (positive = good for the candidate, negative = bad) and the resulting "
        + ("poll" if p == 1 else f"{p} polls") + " (support, 0-100).",
        "",
        describe_structure(struct),
        "",
        "Everything above is known EXCEPT the gain g, which is fixed for this series. Use the history "
        "below to pin down g, then forecast.",
        "",
    ]

    header = "| Week | " + " | ".join(ncols + pcols) + " |"
    sep = "|" + "|".join(["------"] * (1 + m + p)) + "|"
    rows = [header, sep]
    for t in range(len(U)):
        cells = [f"{t+1:>4}"] + [f"{v:>+4.0f}" for v in U[t]] + [f"{v:>4.0f}" for v in Y[t]]
        rows.append("| " + " | ".join(cells) + " |")
    if len(U) == 0:
        rows = ["(no weeks observed yet.)"]

    k = len(U)
    if horizon == 1:
        scen = ["", "Consider these scenarios for next week's news:"]
        ask = f"For EACH scenario, predict next week's {target}."
    else:
        scen = ["", f"Consider these scenarios for next week's news (week {k + 1}). "
                    f"Assume there is NO further news after that week (news = 0 in weeks "
                    f"{k + 2}-{k + horizon}):"]
        ask = (f"For EACH scenario, predict the {target} in week {k + horizon}, i.e. {horizon} "
               f"weeks from now -- long enough for that news to reach the poll.")
    for lab, s in zip(labels, shocks):
        u = np.zeros(m); u[struct.probe_input] = float(s)
        if m == 1:
            scen.append(f"- Scenario {lab}: news = {u[0]:+.0f}")
        else:
            scen.append(f"- Scenario {lab}: " + ", ".join(f"{c}={v:+.0f}" for c, v in zip(ncols, u)))
    keys = ", ".join(f'"poll_{lab}": <integer 0-100>' for lab in labels)
    tail = [
        "",
        f"{ask} Use the SAME inferred g for all of them.",
        "Reason in AT MOST two short sentences. Do NOT write a week-by-week table or long "
        "arithmetic, and do NOT second-guess yourself. Then immediately give the answer.",
        'Output EXACTLY ONE JSON object on its own line and then STOP: '
        '{"rationale": "<=12 words", ' + keys + "}",
    ]
    return "\n".join(head + rows + scen + tail)


_JSON_RE = re.compile(r"\{[^{}]*\"predicted_poll\"[^{}]*\}", re.DOTALL)
_JSON_MULTI_RE = re.compile(r"\{[^{}]*\"poll_A\"[^{}]*\}", re.DOTALL)


def parse_forecasts(text: str, n: int = 4) -> Optional[List[float]]:
    """Extract the n scenario forecasts poll_A..poll_D from the model's reply. Accepts only a
    well-formed JSON object containing ALL n keys (last such object wins); returns None otherwise --
    no fallback to stray numbers, and no partial credit for a subset of scenarios (a missing
    scenario would corrupt the slope)."""
    if not text:
        return None
    labels = SCENARIO_LABELS[:n]
    for m in reversed(_JSON_MULTI_RE.findall(text)):
        try:
            obj = json.loads(m)
        except Exception:
            continue
        vals = [obj.get(f"poll_{lab}") for lab in labels]
        if all(isinstance(v, (int, float)) for v in vals):
            return [float(np.clip(v, 0, 100)) for v in vals]
    return None


def parse_forecast(text: str) -> Optional[float]:
    """Extract predicted_poll from the model's reply. Accepts only a well-formed JSON object that
    contains predicted_poll (last one wins); returns None for a missing/malformed object -- no
    fallback to scraping stray numbers from the reasoning (matches the exp2 hardened parser)."""
    if not text:
        return None
    matches = _JSON_RE.findall(text)
    for m in reversed(matches):
        try:
            obj = json.loads(m)
        except Exception:
            continue
        v = obj.get("predicted_poll")
        if isinstance(v, (int, float)):
            return float(np.clip(v, 0, 100))
    return None


if __name__ == "__main__":
    # show an example prompt for a couple of structures
    import sys
    sys.path.insert(0, ".")
    from lg_dag import generate_episode
    from catalog import BY_NAME
    from lg_dag import SHOCKS
    for name in ("direct", "two_news_feedback"):
        st = BY_NAME[name]
        ep = generate_episode(st, seed=7, T=4)
        print("=" * 70, f"\n{name} (multi-shock)\n", "=" * 70, sep="")
        print(build_prompt_multi(st, ep["U"][:3], ep["Y"][:3], SHOCKS))
        print()
