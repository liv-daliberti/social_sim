#!/usr/bin/env python3
"""Run the Experiment 2 v2 two-reference-cities → third-city probe.

The default ``--backend none`` is an offline oracle simulation.  With an Ollama
or Azure backend, one model call is made at every target prefix k=0..K.  A single
call predicts four counterfactual next polls, which lets us recover a behavioural
gain from their slope without conflating gain with the forecast's level.

Examples (from ``exp2_v2/biased_news``):

    python eval/run_three_city.py --dry-run
    python eval/run_three_city.py --backend none --n 500
    python eval/run_three_city.py --backend ollama --model qwen2.5:7b --n 100
    python eval/run_three_city.py --backend azure --model gpt-5.4 --n 100
    python eval/run_three_city.py --backend none --weak-gain 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

from engine.three_city import (
    PHI,
    PROBE_SHOCKS,
    REFERENCE_WEEKS,
    SIGMA_OPINION,
    SIGMA_SURVEY,
    STRONG_GAIN,
    TARGET_WEEKS,
    WEAK_GAIN,
    compute_clairvoyant_ceiling,
    compute_frequentist_baseline,
    compute_hierarchical_bayes,
    compute_naive_baseline,
    compute_pooled_baseline,
    compute_target_only_bayes,
    counterfactual_gold,
    generate_triplet,
    position_score,
    type_score,
)

_OUTDIR = _ROOT / "data" / "three_city"
_AZURE_ENDPOINT = "https://liv-forecast.services.ai.azure.com/openai/v1"
MAX_TRIES = 3


def _city_table(city: Mapping[str, Any], weeks: Optional[int] = None) -> list[str]:
    if weeks is None:
        weeks = len(city["news"])
    rows = [
        "| Week | Net news | Poll |",
        "|------|----------|------|",
        f"| {0:>4} | {'baseline':>8} | {city['initial_poll']:>4} |",
    ]
    rows.extend(
        f"| {i:>4} | {dose:>+8d} | {poll:>4} |"
        for i, (dose, poll) in enumerate(
            zip(city["news"][:weeks], city["polls"][:weeks]),
            1,
        )
    )
    return rows


def build_prompt(
    triplet: Mapping[str, Any],
    k: int,
    *,
    shocks: Sequence[int] = PROBE_SHOCKS,
    no_think: bool = False,
) -> str:
    """Prompt with two complete examples and the target's first ``k`` weeks."""
    lines = [
        "You are forecasting weekly tracking polls in three cities.",
        "",
        "Across this region there are exactly TWO stable city response types. "
        "Reference City A exemplifies one type and Reference City B exemplifies "
        "the other. Target City C is equally likely (50/50) to share A's exact "
        "type or B's exact type.",
        "",
        "All cities start at a known poll of 50. In every later week:",
        "  - 90% of the prior deviation from 50 carries forward;",
        "  - signed net news then moves opinion by a city-specific fixed number "
        "of poll points per news point;",
        "  - small random opinion drift (about 1 point) and poll noise (about 2 "
        "points) are present.",
        "Positive news helps and negative news hurts. The two cities' response "
        "rates are not stated: infer them from the reference histories.",
        "",
        "REFERENCE CITY A (complete example)",
        *_city_table(triplet["reference_a"]),
        "",
        "REFERENCE CITY B (complete example)",
        *_city_table(triplet["reference_b"]),
        "",
        f"TARGET CITY C (observed through Week {k})",
        *_city_table(triplet["target"], weeks=k),
    ]
    if k == 0:
        lines += [
            "",
            "No post-baseline poll has yet been observed for City C. Use the "
            "50/50 prior over the two demonstrated city types.",
        ]

    shock_list = ", ".join(f"{shock:+d}" for shock in shocks)
    prediction_schema = ", ".join(
        f'{{"news": {shock}, "poll": <0-100>}}' for shock in shocks
    )
    lines += [
        "",
        f"For each counterfactual next-week net-news value [{shock_list}], predict "
        "City C's next poll. These are separate alternatives from the same current "
        "history, not a sequence of future weeks.",
        "",
        "Also report (i) the probability from 0 to 1 that C matches A rather than "
        "B, and (ii) C's estimated poll points per +1 news point.",
        "",
        "Respond with one JSON object only:",
        "{",
        '  "rationale": "one short sentence",',
        '  "prob_matches_a": <number 0-1>,',
        '  "points_per_news": <number 0-1>,',
        f'  "predictions": [{prediction_schema}]',
        "}",
    ]
    if no_think:
        lines.append("/no_think")
    return "\n".join(lines)


def _json_objects(text: str):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def parse_reply(
    raw: str,
    *,
    shocks: Sequence[int] = PROBE_SHOCKS,
) -> Dict[str, Any]:
    """Parse the last usable structured answer in a model response."""
    wanted = {int(shock) for shock in shocks}
    parsed: Dict[str, Any] = {
        "prob_matches_a": None,
        "points_per_news": None,
        "predictions": {},
    }
    for obj in _json_objects(raw):
        predictions: Dict[str, float] = {}
        for item in obj.get("predictions", []):
            if not isinstance(item, dict):
                continue
            try:
                shock = int(round(float(item["news"])))
                poll = max(0.0, min(100.0, float(item["poll"])))
            except (KeyError, TypeError, ValueError):
                continue
            if shock in wanted:
                predictions[str(shock)] = poll
        if len(predictions) < 2:
            continue

        probability = obj.get("prob_matches_a")
        try:
            probability = float(probability)
            if 1.0 < probability <= 100.0:
                probability /= 100.0
            probability = max(0.0, min(1.0, probability))
        except (TypeError, ValueError):
            probability = None

        gain = obj.get("points_per_news")
        try:
            gain = float(gain)
        except (TypeError, ValueError):
            gain = None

        parsed = {
            "prob_matches_a": probability,
            "points_per_news": gain,
            "predictions": predictions,
        }
    return parsed


def _slope(predictions: Mapping[str, float]) -> Optional[float]:
    pairs = [(float(shock), float(poll)) for shock, poll in predictions.items()]
    if len(pairs) < 2:
        return None
    mean_x = sum(x for x, _ in pairs) / len(pairs)
    mean_y = sum(y for _, y in pairs) / len(pairs)
    denominator = sum((x - mean_x) ** 2 for x, _ in pairs)
    if denominator == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in pairs) / denominator


def _forecast_mae(
    predictions: Mapping[str, float],
    gold: Mapping[str, float],
) -> Optional[float]:
    errors = [
        abs(float(prediction) - float(gold[shock]))
        for shock, prediction in predictions.items()
        if shock in gold
    ]
    return sum(errors) / len(errors) if errors else None


def _slug(value: str) -> str:
    return value.replace(":", "-").replace("/", "-").replace(" ", "-")


def _load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open() as handle:
        for line in handle:
            try:
                done.add(json.loads(line)["episode_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def _make_model_client(args):
    # Reuse the original Exp 2's battle-tested Ollama/Azure plumbing only when a
    # model run is requested.  Offline oracle simulation has no API dependency.
    from run_batch import make_client

    if args.backend == "azure" and "claude" in args.model.lower():
        from anthropic import AnthropicFoundry

        return AnthropicFoundry(
            azure_ad_token_provider=lambda: args.api_key,
            base_url=args.endpoint,
        )
    return make_client(
        args.backend,
        endpoint=args.endpoint,
        api_key=args.api_key,
        azure_auth=args.azure_auth,
        timeout=args.timeout,
    )


def _call_model(client, args, prompt: str) -> Tuple[Dict[str, Any], str]:
    from run_batch import _strip_reasoning
    from run_recovery_probe import chat_call

    last_raw = ""
    for _ in range(MAX_TRIES):
        try:
            response = chat_call(
                client,
                args.model,
                [{"role": "user", "content": prompt}],
                args.max_tokens,
                args.temperature,
            )
            last_raw = _strip_reasoning(
                (response.choices[0].message.content or "").strip()
            )
            parsed = parse_reply(last_raw)
            if len(parsed["predictions"]) >= 2:
                return parsed, last_raw
        except Exception as exc:  # record the final API error, then retry
            last_raw = f"ERROR: {exc}"
        if args.delay:
            time.sleep(args.delay)
    return parse_reply(last_raw), last_raw


def _curve_entry(
    triplet: Mapping[str, Any],
    k: int,
    *,
    parsed: Optional[Mapping[str, Any]] = None,
    raw: str = "",
) -> Dict[str, Any]:
    oracle = compute_hierarchical_bayes(triplet, k)
    clairvoyant = compute_clairvoyant_ceiling(triplet, k)
    frequentist = compute_frequentist_baseline(triplet, k)
    naive = compute_naive_baseline(triplet, k)
    target_only = compute_target_only_bayes(triplet, k)
    pooled = compute_pooled_baseline(triplet, k)
    gold = counterfactual_gold(triplet, k)

    p_correct = type_score(triplet, oracle["p_match_a"])
    oracle_position = position_score(triplet, oracle["gain_hat"])
    entry: Dict[str, Any] = {
        "k": k,
        "target_news": triplet["target"]["news"][:k],
        "target_polls": triplet["target"]["polls"][:k],
        "gold_forecasts": gold,
        "oracle": {
            **oracle,
            **p_correct,
            **oracle_position,
            "forecast_mae": round(
                _forecast_mae(oracle["forecasts"], gold) or 0.0,
                6,
            ),
        },
        "clairvoyant": {
            **clairvoyant,
            **type_score(triplet, clairvoyant["p_match_a"]),
            **position_score(triplet, clairvoyant["gain_hat"]),
            "forecast_mae": round(
                _forecast_mae(clairvoyant["forecasts"], gold) or 0.0,
                6,
            ),
        },
        "frequentist": {
            **frequentist,
            **(
                position_score(triplet, frequentist["gain_hat"])
                if frequentist["gain_hat"] is not None
                else {
                    "gain_error": None,
                    "normalised_gain_error": None,
                    "adaptation": None,
                }
            ),
            "forecast_mae": (
                round(_forecast_mae(frequentist["forecasts"], gold), 6)
                if frequentist["forecasts"]
                else None
            ),
        },
        "naive": {
            **naive,
            **position_score(triplet, naive["gain_hat"]),
            "forecast_mae": round(
                _forecast_mae(naive["forecasts"], gold) or 0.0,
                6,
            ),
        },
        "target_only": {
            **target_only,
            **position_score(triplet, target_only["gain_hat"]),
            "forecast_mae": round(
                _forecast_mae(target_only["forecasts"], gold) or 0.0,
                6,
            ),
        },
        "pooled": {
            **pooled,
            **position_score(triplet, pooled["gain_hat"]),
            "forecast_mae": round(
                _forecast_mae(pooled["forecasts"], gold) or 0.0,
                6,
            ),
        },
        "lm": None,
    }

    if parsed is not None:
        predictions = dict(parsed["predictions"])
        behavioural_gain = _slope(predictions)
        lm: Dict[str, Any] = {
            "prob_matches_a": parsed.get("prob_matches_a"),
            "points_per_news": parsed.get("points_per_news"),
            "behavioural_gain": (
                round(behavioural_gain, 6)
                if behavioural_gain is not None
                else None
            ),
            "predictions": predictions,
            "forecast_mae": (
                round(_forecast_mae(predictions, gold), 6)
                if predictions
                else None
            ),
            "raw": raw[:2000],
        }
        if parsed.get("prob_matches_a") is not None:
            lm.update(type_score(triplet, float(parsed["prob_matches_a"])))
        if behavioural_gain is not None:
            lm.update(position_score(triplet, behavioural_gain))
        entry["lm"] = lm
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--backend",
        choices=["none", "ollama", "azure"],
        default="none",
        help="none runs the matched oracle only",
    )
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--azure-auth", action="store_true")
    parser.add_argument("--n", type=int, default=250, help="city triplets")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--reference-weeks", type=int, default=REFERENCE_WEEKS)
    parser.add_argument("--target-weeks", type=int, default=TARGET_WEEKS)
    parser.add_argument("--strong-gain", type=float, default=STRONG_GAIN)
    parser.add_argument("--weak-gain", type=float, default=WEAK_GAIN)
    parser.add_argument(
        "--target-evidence",
        choices=["diagnostic", "ambiguous", "random"],
        default="diagnostic",
    )
    parser.add_argument("--sigma-opinion", type=float, default=SIGMA_OPINION)
    parser.add_argument("--sigma-survey", type=float, default=SIGMA_SURVEY)
    parser.add_argument("--phi", type=float, default=PHI)
    parser.add_argument("--out", type=Path, default=_OUTDIR)
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--delay", type=float, default=0.02)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    model_lower = args.model.lower()
    if args.endpoint is None:
        if args.backend == "azure" and "claude" in model_lower:
            args.endpoint = "https://liv-forecast.services.ai.azure.com/anthropic"
        elif args.backend == "azure" and "deepseek" in model_lower:
            # The older cos-tiktok deployment became unhealthy during the
            # original Exp 2 runs. DeepSeek-V4-Pro is also deployed on the
            # healthy shared Foundry resource used by GPT.
            args.endpoint = _AZURE_ENDPOINT
        elif args.backend == "azure":
            args.endpoint = _AZURE_ENDPOINT
        elif args.backend == "ollama":
            args.endpoint = os.environ.get(
                "OLLAMA_ENDPOINT",
                "http://localhost:11434/v1",
            )
    if args.backend == "azure":
        if not args.api_key and "deepseek" in model_lower:
            args.api_key = (
                os.environ.get("AZURE_AI_API_KEY", "")
                or os.environ.get("DEEPSEEK_AZURE_API_KEY", "")
            )
        elif not args.api_key and "claude" in model_lower:
            args.api_key = (
                os.environ.get("AZURE_AI_API_KEY", "")
                or os.environ.get("CLAUDE_AZURE_API_KEY", "")
            )
        elif not args.api_key:
            args.api_key = os.environ.get("AZURE_AI_API_KEY", "")
        if not args.api_key and not args.dry_run:
            parser.error("azure requires --api-key or an Azure API key environment variable")

    no_think = "qwen" in args.model.lower()
    first = generate_triplet(
        args.seed_offset,
        reference_weeks=args.reference_weeks,
        target_weeks=args.target_weeks,
        strong_gain=args.strong_gain,
        weak_gain=args.weak_gain,
        target_evidence=args.target_evidence,
        sigma_opinion=args.sigma_opinion,
        sigma_survey=args.sigma_survey,
        phi=args.phi,
    )
    if args.dry_run:
        print(
            f"Hidden check: strong reference={first['strong_reference']}; "
            f"target matches={first['target_matches']}; "
            f"target g={first['target']['gain']}\n"
        )
        print(build_prompt(first, min(1, args.target_weeks), no_think=no_think))
        return

    model_slug = "oracle" if args.backend == "none" else _slug(args.model)
    gain_slug = str(args.weak_gain).replace(".", "p")
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    args.out.mkdir(parents=True, exist_ok=True)
    stem = (
        f"results_{model_slug}_{args.target_evidence}_"
        f"weak{gain_slug}_{date}"
    )
    out_path = args.out / f"{stem}.jsonl"
    manifest_path = args.out / f"{stem}.manifest.json"
    done = _load_done(out_path)
    client = _make_model_client(args) if args.backend != "none" else None

    n_ok = 0
    n_error = 0
    with out_path.open("a") as handle:
        for index in range(args.n):
            episode_id = f"triplet_{args.seed_offset + index:06d}"
            if episode_id in done:
                continue
            triplet = generate_triplet(
                args.seed_offset + index,
                reference_weeks=args.reference_weeks,
                target_weeks=args.target_weeks,
                strong_gain=args.strong_gain,
                weak_gain=args.weak_gain,
                target_evidence=args.target_evidence,
                sigma_opinion=args.sigma_opinion,
                sigma_survey=args.sigma_survey,
                phi=args.phi,
            )
            curve = []
            episode_failed = False
            for k in range(args.target_weeks + 1):
                if client is None:
                    curve.append(_curve_entry(triplet, k))
                    continue
                prompt = build_prompt(triplet, k, no_think=no_think)
                parsed, raw = _call_model(client, args, prompt)
                if len(parsed["predictions"]) < 2:
                    episode_failed = True
                curve.append(
                    _curve_entry(
                        triplet,
                        k,
                        parsed=parsed,
                        raw=raw,
                    )
                )
                if args.delay:
                    time.sleep(args.delay)

            public_triplet = {
                "reference_a": {
                    "initial_poll": triplet["reference_a"]["initial_poll"],
                    "news": triplet["reference_a"]["news"],
                    "polls": triplet["reference_a"]["polls"],
                },
                "reference_b": {
                    "initial_poll": triplet["reference_b"]["initial_poll"],
                    "news": triplet["reference_b"]["news"],
                    "polls": triplet["reference_b"]["polls"],
                },
                "target": {
                    "initial_poll": triplet["target"]["initial_poll"],
                    "news": triplet["target"]["news"],
                    "polls": triplet["target"]["polls"],
                },
            }
            record = {
                "episode_id": episode_id,
                "model": model_slug,
                "public_triplet": public_triplet,
                "hidden": {
                    "reference_a_gain": triplet["reference_a"]["gain"],
                    "reference_b_gain": triplet["reference_b"]["gain"],
                    "target_gain": triplet["target"]["gain"],
                    "target_matches": triplet["target_matches"],
                    "strong_reference": triplet["strong_reference"],
                },
                "curve": curve,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            if episode_failed:
                n_error += 1
            else:
                n_ok += 1
            if args.verbose or (index + 1) % 25 == 0:
                print(
                    f"{index + 1}/{args.n} triplets; "
                    f"ok={n_ok}, parse_errors={n_error}"
                )

    manifest = {
        "world": "three_city_latent_type_v1",
        "model": model_slug,
        "backend": args.backend,
        "n": args.n,
        "n_ok_this_run": n_ok,
        "n_error_this_run": n_error,
        "reference_weeks": args.reference_weeks,
        "target_weeks": args.target_weeks,
        "strong_gain": args.strong_gain,
        "weak_gain": args.weak_gain,
        "target_evidence": args.target_evidence,
        "sigma_opinion": args.sigma_opinion,
        "sigma_survey": args.sigma_survey,
        "phi": args.phi,
        "probe_shocks": list(PROBE_SHOCKS),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Done: {out_path}")


if __name__ == "__main__":
    main()
