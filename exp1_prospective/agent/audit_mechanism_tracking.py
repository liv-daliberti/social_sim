#!/usr/bin/env python3
"""Does a model's stated reasoning track the packet it read, or only its surface?

Two analyses. The first asks whether rationales identify the specific mechanism a
packet was written to target; it turns out to be uninformative and we report why.
The second asks whether what a model says about relevance predicts what it does to
the forecast, and is the analysis the manuscript uses.

The selectivity result shows models move further on directional packets than on
topic-matched ones, but the two arms are separable from wording alone, so movement
could in principle be routed on surface marking rather than on probative content
(see the construct-validity discussion). This script tests the stronger claim
directly: after reading a snippet, does the model's stated reasoning pick out the
specific causal mechanism the generator wrote that snippet to target?

Design. Each market has nine packets, each with a private `mechanism_targeted`
sentence that never enters the prompt. For one update record we score its
rationale against all nine mechanisms from its own market and ask whether the true
one ranks first. Distractors come from the same market, so the question, actors,
and domain vocabulary are held constant and only the mechanism differs. Chance is
1/9.

Three conditions separate mechanism tracking from text echo:

  rationale        the model's rationale as the query.
  rationale-minus  the rationale with every token that appears in its own snippet
                   removed, so the match must rest on mechanism vocabulary the
                   snippet did not hand the model.
  length-matched   the stripped rationale cut to a fixed 12 tokens sampled at
                   random, over records that have at least that many. Hosted
                   rationales survive stripping with 37-48 tokens and local ones
                   with 13-17, and longer queries retrieve better for purely
                   mechanical reasons, so this condition equalises query length
                   before comparing deployment classes.
  snippet          the snippet itself as the query. The generator wrote snippet
                   and mechanism together, so this measures how much of the
                   retrieval is available from the evidence text alone. It is a
                   reference level, not a baseline to beat.

A within-market permutation null re-scores each rationale against a shuffled
assignment of its market's packets and should land at chance.

Records come from the row-level update directory, since the frozen aggregate
stores scored probabilities rather than rationale text.

ANALYSIS 2, say-do coupling. Orthogonal packets are the ones a model should
declare irrelevant. We flag rationales containing explicit non-relevance language
and compare, within the orthogonal arm, the mean absolute revision of flagged
against unflagged records. Surface routing predicts no particular relation between
the wording of the rationale and the size of the update. A model whose relevance
judgment is doing work should move less precisely when it says the evidence does
not bear on the question. The contrast is bootstrapped over markets.

Outputs:
  data/results/mechanism_tracking.json
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import evaluate_consistency as ec

ROOT = _HERE.parent
CF_FILE = ROOT / "data" / "counterfactuals" / "counterfactuals_2026-06-10.jsonl"
OUT_JSON = ROOT / "data" / "results" / "mechanism_tracking.json"

MODEL_ORDER = (
    "claude-opus-4-8", "gpt-5.4", "DeepSeek-V4-Pro",
    "qwen2.5:72b", "llama3.3:70b", "llama3.1:70b",
    "qwen2.5:32b", "qwen2.5:14b", "llama3.1:8b", "qwen2.5:7b",
)
LABELS = {
    "claude-opus-4-8": "Claude Opus~4.8", "gpt-5.4": "GPT-5.4",
    "DeepSeek-V4-Pro": "DeepSeek V4-Pro", "qwen2.5:72b": "Qwen~2.5-72B",
    "llama3.3:70b": "Llama~3.3-70B", "llama3.1:70b": "Llama~3.1-70B",
    "qwen2.5:32b": "Qwen~2.5-32B", "qwen2.5:14b": "Qwen~2.5-14B",
    "llama3.1:8b": "Llama~3.1-8B", "qwen2.5:7b": "Qwen~2.5-7B",
}
SEED = 20260617
N_BOOT = 2000

STOP = set("""the a an and or but if then than that this these those of in on at to for with
from by as is are was were be been being it its his her their our your they we you he she
not no nor so such into over under about after before during while which who whom whose what
when where how why all any both each few more most other some only own same very can will just
should now also may might must would could has have had do does did done up down out off""".split())
TOKEN_RE = re.compile(r"[a-z][a-z\-]{2,}")


def tokens(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall((text or "").lower()) if t not in STOP]


NONRELEVANCE = re.compile(
    r"\b(unrelated|irrelevant|no signal|no bearing|no direct bearing|"
    r"does not (affect|change|alter|bear|shift|move)|not material|immaterial|"
    r"no material (effect|impact|change)|no impact|no effect|"
    r"does not provide (any )?information|leaves? .{0,20}unchanged|"
    r"remain(s)? unchanged)\b", re.I)


def say_do_coupling(by_cf, markets_of_cf):
    """Within the orthogonal arm, does declared irrelevance predict smaller moves?"""
    rows = defaultdict(lambda: defaultdict(list))   # model -> arm -> (flag, |d|, market)
    for (tid, model), recs in ec._load_all_updated_forecasts().items():
        for rec in recs:
            packet = by_cf.get(rec.get("cf_id"))
            delta = rec.get("delta_yes_prob")
            if packet is None or delta is None:
                continue
            sf = rec.get("updated_structured_forecast") or {}
            rationale = sf.get("rationale")
            if not isinstance(rationale, str) or not rationale.strip():
                continue
            arm = "orthogonal" if packet["direction"] == "orthogonal" else "directional"
            rows[model][arm].append((bool(NONRELEVANCE.search(rationale)),
                                     abs(delta) * 100, packet["task_id"]))

    out = {}
    rng = np.random.RandomState(SEED)
    for model, arms in rows.items():
        rec = {}
        for arm, vals in arms.items():
            flags = [f for f, _, _ in vals]
            rec[f"{arm}_flag_rate"] = round(float(np.mean(flags)), 3)
            rec[f"{arm}_n"] = len(vals)
        orth = arms.get("orthogonal", [])
        f = [d for fl, d, _ in orth if fl]
        u = [d for fl, d, _ in orth if not fl]
        if f and u:
            rec["orthogonal_mean_abs_delta_flagged"] = round(float(np.mean(f)), 3)
            rec["orthogonal_mean_abs_delta_unflagged"] = round(float(np.mean(u)), 3)
            rec["coupling_gap_pp"] = round(float(np.mean(u) - np.mean(f)), 3)
            idx = defaultdict(list)
            for i, (_, _, mk) in enumerate(orth):
                idx[mk].append(i)
            keys = list(idx)
            draws = []
            for _ in range(N_BOOT):
                pick = rng.randint(0, len(keys), len(keys))
                sel = [i for k in pick for i in idx[keys[k]]]
                ff = [orth[i][1] for i in sel if orth[i][0]]
                uu = [orth[i][1] for i in sel if not orth[i][0]]
                if ff and uu:
                    draws.append(np.mean(uu) - np.mean(ff))
            lo, hi = np.percentile(draws, [2.5, 97.5])
            rec["coupling_gap_ci95"] = [round(float(lo), 3), round(float(hi), 3)]
            rec["coupling_gap_excludes_zero"] = bool(lo > 0)
        out[model] = rec
    return out


def main() -> None:
    packets = [json.loads(l) for l in CF_FILE.read_text().splitlines() if l.strip()]
    by_market: dict[str, list] = defaultdict(list)
    for p in packets:
        by_market[p["task_id"]].append(p)
    by_cf = {p["cf_id"]: p for p in packets}

    # IDF over the mechanism corpus, so shared domain words are downweighted.
    mech_tokens = {p["cf_id"]: tokens(p.get("mechanism_targeted", "")) for p in packets}
    df = Counter()
    for toks in mech_tokens.values():
        df.update(set(toks))
    n_docs = len(mech_tokens)
    idf = {t: math.log((1 + n_docs) / (1 + c)) + 1.0 for t, c in df.items()}

    def vec(toks: list[str]) -> dict[str, float]:
        tf = Counter(toks)
        v = {t: (1 + math.log(c)) * idf.get(t, math.log(1 + n_docs) + 1.0)
             for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    mech_vec = {cf: vec(toks) for cf, toks in mech_tokens.items()}

    def cosine(a: dict, b: dict) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(x * b.get(t, 0.0) for t, x in a.items())

    MATCH_LEN = 12
    CONDITIONS = ("rationale", "rationale_minus_snippet",
                  "rationale_minus_snippet_length_matched", "snippet")
    hits = {c: defaultdict(list) for c in CONDITIONS}       # model -> [0/1]
    ranks = {c: defaultdict(list) for c in CONDITIONS}
    by_arm = {c: defaultdict(lambda: defaultdict(list)) for c in CONDITIONS}
    market_of: dict[str, list] = defaultdict(list)          # model -> [task_id]
    qlen: dict = {c: defaultdict(list) for c in
                  ("rationale", "rationale_minus_snippet",
                   "rationale_minus_snippet_length_matched")}
    null_hits: dict[str, list] = defaultdict(list)
    skipped = Counter()
    rng = np.random.RandomState(SEED)

    for (tid, model), recs in ec._load_all_updated_forecasts().items():
        for rec in recs:
            packet = by_cf.get(rec.get("cf_id"))
            if packet is None:
                continue
            cands = by_market.get(packet["task_id"], [])
            if len(cands) < 2:
                skipped["market_without_distractors"] += 1
                continue
            sf = rec.get("updated_structured_forecast") or {}
            rationale = sf.get("rationale")
            if not isinstance(rationale, str) or not rationale.strip():
                skipped["no_rationale"] += 1
                continue

            snippet_toks = set(tokens(packet.get("evidence_text", "")))
            rat_toks = tokens(rationale)
            stripped = [t for t in rat_toks if t not in snippet_toks]
            if len(stripped) >= MATCH_LEN:
                pick = rng.choice(len(stripped), MATCH_LEN, replace=False)
                matched = [stripped[i] for i in sorted(pick)]
            else:
                matched = []          # too short to length-match; excluded below
            queries = {
                "rationale": rat_toks,
                "rationale_minus_snippet": stripped,
                "rationale_minus_snippet_length_matched": matched,
                "snippet": tokens(packet.get("evidence_text", "")),
            }
            true_cf = packet["cf_id"]
            for cond, qt in queries.items():
                if not qt:
                    continue
                if cond in qlen:
                    qlen[cond][model].append(len(qt))
                qv = vec(qt)
                scored = sorted(
                    ((cosine(qv, mech_vec[c["cf_id"]]), c["cf_id"]) for c in cands),
                    key=lambda x: -x[0])
                order = [cf for _, cf in scored]
                rank = order.index(true_cf) + 1
                hit = 1 if rank == 1 else 0
                hits[cond][model].append(hit)
                ranks[cond][model].append(1.0 / rank)
                by_arm[cond][model][packet["direction"]].append(hit)
                if cond == "rationale":
                    market_of[model].append(packet["task_id"])
                    # permutation null: score against a randomly chosen packet
                    fake = order[rng.randint(0, len(order))]
                    null_hits[model].append(1 if fake == true_cf else 0)

    def clustered_ci(values, markets):
        """Bootstrap over markets, retaining every record in a sampled market."""
        idx = defaultdict(list)
        for i, m in enumerate(markets):
            idx[m].append(i)
        keys = list(idx)
        arr = np.array(values, dtype=float)
        draws = []
        r = np.random.RandomState(SEED)
        for _ in range(N_BOOT):
            pick = r.randint(0, len(keys), len(keys))
            sel = np.concatenate([idx[keys[k]] for k in pick])
            draws.append(arr[sel].mean())
        lo, hi = np.percentile(draws, [2.5, 97.5])
        return round(float(lo), 3), round(float(hi), 3)

    per_model = {}
    for model in MODEL_ORDER:
        if not hits["rationale"][model]:
            continue
        row = {"label": LABELS[model], "n_records": len(hits["rationale"][model])}
        for cond in CONDITIONS:
            h = hits[cond][model]
            if not h:
                continue
            row[cond] = {
                "top1": round(float(np.mean(h)), 3),
                "mrr": round(float(np.mean(ranks[cond][model])), 3),
                "n": len(h),
            }
        for cond in ("rationale", "rationale_minus_snippet",
                     "rationale_minus_snippet_length_matched"):
            if cond not in row:
                continue
            m = market_of[model]
            row[cond]["ci95_market_clustered"] = clustered_ci(
                hits[cond][model], m[:len(hits[cond][model])])
            if qlen.get(cond, {}).get(model):
                row[cond]["mean_query_tokens"] = round(
                    float(np.mean(qlen[cond][model])), 1)
            row[cond]["excludes_chance"] = bool(
                row[cond]["ci95_market_clustered"][0] > 1 / 9)
        row["permutation_null_top1"] = round(float(np.mean(null_hits[model])), 3)
        row["rationale_top1_by_arm"] = {
            arm: round(float(np.mean(v)), 3)
            for arm, v in by_arm["rationale"][model].items() if v}
        per_model[model] = row

    pooled = {}
    for cond in CONDITIONS:
        allh = [x for m in MODEL_ORDER for x in hits[cond][m]]
        if allh:
            pooled[cond] = {"top1": round(float(np.mean(allh)), 3), "n": len(allh)}

    # Ceiling: the generator's own rationale for why the snippet targets the
    # mechanism, scored under the identical stripped and length-matched rule.
    # It is written by the same process that wrote the mechanism, so it bounds
    # what this retrieval metric can award to a perfectly mechanism-aware writer.
    ceil_hits, ceil_hits_lm = [], []
    for pk in packets:
        cands = by_market.get(pk["task_id"], [])
        if len(cands) < 2:
            continue
        stoks = set(tokens(pk.get("evidence_text", "")))
        gt = [t for t in tokens(pk.get("rationale", "")) if t not in stoks]
        if not gt:
            continue
        for toks_, bucket in ((gt, ceil_hits),
                              (gt if len(gt) < MATCH_LEN else
                               [gt[i] for i in sorted(rng.choice(len(gt), MATCH_LEN,
                                                                 replace=False))],
                               ceil_hits_lm)):
            if len(bucket) is not None and (bucket is ceil_hits or len(gt) >= MATCH_LEN):
                qv = vec(toks_)
                best = max(((cosine(qv, mech_vec[c["cf_id"]]), c["cf_id"]) for c in cands),
                           key=lambda x: x[0])[1]
                bucket.append(1 if best == pk["cf_id"] else 0)

    report = {
        "analysis": "exp1_mechanism_tracking",
        "generator_rationale_ceiling": {
            "minus_snippet_top1": round(float(np.mean(ceil_hits)), 3),
            "minus_snippet_length_matched_top1": round(float(np.mean(ceil_hits_lm)), 3),
            "n": len(ceil_hits),
            "note": "generator's own rationale as query, same stripping rule",
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_source": "row-level update directory (rationale text is not in the frozen aggregate)",
        "task": "rank the true mechanism among all packets of the same market",
        "chance_top1": round(1 / 9, 3),
        "conditions": {
            "rationale": "model rationale as query",
            "rationale_minus_snippet": "rationale with its own snippet's tokens removed",
            "rationale_minus_snippet_length_matched":
                f"stripped rationale cut to {MATCH_LEN} randomly sampled tokens",
            "snippet": "snippet as query; reference level for generator-induced overlap",
        },
        "skipped": dict(skipped),
        "pooled": pooled,
        "per_model": per_model,
        "say_do_coupling": {
            "excluded_from_reporting": {
                "qwen2.5:7b": "archived updates mix 0-1 and 0-100 probability scales"},
            "flag_lexicon": NONRELEVANCE.pattern,
            "per_model": say_do_coupling(by_cf, None),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")

    print(f"chance top-1 = {1/9:.3f}   (rank true mechanism among 9 same-market candidates)\n")
    print(f"{'model':18s} {'n':>6s} {'ration.':>8s} {'95% CI':>16s} {'-snippet':>9s} "
          f"{'snippet':>8s} {'null':>6s}")
    for model in MODEL_ORDER:
        r = per_model.get(model)
        if not r:
            continue
        ci = r["rationale"]["ci95_market_clustered"]
        print(f"{r['label']:18s} {r['n_records']:6d} {r['rationale']['top1']:8.3f} "
              f"  [{ci[0]:.3f},{ci[1]:.3f}] "
              f"{r['rationale_minus_snippet']['top1']:9.3f} "
              f"{r['snippet']['top1']:8.3f} {r['permutation_null_top1']:6.3f}")
    print(f"\npooled: " + "  ".join(f"{k}={v['top1']:.3f} (n={v['n']})" for k, v in pooled.items()))
    c = report["generator_rationale_ceiling"]
    print(f"generator-rationale ceiling under the same stripping rule: "
          f"{c['minus_snippet_top1']:.3f} (length-matched {c['minus_snippet_length_matched_top1']:.3f}) "
          f"-- at chance, so the stripped retrieval comparison is uninformative")
    print(f"\n{'model':18s} {'flag% orth':>10s} {'flag% dir':>9s} {'|d| flag':>9s} "
          f"{'|d| unflag':>11s} {'gap':>7s} {'95% CI':>16s}")
    for model in MODEL_ORDER:
        r = report["say_do_coupling"]["per_model"].get(model)
        if not r or "coupling_gap_pp" not in r:
            continue
        if model == "qwen2.5:7b":
            # Mixed 0-1 / 0-100 stored updates make its magnitudes incomparable.
            continue
        ci = r["coupling_gap_ci95"]
        print(f"{LABELS[model]:18s} {100*r['orthogonal_flag_rate']:10.1f} "
              f"{100*r['directional_flag_rate']:9.1f} "
              f"{r['orthogonal_mean_abs_delta_flagged']:9.2f} "
              f"{r['orthogonal_mean_abs_delta_unflagged']:11.2f} "
              f"{r['coupling_gap_pp']:7.2f}   [{ci[0]:.2f},{ci[1]:.2f}]")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
