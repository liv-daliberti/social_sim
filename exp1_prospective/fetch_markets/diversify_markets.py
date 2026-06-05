#!/usr/bin/env python3
"""Pick N maximally-diverse markets from a selected_markets JSONL.

Uses greedy Jaccard similarity on question text (unigrams + bigrams) to
avoid near-duplicate questions.  Also caps representation per broad topic
cluster so no single theme dominates.

Usage (from exp1_prospective/):
    python fetch_markets/diversify_markets.py
    python fetch_markets/diversify_markets.py --n 100
    python fetch_markets/diversify_markets.py --input data/selected_markets/selected_2026-06-05.jsonl
    python fetch_markets/diversify_markets.py --sim-threshold 0.25 --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_SEL_DIR = _ROOT / "data" / "selected_markets"

# ── stopwords (minimal; we WANT topic words to drive similarity) ───────────────
_STOP = {
    "a", "an", "the", "and", "or", "of", "in", "to", "by", "for",
    "on", "at", "is", "be", "will", "would", "that", "this", "with",
    "from", "into", "it", "its", "as", "if", "not", "any", "before",
    "after", "by", "within", "between", "have", "has", "had",
}


def _tokens(text: str) -> set[str]:
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1]
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
    return set(words) | set(bigrams)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


# ── broad topic buckets for per-bucket caps ────────────────────────────────────
# Each entry: (bucket_label, keyword_list).  First match wins.
_BUCKETS: list[tuple[str, list[str]]] = [
    ("ukraine_map",     ["serhiivka", "chasiv yar", "pokrovsk", "toretsk", "kupyansk",
                         "isw map", "donetsk oblast", "zaporizhzhia oblast", "kherson"]),
    ("iran_nuclear",    ["iran", "enrichment", "enriched uranium", "nuclear deal",
                         "hormuz", "blockade", "strait"]),
    ("us_2026_midterms",["midterm", "house after", "senate after", "2026 balance",
                         "2026 midterm", "november 2026"]),
    ("us_elections_state", ["governor", "california", "florida", "texas", "new york state",
                            "illinois", "pennsylvania", "ohio", "georgia election",
                            "michigan election", "arizona election"]),
    ("us_federal_policy", ["trump", "congress", "senate", "supreme court", "white house",
                           "legislation", "executive order", "tariff", "trade deal"]),
    ("brazil",          ["brazil", "lula", "bolsonaro", "renan santos", "brazili"]),
    ("russia_ukraine",  ["russia", "ukraine", "zelenskyy", "zelensky", "putin",
                         "ceasefire", "peace deal", "nato"]),
    ("ai_tech",         ["openai", "anthropic", "chatgpt", "deepseek", "gemini",
                         "gpt", "llm", "ai model", "claude", "mistral", "nvidia",
                         "spacex", "elon musk", "tesla"]),
    ("global_elections",["election", "presidential", "parliamentary", "prime minister",
                         "chancellor", "afg", "poland", "germany", "france", "uk election",
                         "mexico", "colombia", "argentina", "south korea", "japan",
                         "india election", "israel", "turkey", "canada election"]),
    ("economics_macro", ["gdp", "recession", "inflation", "interest rate", "federal reserve",
                         "imf", "cpi", "unemployment", "debt ceiling", "budget",
                         "s&p", "nasdaq", "dow", "stock market"]),
    ("china_taiwan",    ["china", "taiwan", "xi jinping", "prc", "ccp"]),
    ("other",           []),  # catch-all
]


def _bucket(r: dict) -> str:
    text = (r.get("question", "") + " " + r.get("event_title", "")).lower()
    for label, kws in _BUCKETS:
        for kw in kws:
            if kw in text:
                return label
    return "other"


# ── greedy diversity selection ─────────────────────────────────────────────────

def select_diverse(
    records: list[dict],
    *,
    n: int,
    sim_threshold: float,
    bucket_cap: int,
) -> list[dict]:
    """Greedy selection: add a market only if it's < sim_threshold similar to all
    already-selected markets, and its bucket hasn't hit bucket_cap yet."""

    # Precompute tokens and bucket for every record
    tok  = {r["market_id"]: _tokens(r.get("question", "") + " " + r.get("event_title", ""))
            for r in records}
    buck = {r["market_id"]: _bucket(r) for r in records}

    # Sort candidates: highest volume first so we always prefer well-traded markets
    sorted_recs = sorted(records, key=lambda r: r.get("volume_usd") or 0.0, reverse=True)

    selected: list[dict] = []
    selected_tokens: list[set[str]] = []
    bucket_counts: Counter = Counter()

    for r in sorted_recs:
        if len(selected) >= n:
            break
        mid = r["market_id"]
        b   = buck[mid]
        t   = tok[mid]

        if bucket_counts[b] >= bucket_cap:
            continue

        # Check similarity against every already-selected market
        too_similar = any(jaccard(t, st) >= sim_threshold for st in selected_tokens)
        if too_similar:
            continue

        selected.append(r)
        selected_tokens.append(t)
        bucket_counts[b] += 1

    return selected


# ── cli ────────────────────────────────────────────────────────────────────────

def _latest_selected(sel_dir: Path) -> Path | None:
    for f in sorted(sel_dir.glob("selected_*.jsonl"), reverse=True):
        if "diverse" not in f.name:
            return f
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pick N diverse markets from a selected_markets JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--input",         type=Path,  default=None)
    ap.add_argument("--out",           type=Path,  default=_SEL_DIR)
    ap.add_argument("--n",             type=int,   default=100,
                    help="Target number of markets to keep.")
    ap.add_argument("--sim-threshold", type=float, default=0.20,
                    help="Jaccard similarity above which two markets are considered duplicates.")
    ap.add_argument("--bucket-cap",    type=int,   default=12,
                    help="Max markets allowed per broad topic bucket.")
    ap.add_argument("--dry-run",       action="store_true")
    args = ap.parse_args()

    input_path = args.input or _latest_selected(_SEL_DIR)
    if input_path is None or not input_path.exists():
        print(f"No selected JSONL found in {_SEL_DIR}.")
        sys.exit(1)

    with input_path.open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"Input: {len(records):,} markets from {input_path.name}")

    chosen = select_diverse(
        records,
        n=args.n,
        sim_threshold=args.sim_threshold,
        bucket_cap=args.bucket_cap,
    )

    # ── report ────────────────────────────────────────────────────────────────
    buckets = Counter(_bucket(r) for r in chosen)
    print(f"\nSelected {len(chosen)} diverse markets  "
          f"(sim_threshold={args.sim_threshold}, bucket_cap={args.bucket_cap})\n")

    print("Bucket breakdown:")
    for b, cnt in buckets.most_common():
        bar = "▪" * cnt
        print(f"  {b:<30s} {cnt:>3}  {bar}")

    vols = sorted(r.get("volume_usd") or 0.0 for r in chosen)
    n = len(vols)
    print(f"\nVolume (USD):  min ${vols[0]:,.0f}  "
          f"median ${vols[n//2]:,.0f}  "
          f"p90 ${vols[int(n*0.9)]:,.0f}  "
          f"max ${vols[-1]:,.0f}")

    prices = [r["yes_price"] for r in chosen if r.get("yes_price") is not None]
    print(f"Yes-price:     min {min(prices):.2f}  "
          f"mean {sum(prices)/len(prices):.2f}  "
          f"max {max(prices):.2f}")

    dtds = [r["days_to_resolution"] for r in chosen if r.get("days_to_resolution") is not None]
    print(f"Days-to-res:   min {min(dtds):.0f}  "
          f"median {sorted(dtds)[len(dtds)//2]:.0f}  "
          f"max {max(dtds):.0f}")

    print(f"\nFull list:")
    for i, r in enumerate(chosen, 1):
        vol = r.get("volume_usd") or 0.0
        yp  = r.get("yes_price") or 0.0
        dtd = r.get("days_to_resolution") or 0.0
        b   = _bucket(r)
        print(f"  {i:>3}. [{yp:.2f}] ${vol:>10,.0f}  {dtd:>4.0f}d  [{b}]  {r['question'][:65]}")

    if args.dry_run:
        print(f"\n--dry-run: skipping write")
        return

    fetch_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path   = args.out / f"diverse_{fetch_date}.jsonl"
    mani_path  = args.out / f"diverse_{fetch_date}.manifest.json"

    with out_path.open("w") as f:
        for r in chosen:
            f.write(json.dumps(r) + "\n")

    manifest = {
        "created_at":     datetime.now(timezone.utc).isoformat(),
        "source_file":    str(input_path),
        "n_input":        len(records),
        "n_selected":     len(chosen),
        "sim_threshold":  args.sim_threshold,
        "bucket_cap":     args.bucket_cap,
        "bucket_counts":  dict(buckets),
    }
    with mani_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {len(chosen)} records → {out_path}")
    print(f"Wrote manifest        → {mani_path}")


if __name__ == "__main__":
    main()
