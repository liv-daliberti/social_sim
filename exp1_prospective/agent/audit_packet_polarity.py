#!/usr/bin/env python3
"""Prompt-content audit for Experiment 1: what the Stage 3 update prompt withholds.

Two read-only checks over frozen artifacts.

  1. WITHHOLDING  The update template substitutes exactly one packet field,
     `evidence_text`.  For every stored Stage 3 prompt we re-extract the NEW
     EVIDENCE block and verify that it equals that packet's `evidence_text`
     verbatim, and that no private packet field --- intended direction, target
     hypothesis, expected shift, slot type, headline, targeted mechanism,
     generator rationale, plausibility note --- appears anywhere in the block.
     Those fields exist only in the packet record and are read only at scoring
     time.  Matching is confined to the evidence block because the surrounding
     prompt legitimately contains the agent's own prior forecast, which names
     H1 and H2.

  2. POLARITY  How often the snippet the agent does see states its own bearing
     on the market.  `outcome` cues name the market, its hypotheses, or its
     resolution; `likelihood` cues are any explicit probability/odds/likelihood
     language, whether or not it refers to the market outcome.

Outputs:
  data/results/packet_polarity_audit.json
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CF_FILE = ROOT / "data" / "counterfactuals" / "counterfactuals_2026-06-10.jsonl"
UPDATE_DIR = ROOT / "data" / "updated_forecasts"
OUT_JSON = ROOT / "data" / "results" / "packet_polarity_audit.json"

EVIDENCE_BLOCK_RX = re.compile(r"NEW EVIDENCE[^\n]*\n\n(.*?)\n---", re.S)

# Packet-record fields that must never reach the agent.
PRIVATE_FIELDS = (
    "direction",
    "target_hypothesis",
    "slot_type",
    "evidence_headline",
    "mechanism_targeted",
    "rationale",
    "plausibility_note",
)

# `expected_hypothesis_shift` is excluded from the substring test: its values are
# the ordinary English words "increase"/"decrease", which occur in wire copy by
# coincidence and carry no packet-identifying information.  They are counted
# separately so the exclusion is visible rather than silent.
GENERIC_LABEL_FIELD = "expected_hypothesis_shift"

# Names the market, its hypotheses, or its resolution.
OUTCOME_RX = re.compile(
    r"\b(H1|H2|resolve[sd]?\s+(yes|no)|resolution criteria|prediction market|"
    r"this market|the market will resolve)\b",
    re.I,
)

# Any explicit likelihood language, about anything.
LIKELIHOOD_RX = re.compile(
    r"\b(more likely|less likely|likelihood|probabilit\w*|odds|chances? of)\b",
    re.I,
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _audit_prompts(by_cf: dict) -> dict:
    verified = 0
    block_mismatch = 0
    no_stored_prompt: Counter = Counter()
    unparseable_prompt: Counter = Counter()
    leaks: dict[str, int] = defaultdict(int)
    generic_word_coincidences = 0
    verified_by_file: Counter = Counter()

    files = sorted(UPDATE_DIR.glob("updated_*.jsonl"))
    for path in files:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            pkt = by_cf.get(rec.get("cf_id"))
            if pkt is None:
                continue
            prompt = (rec.get("generation") or {}).get("prompt", "") or ""
            if not prompt:
                no_stored_prompt[path.name] += 1
                continue
            m = EVIDENCE_BLOCK_RX.search(prompt)
            if not m:
                unparseable_prompt[path.name] += 1
                continue
            block = _norm(m.group(1))
            if block != _norm(pkt.get("evidence_text", "")):
                block_mismatch += 1
                continue
            verified += 1
            verified_by_file[path.name] += 1
            low = block.lower()
            for field in PRIVATE_FIELDS:
                val = _norm(str(pkt.get(field) or "")).lower()
                if not val:
                    continue
                if re.search(rf"(?<!\w){re.escape(val)}(?!\w)", low):
                    leaks[field] += 1
            generic = _norm(str(pkt.get(GENERIC_LABEL_FIELD) or "")).lower()
            if generic and re.search(rf"(?<!\w){re.escape(generic)}(?!\w)", low):
                generic_word_coincidences += 1

    return {
        "update_files": len(files),
        "prompts_verified_identical_to_packet_text": verified,
        "prompts_with_block_mismatch": block_mismatch,
        "private_field_occurrences_in_evidence_block": {
            f: leaks.get(f, 0) for f in PRIVATE_FIELDS
        },
        "total_private_field_occurrences": sum(leaks.values()),
        "generic_shift_word_coincidences": generic_word_coincidences,
        "prompts_verified_by_file": dict(sorted(verified_by_file.items())),
        "records_with_no_stored_prompt": dict(no_stored_prompt),
        "records_with_unparseable_stored_prompt": dict(unparseable_prompt),
        "note": (
            "Prompt persistence differs by runner: the local Ollama runner and "
            "several hosted batches did not store the prompt string, and one "
            "GPT-5.4 top-up file stored a truncated trace. Those records carry "
            "the numeric update fields the frozen evaluation uses, so coverage "
            "of this check is a subset of the record set by logging choice, not "
            "by outcome. The template itself substitutes only `evidence_text`."
        ),
    }


def main() -> None:
    packets = [json.loads(l) for l in CF_FILE.read_text().splitlines() if l.strip()]
    by_cf = {p["cf_id"]: p for p in packets}
    n = len(packets)

    outcome_hits, likelihood_hits = [], []
    cue_counts: Counter = Counter()
    for p in packets:
        t = p.get("evidence_text") or ""
        if OUTCOME_RX.search(t):
            outcome_hits.append(p)
        if LIKELIHOOD_RX.search(t):
            likelihood_hits.append(p)
            for mm in LIKELIHOOD_RX.finditer(t):
                cue_counts[mm.group(0).lower()] += 1

    report = {
        "analysis": "exp1_packet_polarity_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "counterfactual_file": CF_FILE.name,
        "packets": n,
        "packets_by_direction": dict(Counter(p["direction"] for p in packets)),
        "prompt_withholding": _audit_prompts(by_cf),
        "snippet_polarity": {
            "names_market_or_resolution_n": len(outcome_hits),
            "names_market_or_resolution_rate": len(outcome_hits) / n,
            "explicit_likelihood_n": len(likelihood_hits),
            "explicit_likelihood_rate": len(likelihood_hits) / n,
            "explicit_likelihood_by_direction": dict(
                Counter(p["direction"] for p in likelihood_hits)
            ),
            "explicit_likelihood_cue_counts": dict(cue_counts.most_common()),
            "no_explicit_likelihood_rate": 1 - len(likelihood_hits) / n,
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("prompt_withholding", "snippet_polarity")}, indent=2))
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
