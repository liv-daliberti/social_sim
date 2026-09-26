#!/usr/bin/env python3
"""Build a small development-only mechanism-reversal candidate packet."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parents[1]
DEFAULT_MARKETS = REPO / "exp1_prospective/data/selected_markets/diverse_2026-08-12.jsonl"
CORE_PACKETS = REPO / "exp1_prospective/data/counterfactuals/counterfactuals_2026-06-10.jsonl"
DEFAULT_OUTPUT = ROOT / "data/development/pilot_candidates.jsonl"
PROTOCOL_VERSION = "indirect-pilot-v1"

FORBIDDEN_EVIDENCE = re.compile(
    r"\b(?:yes|no|more likely|less likely|odds|probability|forecast|on-track|"
    r"setback|boost|hurt|win|lose|approve|reject)\b",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


FAMILIES = [
    {
        "market_id": "908713",
        "candidate_id": "pilot_fed_investment_reaction",
        "initial_event_model": {
            "key_actors": ["Federal Open Market Committee", "Federal Reserve staff", "Bureau of Economic Analysis"],
            "key_mechanisms": ["the committee's reaction function", "staff inflation projections", "the voting coalition"],
            "latent_variables": ["demand relative to productive capacity", "inflation persistence"],
        },
        "evidence": {
            "headline": "Government revision shows faster business equipment spending in second quarter",
            "text": "A revised government report says U.S. businesses spent 6 percent more on equipment, up from the earlier estimate of 3 percent. The change came from newly processed company filings.",
        },
        "bridge_prefix": "For this scenario, Fed officials interpret faster equipment spending as ",
        "positive_clause": "a sign that demand is heating up and could push inflation higher",
        "negative_clause": "a sign that factories are expanding capacity and could ease inflation",
        "bridge_suffix": ". Their stated policy is to raise rates when inflation pressure rises and leave rates unchanged when it eases.",
        "broken_bridge": "For this scenario, the Fed ignores revisions to equipment-spending data until the following year. This report therefore does not enter any 2026 rate decision.",
        "chains": {
            "positive": (["spending revision", "stronger demand", "inflation pressure", "2026 rate increase"], ["+", "+", "+"]),
            "negative": (["spending revision", "more factory capacity", "inflation pressure", "2026 rate increase"], ["+", "-", "+"]),
            "broken": (["spending revision", "excluded 2026 input", "2026 rate decision"], ["0", "0"]),
        },
    },
    {
        "market_id": "562802",
        "candidate_id": "pilot_house_registration_cohort",
        "initial_event_model": {
            "key_actors": ["Democratic House campaigns", "Republican House campaigns", "state election offices", "first-time voters"],
            "key_mechanisms": ["registration-to-turnout conversion", "district vote margins", "seat aggregation"],
            "latent_variables": ["partisan composition of new registrants", "turnout among first-time voters"],
        },
        "evidence": {
            "headline": "Election offices add 180,000 voters across twelve close House districts",
            "text": "State election offices completed identity and residency checks for the new registrations before publishing the total. The report does not identify the voters' party preferences.",
        },
        "bridge_prefix": "For this scenario, records show that the newly registered voters usually support ",
        "positive_clause": "Democratic House candidates",
        "negative_clause": "Republican House candidates",
        "bridge_suffix": ". They vote at high rates in close districts, and those districts determine which party controls the House.",
        "broken_bridge": "For this scenario, the registrations were completed after the 2026 voting deadline. These voters therefore cannot affect district results or control of the House.",
        "chains": {
            "positive": (["new registrations", "Democratic voters", "close-district ballots", "Democratic House control"], ["+", "+", "+"]),
            "negative": (["new registrations", "Republican voters", "close-district ballots", "Democratic House control"], ["+", "+", "-"]),
            "broken": (["new registrations", "ineligible voters", "House control"], ["0", "0"]),
        },
    },
    {
        "market_id": "700396",
        "candidate_id": "pilot_perplexity_board_rule",
        "initial_event_model": {
            "key_actors": ["Perplexity's board", "a special transaction committee", "potential acquirers", "competition regulators"],
            "key_mechanisms": ["board authorization", "proposal negotiation", "transaction signing and closing"],
            "latent_variables": ["proposal quality", "board willingness", "regulatory feasibility"],
        },
        "evidence": {
            "headline": "Perplexity schedules shareholder meeting after receiving updated ownership list",
            "text": "The company's transfer agent delivered a certified list of current shareholders. Perplexity scheduled a special meeting for the following month but did not publish the meeting agenda.",
        },
        "bridge_prefix": "For this scenario, the updated ownership list ",
        "positive_clause": "gives one buyer enough shareholder support to pass a sale vote",
        "negative_clause": "shows that every potential buyer lacks enough support to pass a sale vote",
        "bridge_suffix": ". The board follows that vote when deciding whether to sign an acquisition agreement this year.",
        "broken_bridge": "For this scenario, the special meeting concerns an employee stock plan, and acquisition votes are excluded. The ownership list therefore cannot affect any sale decision this year.",
        "chains": {
            "positive": (["ownership list", "buyer voting support", "sale vote", "signed acquisition"], ["+", "+", "+"]),
            "negative": (["ownership list", "buyer voting support", "sale vote", "signed acquisition"], ["-", "+", "+"]),
            "broken": (["ownership list", "excluded acquisition vote", "signed acquisition"], ["0", "0"]),
        },
    },
]


def context(bridge_text: str, label: str, chain: tuple[list[str], list[str]]) -> dict:
    return {
        "bridge_text": bridge_text,
        "intended_label": label,
        "chain_nodes": chain[0],
        "chain_edges": chain[1],
    }


def validate(candidate: dict, core_ids: set[str]) -> None:
    assert candidate["development_only"] is True
    assert candidate["market"]["task_id"] not in core_ids
    evidence = candidate["evidence"]["headline"] + " " + candidate["evidence"]["text"]
    hit = FORBIDDEN_EVIDENCE.search(evidence)
    assert hit is None, f"forbidden evidence token {hit.group(0)!r}"
    assert 20 <= len(evidence.split()) <= 45
    labels = {
        "positive": "increase_yes",
        "negative": "decrease_yes",
        "broken": "no_material_effect",
    }
    for role, expected in labels.items():
        item = candidate["contexts"][role]
        assert item["intended_label"] == expected
        assert len(item["chain_edges"]) >= 2
        assert len(item["chain_nodes"]) == len(item["chain_edges"]) + 1
    # The directional variants must share a frame and differ in the decisive clause.
    positive = candidate["contexts"]["positive"]["bridge_text"]
    negative = candidate["contexts"]["negative"]["bridge_text"]
    assert positive != negative
    assert candidate["bridge_frame"]["prefix"] in positive and candidate["bridge_frame"]["prefix"] in negative
    assert positive.endswith(candidate["bridge_frame"]["suffix"])
    assert negative.endswith(candidate["bridge_frame"]["suffix"])
    assert 20 <= len(positive.split()) <= 40
    assert 20 <= len(negative.split()) <= 40
    assert 15 <= len(candidate["contexts"]["broken"]["bridge_text"].split()) <= 28


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", type=Path, default=DEFAULT_MARKETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    markets = {str(row["market_id"]): row for row in read_jsonl(args.markets)}
    core_ids = {row["task_id"] for row in read_jsonl(CORE_PACKETS)}
    candidates = []
    for spec in FAMILIES:
        market = markets.get(spec["market_id"])
        if market is None:
            raise SystemExit(f"market {spec['market_id']} missing from {args.markets}")
        prefix = spec["bridge_prefix"]
        suffix = spec["bridge_suffix"]
        candidate = {
            "protocol_version": PROTOCOL_VERSION,
            "candidate_id": spec["candidate_id"],
            "development_only": True,
            "split": "development",
            "family_index": len(candidates),
            "market": {
                "task_id": market["task_id"],
                "market_id": str(market["market_id"]),
                "question": market["question"],
                "resolution_criteria": market.get("description") or market.get("rules") or "",
                "category": market.get("category", ""),
                "end_time": market.get("end_time"),
            },
            "initial_event_model": spec["initial_event_model"],
            "evidence": {**spec["evidence"], "surface_valence": ("positive", "negative", "neutral")[len(candidates)]},
            "bridge_frame": {"prefix": prefix, "suffix": suffix},
            "contexts": {
                "positive": context(prefix + spec["positive_clause"] + suffix, "increase_yes", spec["chains"]["positive"]),
                "negative": context(prefix + spec["negative_clause"] + suffix, "decrease_yes", spec["chains"]["negative"]),
                "broken": context(spec["broken_bridge"], "no_material_effect", spec["chains"]["broken"]),
            },
            "generation": {
                "generator_id": "codex-gpt5-protocol-draft",
                "prompt_sha256": "0" * 64,
                "status": "draft_for_human_validation",
                "notes": "Development-only draft; no target-model behavior inspected.",
            },
        }
        validate(candidate, core_ids)
        candidates.append(candidate)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in candidates))
    print(f"wrote {len(candidates)} development candidates to {args.output}")


if __name__ == "__main__":
    main()
