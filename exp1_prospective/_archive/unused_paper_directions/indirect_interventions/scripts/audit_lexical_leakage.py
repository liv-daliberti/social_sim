#!/usr/bin/env python3
"""Fail-closed structural and lexical audit for indirect candidates."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_CANDIDATES = ROOT / "data/candidates/core_candidates.jsonl"
DEFAULT_CONTROLS = ROOT / "data/candidates/core_controls.jsonl"
DEFAULT_REPORT = ROOT / "reports/core_candidate_audit.json"
DEVELOPMENT_CANDIDATES = ROOT / "data/development/development_candidates_readable_v3.jsonl"
DEVELOPMENT_REPORT = ROOT / "reports/development_candidate_audit_readable_v3.json"
SCHEMA_PATH = ROOT / "schema/candidate.schema.json"
FORBIDDEN = re.compile(
    r"\b(?:yes|no|more likely|less likely|odds|probability|forecast|on-track|"
    r"setback|boost|hurt|win|lose|approve|reject)\b",
    re.I,
)
EXPECTED = {
    "positive": ("increase_yes", 1),
    "negative": ("decrease_yes", -1),
    "broken": ("no_material_effect", 0),
}
EVIDENCE_WORDS = (20, 45)
DIRECTIONAL_BRIDGE_WORDS = (20, 40)
BROKEN_BRIDGE_WORDS = (15, 28)
DECISIVE_CLAUSE_WORDS = (3, 12)
BRIDGE_ACRONYM = re.compile(r"\b[A-Z]{2,}\b")
MAX_BRIDGE_SENTENCES = 2
MAX_BRIDGE_SENTENCE_WORDS = 24


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def edge_product(edges: list[str]) -> int:
    value = 1
    for edge in edges:
        if edge == "0":
            return 0
        if edge == "-":
            value *= -1
        elif edge != "+":
            raise ValueError(edge)
    return value


def audit(
    candidates: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    *,
    complete: bool,
    development: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    by_market: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ids = [row.get("candidate_id") for row in candidates]
    if len(ids) != len(set(ids)):
        errors.append("candidate IDs are not unique")

    try:
        import jsonschema
    except ImportError:
        errors.append("jsonschema is unavailable; refusing to skip schema validation")
    else:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        for row in candidates:
            try:
                jsonschema.validate(row, schema)
            except jsonschema.ValidationError as exc:
                errors.append(
                    f"{row.get('candidate_id', '<missing>')}: schema violation: "
                    f"{exc.message}"
                )

    for row in candidates:
        cid = row.get("candidate_id", "<missing>")
        market = row.get("market", {})
        task_id = market.get("task_id", "")
        by_market[task_id].append(row)
        expected_split = "development" if development else "core"
        if row.get("development_only") is not development or row.get("split") != expected_split:
            errors.append(f"{cid}: record has wrong split/development flag")
        generation = row.get("generation", {})
        if not development and generation.get("status") != "human_edited":
            errors.append(f"{cid}: one-generator design requires recorded human editing")
        if not development and (
            not generation.get("editor_id") or not generation.get("edited_at")
        ):
            errors.append(f"{cid}: human-editor provenance is incomplete")
        evidence = row.get("evidence", {})
        joined = f"{evidence.get('headline', '')} {evidence.get('text', '')}".strip()
        words = len(joined.split())
        if not EVIDENCE_WORDS[0] <= words <= EVIDENCE_WORDS[1]:
            message = (
                f"{cid}: evidence has {words} words, expected "
                f"{EVIDENCE_WORDS[0]}--{EVIDENCE_WORDS[1]}"
            )
            (errors if generation.get("status") == "human_edited" else warnings).append(message)
        hit = FORBIDDEN.search(joined)
        if hit:
            errors.append(f"{cid}: forbidden evidence cue {hit.group(0)!r}")
        frame = row.get("bridge_frame", {})
        prefix, suffix = frame.get("prefix", ""), frame.get("suffix", "")
        contexts = row.get("contexts", {})
        if set(contexts) != set(EXPECTED):
            errors.append(f"{cid}: contexts are not exactly positive/negative/broken")
            continue
        for role, (label, product) in EXPECTED.items():
            context = contexts[role]
            if context.get("intended_label") != label:
                errors.append(f"{cid}/{role}: wrong intended label")
            edges, nodes = context.get("chain_edges", []), context.get("chain_nodes", [])
            if len(edges) < 2 or len(nodes) != len(edges) + 1:
                errors.append(f"{cid}/{role}: chain does not have >=2 edges and N+1 nodes")
            else:
                try:
                    observed = edge_product(edges)
                except ValueError as exc:
                    errors.append(f"{cid}/{role}: invalid edge {exc}")
                else:
                    if observed != product:
                        errors.append(f"{cid}/{role}: signed-edge product {observed}, expected {product}")
        for role in ("positive", "negative"):
            context = contexts[role]
            clause = context.get("decisive_clause", "")
            exact = prefix + clause + suffix
            if context.get("bridge_text") != exact:
                errors.append(f"{cid}/{role}: bridge differs outside the registered decisive clause")
            bridge_words = len(context.get("bridge_text", "").split())
            clause_words = len(clause.split())
            destination = errors if generation.get("status") == "human_edited" else warnings
            if not DIRECTIONAL_BRIDGE_WORDS[0] <= bridge_words <= DIRECTIONAL_BRIDGE_WORDS[1]:
                destination.append(
                    f"{cid}/{role}: bridge has {bridge_words} words, expected "
                    f"{DIRECTIONAL_BRIDGE_WORDS[0]}--{DIRECTIONAL_BRIDGE_WORDS[1]}"
                )
            if not DECISIVE_CLAUSE_WORDS[0] <= clause_words <= DECISIVE_CLAUSE_WORDS[1]:
                destination.append(
                    f"{cid}/{role}: decisive clause has {clause_words} words, expected "
                    f"{DECISIVE_CLAUSE_WORDS[0]}--{DECISIVE_CLAUSE_WORDS[1]}"
                )
        broken_words = len(contexts["broken"].get("bridge_text", "").split())
        destination = errors if generation.get("status") == "human_edited" else warnings
        if not BROKEN_BRIDGE_WORDS[0] <= broken_words <= BROKEN_BRIDGE_WORDS[1]:
            destination.append(
                f"{cid}/broken: bridge has {broken_words} words, expected "
                f"{BROKEN_BRIDGE_WORDS[0]}--{BROKEN_BRIDGE_WORDS[1]}"
            )
        for role, context in contexts.items():
            bridge_text = context.get("bridge_text", "")
            acronyms = sorted(set(BRIDGE_ACRONYM.findall(bridge_text)))
            acronyms = [value for value in acronyms if value not in {"YES", "NO"}]
            if acronyms:
                destination = (
                    errors if generation.get("status") == "human_edited" else warnings
                )
                destination.append(
                    f"{cid}/{role}: replace all-caps abbreviation(s) with plain "
                    f"language: {', '.join(acronyms)}"
                )
            sentences = [
                sentence.strip() for sentence in re.split(r"[.!?]+", bridge_text)
                if sentence.strip()
            ]
            destination = errors if generation.get("status") == "human_edited" else warnings
            if len(sentences) > MAX_BRIDGE_SENTENCES:
                destination.append(
                    f"{cid}/{role}: bridge has {len(sentences)} sentences, expected "
                    f"at most {MAX_BRIDGE_SENTENCES}"
                )
            if any(
                len(sentence.split()) > MAX_BRIDGE_SENTENCE_WORDS
                for sentence in sentences
            ):
                destination.append(
                    f"{cid}/{role}: bridge sentence exceeds "
                    f"{MAX_BRIDGE_SENTENCE_WORDS} words"
                )
            if ";" in bridge_text:
                destination.append(f"{cid}/{role}: replace semicolon with short sentences")
        p_clause = contexts["positive"].get("decisive_clause", "")
        n_clause = contexts["negative"].get("decisive_clause", "")
        if p_clause == n_clause:
            errors.append(f"{cid}: positive/negative decisive clauses are identical")
        elif difflib.SequenceMatcher(None, p_clause.lower(), n_clause.lower()).ratio() < 0.25:
            warnings.append(f"{cid}: reversal clauses have low lexical overlap; human minimality review needed")

    for task_id, rows in by_market.items():
        indices = sorted(row.get("family_index") for row in rows)
        if indices != [0, 1, 2]:
            errors.append(f"{task_id}: family indices are {indices}, expected [0, 1, 2]")
        valences = Counter(row.get("evidence", {}).get("surface_valence") for row in rows)
        if valences != Counter({"positive": 1, "negative": 1, "neutral": 1}):
            errors.append(f"{task_id}: surface-valence schedule is unbalanced: {dict(valences)}")
        headlines = [row.get("evidence", {}).get("headline") for row in rows]
        if len(headlines) != len(set(headlines)):
            errors.append(f"{task_id}: family evidence is duplicated")
        prompt_hashes = {
            row.get("generation", {}).get("prompt_sha256") for row in rows
        }
        if len(prompt_hashes) != 1:
            errors.append(f"{task_id}: families do not share one market prompt hash")

    control_by_market: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not development:
        for row in controls:
            control_by_market[row.get("market", {}).get("task_id", "")].append(row)
        for task_id, rows in control_by_market.items():
            conditions = Counter(row.get("condition") for row in rows)
            expected = Counter({
                "direct_pro": 1,
                "direct_anti": 1,
                "orthogonal": 1,
                "literal_null": 1,
            })
            if conditions != expected:
                errors.append(f"{task_id}: control battery is malformed: {dict(conditions)}")

    candidate_market_ids = set(by_market)
    control_market_ids = set(control_by_market)
    if not development and candidate_market_ids != control_market_ids:
        errors.append("candidate and control market sets differ")
    if complete:
        expected_candidates = 60 if development else 300
        expected_markets = 20 if development else 100
        if len(candidates) != expected_candidates or len(candidate_market_ids) != expected_markets:
            errors.append(
                f"pool is incomplete: {len(candidates)} candidates across "
                f"{len(candidate_market_ids)} markets; expected "
                f"{expected_candidates} across {expected_markets}"
            )
        if not development and len(controls) != 400:
            errors.append(f"control battery is incomplete: {len(controls)} rows")

    edited_count = sum(
        row.get("generation", {}).get("status") == "human_edited"
        and bool(row.get("generation", {}).get("editor_id"))
        and bool(row.get("generation", {}).get("edited_at"))
        for row in candidates
    )
    market_prompt_hash_count = len({
        row.get("generation", {}).get("prompt_sha256") for row in candidates
    })
    generator_counts = Counter(
        row.get("generation", {}).get("generator_id") for row in candidates
    )
    generation_status_counts = Counter(
        row.get("generation", {}).get("status") for row in candidates
    )

    return {
        "protocol_version": (
            "indirect-development-v1" if development else "indirect-core-v1"
        ),
        "pass": not errors,
        "ready_for_human_editing": development and not errors,
        "ready_for_ratings": development and not errors and edited_count == len(candidates),
        "complete_required": complete,
        "candidate_count": len(candidates),
        "market_count": len(candidate_market_ids),
        "control_count": len(controls),
        "human_edited_count": edited_count,
        "market_prompt_hash_count": market_prompt_hash_count,
        "generator_counts": dict(generator_counts),
        "generation_status_counts": dict(generation_status_counts),
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "forbidden_evidence_cues": True,
            "evidence_word_bounds": list(EVIDENCE_WORDS),
            "directional_bridge_word_bounds": list(DIRECTIONAL_BRIDGE_WORDS),
            "broken_bridge_word_bounds": list(BROKEN_BRIDGE_WORDS),
            "decisive_clause_word_bounds": list(DECISIVE_CLAUSE_WORDS),
            "no_all_caps_bridge_abbreviations": True,
            "maximum_bridge_sentences": MAX_BRIDGE_SENTENCES,
            "maximum_bridge_sentence_words": MAX_BRIDGE_SENTENCE_WORDS,
            "no_bridge_semicolons": True,
            "bridge_only_minimality": True,
            "signed_chain_products": True,
            "latin_square_surface_valence": True,
            "three_distinct_families_per_market": True,
            "four_controls_per_market": None if development else True,
            "json_schema": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--controls", type=Path, default=DEFAULT_CONTROLS)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    if args.development:
        if args.candidates == DEFAULT_CANDIDATES:
            args.candidates = DEVELOPMENT_CANDIDATES
        if args.output == DEFAULT_REPORT:
            args.output = DEVELOPMENT_REPORT
    if not args.candidates.exists():
        raise SystemExit(f"candidate file not found: {args.candidates}")
    if not args.development and not args.controls.exists():
        raise SystemExit(f"control file not found: {args.controls}")
    report = audit(
        read_jsonl(args.candidates),
        [] if args.development else read_jsonl(args.controls),
        complete=not args.allow_incomplete,
        development=args.development,
    )
    report["candidates_sha256"] = hashlib.sha256(args.candidates.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
