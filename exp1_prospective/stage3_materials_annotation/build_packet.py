#!/usr/bin/env python3
"""Build the frozen, blinded Stage 3 materials-annotation packet."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_SOURCE = (
    REPO
    / "exp1_prospective/data/counterfactuals/counterfactuals_2026-06-10.jsonl"
)
DEFAULT_MARKETS = (
    REPO / "exp1_prospective/data/selected_markets/diverse_2026-06-09.jsonl"
)
DEFAULT_FORECASTS = (
    REPO
    / "exp1_prospective/data/initial_forecasts/forecasts_DeepSeek-V4-Pro_2026-06-10.jsonl"
)
DEFAULT_OUTPUT = HERE / "generated_v8"
SEED = "stage3_materials_annotation_v1_20260820"
VERSION = "stage3_materials_annotation_v8"
DIRECTIONS = ("pro_H1", "anti_H1", "orthogonal")
EXPECTED = {
    "pro_H1": "more_likely",
    "anti_H1": "less_likely",
    "orthogonal": "no_material_effect",
}
REVIEWERS = tuple(f"annotator_{index:02d}" for index in range(1, 10))
RESPONSE_FIELDS = (
    "consent_confirmed",
    "conditional_direction",
    "direction_confidence",
    "clarity",
    "plausibility",
    "usable_premise",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def rank_key(*parts: str) -> str:
    joined = "|".join((SEED, *parts)).encode("utf-8")
    return sha256_bytes(joined)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise SystemExit(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows
    )


def csv_bytes(rows: list[dict[str, Any]], fieldnames: list[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def write_frozen(path: Path, data: bytes) -> None:
    if path.exists():
        current = path.read_bytes()
        if current != data:
            raise SystemExit(f"refusing to overwrite different frozen artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def validate_sources(
    packets: list[dict[str, Any]], markets: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, list[dict[str, Any]]]]]:
    if len(packets) != 900:
        raise SystemExit(f"expected 900 frozen packets, found {len(packets)}")
    market_map = {str(row.get("market_id")): row for row in markets}
    if len(market_map) != 100:
        raise SystemExit(f"expected 100 distinct frozen markets, found {len(market_map)}")

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seen_cf: set[str] = set()
    for row in packets:
        market_id = str(row.get("market_id"))
        direction = str(row.get("direction"))
        cf_id = str(row.get("cf_id"))
        if market_id not in market_map:
            raise SystemExit(f"packet market is absent from frozen market file: {market_id}")
        if direction not in DIRECTIONS:
            raise SystemExit(f"unexpected direction {direction!r} for {cf_id}")
        if not cf_id or cf_id in seen_cf:
            raise SystemExit(f"missing or duplicate cf_id: {cf_id!r}")
        seen_cf.add(cf_id)
        grouped[market_id][direction].append(row)

    if set(grouped) != set(market_map):
        raise SystemExit("packet and market ID sets differ")
    for market_id, by_direction in grouped.items():
        if set(by_direction) != set(DIRECTIONS):
            raise SystemExit(f"market {market_id} lacks a registered direction")
        counts = {direction: len(rows) for direction, rows in by_direction.items()}
        if set(counts.values()) != {3}:
            raise SystemExit(f"market {market_id} direction counts are {counts}")
    return market_map, grouped


def build_forecast_context(
    forecasts: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Extract the shared pre-report rationale and observed forecast probabilities."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in forecasts:
        market_id = str(row.get("market_id"))
        if not market_id or market_id == "None":
            raise SystemExit("frozen forecast row lacks market_id")
        grouped[market_id].append(row)

    contexts: dict[str, dict[str, Any]] = {}
    for market_id, rows in grouped.items():
        rationales = {
            str((row.get("structured_forecast") or {}).get("rationale") or "").strip()
            for row in rows
        }
        rationales.discard("")
        if len(rationales) != 1:
            raise SystemExit(
                f"market {market_id} has {len(rationales)} distinct nonempty "
                "frozen forecast rationales"
            )
        yes_probs = {
            float(row["yes_prob"])
            for row in rows
            if row.get("yes_prob") is not None
        }
        if not yes_probs:
            raise SystemExit(f"market {market_id} lacks a frozen yes_prob")
        contexts[market_id] = {
            "rationale": rationales.pop(),
            "yes_probs": yes_probs,
        }
    return contexts


def select_items(
    market_map: dict[str, dict[str, Any]],
    grouped: dict[str, dict[str, list[dict[str, Any]]]],
    forecast_context: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_markets = sorted(market_map, key=lambda value: rank_key("market", value))[:18]
    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for position, market_id in enumerate(selected_markets):
        direction = DIRECTIONS[position % len(DIRECTIONS)]
        candidates = grouped[market_id][direction]
        chosen = min(candidates, key=lambda row: rank_key("packet", str(row["cf_id"])))
        market = market_map[market_id]
        context = forecast_context.get(market_id)
        if context is None:
            raise SystemExit(f"selected market {market_id} lacks frozen forecast context")
        generated_at_prob = float(chosen["yes_prob_at_generation"])
        if not any(
            abs(generated_at_prob - value) < 1e-12 for value in context["yes_probs"]
        ):
            raise SystemExit(
                f"selected packet {chosen['cf_id']} does not match a frozen "
                "June 10 forecast probability"
            )
        item_id = "s3mat_" + rank_key("item", str(chosen["cf_id"]))[:14]
        public_rows.append(
            {
                "item_id": item_id,
                "question": str(chosen.get("question") or market.get("question") or ""),
                "resolution_criteria": str(
                    market.get("description") or market.get("rules") or ""
                ),
                "evidence_headline": str(chosen.get("evidence_headline") or ""),
                "evidence_text": str(chosen.get("evidence_text") or ""),
                "forecast_context": context["rationale"],
                "synthetic_notice": (
                    "This report is fictional research material dated June 10, "
                    "2026. Treat June 10, 2026 as today; judge the report as "
                    "written and do not search for it online."
                ),
            }
        )
        private_rows.append(
            {
                "item_id": item_id,
                "market_id": market_id,
                "cf_id": str(chosen["cf_id"]),
                "registered_direction": direction,
                "expected_conditional_direction": EXPECTED[direction],
                "cf_index": int(chosen["cf_index"]),
                "source": str(chosen.get("source") or ""),
            }
        )

    if Counter(row["registered_direction"] for row in private_rows) != Counter(
        {direction: 6 for direction in DIRECTIONS}
    ):
        raise AssertionError("selected packet is not direction-balanced")
    if len({row["market_id"] for row in private_rows}) != 18:
        raise AssertionError("selected packet repeats a market")
    return public_rows, private_rows


def build_assignments(public_rows: list[dict[str, Any]]) -> dict[str, Any]:
    assignments = []
    all_ids = {row["item_id"] for row in public_rows}
    for reviewer_id in REVIEWERS:
        ordered = sorted(
            public_rows,
            key=lambda row: rank_key("order", reviewer_id, str(row["item_id"])),
        )
        item_ids = [str(row["item_id"]) for row in ordered]
        if set(item_ids) != all_ids or len(item_ids) != 18:
            raise AssertionError(f"invalid assignment for {reviewer_id}")
        assignments.append(
            {
                "reviewer_id": reviewer_id,
                "item_ids_in_order": item_ids,
            }
        )
    return {
        "version": VERSION,
        "seed": SEED,
        "reviewer_count": len(REVIEWERS),
        "items_per_reviewer": 18,
        "ratings_per_item": len(REVIEWERS),
        "expected_rating_count": len(REVIEWERS) * 18,
        "assignments": assignments,
    }


def reviewer_rows(
    reviewer_id: str,
    item_ids: list[str],
    public_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for order, item_id in enumerate(item_ids, 1):
        item = public_by_id[item_id]
        row = {
            "reviewer_id": reviewer_id,
            "display_order": order,
            **item,
        }
        row.update({field: "" for field in RESPONSE_FIELDS})
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--markets", type=Path, default=DEFAULT_MARKETS)
    parser.add_argument("--forecasts", type=Path, default=DEFAULT_FORECASTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    packets = read_jsonl(args.source)
    markets = read_jsonl(args.markets)
    forecasts = read_jsonl(args.forecasts)
    market_map, grouped = validate_sources(packets, markets)
    forecast_context = build_forecast_context(forecasts)
    public_rows, private_rows = select_items(
        market_map, grouped, forecast_context
    )
    assignments = build_assignments(public_rows)
    public_by_id = {str(row["item_id"]): row for row in public_rows}

    output = args.output_dir
    public_data = jsonl_bytes(public_rows)
    private_data = jsonl_bytes(private_rows)
    assignment_data = json_bytes(assignments)
    write_frozen(output / "public_items.jsonl", public_data)
    write_frozen(output / "private_key.jsonl", private_data)
    write_frozen(output / "assignments.json", assignment_data)

    packet_hashes: dict[str, str] = {}
    fieldnames = [
        "reviewer_id",
        "display_order",
        "item_id",
        "question",
        "resolution_criteria",
        "evidence_headline",
        "evidence_text",
        "forecast_context",
        "synthetic_notice",
        *RESPONSE_FIELDS,
    ]
    for assignment in assignments["assignments"]:
        reviewer_id = str(assignment["reviewer_id"])
        rows = reviewer_rows(
            reviewer_id,
            list(assignment["item_ids_in_order"]),
            public_by_id,
        )
        data = csv_bytes(rows, fieldnames)
        relative = f"reviewer_packets/{reviewer_id}.csv"
        write_frozen(output / relative, data)
        packet_hashes[relative] = sha256_bytes(data)

    combined_rows = []
    for assignment in assignments["assignments"]:
        reviewer_id = str(assignment["reviewer_id"])
        combined_rows.extend(
            reviewer_rows(
                reviewer_id,
                list(assignment["item_ids_in_order"]),
                public_by_id,
            )
        )
    combined_data = csv_bytes(combined_rows, fieldnames)
    write_frozen(output / "responses_template.csv", combined_data)

    manifest = {
        "version": VERSION,
        "freeze_date": "2026-08-21",
        "seed": SEED,
        "source": str(args.source.relative_to(REPO)),
        "source_sha256": file_sha256(args.source),
        "markets": str(args.markets.relative_to(REPO)),
        "markets_sha256": file_sha256(args.markets),
        "forecast_context_source": str(args.forecasts.relative_to(REPO)),
        "forecast_context_source_sha256": file_sha256(args.forecasts),
        "forecast_context_field": "structured_forecast.rationale",
        "protocol_sha256": file_sha256(HERE / "PROTOCOL.md"),
        "annotation_form_sha256": file_sha256(HERE / "ANNOTATION_FORM.md"),
        "item_count": 18,
        "market_count": 18,
        "direction_counts": dict(Counter(row["registered_direction"] for row in private_rows)),
        "reviewer_count": len(REVIEWERS),
        "expected_rating_count": len(REVIEWERS) * 18,
        "public_items_sha256": sha256_bytes(public_data),
        "private_key_sha256": sha256_bytes(private_data),
        "assignments_sha256": sha256_bytes(assignment_data),
        "responses_template_sha256": sha256_bytes(combined_data),
        "reviewer_packet_sha256": packet_hashes,
    }
    write_frozen(output / "manifest.json", json_bytes(manifest))

    print("PASS: selected 18 distinct markets; 6 pro, 6 anti, 6 orthogonal")
    print(
        f"PASS: built {len(REVIEWERS)} independently ordered packets and "
        f"{len(REVIEWERS) * 18} response rows"
    )
    print(f"manifest: {output / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
