"""Export blinded development candidates for future independent human review.

This export does not collect annotations, validate materials, or change the pilot.
Give reviewers only one review_batch from the public candidate file plus the
reviewer README. Never distribute the separate private mapping or target outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
from pathlib import Path
from typing import Any

from exp1_prospective.context_reversal.design import CONTEXTS, visible_context
from exp1_prospective.context_reversal.materials import DEFAULT_MATERIALS_PATH, load_families

HERE = Path(__file__).resolve().parent
DEFAULT_SEED = 20260921
FIELDS = (
    "review_id", "review_batch", "material_status", "scenario_text", "new_message",
    "direction_judgment", "relevance_judgment", "confidence_1_to_5",
    "ambiguity_judgment", "alternative_causal_paths", "reviewer_notes",
)
RESPONSE_FIELDS = FIELDS[5:]


def build_review_packet(families: list[dict], seed: int = DEFAULT_SEED) -> tuple[list[dict], list[dict]]:
    """Return public rows and a separate coordinator-only identification mapping.

    A reviewer batch contains one context per family. For each complete block of
    four families, every batch contains all four context types once. Batch names
    therefore do not identify intended direction. Assignment and row order are
    deterministic given the source and seed; IDs carry no condition information.
    """
    rng = random.Random(seed)
    ordered = list(families)
    rng.shuffle(ordered)
    rows: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for start in range(0, len(ordered), len(CONTEXTS)):
        block = ordered[start:start + len(CONTEXTS)]
        context_order = list(CONTEXTS)
        rotations = list(range(len(CONTEXTS)))
        rng.shuffle(context_order)
        rng.shuffle(rotations)
        for family, rotation in zip(block, rotations):
            for batch in range(len(CONTEXTS)):
                context_id = context_order[(batch + rotation) % len(CONTEXTS)]
                review_id = f"R{rng.getrandbits(80):020x}"
                if review_id in identifiers:
                    raise ValueError("Unexpected anonymous review-ID collision")
                identifiers.add(review_id)
                row = {key: "" for key in FIELDS}
                row.update({
                    "review_id": review_id,
                    "review_batch": f"batch_{batch + 1}",
                    "material_status": family["provenance"],
                    "scenario_text": visible_context(family, context_id),
                    "new_message": family["evidence"],
                })
                rows.append(row)
                mapping.append({
                    "review_id": review_id,
                    "review_batch": row["review_batch"],
                    "family_id": family["family_id"],
                    "context_id": context_id,
                    "author_direction_not_validated": family["private_metadata"]["expected_direction"].get(context_id),
                })
    rng.shuffle(rows)
    ordered_mapping = {entry["review_id"]: entry for entry in mapping}
    return rows, [ordered_mapping[row["review_id"]] for row in rows]


def write_review_packet(materials: Path, output: Path, seed: int = DEFAULT_SEED) -> dict:
    """Write CSV/JSON reviewer packets and a separate private mapping, fail closed.

    Existing identical files are permitted. A changed source, seed, or packet
    requires a new output stem, preserving the exported instrument's identity.
    """
    source_hash = hashlib.sha256(materials.read_bytes()).hexdigest()
    families = load_families(materials)
    rows, mapping = build_review_packet(families, seed)
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    csv_text = csv_buffer.getvalue()
    public_payload = {
        "packet_status": "candidates_for_future_independent_review",
        "material_status": "authored_development_unvalidated",
        "independent_validation": "not_collected",
        "source_materials_sha256": source_hash,
        "reviewer_instructions": "HUMAN_REVIEW_README.md",
        "candidates": rows,
    }
    private_payload = {
        "warning": "COORDINATOR ONLY: contains author labels; do not show reviewers or target models.",
        "validation_status": "not_collected",
        "source_materials_sha256": source_hash,
        "reviewer_csv_sha256": hashlib.sha256(csv_text.encode()).hexdigest(),
        "seed": seed,
        "expected_direction_source": "development author, not independent annotation",
        "mapping": mapping,
    }
    files = {
        output.with_suffix(".csv"): csv_text,
        output.with_suffix(".json"): json.dumps(public_payload, indent=2, sort_keys=True) + "\n",
        output.with_suffix(".private.json"): json.dumps(private_payload, indent=2, sort_keys=True) + "\n",
    }
    # Check all conflicts before creating any output.
    for path, contents in files.items():
        if path.exists() and path.read_text() != contents:
            raise ValueError(f"Refusing to replace a different review packet: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    for path, contents in files.items():
        path.write_text(contents)
    if hashlib.sha256(materials.read_bytes()).hexdigest() != source_hash:
        raise ValueError("Source materials changed while the packet was being exported")
    return {"candidate_rows": len(rows), "families": len(families),
            "review_batches": len(CONTEXTS), "source_materials_sha256": source_hash,
            "outputs": [str(path) for path in files]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materials", type=Path, default=DEFAULT_MATERIALS_PATH)
    parser.add_argument("--output", type=Path, default=HERE / "data/human_review_candidates.csv")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    print(json.dumps(write_review_packet(args.materials, args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
