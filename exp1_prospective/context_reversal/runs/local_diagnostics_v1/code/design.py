"""Compile a frozen development design without exposing scoring metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from exp1_prospective.context_reversal.materials import (
    DEFAULT_MATERIALS_PATH,
    load_families,
    validate_families,
)

HERE = Path(__file__).resolve().parent
CONTEXTS = ("positive", "negative", "broken", "masked")
CONDITIONS = ("new_news", "no_news", "repeated_news")
PRIOR_TOKEN = "__PRIOR_PROBABILITY__"
PROTOCOL = "exp1_context_reversal_local_development_v1"
STATUS = "authored_development_unvalidated"
RESPONSE_INSTRUCTION = (
    'Return exactly one JSON object with the key "probability" and a decimal '
    'number between 0 and 1 for the probability of a YES resolution. '
    'Do not include an explanation or any other fields.'
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def visible_context(family: dict, context_id: str) -> str:
    # Deliberate whitelist: never render role labels, gold signs, chains, or provenance.
    pieces = [
        "This is a fictional forecasting scenario. Use only the information supplied "
        "here, and treat the stated relationships as applying within this scenario. "
        "Do not search or use the outcomes of real events.",
        "QUESTION\n" + family["question"],
        "RESOLUTION RULE\n" + family["resolution"],
        "BACKGROUND\n" + family["background"],
        "INFORMATION ALREADY AVAILABLE\n" + family["prior_information"],
    ]
    if context_id != "masked":
        pieces.append("ADDITIONAL CONTEXT\n" + family["contexts"][context_id])
    return "\n\n".join(pieces)


def compile_plan(families: list[dict], repeats: int = 3) -> list[dict]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    validate_families(families)
    units = []
    for family in families:
        for context_id in CONTEXTS:
            context = visible_context(family, context_id)
            baseline = (
                context + "\n\nEstimate the probability of a YES resolution using "
                "only the information above.\n" + RESPONSE_INSTRUCTION
            )
            messages = {
                "new_news": family["evidence"],
                "no_news": "No new information about this scenario has become available.",
                "repeated_news": family["repeat_news"],
            }
            templates = {
                condition: context
                + "\n\nYOUR PREVIOUS FORECAST\n"
                + PRIOR_TOKEN
                + "\n\nMESSAGE\n"
                + news
                + "\n\nRevise your previous probability only to the extent that this "
                "message supplies new information bearing on the question. Repeated "
                "information is not independent evidence. The context above is unchanged.\n"
                + RESPONSE_INSTRUCTION
                for condition, news in messages.items()
            }
            for repeat in range(repeats):
                units.append({
                    "protocol": PROTOCOL,
                    "trial_id": f'{family["family_id"]}:{context_id}:r{repeat}',
                    "family_id": family["family_id"],
                    "domain": family["domain"],
                    "context_id": context_id,
                    "repeat": repeat,
                    "material_status": STATUS,
                    "baseline_prompt": baseline,
                    "update_templates": templates,
                })
    return units


def write_design(materials: Path, output: Path, repeats: int, limit: int | None = None) -> dict:
    families = load_families(materials)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit-families must be positive")
        families = families[:limit]
    units = compile_plan(families, repeats)
    payload = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in units)
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() and output.read_text() != payload:
        raise ValueError("refusing to overwrite a different frozen design; choose a new output")
    manifest = {
        "protocol": PROTOCOL,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": "development",
        "material_status": STATUS,
        "human_validation": "not_collected",
        "materials_path": str(materials.resolve()),
        "materials_sha256": hashlib.sha256(materials.read_bytes()).hexdigest(),
        "design_path": str(output.resolve()),
        "design_sha256": digest(payload),
        "family_ids": [row["family_id"] for row in families],
        "contexts": list(CONTEXTS),
        "update_conditions": list(CONDITIONS),
        "repeats": repeats,
        "baseline_calls": len(units),
        "update_calls": len(units) * len(CONDITIONS),
        "total_calls_per_model": len(units) * (1 + len(CONDITIONS)),
        "sampling_unit": "independent authored family; repeats are not independent families",
        "for_confirmatory_inference": False,
        "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in [Path(__file__), HERE / "materials.py"]},
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        for key in ("design_sha256", "materials_sha256", "repeats", "family_ids", "code_sha256"):
            if existing.get(key) != manifest[key]:
                raise ValueError(f"existing manifest disagrees with frozen provenance: {key}")
        return existing
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materials", type=Path, default=DEFAULT_MATERIALS_PATH)
    parser.add_argument("--output", type=Path, default=HERE / "data/development_plan.jsonl")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit-families", type=int)
    args = parser.parse_args()
    print(json.dumps(write_design(args.materials, args.output, args.repeats, args.limit_families), indent=2))


if __name__ == "__main__":
    main()
