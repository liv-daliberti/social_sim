"""Freeze a direction-only companion by extracting exact visible parent text."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from exp1_prospective.context_reversal.design import RESPONSE_INSTRUCTION
from exp1_prospective.context_reversal import run_local as common

CONDITIONS = common.CONDITIONS
CONTEXTS = ("positive", "negative", "broken", "masked")
DIRECTIONS = ("increase", "decrease", "unchanged", "unclear")
PROTOCOL = "exp1_context_reversal_direction_companion_v1"
BASELINE_SUFFIX = "\n\nEstimate the probability of a YES resolution using only the information above.\n" + RESPONSE_INSTRUCTION
UPDATE_SUFFIX = (
    "\n\nRevise your previous probability only to the extent that this "
    "message supplies new information bearing on the question. Repeated "
    "information is not independent evidence. The context above is unchanged.\n"
    + RESPONSE_INSTRUCTION
)
DIRECTION_INSTRUCTION = (
    "Compare the scenario before this message with the same scenario after this "
    "message. Classify the direction in which the message should change the "
    "probability of a YES resolution, using only the information supplied. "
    "Repeated information is not independent evidence. The context above is unchanged. "
    "Choose increase if the probability should rise, decrease if it should fall, "
    "unchanged if the message supplies no new information bearing on the question, "
    "or unclear if the supplied information does not determine the direction. "
    "Do not estimate or report any numerical probability. "
    'Return exactly one JSON object with the key "direction" and one of the strings '
    '"increase", "decrease", "unchanged", or "unclear". '
    "Do not include an explanation or any other fields."
)
METADATA = ("trial_id", "family_id", "domain", "context_id", "repeat", "material_status")


def extract_visible_text(unit: dict) -> tuple[str, dict[str, str]]:
    baseline = unit["baseline_prompt"]
    if not baseline.endswith(BASELINE_SUFFIX):
        raise ValueError(f"{unit['trial_id']}: unrecognized parent baseline instruction")
    context = baseline[:-len(BASELINE_SUFFIX)]
    prefix = context + "\n\nYOUR PREVIOUS FORECAST\n" + common.PRIOR_TOKEN + "\n\nMESSAGE\n"
    messages = {}
    for condition in CONDITIONS:
        template = unit["update_templates"][condition]
        if not template.startswith(prefix) or not template.endswith(UPDATE_SUFFIX):
            raise ValueError(f"{unit['trial_id']}/{condition}: parent context or instruction mismatch")
        message = template[len(prefix):-len(UPDATE_SUFFIX)]
        if not message.strip():
            raise ValueError("Empty parent message")
        messages[condition] = message
    return context, messages


def render_prompt(context: str, message: str) -> str:
    return context + "\n\nMESSAGE\n" + message + "\n\n" + DIRECTION_INSTRUCTION


def compile_plan(parent_units: list[dict], parent_sha256: str) -> list[dict]:
    units = []
    for parent in parent_units:
        if parent["repeat"] != 0:
            raise ValueError("Direction companion requires the frozen single-repeat parent plan")
        context, messages = extract_visible_text(parent)
        prompts = {condition: render_prompt(context, message) for condition, message in messages.items()}
        units.append({
            **{key: parent[key] for key in METADATA}, "protocol": PROTOCOL,
            "parent_plan_sha256": parent_sha256,
            "parent_unit_sha256": common.text_sha256(common.canonical_json(parent)),
            "scenario_text": context, "messages": messages,
            "direction_prompts": prompts,
            "direction_prompt_sha256": {key: common.text_sha256(value) for key, value in prompts.items()},
        })
    validate_units(units)
    return units


def validate_units(units: list[dict]) -> dict[str, dict]:
    if not units:
        raise ValueError("Direction plan is empty")
    by_id, cells, contexts, domains, parent_hashes = {}, set(), defaultdict(set), {}, set()
    for unit in units:
        if not isinstance(unit, dict) or unit.get("protocol") != PROTOCOL:
            raise ValueError("Invalid direction protocol")
        for key in (*METADATA, "scenario_text", "parent_plan_sha256", "parent_unit_sha256"):
            if key == "repeat":
                if type(unit.get(key)) is not int or unit[key] != 0:
                    raise ValueError("Direction plan requires repeat 0")
            elif not isinstance(unit.get(key), str) or not unit[key].strip():
                raise ValueError(f"Invalid direction-plan field: {key}")
        context, family = unit["context_id"], unit["family_id"]
        if context not in CONTEXTS:
            raise ValueError("Unknown direction context")
        cell = (family, context, unit["repeat"])
        if unit["trial_id"] in by_id or cell in cells:
            raise ValueError("Duplicate direction trial or cell")
        if family in domains and domains[family] != unit["domain"]:
            raise ValueError("Inconsistent family domain")
        domains[family] = unit["domain"]
        cells.add(cell)
        contexts[family].add(context)
        parent_hashes.add(unit["parent_plan_sha256"])
        for field in ("messages", "direction_prompts", "direction_prompt_sha256"):
            if not isinstance(unit.get(field), dict) or set(unit[field]) != set(CONDITIONS):
                raise ValueError(f"{field} must contain exactly the three independent message conditions")
        for condition in CONDITIONS:
            message, prompt = unit["messages"][condition], unit["direction_prompts"][condition]
            if not isinstance(message, str) or not message.strip():
                raise ValueError("Empty message")
            if prompt != render_prompt(unit["scenario_text"], message):
                raise ValueError("Direction prompt does not match its exact scenario and message")
            if common.PRIOR_TOKEN in prompt or "YOUR PREVIOUS FORECAST" in prompt:
                raise ValueError("Numeric forecast placeholder in direction prompt")
            if unit["direction_prompt_sha256"][condition] != common.text_sha256(prompt):
                raise ValueError("Direction prompt hash mismatch")
        by_id[unit["trial_id"]] = unit
    if len(parent_hashes) != 1:
        raise ValueError("Direction units refer to different parent plans")
    if any(value != set(CONTEXTS) for value in contexts.values()):
        raise ValueError("Each direction family must contain all four contexts")
    return by_id


def read_units(path: Path) -> list[dict]:
    units = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    validate_units(units)
    return units


def write_design(parent_plan: Path, output: Path) -> dict:
    if output.resolve() == parent_plan.resolve():
        raise ValueError("Parent plan and direction output must differ")
    units = compile_plan(common.read_units(parent_plan), common.file_sha256(parent_plan))
    payload = "".join(common.canonical_json(row) + "\n" for row in units)
    manifest_path = output.with_suffix(".manifest.json")
    manifest = {
        "protocol": PROTOCOL, "created_at": common.utc_now(),
        "parent_plan_path": str(parent_plan.resolve()), "parent_plan_sha256": common.file_sha256(parent_plan),
        "design_path": str(output.resolve()), "design_sha256": common.text_sha256(payload),
        "family_ids": sorted({u["family_id"] for u in units}), "contexts": list(CONTEXTS),
        "conditions": list(CONDITIONS), "repeats": 1, "unit_count": len(units),
        "total_calls_per_model": 3 * len(units), "numeric_forecasts_elicited": False,
        "independent_messages": True, "for_confirmatory_inference": False,
        "human_review": "User reports review already completed; no new human validation requested or claimed.",
        "interpretation": "Exploratory direction companion; original numeric protocol remains the parent analysis.",
        "code_sha256": {path.name: common.file_sha256(path) for path in (Path(__file__), Path(common.__file__))},
    }
    if output.exists() and output.read_text() != payload:
        raise ValueError("Refusing to overwrite a different frozen direction design")
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        for key in ("protocol", "parent_plan_sha256", "design_sha256", "code_sha256"):
            if existing.get(key) != manifest[key]:
                raise ValueError(f"Frozen direction manifest disagrees: {key}")
        if not output.exists():
            raise ValueError("Frozen direction manifest exists but its design is missing")
        return existing
    common.atomic_write(output, payload)
    common.atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_design(args.parent_plan, args.output), indent=2))


if __name__ == "__main__":
    main()
