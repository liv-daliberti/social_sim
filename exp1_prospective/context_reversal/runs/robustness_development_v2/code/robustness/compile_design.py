"""Freeze all robustness materials and both compatible inference plans; no inference."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path

from exp1_prospective.context_reversal import design as parent_design
from exp1_prospective.context_reversal import direction_design
from exp1_prospective.context_reversal import run_local as common
from . import materials

HERE = Path(__file__).resolve().parent
METADATA = ("parent_family_id", "sampling_family_id", "variant")


def compile_probability(rows: list[dict]) -> list[dict]:
    materials.validate_families(rows)
    by_id = {row["family_id"]: row for row in rows}
    units = parent_design.compile_plan(rows, repeats=1)
    for unit in units:
        unit.update({key: by_id[unit["family_id"]][key] for key in METADATA})
        unit["protocol"] = materials.PROTOCOL + "_probability"
        unit["sampling_note"] = "Cluster uncertainty by parent_family_id; four variants are not independent families."
    validate_plan(units, direction=False)
    return units


def compile_direction(probability_units: list[dict], probability_payload_sha256: str) -> list[dict]:
    units = direction_design.compile_plan(probability_units, probability_payload_sha256)
    by_trial = {unit["trial_id"]: unit for unit in probability_units}
    for unit in units:
        unit.update({key: by_trial[unit["trial_id"]][key] for key in METADATA})
        unit["robustness_protocol"] = materials.PROTOCOL
        unit["sampling_note"] = "Cluster uncertainty by parent_family_id; four variants are not independent families."
    validate_plan(units, direction=True)
    return units


def validate_plan(units: list[dict], *, direction: bool) -> None:
    if len(units) != 320 or len({u["trial_id"] for u in units}) != 320:
        raise ValueError("Expected 320 unique parent/variant/context units")
    if {u["parent_family_id"] for u in units} != {f"dev_{i:02d}" for i in range(1, 21)}:
        raise ValueError("Expected 20 original parent sampling families")
    expected_cells = {(f"dev_{i:02d}", variant, context) for i in range(1, 21)
                      for variant in materials.VARIANTS for context in parent_design.CONTEXTS}
    cells = {(u["parent_family_id"], u["variant"], u["context_id"]): u for u in units}
    if set(cells) != expected_cells:
        raise ValueError("Robustness plan is missing or duplicating a planned cell")
    if any(u["sampling_family_id"] != u["parent_family_id"] or u["repeat"] != 0 for u in units):
        raise ValueError("Invalid sampling group or repeat")
    if direction:
        direction_design.validate_units(units)
    for parent in sorted({u["parent_family_id"] for u in units}):
        for variant in materials.VARIANTS:
            evidence = []
            for context in parent_design.CONTEXTS:
                unit = cells[parent, variant, context]
                if direction:
                    evidence.append(unit["messages"]["new_news"])
                else:
                    _, messages = direction_design.extract_visible_text(unit)
                    evidence.append(messages["new_news"])
            if len(set(evidence)) != 1:
                raise ValueError("Evidence must be identical across contexts")
        for context in parent_design.CONTEXTS:
            base, resample = (cells[parent, variant, context] for variant in ("repaired_base", "resample"))
            fields = ("scenario_text", "messages", "direction_prompts") if direction else ("baseline_prompt", "update_templates")
            if any(base[field] != resample[field] for field in fields):
                raise ValueError("Resample prompts differ from repaired base")
            stages = [("direction", arm) for arm in direction_design.CONDITIONS] if direction else [("baseline", "baseline"), *[("update", arm) for arm in parent_design.CONDITIONS]]
            for stage, arm in stages:
                if common.request_seed(20260921, base["trial_id"], stage, arm) == common.request_seed(20260921, resample["trial_id"], stage, arm):
                    raise ValueError("Base/resample request seeds collided")


def jsonl(rows: list[dict]) -> str:
    return "".join(common.canonical_json(row) + "\n" for row in rows)


def write_bundle(data_dir: Path, run_dir: Path, parent_path: Path = materials.PARENT_PATH) -> dict:
    parents = materials.parent_materials.load_families(parent_path)
    rows = materials.build_families(parents)
    probability = compile_probability(rows)
    probability_payload = jsonl(probability)
    directions = compile_direction(probability, common.text_sha256(probability_payload))
    payloads = {
        data_dir / "families.jsonl": jsonl(rows),
        data_dir / "repair_log.jsonl": jsonl(materials.repair_log(parents, rows)),
        run_dir / "probability_plan.jsonl": probability_payload,
        run_dir / "direction_plan.jsonl": jsonl(directions),
    }
    code_files = [Path(__file__), Path(materials.__file__), HERE / "flow_specs.py", HERE / "nonflow_specs.py", HERE / "__init__.py",
                  Path(parent_design.__file__), Path(direction_design.__file__), Path(common.__file__), Path(materials.parent_materials.__file__)]
    manifest = {
        "protocol": materials.PROTOCOL, "created_at": common.utc_now(), "development_only": True,
        "parent_materials": {"path": str(parent_path.resolve()), "sha256": common.file_sha256(parent_path)},
        "n_original_sampling_families": 20, "n_material_rows": 80, "variants": list(materials.VARIANTS),
        "contexts": list(parent_design.CONTEXTS), "repeats_per_variant": 1,
        "probability_units": 320, "probability_calls_per_model": 1280,
        "direction_units": 320, "direction_calls_per_model": 960,
        "sampling_unit": "parent_family_id; all contexts, variants, and message arms stay grouped within 20 parents",
        "resample": "Same visible text as repaired_base; distinct trial IDs produce distinct existing per-request seeds.",
        "paraphrase_scope": ["background", "mechanism", "relation_schedule"],
        "fixed_base_paraphrase_fields": ["question", "resolution", "prior_information", "evidence", "repeat_news"],
        "name_only_policy": "Explicit simultaneous whole-phrase one-to-one map, exactly reversible on every visible field.",
        "context_balance": "All three relation roles and actors appear once in every schedule; context changes entity assignment only. Role order rotates by parent, and the evidence actor occupies every position across contexts.",
        "new_human_review_requested": False, "model_outputs_used_to_select_families": False,
        "all_original_families_retained": True,
        "artifacts": {path.name: {"path": str(path.resolve()), "sha256": common.text_sha256(payload)} for path, payload in payloads.items()},
        "code_sha256": {str(path.resolve()): common.file_sha256(path) for path in code_files},
    }
    manifest_path = run_dir / "plan.manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        for key in ("parent_materials", "artifacts", "code_sha256", "variants", "n_original_sampling_families"):
            if existing.get(key) != manifest[key]:
                raise ValueError(f"Existing frozen robustness provenance differs: {key}")
        for path, payload in payloads.items():
            if not path.exists() or path.read_text() != payload:
                raise ValueError("Frozen robustness artifact is missing or differs")
        return existing
    for path, payload in payloads.items():
        if path.exists() and path.read_text() != payload:
            raise ValueError(f"Refusing to overwrite different robustness artifact: {path}")
    for path, payload in payloads.items():
        common.atomic_write(path, payload)
    common.atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-materials", type=Path, default=materials.PARENT_PATH)
    parser.add_argument("--data-dir", type=Path, default=HERE.parent / "data/robustness_development_v2")
    parser.add_argument("--run-dir", type=Path, default=HERE.parent / "runs/robustness_development_v2")
    args = parser.parse_args()
    print(json.dumps(write_bundle(args.data_dir, args.run_dir, args.parent_materials), indent=2))


if __name__ == "__main__":
    main()
