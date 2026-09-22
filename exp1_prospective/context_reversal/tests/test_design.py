from __future__ import annotations

import copy
import json

import pytest

from exp1_prospective.context_reversal.design import compile_plan, write_design
from exp1_prospective.context_reversal.materials import load_families


def test_context_specific_baseline_and_independent_update_templates():
    family = load_families()[0]
    rows = compile_plan([family], repeats=2)
    assert len(rows) == 8
    for row in rows:
        context = row["context_id"]
        if context != "masked":
            assert family["contexts"][context] in row["baseline_prompt"]
            assert all(family["contexts"][context] in t for t in row["update_templates"].values())
        else:
            assert "ADDITIONAL CONTEXT" not in row["baseline_prompt"]
        assert family["repeat_news"] in row["baseline_prompt"]
        assert family["evidence"] not in row["baseline_prompt"]
        for condition, template in row["update_templates"].items():
            assert template.count("__PRIOR_PROBABILITY__") == 1
            if condition == "new_news":
                assert family["evidence"] in template
            else:
                assert family["evidence"] not in template
    assert len({r["baseline_prompt"] for r in rows}) == 4


def test_private_metadata_never_changes_prompts():
    family = copy.deepcopy(load_families()[0])
    expected = compile_plan([family])
    family["private_metadata"]["secret_test_label"] = "PRIVATE_GOLD_SHOULD_NOT_RENDER"
    result = compile_plan([family])
    assert expected == result
    assert "PRIVATE_GOLD_SHOULD_NOT_RENDER" not in json.dumps(result)


def test_freeze_refuses_material_changes_and_retains_manifest(tmp_path):
    families = load_families()[:1]
    materials = tmp_path / "materials.jsonl"
    materials.write_text(json.dumps(families[0]) + "\n")
    output = tmp_path / "plan.jsonl"
    first = write_design(materials, output, 1)
    assert first["total_calls_per_model"] == 16
    assert write_design(materials, output, 1) == first
    families[0]["background"] += " Additional background sentence."
    materials.write_text(json.dumps(families[0]) + "\n")
    with pytest.raises(ValueError, match="overwrite"):
        write_design(materials, output, 1)
