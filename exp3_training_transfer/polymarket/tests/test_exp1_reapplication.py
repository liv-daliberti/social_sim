from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


evaluate = load_module("evaluate_exp1_reapplication", "evaluate_exp1_reapplication.py")
analyze = load_module("analyze_exp1_reapplication", "analyze_exp1_reapplication.py")


def assignment_literal(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"missing assignment {name} in {path}")


def test_update_prompt_is_byte_identical_to_frozen_exp1_prompt():
    source = Path(__file__).resolve().parents[3] / "exp1_prospective/agent/local_updated_forecast.py"
    assert assignment_literal(source, "_UPDATE_PROMPT") == evaluate.UPDATE_PROMPT


def test_frozen_instrument_is_balanced_and_complete():
    examples = evaluate.load_instrument()
    assert len(examples) == 900
    assert len({row["task_id"] for row in examples}) == 100
    assert {direction: sum(row["direction"] == direction for row in examples) for direction in evaluate.DIRECTION_ORDER} == {
        "pro_H1": 300,
        "anti_H1": 300,
        "orthogonal": 300,
    }
    assert all(0.0 <= row["initial_yes_prob"] <= 1.0 for row in examples)


def test_parser_extracts_structured_probabilities():
    text = '{"hypotheses":[{"id":"H1","posterior_probability":0.7},{"id":"H2","posterior_probability":0.3}],"yes_prob":0.7}'
    parsed, yes_prob, h1_prob, error = evaluate.resolve_updated_fields(text)
    assert parsed is not None
    assert yes_prob == 0.7
    assert h1_prob == 0.7
    assert error is None


def test_parser_rejects_percentage_scale():
    text = '{"hypotheses":[{"id":"H1","posterior_probability":70}],"yes_prob":70}'
    _, yes_prob, h1_prob, error = evaluate.resolve_updated_fields(text)
    assert yes_prob is None
    assert h1_prob is None
    assert "invalid" in error


def test_metric_thresholds_and_directions():
    pro = {
        "direction": "pro_H1",
        "delta_yes_prob": 0.05,
        "delta_h1_prob": 0.04,
        "updated_yes_prob": 0.54,
        "updated_h1_prob": 0.54,
    }
    anti_wrong = {
        "direction": "anti_H1",
        "delta_yes_prob": 0.05,
        "delta_h1_prob": 0.04,
        "updated_yes_prob": 0.54,
        "updated_h1_prob": 0.54,
    }
    small = {**pro, "delta_yes_prob": 0.029, "delta_h1_prob": 0.029}
    assert analyze.record_metrics(pro)["EHC"] == 1
    assert analyze.record_metrics(pro)["HFC"] == 1
    assert analyze.record_metrics(anti_wrong)["EHC"] == 0
    assert analyze.record_metrics(anti_wrong)["HFC"] == 0
    assert analyze.record_metrics(small)["EHC"] is None
    assert analyze.record_metrics(small)["HFC"] is None


def test_analyzer_tolerates_missing_metric_values():
    assert analyze.mean([None, 0.5, None]) == 0.5
    assert analyze.percentile_interval([]) == {
        "ci95_low": None,
        "ci95_high": None,
    }
