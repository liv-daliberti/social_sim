"""Regression tests for the paper-cited coin probe's local parsing helpers."""

from eval.run_coin_probe import _strip_reasoning, parse_p


def test_parse_p_uses_last_valid_flat_object():
    raw = 'draft {"p_heads": 0.50} final {"rationale": "gut", "p_heads": "0.67"}'
    assert parse_p(raw) == 0.67


def test_parse_p_ignores_invalid_objects():
    assert parse_p('{"p_heads": "unknown"}') is None
    assert parse_p("no JSON answer") is None


def test_strip_reasoning_removes_think_block():
    assert _strip_reasoning("<think>private</think>  {\"p_heads\": 0.6}") == '{"p_heads": 0.6}'
