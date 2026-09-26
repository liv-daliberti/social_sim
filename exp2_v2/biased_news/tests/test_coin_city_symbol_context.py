import json
import re
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.coin_city_stable_relationship_claude_n250 import EXPERIMENT
from engine.coin_city_symbol_context_arm import (
    ARM,
    SYMBOLS,
    city_symbol,
    make_prompt,
    strong_symbol,
    validate_arm_prompt,
    weak_symbol,
)
from analysis.analyze_coin_city_symbol_context import CONTROL_MODELS, EXPECTED_TASKS
from eval.run_coin_city_symbol_context_open_model import format_prompts
from local_results.symbol_context_model_comparison_20260824.reparse_qwen2_5_32b import (
    parse_explicit_arithmetic,
)


DESIGN = ROOT / "data" / EXPERIMENT / "design"
NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_open_model_chat_template_modes_keep_qwen_option_model_specific():
    class RecordingTokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return "formatted"

    tasks = [{"prompt": "frozen prompt"}]
    tokenizer = RecordingTokenizer()
    assert format_prompts(tokenizer, tasks, "auto") == ["formatted"]
    assert tokenizer.calls[-1] == (
        [{"role": "user", "content": "frozen prompt"}],
        {"tokenize": False, "add_generation_prompt": True},
    )

    assert format_prompts(tokenizer, tasks, "qwen-no-thinking") == ["formatted"]
    assert tokenizer.calls[-1][1]["enable_thinking"] is False


def test_arithmetic_reparse_preserves_primary_forecast_range():
    expression, value = parse_explicit_arithmetic('{"predicted_poll": 40 + 2.5}')
    assert expression == "40 + 2.5"
    assert value == 42.5
    with pytest.raises(ValueError, match="outside the 0--100 forecast range"):
        parse_explicit_arithmetic('{"predicted_poll": 100 + 0.1}')


def test_mapping_reverses_and_target_symbol_is_balanced():
    episodes = read_jsonl(DESIGN / "episodes.jsonl")
    assert strong_symbol(episodes[0]) == SYMBOLS[0]
    assert strong_symbol(episodes[1]) == SYMBOLS[1]
    cross = {
        (regime, symbol): 0
        for regime in ("higher", "lower")
        for symbol in SYMBOLS
    }
    for episode in episodes:
        assert strong_symbol(episode) != weak_symbol(episode)
        regime = "higher" if episode["cue_strong"] else "lower"
        cross[(regime, city_symbol(episode, "C"))] += 1
    assert max(cross.values()) - min(cross.values()) <= 1


def test_only_examples_reveal_the_episode_local_mapping():
    for episode in read_jsonl(DESIGN / "episodes.jsonl")[:20]:
        prompt = make_prompt(episode, 0)
        validate_arm_prompt(prompt, ARM)
        assert "KIV and ZOR are arbitrary labels" in prompt
        assert city_symbol(episode, episode["strong_reference_city"]) == strong_symbol(
            episode
        )
        other = "B" if episode["strong_reference_city"] == "A" else "A"
        assert city_symbol(episode, other) == weak_symbol(episode)
        assert city_symbol(episode, "C") == (
            strong_symbol(episode) if episode["cue_strong"] else weak_symbol(episode)
        )
        lowered = prompt.lower()
        assert "national news" not in lowered
        assert "local news" not in lowered
        assert "strongly responsive" not in lowered
        assert "weakly responsive" not in lowered


def test_symbol_arm_preserves_all_numeric_prompt_content():
    episodes = read_jsonl(DESIGN / "episodes.jsonl")
    semantic = {
        row["task_id"]: row["prompt"]
        for row in read_jsonl(DESIGN / "tasks_abc_context.jsonl")
    }
    for episode in episodes[::37]:
        for k in range(5):
            task_id = f"coincityclaude250v4_{episode['episode']:04d}_c{k}"
            assert NUMBER.findall(make_prompt(episode, k)) == NUMBER.findall(
                semantic[task_id]
            )


def test_frozen_symbol_analysis_matches_the_executed_control_scope():
    result = json.loads(
        (DESIGN.parent / "analysis" / "symbol_context_results.json").read_text()
    )

    assert CONTROL_MODELS == ("DeepSeek-V4-Pro", "Qwen3-4B-Instruct-2507")
    assert result["complete"] is True
    assert result["complete_models"] == list(CONTROL_MODELS)
    assert set(result["models"]) == set(CONTROL_MODELS)
    for model in CONTROL_MODELS:
        for counts in result["models"][model]["record_counts"].values():
            assert counts == EXPECTED_TASKS
