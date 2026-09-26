#!/usr/bin/env python3

from __future__ import annotations

import copy
import sys
from argparse import Namespace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_exp4_scale_holdout as builder  # noqa: E402
import evaluate_scale_holdout as evaluator  # noqa: E402
import launch_scale_extension as launcher  # noqa: E402
import render_scale_extension as renderer  # noqa: E402


def task(
    task_id: str,
    *,
    event: str,
    series: str = "",
    template: str,
    settlement: bool = False,
    market: float = 0.4,
) -> dict:
    return {
        "task_id": task_id,
        "event_id": event,
        "series_id": series,
        "template_key": template,
        "decision_ts": "2026-03-01T00:00:00Z",
        "settlement_yes": settlement,
        "market_yes_prob": market,
    }


def fixtures() -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    registered = {
        "train": [task("opened-train", event="opened-event", template="opened-train")],
        "dev": [
            task(
                "opened-dev",
                event="opened-dev-event",
                series="opened-series",
                template="opened-dev",
            )
        ],
        "test": [
            task("opened-test", event="opened-test-event", template="opened-template")
        ],
    }
    candidates = {
        "train": [
            task(
                "candidate-train",
                event="candidate-train-event",
                template="candidate-train",
            )
        ],
        "dev": [
            task(
                "candidate-dev",
                event="candidate-dev-event",
                template="candidate-dev",
            )
        ],
        "test": [
            copy.deepcopy(registered["test"][0]),
            task("shares-opened-event", event="opened-event", template="fresh-a"),
            task(
                "shares-opened-series",
                event="fresh-event-b",
                series="opened-series",
                template="fresh-b",
            ),
            task(
                "shares-opened-template",
                event="fresh-event-c",
                template="opened-template",
            ),
            task(
                "shares-candidate-train",
                event="candidate-train-event",
                template="fresh-d",
            ),
            task(
                "shares-candidate-dev",
                event="candidate-dev-event",
                template="fresh-e",
            ),
            task("keep-z", event="clean-z", template="clean-z"),
            task("keep-a", event="clean-a", template="clean-a"),
        ],
    }
    return candidates, registered


def test_scale_holdout_excludes_every_previously_exposed_family() -> None:
    candidates, registered = fixtures()

    selected = builder.select_scale_holdout(candidates, registered)

    assert {row["task_id"] for row in selected} == {"keep-a", "keep-z"}
    exposed = [row for split in ("train", "dev", "test") for row in registered[split]]
    assert not builder.union_keys(selected) & builder.union_keys(exposed)


def test_scale_holdout_order_is_outcome_blind_and_input_order_independent() -> None:
    candidates, registered = fixtures()
    changed = copy.deepcopy(candidates)
    changed["test"].reverse()
    for row in changed["test"]:
        row["settlement_yes"] = not row["settlement_yes"]
        row["market_yes_prob"] = 1.0 - row["market_yes_prob"]

    first = builder.select_scale_holdout(candidates, registered)
    second = builder.select_scale_holdout(changed, registered)

    assert [row["task_id"] for row in first] == [row["task_id"] for row in second]


def test_public_holdout_description_does_not_summarize_labels_or_scores() -> None:
    candidates, registered = fixtures()

    description = builder.public_description(
        builder.select_scale_holdout(candidates, registered)
    )

    assert description["outcomes_summarized"] is False
    assert "settlement_yes" not in description
    assert "yes_rate" not in description
    assert "market_brier" not in description


def test_five_draw_aggregation_is_strict() -> None:
    assert evaluator.aggregate_draws([0.1, 0.2, 0.3, 0.4, 0.5]) == pytest.approx(0.3)
    assert evaluator.aggregate_draws([0.1, 0.2, None, 0.4, 0.5]) is None
    with pytest.raises(AssertionError, match="expected 5 draws"):
        evaluator.aggregate_draws([0.1])
    with pytest.raises(AssertionError, match=r"outside \[0, 1\]"):
        evaluator.aggregate_draws([0.1, 0.2, 0.3, 0.4, 1.1])


def test_endpoint_summary_counts_draw_and_complete_task_parse_coverage() -> None:
    draws = [
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [None, 0.3, 0.3, 0.3, 0.3],
    ]
    rows = [{"settlement_yes": False}, {"settlement_yes": True}]

    summary = evaluator.endpoint_summary([0.2, None], draws, rows)

    assert summary["brier"] == pytest.approx(0.52)
    assert summary["complete_task_parse_coverage"] == pytest.approx(0.5)
    assert summary["draw_parse_coverage"] == pytest.approx(0.9)
    assert summary["draws_per_task"] == 5.0


def test_adapter_specs_preserve_colons_in_paths() -> None:
    specs = evaluator.parse_adapter_specs(
        "42:path=/tmp/debug_01:02/step;43:job=101;44:job=102"
    )

    assert specs[42] == "path=/tmp/debug_01:02/step"
    assert specs[43] == "job=101"


def test_non_thinking_template_is_explicit() -> None:
    class Tokenizer:
        def apply_chat_template(self, messages: list[dict], **kwargs: object) -> str:
            assert messages == [{"role": "user", "content": "forecast"}]
            assert kwargs["tokenize"] is False
            assert kwargs["add_generation_prompt"] is True
            assert kwargs["enable_thinking"] is False
            return "rendered"

    assert (
        evaluator.format_prompt(Tokenizer(), "forecast", "auto_no_think") == "rendered"
    )


def test_scale_finalizer_requires_the_exact_model_roster() -> None:
    specs = renderer.parse_evaluation_specs("qwen3_4b:100;qwen3_8b:101;qwen3_14b:102")

    assert specs == {
        "qwen3_4b": "100",
        "qwen3_8b": "101",
        "qwen3_14b": "102",
    }
    with pytest.raises(AssertionError, match="requires"):
        renderer.parse_evaluation_specs("qwen3_8b:101;qwen3_14b:102")


def test_scale_finalizer_averages_seed_losses_not_probabilities() -> None:
    rows = [
        {
            "settlement_yes": False,
            "forecasts": {"seed_42": 0.1, "seed_43": 0.2, "seed_44": 0.3},
        }
    ]

    assert renderer.trained_loss_vector(rows) == [
        pytest.approx((0.01 + 0.04 + 0.09) / 3)
    ]


def test_model_cache_validator_requires_every_indexed_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_id = "Test/Model"
    snapshot = tmp_path / "models--Test--Model" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors.index.json").write_text(
        '{"weight_map":{"a":"model-1.safetensors","b":"model-2.safetensors"}}',
        encoding="utf-8",
    )
    (snapshot / "model-1.safetensors").write_bytes(b"one")
    monkeypatch.setattr(launcher, "HF_HUB", tmp_path)

    with pytest.raises(SystemExit, match="missing or incomplete"):
        launcher.validate_model_cache(model_id)

    (snapshot / "model-2.safetensors").write_bytes(b"two")
    launcher.validate_model_cache(model_id)


def test_launcher_and_training_script_lock_the_registered_scale_recipe() -> None:
    args = Namespace(
        partition="all",
        canary_walltime="06:00:00",
        exclude="node206",
        vllm_ratio=0.78,
    )
    command = launcher.canary_command(args)
    export = next(value for value in command if value.startswith("--export="))
    training = (SCRIPTS / "polymarket_rl_scale_extension.sh").read_text(
        encoding="utf-8"
    )

    assert "--gres=gpu:a6000:2" in command
    assert "MODEL=Qwen/Qwen3-14B" in export
    assert "MAX_TRAIN=160" in export
    assert launcher.EXISTING_MODELS["qwen3_4b"]["template"] == "biased_news"
    assert launcher.EXISTING_MODELS["qwen3_8b"]["template"] == "auto_no_think"
    assert "--learning_rate 0.000001" in training
    assert "--lora_rank 32" in training
    assert '--max-train "$MAX_TRAIN"' in training
    assert "--eval_temperature 0.7" in training
    assert "--eval_top_p 0.8" in training
    assert "--eval_top_k 20" in training
    assert "--eval_n 5" in training
    assert "--eval_temperature 0 " not in training
