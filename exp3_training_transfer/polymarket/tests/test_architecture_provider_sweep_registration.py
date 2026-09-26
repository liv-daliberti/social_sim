#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path


POLY = Path(__file__).resolve().parents[1]
RUNS = POLY / "runs"
REGISTRATION = (
    RUNS / "exp4_architecture_provider_sweep_registration_20260825T203304Z.json"
)
GATE_AMENDMENT = (
    RUNS
    / "exp4_architecture_provider_sweep_canary_gate_amendment_20260825T205750Z.json"
)
SCHEDULER_CORRECTION = (
    RUNS / "exp4_architecture_provider_sweep_scheduler_correction_20260825T210434Z.json"
)
HOSTED_RESOURCE = RUNS / (
    "exp4_architecture_provider_sweep_"
    "hosted_resource_amendment_20260825T221237Z.json"
)
WRAPPER = POLY / "scripts" / "polymarket_rl_architecture_extension.sh"
HOSTED_RUNNER = POLY / "scripts" / "evaluate_hosted_scale_holdout.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_model_and_provider_rosters_are_pinned_before_holdout() -> None:
    registration = load(REGISTRATION)

    assert registration["classification"] == "post_hoc_descriptive_extension"
    assert registration["status"] == "registered"
    assert registration["local_models"]["qwen3_1_7b"]["checkpoint"] == "Qwen/Qwen3-1.7B"
    assert registration["local_models"]["qwen3_1_7b"]["revision"] == (
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    )
    assert (
        registration["local_models"]["llama3_2_3b"]["checkpoint"]
        == "unsloth/Llama-3.2-3B-Instruct"
    )
    assert registration["local_models"]["llama3_2_3b"]["revision"] == (
        "006f5dcd1393c3add266de40994ba96225e9689d"
    )
    assert registration["hosted_models"] == [
        "claude-opus-4-8",
        "gpt-5.6-sol",
        "FW-Kimi-K3",
        "DeepSeek-V4-Pro",
    ]
    assert registration["hosted_reporting"]["holdout_calls_started"] is False
    protocol = POLY / "ARCHITECTURE_PROVIDER_SWEEP_PROTOCOL.md"
    assert sha256(protocol) == registration["protocol"]["sha256"]


def test_canary_gate_amendment_is_development_only_and_fail_closed() -> None:
    registration = load(REGISTRATION)
    amendment = load(GATE_AMENDMENT)

    assert amendment["timing_disclosure"].endswith("or any hosted-model holdout call.")
    assert amendment["trigger"]["development_tasks"] == 512
    assert amendment["trigger"]["draws"] == 2560
    assert amendment["trigger"]["parseable_draws_under_frozen_parser"] == 2555
    assert amendment["trigger"]["tasks_with_at_least_one_parseable_draw"] == 512
    assert amendment["amended_gate"]["required_parse_coverage"] == 0.99
    assert amendment["amended_gate"]["required_task_coverage"] == 1.0
    assert amendment["amended_gate"]["holdout_incomplete_draw_policy_changed"] is False
    assert amendment["amended_gate"]["holdout_policy"] == (
        registration["locked_evaluation"]["decoding"]["incomplete_draw_policy"]
    )
    assert amendment["wrapper"]["amended_sha256"] == sha256(WRAPPER)

    wrapper = WRAPPER.read_text(encoding="utf-8")
    assert "from forecast_scoring import parse_yes_probability" in wrapper
    assert "coverage < 0.99" in wrapper
    assert "covered_tasks != len(rows)" in wrapper


def test_scheduler_correction_preserves_registered_job_graph() -> None:
    correction = load(SCHEDULER_CORRECTION)

    scheduler = correction["scheduler_correction"]
    assert scheduler["account"] == "mltheory"
    assert scheduler["corrected_partition"] == "all"
    assert scheduler["scientific_settings_changed"] is False
    assert set(scheduler["preserved"]) == {
        "job_ids",
        "model_checkpoints",
        "seeds",
        "dependencies",
        "walltimes",
        "resources",
        "scientific_settings",
    }
    qwen_jobs = correction["qwen3_1_7b"]["full_training_jobs"]
    llama_jobs = correction["llama3_2_3b"]["preserved_full_training_jobs"]
    assert {entry["job_id"] for entry in qwen_jobs.values()} == {
        "30890065",
        "30890108",
        "30890109",
    }
    assert {entry["job_id"] for entry in llama_jobs.values()} == {
        "30890112",
        "30890113",
        "30890114",
    }
    assert correction["hosted_holdout_calls_started"] is False


def test_latest_hosted_runner_hash_is_registered_before_holdout() -> None:
    amendment = load(HOSTED_RESOURCE)

    assert amendment["software"]["runner_sha256"] == sha256(HOSTED_RUNNER)
    assert amendment["second_failed_canary_archive"]["successful_model_outputs"] == 0
    assert amendment["frozen_scientific_inputs"]["holdout_calls_started"] is False
    assert amendment["authorization"]["external_prompt_transmission_authorized"]
    assert amendment["frozen_scientific_inputs"]["prompt_text_changed"] is False


def test_readme_exposes_current_review_interface_and_invariants() -> None:
    readme = (POLY / "README.md").read_text(encoding="utf-8")

    for checkpoint in ("Qwen3-1.7B", "Llama-3.2-3B"):
        assert checkpoint in readme
    for provider in (
        "claude-opus-4-8",
        "gpt-5.6-sol",
        "FW-Kimi-K3",
        "DeepSeek-V4-Pro",
    ):
        assert provider in readme
