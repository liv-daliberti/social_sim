#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import evaluate_qwen32_holdout as evaluator  # noqa: E402
import launch_qwen32_extension as launcher  # noqa: E402
import render_qwen32_extension as renderer  # noqa: E402


def test_qwen32_canary_is_locked_to_mltheory_all_and_four_a6000s() -> None:
    args = Namespace(
        account="mltheory",
        partition="all",
        submit_account="allcs",
        submit_partition="cs",
        submit_qos="medium",
        canary_walltime="04:00:00",
        exclude="node206",
    )

    command = launcher.canary_command(args)
    export = next(value for value in command if value.startswith("--export="))

    assert "--account=allcs" in command
    assert "--partition=cs" in command
    assert "--qos=medium" in command
    assert "--gres=gpu:a6000:4" in command
    assert "MODEL=Qwen/Qwen3-32B" in export
    assert "SEED=42" in export
    assert "MAX_TRAIN=160" in export


def test_qwen32_trainer_preserves_scientific_recipe_and_global_batches() -> None:
    training = (SCRIPTS / "polymarket_rl_qwen32_extension.sh").read_text(
        encoding="utf-8"
    )

    for required in (
        "--gpus 4",
        "--num_gpus_per_actor 2",
        "--zero-stage 3",
        "--rollout_batch_size 16",
        "--rollout_batch_size_per_device 8",
        "--pi_buffer_maxlen_per_device 64",
        "--lora_rank 32",
        "--lora_alpha 64",
        "--lora_sync_only",
        "--disable-custom-all-reduce",
        "--gradient-checkpointing-use-reentrant",
        "--no-use_fused_lm_head",
        "--learning_rate 0.000001",
        "--temperature 1.3",
        "--eval_temperature 0.7",
        "--eval_top_p 0.8",
        "--eval_top_k 20",
        "--eval_n 5",
    ):
        assert required in training
    assert "Qwen/Qwen3-30B-A3B" not in training
    assert "--eval_temperature 0 " not in training


def test_qwen32_evaluator_rejects_identity_or_template_drift() -> None:
    evaluator.require_locked_identity(
        [
            "evaluate_qwen32_holdout.py",
            "--model-key",
            "qwen3_32b",
            "--model",
            "Qwen/Qwen3-32B",
            "--template",
            "auto_no_think",
        ]
    )
    with pytest.raises(SystemExit, match="unregistered model"):
        evaluator.require_locked_identity(
            [
                "evaluate_qwen32_holdout.py",
                "--model-key",
                "qwen3_30b_a3b",
                "--model",
                "Qwen/Qwen3-30B-A3B",
                "--template",
                "auto_no_think",
            ]
        )

def test_qwen32_evaluator_resolves_registered_report_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reports = tmp_path / "reports"
    adapter = (
        reports
        / "qwen32_train_qwen3_32b_market_20260827_025318_j30913531"
        / "debug_0827T03:11:10"
        / "saved_models"
        / "step_00301"
    )
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(evaluator, "REPORTS", reports)

    assert (
        evaluator.resolve_job_adapter("qwen3_32b", "30913531")
        == adapter.resolve()
    )



def test_qwen32_finalizer_requires_exact_four_model_roster() -> None:
    specs = renderer.parse_evaluation_specs(
        "qwen3_4b:100;qwen3_8b:101;qwen3_14b:102;qwen3_32b:103"
    )

    assert tuple(specs) == renderer.MODEL_KEYS
    with pytest.raises(AssertionError, match="requires"):
        renderer.parse_evaluation_specs("qwen3_4b:100;qwen3_8b:101;qwen3_14b:102")


def test_slurm_fields_falls_back_to_accounting_for_purged_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_check_output(command: list[str], *, text: bool) -> str:
        if command[0] == "scontrol":
            raise launcher.subprocess.CalledProcessError(1, command)
        assert command[0] == "sacct"
        return "30870546|COMPLETED|2026-08-25T14:04:10\n"

    monkeypatch.setattr(launcher.subprocess, "check_output", fake_check_output)
    assert launcher.slurm_fields("30870546") == {
        "JobState": "COMPLETED",
        "Reason": "AccountingOnly",
        "Dependency": "(purged)",
        "StartTime": "2026-08-25T14:04:10",
    }


def test_parent_jobs_remain_the_registered_ids() -> None:
    assert launcher.PARENT_EVALUATIONS == {
        "qwen3_4b": "30870546",
        "qwen3_8b": "30870547",
        "qwen3_14b": "30870548",
    }
    assert launcher.PARENT_FINALIZER == "30870549"


def test_opening_amendment_requires_exact_parent_roster(tmp_path: Path) -> None:
    amendment = {
        "kind": "holdout_opening_amendment",
        "protocol_version": launcher.PROTOCOL_VERSION,
        "parent_evaluation_jobs": list(launcher.PARENT_EVALUATIONS.values()),
        "parent_finalizer_job": launcher.PARENT_FINALIZER,
        "holdout_will_open_before_qwen3_32b_full_training_completes": True,
        "qwen3_32b_will_run_regardless_of_parent_results": True,
        "qwen3_32b_training_recipe_remains_frozen": True,
    }
    path = tmp_path / "opening.json"
    path.write_text(json.dumps(amendment), encoding="utf-8")

    assert launcher.validate_opening_amendment(path) == amendment
    amendment["parent_evaluation_jobs"][-1] = "wrong"
    path.write_text(json.dumps(amendment), encoding="utf-8")
    with pytest.raises(SystemExit, match="roster drifted"):
        launcher.validate_opening_amendment(path)


def test_amended_registration_accepts_original_canary_binding(tmp_path: Path) -> None:
    original_registration = tmp_path / "registration.json"
    original_registration.write_text("{}\n", encoding="utf-8")
    original_hash = hashlib.sha256(original_registration.read_bytes()).hexdigest()
    canary = {
        "kind": "canary",
        "job_id": "123",
        "registration_ledger": str(original_registration),
        "registration_ledger_sha256": original_hash,
        "dry_run": False,
    }
    canary_path = tmp_path / "canary.json"
    canary_path.write_text(json.dumps(canary), encoding="utf-8")
    amended_registration_path = tmp_path / "amended_registration.json"
    amended_registration_path.write_text("{}\n", encoding="utf-8")
    amended_registration = {"canary_registration_ledger": str(original_registration)}

    assert (
        launcher.validate_canary_ledger(
            canary_path,
            amended_registration_path,
            "123",
            amended_registration,
        )
        == canary
    )


def test_amended_registration_accepts_replacement_canary_binding(
    tmp_path: Path,
) -> None:
    original_registration = tmp_path / "original_registration.json"
    original_registration.write_text("{}\n", encoding="utf-8")
    replacement_registration = tmp_path / "replacement_registration.json"
    replacement_registration.write_text("{}\n", encoding="utf-8")
    replacement_hash = hashlib.sha256(replacement_registration.read_bytes()).hexdigest()
    canary = {
        "kind": "canary",
        "job_id": "456",
        "registration_ledger": str(replacement_registration),
        "registration_ledger_sha256": replacement_hash,
        "dry_run": False,
    }
    canary_path = tmp_path / "canary.json"
    canary_path.write_text(json.dumps(canary), encoding="utf-8")
    amended_registration = {"canary_registration_ledger": str(original_registration)}

    assert (
        launcher.validate_canary_ledger(
            canary_path,
            replacement_registration,
            "456",
            amended_registration,
        )
        == canary
    )


def test_zero3_compatibility_amendment_is_execution_only(tmp_path: Path) -> None:
    amendment = {
        "kind": "zero3_compatibility_amendment",
        "protocol_version": launcher.PROTOCOL_VERSION,
        "failed_canary_job_id": "30885396",
        "execution_only_changes": [
            "--no-use_fused_lm_head",
            "--lora_sync_only",
        ],
        "scientific_hyperparameters_changed": False,
        "corrected_trainer_sha256": launcher.sha256(launcher.TRAIN),
        "corrected_launcher_sha256": launcher.sha256(Path(launcher.__file__)),
        "replacement_canary": {
            "seed": 42,
            "max_train": 160,
            "eval_steps": 10,
            "performance_threshold": False,
        },
    }
    path = tmp_path / "compatibility.json"
    path.write_text(json.dumps(amendment), encoding="utf-8")
    assert launcher.validate_compatibility_amendment(path) == amendment

    amendment["execution_only_changes"].append("--learning_rate=changed")
    path.write_text(json.dumps(amendment), encoding="utf-8")
    with pytest.raises(SystemExit, match="flag set drifted"):
        launcher.validate_compatibility_amendment(path)


def test_allreduce_compatibility_amendment_is_execution_only(tmp_path: Path) -> None:
    superseded = tmp_path / "zero3.json"
    superseded.write_text("{}\n", encoding="utf-8")

    amendment = {
        "kind": "vllm_allreduce_compatibility_amendment",
        "protocol_version": launcher.PROTOCOL_VERSION,
        "failed_canary_job_id": "30892281",
        "execution_only_changes_added": ["--disable-custom-all-reduce"],
        "execution_only_changes_retained": [
            "--no-use_fused_lm_head",
            "--lora_sync_only",
        ],
        "supersedes_compatibility_amendment": str(superseded),
        "supersedes_compatibility_amendment_sha256": launcher.sha256(superseded),
        "scientific_hyperparameters_changed": False,
        "corrected_trainer_sha256": launcher.sha256(launcher.TRAIN),
        "corrected_shared_trainer_sha256": launcher.sha256(launcher.SHARED_TRAINER),
        "corrected_launcher_sha256": launcher.sha256(Path(launcher.__file__)),
        "replacement_canary": {
            "seed": 42,
            "max_train": 160,
            "eval_steps": 10,
            "performance_threshold": False,
        },
    }
    path = tmp_path / "allreduce.json"
    path.write_text(json.dumps(amendment), encoding="utf-8")
    assert launcher.validate_compatibility_amendment(path) == amendment

    amendment["corrected_shared_trainer_sha256"] = "wrong"
    path.write_text(json.dumps(amendment), encoding="utf-8")
    with pytest.raises(SystemExit, match="shared trainer hash drifted"):
        launcher.validate_compatibility_amendment(path)


def test_reentrant_checkpoint_amendment_is_execution_only(tmp_path: Path) -> None:
    superseded = tmp_path / "allreduce.json"
    superseded.write_text("{}\n", encoding="utf-8")
    amendment = {
        "kind": "zero3_reentrant_checkpoint_compatibility_amendment",
        "protocol_version": launcher.PROTOCOL_VERSION,
        "failed_canary_job_id": "30908504",
        "execution_only_changes_added": ["--gradient-checkpointing-use-reentrant"],
        "execution_only_changes_retained": [
            "--no-use_fused_lm_head",
            "--lora_sync_only",
            "--disable-custom-all-reduce",
        ],
        "supersedes_compatibility_amendment": str(superseded),
        "supersedes_compatibility_amendment_sha256": launcher.sha256(superseded),
        "scientific_hyperparameters_changed": False,
        "corrected_trainer_sha256": launcher.sha256(launcher.TRAIN),
        "corrected_shared_trainer_sha256": launcher.sha256(launcher.SHARED_TRAINER),
        "corrected_launcher_sha256": launcher.sha256(Path(launcher.__file__)),
        "replacement_canary": {
            "seed": 42,
            "max_train": 160,
            "eval_steps": 10,
            "performance_threshold": False,
        },
    }
    path = tmp_path / "reentrant.json"
    path.write_text(json.dumps(amendment), encoding="utf-8")
    assert launcher.validate_compatibility_amendment(path) == amendment

    amendment["execution_only_changes_added"] = ["--no-gradient-checkpointing"]
    path.write_text(json.dumps(amendment), encoding="utf-8")
    with pytest.raises(SystemExit, match="flag set drifted"):
        launcher.validate_compatibility_amendment(path)

def test_canary_probability_parser_accepts_one_json_fence() -> None:
    fence = chr(96) * 3
    assert launcher.parse_probability('{"yes_prob": 0.37}') == pytest.approx(0.37)
    assert launcher.parse_probability(
        fence + 'json\n{"yes_prob": 0.37}\n' + fence
    ) == pytest.approx(0.37)
    assert launcher.parse_probability(
        'prefix\n' + fence + 'json\n{"yes_prob": 0.37}\n' + fence
    ) is None


def test_adapter_artifact_contract_amendment_is_execution_only(
    tmp_path: Path,
) -> None:
    superseded = tmp_path / "reentrant.json"
    superseded.write_text("{}\n", encoding="utf-8")
    source = tmp_path / "adapter_model.bin"
    source.write_bytes(b"consolidated-bin")
    converted = tmp_path / "adapter_model.safetensors"
    converted.write_bytes(b"verified-safetensors")
    amendment = {
        "kind": "zero3_adapter_artifact_contract_compatibility_amendment",
        "protocol_version": launcher.PROTOCOL_VERSION,
        "completed_canary_job_id": "30910721",
        "failed_advance_job_id": "30910722",
        "execution_only_changes": [
            "write consolidated ZeRO-3 PEFT weights as adapter_model.safetensors while retaining adapter_model.bin",
            "strip one Markdown JSON fence before strict canary JSON parsing",
        ],
        "reuses_completed_canary_without_retraining": True,
        "corrected_oat_deepspeed_sha256": launcher.sha256(launcher.OAT_DEEPSPEED),
        "source_adapter_bin": str(source),
        "source_adapter_bin_sha256": launcher.sha256(source),
        "converted_adapter_safetensors": str(converted),
        "converted_adapter_safetensors_sha256": launcher.sha256(converted),
        "canary_parse_audit": {
            "draws": 2560,
            "strict_json_before_fence_normalization": 0,
            "single_fenced_json": 2560,
            "invalid_after_fence_normalization": 0,
        },
        "supersedes_compatibility_amendment": str(superseded),
        "supersedes_compatibility_amendment_sha256": launcher.sha256(superseded),
        "scientific_hyperparameters_changed": False,
        "corrected_trainer_sha256": launcher.sha256(launcher.TRAIN),
        "corrected_launcher_sha256": launcher.sha256(Path(launcher.__file__)),
    }
    amendment_path = tmp_path / "artifact_contract.json"
    amendment_path.write_text(json.dumps(amendment), encoding="utf-8")
    assert launcher.validate_compatibility_amendment(amendment_path) == amendment

    amendment["canary_parse_audit"]["invalid_after_fence_normalization"] = 1
    amendment_path.write_text(json.dumps(amendment), encoding="utf-8")
    with pytest.raises(SystemExit, match="parse audit drifted"):
        launcher.validate_compatibility_amendment(amendment_path)

