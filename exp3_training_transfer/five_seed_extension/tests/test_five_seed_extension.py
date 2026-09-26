from __future__ import annotations

import importlib.util
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]


def load_launcher():
    spec = importlib.util.spec_from_file_location("five_seed_launch", HERE / "launch.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_training_roster() -> None:
    module = load_launcher()
    rows = module.training_records()
    assert len(rows) == 84
    assert {row["seed"] for row in rows} == {45, 46}
    assert len({(row["campaign"], row["identity"], row["seed"]) for row in rows}) == 84
    assert sum(row["campaign"].startswith("exp3_") for row in rows) == 70
    assert sum(row["campaign"].startswith("exp4_") for row in rows) == 14


def test_mechanism_probe_roster_is_48_endpoints() -> None:
    module = load_launcher()
    rows = module.training_records()
    probed = [row for row in rows if row["campaign"] in {
        "exp3_mechanism_current", "exp3_mechanism_large"
    }]
    assert len(probed) == 48
    assert all("structureless" not in row["identity"] for row in probed)


def test_large_and_qwen32_jobs_inherit_audited_gates() -> None:
    module = load_launcher()
    rows = module.training_records()
    large = [row for row in rows if row["campaign"] == "exp3_mechanism_large"]
    qwen32 = [row for row in rows if row["campaign"] == "exp4_qwen3_32b"]
    assert len(large) == 24 and len(qwen32) == 2
    assert all(f"--dependency=afterok:{module.LARGE_GATE_JOB}" in row["command"] for row in large)
    assert all(f"--dependency=afterok:{module.QWEN32_GATE_JOB}" in row["command"] for row in qwen32)


def test_five_seed_evaluator_has_exact_seed_set() -> None:
    evaluator = (HERE / "evaluate_exp4_five_seed.py").read_text()
    assert "SEEDS = (42, 43, 44, 45, 46)" in evaluator
    assert "TENSOR_PARALLEL_SIZE" in evaluator


def test_copied_zero3_wrappers_keep_compatibility_flags() -> None:
    for name in ("large_mechanism_rl_five_seed.sh", "polymarket_qwen32_five_seed.sh"):
        text = (HERE / name).read_text()
        assert "--no-use_fused_lm_head" in text
        assert "--lora_sync_only" in text
        assert "45|46" in text
    qwen32 = (HERE / "polymarket_qwen32_five_seed.sh").read_text()
    assert "--disable-custom-all-reduce" in qwen32
    assert "--gradient-checkpointing-use-reentrant" in qwen32
