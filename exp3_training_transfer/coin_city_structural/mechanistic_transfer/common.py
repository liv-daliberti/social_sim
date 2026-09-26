"""Shared, fail-closed utilities for the frozen Exp3 mechanism study."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
COIN_ROOT = HERE.parent
REPO_ROOT = COIN_ROOT.parents[1]
DATA_DIR = HERE / "data"
RUNS_DIR = HERE / "runs"
LOGS_DIR = HERE / "logs"
PROTOCOL_DIR = HERE / "protocol"

STUDY = "coin_city_exp3_reference_transplant_qwen8_v1"
MODEL = "Qwen/Qwen3-8B"
MODEL_COMMIT = "b968826d9c46dd6066d109eabc6255188de91218"
TRAINING_SEEDS = (42, 43, 44)
ARMS = ("matched", "prior")
K_VALUES = (0, 8)
VARIANTS = ("original", "transplant")
ANCHORS = (
    "reference_1_end",
    "reference_2_end",
    "target_background_end",
    "target_evidence_end",
    "query_end",
    "final_prompt",
)
PRIMARY_ANCHOR = "query_end"
EPISODE_COUNT = 128
TEST_EPISODES = 32
DEV_EPISODES = EPISODE_COUNT - TEST_EPISODES
SEED_BASE = 93_000_000
REFERENCE_LENGTH = 14
TARGET_LENGTH = 8
MAX_INPUT_TOKENS = 2304

# These are the six canonical Qwen3-8B endpoint jobs from the registered campaign.
ADAPTER_JOBS = {
    42: {"matched": "30730378", "prior": "30730381"},
    43: {"matched": "30730379", "prior": "30730382"},
    44: {"matched": "30730380", "prior": "30730383"},
}
ARM_REPORT_NAMES = {"matched": "causal", "prior": "population_prior"}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(encoded)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def impulse_kernel(parameters: dict[str, float]) -> np.ndarray:
    """Per-unit response at horizons one and three for a zero-state impulse."""
    gain = float(parameters["gain"])
    rho = float(parameters["rho"])
    phi = float(parameters["phi"])
    coupling = float(parameters["lambda"])
    h1 = coupling * gain
    h3 = coupling * gain * (phi * phi + phi * rho + rho * rho)
    return np.asarray((h1, h3), dtype=float)


def persistence_ratio(parameters: dict[str, float]) -> float:
    kernel = impulse_kernel(parameters)
    return float(kernel[1] / kernel[0])


def resolve_adapter(seed: int, arm: str) -> Path:
    if seed not in ADAPTER_JOBS or arm not in ARMS:
        raise ValueError(f"unregistered endpoint seed={seed}, arm={arm}")
    report_arm = ARM_REPORT_NAMES[arm]
    job_id = ADAPTER_JOBS[seed][arm]
    pattern = f"{report_arm}_qwen3_8b_s{seed}_*_j{job_id}"
    reports = sorted(path for path in (COIN_ROOT / "reports").glob(pattern) if path.is_dir())
    if len(reports) != 1:
        raise RuntimeError(f"expected one canonical report for {pattern}, found {len(reports)}")
    adapters = sorted(reports[0].glob("**/saved_models/step_00301"))
    if len(adapters) != 1:
        raise RuntimeError(f"expected one step_00301 under {reports[0]}, found {len(adapters)}")
    adapter = adapters[0]
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter / name).is_file():
            raise FileNotFoundError(adapter / name)
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("base_model_name_or_path") != MODEL:
        raise ValueError(f"wrong base model in {adapter}")
    return adapter.resolve()


def adapter_freeze(seed: int, arm: str) -> dict[str, Any]:
    path = resolve_adapter(seed, arm)
    return {
        "seed": seed,
        "arm": arm,
        "job_id": ADAPTER_JOBS[seed][arm],
        "path": str(path),
        "adapter_config_sha256": file_sha256(path / "adapter_config.json"),
        "adapter_weights_sha256": file_sha256(path / "adapter_model.safetensors"),
    }


def load_frozen_tasks(data_dir: Path = DATA_DIR) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tasks_path = data_dir / "tasks.jsonl"
    manifest_path = data_dir / "task_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("study") != STUDY or manifest.get("status") != "tasks_frozen":
        raise ValueError("unexpected or unfrozen task manifest")
    if file_sha256(tasks_path) != manifest["tasks_sha256"]:
        raise ValueError("frozen task hash mismatch")
    rows = read_jsonl(tasks_path)
    if len(rows) != manifest["record_count"]:
        raise ValueError("frozen task count mismatch")
    return rows, manifest


def paired_seed_bootstrap(
    values: dict[int, np.ndarray], *, repetitions: int = 10_000, seed: int = 20260825
) -> dict[str, Any]:
    """Resample training seeds first, then paired episode effects within seed."""
    if tuple(sorted(values)) != TRAINING_SEEDS:
        raise ValueError(f"requires all training seeds, observed {sorted(values)}")
    rng = np.random.default_rng(seed)
    seed_ids = np.asarray(TRAINING_SEEDS)
    seed_means = {item: float(np.mean(values[item])) for item in TRAINING_SEEDS}
    draws = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        selected = rng.choice(seed_ids, len(seed_ids), replace=True)
        means = []
        for selected_seed in selected:
            episode_values = np.asarray(values[int(selected_seed)], dtype=float)
            means.append(float(np.mean(rng.choice(episode_values, len(episode_values), replace=True))))
        draws[index] = float(np.mean(means))
    low, high = np.quantile(draws, (0.025, 0.975))
    return {
        "estimate": float(np.mean(list(seed_means.values()))),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "seed_estimates": {str(key): value for key, value in seed_means.items()},
        "all_seed_direction_positive": all(value > 0.0 for value in seed_means.values()),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
    }
