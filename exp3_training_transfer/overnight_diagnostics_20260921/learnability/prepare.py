#!/usr/bin/env python3
"""Freeze presentation-matched SFT diagnostics from existing training data.

Preparation does not authorize or launch training. gate.json records the
component-evaluation evidence required before submission.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_frozen(path, payload):
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != content:
        raise RuntimeError(f"Refusing to overwrite frozen file: {path}")
    path.write_text(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=REPO / "exp3_training_transfer/coin_city_structure_selection/data/causal/train")
    args = parser.parse_args()
    from datasets import load_from_disk
    source = load_from_disk(str(args.source))["train"]
    rows = []
    for row in source:
        ref = json.loads(row["reference"])
        assert ref["domain"] == "coin_city" and ref["label_kind"] == "semantic"
        assert ref["cue"] == "correct" and ref["mode"] == "causal"
        forecasts = json.loads(ref["gold"])["forecasts"]
        assert len(forecasts) == 10
        assert max(abs(a - b) for a, b in zip(forecasts, ref["truth_targets"])) <= 0.000501
        rows.append({"task_id": ref["task_id"], "input": row["input"], "completion": ref["gold"]})
    assert len(rows) == 4800
    data = ROOT / "forecast_train.jsonl"
    content = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    if data.exists() and data.read_text() != content:
        raise RuntimeError("Refusing to change frozen training examples")
    data.write_text(content)
    configs = []
    for name, lr in (("forecast_lr1e6", 1e-6), ("forecast_lr2e5", 2e-5)):
        config = {
            "name": name, "data": str(data), "data_sha256": sha(data),
            "source_dataset": str(args.source),
            "source_files": {str(p.relative_to(args.source)): sha(p) for p in sorted(args.source.rglob("*")) if p.is_file()},
            "model": str(REPO / ".runtime/hf_home/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"),
            "seed": 42, "unique_prompts": 4800, "epochs": 1, "batch_size": 16,
            "learning_rate": lr, "lora_rank": 32, "lora_alpha": 64,
            "target_modules": MODULES, "max_length": 3072,
            "template": "auto_no_think", "loss": "completion_only_per_sequence_mean",
            "scheduler": "constant", "max_grad_norm": 1.0,
            "comparison": "same 4800 prompt presentations; SFT 300 updates vs RL 2400 updates and eight sampled completions per prompt",
        }
        path = ROOT / f"{name}.json"
        write_frozen(path, config)
        configs.append(str(path))
    print(json.dumps({"prepared_not_launched": configs, "rows": len(rows), "data_sha256": sha(data)}, indent=2))


if __name__ == "__main__":
    main()
