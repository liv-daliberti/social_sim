#!/usr/bin/env python3
"""Generate the fixed-window patch confirmation for one frozen checkpoint.

Every window, condition and episode comes from protocol.json, which was
committed before this script ran on any fresh episode. Generation and state
capture are imported from the registered patching module, so a number here
means the same thing as a registered number.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

# The registered module reads these at import time; the compute nodes have no
# route to huggingface.co.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import patch_symbol_label_activations as base  # noqa: E402
from freeze_fixed_window_confirmation import RUN_DIR  # noqa: E402

ARMS = ("abc_context", "abc_wrong_context")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen(run_dir: Path) -> tuple[dict, list[dict]]:
    protocol = json.loads((run_dir / "protocol.json").read_text())
    if protocol["status"] != "frozen_before_any_fresh_episode_generation":
        raise ValueError("protocol status drifted")
    tasks_path = run_dir / "tasks.jsonl"
    if file_sha256(tasks_path) != protocol["tasks_sha256"]:
        raise ValueError("tasks.jsonl does not match the frozen hash")
    for name, digest in protocol["code_sha256"].items():
        path = (HERE.parent / name) if name.startswith("engine/") else (HERE / name)
        if file_sha256(path) != digest:
            raise ValueError(f"{name} changed after the protocol was frozen")
    tasks = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
    return protocol, tasks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--device-map")
    ap.add_argument("--gpu-memory-gib", type=int)
    ap.add_argument("--stop-after-seconds", type=int, default=10**9)
    a = ap.parse_args()

    protocol, tasks = load_frozen(a.run_dir)
    spec = protocol["models"][a.model_key]
    n_layers = int(spec["transformer_layers"])
    mid, late = list(spec["mid_window"]), list(spec["late_window"])
    for window in (mid, late):
        if window != list(range(window[0], window[0] + 3)) or window[0] < 1 or window[-1] > n_layers - 1:
            raise ValueError(f"window {window} is not three causally effective layers")
    max_new_tokens = 160

    by_key = {(int(r["episode"]), int(r["c_cases"]), str(r["arm"])): r
              for r in tasks if r["arm"] in ARMS}
    episodes = sorted({k[0] for k in by_key})
    if len(episodes) != protocol["episodes"]["sealed_test_episodes"]:
        raise ValueError("episode count differs from the frozen protocol")
    episodes = [e for i, e in enumerate(episodes) if i % a.shards == a.shard]
    out_dir = a.run_dir / a.model_key
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"shard{a.shard:02d}of{a.shards:02d}.jsonl"
    records = [json.loads(l) for l in out.read_text().splitlines() if l.strip()] if out.exists() else []
    done = {(r["episode"], r["c_cases"], r["recipient_arm"], r["condition"]) for r in records}
    print(f"{a.model_key} shard {a.shard}/{a.shards}: {len(episodes)} episodes, "
          f"{len(records)} records already written", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    torch.manual_seed(base.PATCH_SEED)
    torch.cuda.manual_seed_all(base.PATCH_SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(spec["model_id"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"local_files_only": True, "torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    if a.device_map:
        kwargs["device_map"] = a.device_map
        if a.gpu_memory_gib is not None:
            kwargs["max_memory"] = {i: f"{a.gpu_memory_gib}GiB" for i in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(spec["model_id"], **kwargs)
    model.eval()
    if not a.device_map:
        model.to("cuda")
    observed = getattr(model.config, "_commit_hash", None)
    if observed != spec["model_commit"]:
        raise RuntimeError(f"checkpoint commit {observed} != frozen {spec['model_commit']}")
    if int(model.config.num_hidden_layers) != n_layers:
        raise RuntimeError("model depth differs from the frozen protocol")

    def flush() -> None:
        tmp = out.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
        tmp.replace(out)

    for episode in episodes:
        for depth in protocol["depths"]:
            pair = {arm: by_key[(episode, depth, arm)] for arm in ARMS}
            if all((episode, depth, arm, c) in done for arm in ARMS for c in protocol["conditions"]):
                continue
            states, positions, ids = {}, {}, {}
            for arm, task in pair.items():
                states[arm], positions[arm], ids[arm] = base.capture_city_c_states(
                    model=model, tokenizer=tokenizer, task=task)
            base.validate_matched_pair(correct_positions=positions[ARMS[0]],
                                       wrong_positions=positions[ARMS[1]],
                                       correct_ids=ids[ARMS[0]], wrong_ids=ids[ARMS[1]])
            for recipient, task in pair.items():
                donor = ARMS[1] if recipient == ARMS[0] else ARMS[0]
                jobs = {
                    "unpatched": (None, None, []),
                    "self_mid_window": (recipient, states[recipient], mid),
                    "cross_mid_window": (donor, states[donor], mid),
                    "cross_late_window": (donor, states[donor], late),
                    "cross_embedding_only": (donor, states[donor], [0]),
                }
                if list(jobs) != protocol["conditions"]:
                    raise ValueError("conditions differ from the frozen protocol")
                for condition, (source_arm, source, layers) in jobs.items():
                    if (episode, depth, recipient, condition) in done:
                        continue
                    row = base.generate_one(model=model, tokenizer=tokenizer, task=task,
                                            source_states=source, hidden_state_layers=layers,
                                            max_new_tokens=max_new_tokens)
                    row.update(study=protocol["study"], model_key=a.model_key,
                               episode=episode, c_cases=depth, sample_id=task["sample_id"],
                               recipient_arm=recipient, source_arm=source_arm,
                               condition=condition, patched_hidden_state_layers=layers,
                               target_strong=bool(task["target_strong"]),
                               recipient_cue_strong=bool(task["cue_strong"]))
                    records.append(row)
                    done.add((episode, depth, recipient, condition))
            del states
            torch.cuda.empty_cache()
            flush()
            elapsed = time.time() - started
            print(f"episode {episode} depth {depth}: {len(records)} records, "
                  f"{elapsed / 60:.1f} min", flush=True)
            if elapsed >= a.stop_after_seconds:
                print("walltime budget reached; rerun resumes from saved records", flush=True)
                return
    print("shard complete", flush=True)


if __name__ == "__main__":
    main()
