#!/usr/bin/env python3
"""Run the sealed bidirectional, same/wrong-episode activation intervention."""
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "mechanism_family"))

from common import (  # noqa: E402
    MODEL,
    MODEL_COMMIT,
    RUNS_DIR,
    STUDY,
    TRAINING_SEEDS,
    atomic_json,
    atomic_jsonl,
    file_sha256,
    load_frozen_tasks,
    resolve_adapter,
)
from output_contract import FORECAST_ARRAY_GBNF  # noqa: E402

CONDITIONS = (
    "prior_unpatched",
    "prior_self",
    "matched_same_to_prior",
    "matched_wrong_to_prior",
    "matched_unpatched",
    "matched_self",
    "prior_same_to_matched",
    "prior_wrong_to_matched",
)


def formatted_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def scenario_token_positions(tokenizer, task: dict[str, Any], rendered: str) -> list[int]:
    raw_start = rendered.find(task["prompt"])
    if raw_start < 0:
        raise ValueError("raw prompt is not embedded in chat rendering")
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    output = []
    for raw_end in task["patch_scenario_char_ends"]:
        boundary = raw_start + int(raw_end)
        candidates = [
            index for index, (start, end) in enumerate(offsets) if end > start and end <= boundary
        ]
        if not candidates:
            raise ValueError(f"cannot map scenario site for {task['sample_id']}")
        output.append(candidates[-1])
    if len(output) != 10 or len(set(output)) != 10:
        raise ValueError("scenario patch positions are not ten unique tokens")
    return output


def transformer_layers(model):
    candidates = (
        ("base_model", "model", "model", "layers"),
        ("model", "model", "layers"),
        ("model", "layers"),
    )
    for candidate in candidates:
        value = model
        try:
            for name in candidate:
                value = getattr(value, name)
        except AttributeError:
            continue
        if len(value) == int(model.config.num_hidden_layers):
            return value
    raise AttributeError("could not locate transformer layers")


def capture_states(model, encoded: dict[str, Any], state_layers: list[int], positions: list[int]):
    import torch

    # Capture from the same five-way generation prefill used by every intervention.
    # A batch-one forward can differ slightly from the expanded generation kernel,
    # which makes an otherwise valid self-patch fail the exact-fidelity control.
    layers = transformer_layers(model)
    captured: dict[int, Any] = {}
    handles = []
    try:
        for state_layer in state_layers:
            block_index = int(state_layer)

            def hook(_module, inputs, *, key=state_layer):
                hidden = inputs[0]
                if (
                    key not in captured
                    and hidden.ndim == 3
                    and hidden.shape[1] > max(positions)
                ):
                    captured[key] = hidden[:, positions, :].detach().cpu()
                return None

            handles.append(layers[block_index].register_forward_pre_hook(hook))
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        with torch.inference_mode():
            model.generate(
                **encoded,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                num_return_sequences=5,
                max_new_tokens=1,
                use_cache=True,
                pad_token_id=model.config.eos_token_id,
                eos_token_id=model.config.eos_token_id,
            )
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(state_layers):
        raise RuntimeError("failed to capture every registered patch layer")
    return captured


@contextmanager
def patch_hooks(model, donor: dict[int, Any] | None, positions: list[int]):
    if donor is None:
        yield
        return
    layers = transformer_layers(model)
    handles = []
    try:
        for state_layer, donor_values in donor.items():
            block_index = int(state_layer)
            if block_index >= len(layers):
                raise ValueError(f"state layer {state_layer} cannot be patched into a later block")

            def hook(_module, inputs, *, values=donor_values):
                hidden = inputs[0]
                # Patches apply only to the prefill. Cached one-token decode steps are untouched.
                if hidden.ndim != 3 or hidden.shape[1] <= max(positions):
                    return None
                replaced = hidden.clone()
                source = values.to(device=hidden.device, dtype=hidden.dtype)
                if source.ndim == 2:
                    source = source.unsqueeze(0).expand(hidden.shape[0], -1, -1)
                if source.shape[0] != hidden.shape[0]:
                    raise ValueError(
                        f"donor batch {source.shape[0]} does not match recipient batch "
                        f"{hidden.shape[0]}"
                    )
                replaced[:, positions, :] = source
                return (replaced, *inputs[1:])

            handles.append(layers[block_index].register_forward_pre_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def compile_grammar(tokenizer, vocab_size: int):
    import xgrammar

    info = xgrammar.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    return xgrammar.GrammarCompiler(info).compile_grammar(FORECAST_ARRAY_GBNF)


def generate_five(
    *, model, tokenizer, compiled_grammar, encoded, donor, positions, sampling_seed: int
) -> list[str]:
    import torch
    import xgrammar

    # This transformers version does not accept a per-call ``generator`` kwarg.
    # Reset the global CPU/CUDA RNGs immediately before every condition instead;
    # using the same registered seed preserves paired draws across conditions.
    torch.manual_seed(sampling_seed)
    torch.cuda.manual_seed_all(sampling_seed)
    processor = xgrammar.contrib.hf.LogitsProcessor(compiled_grammar)
    with patch_hooks(model, donor, positions), torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            num_return_sequences=5,
            max_new_tokens=192,
            logits_processor=[processor],
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_length = int(encoded["input_ids"].shape[1])
    return tokenizer.batch_decode(generated[:, prompt_length:], skip_special_tokens=True)


def condition_spec(condition: str, same: dict[str, dict[int, Any]], wrong: dict[str, dict[int, Any]]):
    mapping = {
        "prior_unpatched": ("prior", None),
        "prior_self": ("prior", same["prior"]),
        "matched_same_to_prior": ("prior", same["matched"]),
        "matched_wrong_to_prior": ("prior", wrong["matched"]),
        "matched_unpatched": ("matched", None),
        "matched_self": ("matched", same["matched"]),
        "prior_same_to_matched": ("matched", same["prior"]),
        "prior_wrong_to_matched": ("matched", wrong["prior"]),
    }
    return mapping[condition]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=TRAINING_SEEDS)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--limit", type=int, help="smoke-test reciprocal donor-pair count")
    parser.add_argument("--pair-shard", help="zero-based reciprocal donor-pair shard INDEX/TOTAL")
    args = parser.parse_args()
    run_dir = args.run_dir or RUNS_DIR / f"qwen3_8b_s{args.seed}"
    selection_path = RUNS_DIR / "common_probe" / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "development_selection_complete_test_unopened":
        raise ValueError("patch layer window was not frozen on development data")
    state_layers = [int(value) for value in selection["patch_state_layer_window"]]
    if len(state_layers) != 3 or max(state_layers) > 35:
        raise ValueError("invalid three-layer patch window")
    tasks, manifest = load_frozen_tasks()
    test_tasks = [row for row in tasks if row["split"] == "test" and row["variant"] == "original"]
    test_tasks.sort(key=lambda row: row["sample_id"])
    if len(test_tasks) != 64:
        raise ValueError("expected 32 sealed episodes at two evidence depths")
    pair_ids = sorted({row["donor_pair_id"] for row in test_tasks})
    shard_suffix = ""
    if args.pair_shard:
        index_text, total_text = args.pair_shard.split("/", 1)
        shard_index, shard_total = int(index_text), int(total_text)
        if shard_total < 1 or not 0 <= shard_index < shard_total:
            raise ValueError("invalid --pair-shard INDEX/TOTAL")
        selected_pairs = set(pair_ids[shard_index::shard_total])
        test_tasks = [row for row in test_tasks if row["donor_pair_id"] in selected_pairs]
        shard_suffix = f"_shard_{shard_index:02d}_of_{shard_total:02d}"
    elif args.limit is not None:
        selected_pairs = set(pair_ids[: args.limit])
        test_tasks = [row for row in test_tasks if row["donor_pair_id"] in selected_pairs]
    by_episode_k = {(row["episode_id"], row["k"]): row for row in test_tasks}
    for row in test_tasks:
        if (row["donor_episode_id"], row["k"]) not in by_episode_k:
            raise ValueError("wrong-episode donor missing from the same sealed run")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(
        base, str(resolve_adapter(args.seed, "matched")), adapter_name="matched", is_trainable=False
    )
    model.load_adapter(
        str(resolve_adapter(args.seed, "prior")), adapter_name="prior", is_trainable=False
    )
    model.eval().to("cuda")
    compiled_grammar = compile_grammar(tokenizer, int(model.config.vocab_size))

    rendered: dict[str, str] = {}
    positions: dict[str, list[int]] = {}
    encoded: dict[str, dict[str, Any]] = {}
    for task in test_tasks:
        sample_id = task["sample_id"]
        rendered[sample_id] = formatted_prompt(tokenizer, task["prompt"])
        positions[sample_id] = scenario_token_positions(tokenizer, task, rendered[sample_id])
        item = tokenizer(
            rendered[sample_id], add_special_tokens=False, return_tensors="pt"
        )
        encoded[sample_id] = {key: value.to("cuda") for key, value in item.items()}

    captures: dict[tuple[str, str], dict[int, Any]] = {}
    for arm in ("matched", "prior"):
        model.set_adapter(arm)
        for task_index, task in enumerate(test_tasks, start=1):
            sample_id = task["sample_id"]
            captures[(sample_id, arm)] = capture_states(
                model, encoded[sample_id], state_layers, positions[sample_id]
            )
            if task_index % 8 == 0:
                print(f"captured {arm} donors {task_index}/{len(test_tasks)}", flush=True)

    output_path = run_dir / (
        "patch_smoke.jsonl" if args.limit else f"patch_outputs{shard_suffix}.jsonl"
    )
    existing = []
    if output_path.exists():
        existing = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    completed = {(row["sample_id"], row["condition"]) for row in existing}
    records = list(existing)
    for task_index, task in enumerate(test_tasks):
        sample_id = task["sample_id"]
        wrong_task = by_episode_k[(task["donor_episode_id"], task["k"])]
        same = {arm: captures[(sample_id, arm)] for arm in ("matched", "prior")}
        wrong = {arm: captures[(wrong_task["sample_id"], arm)] for arm in ("matched", "prior")}
        sampling_seed = 20260819 + 2 * int(task["episode_index"]) + (1 if task["k"] == 8 else 0)
        for condition in CONDITIONS:
            if (sample_id, condition) in completed:
                continue
            recipient, donor = condition_spec(condition, same, wrong)
            model.set_adapter(recipient)
            outputs = generate_five(
                model=model,
                tokenizer=tokenizer,
                compiled_grammar=compiled_grammar,
                encoded=encoded[sample_id],
                donor=donor,
                positions=positions[sample_id],
                sampling_seed=sampling_seed,
            )
            records.append(
                {
                    "study": STUDY,
                    "seed": args.seed,
                    "sample_id": sample_id,
                    "episode_id": task["episode_id"],
                    "donor_episode_id": task["donor_episode_id"],
                    "k": task["k"],
                    "condition": condition,
                    "recipient_arm": recipient,
                    "sampling_seed": sampling_seed,
                    "state_layers": state_layers,
                    "outputs": outputs,
                }
            )
        records.sort(key=lambda row: (row["sample_id"], row["condition"]))
        atomic_jsonl(output_path, records)
        print(f"patched generations {task_index + 1}/{len(test_tasks)}", flush=True)
    expected = len(test_tasks) * len(CONDITIONS)
    if len(records) != expected:
        raise RuntimeError(f"expected {expected} completed conditions, observed {len(records)}")
    by_sample: dict[str, dict[str, list[str]]] = {}
    for record in records:
        by_sample.setdefault(record["sample_id"], {})[record["condition"]] = record["outputs"]
    for sample_id, sample_conditions in by_sample.items():
        for unpatched, self_patched in (
            ("prior_unpatched", "prior_self"),
            ("matched_unpatched", "matched_self"),
        ):
            left = [value.strip() for value in sample_conditions[unpatched]]
            right = [value.strip() for value in sample_conditions[self_patched]]
            if left != right:
                raise RuntimeError(
                    f"exact self-patch fidelity failed for {sample_id}: "
                    f"{unpatched} != {self_patched}"
                )
    receipt = {
        "study": STUDY,
        "status": "smoke_complete" if args.limit else "patching_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "model": MODEL,
        "model_commit": MODEL_COMMIT,
        "tasks_sha256": manifest["tasks_sha256"],
        "selection_sha256": file_sha256(selection_path),
        "state_layers": state_layers,
        "scenario_positions": 10,
        "test_tasks": len(test_tasks),
        "conditions": list(CONDITIONS),
        "draws_per_condition": 5,
        "output_path": str(output_path),
        "output_sha256": file_sha256(output_path),
    }
    atomic_json(
        run_dir / (
            "patch_smoke_manifest.json" if args.limit
            else f"patch_manifest{shard_suffix}.json"
        ), receipt
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
