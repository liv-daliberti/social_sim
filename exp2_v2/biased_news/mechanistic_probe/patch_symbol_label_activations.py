#!/usr/bin/env python3
"""Bidirectionally patch City C label states between correct and swapped prompts."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from symbol_relational_common import (
    ARMS,
    DEFAULT_RUN_DIR,
    LAST_CAUSALLY_EFFECTIVE_HIDDEN_LAYER,
    MODEL_COMMIT,
    MODEL_ID,
    PATCH_SEED,
    file_sha256,
    find_label_anchors,
    formatted_prompt,
    design_counts,
    load_tasks,
    run_spec,
    read_jsonl,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EVAL_DIR = ROOT / "eval"
import sys

sys.path.insert(0, str(EVAL_DIR))
from run_frozen_task_shard import _parse_reply, _strip_reasoning  # noqa: E402


def model_input_device(model):
    return model.get_input_embeddings().weight.device


def get_decoder_layers(model):
    core = getattr(model, "model", None)
    layers = getattr(core, "layers", None)
    embeddings = getattr(core, "embed_tokens", None)
    if layers is None or embeddings is None:
        raise TypeError("expected Qwen-style model.model.layers and embed_tokens")
    return embeddings, layers


def encode_task(tokenizer, task: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
    rendered = formatted_prompt(tokenizer, task["prompt"])
    anchors = find_label_anchors(tokenizer, rendered)
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_tensors="pt",
    )
    return rendered, anchors, encoded


def capture_city_c_states(
    *,
    model,
    tokenizer,
    task: dict[str, Any],
) -> tuple[dict[int, Any], list[int], list[int]]:
    import torch

    _, anchors, encoded = encode_task(tokenizer, task)
    positions = anchors["city_c"]["token_positions"]
    input_ids = encoded["input_ids"][0].tolist()
    encoded = {
        key: value.to(model_input_device(model)) for key, value in encoded.items()
    }
    with torch.inference_mode():
        output = model(
            **encoded,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    source = {
        layer: hidden[0, positions, :].detach().clone()
        for layer, hidden in enumerate(output.hidden_states)
    }
    if len(source) != int(model.config.num_hidden_layers) + 1:
        raise RuntimeError("captured hidden-state layer count changed")
    if not all(torch.isfinite(value).all() for value in source.values()):
        raise RuntimeError(f"nonfinite donor activation: {task['sample_id']}")
    del output, encoded
    return source, positions, input_ids


def validate_matched_pair(
    *,
    correct_positions: list[int],
    wrong_positions: list[int],
    correct_ids: list[int],
    wrong_ids: list[int],
) -> None:
    if correct_positions != wrong_positions or len(correct_positions) != 2:
        raise ValueError("correct/swapped City C spans do not align")
    differences = [
        index
        for index, (correct, wrong) in enumerate(zip(correct_ids, wrong_ids))
        if correct != wrong
    ]
    if differences != correct_positions:
        raise ValueError(
            "correct/swapped prompts differ outside the exact City C label span: "
            f"{differences} vs {correct_positions}"
        )


def patch_hook(source, positions: list[int]):
    """Return a hook that replaces two prefill positions and leaves decoding alone."""

    def hook(_module, _inputs, output):
        import torch

        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[1] <= positions[-1]:
            return output
        replacement = source.to(device=hidden.device, dtype=hidden.dtype)
        if replacement.shape != (2, hidden.shape[-1]):
            raise RuntimeError("donor activation shape changed")
        patched = hidden.clone()
        patched[0, positions, :] = replacement
        if isinstance(output, tuple):
            return (patched, *output[1:])
        if torch.is_tensor(output):
            return patched
        raise TypeError(f"unsupported hooked output type: {type(output)}")

    return hook


def generate_one(
    *,
    model,
    tokenizer,
    task: dict[str, Any],
    source_states: dict[int, Any] | None,
    hidden_state_layers: list[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    _, anchors, encoded = encode_task(tokenizer, task)
    positions = anchors["city_c"]["token_positions"]
    encoded = {
        key: value.to(model_input_device(model)) for key, value in encoded.items()
    }
    input_width = int(encoded["input_ids"].shape[1])
    embeddings, decoder_layers = get_decoder_layers(model)
    handles = []
    if source_states is not None:
        for hidden_layer in hidden_state_layers:
            if hidden_layer == 0:
                module = embeddings
            elif 1 <= hidden_layer <= len(decoder_layers):
                module = decoder_layers[hidden_layer - 1]
            else:
                raise ValueError(f"invalid hidden-state layer {hidden_layer}")
            handles.append(
                module.register_forward_hook(
                    patch_hook(source_states[hidden_layer], positions)
                )
            )
    try:
        with torch.inference_mode():
            sequences = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    generated = sequences[:, input_width:]
    text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    raw = _strip_reasoning(text.strip())
    parsed = _parse_reply(raw)
    prediction = parsed["predicted_poll"]
    implied_slope = None
    if prediction is not None:
        implied_slope = (
            float(prediction) - float(task["query_starting_poll"])
        ) / float(task["query_net_news"])
    del encoded, sequences, generated
    return {
        "predicted_poll": prediction,
        "implied_slope": implied_slope,
        "rationale": parsed["rationale"],
        "raw": raw[:8000],
        "parsed": prediction is not None,
        "city_c_token_positions": positions,
    }


def write_partial(path: Path, records: list[dict[str, Any]]) -> None:
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[0, 4],
        help="evidence depths to generate in this allocation",
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_path = args.run_dir / "activation_patch_generations.jsonl"
    manifest_path = args.run_dir / "activation_patch_manifest.json"
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("patch outputs already exist; refusing to overwrite")

    protocol = json.loads(
        (args.run_dir / "relational_protocol_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    if protocol.get("status") != "frozen_before_label_activation_extraction":
        raise ValueError("relational protocol status drifted")
    selection_path = args.run_dir / "development_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "development_selection_complete_test_unread":
        raise ValueError("development selection status drifted")
    selected_window = [
        int(layer)
        for layer in selection["causal_patch_window_hidden_state_layers"]
    ]
    if len(selected_window) != 3 or selected_window != list(
        range(selected_window[0], selected_window[0] + 3)
    ):
        raise ValueError("selected patch window is not three contiguous layers")
    if selected_window[0] < 1 or selected_window[-1] > LAST_CAUSALLY_EFFECTIVE_HIDDEN_LAYER:
        raise ValueError("selected patch window includes a causally ineligible layer")

    tasks, task_manifest = load_tasks(args.run_dir)
    counts = design_counts(task_manifest)
    study = run_spec(args.run_dir)["study"]
    task_by_key = {
        (int(row["episode"]), int(row["c_cases"]), str(row["arm"])): row
        for row in tasks
        if row["split"] == "test" and row["arm"] in ARMS
    }
    episodes = sorted({key[0] for key in task_by_key})
    sealed = counts["sealed_test_episodes"]
    if len(episodes) != sealed or len(task_by_key) != sealed * 2 * len(ARMS):
        raise ValueError(f"expected {sealed} sealed episodes x 2 depths x 2 arms")
    depths = tuple(args.depths)
    if not set(depths) <= {0, 4}:
        raise ValueError("--depths accepts only 0 and 4")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(PATCH_SEED)
    torch.cuda.manual_seed_all(PATCH_SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.eval()
    model.to("cuda")
    observed_commit = getattr(model.config, "_commit_hash", None)
    if observed_commit != MODEL_COMMIT:
        raise RuntimeError(
            f"checkpoint commit changed: observed={observed_commit}, expected={MODEL_COMMIT}"
        )
    if int(model.config.num_hidden_layers) - 1 != LAST_CAUSALLY_EFFECTIVE_HIDDEN_LAYER:
        raise RuntimeError("model depth differs from the frozen causal-layer protocol")
    all_layers = list(range(LAST_CAUSALLY_EFFECTIVE_HIDDEN_LAYER + 1))
    # Resume from a shard written by an earlier allocation. Records are keyed by
    # (episode, depth, recipient arm, condition), so a completed episode-depth
    # block is skipped rather than regenerated.
    records: list[dict[str, Any]] = []
    partial_path = output_path.with_suffix(".partial.jsonl")
    for candidate in (output_path, partial_path):
        if candidate.exists():
            records = read_jsonl(candidate)
            break
    done_blocks = {(int(row["episode"]), int(row["c_cases"])) for row in records}
    expected_total = counts["patch_records"]
    started = time.time()

    for episode in episodes:
        for c_cases in depths:
            if (episode, c_cases) in done_blocks:
                continue
            correct = task_by_key[(episode, c_cases, "abc_context")]
            wrong = task_by_key[(episode, c_cases, "abc_wrong_context")]
            correct_states, correct_positions, correct_ids = capture_city_c_states(
                model=model,
                tokenizer=tokenizer,
                task=correct,
            )
            wrong_states, wrong_positions, wrong_ids = capture_city_c_states(
                model=model,
                tokenizer=tokenizer,
                task=wrong,
            )
            validate_matched_pair(
                correct_positions=correct_positions,
                wrong_positions=wrong_positions,
                correct_ids=correct_ids,
                wrong_ids=wrong_ids,
            )
            state_by_arm = {
                "abc_context": correct_states,
                "abc_wrong_context": wrong_states,
            }
            task_by_arm = {
                "abc_context": correct,
                "abc_wrong_context": wrong,
            }
            for recipient_arm in ARMS:
                recipient = task_by_arm[recipient_arm]
                donor_arm = (
                    "abc_wrong_context"
                    if recipient_arm == "abc_context"
                    else "abc_context"
                )
                specifications = (
                    ("unpatched", None, []),
                    (
                        "self_selected_window",
                        state_by_arm[recipient_arm],
                        selected_window,
                    ),
                    (
                        "cross_selected_window",
                        state_by_arm[donor_arm],
                        selected_window,
                    ),
                    ("cross_embedding_only", state_by_arm[donor_arm], [0]),
                    ("cross_all_layers", state_by_arm[donor_arm], all_layers),
                )
                for condition, source_states, layers in specifications:
                    generated = generate_one(
                        model=model,
                        tokenizer=tokenizer,
                        task=recipient,
                        source_states=source_states,
                        hidden_state_layers=layers,
                        max_new_tokens=args.max_new_tokens,
                    )
                    records.append(
                        {
                            "study": study,
                            "episode": episode,
                            "c_cases": c_cases,
                            "sample_id": recipient["sample_id"],
                            "recipient_arm": recipient_arm,
                            "source_arm": (
                                None
                                if condition == "unpatched"
                                else (
                                    recipient_arm
                                    if condition == "self_selected_window"
                                    else donor_arm
                                )
                            ),
                            "condition": condition,
                            "patched_hidden_state_layers": layers,
                            "target_strong": bool(recipient["target_strong"]),
                            "recipient_cue_strong": bool(recipient["cue_strong"]),
                            **generated,
                        }
                    )
            write_partial(output_path.with_suffix(".partial.jsonl"), records)
            del correct_states, wrong_states
            torch.cuda.empty_cache()
            print(
                f"patch generations {len(records)}/{expected_total} "
                f"({time.time() - started:.1f}s)",
                flush=True,
            )

    records.sort(key=lambda row: (int(row["episode"]), int(row["c_cases"])))
    if len(records) != expected_total:
        # An intentional single-depth shard is written to the partial file and
        # finalised by the allocation that completes the remaining depth.
        write_partial(partial_path, records)
        print(
            f"wrote {len(records)}/{expected_total} records for depths {depths}; "
            "rerun with the remaining depths to finalise",
            flush=True,
        )
        return
    write_partial(output_path, records)
    if partial_path.exists():
        partial_path.unlink()
    manifest = {
        "study": study,
        "status": "complete",
        "model": MODEL_ID,
        "model_commit": observed_commit,
        "patch_site": "both exact City C symbol subtokens",
        "selected_window_hidden_state_layers": selected_window,
        "hidden_state_layer_zero": "token embedding output",
        "hidden_state_layer_n": "output of transformer block n",
        "conditions": [
            "unpatched",
            "self_selected_window",
            "cross_selected_window",
            "cross_embedding_only",
            "cross_all_layers",
        ],
        "directionality": "correct_into_swapped and swapped_into_correct",
        "record_count": len(records),
        "parsed_count": sum(bool(row["parsed"]) for row in records),
        "max_new_tokens": args.max_new_tokens,
        "decode": "greedy; enable_thinking=False",
        "selection_sha256": file_sha256(selection_path),
        "protocol_sha256": file_sha256(
            args.run_dir / "relational_protocol_manifest.json"
        ),
        "tasks_sha256": file_sha256(args.run_dir / "tasks.jsonl"),
        "generations_sha256": file_sha256(output_path),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "elapsed_seconds": time.time() - started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
