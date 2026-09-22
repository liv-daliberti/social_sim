#!/usr/bin/env python3
"""Completion-only LoRA SFT diagnostic; never selects a checkpoint on evaluation.

The launch manifest, rather than command-line defaults, fixes the experimental
budget. Both the data and configuration are hashed before model initialization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode_completion(tokenizer, prompt: str, completion: str, max_length: int):
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    answer_ids = tokenizer.encode(completion, add_special_tokens=False)
    answer_ids.append(tokenizer.eos_token_id)
    ids = prefix_ids + answer_ids
    if len(ids) > max_length:
        raise ValueError(f"Refusing to truncate training example: {len(ids)} > {max_length}")
    labels = [-100] * len(prefix_ids) + answer_ids
    assert answer_ids and all(x == -100 for x in labels[:len(prefix_ids)])
    return {"input_ids": ids, "labels": labels, "prompt_length": len(prefix_ids)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-training-seconds", type=int, default=2500)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    data = Path(config["data"])
    if sha(data) != config["data_sha256"]:
        raise RuntimeError("Training data hash differs from frozen configuration")
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    assert len(rows) == config["unique_prompts"]
    assert len({r["task_id"] for r in rows}) == len(rows)
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    seed = int(config["seed"])
    torch.set_num_threads(4)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(config["model"], local_files_only=True)
    encoded = [encode_completion(tokenizer, r["input"], r["completion"], config["max_length"])
               for r in rows]
    receipt = {
        "config": config, "config_sha256": sha(args.config),
        "code_sha256": sha(Path(__file__)), "data_sha256": sha(data),
        "examples": len(rows), "maximum_tokens": max(len(x["input_ids"]) for x in encoded),
        "completion_tokens": sum(sum(t != -100 for t in x["labels"]) for x in encoded),
        "torch_version": torch.__version__, "job_id": os.environ.get("SLURM_JOB_ID"),
    }
    print(json.dumps(receipt), flush=True)
    if args.validate_only:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "complete.json").exists():
        print("Training already complete", flush=True)
        return
    if (args.output / "receipt.json").exists():
        previous = json.loads((args.output / "receipt.json").read_text())
        assert previous["config_sha256"] == receipt["config_sha256"]
        assert previous["code_sha256"] == receipt["code_sha256"]
        if not (args.output / "resume.pt").exists():
            prior_log = args.output / "train.jsonl"
            if prior_log.exists() and prior_log.stat().st_size:
                raise RuntimeError("Optimizer work exists but its resume state is missing")
            print("Deterministic restart after initialization-only failure", flush=True)
    else:
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map={"": 0},
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=config["lora_rank"], lora_alpha=config["lora_alpha"],
        lora_dropout=0.0, bias="none", target_modules=config["target_modules"],
    ))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config["learning_rate"],
                                 betas=(0.9, 0.95), weight_decay=0.0)
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    if len(rows) % batch_size:
        raise ValueError("Frozen data size must be divisible by the effective batch size")
    total_steps = len(rows) * epochs // batch_size
    start = time.monotonic()
    step = 0
    if (args.output / "resume.pt").exists():
        state = torch.load(args.output / "resume.pt", map_location="cuda", weights_only=False)
        assert state["config_sha256"] == receipt["config_sha256"]
        set_peft_model_state_dict(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        torch.set_rng_state(state["rng_cpu"].cpu())
        torch.cuda.set_rng_state(state["rng_cuda"].cpu())
        print(f"Resuming at optimizer step {step}", flush=True)
    completed_steps = step
    optimizer.zero_grad(set_to_none=True)
    log = (args.output / "train.jsonl").open("a")
    def checkpoint():
        state = {"adapter": get_peft_model_state_dict(model), "optimizer": optimizer.state_dict(),
                 "step": step, "config_sha256": receipt["config_sha256"],
                 "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state()}
        torch.save(state, args.output / "resume.tmp")
        (args.output / "resume.tmp").replace(args.output / "resume.pt")
    if not (args.output / "resume.pt").exists():
        checkpoint()
    for epoch in range(epochs):
        order = np.random.default_rng(seed + epoch).permutation(len(encoded)).tolist()
        for offset in range(0, len(order), batch_size):
            planned_step = epoch * (len(order) // batch_size) + offset // batch_size + 1
            if planned_step <= completed_steps:
                continue
            losses = []
            for index in order[offset:offset + batch_size]:
                row = encoded[index]
                inputs = torch.tensor([row["input_ids"]], device="cuda", dtype=torch.long)
                labels = torch.tensor([row["labels"]], device="cuda", dtype=torch.long)
                result = model(input_ids=inputs, labels=labels, attention_mask=torch.ones_like(inputs))
                loss = result.loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at step {step}, example {index}")
                (loss / batch_size).backward()
                losses.append(float(loss.detach()))
            norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            entry = {"step": step, "total_steps": total_steps, "epoch": epoch,
                     "presentations": step * batch_size, "loss": sum(losses) / len(losses),
                     "grad_norm": float(norm), "elapsed_seconds": time.monotonic() - start}
            log.write(json.dumps(entry) + "\n")
            log.flush()
            if step % 10 == 0 or step == 1:
                print(json.dumps(entry), flush=True)
            if step % 25 == 0:
                checkpoint()
            if time.monotonic() - start > args.max_training_seconds and step < total_steps:
                checkpoint()
                log.close()
                print(f"Cooperative pause at step {step}; endpoint is still fixed at {total_steps}", flush=True)
                return
    log.close()
    assert step == total_steps
    model.save_pretrained(args.output / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(args.output / "adapter")
    complete = {**receipt, "steps": step, "prompt_presentations": len(rows) * epochs,
                "wall_seconds": time.monotonic() - start,
                "adapter_sha256": sha(args.output / "adapter/adapter_model.safetensors")}
    (args.output / "complete.json").write_text(json.dumps(complete, indent=2) + "\n")
    print(json.dumps(complete), flush=True)


if __name__ == "__main__":
    main()
