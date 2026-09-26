#!/usr/bin/env python3
"""Evaluate a base model or saved LoRA endpoint with greedy or repeated decoding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_from_disk

from output_contract import FORECAST_ARRAY_GBNF


def formatted_prompts(tokenizer, prompts: list[str], template: str) -> list[str]:
    if template == "biased_news":
        return [f"<|im_start|>user\n{item}<|im_end|>\n<|im_start|>assistant\n" for item in prompts]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if template == "auto_no_think":
        kwargs["enable_thinking"] = False
    return [tokenizer.apply_chat_template([{"role": "user", "content": item}], **kwargs)
            for item in prompts]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--template", choices=("biased_news", "auto", "auto_no_think"),
                        default="biased_news")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--secondary-output", type=Path)
    parser.add_argument("--secondary-temperature", type=float, default=0.7)
    parser.add_argument("--secondary-n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=3072)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.86)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--structured-output", choices=("none", "forecast_array"),
                        default="none")
    args = parser.parse_args()
    if args.temperature == 0 and args.n != 1:
        parser.error("greedy decoding has exactly one draw; use temperature > 0 for repeats")
    if args.n < 1:
        parser.error("--n must be positive")
    if args.secondary_output is not None and args.secondary_temperature <= 0:
        parser.error("--secondary-temperature must be positive")
    if args.secondary_n < 1:
        parser.error("--secondary-n must be positive")
    if args.max_prompts is not None and args.max_prompts < 1:
        parser.error("--max-prompts must be positive")
    if args.max_lora_rank < 1:
        parser.error("--max-lora-rank must be positive")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import GuidedDecodingParams

    dataset = load_from_disk(str(args.data))["train"]
    if args.max_prompts is not None:
        dataset = dataset.select(range(min(args.max_prompts, len(dataset))))
    engine_kwargs = {
        "model": args.model,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "seed": args.seed,
        "enable_prefix_caching": True,
    }
    if args.disable_custom_all_reduce:
        engine_kwargs["disable_custom_all_reduce"] = True
    if args.adapter:
        engine_kwargs.update({"enable_lora": True, "max_lora_rank": args.max_lora_rank})
    if args.structured_output == "forecast_array":
        # vLLM V1 requires request- and engine-level backends to match.
        engine_kwargs["guided_decoding_backend"] = "xgrammar"
    llm = LLM(**engine_kwargs)
    raw_prompts = list(dataset["input"])
    prompts = formatted_prompts(llm.get_tokenizer(), raw_prompts, args.template)
    request = (LoRARequest("endpoint", 1, str(args.adapter.resolve()))
               if args.adapter else None)

    def generate_to(output: Path, temperature: float, n: int) -> None:
        sampling = SamplingParams(
            temperature=temperature,
            top_p=0.95 if temperature > 0 else 1.0,
            n=n,
            max_tokens=args.max_tokens,
            seed=args.seed,
            guided_decoding=(
                GuidedDecodingParams(
                    grammar=FORECAST_ARRAY_GBNF,
                    backend="xgrammar",
                )
                if args.structured_output == "forecast_array" else None
            ),
        )
        generated = llm.generate(prompts, sampling, lora_request=request)
        records = []
        for index, result in enumerate(generated):
            records.append({
                "input": raw_prompts[index],
                "output": [candidate.text for candidate in result.outputs],
                "reference": dataset[index]["reference"],
                "model": args.model,
                "adapter": str(args.adapter) if args.adapter else None,
                "temperature": temperature,
                "decode_seed": args.seed,
                "structured_output": args.structured_output,
            })
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {len(records)} prompts x {n} draws -> {output}")

    generate_to(args.output, args.temperature, args.n)
    if args.secondary_output is not None:
        generate_to(
            args.secondary_output,
            args.secondary_temperature,
            args.secondary_n,
        )


if __name__ == "__main__":
    main()
