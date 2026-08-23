#!/usr/bin/env python3
"""Quick probe: at rollout sampling settings, what fraction of completions finish
the JSON? Decides GEN_LEN / temperature before committing to a full RL run."""
import re
import sys
from pathlib import Path
from datasets import load_from_disk
from vllm import LLM, SamplingParams

REPO = Path(__file__).resolve().parents[2]
GEN = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
TEMP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
NP = int(sys.argv[3]) if len(sys.argv) > 3 else 24
N = 4  # samples per prompt (like GRPO group)

rx = re.compile(r'predicted_poll"?\s*:\s*"?(-?\d+)', re.I)
ds = load_from_disk(str(REPO / "exp3_training_transfer/biased_news/data/train"))["train"]
prompts = ["<|im_start|>user\n" + ds[i]["input"] + "<|im_end|>\n<|im_start|>assistant\n"
           for i in range(NP)]

llm = LLM(model="Qwen/Qwen3-4B-Instruct-2507", max_model_len=GEN + 700,
          gpu_memory_utilization=0.85, enable_prefix_caching=True)
sp = SamplingParams(temperature=TEMP, top_p=1.0, n=N, max_tokens=GEN)
outs = llm.generate(prompts, sp)

tot = fin = 0
lens = []
for o in outs:
    for c in o.outputs:
        tot += 1
        lens.append(len(c.token_ids))
        if rx.search(c.text) and c.finish_reason != "length":
            fin += 1
lens.sort()
print(f"\n=== GEN={GEN} TEMP={TEMP} n={tot} samples ===")
print(f"formatted (JSON + not truncated): {fin}/{tot} = {fin/tot:.1%}")
print(f"resp token len: median={lens[len(lens)//2]} p90={lens[int(len(lens)*0.9)]} max={lens[-1]}")
