#!/usr/bin/env python3
# Copyright 2025 Garena Online Private Limited
# Licensed under the Apache License, Version 2.0 (see OAT for full text).
"""Dr. GRPO training for structured and binary forecasting tasks (OAT).

Adapted from oat's self-contained ``run_math_rl.py``.  The only things that change
relative to the math example are:

  * the prompt template is the biased-news information-parity prompt (built in
    ``make_dataset.py``); we wrap it with the Qwen3 chat format here;
  * the verifiable reward is forecast accuracy on the held-out test shock --- the
    agent outputs a JSON ``predicted_poll`` and is rewarded for how close it is to
    the noise-free next poll (``BiasedNewsOracle``).  This is improvable only by
    inferring the city's hidden news-responsiveness g, exactly the Experiment-2
    latent.

Dr. GRPO (``--critic_type drgrpo``) keeps the two bias fixes from the math example:
no length normalization (masked_sum with a constant normalizer) and no std
division in the Monte-Carlo advantage.

Run with examples/biased_news_rl.sh.
"""
from __future__ import annotations

import functools
import itertools
import json
import logging
import re
import time

# --- 2026-08-11 debug aid: dump every Python thread's stack on SIGUSR1 ------------------------
# These nodes run with kernel.yama.ptrace_scope=2, so py-spy and gdb cannot attach and a hung
# process is otherwise opaque. faulthandler needs no ptrace: signal the pid and the stacks land
# in the file below. Registered at import so it is live in every launchpad process (learner and
# actor) before anything can deadlock. Costs nothing when unused.
#     kill -USR1 <pid>;  cat /tmp/oat_stacks_<jobid>_<pid>.txt
def _install_stack_dumper() -> None:
    try:
        import faulthandler
        import os
        import signal

        path = f"/tmp/oat_stacks_{os.environ.get('SLURM_JOB_ID', 'na')}_{os.getpid()}.txt"
        handle = open(path, "w")          # kept open for the process lifetime, by design
        faulthandler.register(signal.SIGUSR1, file=handle, all_threads=True, chain=False)
    except Exception:                     # never let a debug aid break training
        pass


_install_stack_dumper()
from dataclasses import dataclass, field
from typing import List, Literal, Tuple

import numpy as np
import torch
import tree
import vllm
from vllm.sampling_params import GuidedDecodingParams
from torch.utils.data import DataLoader

# --- Use torch AdamW instead of DeepSpeed's FusedAdam/CPUAdam so nothing needs
#     to JIT-compile against nvcc (the cluster has no CUDA toolkit). With LoRA the
#     optimizer is tiny; create_optimizer is called with {lr, betas, weight_decay},
#     all valid AdamW kwargs, and ZeRO-2 wraps any torch optimizer fine.
import torch.optim as _optim  # noqa: E402
import oat.utils.deepspeed as _oat_ds  # noqa: E402
_oat_ds.FusedAdam = _optim.AdamW
_oat_ds.DeepSpeedCPUAdam = _optim.AdamW

from oat.actors.base import ActorBase
from oat.algorithms.ppo import PPOActor, PPOArgs, PPOLearner
from oat.args import default_args_validation, get_default_args
from oat.interface import get_program, lp
from oat.oracles.base import PreferenceOracleBase, RewardOracleBase
from oat.types import Metric, TransitionData
from oat.utils.data import PromptDataset, load_data_from_disk_or_hf
from oat.utils.ops import masked_mean, masked_sum

from forecast_scoring import score_binary_forecast
from output_contract import FORECAST_ARRAY_GBNF, parse_forecast_array


# ----------------------------------------------------------------------------- #
# 1. Prompt template: wrap the (already fully written) parity prompt in the Qwen3
#    instruct chat format.  The biased-news prompt is self-contained, so no extra
#    task instructions are added.
# ----------------------------------------------------------------------------- #
def apply_biased_news_template(question: str) -> str:
    return (
        "<|im_start|>user\n" + question + "<|im_end|>\n<|im_start|>assistant\n"
    )


def apply_no_template(question: str) -> str:
    return question


TEMPLATE_FACTORY = {
    "biased_news": apply_biased_news_template,
    "no": apply_no_template,
}
# "auto" is handled specially (needs the tokenizer): each model's own chat template.
# Use it for non-Qwen models (Llama/Gemma/Mistral) whose chat format differs from the
# hardcoded Qwen ChatML wrap above.


# ----------------------------------------------------------------------------- #
# 2. Oracle: rule-based forecast-accuracy reward.
# ----------------------------------------------------------------------------- #
_PRED_RE = re.compile(r'"?predicted_poll"?\s*:\s*"?(-?\d+(?:\.\d+)?)', re.IGNORECASE)
_PPN_RE = re.compile(r'"?points_per_news"?\s*:\s*"?(-?\d+(?:\.\d+)?)', re.IGNORECASE)
# Multi-scenario rows historically used A-D. The paired mechanism-family benchmark uses A-J;
# compile a bounded alphabet so legacy rows remain byte-for-byte compatible.
# `forecast_X` is the domain-neutral key used by the domain_family arm (2026-08-11); `poll_X` is
# kept so the structure-transfer arm's datasets and checkpoints parse unchanged.
_SCEN_RES = {lab: re.compile(r'"?(?:poll|forecast)_' + lab + r'"?\s*:\s*"?(-?\d+(?:\.\d+)?)',
                             re.IGNORECASE)
             for lab in tuple(chr(ord("A") + i) for i in range(26))}


def _last_match(rx: re.Pattern, text: str):
    m = None
    for m in rx.finditer(text):
        pass
    return float(m.group(1)) if m else None


def _parse_scenarios(text: str, n: int, lo: float = 0.0, hi: float = 100.0):
    """Parse forecast_A onward (last match per key wins). Every requested key is required.

    [lo, hi] is the row's OBSERVABLE range. It defaults to the poll range so legacy rows behave
    exactly as before; domain_family rows carry their own range (e.g. 0-400 crates), without which
    every out-of-poll-range forecast would be clamped to 100 and the slope destroyed."""
    if n > len(_SCEN_RES):
        raise ValueError(f"requested {n} scenarios; parser supports {len(_SCEN_RES)}")
    preds = []
    labels = tuple(chr(ord("A") + i) for i in range(n))
    for lab in labels:
        v = _last_match(_SCEN_RES[lab], text)
        if v is None:
            return None
        preds.append(max(lo, min(hi, v)))
    return preds


class BiasedNewsOracle(RewardOracleBase, PreferenceOracleBase):
    """Reward = forecast quality against the noise-free target(s).

    Legacy single-shock rows (reference has `target`):
        reward = max(0, 1 - |pred - target| / reward_scale_pts), clipped to [0, 1].

    Multi-shock rows (reference has `targets`, dag_family 2026-07-08 audit fix): the reply carries
    four scenario forecasts poll_A..poll_D, and

        reward = (1 - slope_weight) * mean_i max(0, 1 - |pred_i - target_i| / reward_scale_pts)
               +      slope_weight  * max(0, 1 - |slope(preds) - slope_target| / slope_scale_g)

    The slope term scores the RESPONSE of the forecasts to the probed news, per episode, against
    the true one-step sensitivity. A constant 'ignore the news' policy caps its slope term at the
    prior-guess level in gain-hiding worlds, so the reward is no longer collectable by global
    shrinkage — the failure mode the Jul-7 single-shock runs converged to.

    A response missing any required field gets reward 0 and formatted=False.
    """

    def __init__(self, reward_scale_pts: float = 30.0, slope_weight: float = 0.5,
                 slope_scale_g: float = 0.5) -> None:
        super().__init__()
        self.scale = float(reward_scale_pts)
        self.slope_weight = float(slope_weight)
        self.slope_scale_g = float(slope_scale_g)

    def _reward_multi(self, resp: str, meta: dict):
        targets = [float(t) for t in meta["targets"]]
        shocks = np.asarray([float(s) for s in meta["shocks"]])
        # Per-row observable range and reward tolerances. Domains differ in scale (poll points vs
        # crates vs minutes), so a fixed points-valued tolerance would make a wide-scale domain
        # arbitrarily harder and confound "which domain" with "how hard the reward is". The
        # domain_family builder therefore scales both tolerances by the domain's display scale;
        # legacy dag_family rows omit these keys and fall back to the CLI values unchanged.
        lo, hi = (float(v) for v in meta.get("clip", (0.0, 100.0)))
        scale = float(meta.get("reward_scale", self.scale))
        slope_scale = float(meta.get("slope_scale", self.slope_scale_g))
        # C3 mechanism rows use one strict compact JSON contract in both the online
        # reward and downstream reporting. Legacy rows retain their keyed parser.
        if "response_targets" in meta:
            preds = parse_forecast_array(resp, len(targets), (lo, hi))
        else:
            preds = _parse_scenarios(resp, len(targets), lo, hi)
        if preds is None:
            return 0.0, {"formatted": False}
        preds = np.asarray(preds)
        level = float(np.mean([max(0.0, 1.0 - abs(p - t) / scale)
                               for p, t in zip(preds, targets)]))
        if "response_targets" in meta:
            pairs = [(int(i), int(j)) for i, j in meta["response_pairs"]]
            target_response = np.asarray(meta["response_targets"], dtype=float)
            predicted_response = np.asarray([preds[i] - preds[j] for i, j in pairs])
            response_scale = float(meta.get("response_scale", self.slope_scale_g))
            response_errors = np.abs(predicted_response - target_response)
            response_reward = float(np.mean(
                np.maximum(0.0, 1.0 - response_errors / response_scale)
            ))
            reward = ((1.0 - self.slope_weight) * level
                      + self.slope_weight * response_reward)
            return reward, {
                "formatted": True,
                "abs_err": float(np.mean(np.abs(preds - np.asarray(targets)))),
                "response_err": float(np.mean(response_errors)),
                "g": float(meta.get("g", float("nan"))),
            }

        # Legacy DAG/domain rows retain their registered slope reward.
        slope = float(np.cov(shocks, preds, bias=True)[0, 1] / np.var(shocks))
        slope_err = abs(slope - float(meta["slope_target"]))
        slope_r = max(0.0, 1.0 - slope_err / slope_scale)
        r = (1.0 - self.slope_weight) * level + self.slope_weight * slope_r
        return r, {
            "formatted": True,
            "abs_err": float(np.mean(np.abs(preds - np.asarray(targets)))),
            "implied_slope": slope,
            "slope_err": slope_err,
            "g": float(meta["g"]),
        }

    def get_reward(
        self,
        inputs: List[str],
        responses: List[str],
        references: List[str],
        batch_size: int = 4,
    ) -> Tuple[torch.Tensor, Metric]:
        del inputs, batch_size
        rewards, infos = [], []
        for resp, ref in zip(responses, references):
            meta = json.loads(ref)
            if "settlement_yes" in meta:              # Experiment 3B proper-score row
                reward, info = score_binary_forecast(
                    resp,
                    bool(meta["settlement_yes"]),
                    float(meta["market_yes_prob"]),
                )
                rewards.append(reward)
                infos.append(info)
                continue
            if "targets" in meta:                      # multi-shock joint row
                r, info = self._reward_multi(resp, meta)
                rewards.append(r)
                infos.append(info)
                continue
            target = float(meta["target"])
            pred = _last_match(_PRED_RE, resp)
            if pred is None:
                rewards.append(0.0)
                infos.append({"formatted": False})
                continue
            pred = max(0.0, min(100.0, pred))
            r = max(0.0, 1.0 - abs(pred - target) / self.scale)
            # implied gain for diagnostics (the Exp-2 recovery axis)
            tn = float(meta["test_news"])
            ig = (pred - float(meta["last_poll"])) / tn if tn else float("nan")
            stated = _last_match(_PPN_RE, resp)
            infos.append({
                "formatted": True,
                "abs_err": abs(pred - target),
                "implied_g": ig,
                "stated_g": stated,
                "g": float(meta["g"]),
            })
            rewards.append(r)
        return torch.tensor(rewards, dtype=torch.float32), infos

    def compare(self, inputs, candidates_A, candidates_B, batch_size=4,
                return_probs=False, disable_tqdm=False):
        del batch_size, return_probs, disable_tqdm
        rewards, info = self.get_reward(inputs, candidates_A, candidates_B)
        return rewards.numpy(), info


# ----------------------------------------------------------------------------- #
# 3. Args.
# ----------------------------------------------------------------------------- #
@dataclass
class BiasedNewsArgs(PPOArgs):
    prompt_template: Literal["biased_news", "no", "auto", "auto_no_think"] = field(default="biased_news")
    reward_scale_pts: float = field(default=30.0)
    # multi-shock composite reward (dag_family rows with `targets`); ignored for legacy rows
    slope_weight: float = field(default=0.5)
    slope_scale_g: float = field(default=0.5)
    # Dr. GRPO ablation toggles (kept identical to the math example).
    remove_std_bias: bool = False
    remove_len_bias: bool = False
    max_train: str = "999999"
    # Synthetic C3 uses syntax-only GBNF guidance. Keep ``none`` available
    # because the Polymarket extension intentionally shares this trainer.
    structured_output: Literal["none", "forecast_array"] = "none"
    # Execution-only vLLM compatibility switch. Some multi-GPU A6000
    # allocations fail during custom-all-reduce initialization; in that case
    # use vLLM's supported NCCL fallback without changing the training recipe.
    disable_custom_all_reduce: bool = False
    # Tensor-parallel NCCL workers can stall while capturing CUDA graphs on the
    # cluster's A100 stack. Eager execution skips that initialization path.
    enforce_eager: bool = False


# ----------------------------------------------------------------------------- #
# 4. Actor: generation + rule-based reward (step() copied verbatim from the math
#    example -- it is task-agnostic).
# ----------------------------------------------------------------------------- #
class BiasedNewsActor(PPOActor):
    @staticmethod
    def _forecast_guidance():
        return GuidedDecodingParams(
            grammar=FORECAST_ARRAY_GBNF,
            backend="xgrammar",
        )

    def init(self, actor_id, save_path):
        if self.args.disable_custom_all_reduce:
            self.vllm_args["disable_custom_all_reduce"] = True
        if self.args.enforce_eager:
            self.vllm_args["enforce_eager"] = True
        if self.args.structured_output == "forecast_array":
            # vLLM V1 requires the per-request backend to match the engine-level
            # backend. Selecting xgrammar explicitly also disables auto fallback.
            self.vllm_args["guided_decoding_backend"] = "xgrammar"
        super().init(actor_id, save_path)
        self.oracle = BiasedNewsOracle(reward_scale_pts=args.reward_scale_pts,
                                       slope_weight=args.slope_weight,
                                       slope_scale_g=args.slope_scale_g)
        # Qwen instruct models stop themselves; unset external stop conditions.
        self.sampling_params.stop = None
        self.sampling_params.stop_token_ids = None
        self.eval_sampling_params.stop = None
        self.eval_sampling_params.stop_token_ids = None
        if args.structured_output == "forecast_array":
            # Mask only structurally invalid continuations. Values remain
            # unrestricted JSON numbers and are scored by the unchanged oracle.
            self.sampling_params.guided_decoding = self._forecast_guidance()
            self.eval_sampling_params.guided_decoding = self._forecast_guidance()

    def step(self, prompts, formatted_prompts, references=None):
        assert not self.eval_mode
        info = {}
        num_samples = self.sampling_params.n
        num_prompts = len(formatted_prompts)

        st = time.time()
        expanded_prompts = []
        for p in formatted_prompts:
            expanded_prompts.extend([p] * num_samples)

        sampling_params_n1 = vllm.SamplingParams(
            temperature=self.sampling_params.temperature,
            top_p=self.sampling_params.top_p,
            top_k=self.sampling_params.top_k,
            max_tokens=self.sampling_params.max_tokens,
            n=1,
            logprobs=1,
            guided_decoding=(
                self._forecast_guidance()
                if self.args.structured_output == "forecast_array" else None
            ),
        )
        outputs = self.generate(expanded_prompts, sampling_params_n1)

        candidates, prompt_token_ids, no_eos = [], [], []
        response_ids, response_logprobs, resp_lens = [], [], []
        for i in range(num_prompts):
            start_idx = i * num_samples
            prompt_outputs = outputs[start_idx:start_idx + num_samples]
            prompt_token_ids.append(prompt_outputs[0].prompt_token_ids)
            candidates.append([]); response_logprobs.append([]); response_ids.append([])
            for out in prompt_outputs:
                completion = out.outputs[0]
                candidates[i].append(completion.text)
                no_eos.append(completion.finish_reason == "length")
                token_ids = completion.token_ids
                logps = completion.logprobs
                logps = [item[token_ids[j]].logprob for j, item in enumerate(logps)]
                response_logprobs[i].append(logps)
                response_ids[i].append(token_ids)
                resp_lens.append(len(token_ids))
        info["actor/generate_time"] = time.time() - st

        st = time.time()
        rewards, oracle_infos = self.oracle.get_reward(
            list(itertools.chain.from_iterable(
                itertools.repeat(x, self.sampling_params.n) for x in prompts)),
            tree.flatten(candidates),
            list(itertools.chain.from_iterable(
                itertools.repeat(x, self.sampling_params.n) for x in references)),
        )
        info["actor/verify_time"] = time.time() - st
        logging.info(f"actor reward {rewards.mean()}")
        info["actor/rewards"] = rewards.mean().item()
        info["actor/num_data"] = rewards.numel()
        info["actor/formatted"] = np.mean([i["formatted"] for i in oracle_infos])
        info["actor/response_tok_len"] = np.mean(resp_lens)
        info["actor/sampling_max_tokens"] = self.sampling_params.max_tokens
        info["actor/sampling_temperature"] = self.sampling_params.temperature

        rewards = rewards.reshape(len(prompts), -1)
        no_eos = np.array(no_eos).reshape(len(prompts), -1)
        info["actor/no_eos_count"] = no_eos.sum()

        trajectory_data = []
        for i in range(len(candidates)):
            prompt = prompts[i]
            for j in range(len(candidates[i])):
                reward = rewards[i][j].item()
                if no_eos[i][j]:
                    reward = 0
                dense_rewards = [0] * len(response_ids[i][j])
                dense_rewards[-1] = reward
                trajectory_data.append(TransitionData(
                    prompt=prompt,
                    prompt_ids=prompt_token_ids[i],
                    response=candidates[i][j],
                    response_ids=response_ids[i][j],
                    response_logprobs=response_logprobs[i][j],
                    rewards=dense_rewards,
                    loss_mask=not no_eos[i][j] if self.args.ignore_no_eos else True,
                    info=info,
                ))
        logging.info(f"actor finished data_len={len(trajectory_data)}")
        return self.ipc_client.serialize_ipc(trajectory_data)


# ----------------------------------------------------------------------------- #
# 5. Learner: Dr. GRPO advantage + biased-news data wiring.  No evaluate override
#    -- the base evaluate reports eval/score = mean reward on the held-out cities
#    and dumps per-row responses+references (with g) for the offline recovery read.
# ----------------------------------------------------------------------------- #
class BiasedNewsLearner(PPOLearner):
    def _init(self, args: BiasedNewsArgs, actors: List[ActorBase]) -> None:
        # Launchpad reconstructs learner nodes in fresh Python processes. The
        # module-level optimizer swap above is not reliably preserved by that
        # serialization boundary, so apply it again inside every learner rank
        # before OAT creates its DeepSpeedStrategy. This keeps the LoRA-only
        # optimizer on torch AdamW and avoids a fragile fused_adam CUDA JIT.
        import oat.utils.deepspeed as learner_oat_ds

        learner_oat_ds.FusedAdam = torch.optim.AdamW
        learner_oat_ds.DeepSpeedCPUAdam = torch.optim.AdamW
        super()._init(args, actors)
        self.args = args
        self.args.max_queries = np.inf
        # Dr. GRPO Mod 1: remove length bias.
        self.masked_aggregator = (
            functools.partial(masked_sum, constant_normalizer=args.generate_max_length)
            if args.critic_type == "drgrpo" else masked_mean
        )
        if args.critic_type in ["grpo", "ppo"] and args.remove_len_bias:
            self.masked_aggregator = functools.partial(
                masked_sum, constant_normalizer=args.generate_max_length)

    # Dr. GRPO Mod 2: MC advantage without std division.
    def compute_monte_carlo_advantages(self, rewards, response_masks):
        rewards = rewards.sum(-1)
        values = rewards.view(-1, self.args.num_samples).mean(dim=1)
        values = values.repeat_interleave(self.args.num_samples, dim=0)
        advantages = rewards - values
        if (self.args.critic_type == "grpo") and (not self.args.remove_std_bias):
            std = rewards.view(-1, self.args.num_samples).std(dim=1)
            std = std.repeat_interleave(self.args.num_samples, dim=0)
            advantages = advantages / (std + 1e-8)
        return advantages

    def _format(self, q):
        # "auto": each model's own chat template (Llama/Gemma/Mistral). Otherwise the
        # hardcoded Qwen ChatML wrap (biased_news) or raw (no).
        if self.args.prompt_template in {"auto", "auto_no_think"}:
            kwargs = {}
            if self.args.prompt_template == "auto_no_think":
                # Hybrid Qwen3 checkpoints otherwise spend the short JSON budget thinking.
                kwargs["enable_thinking"] = False
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": q}], tokenize=False,
                add_generation_prompt=True, **kwargs)
        return TEMPLATE_FACTORY[self.args.prompt_template](q)

    def _apply_template(self, example):
        example[self.args.input_key] = self._format(example[self.args.input_key])
        return example

    def prepare_data(self, strategy, tokenizer):
        self.tokenizer = tokenizer
        train = load_data_from_disk_or_hf(self.args.prompt_data)[self.args.train_split]
        train = train.select(
            range(min(int(self.args.max_train), len(train)))
        ).select_columns([self.args.input_key, self.args.output_key])
        # load_from_cache_file=False: never reuse a stale .map() cache from a prior
        # run whose dataset had a different prompt.
        train = train.map(lambda x: self._apply_template(x), load_from_cache_file=False)

        self.prompts_dataset = PromptDataset(
            train, tokenizer, strategy,
            input_key=self.args.input_key, output_key=self.args.output_key,
            apply_chat_template=False, get_reference=True,
        )
        self.prompts_dataloader = strategy.setup_dataloader(
            self.prompts_dataset,
            strategy.args.rollout_batch_size_per_device,
            pin_memory=True, shuffle=True,
        )

        eval_set = load_data_from_disk_or_hf(self.args.eval_data)[self.args.train_split]
        self.eval_prompts_dataset = eval_set
        self.eval_prompts_dataloader = DataLoader(
            eval_set, batch_size=self.args.eval_batch_size,
            shuffle=False, drop_last=False,
            collate_fn=self.eval_dataloader_collate_fn,
        )

    def eval_dataloader_collate_fn(self, item_list):
        formatted, raw, refs = [], [], []
        for item in item_list:
            q = item[self.args.input_key]
            raw.append(q)
            formatted.append(self._format(q))
            refs.append(item[self.args.output_key])
        return formatted, raw, refs


def run(args: BiasedNewsArgs):
    program, local_resources = get_program(
        args, learner_cls=BiasedNewsLearner, actor_cls=BiasedNewsActor)
    lp.launch(program, launch_type=args.launch_type,
              local_resources=local_resources, terminal="current_terminal")


if __name__ == "__main__":
    args: BiasedNewsArgs = get_default_args(BiasedNewsArgs)
    args.algo = "PPO"
    args.online_evaluation = True
    args = default_args_validation(args)
    run(args)
