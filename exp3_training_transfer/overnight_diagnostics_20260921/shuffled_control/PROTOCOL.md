# Actual shuffled-target control: frozen 2026-09-22 UTC

The experiment tests whether Qwen3-4B mechanism-family improvement benefits from learning episode-specific relationships, relative to distribution-matched supervision with broken episode/target assignment. The existing population-prior arm remains a population prior. It is not called a shuffled control.

## Fixed roster and reuse audit

Ten new shuffled-target runs: disclosed and undisclosed × seeds 42, 43, 44, 45, 46. Pair each with the corresponding causal-family training seed. Use Qwen/Qwen3-4B-Instruct-2507 snapshot `cdbee75f17c01a7cc42f958dc650907174af0554`.

The audit checks actual weights, runtime arguments, and registered submission provenance. Seeds 45/46 have real reusable matched adapters in both disclosures. Seeds 42–44 retain completed scores and checkpoint metadata but their adapter weights are absent. A bounded project/runtime search found no replacement. Six matched recovery runs, exactly the missing disclosure/seed cells, are therefore required for the fresh five-seed comparison. Their runtime uses the same frozen current trainer as the controls. Existing seed45/46 runs are reused. Missing prior weights do not require prior retraining for this comparison.

The registered wrapper, output parser, simulator and renderer hashes still match the historical ledger. The historical trainer hash differs. Current tracked changes add disabled execution switches and reapply the same AdamW patch inside the learner. Historical and recent logs both verify `torch.optim.adamw.AdamW`; key effective scientific settings match. The historical git revision is unavailable locally, so exact historical trainer byte identity is not asserted. `checkpoint_audit.json` preserves settings, weight checks and this caveat. New runs use a frozen source copy and runtime source hashes.

## Target permutation

Copy each existing 4,800-row causal-family training dataset with byte-identical input strings and identical row order. For each training seed independently, make one deterministic Sattolo derangement within each compatibility stratum: `(world, k, scenario_labels, scenario_shocks, scenario_horizons, response_pairs, clip, reward_scale, response_scale)`. These are 24 groups of 200 episodes. Draw the same donor mapping for both disclosure conditions. Never permute across world, evidence depth, units, scenario order or reward normalization.

Move each donor's complete `targets`, `response_targets`, and rendered `gold` together. Preserve all recipient displayed data, true-target metadata and reference identity. Add donor task ID and permutation seed only to reference metadata, which is never part of the prompt. Invariants require exact target-vector/reward-target multiset equality both globally and within every stratum, a bijective mapping, no self assignment, compatible donors, coherent response differences (allowing the source's four-decimal rounding), and identical prompts. Numerically identical vectors can occur naturally; their count is reported, without resampling the permutation based on effect size.

This control retains world-level mechanism information. Its contrast specifically tests episode-specific gain/history learning beyond within-mechanism marginal targets. It does not isolate the contribution of learning mechanism labels across worlds.

## Unchanged training recipe

Dr. GRPO, rank32/alpha64/dropout0 LoRA on the existing seven projection modules; torch AdamW, learning rate 1e-6, constant scheduler, warmup ratio .03, beta/KL0; 4,800 prompts once, rollout batch16, eight draws per prompt, one PPO epoch, per-device train batch1, total300 rollout rounds and2,400 optimizer steps. Temperature1.3, top_p1, max generated tokens192, prompt cap2304, context3072, syntax-only forecast-array grammar. Reward .4 level + .6 response, scales10 and4. ZeRO2, BF16, gradient checkpointing, prefix caching, noncollocated actor/learner on2 A6000 GPUs. Seed matches its causal-family comparator.

The original heldout set remains the online evaluation set at every25 rounds and original endpoint evaluation. It is never reward data. Fixed terminal adapter step00301 is used, with no checkpoint selection or early stopping. Fresh evaluation below is evaluated only after terminal weights exist.

## Fresh frozen endpoint evaluation and inference

The shared evidence-use agent creates a new paired ordinary heldout set before any new training: seed921000000 + world_index×1000000 + episode_index, 4 heldout worlds ×60 independent trajectory seeds × k={3,6,9}, yielding720 prompts per disclosure. Its saved JSONL and HF datasets are under `../evidence_use/data/heldout_{disclosure}`; code, manifest and file hashes enter `freeze.json` before submission. The same numeric episodes appear across disclosures and no fresh task/seed overlaps train, old holdout or paired evidence-use evaluation.

Primary endpoint: mean of five stochastic draws, temperature.7/top_p.95, max192 tokens, syntax-only grammar; decode seed=training_seed+20260814. Greedy with the same limits is secondary. Existing matched45/46 receive exactly the same fresh evaluation. Recovered matched42–44 and shuffled42–46 receive fresh evaluation in their own terminal jobs.

Primary contrast per disclosure is shuffled minus matched response MAE, positive favoring matched. Average decodes within episode, weight four worlds and three evidence depths equally, average paired seed-level effects. Preserve the shared trajectory identity across evidence depths when bootstrapping. Report five seed effects, Student-t95% interval on those means, hierarchical seed/trajectory bootstrap with5,000 replicates seed20260818, exact one-sided seed-level sign test, and the registered wild cluster bootstrap9,999 replicates. Report fresh versus historical evaluation separately; historical results cannot fill missing fresh checkpoints. Greedy, level MAE, each heldout block and each depth are secondary. Parse failures retain the benchmark's failure penalty and parse coverage is reported. With incomplete jobs report the fixed roster and missing cells; do not silently claim G=5 or substitute historical test scores.

## Compute, failures and reporting

Ten controls are the requested new scientific control budget. Six matched runs restore missing checkpoints for the fresh evaluation; they are disclosed as additional recovery cost. Each requests2 A6000/8 CPUs/100G/20h, accountallcs, partitioncs, qosnone, exclude node206, nice1000 for controls and nice2000 for matched recovery. Arrays allow up to10 controls and6 recoveries but available scheduler capacity determines throughput. Existing full runs took about13.5–16.25h each; all results by morning are not guaranteed. Inference jobs are submitted first.

At most three identical infrastructure attempts per fixed cell; record attempt job IDs and failure reasons. Never drop a seed or modify the training/evaluation design based on results. Do not treat interim completed cells as a final five-seed result. All source, data, donor-map and job-roster hashes are frozen before submission.

Pre-submission resource correction at2026-09-22T03:24UTC: the scheduler rewrites long mltheory/all jobs to the owner partition, which has no A6000s. Preserve the original2A6000 allocation under allcs/cs. Fresh matched inference runs use1A5000 under mltheory/all. Numerical settings and frozen data are unchanged.
