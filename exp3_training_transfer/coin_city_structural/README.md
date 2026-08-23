# Coin City structural and domain transfer (locked v1)

This is the cohesive main-body Experiment 3 extension of the fixed-agent Coin
City experiment.  Training uses only the direct Exp2 relationship

`p_end = p_start + g_j x + epsilon`.

At evaluation, equations and internal structure names are withheld.  Every
prompt shows a complete direct/immediate reference system, a complete unseen
persistent/mediated reference system, a target with `k in {0,2,4,8}`
observations, and ten short/long-horizon interventions.  The qualitative target
description is correct, absent, or misleading while all numeric content is
paired.  Evaluation crosses familiar Coin City and held-out Coin Harbor surfaces
with the familiar and unseen structures: ID, domain transfer, structural
transfer, and joint domain-plus-structure transfer.

## Registered grid

- Models: Qwen3-4B-Instruct-2507, Qwen3-8B, Llama-3.1-8B-Instruct.
- Confirmatory arms: causal versus population-prior.
- Seeds: 42, 43, 44 (18 full training jobs).
- Diagnostic: three Qwen3-4B structureless runs.
- Untrained endpoints: one per model.
- Evaluation: 2 domains x 2 target structures x 3 hints x 4 evidence depths x
  30 episodes = 1,440 prompts per checkpoint.
- Decoding: greedy plus five draws at temperature 0.7.

The causal and population-prior training prompts are byte-identical.  Causal
reward targets are the episode truth.  The prior arm receives an exact within-k
derangement of those targets: the complete target marginal is unchanged, but no
reward remains attached to its own reference/target evidence.  The structureless
arm is an appendix diagnostic, not a confirmatory comparator.

The primary estimand is population-prior minus causal response MAE on the joint
Coin Harbor / unseen-structure / correct-hint cell, weighting k equally.  Positive
values favor causal training.  Prespecified secondary cells isolate domain
transfer (new domain, familiar structure) and structural transfer (familiar
domain, unseen structure).  Cue contrasts compare correct with no and misleading
hints; the misleading penalty by k tests whether observations override the
qualitative prior.  Training seed is the top-level variance unit; episodes are
resampled within seed and stochastic draws are averaged within episode.

## Training setup

Every full run uses 4,800 prompts, 300 rollout-and-update rounds, 16 prompts per
round, eight rollouts per prompt, and 2,400 AdamW policy steps.  Dr. GRPO uses
rank-32 LoRA with alpha 64, learning rate 1e-6, constant schedule, 3% warmup,
one PPO epoch, beta 0, BF16, gradient checkpointing, FlashAttention, ZeRO stage
2, reference offload, and separated actor/learner A6000 GPUs.  Rollouts use
temperature 1.3, top-p 1, a 192-token generation limit, 2,304 prompt-token limit,
and 3,072 total model-token limit.  Reward weights level 0.40 and intervention
response 0.60.  Syntax-only XGrammar requires one ten-number JSON array but does
not constrain values.  Greedy online evaluation runs every 50 rounds.

Six short model-by-arm canaries wait for the pre-existing C3 and Polymarket
campaign.  Full jobs have an `afterok` dependency on a fail-closed canary audit;
therefore a parser, artifact, reward-variation, or scheduler failure cannot
silently unlock the confirmatory grid.

## Submitted campaign and paper handoff

The immutable scientific ledger is
`runs/coin_city_structural_20260818T203858Z.json`.  The registered scheduler DAG
is:

- canaries `30723972`--`30723977`, with `afterany` dependencies on all 56 jobs in
  the effective broad-mechanism and Polymarket-roster replacement ledgers;
- fail-closed gate `30723978`, with `afterok` on all six canaries;
- 18 confirmatory, three diagnostic, and three base jobs were initially registered
  as `30723979`--`30724003` (job ID `30723986` is unrelated), all with `afterok`
  on the gate. Before any of these jobs started, the twelve Qwen3-8B/Llama cells
  were superseded by scientifically identical 30-hour jobs `30730378`--`30730389`
  after upstream throughput showed that 24 hours was marginal; Qwen3-4B and base
  IDs are unchanged;
- upstream disclosed Qwen3-8B causal seed 42 job `30730348` stalled after an NCCL
  communicator bind timeout before completing its first round. Scientifically
  identical replacement `30788373` now occupies that cell in the exact 56-job
  canary barrier;
- hash-pinned paper finalizer `30788375`, with `afterany` on the effective 24 scientific
  endpoints.  Its exact-roster renderer refuses partial output, then writes the
  numerical summary and three tables to both paper trees and compiles both PDFs.
- cross-paper completion verifier `30788376`, with `afterok` on both the current-
  campaign finalizer `30788374` and Coin City finalizer `30788375`. It requires all
  89 upstream registered jobs to have completed, revalidates result-row counts and
  artifact hashes, compares all nine generated tables across the two paper trees,
  and requires both PDFs to be newer than their tables and to contain the expected
  result markers. It also fails closed unless the verifier and all four registered
  inputs retain their submission-time SHA-256 hashes. Earlier verifier registrations
  were canceled while pending and without execution as the fail-closed hash checks
  and complete prespecified estimate schema were added.

Run `python monitor.py` for one read-only snapshot or `python monitor.py --watch
60` for continuous scheduler and training-step monitoring.  The cross-campaign
paper handoff and renderer hashes are recorded in
`../runs/coin_analysis_repair_20260818T213628Z.json`. The pre-execution inference
extension is recorded in `../runs/coin_analysis_extension_20260818T220924Z.json`;
the active terminal verifier, exact command, dependencies, and hashes are recorded
in `../runs/paper_verifier_20260819T172807Z.json`. The single-cell NCCL recovery
and replacement finalizer lineage are recorded in
`../runs/runtime_recovery_20260819T172806Z.json`. The original
ledger remains the scientific freeze; the final derived
`runs/coin_city_structural_20260818T203858Z_walltime30h_20260818T211828Z_effectivebarrier_20260818T212454Z_runtimefix_20260819T172806Z.json`
changes scheduler lineage, wall time, and the ordering dependency only. The
barrier correction was applied while all six canaries were still pending and
ensures they wait for the replacement jobs rather than merely for their cancelled
predecessors; it changes no scientific configuration.

A pre-execution ingestion audit found that the frozen dataset names the no-hint
cue `none` while the paper renderer expected `absent`. The renderer now selects
the registered `none` key; a regression test executes that table path, and the
finalizer refuses to run unless the renderer hash is exactly the one recorded in
the analysis-repair handoff. This was an analysis-only label repair made before
any Coin City endpoint ran.

A second pre-execution audit made the already-prespecified secondary estimands
explicit in the output contract. The renderer now reports seed-first paired
intervals for all four structure-by-domain transfer cells, absent-minus-correct
and misleading-minus-correct at every evidence depth, and the paired
`k=8`-minus-`k=0` change in misleading-hint penalty. The same estimates are stored
in `registered_results.json`; this changed no prompt, data, trained checkpoint,
primary estimand, or outcome-dependent decision.

For the verifier-aware live view, pass the final handoff explicitly:
`python monitor.py --ledger runs/coin_city_structural_20260818T203858Z_walltime30h_20260818T211828Z_effectivebarrier_20260818T212454Z_runtimefix_20260819T172806Z.json --handoff ../runs/paper_verifier_20260819T172807Z.json --watch 60`.
The strict completion audit can also be rerun manually with
`python ../verify_paper_handoff.py` and the same three effective ledgers plus the
analysis-repair handoff; `--allow-incomplete` is the read-only pre-completion
health check.
