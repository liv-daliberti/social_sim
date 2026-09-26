# Archived Experiment 3/4 status snapshot (2026-08-20)

Last verified: **2026-08-20 11:06 ET**. Refresh with the monitors at the end of
this file before acting on job-level details.

This ledger is operational. The manuscript reports completed designs and results;
it does not narrate scheduler repairs, queue transitions, or partial training
curves.

## Portfolio status

| Campaign | Complete | Running | Pending | Total |
|---|---:|---:|---:|---:|
| Structural-OOD mechanism grid | 30 | 8 | 10 | 48 |
| Polymarket 8B extension | 0 | 0 | 8 | 8 |
| Coin City A-to-B extension | 0 | 0 | 31 | 31 |
| **Scientific jobs** | **30** | **8** | **49** | **87** |

Dependent paper renderers and the terminal verifier are excluded from these
scientific-job totals.

## Priority 0 — Results complete and usable in the paper

1. **Qwen3-4B structural-OOD primary factorial.** All 12 matched cells are
   complete: causal-family and population-prior rewards, shown and hidden
   mechanism descriptions, and seeds 42/43/44. The locked hierarchical analysis
   reports repeated-draw prior-minus-causal MAE of 0.323 [0.259, 0.422] when the
   mechanism is shown and 0.340 [0.311, 0.369] when hidden. Greedy intervals
   include zero.
2. **Qwen3-4B structureless diagnostics.** All six cells are complete: both
   disclosures and all three seeds. Repeated-draw structureless-minus-causal MAE
   is 0.641 [0.551, 0.774] with structure shown and 0.677 [0.569, 0.856] when
   hidden. The hidden comparison was added to the paper on August 20.
3. **Structural-OOD untrained endpoints.** All six model-by-disclosure endpoints
   are complete for Qwen3-4B, Qwen3-8B, and Llama-3.1-8B. These are descriptive
   references, not scale or architecture estimates.
4. **Historical Polymarket Qwen3-4B triplet.** Training lowers sealed-test Brier
   score from 0.1316 to 0.1159, a difference of -0.0157
   [-0.0253, -0.0067]. Its difference from contemporaneous market prices is
   unresolved: 0.0022 [-0.0026, 0.0072].
5. **Experiments 1 and 2.** The ten-model prospective update study and six-model
   Coin City inference study are complete and included in the manuscript. Their
   shortcut-compatible interpretations remain explicit.

## Priority 1 — Finish the structural-OOD model roster

This is the highest-value running work because it unlocks the prespecified Qwen
scale comparison, the approximately matched Qwen/Llama comparison, and the full
disclosure interaction.

### Running

| Job | Cell | Progress at 11:06 ET |
|---:|---|---:|
| 30730355 | shown / Llama-8B / causal / seed 43 | 173/300 |
| 30730356 | shown / Llama-8B / causal / seed 44 | 174/300 |
| 30730357 | shown / Llama-8B / population prior / seed 42 | 150/300 |
| 30730358 | shown / Llama-8B / population prior / seed 43 | 145/300 |
| 30730359 | shown / Llama-8B / population prior / seed 44 | 144/300 |
| 30730360 | hidden / Qwen3-8B / causal / seed 42 | 124/300 |
| 30730361 | hidden / Qwen3-8B / causal / seed 43 | 46/300 |
| 30730362 | hidden / Qwen3-8B / causal / seed 44 | 3/300 |

All eight logs were advancing at the verified time. The recurring node-local
vLLM usage-statistics `ENOSPC` message is nonfatal; completed jobs have written
adapters and locked endpoint summaries to the shared filesystem.

### Pending within the roster

1. **Critical paired-cell repair:** job 30788373, shown Qwen3-8B causal seed 42.
   It is scientifically identical to the source cell that stalled during NCCL
   initialization and is waiting for an eligible A6000 allocation.
2. Hidden Qwen3-8B population-prior seeds 42/43/44: jobs 30730363--30730365.
3. All six hidden Llama-8B cells: jobs 30730366--30730371.

No new model-level claim enters the paper until the relevant causal/prior
three-seed pair is complete and its locked evaluation passes.

## Priority 2 — Polymarket 8B extension

Six scheduler-ready training jobs remain pending for Qwen3-8B and Llama-3.1-8B
at seeds 42/43/44 (30730372--30730377). Two locked-test jobs
(30730390--30730391) depend on their model's three adapters. This extension tests
whether the Qwen3-4B historical gain replicates with Qwen scale and across an
approximately matched Qwen/Llama comparison. No extension result is currently in
the paper.

## Priority 3 — Coin City A-to-B extension

All 31 jobs are dependency-gated behind the current structural and Polymarket
work:

1. Six seed-42 model-by-arm canaries.
2. One fail-closed canary gate.
3. Eighteen confirmatory runs: 3 models x 2 rewards x 3 seeds.
4. Three Qwen3-4B structureless diagnostics.
5. Three untrained endpoints.

The extension trains only on Coin City's direct response relation and evaluates an
unseen persistent-mediator mechanism in both Coin City and Coin Harbor, with
correct, absent, and misleading qualitative hints. The appendix records the
registered protocol but explicitly labels it as having no result in the present
paper.

## Priority 4 — Final analysis and paper freeze

1. Render the full structural and Polymarket roster only after every matched cell
   and locked endpoint is complete.
2. Release Coin City only if every canary satisfies the frozen adapter, parse,
   reward-variance, and length-limit gates.
3. Run the Coin City renderer after all 31 scientific jobs finish.
4. Run the terminal paper verifier against exact ledger and artifact hashes.
5. Rebuild both manuscript variants and audit every numerical sentence against
   machine-readable results.

## Authoritative monitors

Structural-OOD and Polymarket roster:

```bash
python exp3_training_transfer/monitor_campaign.py \
  --mechanism-ledger exp3_training_transfer/mechanism_family/runs/c3_mechanism_20260817T215355Z_walltime30h_20260818T211828Z_runtimefix_20260819T172806Z.json \
  --polymarket-extension-ledger exp3_training_transfer/polymarket/runs/exp3b_model_extension_20260817T215356Z_walltime30h_20260818T211828Z.json \
  --handoff exp3_training_transfer/runs/runtime_recovery_20260819T172806Z.json
```

Coin City:

```bash
python exp3_training_transfer/coin_city_structural/monitor.py \
  --ledger exp3_training_transfer/coin_city_structural/runs/coin_city_structural_20260818T203858Z_walltime30h_20260818T211828Z_effectivebarrier_20260818T212454Z_runtimefix_20260819T172806Z.json \
  --handoff exp3_training_transfer/runs/paper_verifier_20260819T172807Z.json
```
