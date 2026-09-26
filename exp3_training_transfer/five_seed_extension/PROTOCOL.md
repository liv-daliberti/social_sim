# Experiment 3/4 five-seed precision extension

Protocol version: `exp3_exp4_five_seed_extension_v1`

Frozen on **2026-08-25 UTC**, after results from several original three-seed
rosters were known but before seeds 45 or 46 were trained for any endpoint in
this extension.  This is a prospective *additional-seed precision extension*;
it does not retroactively turn the original three-seed analyses into
five-seed preregistrations.  Every added seed is reported regardless of
direction, parse rate, or agreement with seeds 42--44.

## Replication unit and fixed seeds

The two added training seeds are exactly **45 and 46**.  Together with the
original seeds 42, 43, and 44, the updated descriptive and inferential roster
uses five independent training runs per trained cell.  Decode draws remain
within-checkpoint repeated measurements and are never counted as training
replications.  Untrained bases and hosted API endpoints do not acquire a
fictitious seed dimension.

No data, prompt, reward, optimizer, learning rate, LoRA setting, rollout
recipe, training budget, checkpoint revision, decoder, or endpoint changes.
Each new job reuses the exact parent wrapper and allocation profile, except
for two copied wrappers whose original files are hash-bound to active gates;
those copies differ only by admitting seeds 45/46 and recording this amendment.

## Exact added training roster

### Experiment 3

| Roster | Added cells | Seeds/cell | Jobs |
|---|---:|---:|---:|
| Mechanism-family current: 2 disclosures x 3 models x 2 confirmatory arms | 12 | 2 | 24 |
| Mechanism-family Qwen3-4B structureless diagnostic: 2 disclosures | 2 | 2 | 4 |
| Mechanism-family large scale: 2 disclosures x 3 models x 2 arms | 12 | 2 | 24 |
| Coin City current: 3 models x 2 confirmatory arms | 6 | 2 | 12 |
| Coin City Qwen3-4B structureless diagnostic | 1 | 2 | 2 |
| Coin City Qwen3-14B scale: 2 confirmatory arms | 2 | 2 | 4 |
| **Experiment 3 total** | **35** | **2** | **70** |

The 48 added mechanism-family confirmatory endpoints (24 current and 24
large-scale) also receive the frozen v2 final-token latent probe.  The
structureless diagnostics are behavioral controls and remain outside that
probe estimand, matching the parent protocol.

### Experiment 4

Seeds 45/46 are added for each locally trained checkpoint: Qwen3-1.7B,
Qwen3-4B-Instruct-2507, Qwen3-8B, Qwen3-14B, Qwen3-32B,
Llama-3.2-3B-Instruct, and Llama-3.1-8B-Instruct.  This is 14 training jobs.
Each five-seed checkpoint is evaluated on the existing 318-market
family-disjoint endpoint with the frozen five-draw stochastic decoder.  The
hosted provider panel is excluded because it has no experimental training
seed; its five calls per task are decode replicates, not seed replicates.

## Gates and dependencies

Completed parent feasibility gates are inherited without rerunning canaries.
The still-pending large mechanism-family and Qwen3-32B additions are locked
behind their already-submitted audited advance jobs.  A successful Slurm exit
alone is not substituted for those artifact gates.  New latent probes depend
on their exact training job, and five-seed Exp4 evaluations depend on all five
model-specific adapters.

## Reporting boundary

Primary reporting preserves the original three-seed estimates and labels the
five-seed result as a later precision extension.  A combined five-seed estimate
must show all per-seed values, use seed-first resampling, and disclose that
seeds 45/46 were selected after some parent results were observed.  The added
seeds improve Monte Carlo precision and robustness checks; they do not solve
learning-rate-by-scale confounding or license post-result hyperparameter
tuning.
