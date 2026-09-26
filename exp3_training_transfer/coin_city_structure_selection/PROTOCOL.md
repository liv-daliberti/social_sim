# Coin City structure-selection transfer

Protocol version: `coin_city_structure_selection_v1`

Frozen **2026-09-16 UTC**, before any training job has been submitted and before
any endpoint exists. Sibling of `coin_city_structural_transfer_v1`, which it
does not modify or supersede.

## Why this experiment exists

Experiment 2 established that models select among *displayed references* when a
cue names one, that inverting the cue inverts the selection, and that the cue
selects a reference rather than triggering a prior on its words.

The parent Experiment 3 was intended to ask whether training strengthens that
operation and whether it transfers. It cannot answer that question on the
structure dimension, for a reason discovered on 2026-09-16 and recorded here
rather than in a footnote. Its training set shows two `direct_a` references
differing in *gain*, with the cue naming the gain band. Structure never varies.
"Emit zero at horizon 3" is therefore the optimal policy under the training
distribution, and the trained checkpoints adopt it unconditionally:

* measured horizon-3 output is **exactly 0.000** in every evaluation cell, on
  every one of seven seeds, in both trained arms;
* the cue moves structure choice by **−0.003 to +0.001** for trained models and
  **−0.011 to −0.003** for the base, against Experiment 2's `.51`–`.86`;
* the two trained arms are identical on structure choice to four decimal places.

The parent's structure-shift cells therefore do not measure transfer of
cue-driven selection. They measure whether a model trained exclusively on
structure A spontaneously emits structure B, which the design guarantees it will
not. This protocol makes structure a selectable dimension so the intended
question becomes answerable.

## Design

Each episode displays two reference systems that differ **only** in structure:
one `direct_a`, one `mediated_b`. Their gains are set so their horizon-1
responses coincide exactly --- a mediated system's horizon-1 response is
`lambda * gain * shock`, so the direct gain is scaled by `lambda`. Persistence
at horizon 3 is consequently the sole discriminating feature, and no model can
select on response magnitude instead of on structure.

The mediated parameters (`rho`, `phi`, `lambda`) are resampled every episode, so
a memorised fixed "mediated template" does not solve the task: the persistence
must be read off whichever reference the cue names.

A cue sentence names the structure the target follows. Reference order is
randomised. Equations, coefficients and internal structure labels are never
rendered.

### Verified solvable before building

Checked before any data was generated, because the parent's defect was an
unexamined assumption that the task was answerable:

| | Coin City | Coin Harbor |
|---|---|---|
| lagged-outcome coefficient, `direct_a` | +0.015 | +0.023 |
| lagged-outcome coefficient, `mediated_b` | +0.677 | +0.603 |
| separation $d$ | 4.47 | 3.10 |
| one-feature classifier accuracy, 14 rows | 99.0% | 97.2% |

Choosing the right structure is worth **2.18 response-MAE points** averaged over
the evaluation set --- roughly twenty times the parent's primary contrast.

## Held out from training

* **Coin Harbor entirely.** The surface-domain transfer test.
* **Arbitrary labels entirely.** Training uses only the semantic context
  sentences. Evaluation additionally presents episodes in which each structure
  is denoted by an arbitrary symbol (`ZETA`/`KAPPA`) whose mapping is redrawn
  every episode, so it can only be recovered by reading the reference
  trajectories. This is the Experiment 2 novel-induction design, and it is what
  separates relationship selection from a learned word-to-shape association.
* **Evaluation episodes**, drawn from a disjoint seed band (71M vs 61M), with
  train/eval disjointness asserted on both task identity and numeric content.

## Arms

| arm | reward |
|---|---|
| `episode_matched` | the cue-named structure's own forecast |
| `population_prior` | the equal-weight blend of both structures' responses --- the policy of ignoring the cue and hedging |

Both arms see byte-identical prompts; only the reward differs, exactly as in the
parent protocol. 4,800 training presentations per arm, 300 rounds, balanced
2,400/2,400 across target structures.

## Evaluation

Fully crossed, 2,880 tasks: 2 domains x 2 label kinds x 2 target structures x 3
cues (correct / none / misleading) x 4 evidence depths x 30 episodes. Every cell
holds 120 tasks. Scored with the parent's frozen five-draw stochastic decoder at
temperature 0.7 and the parent's `report.py`, unmodified.

## Primary endpoint

The parent's `response_mae` conflates magnitude error with structure error, and
its `structure_correct` is a brittle nearest-template label. This protocol's
primary measure is continuous and reads the structure dimension directly.

For each episode define the **implied persistence ratio**

    p_hat = mean |predicted horizon-3 response| / mean |predicted horizon-1 response|

whose true value is 0 for `direct_a` and approximately 0.65 for `mediated_b`.
The **cue-following index** is

    D = mean p_hat given the cue names mediated_b
      - mean p_hat given the cue names direct_a

A model that selects on the cue has `D` near 0.65; one that ignores it has
`D` near 0. The parent's trained checkpoints have `D = 0.00` by construction,
and its base has `D = -0.01`.

**Primary:** `D` for `episode_matched` on Coin Harbor with semantic labels ---
domain transfer of cue-driven structure selection --- against the untrained base.

**Secondary, pre-specified:** `D` on Coin City (in-distribution); `D` under
arbitrary labels in both domains (induction); `episode_matched` minus
`population_prior` on `D`; response MAE and structure accuracy in every cell;
and the graded version, the correlation across episodes between `p_hat` and the
cue-named reference's own persistence ratio, which is the direct analogue of
Experiment 2's implied-slope correlation.

Estimators are the four fixed for the parent's eight-seed amendment:
hierarchical seed-first paired bootstrap, Student-$t$ on seed-level means, wild
cluster bootstrap with Webb weights imposing the null, and the exact one-sided
seed-level sign test.

## Seeds

Three seeds (42, 43, 44) per arm, declared a **pilot**. The trained-versus-base
comparison has an expected effect of roughly 0.65 against a between-seed
standard deviation measured at about 0.2 in this pipeline, so three seeds
resolve it decisively if it exists.

Fixed now: **no claim about the `episode_matched` minus `population_prior`
contrast enters a manuscript at three seeds.** That contrast is the fine one and
this session established what three seeds do to fine contrasts; it requires
eight seeds under a separate amendment before it is reported. The pilot
establishes existence and magnitude of cue-following, nothing else.

## What would falsify the claim

If `D` stays near zero after training on structure-varying data, then models do
not acquire cue-driven structure selection from this kind of supervision, and
the Experiment 2 selection result does not extend from gain to functional form.
That is a publishable negative and is reported as such.

If `D` is large on Coin City but collapses on Coin Harbor, selection is learned
but surface-bound. If it survives semantic labels but collapses under arbitrary
ones, what is learned is a word-to-shape association rather than reference
selection --- the reading Experiment 2 was designed to exclude.

## Allocation and wrapper

A6000, 2 GPUs, `allcs`, matching the parent's profile. Checkpointing flags are
passed **explicitly** in the environment rather than left to `train.sh`
defaults, because the parent protocol was silently split across two wrapper
versions when those defaults changed on 2026-08-28
(`../five_seed_extension/WRAPPER_DEVIATION_RECORD.md`). Every submission for
this protocol records `SAVE_STEPS` and `SAVE_CKPT` in its ledger.

## Boundary

This protocol does not modify, re-analyse or supersede
`coin_city_structural_transfer_v1`. The parent's reported results stand as what
that design measured. This is a new experiment answering the question the parent
was intended to answer.
