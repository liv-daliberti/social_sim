# Qwen3-4B Coin City eight-seed robustness amendment

Amendment version: `exp3_qwen3_4b_coin_city_eight_seed_v1`

Recorded on **2026-09-16 UTC**, before any seed 45--49 training job has been
submitted for the Qwen3-4B Coin City structural roster and before any
added-seed Qwen3-4B endpoint exists.

Parent protocol: `exp3_exp4_five_seed_extension_v1`
(`PROTOCOL.md`).
Sibling amendment: `exp3_qwen3_8b_coin_city_eight_seed_v1`
(`QWEN3_8B_EIGHT_SEED_AMENDMENT.md`).
Interim record that triggered this one: `exp3_qwen3_8b_g7_interim_look_v1`
(`G7_INTERIM_LOOK_DISCLOSURE.md`).

## What this amendment changes

The Qwen3-4B Coin City confirmatory arms go from three training seeds to
**eight**: added seeds are exactly **45, 46, 47, 48, and 49**, in each of the
`causal` (episode-matched) and `population_prior` arms. Ten training jobs.

## Why, stated as it actually happened

This is not a power calculation and is not a bid for significance. It is a
robustness check on a result now suspected of being overstated.

The Qwen3-8B roster was extended from three seeds to eight under the sibling
amendment. Its disclosed $G=7$ interim, computed on 2026-09-16, moved the
primary joint-cell contrast from $.251$ $[.058,.470]$ at three seeds to $.108$
$[-.035,.265]$ at seven, with the hierarchical interval no longer excluding
zero, the structure cell changing sign, and seed-level sign agreement falling
from 12 of 12 to 22 of 28. The added seeds contained genuinely negative draws.
The three-seed Qwen3-8B estimate was a favorable draw, and the between-seed
standard deviation was larger than three seeds could reveal.

The published Qwen3-4B comparator is a three-seed estimate reporting all four
transfer-cell intervals excluding zero. It was produced by the same pipeline,
on the same task universe, at the same seed count, from the same three seed
values. It is exposed to the same failure mode. The manuscript currently reads
the two Qwen sizes as mutually corroborating; that reading is only available if
both are measured at a seed count capable of supporting it.

So this extension exists because the three-seed Qwen3-4B result may be wrong in
the same direction, and the manuscript should not continue to lean on it
untested.

## Recorded expectation, before the data exist

We expect the Qwen3-4B contrast to shrink materially relative to its three-seed
value, for the same reason Qwen3-8B's did. We record that expectation here so
that neither outcome can later be presented as the one we predicted. If it does
not shrink, that is a genuine finding and is reported as such; if it shrinks to
nothing, that is equally reported.

## Binding commitments

1. **$G = 8$ is fixed now.** All ten jobs are submitted as one roster. The
   analysis is run once, after all ten produce endpoints or the roster is
   closed under the failure rule.
2. **The eight-seed Qwen3-4B result replaces the three-seed estimate** as the
   reported comparator, in whichever direction it falls.
3. **No seed is added or dropped on the basis of the result**, and no seed is
   discarded for being an outlier or for disagreeing with seeds 42--44.
4. **The estimators are fixed** and are the four already pre-specified for
   Qwen3-8B: hierarchical seed-first paired bootstrap; Student-$t$ on the
   seed-level means; wild cluster bootstrap with Webb six-point weights
   imposing the null; exact one-sided seed-level sign test. Bootstrap seeds and
   repetition counts are unchanged.
5. **Failure rule**, unchanged from the sibling amendment: an infrastructural
   failure is resubmitted unchanged up to three attempts and recorded; a
   completed seed is never discarded; if a seed cannot be completed the roster
   is reported at whatever $G$ was achieved with the missing seed named.

## Scope

Two confirmatory arms only. The Qwen3-4B `structureless` diagnostic
(seeds 42--44, jobs 30723998--30724000) is **not** extended. It is a
behavioral control on whether training without structure helps, not a term in
the episode-matched-minus-population-prior contrast, and the sibling amendment
set the same boundary. Consequence, stated so it is not discovered later: any
figure or table placing the structureless arm beside the confirmatory arms will
be comparing a three-seed control against eight-seed confirmatory arms, and
must say so.

The untrained Qwen3-4B base (`base_qwen3_4b_j30724001`) is not reseeded; an
untrained base has no training seed dimension.

## Exact added roster

| Arm | Model | Seeds | Jobs |
|---|---|---|---:|
| `causal` | `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f17c01a7cc42f958dc650907174af0554` | 45, 46, 47, 48, 49 | 5 |
| `population_prior` | same | 45, 46, 47, 48, 49 | 5 |

Combined eight-seed roster, 17 endpoint files: the base, `causal` and
`population_prior` at seeds 42--44 (jobs 30723979--30723984), and the ten added
above.

## Superseded on 2026-09-16: training wrapper deviation

The claim below that added jobs reuse `train.sh` at a pinned hash is **not**
equivalent to reusing the wrapper seeds 42--44 trained under. `train.sh` was
modified on 2026-08-28: `--save_steps` went from `999999` to a `40` default and
`--save_ckpt` was added, both applying silently because `coin_env` sets
neither. Read this section together with `WRAPPER_DEVIATION_RECORD.md`
(`exp3_coin_city_wrapper_deviation_v1`), which records the deviation, the
crossover control measuring its effect, and the corrective action taken.

## Nothing scientific changes

No change to data, prompts, reward, optimizer, learning rate, LoRA rank or
alpha, rollout recipe, 4,800-presentation budget, 300 rounds, checkpoint
revision, decoder, five-draw stochastic endpoint, temperature, or the
1,440-task held-out universe. Each added job reuses `train.sh` at its pinned
hash, which trains and then writes and scores both endpoints, so one job per
cell is the complete unit.

## Allocation

A6000 via `allcs`, matching seeds 42--44, which ran on node207 (`causal`) and
node208 (`population_prior`), both A6000, for 13.0--13.6 hours each. Added
seeds request the same CPU count, memory, GPU type, GPU count, node exclusion
and walltime, and additionally pass `--account=allcs --qos=medium`. As recorded
for Qwen3-8B, a site `job_submit` plugin resolves the partition to `cs`
regardless of what is requested. All exported environment variables are
byte-identical to the seed-42--44 submissions except `SEED` and the derived
`TAG`, verified against the parent ledger at submission time.

## Reporting boundary

The three-seed Qwen3-4B estimate remains in the record as what was originally
computed. The eight-seed estimate is what the manuscript reports, labeled as a
later robustness extension, disclosing that it was run after the Qwen3-8B
interim indicated the three-seed estimates were optimistic.

This extension improves the precision and honesty of the Qwen3-4B comparator.
It does not address the learning-rate-by-scale confound, does not license
post-result hyperparameter tuning, and does not extend to the Llama-3.1-8B
comparator, the structureless diagnostic, or any mechanism-family roster, all
of which remain at three seeds and must be described as such.
