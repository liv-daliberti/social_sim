# Qwen3-8B Coin City eight-seed precision amendment

Amendment version: `exp3_qwen3_8b_coin_city_eight_seed_v1`

Recorded on **2026-09-14 UTC**, before any seed 45, 46, 47, 48, or 49 training
job has been submitted for the Qwen3-8B Coin City structural roster, and before
any added-seed endpoint exists anywhere in Experiment 3.

Parent protocol: `exp3_exp4_five_seed_extension_v1`
(`exp3_training_transfer/five_seed_extension/PROTOCOL.md`,
sha256 `4c485b67a497fb36a3b866a964090af067e761bcf9512aca7e9723ebcf9d89ed`).

Parent allocation amendment: `RECOVERY_AMENDMENT.md`,
sha256 `5ec19af03802bbe2fe6e2081402a3d8b3bbe9da77df5e100dbe73820adde4f7e`.

Training wrapper: `exp3_training_transfer/coin_city_structural/train.sh`,
sha256 `7ca287c389ff4f18678de9e381e5de6b0064cdd2635eebbfab3a749476685a21`.

## What this amendment changes

The parent protocol adds exactly seeds 45 and 46 to every trained cell in
Experiments 3 and 4, giving five training seeds per cell. This amendment raises
the added-seed count **for one roster only** — the Qwen3-8B Coin City
structural confirmatory arms, which carry the Experiment 3 primary endpoint and
Figure~8 — from two added seeds to five. The added seeds for that roster are
exactly **45, 46, 47, 48, and 49**, giving eight training seeds per cell
together with the original 42, 43, and 44.

Every other roster in the parent protocol is unchanged and still adds exactly
seeds 45 and 46. No roster loses a seed.

## Disclosure: this seed count was chosen from observed results

This is not a blinded design decision and is not presented as one.

Seeds 42--44 of this roster were complete, analyzed, and written up before this
amendment was recorded. The published appendix
(`paper/experiment3_appendix.tex`, "Sensitivity of the intervals to three
training seeds") reports that under a Student-$t$ interval on three seed-level
means with two degrees of freedom, none of the twelve transfer-cell intervals
excludes zero, and states that no single interval in the experiment is claimed
robust to the choice of small-cluster procedure.

The seed count in this amendment was chosen by projecting that same $t$
procedure forward at larger $G$, holding the between-seed standard deviation
fixed at its three-seed estimate. The projection is reproducible from the
registered endpoints alone:

```
python exp3_training_transfer/five_seed_extension/project_seed_power.py \
    --model qwen3_8b --project 5 6 8 9
```

It recomputes the seed-level contrasts from the registered
`stochastic_n5.scores.jsonl` files, reproduces the published
`.251 [-.259, .762]` primary interval exactly, and gave:

| Cell | mean | seed SD | projected $t$ at $G=5$ | projected $t$ at $G=8$ | smallest $G$ excluding zero |
|---|---:|---:|---|---|---:|
| City / A (ID) | $+.081$ | $.100$ | $[-.043, +.206]$ | $[-.002, +.165]$ | 9 |
| Harbor / A (domain) | $+.241$ | $.281$ | $[-.107, +.590]$ | $[+.007, +.476]$ | 8 |
| City / B (structure) | $+.029$ | $.015$ | $[+.011, +.047]$ | $[+.016, +.041]$ | 4 |
| Harbor / B (joint, primary) | $+.251$ | $.205$ | $[-.004, +.506]$ | $[+.080, +.423]$ | 6 |

Eight was selected as the smallest count that the projection places outside
zero in the primary joint cell and in the domain cell, which is the boundary at
which the added seeds change what the manuscript can state rather than only
narrowing an interval it already reports as inconclusive.

Two limits of that projection are stated here rather than discovered later. A
standard deviation estimated from three values is itself very imprecise — its
95% interval spans roughly $0.5\times$ to $6\times$ the point estimate — so
these projections are indicative and carry no guarantee. And the ID cell is not
projected to exclude zero at eight seeds under this procedure; that is accepted
in advance, not treated as a failure of the extension.

## Fixed stopping rule

$G = 8$ is fixed now. All ten added training jobs are submitted as one roster.

The analysis is run **once**, after all ten have produced endpoints, or after
the roster is declared closed under the failure rule below. Seeds are not
inspected as they land, the analysis is not run at intermediate $G$, and no
further seed is added to this roster on the basis of the result — in either
direction. If the eight-seed intervals span zero, that is the reported result.

Every added seed is reported regardless of direction, parse rate, or agreement
with seeds 42--44. No added seed is dropped for being an outlier.

**Failure rule.** A training job that fails for an infrastructural reason
(scheduler rejection, node fault, preemption, wrapper or environment error) is
resubmitted unchanged, up to three attempts, and the attempts are recorded. A
seed whose training completes is never discarded. If a seed cannot be completed
after three attempts, the roster is reported at whatever $G$ was achieved, with
the missing seed and its failure named.

## Exact added roster

Ten training jobs: 5 seeds $\times$ 2 confirmatory arms $\times$ Qwen3-8B.

| Arm | Model | Seeds | Jobs |
|---|---|---|---:|
| `causal` (episode-matched) | `Qwen/Qwen3-8B` @ `b968826d9c46dd6066d109eabc6255188de91218` | 45, 46, 47, 48, 49 | 5 |
| `population_prior` | same | 45, 46, 47, 48, 49 | 5 |

The untrained Qwen3-8B base endpoint (`base_qwen3_8b_j30724002`) is not
reseeded; an untrained base has no training seed dimension. The Qwen3-4B
structureless diagnostic is a behavioral control on a different model and is
outside this amendment.

Combined eight-seed roster for the primary endpoint, 17 endpoint files:

```
base_qwen3_8b_j30724002                                  (1 base)
causal_qwen3_8b_s{42,43,44}_*_j{30730378,30730379,30730380}
population_prior_qwen3_8b_s{42,43,44}_*_j{30730381,30730382,30730383}
causal_qwen3_8b_s{45,46,47,48,49}                        (this amendment)
population_prior_qwen3_8b_s{45,46,47,48,49}              (this amendment)
```

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
revision, decoder, five-draw stochastic endpoint, temperature, or the 1,440-task
held-out universe. Each added job reuses `train.sh` at the hash pinned above,
which trains and then writes both the greedy and the `stochastic_n5` endpoint
and scores them, so one job per cell is the complete unit.

## Allocation

Added seeds run on **A6000 GPUs via `allcs`/`cs`**, matching the GPU
architecture of seeds 42--44, which ran on A6000 throughout (node103, node104,
node205, node207). No GPU architecture is mixed within this estimand,
so no hardware term is introduced and none has to be disclosed.

Seeds 42--44 were submitted as:

```
sbatch --partition=cs --cpus-per-task=8 --mem=100G \
       --gres=gpu:a6000:2 --exclude=node206 --time=1-06:00:00
```

Added seeds use the same CPU count, memory, GPU type, GPU count, node
exclusion, and walltime (`30:00:00` is `1-06:00:00`), and additionally pass
`--account=allcs --qos=medium` and `--partition=cs,all`. All exported
environment variables are byte-identical to the seed-42--44 submissions except
`SEED` and the derived `TAG`; this is verified against the original ledger on
every submission by `submit_qwen3_8b_eight_seed.py`.

**Partition routing.** Seeds 42--44 were submitted to `--partition=cs` and ran
on node103, node104, node205, and node207. Partition membership has since
changed: `cs` is now `node[202-207]`, so with node206 excluded and node205
drained (`NHC: check_nv_smi_temp: 5 GPUs are overheated`) it offers a single
usable A6000 host, node207, against the 20 GPUs this roster needs.

The roster was submitted twice. Job IDs 31281157--31281166 (`--partition=cs`)
were cancelled while all ten were still `PENDING`, before any job started, and
resubmitted as 31281193--31281202 with `--partition=cs,all` to try to reach the
A6000 hosts that ran seeds 42--44 and now sit in `all`.

That resubmission changed nothing and is recorded for completeness, not as a
fix. A site `job_submit` plugin forces `Partition=cs` on this account: the
submit line records `--partition=cs,all` but Slurm resolves the job to `cs`,
`scontrol update partition=` is refused outright (`Partition may not be
modified after submission`), and `sbatch --test-only` returns the identical
start estimate for `cs`, `all`, `cs,all`, and `all,cs`, and also with the
node206 exclusion removed. The queue position of the second submission is
therefore the same as the first would have been. No training, no endpoint, and
no seed was produced or discarded by either submission.

`--gres=gpu:a6000:2` is unchanged throughout and constrains every job to an
A6000 host, so no GPU architecture change was introduced at any point.

## Pre-specified analysis

Specified before any added endpoint exists, so the choice of estimator cannot
follow the data. All four are reported for all four transfer cells, whatever
they show.

1. **Hierarchical seed-first bootstrap**, unchanged from the parent analysis:
   5,000 repetitions, bootstrap seed 20260818, seeds resampled at the top
   level and paired episodes within each selected seed, draws averaged within
   episode, parse failures penalized at the full observable-range clip width.
   This remains the interval printed in Figure~8.
2. **Student-$t$ interval** on the eight seed-level means with seven degrees of
   freedom, the conservative alternative the appendix already reports at
   $G=3$, carried forward unchanged.
3. **Wild cluster bootstrap** on the seed-level contrasts with Webb six-point
   weights, 9,999 repetitions, bootstrap seed 20260818, imposing the null.
   This is the few-clusters remedy from Cameron, Gelbach and Miller (2008)
   (`cameron2008bootstrap`) — the reference the appendix already cites for
   the $G=3$ problem — and it is
   added here because that citation currently names the problem without
   reporting the recommended correction.
4. **Exact seed-level sign test** on the primary joint-cell contrast,
   one-sided in the pre-registered direction (episode-matched below
   population-prior), reported with its exact binomial $p$. This is the only
   one of the four that requires no variance estimate, and the appendix
   already identifies sign consistency rather than any interval as the
   principal evidence. At $G=8$ a unanimous result gives $p = 2^{-8} = .0039$;
   at $G=3$ the same statistic could not fall below $p = .125$.

Reported seed-level values are shown individually for all eight seeds in every
cell, per the parent protocol's requirement for combined estimates.

## Reporting boundary

Primary reporting preserves the original three-seed estimates. The eight-seed
result is labeled a later precision extension and discloses that its seed count
was selected after the three-seed results were observed and from a power
projection computed on them.

These seeds improve Monte Carlo precision and the credibility of the
small-cluster inference. They do not address the learning-rate-by-scale
confound, do not license post-result hyperparameter tuning, do not extend to
the Llama-3.1-8B comparator or to any other Experiment 3 roster, and do not
convert the original three-seed analysis into an eight-seed preregistration.

## Record

Machine-readable record of this amendment, written before submission:
`exp3_training_transfer/five_seed_extension/runs/qwen3_8b_eight_seed_amendment_20260914T211120Z.json`.
It pins the roster, the allocation, the stopping rule, the four pre-specified
analyses, and the sha256 of this document and of every file it binds.
