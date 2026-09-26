# Disclosed interim analysis at G=7

Record version: `exp3_qwen3_8b_g7_interim_look_v1`

Recorded on **2026-09-16 UTC**, *before* any seed-level value, contrast,
interval, or test statistic from the added seeds was computed, printed, or
looked at by anyone.

Amends: `exp3_qwen3_8b_coin_city_eight_seed_v1`
(`QWEN3_8B_EIGHT_SEED_AMENDMENT.md`).

## What is being done and why

The eight-seed amendment fixes $G=8$ and states that the analysis is run once,
after all ten added training jobs have produced endpoints, and that the
analysis is not run at intermediate $G$.

Nine of the ten added jobs completed. The tenth, `causal` seed 48
(job 31281196), died of a sporadic native crash and is being rerun unchanged as
job 31305731 under the amendment's failure rule. Because the transfer contrast
is paired within training seed, the currently analyzable roster is the seven
seeds both arms share: **42, 43, 44, 45, 46, 47, 49**. Seed 48 is excluded from
this interim look in both arms, including the `population_prior` seed 48
endpoint, which completed successfully.

At the author's direction an interim analysis is being run at $G=7$ today,
ahead of the $G=8$ analysis. This departs from the recorded stopping rule. It
is recorded here rather than left implicit.

## Binding commitments

These are fixed now, before the interim numbers exist.

1. **$G=8$ remains the primary reported result.** The interim $G=7$ estimate is
   secondary and is labeled as an interim look wherever it appears.
2. **The interim is reported regardless of direction.** If it disagrees with
   $G=8$, points the other way, or is less favorable, it is still reported, and
   this record is still cited.
3. **Nothing may be changed on the basis of it.** Not the seed count, not the
   stopping rule, not the four pre-specified estimators, not the transfer cells,
   not the primary cell, not the bootstrap seeds, not the reporting boundary,
   and not the decision to rerun seed 48. The seed-48 rerun proceeds to
   completion whatever the interim shows, and its result enters the $G=8$
   analysis unconditionally.
4. **No further seed is added on the basis of it**, in either direction.
5. **The $G=8$ analysis is not conditioned on it.** It is run with the same code
   and the same bootstrap seeds, differing only in the seed list.

## What this costs, stated plainly

A result reported at two different $G$ values invites the reading that the
larger one was chosen because it looked better. The commitments above are what
separate a disclosed interim look from optional stopping, and they are only
worth anything if both numbers are reported. Both will be.

The honest summary for a reader is: the seed count was fixed at eight in
advance, one training job crashed, the authors looked at the seven-seed result
while the replacement was rerunning, and the eight-seed result is the one the
manuscript claims.

## Analysis identity

The interim uses `make_qwen3_8b_eight_seed_outputs.py` with
`--seeds 42 43 44 45 46 47 49`. The $G=8$ analysis will use the same script,
same bootstrap seeds, and `--seeds 42 43 44 45 46 47 48 49`. The script
reproduces the published three-seed estimates exactly when given
`--seeds 42 43 44`, which is its regression test against the registered
`registered_results.json`.

Outputs are written to distinct paths so neither overwrites the other:

```
reports/qwen3_8b_step300_stochastic_g7_interim.json
reports/qwen3_8b_step300_stochastic_eight_seed.json
```
