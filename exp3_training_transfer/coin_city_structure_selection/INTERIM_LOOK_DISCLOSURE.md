# Disclosed mid-training interim look

Record version: `coin_city_structure_selection_interim_v1`

Recorded **2026-09-20 UTC**, before any cue-following index was computed from
any trained checkpoint of this protocol.

Amends: `coin_city_structure_selection_v1` (`PROTOCOL.md`).

## What is being done

The protocol fixes one analysis, at round 300, on the five-draw stochastic
endpoint. At the author's direction an interim look is taken now, mid-training,
from the temperature-zero greedy monitors that the training loop writes every
50 rounds. This departs from the recorded analysis plan and is recorded here
rather than left implicit.

At the time of this record the roster stands at:

| job | cell | deepest monitor |
|---|---|---|
| 31366205 | `causal` s42 | step 150 |
| 31366206 | `causal` s43 | step 150 |
| 31366207 | `causal` s44 | step 100 |
| 31366208 | `population_prior` s42 | step 50 |
| 31366209 | `population_prior` s43 | not started |
| 31366210 | `population_prior` s44 | not started |

The `population_prior` arm therefore has one seed at one shallow step and
cannot support any arm contrast. Only the `causal` arm against the untrained
base is readable, and only to the depth its seeds share.

## What these numbers are not

* **Not the registered endpoint.** Greedy, temperature zero, one draw per task,
  against a protocol that reports five stochastic draws at temperature 0.7.
* **Not round 300.** Training is half done; a policy still moving.
* **Not an arm contrast.** The control arm is one seed deep.

One property does favour the interim: the cue-following index is computed from
`predicted_response`, which is absent on an unparsed draw, so unparsed draws
drop out rather than taking the 280-point clip penalty. The parse-penalty
leverage that made the parent protocol's greedy monitors swing by +/-4.8 MAE
points between adjacent steps does not propagate into this statistic. It is
noisy from single-draw decoding and from being mid-training, not from parse
failures.

## Binding commitments

Fixed now, before the numbers exist.

1. **The round-300 five-draw analysis remains the reported result**, run once
   when the roster completes, exactly as the protocol specifies.
2. **This interim changes nothing**: not the seed count, not the stopping rule,
   not the estimators, not the primary endpoint, not the decision to run the
   `population_prior` arm to completion. Every queued job proceeds unchanged
   whatever this shows.
3. **It is reported regardless of direction**, and this record is cited wherever
   it appears. If it disagrees with the round-300 result, both are shown.
4. **No new seed is added on the basis of it**, in either direction.

## Reference point

The untrained base was evaluated on the same 2,880-task set (job 31366588,
2,880 greedy and 14,400 stochastic rows, 99.91% parse). Its cue-following index
is +0.020 and -0.005 on Coin City under semantic and arbitrary labels, and
+0.068 and +0.002 on Coin Harbor. Implied persistence is 1.05-1.21 in every
condition against a truth of 0.00 for direct targets and ~0.65 for mediated:
the untrained model predicts full persistence regardless of cue or truth.

A trained model that has learned cue-driven structure selection would show an
index approaching 0.65. The parent protocol's trained checkpoints sit at 0.000.
