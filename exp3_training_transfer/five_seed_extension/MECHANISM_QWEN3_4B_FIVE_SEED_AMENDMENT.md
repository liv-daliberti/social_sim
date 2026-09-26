# Mechanism-family Qwen3-4B five-seed amendment

Amendment version: `exp3_mechanism_qwen3_4b_five_seed_v1`

Recorded on **2026-09-16 UTC**, before any seed 45 or 46 mechanism-family
training job has been submitted and before any added-seed mechanism endpoint
exists.

Parent protocol: `exp3_exp4_five_seed_extension_v1` (`PROTOCOL.md`), whose
`exp3_mechanism_current` roster already specifies seeds 45 and 46 for this
cell. This amendment does not add seeds beyond that roster; it narrows the
submission to the four Qwen3-4B cells and records why those are being run
first, together with the evidence that motivated it.

## What is being added

Four cells, two added seeds each, eight training jobs:

| disclosure | arm | model | seeds |
|---|---|---|---|
| disclosed | `causal_family` | Qwen3-4B-Instruct-2507 | 45, 46 |
| disclosed | `population_prior` | same | 45, 46 |
| undisclosed | `causal_family` | same | 45, 46 |
| undisclosed | `population_prior` | same | 45, 46 |

Taken with the registered seeds 42--44 this gives $G = 5$ per cell.

## Why these four cells, and why now

Applying the four estimators fixed for the Coin City eight-seed amendment to
the registered mechanism-family endpoints
(`mechanism_family/project_mechanism_seed_power.py`) gives:

| cell | mean | sd | Student-$t$ df=2 | wild cluster $p$ | min $G$ |
|---|---:|---:|---|---:|---:|
| Qwen3-4B disclosed | $+.323$ | $.078$ | $[+.130,+.516]$ excl 0 | $.111$ | 3 |
| Qwen3-4B undisclosed | $+.340$ | $.026$ | $[+.276,+.403]$ excl 0 | $.028$ | 3 |
| Qwen3-8B disclosed | $-.125$ | $.093$ | spans 0 | $.096$ | 5 |
| Qwen3-8B undisclosed | $-.013$ | $.048$ | spans 0 | $.786$ | 56 |
| Llama-3.1-8B disclosed | $+.035$ | $.778$ | spans 0 | $.933$ | $>60$ |
| Llama-3.1-8B undisclosed | $+1.093$ | $1.153$ | spans 0 | $.121$ | 7 |

Qwen3-4B undisclosed is the only cell anywhere in Experiment 3 that excludes
zero under the conservative Student-$t$ at three seeds --- the alternative the
manuscript's Coin City appendix reports as excluding zero for none of its
twelve transfer cells --- and it also clears the wild cluster bootstrap. Its
between-seed standard deviation, $.0255$, is an order of magnitude below Coin
City's $.205$.

The one thing it cannot have at three seeds is the distribution-free evidence.
An exact one-sided sign test on three unanimous seeds returns $p = .125$ and
cannot return less; the statistic is floored by $G$, not by the data. Five
seeds make a unanimous result $p = .031$. That is the entire purpose of this
amendment: to supply the one form of evidence the current seed count makes
impossible, for the strongest result in the experiment.

## Binding commitments

1. **$G = 5$ is fixed.** All eight jobs are submitted as one roster; the
   analysis is run once, after all eight produce endpoints or the roster is
   closed under the failure rule.
2. **Reported regardless of direction.** If the added seeds move the estimate
   toward or across zero, that is the reported result, and this record is
   cited. The three-seed estimate remains in the record as what was originally
   computed.
3. **Estimators are fixed** and unchanged: hierarchical seed-first paired
   bootstrap, Student-$t$ on seed-level means, wild cluster bootstrap with Webb
   six-point weights imposing the null, exact one-sided seed-level sign test.
   Bootstrap seed 20260818, 5,000 and 9,999 repetitions.
4. **No seed is added or dropped on the basis of the result**, and no completed
   seed is discarded for disagreeing with seeds 42--44.
5. **Failure rule**, as for the sibling amendments: an infrastructural failure
   is resubmitted unchanged up to three attempts and recorded; if a seed cannot
   be completed, the roster is reported at whatever $G$ was achieved with the
   missing seed named.

## Disclosure

The seed count was chosen after the three-seed results were observed, and these
four cells were selected because they are the strongest in the experiment. Both
facts are stated wherever the five-seed result is reported. This is a precision
extension on a result that already clears its interval-based tests, not a
search for one that does not.

Not extended here, and remaining at three seeds: the Qwen3-8B cells (one of
which points in the opposite direction, $-.125$ with 0 of 3 seeds positive),
the Llama-3.1-8B cells, the `structureless` diagnostics, and the large-scale
mechanism roster. All must be described as three-seed wherever they appear.

**The manuscript must not claim cross-size corroboration in Experiment 3.**
Qwen3-8B disclosed points opposite to Qwen3-4B, and the Coin City Qwen3-8B
contrast fell from $+.251$ to $+.108$ spanning zero at seven seeds. There is no
cell in this experiment where the two Qwen sizes agree at a defensible seed
count.

## Wrapper

`mechanism_family/mechanism_rl.sh`, unmodified, matching git HEAD with no
working-tree edit. Verified against the behaviour of the registered runs rather
than assumed: the wrapper defaults `SAVE_STEPS` to `999999`, passes no
`--save_ckpt`, and hardcodes `--enable_prefix_caching`; the registered Aug-18
and Aug-19 endpoints contain zero `checkpoints/` directories and a single
`saved_models/step_00301`, exactly as those flags imply.

This check exists because the Coin City rosters were silently split across two
wrapper versions when `train.sh` defaults changed on 2026-08-28
(`WRAPPER_DEVIATION_RECORD.md`). `SAVE_STEPS` is nonetheless passed explicitly
in every submission under this amendment so the value is recorded rather than
inherited.

## Allocation

Matching the registered seed-42 submissions exactly:
`--partition=all --gres=gpu:a6000:2 --cpus-per-task=8 --mem=100G
--time=20:00:00 --exclude=node206`. All exported environment variables are
byte-identical to those submissions except `SEED`, verified against the parent
ledger at submission time.
