# Contradiction-resolution seed extensions

Amendment version: `exp3_contradiction_resolution_v1`

Recorded **2026-09-19 UTC**, before either roster below has been submitted.

Parent protocol: `exp3_exp4_five_seed_extension_v1` (`PROTOCOL.md`), whose
seeds-45/46 roster already covers both cells. This amendment does not add seeds
beyond that roster; it narrows submission to two cells and records why they are
being run now.

## The two cells

| roster | cell | current | seeds positive | min $G$ | jobs |
|---|---|---|---:|---:|---:|
| Coin City | Llama-3.1-8B, in-distribution | $-0.351$ | 0 of 3 | 4 | 4 |
| Mechanism family | Qwen3-8B, disclosed | $-0.125$ | 0 of 3 | 5 | 4 |

Both go to $G = 5$ by adding seeds 45 and 46 in each of the two arms.

## Why: these resolve disagreements that already exist in the data

Neither extension is a search for significance. Both target cells whose current
estimate **contradicts** a claim the manuscript rests on, and in both the
contradiction is what needs resolving, whichever way it lands.

**Coin City Llama.** After the eight-seed extensions, the only surviving Coin
City claim is the in-distribution contrast. Qwen3-8B gives $+0.060$ clearing all
four estimators at $G=8$; Qwen3-4B gives $+0.067$ but spans zero with 4 of 8
seeds positive. Llama-3.1-8B gives $-0.351$ with all three seeds negative and an
interval excluding zero on the negative side --- opposite in sign and larger in
magnitude than either Qwen. A claim that episode-matched training beats
population-prior training in distribution cannot stand while its only
non-Qwen model says the reverse, and the three-seed estimate is not a basis for
dismissing it either.

**Mechanism-family Qwen3-8B.** Qwen3-4B is the strongest result in Experiment 3
($+0.340$ undisclosed, clearing the conservative $t$ and the wild cluster
bootstrap at $G=3$). Qwen3-8B disclosed runs $-0.125$ with 0 of 3 seeds
positive. The two sizes point opposite ways in the same roster. Reporting the
4B result without resolving the 8B one would be selecting the agreeable half.

## Recorded expectations, before the data exist

Written down so neither outcome can later be presented as predicted.

We expect **Llama's negative in-distribution contrast to persist** --- all three
seeds agree in direction, its minimum $G$ is 4, and the magnitude is large. If
it persists, the in-distribution claim is Qwen-specific and must be stated that
way.

We expect **Qwen3-8B's mechanism contrast to remain near zero or mildly
negative**, not to turn positive. If it does turn positive, that is a genuine
finding about seed sensitivity at $G=3$ and is reported as such.

## Binding commitments

1. **$G = 5$ is fixed** for both cells. Each roster is submitted whole; each
   analysis is run once.
2. **Reported regardless of direction**, and this record is cited wherever the
   five-seed values appear. The three-seed estimates remain in the record as
   what was originally computed.
3. **No seed is added or dropped on the basis of the result**; no completed seed
   is discarded for disagreeing with seeds 42--44.
4. **Estimators unchanged**: hierarchical seed-first paired bootstrap,
   Student-$t$ on seed-level means, wild cluster bootstrap with Webb six-point
   weights imposing the null, exact one-sided seed-level sign test. Bootstrap
   seed 20260818.
5. **Failure rule** as in the sibling amendments: infrastructural failures are
   resubmitted unchanged up to three attempts and recorded; a completed seed is
   never discarded; a roster that cannot be completed is reported at whatever
   $G$ was achieved with the missing seed named.

## Disclosure

Both seed counts were chosen after the three-seed results were observed, and
both cells were selected **because they disagree** with results reported
elsewhere in the experiment. That is stated wherever these values appear.

At $G=5$ the exact sign test reaches $p = .031$ if unanimous; it is floored at
$p = .125$ at $G=3$ and cannot go lower regardless of the data.

## Wrapper integrity

Both wrappers are verified against the hash their own ledger recorded, not
against git:

* `mechanism_family/mechanism_rl.sh` matches `train_script_sha256` in
  `c3_mechanism_20260817T215355Z_walltime30h_...json`.
* `coin_city_structural/train.sh` matches **no** ledger hash, because the Coin
  City ledger is the only roster ledger that records no train-script hash. It
  is known to have changed on 2026-08-28
  (`WRAPPER_DEVIATION_RECORD.md`). The crossover control showed that change is
  inert --- the same seeds under the new wrapper reproduce their original
  contrasts ($+0.251$ vs $+0.255$ on the joint cell) --- so the Llama addition
  uses the current wrapper with `SAVE_STEPS=999999 SAVE_CKPT=0`, which
  reproduces the pre-change emitted command line exactly.

Every submission under this amendment records its wrapper hash in its ledger.
