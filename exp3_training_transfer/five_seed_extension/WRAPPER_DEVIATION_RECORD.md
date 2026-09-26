# Training wrapper deviation: seeds 42--44 and 45--49 did not share a wrapper

Record version: `exp3_coin_city_wrapper_deviation_v1`

Recorded on **2026-09-16 UTC**, on discovery.

Affects: `exp3_qwen3_8b_coin_city_eight_seed_v1`,
`exp3_qwen3_4b_coin_city_eight_seed_v1`, `exp3_qwen3_8b_g7_interim_look_v1`.

## The deviation

`exp3_training_transfer/coin_city_structural/train.sh` was modified on
2026-08-28, after Coin City seeds 42--44 trained (2026-08-20 to 2026-08-23) and
before seeds 45--49 (2026-09-14). Two flags changed:

| flag | seeds 42--44 (git HEAD, committed 2026-08-22) | seeds 45--49 (working tree) |
|---|---|---|
| `--save_steps` | `999999` | `40` (`SAVE_STEPS` default) |
| `--save_ckpt` | absent | present (`SAVE_CKPT` default) |
| `--enable_prefix_caching` | hardcoded | via `ENABLE_PREFIX_CACHING` default 1 — unchanged |

Both changed flags take effect silently: `launch.py`'s `coin_env` sets neither
`SAVE_STEPS` nor `SAVE_CKPT`, so the new defaults applied without appearing in
any submission command or ledger. The observable trace is that seeds 45--49
carry a `debug_*/checkpoints/step_00301` directory and seeds 42--44 do not.

The parent protocol requires that each added job "reuses the exact parent
wrapper". It did not.

## What was and was not verified

The eight-seed amendment states that added jobs reuse `train.sh` "at the hash
pinned above". The pinned hash was the *current* file, not the file that ran
seeds 42--44, and that difference was not checked. The verification that was
performed --- and that does hold --- covered the *environment*: every added
job's exported variables are byte-identical to the corresponding seed-42
submission except `SEED` and the derived `TAG`. Script identity was assumed,
not tested. That is the gap.

## What this does and does not compromise

Within any one seed, both arms trained under the same wrapper, so each
seed-level contrast is internally valid.

Across seed blocks it is confounded. Specifically, the split-half diagnostic
computed during the $G=7$ interim ---

    seeds 42--44 joint-cell contrast  +0.2513
    seeds 45--49 joint-cell contrast  +0.0010

--- varies wrapper version together with seed block and **cannot** be read as
"fresh seeds show no effect". That reading is withdrawn pending the control
below. The pooled $G=7$ estimate remains an estimate of the contrast averaged
over two wrapper versions, which is not the quantity the protocol defines.

The evaluated checkpoint was checked separately and is **not** implicated:
every seed in both blocks evaluates `saved_models/step_00301`, and `train.sh`
resolves its adapter by globbing `saved_models/` only, never `checkpoints/`.
Parse rates are 100.00% on all fourteen endpoints.

## Control

`submit_wrapper_crossover.py`, protocol `exp3_coin_city_wrapper_crossover_v1`,
jobs 31306100--31306105. Qwen3-8B seeds 42, 43 and 44 re-run in both arms under
the current wrapper, seed held fixed, so each seed's contrast can be compared
against its archived original. Tagged `wrapxover_` so the report directories
cannot match the renderer's `{arm}_{model}_s{seed}_*` glob. These runs are a
diagnostic control; they are not endpoints and enter no estimand.

Reading, fixed before the results exist:

- If the three crossover contrasts track their originals, the wrapper change is
  inert, the split-half stands, and the eight-seed roster may be pooled as
  planned with this deviation disclosed.
- If they move toward the seeds 45--49 values, the wrapper change is doing the
  work, the split-half is void, and the eight-seed roster must be rebuilt under
  a single wrapper before any pooled estimate is reported.

Either way this record is cited and the crossover values are reported.

## Corrective action taken

The Qwen3-4B eight-seed roster had been submitted on the current wrapper
(jobs 31305819--31305828) and was cancelled 16 minutes in, before any endpoint
existed, and resubmitted as 31306083--31306092 with
`SAVE_STEPS=999999 SAVE_CKPT=0`. Those two values reproduce the emitted oat
command line of the wrapper at HEAD: `PREFIX_CACHE_ARGS` expands to the same
`--enable_prefix_caching`, and `SAVE_CKPT_ARGS` and `RESUME_ARGS` expand empty.
The Qwen3-4B roster therefore shares one wrapper across all eight seeds and its
own split-half is interpretable. The four cancelled output directories are
archived under `reports/failed_attempts/` with `FAILURE.json` markers; none
contained endpoint files.

The Qwen3-8B seed-48 rerun (job 31305731) was left running. It is
wrapper-matched to seeds 45--49, which is the correct match for its block.

`submit_coin_city_eight_seed.py` gained `--restore-parent-wrapper`, which
passes `WRAPPER_RESTORE = {"SAVE_STEPS": "999999", "SAVE_CKPT": "0"}` and
teaches the environment check to expect exactly those two keys.

## Standing lesson

Pinning the hash of a file as it stands today does not establish that it
matches what produced the data being extended. Any future extension of an
existing roster must diff the wrapper against the version in effect when the
original runs executed, and record the comparison, not merely hash the current
file.
