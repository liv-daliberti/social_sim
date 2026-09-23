# Amendment: train.sh checkpointing, Qwen3-14B Coin City scale extension

Protocol: `coin_city_qwen3_14b_scale_v1`
Ledger: `runs/coin_city_qwen3_14b_scale_full_20260825T134340Z.json`

## What happened

Of the six registered trainings, five completed on 2026-08-26. One did not:
`30880156`, the causal arm at seed 43, hit its walltime and was killed. The
finalizer `30880231` was `afterok`-locked to all six, so it was cancelled and
never ran. The extension has been renderable-but-unrendered since the seed was
recovered; this file records why the recovery required a script change.

The registered `train.sh` passed `--save_steps 999999`, so a training that ran
out of walltime left nothing to restart from and had to be rerun end to end
inside one allocation. On 2026-08-28 the script gained periodic checkpointing so
the seed could be recovered:

- `--save_steps 999999` became `--save_steps "$SAVE_STEPS"`, default `40`
- `--save_ckpt` added, behind `SAVE_CKPT`, default on
- `--resume_dir` / `--resume_tag` added, behind `RESUME_DIR`, unset by default
- `--enable_prefix_caching` moved behind `ENABLE_PREFIX_CACHING`, default on,
  which reproduces the registered invocation exactly

The registered `train_script_sha256` is
`46c95d79063e2b89c6db461aedd1bde163f7ec1730a1e3947faf2eaca9919ffb`; the amended
script is
`7ca287c389ff4f18678de9e381e5de6b0064cdd2635eebbfab3a749476685a21`.

## Why the recovered seed is comparable

Only the causal seed 43 ran under the amended script. Seeds 42 and 44, both
population-prior seeds, the canary and the base endpoint all ran on 2026-08-26
under the registered script.

The recovered run, `scale_causal_qwen3_14b_s43_recovery2_20260828_204430_j30948628`,
trained from scratch rather than resuming: its learner configuration records
`'resume_dir': ''`, so the resume path the amendment added was never taken. It
reached `step 301`, the same endpoint as seed 42's registered run. The remaining
changes save checkpoints more often and restate an existing flag through a
variable whose default is the registered value. None of them enters the loss,
the gradient, the data order or the sampling, so the recovered adapter is the
result of the registered procedure run to the registered endpoint.

What the amendment cannot claim is that the run is bit-identical to one that had
not been interrupted. It is a fresh training under a script that differs in
checkpointing, and it is reported as such.

## Scope

This amendment authorizes the renderer to accept the amended hash for the
`train_script_sha256` key alone, for this protocol alone. Every other registered
hash -- launcher, advance, base evaluation, finalizer, core analysis, protocol,
frozen dataset, parent ledger, parent manifest, parent results -- is unchanged
and is still checked exactly.
