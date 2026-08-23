# Experiment 3D: domain transfer (Coin-* family)

_Created 2026-08-11._

Replaces structure transfer as the **primary** synthetic arm of Experiment 3. The structure-family
arm (`../dag_family`) is retained and now reported in the paper appendix.

## The question

Structure transfer varies the causal graph but keeps one cover story: every world is an election.
That tests whether a model can use a described-but-unseen graph, not whether the forecasting skill
is tied to the domain it was learned in. Here the graph is held **fixed** (the Exp-2 `direct` world:
driver -> persistent latent -> noisy observable, one hidden per-episode sensitivity `g`) and the
**domain** varies.

## Domains

| Domain | Role | Driver | Observable | Range | sigma | Implied g | Period |
|---|---|---|---|---|---|---|---|
| CoinCity | train | news | poll (points) | 0–100 | 2.0 | 0.10–1.00 | week |
| CoinFishing | train | sea_state | catch (crates) | 0–400 | 12.0 | 0.20–2.00 | trip |
| CoinFarm | train | rainfall | yield (bu/acre) | 0–240 | 3.8 | 0.16–1.60 | season |
| **CoinBasketball** | **held out** | roster | points | 70–130 | 1.4 | 0.12–1.20 | game |
| **CoinClinic** | **held out** | staffing | wait (minutes) | 0–180 | 4.0 | 0.18–1.80 | day |

Domains are not a renaming of one another. Each carries its own offset, observable scale, driver
scale, and noise, so the *implied sensitivity* differs across domains and no memorized numeric
regime covers the family. Held-out ranges sit inside the hull of the training ranges, so this is an
**interpolation** test of domain generality (`domains.py::_check` enforces this and fails loudly if
an edit breaks it).

Implementation is an affine display skin over the canonical world, so `lg_dag.py`'s verified
stability and clipping gates are untouched:

    y_d = offset_d + obs_scale_d * (y_canonical - 50)
    x_d = in_scale_d * u_canonical
    g_d = (obs_scale_d / in_scale_d) * g_canonical

## Arms

Nested ladder over the training pool, **row-matched at 4,800** so diversity is separated from data
quantity:

| Arm | Trains on | Rows/domain |
|---|---|---|
| `d1` | CoinCity | 4800 |
| `d2` | CoinCity + CoinFishing | 2400 |
| `d3` | CoinCity + CoinFishing + CoinFarm | 1600 |
| `structureless` | same three, driver decoupled from observable | 1600 |

`d3 - d1` is the domain-diversity effect. `d3 - structureless` separates real driver->observable
inference from adaptation to format, units, and levels. The untrained base model is the fourth
reference point and needs no training run.

## Reward

Unchanged in form (level + slope composite, `slope_weight` 0.5) but **tolerances now travel in the
dataset**, scaled by each domain's skin: `reward_scale = 15 * obs_scale`, `slope_scale = 0.5 *
gain_ratio`, plus the domain's `clip` range. Without this, CoinFishing (0–400) would be ~4x harder
than CoinCity at equal relative accuracy, and every forecast above 100 crates would be clipped to
100 — destroying the measured slope. `run_biased_news_rl.py` reads these per row and falls back to
its CLI values for legacy `dag_family` rows, which are unaffected.

## Files

- `domains.py` — registry + affine skin + health gates
- `prompt_domain.py` — domain-skinned prompt, `forecast_A..D` reply keys
- `make_dataset_domains.py` — builds all four arms
- `preflight_domains.py` — fail-closed audit, writes `protocol/exp3d_manifest.json`
- `domain_rl.sh` — Slurm launcher
- `launch_domains.py` — registered submission (runs preflight, writes `runs/` ledger)
- `make_domain_table.py` — emits the paper's roster table from the registry

## Reproduce

```bash
python make_dataset_domains.py --all
python preflight_domains.py --write-manifest protocol/exp3d_manifest.json
python launch_domains.py --canary     # prove the pipeline trains
python launch_domains.py --all        # 4 arms x 3 seeds
```

## Preflight (passing as of 2026-08-11)

All four arms at 4,800 rows in equal per-domain shares; ladder nested; no held-out domain in any
training split; train/eval seed blocks disjoint; targets inside declared ranges; tolerances
skin-scaled; held-out split byte-identical across arms. `corr(true_gain, slope_target)` = 1.00 for
`d1`–`d3`, undefined for the control (slope targets identically zero). Identifiability on held-out
domains, oracle vs constant policy: CoinBasketball +0.548, CoinClinic +0.560.

## Note on the launch hang

The previous campaign (3 x exp3a prior-mean, 3 x exp3b) deadlocked during model init and burned its
full 8h wall without a single training step, while the monitor reported `300/300 (100%)`. Cause: a
HuggingFace Hub request whose peer hung up, leaving the learner parked on a lock (43 threads in
futex, idle GPU, CLOSE-WAIT socket to huggingface.co). Fixed by pinning offline mode in
`.runtime/oat_env.sh` — every model we train is already cached locally. `monitor_campaign.py` still
mis-reports progress by matching the dataset-map tqdm bar; treat `grep -c global_step train.log` as
the source of truth until that is fixed.
