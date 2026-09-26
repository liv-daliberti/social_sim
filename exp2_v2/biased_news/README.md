# Experiment 2: Coin City

This directory contains the maintained ICLR Coin City implementation. The
original 250-episode design remains frozen at
`data/coin_city_stable_relationship_claude_n250_v4/`; additive robustness work
is governed by [`ROBUSTNESS_EXTENSION_PROTOCOL.md`](ROBUSTNESS_EXTENSION_PROTOCOL.md).

## Maintained entry points

- Build and validate the harder generator population:
  `python eval/build_coin_city_robustness_population.py`
- Register and submit its ten open-checkpoint jobs:
  `python eval/launch_coin_city_robustness_population.py --submit`
- Analyze the completed population:
  `python analysis/analyze_coin_city_robustness_population.py`
- Run the already-registered rationale audit:
  `python analysis/analyze_coin_city_rationales.py`
- Prepare the hosted-model runtime:
  `python -m venv .venv_hosted && .venv_hosted/bin/pip install -r eval/requirements-hosted.txt`
- Run and analyze the independent GPT-5.6 k=0 repeat:
  `bash eval/run_coin_city_gpt56_k0_repeat_local.sh` and
  `python analysis/analyze_coin_city_gpt56_k0_repeat.py`

Hosted credentials are read from `GPT56_AZURE_API_KEY`; they are never stored in
tracked files. Generated response files, manifests, registration ledgers, and
analysis artifacts live below the corresponding `data/<experiment>/` root.


## Completed extension artifacts

- Harder-population analysis:
  \`data/coin_city_population_075_040_sd010_v1/analysis/population_robustness_results_v1.json\`
- Rationale audit:
  \`data/coin_city_stable_relationship_claude_n250_v4/analysis/rationale_audit_v1.json\`
- GPT-5.6 independent repeat:
  \`data/coin_city_stable_relationship_claude_n250_v4/analysis/gpt56_k0_repeat_v1.json\`
- Paper tables: \`paper/tables/exp2_generator_population_robustness.tex\` and
  \`paper/tables/exp2_gpt56_repeat.tex\`

## Checks

Run focused checks from this directory with:

```bash
pytest -q tests/test_coin_city_robustness_extension.py
```

The original generator and response files are immutable inputs. New populations,
repeats, and post-hoc analyses use new output directories rather than overwriting
the paper's frozen artifacts.
