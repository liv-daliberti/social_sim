# Experiment 2: Coin City

The paper result is the frozen six-deployment experiment
`coin_city_stable_relationship_claude_n250_v4`. Its active code is under
`biased_news/`; its frozen tasks, keys, responses, and analyses are under
`biased_news/data/coin_city_stable_relationship_claude_n250_v4/`.

The maintained workflow consists of:

- `engine/coin_city_stable_relationship_claude_n250.py` for episode generation;
- `engine/coin_city_wrong_context_arm.py` and
  `engine/coin_city_symbol_context_arm.py` for the two diagnostic arms (the
  arbitrary-symbol control was executed only on DeepSeek V4 Pro);
- the corresponding builders and validators in `eval/`;
- `eval/run_frozen_task_shard.py` for shared append-only execution and parsing;
- the model-specific `eval/run_coin_city_stable_relationship_*.py` wrappers;
- the two frozen result analyzers in `analysis/`, plus the hash-frozen live
  renderer and its compatibility base;
- `analysis/analyze_coin_city_reference_selection.py`, which asks whether the
  forecast tracks the reference city the City C cue names rather than a prior
  attached to the cue words, from the frozen responses and with no model call;
  and
- `paper/generate_coin_city_split_figures.py` plus
  `paper/generate_coin_city_appendix_tables.py` for manuscript outputs.

The original Opus 4.8 wrapper is byte-for-byte frozen by the design manifest.
Its historical import name now aliases the maintained shard harness, which
removes runtime dependencies on retired three-city code without changing the
hash-locked wrapper.

From `exp2_v2/biased_news`, run:

```bash
python eval/validate_coin_city_stable_relationship_claude_n250.py
python -m pytest -q tests
python eval/run_coin_city_stable_relationship_claude_n250.py \
  --model claude-opus-4-8 --arm baseline \
  --episodes 0 --prefixes 0 --dry-run
```

No command above makes a model API call.

Earlier three-city and Coin City iterations are retained under `_archive/` as
an audit trail. They are not part of the current paper workflow or default test
suite and should not be imported by maintained code.
