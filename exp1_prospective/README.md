# Experiment 1: prospective evidence interventions

Experiment 1 measures whether forecasting agents revise a focal binary forecast
in the direction implied by new evidence while remaining comparatively stable
under orthogonal evidence. The manuscript's numerical authority is the frozen
June 17 aggregate:

`data/results/consistency_report_2026-06-17.json`

The mutable row-level collection directories contain later retries and top-ups, so
running `agent/evaluate_consistency.py` over their present contents defines a new
dataset rather than reproducing the freeze. The freeze itself enumerates every
scored run, so the published numbers can be rebuilt from it directly:

```bash
python exp1_prospective/agent/rebuild_from_freeze.py
```

This recomputes the model-summary, anchoring, Figure 5(b), and arm-coverage values
from `consistency_report_2026-06-17.json`, checks them against what the LaTeX
sources print, and exits non-zero on any disagreement. All 116 checked values
currently reproduce. Three published artifacts are derived from the row-level
records instead and are listed as such in the check output: the threshold
robustness table, the Stage 3 human review, and the post-freeze settlement audit.

## Maintained workflow

| Stage | Maintained source or artifact |
|---|---|
| Market selection | `fetch_markets/select_markets.py` and `data/selected_markets/diverse_2026-06-09.jsonl` |
| Initial forecasts | hosted/local runners in `agent/` and frozen manifests under `data/initial_forecasts/` |
| Evidence packets | `agent/build_counterfactuals.py` and `data/counterfactuals/counterfactuals_2026-06-10.jsonl` |
| Conditional updates | provider-specific update runners in `agent/` |
| Frozen evaluation | `agent/evaluate_consistency.py` and the June 17 report above |
| Clustered uncertainty | `agent/evaluate_clustered_uncertainty.py` and `data/results/clustered_movement_uncertainty.json` |
| Threshold robustness | `agent/evaluate_threshold_robustness.py` and `data/results/threshold_robustness.json` |
| Prompt-withholding and packet-polarity audit | `agent/audit_packet_polarity.py` and `data/results/packet_polarity_audit.json` |
| Selectivity robustness checks | `agent/audit_selectivity_robustness.py` and `data/results/selectivity_robustness.json` |
| Lexical sign-classifier baseline | `agent/audit_lexical_sign_baseline.py` and `data/results/lexical_sign_baseline.json` |
| Hosted vs. open-weight group contrast | `agent/audit_group_contrast.py` and `data/results/group_contrast.json` |
| Freeze reproduction check | `agent/rebuild_from_freeze.py` and `data/results/freeze_reproduction.json` |
| Rationale mechanism-tracking and say/do coupling | `agent/audit_mechanism_tracking.py` and `data/results/mechanism_tracking.json` |
| Post-freeze outcomes | `agent/evaluate_resolved_outcomes.py`, the dated market snapshot, and `data/results/resolved_outcome_evaluation_2026-08-24.json` |
| Human materials review | `stage3_materials_annotation/` |

The paper's 95% movement intervals resample whole markets 20,000 times while
retaining every packet and repeated initial-run update belonging to a sampled
market. Rebuild that deterministic artifact from the repository root with:

```bash
python exp1_prospective/agent/evaluate_clustered_uncertainty.py
python exp1_prospective/agent/evaluate_resolved_outcomes.py
python -m pytest -q exp1_prospective/agent/tests
```

The generator records the input checksum and fails if reconstructed point
estimates or record counts disagree with the frozen report. Qwen 2.5-7B is
retained for compliance auditing but excluded from magnitude and sensitivity
comparisons because its archived records mix probability scales.

The post-freeze outcome report is a descriptive follow-up on the 60 markets that
had strict terminal 0/1 prices on August 24. Its paired comparisons use the
frozen June 9 market price; pass `--fetch` only to create a new dated snapshot.

Historical plans and unused manuscript directions are under `_archive/` and are
not maintained entrypoints.

## Context-reversal development pilot

`context_reversal/` is the maintained local-model follow-up. It tests identical
news under reversed and broken contextual relationships, with context-specific
initial forecasts and independent no-news/repeated-news controls. Its first
20-family synthetic cohort is exploratory and has not been independently
human-validated; it does not replace the frozen results above. See
`context_reversal/PROTOCOL.md` and `context_reversal/README.md`.
