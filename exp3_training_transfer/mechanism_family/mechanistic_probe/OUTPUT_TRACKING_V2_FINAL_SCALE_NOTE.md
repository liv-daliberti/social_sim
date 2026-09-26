# Output tracking v2 final response-scale note

This note supersedes `OUTPUT_TRACKING_V2_CORRECTION.md`.

The response-scale audit initially suspected that `prior_response` was being
subtracted twice.  That suspicion was rejected after tracing both sides of the
registered estimand.  The hidden-state response target is

`(truth_response - prior_response) / 4`,

so output-level response tracking must use

`(predicted_response - prior_response) / 4`.

Their prediction error is therefore exactly
`(truth_response - predicted_response) / 4`; no prior is present in the error.
Because `prior_response` is fixed within a world, it also cancels from the
within-world correlation.  The large negative response R-squared values are
not a scale artifact: they arise because episode-level truth variation around
the development-estimated world mean is small relative to forecast error.

The authoritative entry point and canonical artifacts are consequently:

- `run_output_tracking_v2.py`
- `runs/output_tracking_v2.json`
- `runs/output_tracking_predictions_v2.jsonl`

`run_output_tracking_v2_corrected.py` is retained only as a transparent record
of the rejected diagnostic transformation and must not be used for results.
