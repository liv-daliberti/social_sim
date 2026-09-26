# Output tracking v2 response-scale correction

The first local bootstrap pass on 2026-08-25 exposed a response-scale mismatch
before confirmatory interpretation.  Registered `*.scores.jsonl` files store
`predicted_response` as a forecast contrast on the same scale as
`truth_response`.  The initial `analyze_output_tracking.py` implementation
mistakenly treated that field as a forecast level and subtracted
`prior_response` a second time.

`run_output_tracking_v2_corrected.py` is the authoritative v2 output-control
entry point.  It preserves the frozen split, parser, all-failed-task imputation,
gain decoder, bootstrap seed, and bootstrap scheme, but ensures that the value
entering response tracking is exactly the registered predicted response.  The
incorrect first-pass outputs were overwritten by a 10,000-replicate corrected
run at:

- `runs/output_tracking_v2.json`
- `runs/output_tracking_predictions_v2.jsonl`

This correction does not affect hidden-state extraction or `analyze_probe_v2.py`.
