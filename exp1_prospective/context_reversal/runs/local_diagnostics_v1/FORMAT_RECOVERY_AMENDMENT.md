# Format-only recovery amendment

Recorded after the first unconstrained non-thinking output, before any recovery
inference. The original version required bare final JSON. Qwen3 returned a single
correctly formed probability object inside a Markdown JSON fence in all 103 parse
failures. This blocked 198 update branches because 66 of 80 baseline responses
were rejected. These are serialization failures under the initial contract, not
evidence of 103 semantically uninterpretable probability answers.

Apply the same deterministic repair to BOTH thinking and non-thinking arms:
accept either a bare strict probability JSON object, or exactly one surrounding
Markdown code fence (`json` or no language) containing that same strict object.
No other surrounding prose, multiple objects, missing thinking closure, malformed
number, duplicate field, out-of-range value, or truncation becomes valid. The
probability value must come from the final answer after any thinking block.

Preserve all original raw outputs, statuses, manifests, and per-arm analyses.
Record a separate parser version and recovery provenance. Reparse every existing
received completion under the new rule; do not regenerate any of those calls,
regardless of correctness or value. Generate only update branches previously
marked blocked and never attempted, after their own baseline becomes parseable.
A baseline that still fails remains failed, with its updates blocked. Both modes
retain the same model, prompts, seeds, temperature, token budget, and scoring.

Report original format compliance separately from recovered task outcomes. This
is a disclosed post-output serialization amendment; it is not a preregistered
analysis rule. The original direction-task results are unaffected. No frontier
calls or additional human review are introduced.
