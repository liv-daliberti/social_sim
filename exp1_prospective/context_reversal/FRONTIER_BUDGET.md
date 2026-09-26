# One future frontier evaluation: freeze and budget gate

The new authorization covers at most **$25 total for one future evaluation**. The recommended manifest cap is `$24`, preserving a $1 buffer. The earlier development pilot is a separate completed authorization. No future evaluation, credential lookup, token-count request, or API inference has been performed by these new tools.

New modules are `frontier_budget.py`, `run_frozen_frontier.py`, `frontier_token_certificate.py`, `collect_frontier_token_counts.py`, and `run_frozen_frontier_batch.py`. Existing frozen runners remain unchanged. Standard Responses, Batch Responses, and the official input-count collector are implemented and tested only with fake clients. No live transport has been exercised. The standard runner deliberately rejects Batch pricing; the separate Batch runner uses its own explicit workflow below.

## Budget and resume contract

The complete frozen cohort must fit its worst-case reservation before the first generation request. Admission cannot select families or stop midway through a family because of observed spending. All planned baseline and update requests, including future baseline-dependent updates, are reserved at once. The fixed shared ledger is `runs/frontier_budget_authorization_v1/ledger.json`. It binds the authorization to one freeze hash and output path; changing paths or resuming cannot reset the budget.

Currency uses integer nanodollars. Every concurrent request is durably marked in flight before submission. Unknown-billed failures and interrupted requests retain their full reservations and are never retried. Known token usage releases only the unused reservation; input is still charged at the maximum cache-write rate, with no assumed discount. A failed baseline releases only its three provably unattempted branches. All failures remain in planned response denominators. Provider-reported token-bound violations halt further generation and remain explicit in the ledger.

Requests use `service_tier="default"`, no tools, `store=False`, low reasoning, the frozen JSON schema, and either 1024 or2048 maximum output tokens. SDK retries are disabled by the existing reviewed client constructor. Output-token limits include reasoning tokens. Pricing validity is explicitly limited through 2026-11-21; a later run fails closed until reviewed pricing is updated.

Official references:

- [GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol)
- [Pricing](https://developers.openai.com/api/docs/pricing)
- [Responses request settings](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)

## Token evidence

Without a certificate, input reservations use the earlier runner's conservative UTF-8 prompt bytes plus schema bytes plus4096 framing tokens. This is intentionally too conservative for the proposed 80-family design.

The official [input-token counting guide](https://developers.openai.com/api/docs/guides/token-counting) says counts include request formatting and schemas. Its [Python endpoint reference](https://developers.openai.com/api/reference/python/resources/responses/subresources/input_tokens/methods/count) accepts the `text` structured-output schema and `reasoning` configuration.

`frontier_token_certificate.py export` writes exact request bodies and hashes; it makes no requests:

```bash
python -m exp1_prospective.context_reversal.frontier_token_certificate export \
  --plan /absolute/frozen/plan.jsonl --max-output-tokens 2048 \
  --output /absolute/frozen/input_count_requests.json
```

`collect_frontier_token_counts.py` obtains and archives official count responses, including HTTP status, provider request ID, timestamp, and exact request-body hash, only with an explicit `--live` flag after all review/freeze gates. Offline import of separately archived receipts remains available:

```bash
python -m exp1_prospective.context_reversal.frontier_token_certificate import \
  --requests /absolute/frozen/input_count_requests.json \
  --receipts /absolute/frozen/input_count_receipts.json \
  --output /absolute/frozen/token_certificate.json
```

Each receipt must contain `request_id`, `endpoint`, `http_status`, `count_body_sha256`, `provider_request_id`, `received_at`, and `response` with `object="response.input_tokens"` and an integer `input_tokens`. Requests cover every baseline and one identical message/schema envelope containing the anchor `x`. Receipts are archived provider evidence, not cryptographically authenticated statements; their collection provenance must be reviewed before the final freeze.

Baseline bounds are exact for their hashed bodies. Unknown-prior update bounds use the certified identical envelope count **plus the full UTF-8 byte length of the entire rendered update**, allowing32 bytes for any serialized finite probability in `[0,1]`. Nothing is subtracted for the anchor. This uses a conservative one-token-per-content-byte ceiling and fixed request structure; it is **not** an exact count of a future update or a guess based on a few sampled priors. All templates and metadata are bound by the plan hash and canonical-unit hash. The Batch adapter also checks an exact count receipt for every actual rendered update before submission, and halts without generation if any receipt exceeds the frozen bound. Its saved probabilities are checked against the archived provider results before these own-context priors may enter downstream requests.

## Freeze manifest

The offline validator requires all of the following, with exact hashes:

- A fresh plan, nonempty protocol, scoring artifact, completed review, and freshness audit.
- Completed local development result artifacts; reviewer identity; zero unresolved findings; no output from any target model in review/selection, no outcome-based family filtering, and an explicit declaration that target inference did not begin before freeze.
- Every family and context covered once, repeat0; three scored contexts with optional masked context; a fixed complete family order.
- Freshness checks exclude the original evaluated development plan and all declared additional development plans by hash, family identity, and exact prompt/template overlap.
- Current hashes for `run_frozen_frontier.py`, `frontier_budget.py`, `frontier_token_certificate.py`, `run_openai.py`, and `run_local.py`.
- Optional `token_certificate` artifact for standard mode; Batch requires both `token_certificate` and `token_count_collection`, matching the exact model, schema, reasoning, output limit, plan, and body hashes.
- Batch and count collection additionally freeze hashes for `collect_frontier_token_counts.py` and `run_frozen_frontier_batch.py`, as applicable, and an absolute `token_certificate_output`. The actual local target roster and settings belong in the hashed protocol. Review may use separate reviewer checkpoints, but no target-model outputs or outcome-based family selection are permitted before the design freeze.

Manifest top-level fields are `schema_version="frozen_frontier_evaluation_v1"`, `status="frozen"`, `freeze_stage="after_local_development_before_frontier"`, `evaluation_id`, `target_inference_started_before_freeze=false`, `authorization_scope="one_future_frozen_frontier_evaluation_25usd_v1"`, `model="gpt-5.6-sol"`, `model_key`, `max_output_tokens`, `reasoning_effort="low"`, `max_retries=0`, `concurrency` in1..4, `pricing` (`standard` or `batch`), `prices_nusd_per_token`, `pricing_valid_through="2026-11-21"`, `budget_usd`, `contexts`, `family_order`, absolute `output_path`, `artifacts`, and `code_sha256`. Artifact objects contain absolute `path` and `sha256`.

The review JSON requires `status="complete"`, `plan_sha256`, exact `family_ids`, `reviewer`, `unresolved_findings=0`, `frontier_model_outputs_used=false`, `target_model_outputs_used=false`, `outcome_based_family_filtering=false`, `local_development_complete=true`, and a nonempty `development_results` list of artifacts. Freshness JSON requires `status="passed"`, `plan_sha256`, exact `family_ids`, and nonempty `excluded_plans` artifacts including the original frontier pilot plan.

```bash
python -m exp1_prospective.context_reversal.run_frozen_frontier \
  --freeze-manifest /absolute/frozen/evaluation.json --dry-run
```

Dry run is the default, never reads a key, writes no outputs, and reports both budget eligibility and whether the selected transport is implemented. `--execute` is a separate flag and still requires every gate and the complete cohort reservation. No actual fresh evaluation manifest has been created by this work.

## Scalable options

At80 families ×3 contexts ×4 calls =960 calls:

| Mode | Maximum output tokens | Output-only worst case | With average1000 reserved input tokens/call |
|---|---:|---:|---:|
| Standard |2048|$39.3216|$44.1216|
| Standard |1024|$19.6608|$24.4608|
| Batch |2048|$19.6608|$22.0608|

These are planning calculations, not a certificate for unbuilt fresh materials. To fit a `$24` cohort cap, Batch2048 requires an average reserved input bound of at most1808 tokens/call; Standard1024 requires at most904. At80 families ×4 contexts, Batch2048 output alone is$26.2144 and cannot fit$25. Batch2048 with three scored contexts best preserves the previous output limit, subject to actual token certificates for the complete fresh cohort.


## Explicit Batch workflow

1. Complete local development and review, then freeze the fresh design, scoring, family order, review, freshness evidence, all transport code hashes, `pricing="batch"`, `budget_usd="24"`, and absolute response/certificate paths. No target-model inference may precede this design freeze. There must be no unresolved findings or target-output-based material selection.
2. Run the collector without `--live` to validate the count plan offline. Once the key is configured and live execution is authorized, use `collect_frontier_token_counts --freeze-manifest /absolute/design_freeze.json --live`. This invokes only `/responses/input_tokens`, never generation. It persists every count receipt before continuing. The fixed count authorization binds the same design and canonical certificate/collection paths across resumptions.
3. Preserve the design freeze. Create the final generation freeze by adding the returned certificate and collection paths/hashes under `artifacts.token_certificate` and `artifacts.token_count_collection`. All design artifacts and settings must remain identical. The final freeze, including these receipts, is immutable once generation begins. If the whole cohort worst case exceeds $24, no generation is admitted; no outcome-based family pruning is allowed.
4. Validate offline with `run_frozen_frontier_batch --freeze-manifest /absolute/generation_freeze.json` (default action `status`, no client/key/writes). Then invoke the following explicit actions with `--live`, in order: `submit-baselines`, `collect-baselines`, `count-updates`, `submit-updates`, `collect-updates`. Collection returns `phase_status="waiting"` for an unfinished batch; invoke the same collection action later. There is no long polling or automatic resubmission.

For example, the baseline submission command is:

```bash
python -m exp1_prospective.context_reversal.run_frozen_frontier_batch \
  --freeze-manifest /absolute/generation_freeze.json \
  --action submit-baselines --live
```

Each phase uses a separate JSONL input file with unique stable custom IDs and endpoint `/v1/responses`. Requests preserve the prior pilot's typed JSON format, low reasoning, and 2048 output tokens when that frozen option is selected. Returned order does not affect context/prior association. Every failed/truncated/malformed completion remains a planned failure; no completion is regenerated. Failed baselines retain three explicitly blocked unattempted branches. Terminal batches retain missing result rows as unknown-billed failures at their full reservation.

Submission intent, input file ID, batch ID and full reservations are written durably. A lost creation response blocks new submission. Reconcile the existing ID using `--action adopt-baseline-batch --remote-id batch_... --live` (or `adopt-update-batch`); the runner verifies endpoint, input file and frozen metadata before adopting it. Ambiguous uploads similarly use `adopt-baseline-file`/`adopt-update-file` and require an exact remote file-body hash. These actions never create a replacement. If an input-count receipt was lost, `--receipt-file /absolute/recovered_receipts.json` explicitly imports the archived receipt list for those prior count attempts; it cannot replace received receipts or silently repeat a count.

Remote result files are archived locally with hashes before any result settles. SDK retries are zero. Fatal model, service-tier or usage-contract violations persist atomically with billing settlement; resume retains failed status and fills every unattempted planned branch. The ledger remains the accounting authority, including in-flight requests that have not yet reached the response JSONL. Keep the shared ledger, its lock, both count evidence files, phase inputs/results and the immutable freeze together. Never delete them to reset a run.

The Batch interface follows the [official Batch guide](https://developers.openai.com/api/docs/guides/batch), including its separate result/error files and terminal states. Readiness here means offline-tested transport code; fresh material review, the real immutable evaluation freeze, provider count receipts and a whole-cohort budget certificate are still required before any frontier generation.
