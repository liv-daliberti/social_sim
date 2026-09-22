# Local direction and reasoning diagnostics

Authorized follow-up on all 20 existing context-reversal families. This run does
not repeat human review and makes no frontier API calls. See PROTOCOL.md and
REVIEW_PROVENANCE.md for the fixed design and existing review scope.

Planned: direction-only classification on Qwen2.5-72B, Llama-3.1-70B and Qwen3-32B
(240 outputs each), plus Qwen3-32B probability updates with thinking enabled and
an otherwise matched unconstrained non-thinking control (320 outputs each).
Total: 1,360 planned records, one repeat per family/context.

The group worker verifies frozen code and input hashes before each arm. Scheduler
submissions and exact commands are recorded in submission.json. Jobs: Qwen72
31443323, Llama70 31443324, Qwen32 group 31443404; finalizer 31443326. Each arm
checkpoints raw outputs and a manifest, then analyzes its own results. The CPU
finalizer records total response coverage and validity separately in completion.json
and writes a summary covering every prescribed arm, including failures.

A planned or submitted run is not a completed result. Check completion.json and
raw response manifests for current status. Original local and frontier pilot
results remain unchanged.

Direction classification is complete: 720/720 valid outputs. See
`results/local_diagnostics_v1/direction_summary.md` and its source/audit hashes.
The completed direction findings are included in the ICLR appendix and rebuilt PDF.

Probability outputs receive the symmetric, documented optional-Markdown-fence
parser amendment in `FORMAT_RECOVERY_AMENDMENT.md`. Strict v1 raw files and
analyses remain separate. Disabled recovery job 31443826 reuses all 122 received
outputs and generates 198 previously unattempted updates, with zero new baselines.
Thinking continuation job 31443835 depends on 31443404, resumes only missing
records if needed, then seeds/completes the same v2 parsing policy. Saved
truncations and other failures are retained. The one-hour scheduler limit changes
allocation boundaries, not prompts, seeds, decoding settings, or token budgets.
Finalizer 31443326 waits for both recovery arms; CPU comparison job 31444035
then writes `results/local_diagnostics_v1/reasoning_comparison.{json,md}`.
Current paths and commands are authoritative in submission.json.
