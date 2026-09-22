# Fresh evaluation status — 2026-09-22

## Completed

- New materials frozen before target inference: 80 finite-rule instances in eight mechanism classes; four wording/sampling variants; three relational contexts; own-context baselines and three independent update branches.
- Every family received model screening and formal author adjudication. No new human validation. Screener agreement was poor (224/720 and 182/720); all families and disagreements are retained. See `ADJUDICATION.md`.
- Frontier evaluation complete and independently rescored: GPT-5.6 Sol, fixed balanced 24-parent subset, base/paraphrase, **576/576 valid records**. Both wordings: **24/24 paired reversals, 48/48 correct signed updates, zero irrelevant/no-news/repeated-news movement**.
- Conservative usage-based frontier cost bound: **$1.773475**. Both batches collected, no unknown billed calls, authenticated transient process closed. No further frontier spending scheduled.
- Completed report: `../../results/fresh_evaluation_v1/frontier_summary.md`.
- Completion audit and publication guard tests pass. The isolated synthetic layout rehearsal preserves nine main pages, with no undefined references or overfull boxes. Synthetic fixtures exist only under `/tmp`, not in experiment results or canonical paper files.

## Running / pending

- Matched Qwen3-32B thinking enabled/disabled: **7,680 planned records** on identical fresh materials. Seven of eight disabled shards are complete; enabled runs are slower. Durably attempted requests and token-limit truncations are terminal, without resampling.
- Jobs resume automatically within the frozen inference settings. `local_jobs/` contains the scheduler history. Live response coverage is in `../../results/fresh_evaluation_v1/status.json`; recorded totals can temporarily include in-flight write-ahead records.
- The CPU completion workflow is scheduled (initial pending job **31460944**) and checks readiness every ten minutes. On completion it reparses raw outputs, verifies own-context priors and chat/token hashes, independently checks primary statistics, writes the appendix and matched tables, builds in isolation, and publishes only after all checks pass.
- Canonical ICLR `main.tex` and `main.pdf` have **not yet received the fresh results**. Publication requires full collection, a passed audit, a clean build with nine main pages, and unchanged canonical source hashes. Publication state: `publication/status.json`. Backups are retained on success; conflicts stop the workflow.

This is a controlled finite-rule diagnostic, not 80 independent mechanisms or a validated natural-news benchmark. All-success empirical intervals do not imply certainty outside these instances. The original packet relevance shortcut remains a limitation.
