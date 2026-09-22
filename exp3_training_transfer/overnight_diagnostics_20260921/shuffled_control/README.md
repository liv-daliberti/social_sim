# Shuffled-target mechanism controls

The frozen control preserves all 4,800 training prompts and the exact target-vector multiset, assigning whole vectors to other compatible episodes within each world/evidence-depth stratum. Five seeds per disclosure are fixed. `PROTOCOL.md` describes the original design; `MATCHED_HARDWARE_AMENDMENT.md` fixes the final primary roster at 10 shuffled + 10 matched trainings, all on two A5000 GPUs with actor fraction 0.76. The six missing historical adapters and four hardware-comparator reruns explain the additional matched budget. No new seeds are added.

`checkpoint_audit.json` verifies actual weight files, runtime arguments and known source drift. `data/manifest.json` and donor maps record exact shuffle invariants. `freeze.json` pins scientific sources and datasets. `HARDWARE_AMENDMENT.md` and `MEMORY_AMENDMENT.md` record the execution changes and startup failures. `runs/` records every scheduler submission. Old source files and manuscript files were not modified.

Commands from the repository root, using `.runtime/oat_conda/bin/python`:

- `.../shuffled_control/release_roster.py --submit` checks the real first-update gate and idempotently submits the fixed remaining 19 training cells. It refuses release before the gate.
- `.../shuffled_control/analyze.py` writes `results.json` and `RESULTS.md` for the same-hardware primary comparison, explicitly listing incomplete cells.
- `.../shuffled_control/analyze.py --require-complete` exits nonzero until all 40 endpoint/decode cells exist.
- `.../shuffled_control/analyze.py --comparator historical_hardware` produces a separately labeled sensitivity using historical A6000 matched seeds 45/46.

The fresh 720-prompt test per disclosure is under `../evidence_use/data/heldout_{disclosure}`. Training jobs append greedy and five-draw stochastic evaluation of final step 00301, using max 192 tokens and the original forecast-array grammar. The primary analysis averages decodes within episode and clusters all evidence depths from one trajectory together.

Validation: 5 shuffle-invariant tests, 2 direct actual-trainer reward-path tests, and 1 trajectory-cluster bootstrap test passed. Read-only independent review found no control/reward-path blocker. Results remain pending while training is in progress.
