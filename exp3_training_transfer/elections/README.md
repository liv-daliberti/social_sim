# Elections simulator

This directory powers the deployed Blue-ian/Red-ian causal-DAG viewer.

- `server.py`, `ui/`, and `engine/` are the deployed service.
- `run.py` is the maintained command-line simulator.
- `scripts/generate_training_tasks.py` deterministically samples the current
  DAG and assigns exact conditional-probability labels.
- `scripts/run_tinker_elections_grpo.py`, `configs/`, and the task JSONL files
  support the training-monitor view exposed by the service.

W&B runs, model reports, and caches are local artifacts and are ignored by Git.
The retired enumerated task-generation mode was removed because the maintained
10,000-row dataset and training workflow use stochastic DAG sampling only.
