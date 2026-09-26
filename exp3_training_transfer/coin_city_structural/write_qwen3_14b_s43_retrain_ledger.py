import json, hashlib, shutil
from pathlib import Path
R = Path("/n/fs/similarity/social_sim/exp3_training_transfer/coin_city_structural")
M = Path("/tmp/claude-363432/-n-fs-similarity-social-sim/54e64e28-f3d1-4f39-8e9a-d9ce92e871cd/scratchpad/mirror/exp3_training_transfer/coin_city_structural")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
old_path = R/"runs/coin_city_qwen3_14b_scale_full_20260825T134340Z.json"
d = json.loads(old_path.read_text())
STAMP = "20260925T234238Z"
name = f"coin_city_qwen3_14b_scale_full_20260825T134340Z_s43retrain_{STAMP}.json"

registered_train = sha(R/"train_registered_s43.sh")
assert registered_train == d["train_script_sha256"] == "46c95d79063e2b89c6db461aedd1bde163f7ec1730a1e3947faf2eaca9919ffb"
current_core = sha(R/"make_paper_outputs.py")

# replace causal seed 43 with the registered retrain
submit_line = ("sbatch --parsable --partition=all --gres=gpu:a6000:2 --cpus-per-task=8 --mem=100G "
  "--time=2-00:00:00 --exclude=node206 --export=ALL,ARM=causal,BATCH=16,EVAL_BATCH=120,EVAL_STEPS=50,"
  "GEN_LEN=192,GPUS=2,LORA_ALPHA=64,LORA_RANK=32,LR=0.000001,MAX_MODEL_LEN=3072,MAX_TRAIN=4800,"
  "MODEL=Qwen/Qwen3-14B,MODEL_KEY=qwen3_14b,PROMPT_MAX_LENGTH=2304,PROMPT_TEMPLATE=auto_no_think,"
  "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,RESPONSE_W=0.60,ROLLOUT_PER_PROMPT=8,SEED=43,"
  "STOCHASTIC_N=5,STOCHASTIC_TEMP=0.7,TAG=scale_causal_qwen3_14b_s43,TEMP=1.3,VLLM_RATIO=0.78,ZERO_STAGE=2 "
  "train_registered_s43.sh").split()
submit_line[-1] = str(R/"train_registered_s43.sh")
new_jobs = []
superseded_s43 = None
for row in d["training_jobs"]:
    if row["arm"] == "causal" and row["seed"] == 43:
        superseded_s43 = row
        env = row["environment"]
        exported = dict(kv.split("=",1) for kv in submit_line[8].split(",")[1:])
        assert exported == env, (exported, env)
        new_jobs.append({
            "arm": "causal", "model": "qwen3_14b", "seed": 43, "job_id": "31481143",
            "environment": env, "command": submit_line,
            "effective_allocation": {"account": "allcs", "gres": "gpu:a6000:2", "partition": "cs"},
            "train_script": "train_registered_s43.sh",
            "train_script_sha256": registered_train,
        })
    else:
        new_jobs.append(row)
d["training_jobs"] = new_jobs
d["finalization_job"] = {**d["finalization_job"], "state": "CANCELLED",
    "note": "cancelled 2026-08-31; its afterok dependency on 30920615 became unsatisfiable when that job timed out"}
d["superseded_training_jobs"] = d["superseded_training_jobs"] + [
    {**superseded_s43, "superseded_reason": "TIMEOUT at 04:00:00 after an unexpected hold/time-limit change; produced no scores"}]
d["s43_retrain_amendment"] = {
    "amended_at": STAMP,
    "arm": "causal", "seed": 43,
    "failed_job_ids": {
        "30880156": "TIMEOUT at 30h; rollout actor aborted at round 230, no final adapter",
        "30920615": "TIMEOUT at 04:00:00; no scores written",
        "30948628": "COMPLETED but trained under the working-tree train.sh (checkpointing/resume edits, sha256 7ca287c389ff4f18678de9e381e5de6b0064cdd2635eebbfab3a749476685a21), which is not the registered script; excluded from the roster",
    },
    "replacement_job_id": "31481143",
    "replacement_submitted": "2026-09-23T16:30:30",
    "replacement_completed": "2026-09-25T00:42:52",
    "replacement_state": "COMPLETED",
    "replacement_elapsed": "23:26:16",
    "replacement_time_limit": "2-00:00:00",
    "replacement_script": "train_registered_s43.sh",
    "replacement_script_sha256": registered_train,
    "replacement_script_byte_identical_to_registered_train_sh": True,
    "replacement_report": str(R/"reports/scale_causal_qwen3_14b_s43_20260924_011636_j31481143"),
    "resume_dir": "",
    "scientific_change": False,
    "reason": "walltime is an sbatch parameter, not part of the registered script; the retrain uses the byte-identical registered script with a 2-day allocation",
    "finalization_job_30920625": "CANCELLED; rendering performed from this ledger instead",
    "core_analysis_sha256_note": (
        "core_analysis_sha256 above records the make_paper_outputs.py present at amendment time "
        f"({current_core}); the value registered on 2026-08-25 "
        f"({d['core_analysis_sha256']}) is in neither git history nor any stash and cannot be recovered."),
    "core_analysis_sha256_registered_20260825": d["core_analysis_sha256"],
    "train_script_sha256_note": (
        "train_script_sha256 above is the registered script under which all six trainings ran "
        "(seeds 42/44 and the prior arm via the then-unmodified train.sh; seed 43 via train_registered_s43.sh). "
        "The working-tree train.sh has since gained uncommitted checkpoint/resume options."),
}
d["core_analysis_sha256"] = current_core
d["scheduler_corrections"] = d["scheduler_corrections"] + [
    {"action": "replace timed-out job 30920615 with registered-script retrain 31481143 (2-day allocation); cancel finalizer 30920625",
     "job_id": "31481143", "reason": "seed-43 causal rerun timed out; retrain under train_registered_s43.sh", "scientific_change": False}]
d["ledger_name"] = name
d["parent_extension_ledger"] = old_path.name
d["parent_extension_ledger_sha256"] = sha(old_path)
out = R/"runs"/name
out.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n", encoding="utf-8")
shutil.copy(out, M/"runs"/name)
print("wrote", out, sha(out))
