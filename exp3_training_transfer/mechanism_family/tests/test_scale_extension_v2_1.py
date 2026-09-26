from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


FAMILY = Path(__file__).resolve().parents[1]
EXTENSION = FAMILY / "scale_extension_v2_1"
LAUNCHER = EXTENSION / "launch.py"


def load_auditor():
    spec = importlib.util.spec_from_file_location(
        "scale_extension_v2_1_auditor", EXTENSION / "audit_canaries.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render(stage: str) -> list[str]:
    completed = subprocess.run(
        [sys.executable, str(LAUNCHER), "--stage", stage],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in completed.stdout.splitlines() if line.startswith("sbatch ")]


class ScaleExtensionV21Tests(unittest.TestCase):
    def test_dry_run_freezes_exact_large_model_rosters_and_resources(self):
        canary = render("canary")
        full = render("full-render")
        self.assertEqual(len(canary), 12)
        self.assertEqual(len(full), 36)

        expected = {
            "qwen3_14b": (12, "--gres=gpu:a6000:2", "--time=36:00:00"),
            "qwen3_32b": (12, "--gres=gpu:a6000:4", "--time=48:00:00"),
            "llama3_1_70b": (12, "--gres=gpu:a6000:8", "--time=72:00:00"),
        }
        for model, (count, gpu_flag, time_flag) in expected.items():
            cells = [line for line in full if f"MODEL_KEY={model}" in line]
            self.assertEqual(len(cells), count)
            self.assertTrue(all(gpu_flag in line and time_flag in line for line in cells))

        for disclosure in ("disclosed", "undisclosed"):
            for arm in ("causal_family", "population_prior"):
                for seed in (42, 43, 44):
                    cells = [
                        line for line in full
                        if f"DISCLOSURE={disclosure}" in line
                        and f"ARM={arm}" in line
                        and f"SEED={seed}" in line
                    ]
                    self.assertEqual(len(cells), 3)

        self.assertEqual(sum("RUN_KIND=canary" in line for line in canary), 6)
        self.assertEqual(sum("large_base_eval.sh" in line for line in canary), 6)
        self.assertTrue(all("--exclude=node206" in line for line in canary + full))

    def test_gate_requires_the_exact_six_canaries_and_six_bases(self):
        auditor = load_auditor()
        models = ("qwen3_14b", "qwen3_32b", "llama3_1_70b")
        disclosures = ("disclosed", "undisclosed")
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            reports = temporary / "reports"
            reports.mkdir()
            submissions = []
            states = {}
            next_job = 9000
            summary = [{"n_draws": 60, "parse_rate": 1.0}]

            for model in models:
                for disclosure in disclosures:
                    job_id = str(next_job)
                    next_job += 1
                    states[job_id] = "COMPLETED"
                    submissions.append(
                        {
                            "kind": "canary",
                            "job_id": job_id,
                            "model_key": model,
                            "disclosure": disclosure,
                            "arm": "causal_family",
                            "seed": 42,
                        }
                    )
                    root = reports / f"scale_v2_1_canary_x_j{job_id}"
                    adapter = root / "debug_0" / "saved_models" / "step_10"
                    scores = root / "debug_0" / "eval_results"
                    adapter.mkdir(parents=True)
                    scores.mkdir(parents=True)
                    (adapter / "adapter_config.json").write_text("{}\n")
                    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
                    (scores / "step_10.scores.summary.json").write_text(json.dumps(summary))
                    (root / "stochastic_n1.scores.summary.json").write_text(json.dumps(summary))
                    (root / "train.log").write_text(
                        "actor reward 0.1\n'actor/no_eos_count': 0\nactor reward 0.3\n"
                    )

            for model in models:
                for disclosure in disclosures:
                    job_id = str(next_job)
                    next_job += 1
                    states[job_id] = "COMPLETED"
                    submissions.append(
                        {
                            "kind": "base_evaluation",
                            "job_id": job_id,
                            "model_key": model,
                            "disclosure": disclosure,
                        }
                    )
                    root = reports / f"scale_v2_1_base_{disclosure}_{model}_j{job_id}"
                    root.mkdir()
                    (root / "greedy.scores.summary.json").write_text(json.dumps(summary))
                    (root / "stochastic_n5.scores.summary.json").write_text(json.dumps(summary))

            ledger = {
                "protocol": "c3_mechanism_scale_v2_1",
                "stage": "canary",
                "submissions": submissions,
            }
            ledger_path = temporary / "ledger.json"
            ledger_path.write_text(json.dumps(ledger))
            result = auditor.audit(ledger_path, reports=reports, states=states)
            self.assertEqual(result["status"], "pass")
            self.assertEqual(len(result["jobs"]), 12)

            ledger["submissions"] = submissions[:-1]
            ledger_path.write_text(json.dumps(ledger))
            with self.assertRaisesRegex(AssertionError, "exact six-cell grid"):
                auditor.audit(ledger_path, reports=reports, states=states)


if __name__ == "__main__":
    unittest.main()
