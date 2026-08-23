#!/usr/bin/env python3
"""GRPO training for the elections news-only forecasting task.

The model receives only the four E-layer news signals from the elections DAG
and must learn to forecast the winner (Blue / not-Blue) from that partial
information alone.  No tools, no external evidence — pure Brier-score reward.

Mirrors the structure of kalshi/agentic_forecasting/scripts/run_tinker_polymarket_wiki_grpo.py
but with all evidence/tool machinery removed.

Modes
-----
  dryrun   – run local rollouts against a served model; verify prompt & parsing
  eval     – score a checkpoint on the eval set; write results JSONL + summary
  train    – launch Agent-lightning / Tinker GRPO training

Examples
--------
  # generate data first (from elections/)
  python scripts/generate_training_tasks.py

  # smoke-test a few rollouts locally
  python scripts/run_tinker_elections_grpo.py dryrun --max-tasks 4

  # full evaluation pass
  python scripts/run_tinker_elections_grpo.py eval \\
      --tasks-file data/elections_eval.tasks.jsonl --max-tasks 1000

  # train
  python scripts/run_tinker_elections_grpo.py train \\
      --tasks-file data/elections_train.tasks.jsonl \\
      --eval-tasks-file data/elections_eval.tasks.jsonl \\
      --agent-lightning-examples-path /path/to/agent-lightning/examples/tinker
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("LITELLM_MODE", "PRODUCTION")

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent          # elections/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_TASKS     = ROOT / "data" / "elections_train.tasks.jsonl"
DEFAULT_EVAL_FILE = ROOT / "data" / "elections_eval.tasks.jsonl"
DEFAULT_MODEL     = os.getenv("QWEN_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
DEFAULT_API_BASE  = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_API_KEY   = os.getenv("OPENAI_API_KEY", "local-no-key-required")

REWARD_VERSION = "elections_news_brier_v1"
DEFAULT_FORECAST_REWARD_SCALE = 2.0

_TRAIN_ROLLOUT_LOG_LOCK = threading.Lock()

_YES_PROB_RE = re.compile(
    r'"?(?:yes_prob|probability)"?\s*:\s*"?(-?\d+(?:\.\d+)?)"?',
    re.IGNORECASE,
)
_RATIONALE_RE = re.compile(
    r'"?rationale"?\s*:\s*"(?P<rationale>.*)',
    re.IGNORECASE | re.DOTALL,
)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ElectionForecastTask:
    """One synthetic election forecasting scenario.

    settlement_yes is now a float probability P(Blue wins | E1..E4) estimated
    over many simulations, not a noisy single-draw boolean.  The Brier score
    reward is computed against this true conditional probability.
    """

    task_id:          str
    question:         str
    settlement_yes:   float | None   # true P(Blue wins | news signals), or None if unknown
    news_type:        str
    news_reliability: str
    news_tone:        str
    news_volume:      str

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "ElectionForecastTask":
        return cls(
            task_id          = str(row["task_id"]),
            question         = str(row["question"]),
            settlement_yes   = _parse_float_or_none(row.get("settlement_yes")),
            news_type        = str(row.get("news_type",        "")),
            news_reliability = str(row.get("news_reliability", "")),
            news_tone        = str(row.get("news_tone",        "")),
            news_volume      = str(row.get("news_volume",      "")),
        )


@dataclass(frozen=True)
class ElectionForecastResult:
    """Parsed LLM response + scoring fields for one rollout."""

    task_id:            str
    yes_prob:           float
    rationale:          str
    raw_response:       str
    conversation_trace: list[dict[str, Any]]
    reward:             float | None
    brier_loss:         float | None
    log_loss:           float | None
    is_correct:         bool | None

    def to_json(self) -> dict[str, Any]:
        return {
            "task_id":            self.task_id,
            "yes_prob":           self.yes_prob,
            "rationale":          self.rationale,
            "raw_response":       self.raw_response,
            "conversation_trace": self.conversation_trace,
            "reward":             self.reward,
            "brier_loss":         self.brier_loss,
            "log_loss":           self.log_loss,
            "is_correct":         self.is_correct,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_float_or_none(value: Any) -> float | None:
    """Parse settlement_yes as a float probability (supports both legacy bool and new float)."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        # Legacy boolean strings
        return 1.0 if str(value).strip().lower() in {"true", "yes", "y", "1"} else 0.0


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        parsed = float(raw)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _hash_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        _json_safe(payload), sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _runtime_config_hash(args: argparse.Namespace) -> str:
    return _hash_json({k: v for k, v in vars(args).items() if k != "api_key"})


def _path_manifest(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "sha256": None}
    return {
        "path":   str(path),
        "exists": path.exists(),
        "sha256": _file_sha256(path) if path.exists() else None,
    }


def _git_capture(*cmd: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *cmd], cwd=ROOT, check=False, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _git_manifest() -> dict[str, Any]:
    status = _git_capture("status", "--short") or ""
    lines  = [l for l in status.splitlines() if l.strip()]
    return {
        "commit": _git_capture("rev-parse", "HEAD"),
        "branch": _git_capture("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty":  bool(lines),
        "status_short_count": len(lines),
    }


def _mean_float(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]
    return sum(vals) / len(vals) if vals else None


# ── I/O ───────────────────────────────────────────────────────────────────────

def _read_jsonl_rows(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
                if limit > 0 and len(rows) >= limit:
                    break
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=True))
            fh.write("\n")


def _append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, ensure_ascii=True))
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")


# ── Task selection ────────────────────────────────────────────────────────────

def _select_task_rows(
    rows: list[dict[str, Any]],
    *,
    max_tasks: int = 0,
    task_offset: int = 0,
    task_limit: int = 0,
    task_shard_index: int | None = None,
    task_shard_count: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    full_rows   = len(rows)
    shard_start = 0
    shard_end   = full_rows
    indexed     = list(enumerate(rows))

    if int(task_shard_count) > 0:
        sc    = int(task_shard_count)
        si    = int(task_shard_index or 0)
        size  = math.ceil(full_rows / sc) if full_rows else 0
        shard_start = min(full_rows, si * size)
        shard_end   = min(full_rows, shard_start + size)
        indexed = indexed[shard_start:shard_end]

    offset = min(len(indexed), int(task_offset))
    indexed = indexed[offset:]
    limit = int(task_limit) if int(task_limit) > 0 else int(max_tasks)
    if limit > 0:
        indexed = indexed[:limit]

    selected: list[dict[str, Any]] = []
    for sel_idx, (inp_idx, row) in enumerate(indexed):
        payload = dict(row)
        payload["_input_row_index"]    = inp_idx
        payload["_selected_row_index"] = sel_idx
        selected.append(payload)

    selection = {
        "full_rows":     full_rows,
        "task_offset":   int(task_offset),
        "task_limit":    int(task_limit),
        "max_tasks":     int(max_tasks),
        "selected_rows": len(selected),
    }
    return selected, selection


def _read_selected_tasks(
    path: Path, args: argparse.Namespace
) -> tuple[list[ElectionForecastTask], list[dict[str, Any]], dict[str, Any]]:
    rows = _read_jsonl_rows(path)
    selected, selection = _select_task_rows(
        rows,
        max_tasks         = int(args.max_tasks),
        task_offset       = int(args.task_offset),
        task_limit        = int(args.task_limit),
        task_shard_index  = args.task_shard_index,
        task_shard_count  = int(args.task_shard_count),
    )
    tasks = [ElectionForecastTask.from_json(r) for r in selected]
    return tasks, selected, selection


# ── Prompt ────────────────────────────────────────────────────────────────────

def _build_prompt(task: ElectionForecastTask) -> str:
    # /no_think suppresses Qwen3's extended chain-of-thought at low temperatures,
    # preventing the thinking tokens from exhausting max_tokens before the JSON answer.
    return (
        "You are a political analyst making a probabilistic forecast.\n\n"
        "Based **only** on the news signals in the table below, estimate the "
        "probability that the Blue candidate wins the election.\n"
        "You have no access to any other information.\n\n"
        "| Signal            | Value                  |\n"
        "|-------------------|------------------------|\n"
        f"| News type         | {task.news_type:<22} |\n"
        f"| News reliability  | {task.news_reliability:<22} |\n"
        f"| News tone         | {task.news_tone:<22} |\n"
        f"| News volume       | {task.news_volume:<22} |\n\n"
        f"Question: {task.question}\n\n"
        "Respond with a JSON object only — no prose outside it:\n"
        '{"rationale": "one or two sentence explanation", "yes_prob": <float 0.0–1.0>}'
        "\n/no_think"
    )


# ── Response parsing ──────────────────────────────────────────────────────────

def _parse_yes_prob(raw: str) -> float:
    """Extract yes_prob from JSON-ish LLM output; clamp to (0, 1)."""
    m = _YES_PROB_RE.search(raw)
    if m:
        try:
            p = float(m.group(1))
            return max(1e-6, min(1 - 1e-6, p))
        except ValueError:
            pass
    return 0.5


def _parse_rationale(raw: str) -> str:
    m = _RATIONALE_RE.search(raw)
    if m:
        text = m.group("rationale")
        text = re.sub(r'"\s*,?\s*"?yes_prob.*', "", text, flags=re.DOTALL)
        return text.strip().rstrip('"').strip()
    return ""


# ── Reward ────────────────────────────────────────────────────────────────────

def _compute_reward(
    yes_prob: float,
    settlement_yes: float,
    forecast_scale: float,
) -> tuple[float, float, float, bool]:
    """Return (reward, brier_loss, log_loss, is_correct).

    is_correct: sample an actual binary election outcome from the true
    probability distribution, then check whether the model's directional
    prediction (>= 0.5 = Blue) matches that outcome.  This captures real
    prediction difficulty — even a perfectly calibrated model can only hit
    settlement_yes accuracy on average, matching the ~57.7% Bayes ceiling.
    """
    y            = float(settlement_yes)
    brier_loss   = (yes_prob - y) ** 2
    log_loss     = -(
        y * math.log2(max(yes_prob, 1e-9))
        + (1 - y) * math.log2(max(1 - yes_prob, 1e-9))
    )
    reward       = (0.25 - brier_loss) * forecast_scale
    actual_blue_wins = random.random() < y      # sample one election from true distribution
    is_correct   = (yes_prob >= 0.5) == actual_blue_wins
    return reward, brier_loss, log_loss, is_correct


# ── Core forecast function ────────────────────────────────────────────────────

def forecast_with_news(
    client: OpenAI,
    task: ElectionForecastTask,
    *,
    model: str,
    max_tokens: int = 256,
    temperature: float = 0.2,
) -> ElectionForecastResult:
    """Single-turn LLM call: news table → yes_prob."""
    prompt   = _build_prompt(task)
    messages = [{"role": "user", "content": prompt}]

    response = client.chat.completions.create(
        model       = model,
        messages    = messages,
        max_tokens  = max_tokens,
        temperature = temperature,
        extra_body  = {"enable_thinking": False},
    )
    raw       = (response.choices[0].message.content or "").strip()
    yes_prob  = _parse_yes_prob(raw)
    rationale = _parse_rationale(raw)

    reward = brier_loss = log_loss = None
    is_correct: bool | None = None

    if task.settlement_yes is not None:
        forecast_scale = _env_float("FORECAST_REWARD_SCALE", DEFAULT_FORECAST_REWARD_SCALE)
        reward, brier_loss, log_loss, is_correct = _compute_reward(
            yes_prob, task.settlement_yes, forecast_scale
        )

    return ElectionForecastResult(
        task_id            = task.task_id,
        yes_prob           = yes_prob,
        rationale          = rationale,
        raw_response       = raw,
        conversation_trace = messages + [{"role": "assistant", "content": raw}],
        reward             = reward,
        brier_loss         = brier_loss,
        log_loss           = log_loss,
        is_correct         = is_correct,
    )


# ── Agent-lightning / Tinker integration ─────────────────────────────────────

def _import_agent_lightning() -> Any:
    try:
        import agent_lightning as agl  # type: ignore
    except ImportError:
        import agentlightning as agl  # type: ignore
    return agl


def _import_agl_tinker() -> tuple[Any, Any, Any]:
    try:
        from agl_tinker import AGLDatasetBuilder, Config, Tinker  # type: ignore
    except ImportError:
        from agl_tinker.algo import Tinker  # type: ignore
        from agl_tinker.env  import AGLDatasetBuilder  # type: ignore
        from agl_tinker.train import Config  # type: ignore
    return AGLDatasetBuilder, Config, Tinker


def _patch_agl_tinker_compat() -> None:
    """Patch over version skew between Agent-lightning examples and cookbook."""
    try:
        import agl_tinker.train as agl_tinker_train  # type: ignore
        from agl_tinker.llm import TinkerLLM  # type: ignore
        from tinker_cookbook import checkpoint_utils  # type: ignore
    except ImportError:
        return

    def _canonicalize_messages(_self: Any, messages: Any) -> list[dict[str, Any]]:
        if messages is None:
            return []
        canonical: list[dict[str, Any]] = []
        for message in messages:
            row = message.model_dump() if hasattr(message, "model_dump") else dict(message)
            if row.get("content") is None:
                row["content"] = ""
            canonical.append(row)
        return canonical

    TinkerLLM._canonicalize_messages = _canonicalize_messages  # type: ignore[method-assign]

    if not getattr(TinkerLLM, "_elections_fresh_seed_patched", False):
        original_acompletion = TinkerLLM.acompletion
        seed_rng = random.SystemRandom()

        async def _acompletion_with_fresh_seed(self: Any, **kwargs: Any) -> Any:
            if kwargs.get("seed") is None:
                kwargs["seed"] = seed_rng.randrange(0, 2**31 - 1)
            return await original_acompletion(self, **kwargs)

        TinkerLLM.acompletion = _acompletion_with_fresh_seed  # type: ignore[method-assign]
        TinkerLLM._elections_fresh_seed_patched = True

    if not getattr(checkpoint_utils, "_elections_resume_record_compat_patched", False):
        original_get_last = checkpoint_utils.get_last_checkpoint

        def _get_last_dict(*args: Any, **kwargs: Any) -> Any:
            record = original_get_last(*args, **kwargs)
            return record.to_dict() if hasattr(record, "to_dict") else record

        checkpoint_utils.get_last_checkpoint = _get_last_dict  # type: ignore[method-assign]
        checkpoint_utils._elections_resume_record_compat_patched = True

    if not getattr(agl_tinker_train, "_elections_config_compat_patched", False):
        original_train_step = agl_tinker_train.do_train_step_and_get_sampling_client

        async def _train_step_compat(config: Any, *args: Any, **kwargs: Any) -> Any:
            _fill_tinker_config_compat(config)
            return await original_train_step(config, *args, **kwargs)

        agl_tinker_train.do_train_step_and_get_sampling_client = _train_step_compat
        agl_tinker_train._elections_config_compat_patched = True

    if not getattr(agl_tinker_train, "_elections_sampler_handoff_patched", False):
        import tinker_cookbook.rl.train as cookbook_train  # type: ignore
        from tinker.lib.public_interfaces.sampling_client import SamplingClient  # type: ignore

        original_agl_save  = agl_tinker_train.save_checkpoint_and_get_sampling_client
        original_book_save = cookbook_train.save_checkpoint_and_get_sampling_client

        async def _save_sampler_guardrails(
            training_client: Any, i_batch: int, log_path: str, save_every: int, *args: Any, **kwargs: Any
        ) -> Any:
            start_batch   = int(kwargs.get("start_batch", 0))
            is_initial    = not kwargs.get("ttl_seconds") and kwargs.get("store") is None
            base_model    = os.getenv("TINKER_INITIAL_BASE_SAMPLER_MODEL", "").strip()
            if (
                is_initial
                and i_batch == start_batch
                and os.getenv("TINKER_INITIAL_BASE_SAMPLER", "").strip().lower() in {"1", "true", "yes", "on"}
                and base_model
            ):
                return SamplingClient.create(training_client.holder, base_model=base_model).result(), {}

            force_named = os.getenv("TINKER_FORCE_NAMED_SAMPLER_SAVES", "").strip().lower() in {"1", "true", "yes", "on"}
            if force_named:
                due_full = save_every > 0 and i_batch > start_batch and i_batch % save_every == 0
                name = f"{i_batch:06d}" if due_full else f"sampler_{i_batch:06d}"
                kind = "both" if due_full else "sampler"
                paths = await checkpoint_utils.save_checkpoint_async(
                    training_client=training_client,
                    name=name, log_path=log_path,
                    loop_state={"batch": i_batch}, kind=kind,
                    ttl_seconds=kwargs.get("ttl_seconds"), store=kwargs.get("store"),
                )
                return training_client.create_sampling_client(paths["sampler_path"]), {}

            return await original_book_save(training_client, i_batch, log_path, save_every, *args, **kwargs)

        async def _agl_save_guardrails(
            training_client: Any, i_batch: int, log_path: str, save_every: int, *args: Any, **kwargs: Any
        ) -> Any:
            if os.getenv("TINKER_INITIAL_BASE_SAMPLER", "").strip().lower() in {"1", "true", "yes", "on"}:
                return await _save_sampler_guardrails(training_client, i_batch, log_path, save_every, *args, **kwargs)
            return await original_agl_save(training_client, i_batch, log_path, save_every, *args, **kwargs)

        agl_tinker_train.save_checkpoint_and_get_sampling_client = _agl_save_guardrails
        cookbook_train.save_checkpoint_and_get_sampling_client   = _save_sampler_guardrails
        agl_tinker_train._elections_sampler_handoff_patched = True


def _fill_tinker_config_compat(config: Any) -> None:
    defaults = {
        "loss_fn_config": None, "kl_reference_config": None,
        "temperature": getattr(config, "train_temperature", 1.0),
        "rollout_error_tolerance": False, "span_chart_every": 0,
        "async_config": None, "stream_minibatch_config": None,
        "ttl_seconds": 604800, "rolling_save_every": 0,
        "rolling_ttl_seconds": 7200, "num_groups_to_log": 0,
        "rollout_json_export": True, "max_steps": None,
    }
    for name, value in defaults.items():
        if not hasattr(config, name):
            object.__setattr__(config, name, value)


# ── Rollout function (module-level — Tinker must be able to import this) ─────

def elections_rollout(task: dict[str, Any], llm: Any = None, rollout: Any = None) -> None:
    """Agent-lightning rollout for elections news-only forecasting."""
    agl = _import_agent_lightning()

    endpoint = getattr(llm, "endpoint", os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    model    = getattr(llm, "model",    os.getenv("QWEN_MODEL", DEFAULT_MODEL))
    client   = OpenAI(
        base_url   = endpoint,
        api_key    = os.getenv("OPENAI_API_KEY", "dummy"),
        timeout    = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "90")),
        max_retries = 1,
    )
    parsed_task = ElectionForecastTask.from_json(task)
    is_eval     = _is_eval_rollout(rollout)
    temperature = (
        float(os.getenv("EVAL_TEMPERATURE", os.getenv("TEMPERATURE", "0.2")))
        if is_eval
        else float(os.getenv("TEMPERATURE", "0.2"))
    )

    result = forecast_with_news(
        client,
        parsed_task,
        model       = model,
        max_tokens  = int(os.getenv("MAX_TOKENS", "256")),
        temperature = temperature,
    )
    _append_train_rollout_record(parsed_task, result, rollout)
    if result.reward is not None:
        agl.emit_reward(float(result.reward))


def _is_eval_rollout(rollout: Any = None) -> bool:
    mode = str(getattr(rollout, "mode", "") or "").strip().lower()
    return mode in {"val", "validation", "eval", "rolloutmode.val"} or mode.endswith(".val")


def _append_train_rollout_record(
    task:    ElectionForecastTask,
    result:  ElectionForecastResult,
    rollout: Any = None,
) -> None:
    log_path = os.getenv("TRAIN_ROLLOUT_LOG")
    if not log_path:
        return
    mode = str(getattr(rollout, "mode", None) or "unknown")
    record = {
        **result.to_json(),
        "generated_at":     _utc_now(),
        "mode":             mode,
        "task_id":          task.task_id,
        "question":         task.question,
        "settlement_yes":   task.settlement_yes,
        "news_type":        task.news_type,
        "news_reliability": task.news_reliability,
        "news_tone":        task.news_tone,
        "news_volume":      task.news_volume,
    }
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _TRAIN_ROLLOUT_LOG_LOCK:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, ensure_ascii=True))
            fh.write("\n")


# ── Summary / manifest helpers ────────────────────────────────────────────────

def _summarize_rollout_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored   = [r for r in rows if r.get("reward") is not None]
    accurate = [r for r in rows if r.get("is_correct") is not None]
    tone_buckets: dict[str, list[float]] = {}
    for r in scored:
        tone = str(r.get("news_tone") or "unknown")
        tone_buckets.setdefault(tone, []).append(float(r["reward"]))
    tone_mean = {k: sum(v) / len(v) for k, v in tone_buckets.items()}
    return {
        "rollouts":         len(rows),
        "scored_rollouts":  len(scored),
        "mean_reward":      _mean_float(scored,   "reward"),
        "mean_brier_loss":  _mean_float(scored,   "brier_loss"),
        "mean_log_loss":    _mean_float(scored,   "log_loss"),
        "mean_yes_prob":    _mean_float(rows,      "yes_prob"),
        "accuracy":         (
            sum(1 for r in accurate if r.get("is_correct") is True) / len(accurate)
            if accurate else None
        ),
        "mean_reward_by_tone": tone_mean,
    }


def _summarize_train_rollouts(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "total_rollouts": 0, "by_mode": {}}
    rows   = _read_jsonl_rows(path)
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(str(row.get("mode") or "unknown"), []).append(row)
    return {
        "path":          str(path),
        "exists":        True,
        "total_rollouts": len(rows),
        "all":           _summarize_rollout_rows(rows),
        "by_mode":       {m: _summarize_rollout_rows(mrows) for m, mrows in sorted(by_mode.items())},
    }


def _task_manifest(path: Path, *, max_tasks: int) -> dict[str, Any]:
    rows = _read_jsonl_rows(path) if path.exists() else []
    return {
        "path":      str(path),
        "sha256":    _file_sha256(path) if path.exists() else None,
        "rows":      len(rows),
        "max_tasks": max_tasks,
    }


def _checkpoint_manifest(log_dir: Path) -> dict[str, Any]:
    ckpt_path = log_dir / "checkpoints.jsonl"
    if not ckpt_path.exists():
        return {"checkpoint_record_count": 0}
    rows = _read_jsonl_rows(ckpt_path)
    latest = rows[-1] if rows else {}
    return {
        "checkpoint_record_count": len(rows),
        "latest_state_path":   latest.get("state_path"),
        "latest_sampler_path": latest.get("sampler_path"),
    }


def _train_batch_schedule(
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    n_epochs:   int,
    shuffle:    bool,
    seed:       int,
) -> dict[str, Any]:
    batch_size = max(1, int(batch_size))
    n_epochs   = max(1, int(n_epochs))
    rng        = random.Random(seed)
    batches: list[dict[str, Any]] = []
    for epoch in range(n_epochs):
        indices = list(range(len(rows)))
        if shuffle:
            rng.shuffle(indices)
        kept = indices[: len(indices) - len(indices) % batch_size]
        for b, start in enumerate(range(0, len(kept), batch_size)):
            batch_indices = kept[start: start + batch_size]
            batches.append({
                "global_batch_index":   len(batches),
                "epoch":                epoch,
                "batch_index_in_epoch": b,
                "row_indices":          batch_indices,
                "task_ids":             [str(rows[i].get("task_id") or i) for i in batch_indices],
            })
    return {
        "policy":      "shuffled_per_epoch" if shuffle else "sequential_file_order",
        "batch_size":  batch_size,
        "n_epochs":    n_epochs,
        "train_rows":  len(rows),
        "batch_count": len(batches),
        "batches":     batches,
    }


def _eval_schedule(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"eval_rows": len(rows), "task_ids": [str(r.get("task_id") or i) for i, r in enumerate(rows)]}


def _base_manifest(
    args:       argparse.Namespace,
    *,
    run_label:  str,
    output_dir: Path,
    model:      str | None  = None,
    checkpoint: str | None  = None,
    api_base:   str | None  = None,
    output_paths: dict[str, str] | None = None,
    wandb_run:  Any | None  = None,
) -> dict[str, Any]:
    return {
        "generated_at": _utc_now(),
        "mode":         args.mode,
        "run_label":    run_label,
        "model":        model,
        "checkpoint":   checkpoint,
        "api_base":     api_base,
        "task_file":    _task_manifest(args.tasks_file, max_tasks=int(args.max_tasks)),
        "git":          _git_manifest(),
        "reward_version": REWARD_VERSION,
        "output_dir":   str(output_dir),
        "output_paths": output_paths or {},
        "config_hash":  _runtime_config_hash(args),
        "wandb":        _wandb_manifest(wandb_run),
    }


def _rollout_summary_from_rows(
    args:       argparse.Namespace,
    *,
    rows:       list[dict[str, Any]],
    run_label:  str,
    model:      str,
    api_base:   str,
    output_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    return {
        "generated_at": _utc_now(),
        "run_label":    run_label,
        "model":        model,
        "api_base":     api_base,
        "output_dir":   str(output_dir),
        "manifest":     str(manifest_path),
        "rollouts":     _summarize_rollout_rows(rows),
    }


def _rollout_config(
    args:       argparse.Namespace,
    *,
    model:      str,
    api_base:   str,
    output_dir: Path,
    run_label:  str,
) -> dict[str, Any]:
    return {
        "tasks_file":           str(args.tasks_file),
        "output_dir":           str(output_dir),
        "run_label":            run_label,
        "model":                model,
        "checkpoint":           args.checkpoint,
        "api_base":             api_base,
        "max_tokens":           args.max_tokens,
        "temperature":          args.temperature,
        "eval_temperature":     args.eval_temperature,
        "forecast_reward_scale": args.forecast_reward_scale,
        "max_tasks":            args.max_tasks,
        "task_offset":          args.task_offset,
        "task_limit":           args.task_limit,
    }


# ── W&B ───────────────────────────────────────────────────────────────────────

def _wandb_manifest(run: Any | None) -> dict[str, Any]:
    if run is None:
        return {"enabled": False}
    out: dict[str, Any] = {"enabled": True}
    for key in ("id", "name", "project", "entity", "url", "mode"):
        value = getattr(run, key, None)
        if value not in (None, ""):
            out[key] = str(value)
    return out


def _init_wandb(
    args:     argparse.Namespace,
    *,
    job_type: str,
    run_name: str | None,
    config:   dict[str, Any],
) -> Any | None:
    project = args.wandb_project or os.getenv("WANDB_PROJECT")
    if args.no_wandb or (not args.wandb and not project):
        return None
    try:
        import wandb  # type: ignore
    except ImportError:
        return None
    return wandb.init(
        project  = project,
        entity   = args.wandb_entity,
        name     = run_name,
        job_type = job_type,
        config   = config,
        mode     = args.wandb_mode or os.getenv("WANDB_MODE"),
        tags     = [t for t in (args.wandb_tags or "").split(",") if t.strip()],
        reinit   = True,
    )


def _prepare_train_wandb(args: argparse.Namespace, config: dict[str, Any]) -> Any | None:
    return _init_wandb(args, job_type="train", run_name=args.wandb_run_name, config=config)


def _wandb_log_summary(run: Any | None, summary: dict[str, Any], *, prefix: str = "") -> None:
    if run is None:
        return
    flat: dict[str, Any] = {}
    rollouts = summary.get("rollouts") or {}
    for key in ("mean_reward", "accuracy", "mean_brier_loss", "mean_log_loss"):
        if rollouts.get(key) is not None:
            flat[f"{prefix}/{key}" if prefix else key] = rollouts[key]
    if flat:
        run.log(flat)


# ── Argparse ──────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run, evaluate, or launch a Tinker/Agent-lightning GRPO training job "
            "for the elections news-only forecasting agent."
        )
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "mode", nargs="?", choices=("dryrun", "eval", "train"),
        help="dryrun/eval run local rollouts; train launches Agent-lightning/Tinker.",
    )
    parser.add_argument("--tasks-file",     type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output-dir",     type=Path, default=ROOT / "reports" / "tinker_elections_grpo" / "run")
    parser.add_argument("--model",          default=None)
    parser.add_argument("--checkpoint",     default=None)
    parser.add_argument("--renderer-name",  default=os.getenv("TINKER_RENDERER", "qwen3"))
    parser.add_argument("--api-base",       default=DEFAULT_API_BASE)
    parser.add_argument("--api-key",        default=DEFAULT_API_KEY)
    parser.add_argument("--max-tasks",      type=int, default=8)
    parser.add_argument("--task-offset",    type=int, default=0)
    parser.add_argument("--task-limit",     type=int, default=0)
    parser.add_argument("--task-shard-index",  type=int, default=None)
    parser.add_argument("--task-shard-count",  type=int, default=0)
    parser.add_argument(
        "--resume-results",
        action=argparse.BooleanOptionalAction, default=False,
        help="Continue from an existing results JSONL in --output-dir.",
    )
    # Generation
    parser.add_argument("--max-tokens",     type=int,   default=256)
    parser.add_argument("--temperature",    type=float, default=0.2)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    # Reward
    parser.add_argument("--forecast-reward-scale", type=float, default=DEFAULT_FORECAST_REWARD_SCALE)
    # GRPO / Tinker
    parser.add_argument("--learning-rate",  type=float, default=1e-5)
    parser.add_argument("--batch-size",     type=int,   default=8)
    parser.add_argument("--group-size",     type=int,   default=4)
    parser.add_argument("--n-epochs",       type=int,   default=1)
    parser.add_argument("--train-shuffle",  action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train-seed",     type=int,   default=42)
    parser.add_argument("--n-runners",      type=int,   default=4)
    parser.add_argument("--lora-rank",      type=int,   default=32)
    parser.add_argument("--num-substeps",   type=int,   default=1)
    parser.add_argument("--eval-every",     type=int,   default=0)
    parser.add_argument("--skip-initial-eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-tasks-file", type=Path, default=None)
    parser.add_argument("--eval-max-tasks", type=int,   default=1000)
    parser.add_argument("--save-every",     type=int,   default=20)
    parser.add_argument("--llm-request-timeout-seconds", type=float, default=90.0)
    parser.add_argument(
        "--tinker-initial-base-sampler",
        action=argparse.BooleanOptionalAction,
        default=(os.getenv("TINKER_INITIAL_BASE_SAMPLER", "").strip().lower() in {"1", "true", "yes", "on"}),
    )
    parser.add_argument(
        "--tinker-force-named-sampler-saves",
        action=argparse.BooleanOptionalAction,
        default=(os.getenv("TINKER_FORCE_NAMED_SAMPLER_SAVES", "").strip().lower() in {"1", "true", "yes", "on"}),
    )
    parser.add_argument("--llm-proxy-port",     type=int,   default=12358)
    parser.add_argument("--llm-proxy-launch-mode", choices=("mp", "thread"), default="thread")
    parser.add_argument("--store-port",         type=int,   default=4747)
    parser.add_argument("--execution-strategy", choices=("cs", "shm"), default="cs")
    parser.add_argument("--strategy-main",      choices=("algorithm", "runner"), default="algorithm")
    # W&B
    parser.add_argument("--wandb",       action="store_true")
    parser.add_argument("--no-wandb",    action="store_true")
    parser.add_argument("--wandb-project",  default=os.getenv("WANDB_PROJECT"))
    parser.add_argument("--wandb-entity",   default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--wandb-run-name", default=os.getenv("WANDB_RUN_NAME"))
    parser.add_argument("--wandb-mode",     default=os.getenv("WANDB_MODE"))
    parser.add_argument("--wandb-tags",     default=os.getenv("WANDB_TAGS", ""))
    parser.add_argument("--agent-lightning-examples-path", type=Path, default=None)
    return parser


def _load_yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit("YAML config requested but PyYAML is not installed. Install `pyyaml`.") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"YAML config must contain a mapping at top level: {path}")
    return {str(k).replace("-", "_"): v for k, v in payload.items()}


def _parser_destinations(parser: argparse.ArgumentParser) -> set[str]:
    return {action.dest for action in parser._actions if action.dest != argparse.SUPPRESS}  # pylint: disable=protected-access


def _parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    config_args, _ = config_parser.parse_known_args()
    parser = _build_parser()
    if config_args.config:
        config_defaults = _load_yaml_config(config_args.config)
        unknown = sorted(set(config_defaults) - _parser_destinations(parser))
        if unknown:
            raise SystemExit(f"Unknown key(s) in {config_args.config}: {', '.join(unknown)}")
        path_keys = {"tasks_file", "output_dir", "eval_tasks_file", "agent_lightning_examples_path"}
        for key in path_keys & set(config_defaults):
            if config_defaults[key] is not None:
                config_defaults[key] = Path(str(config_defaults[key]))
        parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    if args.mode is None:
        parser.error("mode is required, either as positional argument or `mode:` in --config YAML")
    return args


def _resolved_model(args: argparse.Namespace) -> str:
    if args.model:
        return str(args.model)
    if args.mode == "eval" and args.checkpoint:
        return str(args.checkpoint)
    return DEFAULT_MODEL


# ── Core rollout / eval loop ──────────────────────────────────────────────────

def _run_rollout_eval(
    args:             argparse.Namespace,
    *,
    run_label:        str,
    model:            str,
    api_base:         str,
    api_key:          str,
    output_dir:       Path,
    results_filename: str,
    summary_filename: str,
    wandb_run:        Any | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks, selected_task_rows, task_selection = _read_selected_tasks(args.tasks_file, args)
    if not tasks:
        raise SystemExit(f"No tasks found in {args.tasks_file}")

    os.environ["FORECAST_REWARD_SCALE"] = str(args.forecast_reward_scale)

    client = OpenAI(
        api_key    = api_key,
        base_url   = api_base,
        timeout    = float(args.llm_request_timeout_seconds),
        max_retries = 1,
    )

    result_path   = output_dir / results_filename
    summary_path  = output_dir / summary_filename
    progress_path = output_dir / results_filename.replace(".jsonl", "_progress.json")
    manifest_path = output_dir / "run_manifest.json"

    rows: list[dict[str, Any]] = []
    if bool(getattr(args, "resume_results", False)) and result_path.exists():
        rows = _read_jsonl_rows(result_path)
        if rows:
            print(f"[{run_label}] resuming from {len(rows)}/{len(tasks)} completed", flush=True)

    if not rows:
        result_path.write_text("", encoding="utf-8")

    _write_json(progress_path, {
        "status": "started", "run_label": run_label,
        "completed_tasks": len(rows), "total_tasks": len(tasks),
        "updated_at": _utc_now(),
    })

    try:
        resume_count = len(rows)
        for idx, (task, task_row) in enumerate(
            zip(tasks[resume_count:], selected_task_rows[resume_count:]),
            start=resume_count + 1,
        ):
            result = forecast_with_news(
                client, task,
                model       = model,
                max_tokens  = int(args.max_tokens),
                temperature = float(args.temperature),
            )
            payload = {
                **result.to_json(),
                "task_id":          task.task_id,
                "question":         task.question,
                "settlement_yes":   task.settlement_yes,
                "news_type":        task.news_type,
                "news_reliability": task.news_reliability,
                "news_tone":        task.news_tone,
                "news_volume":      task.news_volume,
                "input_row_index":  task_row.get("_input_row_index"),
            }
            rows.append(payload)
            _append_jsonl_row(result_path, payload)
            _write_json(progress_path, {
                "status": "running", "run_label": run_label,
                "completed_tasks": len(rows), "total_tasks": len(tasks),
                "last_task_id": payload.get("task_id"), "updated_at": _utc_now(),
            })
            print(
                f"[{run_label} {idx}/{len(tasks)}]  "
                f"reward={payload.get('reward'):.4f}  "
                f"p={float(payload.get('yes_prob') or 0):.4f}  "
                f"correct={payload.get('is_correct')}  "
                f"tone={task.news_tone}  "
                f"{task.question[:80]}",
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log({
                    f"{run_label}/yes_prob":   payload.get("yes_prob"),
                    f"{run_label}/reward":     payload.get("reward"),
                    f"{run_label}/brier_loss": payload.get("brier_loss"),
                    f"{run_label}/is_correct": int(payload["is_correct"]) if payload.get("is_correct") is not None else None,
                })
    except Exception as exc:
        _write_json(progress_path, {
            "status": "failed", "run_label": run_label,
            "completed_tasks": len(rows), "total_tasks": len(tasks),
            "error_type": type(exc).__name__, "error": str(exc), "updated_at": _utc_now(),
        })
        raise

    summary = _rollout_summary_from_rows(
        args, rows=rows, run_label=run_label, model=model,
        api_base=api_base, output_dir=output_dir, manifest_path=manifest_path,
    )
    manifest = _base_manifest(
        args, run_label=run_label, output_dir=output_dir,
        model=model, checkpoint=args.checkpoint if run_label == "eval" else None,
        api_base=api_base,
        output_paths={"results_jsonl": str(result_path), "summary_json": str(summary_path)},
        wandb_run=wandb_run,
    )
    _write_json(summary_path, summary)
    _write_json(manifest_path, manifest)
    _write_json(progress_path, {
        "status": "completed", "run_label": run_label,
        "completed_tasks": len(rows), "total_tasks": len(tasks),
        "results_jsonl": str(result_path), "summary_json": str(summary_path),
        "updated_at": _utc_now(),
    })
    _wandb_log_summary(wandb_run, summary, prefix=run_label)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True))
    return summary


# ── Modes ─────────────────────────────────────────────────────────────────────

def _dryrun(args: argparse.Namespace) -> None:
    model     = _resolved_model(args)
    config    = _rollout_config(args, model=model, api_base=args.api_base, output_dir=args.output_dir, run_label="dryrun")
    wandb_run = _init_wandb(args, job_type="dryrun", run_name=args.wandb_run_name, config=config)
    try:
        _run_rollout_eval(
            args, run_label="dryrun", model=model,
            api_base=args.api_base, api_key=args.api_key,
            output_dir=args.output_dir,
            results_filename="dryrun_results.jsonl",
            summary_filename="dryrun_summary.json",
            wandb_run=wandb_run,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def _eval(args: argparse.Namespace) -> None:
    model     = _resolved_model(args)
    config    = _rollout_config(args, model=model, api_base=args.api_base, output_dir=args.output_dir, run_label="eval")
    wandb_run = _init_wandb(args, job_type="eval", run_name=args.wandb_run_name, config=config)
    try:
        _run_rollout_eval(
            args, run_label="eval", model=model,
            api_base=args.api_base, api_key=args.api_key,
            output_dir=args.output_dir,
            results_filename="eval_results.jsonl",
            summary_filename="eval_summary.json",
            wandb_run=wandb_run,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def _validate_grpo_train_args(
    args: argparse.Namespace, *, task_rows: list[dict[str, Any]]
) -> None:
    if int(args.group_size) < 2:
        raise SystemExit(
            "Invalid GRPO config: --group-size must be at least 2. "
            "GRPO centers rewards within each group (advantage = reward - group_mean); "
            "group_size=1 gives zero advantage and cannot learn."
        )
    if int(args.batch_size) < 1:
        raise SystemExit("Invalid GRPO config: --batch-size must be at least 1.")
    if len(task_rows) < int(args.batch_size):
        raise SystemExit(
            f"Effective train rows ({len(task_rows)}) < batch_size ({args.batch_size}). "
            "The AGLDataset drops partial batches when shuffling."
        )


def _train(args: argparse.Namespace) -> None:
    model = _resolved_model(args)
    if args.execution_strategy == "shm" and int(args.n_runners) != 1:
        raise SystemExit("--execution-strategy shm requires --n-runners 1.")
    if args.execution_strategy == "shm" and args.llm_proxy_launch_mode != "thread":
        raise SystemExit("--execution-strategy shm requires --llm-proxy-launch-mode thread.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = _read_jsonl_rows(args.tasks_file)
    if not all_rows:
        raise SystemExit(f"No task rows found in {args.tasks_file}")

    # honour --task-offset / --task-limit / --max-tasks for train set
    task_rows, task_selection = _select_task_rows(
        all_rows,
        max_tasks        = int(args.max_tasks) if int(args.max_tasks) > 0 else len(all_rows),
        task_offset      = int(args.task_offset),
        task_limit       = int(args.task_limit),
        task_shard_index = args.task_shard_index,
        task_shard_count = int(args.task_shard_count),
    )
    _validate_grpo_train_args(args, task_rows=task_rows)

    if args.eval_tasks_file:
        eval_rows = _read_jsonl_rows(args.eval_tasks_file, args.eval_max_tasks)
        if not eval_rows:
            raise SystemExit(f"No eval rows found in {args.eval_tasks_file}")
    else:
        eval_rows = task_rows[:max(1, min(len(task_rows), int(args.batch_size)))]
        if int(args.eval_every) > 0:
            print(
                "WARNING: --eval-every enabled without --eval-tasks-file; "
                "validation uses a small training fallback.",
                file=sys.stderr,
            )

    checkpoints_path    = args.output_dir / "checkpoints.jsonl"
    train_rollouts_path = args.output_dir / "train_rollouts.jsonl"
    dataset_schedule_path = args.output_dir / "dataset_schedule.json"

    if train_rollouts_path.exists() and not checkpoints_path.exists():
        train_rollouts_path.unlink()

    train_config = {
        "tasks_file":           str(args.tasks_file),
        "output_dir":           str(args.output_dir),
        "model":                model,
        "renderer_name":        args.renderer_name,
        "learning_rate":        float(args.learning_rate),
        "batch_size":           int(args.batch_size),
        "group_size":           int(args.group_size),
        "n_epochs":             int(args.n_epochs),
        "train_shuffle":        bool(args.train_shuffle),
        "train_seed":           int(args.train_seed),
        "n_runners":            int(args.n_runners),
        "lora_rank":            int(args.lora_rank),
        "num_substeps":         int(args.num_substeps),
        "eval_every":           int(args.eval_every),
        "skip_initial_eval":    bool(args.skip_initial_eval),
        "eval_tasks_file":      str(args.eval_tasks_file) if args.eval_tasks_file else None,
        "eval_max_tasks":       int(args.eval_max_tasks),
        "eval_task_rows":       len(eval_rows),
        "save_every":           int(args.save_every),
        "max_tokens":           int(args.max_tokens),
        "temperature":          float(args.temperature),
        "eval_temperature":     float(args.eval_temperature),
        "forecast_reward_scale": float(args.forecast_reward_scale),
        "reward_version":       REWARD_VERSION,
        "task_selection":       task_selection,
    }

    dataset_schedule = {
        "generated_at": _utc_now(),
        "train": _train_batch_schedule(
            task_rows, batch_size=int(args.batch_size),
            n_epochs=int(args.n_epochs),
            shuffle=bool(args.train_shuffle), seed=int(args.train_seed),
        ),
        "eval": _eval_schedule(eval_rows),
    }
    _write_json(dataset_schedule_path, dataset_schedule)
    # Write config immediately so the training monitor can read it before training finishes
    _write_json(args.output_dir / "train_config.json", train_config)

    wandb_run = _prepare_train_wandb(args, train_config)
    if wandb_run is not None:
        wandb_run.summary["train/task_rows"] = len(task_rows)
        wandb_run.summary["train/launch"]    = 1

    if args.agent_lightning_examples_path:
        sys.path.insert(0, str(args.agent_lightning_examples_path.resolve()))
    try:
        agl = _import_agent_lightning()
        AGLDatasetBuilder, Config, Tinker = _import_agl_tinker()
        _patch_agl_tinker_compat()
    except ImportError as exc:
        raise SystemExit(
            "Training mode requires Agent-lightning + agl_tinker + Tinker credentials.\n\n"
            "Typical setup:\n"
            "  git clone https://github.com/microsoft/agent-lightning /path/to/agent-lightning\n"
            "  cd /path/to/agent-lightning && uv sync --frozen --extra apo --group dev --group agents --group tinker\n"
            "  export TINKER_API_KEY=...\n"
            "  python scripts/run_tinker_elections_grpo.py train "
            "--agent-lightning-examples-path /path/to/agent-lightning/examples/tinker\n"
        ) from exc

    os.environ["QWEN_MODEL"]                    = model
    os.environ["FORECAST_REWARD_SCALE"]         = str(args.forecast_reward_scale)
    os.environ["MAX_TOKENS"]                    = str(args.max_tokens)
    os.environ["TEMPERATURE"]                   = str(args.temperature)
    os.environ["EVAL_TEMPERATURE"]              = str(args.eval_temperature)
    os.environ["LLM_REQUEST_TIMEOUT_SECONDS"]   = str(args.llm_request_timeout_seconds)
    os.environ["TRAIN_ROLLOUT_LOG"]             = str(train_rollouts_path)
    os.environ["TINKER_INITIAL_BASE_SAMPLER"]   = "1" if args.tinker_initial_base_sampler else "0"
    os.environ["TINKER_INITIAL_BASE_SAMPLER_MODEL"] = model if args.tinker_initial_base_sampler else ""
    os.environ["TINKER_FORCE_NAMED_SAMPLER_SAVES"]  = "1" if args.tinker_force_named_sampler_saves else "0"
    os.environ["AGL_TINKER_SKIP_INITIAL_EVAL"]  = "1" if bool(args.skip_initial_eval) else "0"

    config = Config(
        learning_rate  = float(args.learning_rate),
        dataset_builder = AGLDatasetBuilder(
            batch_size = int(args.batch_size),
            group_size = int(args.group_size),
            shuffle    = bool(args.train_shuffle),
            seed       = int(args.train_seed),
            n_epochs   = int(args.n_epochs),
        ),
        renderer_name = args.renderer_name,
        model_name    = model,
        log_path      = str(args.output_dir),
        max_tokens    = int(args.max_tokens),
        lora_rank     = int(args.lora_rank),
        num_substeps  = int(args.num_substeps),
        eval_every    = int(args.eval_every),
        save_every    = int(args.save_every),
        llm_proxy_port    = int(args.llm_proxy_port),
        train_temperature = float(args.temperature),
        eval_temperature  = float(args.eval_temperature),
        wandb_project  = args.wandb_project if wandb_run is not None else None,
        wandb_name     = args.wandb_run_name if wandb_run is not None else None,
    )
    _fill_tinker_config_compat(config)

    if args.execution_strategy == "shm":
        strategy = agl.SharedMemoryExecutionStrategy(
            n_runners  = int(args.n_runners),
            main_thread = args.strategy_main,
        )
    else:
        strategy = agl.ClientServerExecutionStrategy(
            n_runners    = int(args.n_runners),
            server_port  = int(args.store_port),
            main_process = args.strategy_main,
        )

    trainer = agl.Trainer(
        algorithm  = Tinker(config),
        llm_proxy  = agl.LLMProxy(
            port        = int(args.llm_proxy_port),
            num_retries = 3,
            launch_mode = args.llm_proxy_launch_mode,
        ),
        n_runners  = int(args.n_runners),
        port       = int(args.store_port),
        strategy   = strategy,
    )
    agent = agl.rollout(elections_rollout) if hasattr(agl, "rollout") else elections_rollout

    try:
        trainer.fit(agent, train_dataset=task_rows, val_dataset=eval_rows)

        train_summary_path  = args.output_dir / "train_summary.json"
        train_manifest_path = args.output_dir / "run_manifest.json"
        checkpoints         = _checkpoint_manifest(args.output_dir)
        rollout_summary     = _summarize_train_rollouts(train_rollouts_path)
        completed           = bool(checkpoints.get("checkpoint_record_count")) or bool(rollout_summary.get("total_rollouts"))

        train_summary = {
            "generated_at":  _utc_now(),
            "status":        "completed" if completed else "no_rollouts_or_checkpoint",
            "task_rows":     len(task_rows),
            "eval_task_rows": len(eval_rows),
            "training":      train_config,
            "checkpoints":   checkpoints,
            "rollouts":      rollout_summary,
        }
        train_manifest = _base_manifest(
            args, run_label="train", output_dir=args.output_dir, model=model,
            output_paths={
                "train_summary_json":    str(train_summary_path),
                "train_rollouts_jsonl":  str(train_rollouts_path),
                "dataset_schedule_json": str(dataset_schedule_path),
                "checkpoints_jsonl":     str(checkpoints_path),
            },
            wandb_run=wandb_run,
        )
        train_manifest["training"]          = train_config
        train_manifest["dataset_schedule"]  = dataset_schedule
        train_manifest["checkpoints"]       = checkpoints
        train_manifest["rollouts"]          = rollout_summary

        _write_json(train_summary_path,  train_summary)
        _write_json(train_manifest_path, train_manifest)

        if wandb_run is not None:
            try:
                all_sum = (rollout_summary.get("all") or {})
                for key in ("mean_reward", "accuracy", "mean_brier_loss", "mean_log_loss"):
                    if all_sum.get(key) is not None:
                        wandb_run.summary[f"rollouts/all/{key}"] = all_sum[key]
                wandb_run.summary["train/finished"] = 1 if completed else 0
            except Exception as exc:
                print(f"W&B summary update failed (run may already be finished): {exc}", file=sys.stderr)
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception as exc:
                print(f"W&B finish failed: {exc}", file=sys.stderr)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    if args.mode == "dryrun":
        _dryrun(args)
    elif args.mode == "eval":
        _eval(args)
    else:
        _train(args)


if __name__ == "__main__":
    main()
