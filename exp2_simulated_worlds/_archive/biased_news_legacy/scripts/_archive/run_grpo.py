#!/usr/bin/env python3
"""GRPO training for the biased-news sequential survey forecasting task.

The model receives a sequence of (event, survey) pairs from a city campaign
and must predict the next survey value.  Latent: whether the city's news
coverage is biased (dampening negative event effects) or neutral (full effects).
A well-calibrated model must infer this hidden variable from the history.

Three context variants (V0/V1/V2) differ in what the prompt says about the
polling company — from nothing (V0) to an explicit bias warning (V2).

Reward: MAE-based.  Perfect prediction → +forecast_scale; error=50 → 0.

Modes
-----
  dryrun  – run local rollouts against a served model; verify prompt & parsing
  eval    – score a checkpoint on the eval set; write results JSONL + summary
  train   – launch Agent-lightning / Tinker GRPO training

Examples
--------
  # generate data first (from biased_news/)
  python scripts/generate_tasks.py

  # smoke-test a few rollouts locally
  python scripts/run_grpo.py dryrun --max-tasks 4

  # full evaluation pass
  python scripts/run_grpo.py eval \\
      --tasks-file data/biased_news_eval.tasks.jsonl --max-tasks 500

  # train
  python scripts/run_grpo.py train \\
      --config configs/biased_news_v1.yml \\
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

ROOT = Path(__file__).resolve().parent.parent   # biased_news/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_TASKS     = ROOT / "data" / "biased_news_train.tasks.jsonl"
DEFAULT_EVAL_FILE = ROOT / "data" / "biased_news_eval.tasks.jsonl"
DEFAULT_MODEL     = os.getenv("QWEN_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
DEFAULT_API_BASE  = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")

_AZURE_AI_PROJECT_BASE = "https://forecasting-agents-resource.services.ai.azure.com/api/projects/forecasting-agents"
AZURE_OPENAI_V1_SUFFIX = "/openai/v1"
DEFAULT_API_KEY   = os.getenv("OPENAI_API_KEY", "local-no-key-required")

REWARD_VERSION = "biased_news_mae_v1"
DEFAULT_FORECAST_REWARD_SCALE = 2.0
MAX_SURVEY_ERROR = 50.0   # normalizer: error >= 50 gets reward <= 0

_TRAIN_ROLLOUT_LOG_LOCK = threading.Lock()

_SURVEY_RE = re.compile(
    r'"?(?:predicted_survey|survey|prediction)"?\s*:\s*"?(-?\d+(?:\.\d+)?)"?',
    re.IGNORECASE,
)
_RATIONALE_RE = re.compile(
    r'"?rationale"?\s*:\s*"(?P<rationale>.*)',
    re.IGNORECASE | re.DOTALL,
)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BiasedNewsForecastTask:
    task_id:          str
    variant:          str
    context_prefix:   str
    history:          tuple           # tuple of {"step", "event", "survey"} dicts
    next_event:       str             # the T+1 event text — shown, survey hidden
    question:         str
    settlement_value: float | None    # true E[Survey_{T+1}]

    @classmethod
    def from_json(cls, row: dict[str, Any]) -> "BiasedNewsForecastTask":
        return cls(
            task_id          = str(row["task_id"]),
            variant          = str(row.get("variant", "V0")),
            context_prefix   = str(row.get("context_prefix", "")),
            history          = tuple(row.get("history", [])),
            next_event       = str(row.get("next_event", "")),
            question         = str(row["question"]),
            settlement_value = _parse_float_or_none(row.get("settlement_value")),
        )


@dataclass(frozen=True)
class BiasedNewsForecastResult:
    task_id:            str
    predicted_survey:   float
    rationale:          str
    raw_response:       str
    conversation_trace: list[dict[str, Any]]
    reward:             float | None
    ae_loss:            float | None
    se_loss:            float | None

    def to_json(self) -> dict[str, Any]:
        return {
            "task_id":            self.task_id,
            "predicted_survey":   self.predicted_survey,
            "rationale":          self.rationale,
            "raw_response":       self.raw_response,
            "conversation_trace": self.conversation_trace,
            "reward":             self.reward,
            "ae_loss":            self.ae_loss,
            "se_loss":            self.se_loss,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_thinking_model(model: str) -> bool:
    lower = model.lower()
    return "qwen" in lower or "qwq" in lower


def _make_openai_client(
    api_key: str,
    api_base: str,
    *,
    timeout: float = 90.0,
    max_retries: int = 1,
    azure_auth: bool = False,
) -> OpenAI:
    """Create an OpenAI client, with optional Azure AI Foundry header-based auth."""
    if azure_auth:
        return OpenAI(
            base_url    = api_base,
            api_key     = "placeholder",
            default_headers = {"api-key": api_key},
            timeout     = timeout,
            max_retries = max_retries,
        )
    return OpenAI(
        base_url    = api_base,
        api_key     = api_key,
        timeout     = timeout,
        max_retries = max_retries,
    )


def _parse_float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )


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
) -> tuple[list[BiasedNewsForecastTask], list[dict[str, Any]], dict[str, Any]]:
    rows = _read_jsonl_rows(path)
    selected, selection = _select_task_rows(
        rows,
        max_tasks         = int(args.max_tasks),
        task_offset       = int(args.task_offset),
        task_limit        = int(args.task_limit),
        task_shard_index  = args.task_shard_index,
        task_shard_count  = int(args.task_shard_count),
    )
    tasks = [BiasedNewsForecastTask.from_json(r) for r in selected]
    return tasks, selected, selection


# ── Prompt ────────────────────────────────────────────────────────────────────

def _build_prompt(task: BiasedNewsForecastTask) -> str:
    """
    Build the full prompt shown to the model.

    Structure:
      [system preamble]
      [optional context_prefix if variant != V0]
      [numbered timeline table]
      [question]
      [output format instruction]
      /no_think
    """
    lines: list[str] = [
        "You are a political analyst tracking a local election campaign.",
        "You have been monitoring public opinion through a sequence of events "
        "and weekly survey results (scale 0–100, where higher means more support "
        "for the leading candidate).",
        "",
    ]

    if task.context_prefix:
        lines += [task.context_prefix, ""]

    lines += [
        "Campaign timeline:",
        "",
        "| Week | Event | Survey |",
        "|------|-------|--------|",
    ]
    for step in task.history:
        lines.append(
            f"| {step['step']:>4} | {step['event']:<70} | {step['survey']:>6} |"
        )
    # T+1 row: event known, survey to predict
    T_plus_1 = len(task.history) + 1
    lines.append(
        f"| {T_plus_1:>4} | {task.next_event:<70} | {'?':>6} |"
    )

    lines += [
        "",
        task.question,
        "",
        "Respond with a JSON object only — no prose outside it:",
        '{"rationale": "one or two sentence explanation", "predicted_survey": <integer 0–100>}',
        "/no_think",
    ]
    return "\n".join(lines)


# ── Response parsing ──────────────────────────────────────────────────────────

def _parse_predicted_survey(raw: str) -> float:
    """Extract predicted_survey from JSON-ish output; clamp to [0, 100]."""
    m = _SURVEY_RE.search(raw)
    if m:
        try:
            v = float(m.group(1))
            return max(0.0, min(100.0, v))
        except ValueError:
            pass
    return 50.0   # fallback to midpoint


def _parse_rationale(raw: str) -> str:
    m = _RATIONALE_RE.search(raw)
    if m:
        text = m.group("rationale")
        text = re.sub(r'"\s*,?\s*"?predicted_survey.*', "", text, flags=re.DOTALL)
        return text.strip().rstrip('"').strip()
    return ""


# ── Reward ────────────────────────────────────────────────────────────────────

def _compute_reward(
    predicted: float,
    settlement_value: float,
    forecast_scale: float,
) -> tuple[float, float, float]:
    """Return (reward, ae_loss, se_loss).

    Reward = (1 - ae / MAX_SURVEY_ERROR) * scale
      Perfect (ae=0) → +scale
      ae=50          → 0
      ae>=50         → negative (capped at -scale)
    """
    ae      = abs(predicted - settlement_value)
    se      = (predicted - settlement_value) ** 2
    reward  = (1.0 - min(ae, MAX_SURVEY_ERROR) / MAX_SURVEY_ERROR) * forecast_scale
    return reward, ae, se


# ── Core forecast function ────────────────────────────────────────────────────

def forecast_survey(
    client: OpenAI,
    task: BiasedNewsForecastTask,
    *,
    model: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> BiasedNewsForecastResult:
    """Single-turn LLM call: event timeline → predicted_survey."""
    prompt   = _build_prompt(task)
    messages = [{"role": "user", "content": prompt}]

    kwargs: dict[str, Any] = dict(
        model       = model,
        messages    = messages,
        max_tokens  = max_tokens,
        temperature = temperature,
    )
    if _is_thinking_model(model):
        kwargs["extra_body"] = {"enable_thinking": False}

    response = client.chat.completions.create(**kwargs)
    raw               = (response.choices[0].message.content or "").strip()
    predicted_survey  = _parse_predicted_survey(raw)
    rationale         = _parse_rationale(raw)

    reward = ae_loss = se_loss = None

    if task.settlement_value is not None:
        forecast_scale = _env_float("FORECAST_REWARD_SCALE", DEFAULT_FORECAST_REWARD_SCALE)
        reward, ae_loss, se_loss = _compute_reward(
            predicted_survey, task.settlement_value, forecast_scale
        )

    return BiasedNewsForecastResult(
        task_id            = task.task_id,
        predicted_survey   = predicted_survey,
        rationale          = rationale,
        raw_response       = raw,
        conversation_trace = messages + [{"role": "assistant", "content": raw}],
        reward             = reward,
        ae_loss            = ae_loss,
        se_loss            = se_loss,
    )


# ── Agent-lightning / Tinker integration ─────────────────────────────────────

def _import_agent_lightning() -> Any:
    try:
        import agent_lightning as agl  # type: ignore
    except ImportError:
        import agentlightning as agl   # type: ignore
    return agl


def _import_agl_tinker() -> tuple[Any, Any, Any]:
    try:
        from agl_tinker import AGLDatasetBuilder, Config, Tinker  # type: ignore
    except ImportError:
        from agl_tinker.algo  import Tinker           # type: ignore
        from agl_tinker.env   import AGLDatasetBuilder # type: ignore
        from agl_tinker.train import Config            # type: ignore
    return AGLDatasetBuilder, Config, Tinker


def _patch_agl_tinker_compat() -> None:
    try:
        import agl_tinker.train as agl_tinker_train  # type: ignore
        from agl_tinker.llm import TinkerLLM          # type: ignore
        from tinker_cookbook import checkpoint_utils   # type: ignore
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

    if not getattr(TinkerLLM, "_biased_news_fresh_seed_patched", False):
        original_acompletion = TinkerLLM.acompletion
        seed_rng = random.SystemRandom()

        async def _acompletion_with_fresh_seed(self: Any, **kwargs: Any) -> Any:
            if kwargs.get("seed") is None:
                kwargs["seed"] = seed_rng.randrange(0, 2**31 - 1)
            return await original_acompletion(self, **kwargs)

        TinkerLLM.acompletion = _acompletion_with_fresh_seed  # type: ignore[method-assign]
        TinkerLLM._biased_news_fresh_seed_patched = True

    if not getattr(checkpoint_utils, "_biased_news_resume_compat_patched", False):
        original_get_last = checkpoint_utils.get_last_checkpoint

        def _get_last_dict(*args: Any, **kwargs: Any) -> Any:
            record = original_get_last(*args, **kwargs)
            return record.to_dict() if hasattr(record, "to_dict") else record

        checkpoint_utils.get_last_checkpoint = _get_last_dict  # type: ignore[method-assign]
        checkpoint_utils._biased_news_resume_compat_patched = True

    if not getattr(agl_tinker_train, "_biased_news_config_compat_patched", False):
        original_train_step = agl_tinker_train.do_train_step_and_get_sampling_client

        async def _train_step_compat(config: Any, *args: Any, **kwargs: Any) -> Any:
            _fill_tinker_config_compat(config)
            return await original_train_step(config, *args, **kwargs)

        agl_tinker_train.do_train_step_and_get_sampling_client = _train_step_compat
        agl_tinker_train._biased_news_config_compat_patched = True

    if not getattr(agl_tinker_train, "_biased_news_sampler_handoff_patched", False):
        import tinker_cookbook.rl.train as cookbook_train       # type: ignore
        from tinker.lib.public_interfaces.sampling_client import SamplingClient  # type: ignore

        original_agl_save  = agl_tinker_train.save_checkpoint_and_get_sampling_client
        original_book_save = cookbook_train.save_checkpoint_and_get_sampling_client

        async def _save_sampler_guardrails(
            training_client: Any, i_batch: int, log_path: str, save_every: int,
            *args: Any, **kwargs: Any
        ) -> Any:
            start_batch = int(kwargs.get("start_batch", 0))
            is_initial  = not kwargs.get("ttl_seconds") and kwargs.get("store") is None
            base_model  = os.getenv("TINKER_INITIAL_BASE_SAMPLER_MODEL", "").strip()
            if (
                is_initial and i_batch == start_batch
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
                    training_client=training_client, name=name, log_path=log_path,
                    loop_state={"batch": i_batch}, kind=kind,
                    ttl_seconds=kwargs.get("ttl_seconds"), store=kwargs.get("store"),
                )
                return training_client.create_sampling_client(paths["sampler_path"]), {}

            return await original_book_save(
                training_client, i_batch, log_path, save_every, *args, **kwargs
            )

        async def _agl_save_guardrails(
            training_client: Any, i_batch: int, log_path: str, save_every: int,
            *args: Any, **kwargs: Any
        ) -> Any:
            if os.getenv("TINKER_INITIAL_BASE_SAMPLER", "").strip().lower() in {"1", "true", "yes", "on"}:
                return await _save_sampler_guardrails(
                    training_client, i_batch, log_path, save_every, *args, **kwargs
                )
            return await original_agl_save(
                training_client, i_batch, log_path, save_every, *args, **kwargs
            )

        agl_tinker_train.save_checkpoint_and_get_sampling_client = _agl_save_guardrails
        cookbook_train.save_checkpoint_and_get_sampling_client   = _save_sampler_guardrails
        agl_tinker_train._biased_news_sampler_handoff_patched = True


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

def biased_news_rollout(task: dict[str, Any], llm: Any = None, rollout: Any = None) -> None:
    """Agent-lightning rollout for biased-news sequential survey forecasting."""
    agl = _import_agent_lightning()

    endpoint = getattr(llm, "endpoint", os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    model    = getattr(llm, "model",    os.getenv("QWEN_MODEL", DEFAULT_MODEL))
    client   = OpenAI(
        base_url    = endpoint,
        api_key     = os.getenv("OPENAI_API_KEY", "dummy"),
        timeout     = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "90")),
        max_retries = 1,
    )
    parsed_task = BiasedNewsForecastTask.from_json(task)
    is_eval     = _is_eval_rollout(rollout)
    temperature = (
        float(os.getenv("EVAL_TEMPERATURE", os.getenv("TEMPERATURE", "0.7")))
        if is_eval
        else float(os.getenv("TEMPERATURE", "0.7"))
    )

    result = forecast_survey(
        client, parsed_task,
        model       = model,
        max_tokens  = int(os.getenv("MAX_TOKENS", "512")),
        temperature = temperature,
    )
    _append_train_rollout_record(parsed_task, result, rollout)
    if result.reward is not None:
        agl.emit_reward(float(result.reward))


def _is_eval_rollout(rollout: Any = None) -> bool:
    mode = str(getattr(rollout, "mode", "") or "").strip().lower()
    return mode in {"val", "validation", "eval", "rolloutmode.val"} or mode.endswith(".val")


def _append_train_rollout_record(
    task:    BiasedNewsForecastTask,
    result:  BiasedNewsForecastResult,
    rollout: Any = None,
) -> None:
    log_path = os.getenv("TRAIN_ROLLOUT_LOG")
    if not log_path:
        return
    mode   = str(getattr(rollout, "mode", None) or "unknown")
    record = {
        **result.to_json(),
        "generated_at":    _utc_now(),
        "mode":            mode,
        "task_id":         task.task_id,
        "variant":         task.variant,
        "settlement_value": task.settlement_value,
    }
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _TRAIN_ROLLOUT_LOG_LOCK:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, ensure_ascii=True))
            fh.write("\n")


# ── Summary helpers ───────────────────────────────────────────────────────────

def _summarize_rollout_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored   = [r for r in rows if r.get("reward") is not None]
    variant_buckets: dict[str, list[float]] = {}
    for r in scored:
        v = str(r.get("variant") or "unknown")
        variant_buckets.setdefault(v, []).append(float(r["reward"]))
    variant_mean = {k: sum(v) / len(v) for k, v in variant_buckets.items()}
    return {
        "rollouts":          len(rows),
        "scored_rollouts":   len(scored),
        "mean_reward":       _mean_float(scored, "reward"),
        "mean_ae_loss":      _mean_float(scored, "ae_loss"),
        "mean_se_loss":      _mean_float(scored, "se_loss"),
        "mean_predicted":    _mean_float(rows,   "predicted_survey"),
        "mean_reward_by_variant": variant_mean,
    }


def _summarize_train_rollouts(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "total_rollouts": 0, "by_mode": {}}
    rows   = _read_jsonl_rows(path)
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(str(row.get("mode") or "unknown"), []).append(row)
    return {
        "path":           str(path),
        "exists":         True,
        "total_rollouts": len(rows),
        "all":            _summarize_rollout_rows(rows),
        "by_mode":        {m: _summarize_rollout_rows(mrows) for m, mrows in sorted(by_mode.items())},
    }


def _task_manifest(path: Path, *, max_tasks: int) -> dict[str, Any]:
    rows = _read_jsonl_rows(path) if path.exists() else []
    return {"path": str(path), "sha256": _file_sha256(path) if path.exists() else None,
            "rows": len(rows), "max_tasks": max_tasks}


def _checkpoint_manifest(log_dir: Path) -> dict[str, Any]:
    ckpt_path = log_dir / "checkpoints.jsonl"
    if not ckpt_path.exists():
        return {"checkpoint_record_count": 0}
    rows   = _read_jsonl_rows(ckpt_path)
    latest = rows[-1] if rows else {}
    return {
        "checkpoint_record_count": len(rows),
        "latest_state_path":   latest.get("state_path"),
        "latest_sampler_path": latest.get("sampler_path"),
    }


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
    args: argparse.Namespace, *, job_type: str, run_name: str | None, config: dict[str, Any]
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


def _base_manifest(
    args: argparse.Namespace,
    *,
    run_label: str,
    output_dir: Path,
    model: str | None = None,
    checkpoint: str | None = None,
    api_base: str | None = None,
    output_paths: dict[str, str] | None = None,
    wandb_run: Any | None = None,
) -> dict[str, Any]:
    return {
        "generated_at":   _utc_now(),
        "mode":           args.mode,
        "run_label":      run_label,
        "model":          model,
        "checkpoint":     checkpoint,
        "api_base":       api_base,
        "task_file":      _task_manifest(args.tasks_file, max_tasks=int(args.max_tasks)),
        "git":            _git_manifest(),
        "reward_version": REWARD_VERSION,
        "output_dir":     str(output_dir),
        "output_paths":   output_paths or {},
        "config_hash":    _runtime_config_hash(args),
        "wandb":          _wandb_manifest(wandb_run),
    }


# ── Core rollout / eval loop ──────────────────────────────────────────────────

def _run_rollout_eval(
    args: argparse.Namespace,
    *,
    run_label: str,
    model: str,
    api_base: str,
    api_key: str,
    output_dir: Path,
    results_filename: str,
    summary_filename: str,
    wandb_run: Any | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks, selected_task_rows, task_selection = _read_selected_tasks(args.tasks_file, args)
    if not tasks:
        raise SystemExit(f"No tasks found in {args.tasks_file}")

    os.environ["FORECAST_REWARD_SCALE"] = str(args.forecast_reward_scale)

    client = _make_openai_client(
        api_key,
        api_base,
        timeout     = float(args.llm_request_timeout_seconds),
        max_retries = 1,
        azure_auth  = bool(getattr(args, "azure_auth", False)),
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
            result = forecast_survey(
                client, task,
                model       = model,
                max_tokens  = int(args.max_tokens),
                temperature = float(args.temperature),
            )
            payload = {
                **result.to_json(),
                "task_id":          task.task_id,
                "variant":          task.variant,
                "settlement_value": task.settlement_value,
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
                f"pred={float(payload.get('predicted_survey') or 50):.1f}  "
                f"true={float(task.settlement_value or 0):.1f}  "
                f"ae={float(payload.get('ae_loss') or 0):.2f}  "
                f"reward={float(payload.get('reward') or 0):.4f}  "
                f"variant={task.variant}",
                flush=True,
            )
    except Exception as exc:
        _write_json(progress_path, {
            "status": "failed", "run_label": run_label,
            "completed_tasks": len(rows), "total_tasks": len(tasks),
            "error_type": type(exc).__name__, "error": str(exc), "updated_at": _utc_now(),
        })
        raise

    rollout_config = {
        "tasks_file": str(args.tasks_file), "output_dir": str(output_dir),
        "run_label": run_label, "model": model, "api_base": api_base,
        "max_tokens": args.max_tokens, "temperature": args.temperature,
        "forecast_reward_scale": args.forecast_reward_scale, "max_tasks": args.max_tasks,
    }
    summary = {
        "generated_at": _utc_now(), "run_label": run_label,
        "model": model, "api_base": api_base, "output_dir": str(output_dir),
        "manifest": str(manifest_path), "rollouts": _summarize_rollout_rows(rows),
    }
    manifest = _base_manifest(
        args, run_label=run_label, output_dir=output_dir,
        model=model, checkpoint=getattr(args, "checkpoint", None),
        api_base=api_base,
        output_paths={"results_jsonl": str(result_path), "summary_json": str(summary_path)},
        wandb_run=wandb_run,
    )
    _write_json(summary_path, summary)
    _write_json(manifest_path, manifest)
    _write_json(progress_path, {
        "status": "completed", "run_label": run_label,
        "completed_tasks": len(rows), "total_tasks": len(tasks),
        "results_jsonl": str(result_path), "updated_at": _utc_now(),
    })
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=True))
    return summary


# ── Argparse ──────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("mode", nargs="?", choices=("dryrun", "eval", "train"))
    parser.add_argument("--tasks-file",     type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output-dir",     type=Path, default=ROOT / "reports" / "tinker_biased_news_grpo" / "run")
    parser.add_argument("--model",          default=None)
    parser.add_argument("--checkpoint",     default=None)
    parser.add_argument("--renderer-name",  default=os.getenv("TINKER_RENDERER", "qwen3"))
    parser.add_argument("--api-base",       default=DEFAULT_API_BASE)
    parser.add_argument("--api-key",        default=DEFAULT_API_KEY)
    parser.add_argument("--azure-auth",     action="store_true",
                        help="Use Azure AI Foundry header-based auth (api-key header) instead of Bearer")
    parser.add_argument("--max-tasks",      type=int, default=8)
    parser.add_argument("--task-offset",    type=int, default=0)
    parser.add_argument("--task-limit",     type=int, default=0)
    parser.add_argument("--task-shard-index",  type=int, default=None)
    parser.add_argument("--task-shard-count",  type=int, default=0)
    parser.add_argument("--resume-results", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-tokens",     type=int,   default=512)
    parser.add_argument("--temperature",    type=float, default=1.0)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--forecast-reward-scale", type=float, default=DEFAULT_FORECAST_REWARD_SCALE)
    parser.add_argument("--learning-rate",  type=float, default=5e-5)
    parser.add_argument("--batch-size",     type=int,   default=8)
    parser.add_argument("--group-size",     type=int,   default=4)
    parser.add_argument("--n-epochs",       type=int,   default=5)
    parser.add_argument("--train-shuffle",  action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-seed",     type=int,   default=42)
    parser.add_argument("--n-runners",      type=int,   default=1)
    parser.add_argument("--lora-rank",      type=int,   default=16)
    parser.add_argument("--num-substeps",   type=int,   default=1)
    parser.add_argument("--eval-every",     type=int,   default=5)
    parser.add_argument("--skip-initial-eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-tasks-file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--eval-max-tasks", type=int,   default=200)
    parser.add_argument("--save-every",     type=int,   default=10)
    parser.add_argument("--llm-request-timeout-seconds", type=float, default=90.0)
    parser.add_argument("--tinker-initial-base-sampler", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tinker-force-named-sampler-saves", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--llm-proxy-port",     type=int,   default=12359)
    parser.add_argument("--llm-proxy-launch-mode", choices=("mp", "thread"), default="thread")
    parser.add_argument("--store-port",         type=int,   default=4748)
    parser.add_argument("--execution-strategy", choices=("cs", "shm"), default="shm")
    parser.add_argument("--strategy-main",      choices=("algorithm", "runner"), default="algorithm")
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
        raise SystemExit("YAML config requested but PyYAML not installed.") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"YAML config must be a mapping: {path}")
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
        parser.error("mode is required (positional or `mode:` in --config YAML)")
    return args


def _resolved_model(args: argparse.Namespace) -> str:
    if args.model:
        return str(args.model)
    if args.mode == "eval" and getattr(args, "checkpoint", None):
        return str(args.checkpoint)
    return DEFAULT_MODEL


# ── Modes ─────────────────────────────────────────────────────────────────────

def _dryrun(args: argparse.Namespace) -> None:
    model     = _resolved_model(args)
    wandb_run = _init_wandb(args, job_type="dryrun", run_name=args.wandb_run_name,
                            config={"model": model, "tasks_file": str(args.tasks_file)})
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
    wandb_run = _init_wandb(args, job_type="eval", run_name=args.wandb_run_name,
                            config={"model": model, "tasks_file": str(args.tasks_file)})
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


def _validate_grpo_train_args(args: argparse.Namespace, *, task_rows: list[dict[str, Any]]) -> None:
    if int(args.group_size) < 2:
        raise SystemExit("--group-size must be at least 2 for GRPO.")
    if int(args.batch_size) < 1:
        raise SystemExit("--batch-size must be at least 1.")
    if len(task_rows) < int(args.batch_size):
        raise SystemExit(
            f"Effective train rows ({len(task_rows)}) < batch_size ({args.batch_size})."
        )


def _train(args: argparse.Namespace) -> None:
    model = _resolved_model(args)
    if args.execution_strategy == "shm" and int(args.n_runners) != 1:
        raise SystemExit("--execution-strategy shm requires --n-runners 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = _read_jsonl_rows(args.tasks_file)
    if not all_rows:
        raise SystemExit(f"No task rows found in {args.tasks_file}")

    task_rows, task_selection = _select_task_rows(
        all_rows,
        max_tasks        = int(args.max_tasks) if int(args.max_tasks) > 0 else len(all_rows),
        task_offset      = int(args.task_offset),
        task_limit       = int(args.task_limit),
        task_shard_index = args.task_shard_index,
        task_shard_count = int(args.task_shard_count),
    )
    _validate_grpo_train_args(args, task_rows=task_rows)

    eval_rows = _read_jsonl_rows(args.eval_tasks_file, args.eval_max_tasks) if args.eval_tasks_file else []
    if not eval_rows:
        eval_rows = task_rows[:max(1, min(len(task_rows), int(args.batch_size)))]

    checkpoints_path      = args.output_dir / "checkpoints.jsonl"
    train_rollouts_path   = args.output_dir / "train_rollouts.jsonl"
    dataset_schedule_path = args.output_dir / "dataset_schedule.json"

    if train_rollouts_path.exists() and not checkpoints_path.exists():
        train_rollouts_path.unlink()

    train_config = {
        "tasks_file": str(args.tasks_file), "output_dir": str(args.output_dir),
        "model": model, "renderer_name": args.renderer_name,
        "learning_rate": float(args.learning_rate), "batch_size": int(args.batch_size),
        "group_size": int(args.group_size), "n_epochs": int(args.n_epochs),
        "train_shuffle": bool(args.train_shuffle), "train_seed": int(args.train_seed),
        "n_runners": int(args.n_runners), "lora_rank": int(args.lora_rank),
        "num_substeps": int(args.num_substeps), "eval_every": int(args.eval_every),
        "skip_initial_eval": bool(args.skip_initial_eval),
        "eval_tasks_file": str(args.eval_tasks_file) if args.eval_tasks_file else None,
        "eval_max_tasks": int(args.eval_max_tasks), "save_every": int(args.save_every),
        "max_tokens": int(args.max_tokens), "temperature": float(args.temperature),
        "eval_temperature": float(args.eval_temperature),
        "forecast_reward_scale": float(args.forecast_reward_scale),
        "reward_version": REWARD_VERSION, "task_selection": task_selection,
    }

    _write_json(args.output_dir / "train_config.json", train_config)

    wandb_run = _init_wandb(args, job_type="train", run_name=args.wandb_run_name,
                            config=train_config)

    if args.agent_lightning_examples_path:
        sys.path.insert(0, str(args.agent_lightning_examples_path.resolve()))
    try:
        agl = _import_agent_lightning()
        AGLDatasetBuilder, Config, Tinker = _import_agl_tinker()
        _patch_agl_tinker_compat()
    except ImportError as exc:
        raise SystemExit(
            "Training mode requires Agent-lightning + agl_tinker.\n"
            "See elections/scripts/run_tinker_elections_grpo.py for setup instructions."
        ) from exc

    os.environ.update({
        "QWEN_MODEL":                    model,
        "FORECAST_REWARD_SCALE":         str(args.forecast_reward_scale),
        "MAX_TOKENS":                    str(args.max_tokens),
        "TEMPERATURE":                   str(args.temperature),
        "EVAL_TEMPERATURE":              str(args.eval_temperature),
        "LLM_REQUEST_TIMEOUT_SECONDS":   str(args.llm_request_timeout_seconds),
        "TRAIN_ROLLOUT_LOG":             str(train_rollouts_path),
        "TINKER_INITIAL_BASE_SAMPLER":   "1" if args.tinker_initial_base_sampler else "0",
        "TINKER_INITIAL_BASE_SAMPLER_MODEL": model if args.tinker_initial_base_sampler else "",
        "TINKER_FORCE_NAMED_SAMPLER_SAVES":  "1" if args.tinker_force_named_sampler_saves else "0",
        "AGL_TINKER_SKIP_INITIAL_EVAL":  "1" if bool(args.skip_initial_eval) else "0",
    })

    config = Config(
        learning_rate   = float(args.learning_rate),
        dataset_builder = AGLDatasetBuilder(
            batch_size = int(args.batch_size),
            group_size = int(args.group_size),
            shuffle    = bool(args.train_shuffle),
            seed       = int(args.train_seed),
            n_epochs   = int(args.n_epochs),
        ),
        renderer_name     = args.renderer_name,
        model_name        = model,
        log_path          = str(args.output_dir),
        max_tokens        = int(args.max_tokens),
        lora_rank         = int(args.lora_rank),
        num_substeps      = int(args.num_substeps),
        eval_every        = int(args.eval_every),
        save_every        = int(args.save_every),
        llm_proxy_port    = int(args.llm_proxy_port),
        train_temperature = float(args.temperature),
        eval_temperature  = float(args.eval_temperature),
        wandb_project     = args.wandb_project if wandb_run is not None else None,
        wandb_name        = args.wandb_run_name if wandb_run is not None else None,
    )
    _fill_tinker_config_compat(config)

    if args.execution_strategy == "shm":
        strategy = agl.SharedMemoryExecutionStrategy(
            n_runners=int(args.n_runners), main_thread=args.strategy_main
        )
    else:
        strategy = agl.ClientServerExecutionStrategy(
            n_runners=int(args.n_runners), server_port=int(args.store_port),
            main_process=args.strategy_main,
        )

    trainer = agl.Trainer(
        algorithm  = Tinker(config),
        llm_proxy  = agl.LLMProxy(
            port=int(args.llm_proxy_port), num_retries=3,
            launch_mode=args.llm_proxy_launch_mode,
        ),
        n_runners  = int(args.n_runners),
        port       = int(args.store_port),
        strategy   = strategy,
    )
    agent = agl.rollout(biased_news_rollout) if hasattr(agl, "rollout") else biased_news_rollout

    try:
        trainer.fit(agent, train_dataset=task_rows, val_dataset=eval_rows)

        train_summary_path  = args.output_dir / "train_summary.json"
        checkpoints         = _checkpoint_manifest(args.output_dir)
        rollout_summary     = _summarize_train_rollouts(train_rollouts_path)

        train_summary = {
            "generated_at": _utc_now(),
            "status": "completed" if checkpoints.get("checkpoint_record_count") else "no_checkpoints",
            "task_rows": len(task_rows), "eval_task_rows": len(eval_rows),
            "training": train_config, "checkpoints": checkpoints, "rollouts": rollout_summary,
        }
        _write_json(train_summary_path, train_summary)
        _write_json(args.output_dir / "run_manifest.json", _base_manifest(
            args, run_label="train", output_dir=args.output_dir, model=model, wandb_run=wandb_run,
        ))
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                pass


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    {"dryrun": _dryrun, "eval": _eval, "train": _train}[args.mode](args)


if __name__ == "__main__":
    main()
