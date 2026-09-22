#!/usr/bin/env python3
"""Bounded, resumable GPT-5.6 Responses API pilot; --dry-run is entirely offline.

The frozen input supplies every prompt. Each context has one baseline and three
independent updates. There are no model fallbacks, automatic retries, or tools.
Request reservations are durable before submission, so an interrupted request
is never silently submitted twice. Failed and unattempted outcomes remain in
the output and therefore in the planned analysis denominators.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import math
import os
from pathlib import Path
import platform
import random
import sys

from exp1_prospective.context_reversal.run_local import (
    CONDITIONS, atomic_write, canonical_json, file_sha256, output_lock,
    package_version, read_units, record_key, render_update, text_sha256, utc_now,
)

MODEL = "gpt-5.6"
BASE_URL = "https://api.openai.com/v1"
MAX_UNITS = 80
MAX_CALLS = 320
MAX_BUDGET_USD = 15.0
MAX_OUTPUT_TOKENS = 2048
SHUFFLE_SEED = 20260921
PRICES = {"input": 4.0, "cached_input": 0.4, "output": 20.0}
PROBABILITY_SCHEMA = {
    "type": "object", "properties": {"probability": {
        "type": "number", "minimum": 0, "maximum": 1,
    }}, "required": ["probability"], "additionalProperties": False,
}
SAFE_API_CODES = {
    "invalid_api_key", "invalid_request_error", "model_not_found",
    "model_not_available", "model_not_supported", "unsupported_model",
    "permission_denied", "insufficient_quota", "rate_limit_exceeded",
    "context_length_exceeded", "server_error", "account_deactivated",
}


def parse_probability(raw: str) -> float:
    def object_pairs(pairs):
        if len({key for key, _ in pairs}) != len(pairs):
            raise ValueError("Duplicate JSON fields")
        return dict(pairs)

    value = json.loads(raw, object_pairs_hook=object_pairs)
    if not isinstance(value, dict) or set(value) != {"probability"}:
        raise ValueError("Expected exactly one probability field")
    if type(value["probability"]) not in (int, float):
        raise ValueError("Probability must be numeric")
    probability = float(value["probability"])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Probability must be finite and in [0, 1]")
    return probability


def make_config(args: argparse.Namespace, units: list[dict]) -> dict:
    if args.model not in {MODEL, "gpt-5.6-sol"}:
        raise ValueError("This bounded pilot permits only gpt-5.6 or its canonical gpt-5.6-sol ID")
    if not units or len(units) > MAX_UNITS or any(u["repeat"] != 0 for u in units):
        raise ValueError("Pilot requires 1..80 frozen units, all from repeat 0")
    if len({u["family_id"] for u in units}) > 20:
        raise ValueError("Pilot permits at most 20 families")
    if not 1 <= args.concurrency <= 4 or not 1 <= args.max_calls <= MAX_CALLS:
        raise ValueError("Concurrency must be 1..4 and maximum calls 1..320")
    if not math.isfinite(args.budget_usd) or not 0 < args.budget_usd <= MAX_BUDGET_USD:
        raise ValueError("Budget must be positive and no greater than $15")
    if args.output.resolve() == args.input.resolve():
        raise ValueError("Input and output must differ")
    return {
        "schema_version": 1, "input_path": str(args.input.resolve()),
        "input_sha256": file_sha256(args.input), "unit_count": len(units),
        "model_key": args.model_key, "model": args.model, "base_url": BASE_URL,
        "code_sha256": {p.name: file_sha256(p) for p in
                        (Path(__file__), Path(__file__).with_name("run_local.py"))},
        "decode": {"reasoning_effort": "low", "max_output_tokens": MAX_OUTPUT_TOKENS,
                   "text_format": {"type": "json_schema", "name": "forecast_probability",
                                   "strict": True, "schema": PROBABILITY_SCHEMA},
                   "temperature": "omitted", "seed": "omitted", "tools": [], "store": False},
        "limits": {"max_calls": args.max_calls, "budget_usd": args.budget_usd,
                   "concurrency": args.concurrency, "max_retries": 0, "timeout_seconds": 120},
        "shuffle_seed": SHUFFLE_SEED, "prices_usd_per_million_tokens": PRICES,
        "reservation_policy": "UTF-8 bytes of prompt and JSON schema + 4096 input tokens; maximum output; no cache discount",
        "software": {"python": platform.python_version(), "openai": package_version("openai"),
                     "httpx": package_version("httpx")},
    }


def base_record(unit: dict, stage: str, condition: str, prompt: str | None,
                config: dict, prior: float | None = None) -> dict:
    return {
        **{key: unit[key] for key in ("trial_id", "family_id", "domain", "context_id", "repeat", "material_status")},
        "stage": stage, "condition": condition, "model_key": config["model_key"],
        "requested_model": config["model"],
        "status": "generation_error", "raw": "", "probability": None,
        "prior_probability": prior, "prompt": prompt,
        "prompt_sha256": text_sha256(prompt) if prompt is not None else None,
        "input_sha256": config["input_sha256"], "run_signature": text_sha256(canonical_json(config)),
        "created_at": utc_now(), "request_attempted": False, "response_received": False,
        "returned_model": None, "response_id": None, "usage": None,
        "billed_cost_estimate_usd": None, "reserved_cost_usd": 0.0, "error": None,
    }


def reservation_usd(prompt: str) -> float:
    # Tokens cannot outnumber their UTF-8 bytes. Extra room covers the message
    # envelope and schema framing, whose exact tokenization is server-owned.
    input_bound = len(prompt.encode("utf-8")) + len(canonical_json(PROBABILITY_SCHEMA).encode("utf-8")) + 4096
    return (input_bound * PRICES["input"] + MAX_OUTPUT_TOKENS * PRICES["output"]) / 1_000_000


def safe_api_error(exc: Exception) -> dict:
    # Never stringify exceptions: an authentication error may echo a secret.
    status = getattr(exc, "status_code", None)
    code = getattr(exc, "code", None)
    return {"kind": "api_error", "http_status": status if type(status) is int and 100 <= status <= 599 else None,
            "code": code if isinstance(code, str) and code in SAFE_API_CODES else None}


def fatal_api_error(error: dict) -> bool:
    return error["http_status"] in (None, 400, 401, 403, 404, 429) or error["code"] in {
        "model_not_found", "model_not_available", "model_not_supported", "unsupported_model",
        "insufficient_quota", "account_deactivated",
    }


def create_client():
    # Explicitly exclude shell endpoint/proxy settings and optional credentials.
    # SDK and transport logs are disabled before either dependency is imported.
    os.environ.pop("OPENAI_LOG", None)
    for name in ("openai", "httpx", "httpcore"):
        logger = logging.getLogger(name)
        logger.disabled = True
        logger.setLevel(logging.CRITICAL + 1)
    import httpx
    from openai import OpenAI
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=BASE_URL,
                  admin_api_key="", organization="", project="", webhook_secret="",
                  max_retries=0, timeout=120.0,
                  http_client=httpx.Client(trust_env=False, follow_redirects=False, timeout=120.0))


def response_usage(response) -> dict | None:
    source = getattr(response, "usage", None)
    if source is None:
        return None
    usage = {
        "input_tokens": getattr(source, "input_tokens", None),
        "cached_input_tokens": getattr(getattr(source, "input_tokens_details", None), "cached_tokens", 0),
        "output_tokens": getattr(source, "output_tokens", None),
        "reasoning_tokens": getattr(getattr(source, "output_tokens_details", None), "reasoning_tokens", 0),
        "total_tokens": getattr(source, "total_tokens", None),
    }
    if any(type(value) is not int or value < 0 for value in usage.values()):
        return None
    if usage["cached_input_tokens"] > usage["input_tokens"] or usage["reasoning_tokens"] > usage["output_tokens"]:
        return None
    if usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]:
        return None
    return usage


def usage_cost(usage: dict) -> float:
    # Output usage already includes reasoning tokens. Do not count them twice.
    return ((usage["input_tokens"] - usage["cached_input_tokens"]) * PRICES["input"]
            + usage["cached_input_tokens"] * PRICES["cached_input"]
            + usage["output_tokens"] * PRICES["output"]) / 1_000_000


def submit(client, row: dict, config: dict) -> tuple[dict, bool]:
    try:
        response = client.responses.create(
            model=config["model"], input=[{"role": "user", "content": row["prompt"]}],
            reasoning={"effort": "low"}, max_output_tokens=MAX_OUTPUT_TOKENS,
            text={"format": config["decode"]["text_format"]}, store=False, tools=[],
        )
    except Exception as exc:
        row["error"] = safe_api_error(exc)
        return row, fatal_api_error(row["error"])
    row.update({"response_received": True, "returned_model": getattr(response, "model", None),
                "response_id": getattr(response, "id", None), "created_at": utc_now(),
                "response_status": getattr(response, "status", None), "usage": response_usage(response)})
    raw = getattr(response, "output_text", "")
    row["raw"] = raw if isinstance(raw, str) else ""
    if row["usage"] is not None:
        row["billed_cost_estimate_usd"] = usage_cost(row["usage"])
    if row["response_status"] != "completed":
        row["error"] = {"kind": "response_not_completed"}
        return row, False
    try:
        row["probability"] = parse_probability(row["raw"])
        row["status"] = "ok"
    except (ValueError, TypeError, OverflowError):
        row["status"] = "parse_error"
        row["error"] = {"kind": "invalid_probability_json"}
    return row, False


def load_existing(output: Path, manifest_path: Path, config: dict, units: list[dict]) -> tuple[dict, dict]:
    signature = text_sha256(canonical_json(config))
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest and (manifest.get("run_signature") != signature or manifest.get("config") != config):
        raise ValueError("Existing run differs from this input/model/configuration/code; choose a new output")
    if output.exists() and output.stat().st_size and not manifest:
        raise ValueError("Nonempty output has no manifest; refusing unsafe resume")
    records, by_id = {}, {u["trial_id"]: u for u in units}
    for line in output.read_text().splitlines() if output.exists() else []:
        row = json.loads(line)
        key = record_key(row)
        if key in records or key[0] not in by_id:
            raise ValueError("Duplicate or unknown existing response")
        if row.get("input_sha256") != config["input_sha256"] or row.get("run_signature") != signature or row.get("model_key") != config["model_key"]:
            raise ValueError("Existing response provenance mismatch")
        for field in ("family_id", "domain", "context_id", "repeat", "material_status"):
            if row.get(field) != by_id[key[0]][field]:
                raise ValueError("Existing response metadata mismatch")
        if key[1:] != ("baseline", "baseline") and not (key[1] == "update" and key[2] in CONDITIONS):
            raise ValueError("Invalid existing stage/condition")
        if row.get("status") not in {"ok", "parse_error", "generation_error", "blocked_baseline"}:
            raise ValueError("Invalid existing status")
        if row["status"] == "ok":
            if type(row.get("probability")) not in (int, float) or parse_probability(row["raw"]) != row["probability"]:
                raise ValueError("Existing probability mismatch")
        elif row.get("probability") is not None:
            raise ValueError("Failed response must have null probability")
        records[key] = row
    ledger = manifest.get("attempt_ledger", [])
    attempts = {}
    for entry in ledger:
        key = tuple(entry["key"])
        if key in attempts or len(key) != 3 or key[0] not in by_id:
            raise ValueError("Invalid attempt ledger")
        if key[1:] != ("baseline", "baseline") and not (key[1] == "update" and key[2] in CONDITIONS):
            raise ValueError("Invalid attempt ledger stage/condition")
        attempts[key] = entry
    if len(attempts) > config["limits"]["max_calls"]:
        raise ValueError("Attempt ledger exceeds call ceiling")
    # Reconstruct requests that could have been submitted before a crash. Their
    # result is unknown and their reservation remains charged; never retry them.
    for key, entry in attempts.items():
        unit = by_id[key[0]]
        if key[1] == "baseline":
            prompt, prior = unit["baseline_prompt"], None
        else:
            baseline = records.get((key[0], "baseline", "baseline"))
            if baseline is None or baseline["status"] != "ok":
                raise ValueError("Attempted update lacks valid baseline")
            prior = baseline["probability"]
            prompt = render_update(unit, key[2], prior)
        if entry.get("prompt_sha256") != text_sha256(prompt) or entry.get("reserved_cost_usd") != reservation_usd(prompt):
            raise ValueError("Attempt ledger prompt/reservation mismatch")
        if key not in records:
            row = base_record(unit, key[1], key[2], prompt, config, prior)
            row.update({"request_attempted": True, "reserved_cost_usd": entry["reserved_cost_usd"],
                        "error": {"kind": "interrupted_request_result_unavailable"}})
            records[key] = row
    for key, row in records.items():
        unit = by_id[key[0]]
        if type(row.get("request_attempted")) is not bool or row["request_attempted"] != (key in attempts):
            raise ValueError("Existing response/attempt ledger mismatch")
        if row["request_attempted"] and row.get("reserved_cost_usd") != attempts[key]["reserved_cost_usd"]:
            raise ValueError("Existing reservation mismatch")
        if row["status"] in {"ok", "parse_error"} and not row["request_attempted"]:
            raise ValueError("Existing completion was never attempted")
        usage = row.get("usage")
        if usage is not None:
            if set(usage) != {"input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens", "total_tokens"} or any(type(v) is not int or v < 0 for v in usage.values()):
                raise ValueError("Invalid existing token usage")
            if usage["cached_input_tokens"] > usage["input_tokens"] or usage["reasoning_tokens"] > usage["output_tokens"] or usage["total_tokens"] != usage["input_tokens"] + usage["output_tokens"]:
                raise ValueError("Inconsistent existing token usage")
            if row.get("billed_cost_estimate_usd") != usage_cost(usage):
                raise ValueError("Existing usage cost mismatch")
        elif row.get("billed_cost_estimate_usd") is not None:
            raise ValueError("Existing cost lacks usage")
        if key[1] == "baseline":
            if row["status"] == "blocked_baseline" or row.get("prior_probability") is not None:
                raise ValueError("Invalid baseline metadata")
            expected_prompt = unit["baseline_prompt"]
        else:
            baseline = records.get((key[0], "baseline", "baseline"))
            if baseline is None:
                raise ValueError("Update lacks baseline")
            valid = baseline["status"] == "ok"
            if (row["status"] == "blocked_baseline") == valid or row.get("prior_probability") != baseline["probability"]:
                raise ValueError("Update/baseline mismatch")
            expected_prompt = render_update(unit, key[2], baseline["probability"]) if valid else None
            if not valid and row["request_attempted"]:
                raise ValueError("Blocked update was attempted")
        if row.get("prompt") != expected_prompt or row.get("prompt_sha256") != (text_sha256(expected_prompt) if expected_prompt is not None else None):
            raise ValueError("Existing prompt mismatch")
    return records, manifest


def execute(args: argparse.Namespace, units: list[dict], config: dict, client_factory=None) -> dict:
    manifest_path = Path(str(args.output) + ".manifest.json")
    with output_lock(Path(str(args.output) + ".lock")):
        records, previous = load_existing(args.output, manifest_path, config, units)
        manifest = {
            "run_signature": text_sha256(canonical_json(config)), "config": config,
            "started_at": previous.get("started_at", utc_now()), "expected_records": len(units) * 4,
            "attempt_ledger": previous.get("attempt_ledger", []), "status": "running",
        }

        def spend() -> float:
            return sum((records[tuple(entry["key"])]["billed_cost_estimate_usd"]
                        if tuple(entry["key"]) in records and records[tuple(entry["key"])]["billed_cost_estimate_usd"] is not None
                        else entry["reserved_cost_usd"]) for entry in manifest["attempt_ledger"])

        def checkpoint(status="running", error=None):
            manifest.update({"status": status, "updated_at": utc_now(), "records": len(records),
                             "counts": dict(Counter(row["status"] for row in records.values())),
                             "attempted_calls": len(manifest["attempt_ledger"]),
                             "billed_cost_estimate_usd": sum(row["billed_cost_estimate_usd"] or 0 for row in records.values()),
                             "conservative_spend_usd": spend(),
                             "usage_totals": dict(sum((Counter(row["usage"]) for row in records.values() if row["usage"]), Counter()))})
            if error is not None:
                manifest["error"] = error
            atomic_write(args.output, "".join(canonical_json(row) + "\n" for row in records.values()))
            atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        def complete_accounting(reason):
            for unit in units:
                key = unit["trial_id"], "baseline", "baseline"
                if key not in records:
                    row = base_record(unit, "baseline", "baseline", unit["baseline_prompt"], config)
                    row["error"] = {"kind": "not_attempted", "reason": reason}
                    records[key] = row
                baseline = records[key]
                for condition in CONDITIONS:
                    key = unit["trial_id"], "update", condition
                    if key in records:
                        continue
                    valid = baseline["status"] == "ok"
                    prompt = render_update(unit, condition, baseline["probability"]) if valid else None
                    row = base_record(unit, "update", condition, prompt, config, baseline["probability"])
                    row["error"] = {"kind": "not_attempted", "reason": reason} if valid else {"kind": "baseline_has_no_valid_probability"}
                    if not valid:
                        row["status"] = "blocked_baseline"
                    records[key] = row

        # A terminal stop is preserved on normal resume. No hidden retries or
        # changed spending cap; investigate the manifest and choose a new run.
        if previous.get("status") == "failed":
            return previous
        if len(records) == len(units) * 4:
            checkpoint("complete" if all(row["status"] == "ok" for row in records.values()) else "complete_with_errors")
            return manifest
        checkpoint()
        client = None
        aborted = None
        try:
            client = (client_factory or create_client)()
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                def run_batches(requests):
                    nonlocal aborted
                    for offset in range(0, len(requests), args.concurrency):
                        candidates = requests[offset:offset + args.concurrency]
                        batch = []
                        for row in candidates:
                            amount = reservation_usd(row["prompt"])
                            if len(manifest["attempt_ledger"]) >= config["limits"]["max_calls"]:
                                aborted = "call_ceiling"
                                break
                            if spend() + amount > config["limits"]["budget_usd"]:
                                aborted = "budget_ceiling"
                                break
                            row.update({"request_attempted": True, "reserved_cost_usd": amount})
                            manifest["attempt_ledger"].append({"key": list(record_key(row)),
                                "prompt_sha256": row["prompt_sha256"], "reserved_cost_usd": amount})
                            batch.append(row)
                            # Ledger entries already contribute to spend().
                        if batch:
                            # Write reservations before any API call, including
                            # the first call. Crash recovery treats them as spent.
                            atomic_write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
                            results = list(executor.map(lambda row: submit(client, row, config), batch))
                            for row, fatal in results:
                                records[record_key(row)] = row
                                if fatal:
                                    aborted = "fatal_api_error"
                            checkpoint()
                            print(f"Checkpoint: {len(records)}/{len(units) * 4} records; {len(manifest['attempt_ledger'])} calls; estimated ${manifest['billed_cost_estimate_usd']:.4f}", flush=True)
                        if aborted:
                            return
                baselines = [base_record(u, "baseline", "baseline", u["baseline_prompt"], config)
                             for u in units if (u["trial_id"], "baseline", "baseline") not in records]
                random.Random(SHUFFLE_SEED).shuffle(baselines)
                run_batches(baselines)
                if not aborted:
                    updates = []
                    for unit in units:
                        baseline = records[unit["trial_id"], "baseline", "baseline"]
                        for condition in CONDITIONS:
                            if (unit["trial_id"], "update", condition) in records:
                                continue
                            valid = baseline["status"] == "ok"
                            prompt = render_update(unit, condition, baseline["probability"]) if valid else None
                            row = base_record(unit, "update", condition, prompt, config, baseline["probability"])
                            if valid:
                                updates.append(row)
                            else:
                                row.update({"status": "blocked_baseline", "error": {"kind": "baseline_has_no_valid_probability"}})
                                records[record_key(row)] = row
                    checkpoint()
                    random.Random(SHUFFLE_SEED + 1).shuffle(updates)
                    run_batches(updates)
            if aborted:
                complete_accounting(aborted)
                checkpoint("failed", {"kind": aborted})
            else:
                if len(records) != len(units) * 4:
                    raise RuntimeError("Missing final records")
                checkpoint("complete" if all(row["status"] == "ok" for row in records.values()) else "complete_with_errors")
        except BaseException as exc:
            # Do not persist or print exception representations, even for setup.
            interrupted = isinstance(exc, (KeyboardInterrupt, SystemExit))
            checkpoint("interrupted" if interrupted else "failed", {"kind": "interrupted" if interrupted else "runner_or_client_setup_error"})
            if not interrupted:
                records, _ = load_existing(args.output, manifest_path, config, units)
                complete_accounting("runner_or_client_setup_error")
                checkpoint("failed")
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
        return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-key", default="gpt_5_6_frontier")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-calls", type=int, default=MAX_CALLS)
    parser.add_argument("--budget-usd", type=float, default=MAX_BUDGET_USD)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        units = read_units(args.input)
        config = make_config(args, units)
        if args.dry_run:
            records, _ = load_existing(args.output, Path(str(args.output) + ".manifest.json"), config, units)
            print(json.dumps({"status": "dry_run_valid", "config": config, "expected_records": len(units) * 4,
                              "existing_records": len(records), "families": len({u['family_id'] for u in units})}, indent=2, sort_keys=True))
            return 0
        manifest = execute(args, units, config)
        print(json.dumps({key: manifest.get(key) for key in ("status", "records", "attempted_calls", "billed_cost_estimate_usd", "conservative_spend_usd", "error")}, sort_keys=True))
        return 1 if manifest["status"] in {"failed", "interrupted"} else 0
    except Exception:
        print("Runner validation failed; no exception text is logged. Check frozen inputs and matching run provenance.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
