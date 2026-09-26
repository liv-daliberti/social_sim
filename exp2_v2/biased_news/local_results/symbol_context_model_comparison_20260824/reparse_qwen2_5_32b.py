#!/usr/bin/env python3
"""Create a provenance-preserving arithmetic-only reparse of an open model."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import operator
import re
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "qwen2_5_32b"
DESTINATION = HERE / "qwen2_5_32b_reparsed"
MODEL = "Qwen2.5-32B-Instruct"
ARMS = ("abc_no_context", "abc_context", "abc_symbol_context")
EXPECTED_ROWS = 1_250
EXPRESSION = re.compile(r'"predicted_poll"\s*:\s*(.+?)\s*}\s*$')
BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, bool):
            raise ValueError("booleans are not numeric predictions")
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in BINOPS:
        return float(BINOPS[type(node.op)](evaluate(node.left), evaluate(node.right)))
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARYOPS:
        return float(UNARYOPS[type(node.op)](evaluate(node.operand)))
    raise ValueError(f"disallowed arithmetic syntax: {ast.dump(node)}")


def parse_explicit_arithmetic(raw: str) -> tuple[str, float]:
    match = EXPRESSION.search(raw.strip())
    if not match:
        raise ValueError("no terminal predicted_poll arithmetic expression")
    expression = match.group(1).strip()
    tree = ast.parse(expression, mode="eval")
    value = evaluate(tree)
    if not math.isfinite(value):
        raise ValueError("arithmetic result is not finite")
    if not 0.0 <= value <= 100.0:
        raise ValueError("arithmetic result is outside the 0--100 forecast range")
    return expression, value


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    model = args.model

    if destination.exists():
        raise SystemExit(f"Refusing to overwrite existing {destination}")

    recovered: list[dict] = []
    unresolved: list[dict] = []
    source_hashes: dict[str, str] = {}
    destination_hashes: dict[str, str] = {}
    counts: dict[str, dict[str, int]] = {}

    with tempfile.TemporaryDirectory(prefix="open_model_reparsed_", dir=HERE) as temporary:
        staging = Path(temporary)
        for arm in ARMS:
            filename = f"responses_{model}_{arm}.jsonl"
            source_path = source / filename
            rows = [
                json.loads(line)
                for line in source_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(rows) != EXPECTED_ROWS:
                raise ValueError(f"{arm}: found {len(rows)} rows, expected {EXPECTED_ROWS}")
            if len({row["task_id"] for row in rows}) != EXPECTED_ROWS:
                raise ValueError(f"{arm}: task IDs are not unique")

            source_hashes[arm] = sha256(source_path)
            for row in rows:
                if row.get("predicted_poll") is not None:
                    continue
                try:
                    expression, value = parse_explicit_arithmetic(row.get("raw", ""))
                except (ArithmeticError, SyntaxError, ValueError) as error:
                    unresolved.append(
                        {"arm": arm, "task_id": row["task_id"], "reason": str(error)}
                    )
                    continue
                row["predicted_poll"] = value
                row["error"] = None
                row["reparse_provenance"] = {
                    "expression": expression,
                    "method": "strict arithmetic-only AST whitelist",
                    "new_model_call": False,
                    "original_predicted_poll": None,
                }
                recovered.append(
                    {
                        "arm": arm,
                        "expression": expression,
                        "task_id": row["task_id"],
                        "value": value,
                    }
                )

            destination_path = staging / filename
            destination_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            destination_hashes[arm] = sha256(destination_path)
            counts[arm] = {
                "responses_received": len(rows),
                "successfully_parsed": sum(
                    row.get("predicted_poll") is not None for row in rows
                ),
            }

        manifest_name = f"responses_{model}.manifest.json"
        manifest = json.loads((source / manifest_name).read_text(encoding="utf-8"))
        manifest["counts"] = counts
        manifest["reparse"] = {
            "allowed_syntax": "numeric constants, parentheses, unary +/-, +, -, *, /",
            "destination_response_sha256": destination_hashes,
            "new_model_calls": 0,
            "original_outputs_preserved": True,
            "recovered": recovered,
            "source_dir": str(source),
            "source_response_sha256": source_hashes,
            "unresolved": unresolved,
        }
        (staging / manifest_name).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.rename(destination)

    print(json.dumps({"counts": counts, "recovered": recovered, "unresolved": unresolved}, indent=2))


if __name__ == "__main__":
    main()
