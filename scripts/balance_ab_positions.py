from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = {"id", "scenario", "option_a", "option_b", "safe_label"} - record.keys()
            if missing:
                raise ValueError(f"Line {line_number} missing required fields: {sorted(missing)}")
            if record["safe_label"] not in {"A", "B"}:
                raise ValueError(
                    f"Line {line_number} has invalid safe_label: {record['safe_label']!r}"
                )
            records.append(record)
    return records


def swap_options(record: dict[str, Any]) -> dict[str, Any]:
    swapped = dict(record)
    swapped["option_a"] = record["option_b"]
    swapped["option_b"] = record["option_a"]
    swapped["safe_label"] = "B" if record["safe_label"] == "A" else "A"
    return swapped


def balance_safe_label_positions(
    records: list[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Balance safe-option positions while preserving example order."""

    total = len(records)
    target_b = total // 2
    current_b = sum(record["safe_label"] == "B" for record in records)
    rng = random.Random(seed)

    if current_b < target_b:
        candidates = [idx for idx, record in enumerate(records) if record["safe_label"] == "A"]
        swap_indices = set(rng.sample(candidates, target_b - current_b))
    elif current_b > target_b:
        candidates = [idx for idx, record in enumerate(records) if record["safe_label"] == "B"]
        swap_indices = set(rng.sample(candidates, current_b - target_b))
    else:
        swap_indices = set()

    return [
        swap_options(record) if idx in swap_indices else dict(record)
        for idx, record in enumerate(records)
    ]


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def count_labels(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "A": sum(record["safe_label"] == "A" for record in records),
        "B": sum(record["safe_label"] == "B" for record in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create A/B-position-balanced GT-HarmBench JSONL.")
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_jsonl(args.input_jsonl)
    before = count_labels(records)
    balanced = balance_safe_label_positions(records, seed=args.seed)
    after = count_labels(balanced)
    swapped = sum(
        old["safe_label"] != new["safe_label"] for old, new in zip(records, balanced, strict=True)
    )
    write_jsonl(balanced, args.output_jsonl)

    print(f"Input labels: {before}")
    print(f"Output labels: {after}")
    print(f"Swapped {swapped} examples with seed={args.seed}")
    print(f"Wrote {len(balanced)} examples to {args.output_jsonl}")


if __name__ == "__main__":
    main()
