from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from pathlib import Path


def normalize_label_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def parse_actions(value: str) -> list[str]:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list) or len(parsed) != 2:
        raise ValueError(f"Expected exactly two actions, got {value!r}")
    return [str(item).strip() for item in parsed]


def parse_target_pairs(value: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for part in str(value or "").split("|"):
        part = part.strip()
        if not part:
            continue
        parsed = ast.literal_eval(part)
        if not isinstance(parsed, tuple) or len(parsed) != 2:
            raise ValueError(f"Expected target pair, got {part!r}")
        pairs.append((normalize_label_text(parsed[0]), normalize_label_text(parsed[1])))
    return pairs


def infer_safe_label(
    *,
    actions: list[str],
    target_pairs: list[tuple[str, str]],
    role_index: int,
) -> str | None:
    target_actions = {pair[role_index] for pair in target_pairs}
    matches = [
        action_index
        for action_index, action in enumerate(actions)
        if normalize_label_text(action) in target_actions
    ]
    if len(matches) != 1:
        return None
    return "A" if matches[0] == 0 else "B"


def convert_csv(
    input_path: Path,
    output_path: Path,
    *,
    target_column: str,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    with (
        input_path.open("r", encoding="utf-8-sig", newline="") as input_file,
        output_path.open("w", encoding="utf-8", newline="\n") as output_file,
    ):
        reader = csv.DictReader(input_file)
        for row_index, row in enumerate(reader):
            row_actions = parse_actions(row["actions_row"])
            column_actions = parse_actions(row["actions_column"])
            target_pairs = parse_target_pairs(row[target_column])
            game_type = row.get("formal_game") or "unknown"
            source_id = row.get("id") or str(row_index)

            role_specs = (
                ("row", "story_row", row_actions, 0),
                ("col", "story_col", column_actions, 1),
            )
            for role_name, story_column, actions, role_index in role_specs:
                safe_label = infer_safe_label(
                    actions=actions,
                    target_pairs=target_pairs,
                    role_index=role_index,
                )
                if safe_label is None:
                    skipped += 1
                    continue

                record = {
                    "id": f"{source_id}_{role_name}",
                    "game_type": str(game_type).strip(),
                    "scenario": str(row[story_column]).strip(),
                    "option_a": actions[0],
                    "option_b": actions[1],
                    "safe_label": safe_label,
                }
                output_file.write(json.dumps(record, ensure_ascii=True) + "\n")
                written += 1

    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GT-HarmBench CSV to clean JSONL.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument(
        "--target-column",
        default="target_nash_social_welfare",
        help="Target column used to infer safe labels.",
    )
    args = parser.parse_args()

    written, skipped = convert_csv(
        args.input_csv,
        args.output_jsonl,
        target_column=args.target_column,
    )
    print(
        f"Wrote {written} examples to {args.output_jsonl}; "
        f"skipped {skipped} ambiguous role examples."
    )


if __name__ == "__main__":
    main()
