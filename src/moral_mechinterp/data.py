"""JSONL loading and normalization for GT-HarmBench-style A/B examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any

import jsonlines


@dataclass(frozen=True)
class NormalizedExample:
    id: str
    game_type: str
    scenario: str
    option_a: str
    option_b: str
    safe_label: str

    @property
    def harmful_label(self) -> str:
        return "B" if self.safe_label == "A" else "A"

    def to_record(self) -> dict[str, str]:
        return {
            "id": self.id,
            "game_type": self.game_type,
            "scenario": self.scenario,
            "option_a": self.option_a,
            "option_b": self.option_b,
            "safe_label": self.safe_label,
            "harmful_label": self.harmful_label,
        }


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _option_from_nested(raw: dict[str, Any], label: str) -> Any:
    options = raw.get("options")
    if isinstance(options, dict):
        return options.get(label) or options.get(label.lower())
    if isinstance(options, list) and len(options) >= 2:
        idx = 0 if label == "A" else 1
        value = options[idx]
        if isinstance(value, dict):
            return (
                value.get("text")
                or value.get("option")
                or value.get("action")
                or value.get("content")
            )
        return value
    return None


def _normalize_label(value: Any, *, field_name: str) -> str:
    if value is None:
        raise ValueError(f"Missing required label field: {field_name}")
    text = str(value).strip().upper()
    if text.startswith("ANSWER:"):
        text = text.replace("ANSWER:", "", 1).strip()
    if text not in {"A", "B"}:
        raise ValueError(f"{field_name} must be A or B, got {value!r}")
    return text


def _normalize_text(value: Any, *, field_name: str) -> str:
    if value is None:
        raise ValueError(f"Missing required text field: {field_name}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Field {field_name} is empty")
    return text


def normalize_example(raw: dict[str, Any], index: int) -> NormalizedExample:
    """Normalize one benchmark row while tolerating small schema variations."""

    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    example_id = _first_present(raw, ("id", "example_id", "uid", "name")) or str(index)
    game_type = (
        _first_present(raw, ("game_type", "game", "task_type"))
        or metadata.get("game_type")
        or metadata.get("game")
        or "unknown"
    )
    scenario = _first_present(raw, ("scenario", "prompt", "context", "question"))
    option_a = (
        _first_present(raw, ("option_a", "action_a", "answer_a", "choice_a", "A"))
        or _option_from_nested(raw, "A")
    )
    option_b = (
        _first_present(raw, ("option_b", "action_b", "answer_b", "choice_b", "B"))
        or _option_from_nested(raw, "B")
    )
    safe_label = _first_present(raw, ("safe_label", "safe_answer", "safe_option"))

    return NormalizedExample(
        id=str(example_id),
        game_type=str(game_type),
        scenario=_normalize_text(scenario, field_name="scenario"),
        option_a=_normalize_text(option_a, field_name="option_a"),
        option_b=_normalize_text(option_b, field_name="option_b"),
        safe_label=_normalize_label(safe_label, field_name="safe_label"),
    )


def load_jsonl_examples(
    path: str | Path,
    *,
    max_examples: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> list[NormalizedExample]:
    """Load and normalize a JSONL benchmark file."""

    examples: list[NormalizedExample] = []
    with jsonlines.open(Path(path), "r") as reader:
        for idx, raw in enumerate(reader):
            if not isinstance(raw, dict):
                raise ValueError(f"Line {idx + 1} must be a JSON object")
            examples.append(normalize_example(raw, idx))

    if shuffle:
        Random(seed).shuffle(examples)

    if max_examples is not None:
        examples = examples[: int(max_examples)]

    return examples
