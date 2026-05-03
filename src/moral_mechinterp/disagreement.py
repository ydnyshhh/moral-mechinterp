"""Disagreement-set assignment and strong flip/regression indicators."""

from __future__ import annotations

from typing import Any


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def assign_disagreement_type(base_safe: Any, ut_safe: Any, game_safe: Any) -> str:
    base = _as_bool(base_safe)
    ut = _as_bool(ut_safe)
    game = _as_bool(game_safe)
    if base is None or ut is None or game is None:
        return "incomplete"

    if base and ut and game:
        return "all_safe"
    if not base and not ut and not game:
        return "all_harmful"
    if not base and ut and game:
        return "base_harmful_ut_safe_game_safe"
    if not base and ut and not game:
        return "base_harmful_ut_safe_game_harmful"
    if not base and not ut and game:
        return "base_harmful_ut_harmful_game_safe"
    if base and not ut and game:
        return "base_safe_ut_harmful_game_safe"
    if base and ut and not game:
        return "base_safe_ut_safe_game_harmful"
    if base and not ut and not game:
        return "base_safe_trained_harmful"
    raise AssertionError("unreachable disagreement state")


def add_margin_differences(row: dict[str, Any]) -> None:
    if "ut_safe_margin" in row and "base_safe_margin" in row:
        row["ut_minus_base_margin"] = row["ut_safe_margin"] - row["base_safe_margin"]
    if "game_safe_margin" in row and "base_safe_margin" in row:
        row["game_minus_base_margin"] = row["game_safe_margin"] - row["base_safe_margin"]
    if "game_safe_margin" in row and "ut_safe_margin" in row:
        row["game_minus_ut_margin"] = row["game_safe_margin"] - row["ut_safe_margin"]


def add_disagreement_fields(row: dict[str, Any], *, tau: float) -> None:
    row["disagreement_type"] = assign_disagreement_type(
        row.get("base_safe"),
        row.get("ut_safe"),
        row.get("game_safe"),
    )

    base_margin = row.get("base_safe_margin")
    ut_margin = row.get("ut_safe_margin")
    game_margin = row.get("game_safe_margin")

    row["strong_ut_flip"] = bool(
        base_margin is not None and ut_margin is not None and base_margin < -tau and ut_margin > tau
    )
    row["strong_game_flip"] = bool(
        base_margin is not None
        and game_margin is not None
        and base_margin < -tau
        and game_margin > tau
    )
    row["strong_ut_regression"] = bool(
        base_margin is not None and ut_margin is not None and base_margin > tau and ut_margin < -tau
    )
    row["strong_game_regression"] = bool(
        base_margin is not None
        and game_margin is not None
        and base_margin > tau
        and game_margin < -tau
    )


def finalize_behavior_rows(rows: list[dict[str, Any]], *, tau: float) -> list[dict[str, Any]]:
    for row in rows:
        add_margin_differences(row)
        add_disagreement_fields(row, tau=tau)
    return rows
