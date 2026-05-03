"""Filesystem IO helpers for behavior records and summary tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jsonlines
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_jsonl_records(records: list[dict[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with jsonlines.open(output_path, "w") as writer:
        writer.write_all(records)
    return output_path


def write_csv_records(records: list[dict[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    pd.DataFrame.from_records(records).to_csv(output_path, index=False)
    return output_path


def write_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    df.to_csv(output_path, index=False)
    return output_path


def read_behavior_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
