"""Summary metrics for behavior files."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from moral_mechinterp.constants import MODEL_LABELS, MODEL_ORDER


def coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes"}
        if not pd.isna(value)
        else np.nan
    )


def available_model_prefixes(df: pd.DataFrame) -> list[str]:
    return [prefix for prefix in MODEL_ORDER if f"{prefix}_safe" in df.columns]


def overall_metrics(df: pd.DataFrame, prefixes: list[str] | None = None) -> pd.DataFrame:
    prefixes = prefixes or available_model_prefixes(df)
    rows: list[dict[str, Any]] = []
    for prefix in prefixes:
        safe_col = f"{prefix}_safe"
        margin_col = f"{prefix}_safe_margin"
        safe = coerce_bool_series(df[safe_col]).astype(float)
        margin = pd.to_numeric(df[margin_col], errors="coerce")
        rows.append(
            {
                "model": prefix,
                "model_label": MODEL_LABELS.get(prefix, prefix),
                "safe_rate": safe.mean(),
                "mean_safe_margin": margin.mean(),
                "median_safe_margin": margin.median(),
                "n_examples": int(safe.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def game_type_metrics(df: pd.DataFrame, prefixes: list[str] | None = None) -> pd.DataFrame:
    prefixes = prefixes or available_model_prefixes(df)
    rows: list[dict[str, Any]] = []
    for game_type, group in df.groupby("game_type", dropna=False):
        for prefix in prefixes:
            safe = coerce_bool_series(group[f"{prefix}_safe"]).astype(float)
            margin = pd.to_numeric(group[f"{prefix}_safe_margin"], errors="coerce")
            rows.append(
                {
                    "game_type": game_type,
                    "model": prefix,
                    "model_label": MODEL_LABELS.get(prefix, prefix),
                    "safe_rate": safe.mean(),
                    "mean_safe_margin": margin.mean(),
                    "median_safe_margin": margin.median(),
                    "n_examples": int(safe.notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def disagreement_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df["disagreement_type"]
        .fillna("missing")
        .value_counts(dropna=False)
        .rename_axis("disagreement_type")
        .reset_index(name="count")
    )
    total = counts["count"].sum()
    counts["percent"] = counts["count"] / total if total else np.nan
    return counts


def strong_flip_counts(df: pd.DataFrame) -> pd.DataFrame:
    indicators = [
        "strong_ut_flip",
        "strong_game_flip",
        "strong_ut_regression",
        "strong_game_regression",
    ]
    rows: list[dict[str, Any]] = []
    n = len(df)
    for indicator in indicators:
        if indicator not in df.columns:
            count = 0
        else:
            count = int(coerce_bool_series(df[indicator]).sum())
        rows.append(
            {
                "indicator": indicator,
                "count": count,
                "percent": count / n if n else np.nan,
                "n_examples": n,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(
    values: np.ndarray | pd.Series | list[float],
    *,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean."""

    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, arr.size, size=(n_bootstrap, arr.size))
    means = arr[sample_indices].mean(axis=1)
    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1 - alpha / 2))
    return low, high


def paired_improvements(
    df: pd.DataFrame,
    *,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base = coerce_bool_series(df["base_safe"]).astype(float)
    for prefix in ("ut", "game"):
        safe_col = f"{prefix}_safe"
        if safe_col not in df.columns:
            continue
        trained = coerce_bool_series(df[safe_col]).astype(float)
        diff = trained - base
        low, high = bootstrap_ci(
            diff,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            seed=seed,
        )
        rows.append(
            {
                "comparison": f"{prefix}_minus_base_safe_rate",
                "model": prefix,
                "model_label": MODEL_LABELS.get(prefix, prefix),
                "mean_paired_improvement": diff.mean(),
                "ci_low": low,
                "ci_high": high,
                "n_examples": int(diff.notna().sum()),
            }
        )
    return pd.DataFrame(rows)
