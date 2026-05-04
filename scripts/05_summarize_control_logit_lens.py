from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from moral_mechinterp.constants import MODEL_ORDER

CONTROL_SUBSETS = {
    "random_pd_150": {
        "subset": "Random Prisoner's Dilemma",
        "dominant_game_type": "Prisoner's Dilemma",
    },
    "random_chicken_150": {
        "subset": "Random Chicken",
        "dominant_game_type": "Chicken",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize late-layer logit-lens margins for random PD/Chicken control subsets. "
            "This reads saved layer_margin_summary.csv files and does not run inference."
        )
    )
    parser.add_argument(
        "--logit-lens-dir",
        type=Path,
        default=Path("outputs/logit_lens_fixed"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/tables_full/random_control_late_layer_logit_lens.csv"),
    )
    parser.add_argument(
        "--subsets",
        default="random_pd_150,random_chicken_150",
        help="Comma-separated subset directory names under --logit-lens-dir.",
    )
    parser.add_argument("--late-layer-start", type=int, default=21)
    parser.add_argument("--late-layer-end", type=int, default=31)
    return parser.parse_args()


def parse_subset_names(subsets: str) -> list[str]:
    names = [name.strip() for name in subsets.split(",") if name.strip()]
    if not names:
        raise ValueError("At least one subset name is required.")
    return names


def read_summary(logit_lens_dir: Path, subset_name: str) -> pd.DataFrame:
    path = logit_lens_dir / subset_name / "layer_margin_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing summary CSV for {subset_name}: {path}. "
            "Run scripts/03_logit_lens_margins.py for this control subset first."
        )
    summary = pd.read_csv(path)
    required = {"model", "layer", "mean_safe_margin", "n"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return summary


def dominant_game_type(logit_lens_dir: Path, subset_name: str) -> str:
    path = logit_lens_dir / subset_name / "layer_margins.csv"
    if path.exists():
        margins = pd.read_csv(path, usecols=["id", "game_type"])
        examples = margins.drop_duplicates()
        if not examples.empty:
            return str(examples["game_type"].value_counts().idxmax())

    known = CONTROL_SUBSETS.get(subset_name)
    if known is not None:
        return str(known["dominant_game_type"])
    return "unknown"


def late_layer_model_means(
    summary: pd.DataFrame,
    *,
    late_layer_start: int,
    late_layer_end: int,
) -> dict[str, float]:
    late = summary[
        (summary["layer"] >= late_layer_start) & (summary["layer"] <= late_layer_end)
    ].copy()
    if late.empty:
        raise ValueError(
            f"No rows found for late layers {late_layer_start}-{late_layer_end}."
        )

    by_model = late.groupby("model")["mean_safe_margin"].mean()
    missing_models = [model_key for model_key in MODEL_ORDER if model_key not in by_model]
    if missing_models:
        raise ValueError(f"Missing model summaries for: {missing_models}")
    return {model_key: float(by_model[model_key]) for model_key in MODEL_ORDER}


def subset_label(subset_name: str) -> str:
    known = CONTROL_SUBSETS.get(subset_name)
    if known is not None:
        return str(known["subset"])
    return subset_name


def summarize_control_subset(
    *,
    logit_lens_dir: Path,
    subset_name: str,
    late_layer_start: int,
    late_layer_end: int,
) -> dict[str, object]:
    summary = read_summary(logit_lens_dir, subset_name)
    means = late_layer_model_means(
        summary,
        late_layer_start=late_layer_start,
        late_layer_end=late_layer_end,
    )
    model_margins = {
        "Base": means["base"],
        "UT": means["ut"],
        "GAME": means["game"],
    }
    winner = max(model_margins, key=lambda key: model_margins[key])
    n = int(pd.to_numeric(summary["n"], errors="coerce").dropna().max())

    return {
        "subset": subset_label(subset_name),
        "subset_name": subset_name,
        "n": n,
        "dominant_game_type": dominant_game_type(logit_lens_dir, subset_name),
        "base_late_margin": means["base"],
        "ut_late_margin": means["ut"],
        "game_late_margin": means["game"],
        "ut_minus_base": means["ut"] - means["base"],
        "game_minus_base": means["game"] - means["base"],
        "game_minus_ut": means["game"] - means["ut"],
        "winner": winner,
        "late_layers": f"{late_layer_start}-{late_layer_end}",
    }


def main() -> None:
    args = parse_args()
    rows = [
        summarize_control_subset(
            logit_lens_dir=args.logit_lens_dir,
            subset_name=subset_name,
            late_layer_start=args.late_layer_start,
            late_layer_end=args.late_layer_end,
        )
        for subset_name in parse_subset_names(args.subsets)
    ]
    table = pd.DataFrame(rows)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_path, index=False)
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(table.replace({np.nan: None}).to_string(index=False))
    print(f"\nWrote random control late-layer table: {args.output_path}")


if __name__ == "__main__":
    main()
