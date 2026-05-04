from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from moral_mechinterp.constants import MODEL_ORDER

if TYPE_CHECKING:
    import pandas as pd


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
        "--base-dir",
        type=Path,
        default=Path("outputs/logit_lens_fixed"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/tables_full/random_control_late_layer_logit_lens.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("outputs/tables_full/random_control_late_layer_logit_lens.md"),
    )
    parser.add_argument("--late-start", type=int, default=21)
    parser.add_argument("--late-end", type=int, default=31)
    return parser.parse_args()


def load_pandas():
    import pandas as pd

    return pd


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def read_summary(base_dir: Path, subset_name: str) -> pd.DataFrame:
    pd = load_pandas()
    path = base_dir / subset_name / "layer_margin_summary.csv"
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


def dominant_game_type(base_dir: Path, subset_name: str) -> str:
    pd = load_pandas()
    path = base_dir / subset_name / "layer_margins.csv"
    if path.exists():
        margins = pd.read_csv(path, usecols=["id", "game_type"])
        examples = margins.drop_duplicates()
        if not examples.empty:
            return str(examples["game_type"].value_counts().idxmax())

    known = CONTROL_SUBSETS[subset_name]
    return str(known["dominant_game_type"])


def late_layer_model_means(
    summary: pd.DataFrame,
    *,
    late_start: int,
    late_end: int,
) -> dict[str, float]:
    late = summary[(summary["layer"] >= late_start) & (summary["layer"] <= late_end)].copy()
    if late.empty:
        raise ValueError(f"No rows found for late layers {late_start}-{late_end}.")

    by_model = late.groupby("model")["mean_safe_margin"].mean()
    missing_models = [model_key for model_key in MODEL_ORDER if model_key not in by_model]
    if missing_models:
        raise ValueError(f"Missing model summaries for: {missing_models}")
    return {model_key: float(by_model[model_key]) for model_key in MODEL_ORDER}


def summarize_subset(
    *,
    base_dir: Path,
    subset_name: str,
    late_start: int,
    late_end: int,
) -> dict[str, object]:
    pd = load_pandas()
    summary = read_summary(base_dir, subset_name)
    means = late_layer_model_means(summary, late_start=late_start, late_end=late_end)
    model_margins = {
        "Base": means["base"],
        "UT": means["ut"],
        "GAME": means["game"],
    }
    winner = max(model_margins, key=lambda key: model_margins[key])
    n = int(pd.to_numeric(summary["n"], errors="coerce").dropna().max())

    return {
        "subset": str(CONTROL_SUBSETS[subset_name]["subset"]),
        "subset_name": subset_name,
        "n": n,
        "dominant_game_type": dominant_game_type(base_dir, subset_name),
        "base_late_margin": means["base"],
        "ut_late_margin": means["ut"],
        "game_late_margin": means["game"],
        "ut_minus_base": means["ut"] - means["base"],
        "game_minus_base": means["game"] - means["base"],
        "game_minus_ut": means["game"] - means["ut"],
        "winner": winner,
        "late_layers": f"{late_start}-{late_end}",
    }


def format_delta(value: float) -> str:
    return f"{value:+.3f}"


def format_markdown_table(table: pd.DataFrame, *, late_start: int, late_end: int) -> str:
    layers = f"{late_start}–{late_end}"
    lines = [
        "| Subset | n | UT−Base late | GAME−Base late | Winner | Layers |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            "| "
            f"{row.subset} | "
            f"{row.n} | "
            f"{format_delta(row.ut_minus_base)} | "
            f"{format_delta(row.game_minus_base)} | "
            f"{row.winner} | "
            f"{layers} |"
        )
    return "\n".join(lines) + "\n"


def print_full_table(table: pd.DataFrame) -> None:
    display = table.copy()
    numeric_columns = [
        "base_late_margin",
        "ut_late_margin",
        "game_late_margin",
        "ut_minus_base",
        "game_minus_base",
        "game_minus_ut",
    ]
    for column in numeric_columns:
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    print(display.to_string(index=False))


def main() -> None:
    configure_stdout()
    args = parse_args()
    if args.late_end < args.late_start:
        raise ValueError("--late-end must be greater than or equal to --late-start.")

    pd = load_pandas()
    rows = [
        summarize_subset(
            base_dir=args.base_dir,
            subset_name=subset_name,
            late_start=args.late_start,
            late_end=args.late_end,
        )
        for subset_name in CONTROL_SUBSETS
    ]
    table = pd.DataFrame(rows)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)

    markdown = format_markdown_table(table, late_start=args.late_start, late_end=args.late_end)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")

    print("Full random-control late-layer table:")
    print_full_table(table)
    print("\nCompact Markdown table:")
    print(markdown, end="")
    print(f"\nWrote CSV: {args.output_csv}")
    print(f"Wrote Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
