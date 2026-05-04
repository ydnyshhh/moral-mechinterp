from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


SUBSET_SPECS = {
    "top_ut_margin_shift": {
        "subset": "UT-favored margin shifts",
        "dominant_game_type": "Prisoner's Dilemma",
    },
    "top_game_margin_shift": {
        "subset": "GAME-favored margin shifts",
        "dominant_game_type": "Chicken",
    },
    "ut_safe_game_harmful": {
        "subset": "UT-safe / GAME-harmful",
        "dominant_game_type": "Prisoner's Dilemma",
    },
    "game_safe_ut_harmful": {
        "subset": "GAME-safe / UT-harmful",
        "dominant_game_type": "Chicken",
    },
    "random_pd_150": {
        "subset": "Random Prisoner's Dilemma",
        "dominant_game_type": "Prisoner's Dilemma",
    },
    "random_chicken_150": {
        "subset": "Random Chicken",
        "dominant_game_type": "Chicken",
    },
}
PAIR_TO_COLUMN = {
    "Base–UT": "base_ut_late_drift",
    "Base–GAME": "base_game_late_drift",
    "UT–GAME": "ut_game_late_drift",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize late-layer representation drift across selected and random "
            "GT-HarmBench subsets. This reads saved cosine_drift_summary.csv files "
            "and does not run model inference."
        )
    )
    parser.add_argument("--base-dir", type=Path, default=Path("outputs/representation_drift"))
    parser.add_argument(
        "--subset-dir",
        type=Path,
        default=Path("outputs/behavior_full/subsets"),
        help="Optional source of subset CSVs for dominant game-type inference.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/tables_full/late_layer_representation_drift.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("outputs/tables_full/late_layer_representation_drift.md"),
    )
    parser.add_argument(
        "--subsets",
        default=",".join(SUBSET_SPECS),
        help="Comma-separated subset directory names under --base-dir.",
    )
    parser.add_argument("--late-start", type=int, default=21)
    parser.add_argument("--late-end", type=int, default=31)
    return parser.parse_args()


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def load_pandas():
    import pandas as pd

    return pd


def parse_subset_names(raw: str) -> list[str]:
    names = [name.strip() for name in raw.split(",") if name.strip()]
    if not names:
        raise ValueError("At least one subset name is required.")
    return names


def read_summary(base_dir: Path, subset_name: str) -> pd.DataFrame:
    pd = load_pandas()
    path = base_dir / subset_name / "cosine_drift_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing representation drift summary for {subset_name}: {path}. "
            "Run scripts/06_representation_drift.py for this subset first."
        )
    summary = pd.read_csv(path)
    required = {"pair", "layer", "mean_cosine_drift", "n"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return summary


def subset_label(subset_name: str) -> str:
    known = SUBSET_SPECS.get(subset_name)
    if known is not None:
        return str(known["subset"])
    return subset_name


def infer_dominant_game_type(subset_dir: Path, subset_name: str) -> str:
    pd = load_pandas()
    subset_csv = subset_dir / f"{subset_name}.csv"
    if subset_csv.exists():
        subset = pd.read_csv(subset_csv, usecols=["game_type"])
        if not subset.empty:
            return str(subset["game_type"].value_counts().idxmax())
    known = SUBSET_SPECS.get(subset_name)
    if known is not None:
        return str(known["dominant_game_type"])
    return "unknown"


def late_pair_means(
    summary: pd.DataFrame,
    *,
    late_start: int,
    late_end: int,
) -> dict[str, float]:
    late = summary[(summary["layer"] >= late_start) & (summary["layer"] <= late_end)]
    if late.empty:
        raise ValueError(f"No rows found for late layers {late_start}-{late_end}.")
    means = late.groupby("pair")["mean_cosine_drift"].mean()
    missing = [pair for pair in PAIR_TO_COLUMN if pair not in means]
    if missing:
        raise ValueError(f"Missing pair summaries for: {missing}")
    return {pair: float(means[pair]) for pair in PAIR_TO_COLUMN}


def dominant_base_adapter_drift(base_ut: float, base_game: float) -> str:
    if abs(base_ut - base_game) <= 1e-12:
        return "tie"
    return "Base–UT" if base_ut > base_game else "Base–GAME"


def summarize_subset(
    *,
    base_dir: Path,
    subset_dir: Path,
    subset_name: str,
    late_start: int,
    late_end: int,
) -> dict[str, object]:
    pd = load_pandas()
    summary = read_summary(base_dir, subset_name)
    means = late_pair_means(summary, late_start=late_start, late_end=late_end)
    n = int(pd.to_numeric(summary["n"], errors="coerce").dropna().max())
    base_ut = means["Base–UT"]
    base_game = means["Base–GAME"]
    ut_game = means["UT–GAME"]
    return {
        "subset": subset_label(subset_name),
        "subset_name": subset_name,
        "n": n,
        "dominant_game_type": infer_dominant_game_type(subset_dir, subset_name),
        "base_ut_late_drift": base_ut,
        "base_game_late_drift": base_game,
        "ut_game_late_drift": ut_game,
        "dominant_base_adapter_drift": dominant_base_adapter_drift(base_ut, base_game),
        "late_layers": f"{late_start}-{late_end}",
    }


def format_markdown_table(table: pd.DataFrame, *, late_start: int, late_end: int) -> str:
    layers = f"{late_start}–{late_end}"
    lines = [
        (
            "| Subset | n | Base–UT late drift | Base–GAME late drift | "
            "UT–GAME late drift | Dominant drift | Layers |"
        ),
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            "| "
            f"{row.subset} | "
            f"{row.n} | "
            f"{row.base_ut_late_drift:.5f} | "
            f"{row.base_game_late_drift:.5f} | "
            f"{row.ut_game_late_drift:.5f} | "
            f"{row.dominant_base_adapter_drift} | "
            f"{layers} |"
        )
    return "\n".join(lines) + "\n"


def print_full_table(table: pd.DataFrame) -> None:
    display = table.copy()
    for column in (
        "base_ut_late_drift",
        "base_game_late_drift",
        "ut_game_late_drift",
    ):
        display[column] = display[column].map(lambda value: f"{value:.5f}")
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
            subset_dir=args.subset_dir,
            subset_name=subset_name,
            late_start=args.late_start,
            late_end=args.late_end,
        )
        for subset_name in parse_subset_names(args.subsets)
    ]
    table = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)

    markdown = format_markdown_table(table, late_start=args.late_start, late_end=args.late_end)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")

    print("Full late-layer representation drift table:")
    print_full_table(table)
    print("\nCompact Markdown table:")
    print(markdown, end="")
    print(f"\nWrote CSV: {args.output_csv}")
    print(f"Wrote Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
