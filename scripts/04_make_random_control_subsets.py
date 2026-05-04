from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from moral_mechinterp.constants import MODEL_ORDER

CONTROL_SPECS = {
    "random_pd": {
        "game_type": "Prisoner's Dilemma",
        "filename_template": "random_pd_{n}.csv",
    },
    "random_chicken": {
        "game_type": "Chicken",
        "filename_template": "random_chicken_{n}.csv",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create or validate random game-type control subsets from the full behavior CSV. "
            "Existing subset files are reused and checked rather than overwritten."
        )
    )
    parser.add_argument(
        "--behavior-csv",
        type=Path,
        default=Path("outputs/behavior_full/model_choices.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/behavior_full/subsets"),
    )
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_behavior_csv(behavior: pd.DataFrame) -> None:
    required = {"game_type"}
    missing = sorted(required - set(behavior.columns))
    if missing:
        raise ValueError(f"Behavior CSV is missing required columns: {missing}")


def subset_filename(spec: dict[str, str], n: int) -> str:
    return spec["filename_template"].format(n=n)


def create_or_read_subset(
    *,
    behavior: pd.DataFrame,
    output_dir: Path,
    spec: dict[str, str],
    n: int,
    seed: int,
) -> tuple[Path, pd.DataFrame, bool]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / subset_filename(spec, n)
    game_type = spec["game_type"]

    if path.exists():
        subset = pd.read_csv(path)
        validate_subset(subset, path=path, game_type=game_type, n=n)
        return path, subset, False

    game_df = behavior[behavior["game_type"] == game_type].copy()
    if len(game_df) < n:
        raise ValueError(f"Only {len(game_df)} rows available for {game_type}; need {n}.")

    subset = game_df.sample(n=n, random_state=seed).sort_index()
    subset.to_csv(path, index=False)
    validate_subset(subset, path=path, game_type=game_type, n=n)
    return path, subset, True


def validate_subset(subset: pd.DataFrame, *, path: Path, game_type: str, n: int) -> None:
    if len(subset) != n:
        raise ValueError(f"{path} has {len(subset)} rows; expected {n}.")
    if "game_type" not in subset.columns:
        raise ValueError(f"{path} is missing required column: game_type")
    bad_types = sorted(set(subset["game_type"].dropna()) - {game_type})
    if bad_types:
        raise ValueError(f"{path} contains unexpected game_type values: {bad_types}")


def boolean_safe_rate(series: pd.Series) -> float:
    if series.dtype == bool:
        return float(series.mean())
    normalized = series.astype(str).str.lower().str.strip()
    return float(normalized.isin({"true", "1", "yes"}).mean())


def summarize_subset(subset_name: str, subset: pd.DataFrame, *, path: Path, created: bool) -> None:
    status = "created" if created else "exists"
    print(f"\n{subset_name} ({status}): {path}")
    print(f"n: {len(subset)}")
    print("game_type counts:")
    for game_type, count in subset["game_type"].value_counts().items():
        print(f"  {game_type}: {count}")

    print("safe rates:")
    for model_key in MODEL_ORDER:
        column = f"{model_key}_safe"
        if column in subset.columns:
            print(f"  {model_key}: {boolean_safe_rate(subset[column]):.3f}")
        else:
            print(f"  {model_key}: missing {column}")

    print("mean safe margins:")
    for model_key in MODEL_ORDER:
        column = f"{model_key}_safe_margin"
        if column in subset.columns:
            margin = pd.to_numeric(subset[column], errors="coerce").mean()
            print(f"  {model_key}: {margin:.3f}")
        else:
            print(f"  {model_key}: missing {column}")


def main() -> None:
    args = parse_args()
    if not args.behavior_csv.exists():
        raise FileNotFoundError(f"Missing behavior CSV: {args.behavior_csv}")

    behavior = pd.read_csv(args.behavior_csv)
    validate_behavior_csv(behavior)

    for spec in CONTROL_SPECS.values():
        path, subset, created = create_or_read_subset(
            behavior=behavior,
            output_dir=args.output_dir,
            spec=spec,
            n=args.n,
            seed=args.seed,
        )
        summarize_subset(path.stem, subset, path=path, created=created)


if __name__ == "__main__":
    main()
