from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from moral_mechinterp.constants import MODEL_COLORS, MODEL_MARKERS

if TYPE_CHECKING:
    import pandas as pd


SUBSET_SPECS = {
    "top_ut_margin_shift": {
        "label": "UT-favored margin shifts",
        "dominant_game_type": "Prisoner's Dilemma",
    },
    "top_game_margin_shift": {
        "label": "GAME-favored margin shifts",
        "dominant_game_type": "Chicken",
    },
    "ut_safe_game_harmful": {
        "label": "UT-safe / GAME-harmful",
        "dominant_game_type": "Prisoner's Dilemma",
    },
    "game_safe_ut_harmful": {
        "label": "GAME-safe / UT-harmful",
        "dominant_game_type": "Chicken",
    },
    "random_pd_150": {
        "label": "Random Prisoner's Dilemma",
        "dominant_game_type": "Prisoner's Dilemma",
    },
    "random_chicken_150": {
        "label": "Random Chicken",
        "dominant_game_type": "Chicken",
    },
}
PANEL_LETTERS = ("A", "B", "C", "D", "E", "F")
REQUIRED_MODELS = ("base", "ut", "game")
DELTA_COLUMNS = ("ut_minus_base", "game_minus_base")
DELTA_LABELS = {
    "ut_minus_base": "UT−Base",
    "game_minus_base": "GAME−Base",
}
DELTA_COLORS = {
    "ut_minus_base": MODEL_COLORS["ut"],
    "game_minus_base": MODEL_COLORS["game"],
}
DELTA_MARKERS = {
    "ut_minus_base": MODEL_MARKERS["ut"],
    "game_minus_base": MODEL_MARKERS["game"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot adapter-delta logit-lens curves from saved layer_margin_summary.csv "
            "files. This subtracts Base from UT/GAME layerwise safe-margin trajectories "
            "and does not run model inference."
        )
    )
    parser.add_argument("--base-dir", type=Path, default=Path("outputs/logit_lens_fixed"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/adapter_delta_logit_lens"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("outputs/figures_adapter_delta"),
    )
    parser.add_argument("--table-dir", type=Path, default=Path("outputs/tables_full"))
    parser.add_argument("--subsets", default=",".join(SUBSET_SPECS))
    parser.add_argument("--late-start", type=int, default=21)
    parser.add_argument("--late-end", type=int, default=31)
    parser.add_argument("--no-individual-plots", action="store_true")
    parser.add_argument("--no-combined-plot", action="store_true")
    return parser.parse_args()


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def load_pandas():
    import pandas as pd

    return pd


def parse_subset_names(raw: str) -> list[str]:
    subsets = [subset.strip() for subset in raw.split(",") if subset.strip()]
    if not subsets:
        raise ValueError("At least one subset is required.")
    return subsets


def subset_label(subset_name: str) -> str:
    spec = SUBSET_SPECS.get(subset_name)
    if spec is None:
        return subset_name
    return str(spec["label"])


def dominant_game_type(subset_name: str) -> str:
    spec = SUBSET_SPECS.get(subset_name)
    if spec is None:
        return "unknown"
    return str(spec["dominant_game_type"])


def read_layer_summary(base_dir: Path, subset_name: str) -> pd.DataFrame:
    pd = load_pandas()
    path = base_dir / subset_name / "layer_margin_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing layer margin summary for {subset_name}: {path}")
    summary = pd.read_csv(path)
    required = {"model", "layer", "mean_safe_margin", "n"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    models = set(summary["model"].astype(str))
    missing_models = [model for model in REQUIRED_MODELS if model not in models]
    if missing_models:
        raise ValueError(f"{path} is missing required model rows: {missing_models}")
    return summary


def compute_adapter_delta(summary: pd.DataFrame, *, subset_name: str) -> pd.DataFrame:
    pd = load_pandas()
    means = summary.pivot_table(
        index="layer",
        columns="model",
        values="mean_safe_margin",
        aggfunc="mean",
    )
    missing_models = [model for model in REQUIRED_MODELS if model not in means.columns]
    if missing_models:
        raise ValueError(f"{subset_name} is missing model columns after pivot: {missing_models}")

    n_by_layer = (
        summary.groupby("layer", as_index=True)["n"]
        .max()
        .rename("n")
        .astype(int)
    )
    delta = pd.DataFrame(
        {
            "subset_name": subset_name,
            "layer": means.index.astype(int),
            "n": n_by_layer.reindex(means.index).to_numpy(),
            "base_mean_safe_margin": means["base"].to_numpy(dtype=float),
            "ut_mean_safe_margin": means["ut"].to_numpy(dtype=float),
            "game_mean_safe_margin": means["game"].to_numpy(dtype=float),
        }
    )
    delta["ut_minus_base"] = (
        delta["ut_mean_safe_margin"] - delta["base_mean_safe_margin"]
    )
    delta["game_minus_base"] = (
        delta["game_mean_safe_margin"] - delta["base_mean_safe_margin"]
    )
    delta["game_minus_ut"] = (
        delta["game_mean_safe_margin"] - delta["ut_mean_safe_margin"]
    )
    return delta.sort_values("layer").reset_index(drop=True)


def write_delta_csv(delta: pd.DataFrame, *, output_dir: Path, subset_name: str) -> Path:
    subset_dir = output_dir / subset_name
    subset_dir.mkdir(parents=True, exist_ok=True)
    path = subset_dir / "adapter_delta_summary.csv"
    delta.to_csv(path, index=False)
    return path


def adapter_delta_winner(ut_minus_base: float, game_minus_base: float) -> str:
    if abs(ut_minus_base - game_minus_base) <= 1e-12:
        return "tie"
    return "UT" if ut_minus_base > game_minus_base else "GAME"


def summarize_late_delta(
    delta: pd.DataFrame,
    *,
    subset_name: str,
    late_start: int,
    late_end: int,
) -> dict[str, object]:
    late = delta[(delta["layer"] >= late_start) & (delta["layer"] <= late_end)]
    if late.empty:
        raise ValueError(f"No rows found for {subset_name} layers {late_start}-{late_end}.")
    ut_minus_base = float(late["ut_minus_base"].mean())
    game_minus_base = float(late["game_minus_base"].mean())
    game_minus_ut = float(late["game_minus_ut"].mean())
    return {
        "subset": subset_label(subset_name),
        "subset_name": subset_name,
        "n": int(late["n"].max()),
        "dominant_game_type": dominant_game_type(subset_name),
        "ut_minus_base_late": ut_minus_base,
        "game_minus_base_late": game_minus_base,
        "game_minus_ut_late": game_minus_ut,
        "adapter_delta_winner": adapter_delta_winner(ut_minus_base, game_minus_base),
        "late_layers": f"{late_start}-{late_end}",
    }


def format_signed(value: float) -> str:
    return f"{value:+.3f}"


def format_late_summary_markdown(summary: pd.DataFrame, *, late_start: int, late_end: int) -> str:
    layers = f"{late_start}–{late_end}"
    lines = [
        (
            "| Subset | n | UT−Base late | GAME−Base late | GAME−UT late | "
            "Winner | Layers |"
        ),
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            "| "
            f"{row.subset} | "
            f"{row.n} | "
            f"{format_signed(row.ut_minus_base_late)} | "
            f"{format_signed(row.game_minus_base_late)} | "
            f"{format_signed(row.game_minus_ut_late)} | "
            f"{row.adapter_delta_winner} | "
            f"{layers} |"
        )
    return "\n".join(lines) + "\n"


def plot_individual_delta(
    delta: pd.DataFrame,
    *,
    subset_name: str,
    figure_dir: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt

    from moral_mechinterp.plot_style import apply_paper_style, despine, save_figure

    figure_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style(font_family="serif")
    fig, ax = plt.subplots(figsize=(4.7, 3.0))
    for column in DELTA_COLUMNS:
        ax.plot(
            delta["layer"],
            delta[column],
            color=DELTA_COLORS[column],
            marker=DELTA_MARKERS[column],
            markevery=max(1, len(delta) // 8),
            linewidth=1.45,
            markersize=3.8,
            label=DELTA_LABELS[column],
        )

    ax.axhline(0, color="#2A2A2A", linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_xlabel("Layer")
    ax.set_ylabel("Adapter-induced safe-margin shift")
    ax.set_title(f"Adapter delta: {subset_label(subset_name)}")
    ax.legend(frameon=False)
    ax.yaxis.grid(True)
    despine(ax)
    return save_figure(fig, figure_dir / f"{subset_name}_adapter_delta")


def combined_ylim(deltas: dict[str, pd.DataFrame]) -> tuple[float, float]:
    values: list[float] = []
    for delta in deltas.values():
        values.extend(delta["ut_minus_base"].dropna().tolist())
        values.extend(delta["game_minus_base"].dropna().tolist())
    if not values:
        return -1.0, 1.0
    low = min(values)
    high = max(values)
    pad = 0.08 * max(1e-6, high - low)
    return low - pad, high + pad


def plot_combined_delta(
    deltas: dict[str, pd.DataFrame],
    *,
    figure_dir: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt

    from moral_mechinterp.plot_style import apply_paper_style, despine, save_figure

    figure_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style(font_family="serif")
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.6), sharex=False, sharey=False)
    axes_flat = axes.ravel()
    y_low, y_high = combined_ylim(deltas)

    handles = []
    labels = []
    for panel_idx, (subset_name, delta) in enumerate(deltas.items()):
        ax = axes_flat[panel_idx]
        for column in DELTA_COLUMNS:
            (line,) = ax.plot(
                delta["layer"],
                delta[column],
                color=DELTA_COLORS[column],
                marker=DELTA_MARKERS[column],
                markevery=max(1, len(delta) // 7),
                linewidth=1.25,
                markersize=3.2,
                label=DELTA_LABELS[column],
            )
            if panel_idx == 0:
                handles.append(line)
                labels.append(DELTA_LABELS[column])
        n = int(delta["n"].max())
        title = f"{PANEL_LETTERS[panel_idx]}. {subset_label(subset_name)} (n={n})"
        ax.set_title(title)
        ax.axhline(0, color="#2A2A2A", linewidth=0.75, linestyle=(0, (3, 2)))
        ax.set_ylim(y_low, y_high)
        ax.set_xlim(delta["layer"].min() - 1, delta["layer"].max() + 1)
        ax.set_xticks([0, 10, 20, 30])
        ax.yaxis.grid(True)
        despine(ax)

    for ax in axes[1, :]:
        ax.set_xlabel("Layer")
    for ax in axes[:, 0]:
        ax.set_ylabel("Adapter-induced safe-margin shift")

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.1, wspace=0.28, hspace=0.36)
    return save_figure(fig, figure_dir / "adapter_delta_logit_lens_2x3")


def write_late_summary_outputs(
    late_summary: pd.DataFrame,
    *,
    table_dir: Path,
    late_start: int,
    late_end: int,
) -> tuple[Path, Path]:
    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / "adapter_delta_late_layer_summary.csv"
    md_path = table_dir / "adapter_delta_late_layer_summary.md"
    late_summary.to_csv(csv_path, index=False)
    md_path.write_text(
        format_late_summary_markdown(
            late_summary,
            late_start=late_start,
            late_end=late_end,
        ),
        encoding="utf-8",
    )
    return csv_path, md_path


def main() -> None:
    configure_stdout()
    args = parse_args()
    if args.late_end < args.late_start:
        raise ValueError("--late-end must be greater than or equal to --late-start.")

    pd = load_pandas()
    subsets = parse_subset_names(args.subsets)
    deltas: dict[str, pd.DataFrame] = {}
    late_rows: list[dict[str, object]] = []
    for subset_name in subsets:
        summary = read_layer_summary(args.base_dir, subset_name)
        delta = compute_adapter_delta(summary, subset_name=subset_name)
        deltas[subset_name] = delta
        write_delta_csv(delta, output_dir=args.output_dir, subset_name=subset_name)
        late_rows.append(
            summarize_late_delta(
                delta,
                subset_name=subset_name,
                late_start=args.late_start,
                late_end=args.late_end,
            )
        )
        if not args.no_individual_plots:
            plot_individual_delta(delta, subset_name=subset_name, figure_dir=args.figure_dir)

    late_summary = pd.DataFrame(late_rows)
    csv_path, md_path = write_late_summary_outputs(
        late_summary,
        table_dir=args.table_dir,
        late_start=args.late_start,
        late_end=args.late_end,
    )
    if not args.no_combined_plot:
        plot_combined_delta(deltas, figure_dir=args.figure_dir)

    print("Adapter-delta late-layer summary:")
    print(late_summary.to_string(index=False))
    print("\nCompact Markdown table:")
    print(
        format_late_summary_markdown(
            late_summary,
            late_start=args.late_start,
            late_end=args.late_end,
        ),
        end="",
    )
    print(f"\nWrote CSV: {csv_path}")
    print(f"Wrote Markdown: {md_path}")
    print(f"Wrote per-subset delta CSVs to: {args.output_dir}")
    if not args.no_individual_plots or not args.no_combined_plot:
        print(f"Wrote adapter-delta figures to: {args.figure_dir}")


if __name__ == "__main__":
    main()
