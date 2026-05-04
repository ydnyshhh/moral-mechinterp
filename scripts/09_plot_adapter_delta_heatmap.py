from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


SUBSET_SPECS = {
    "top_ut_margin_shift": "UT-favored margin shifts",
    "top_game_margin_shift": "GAME-favored margin shifts",
    "ut_safe_game_harmful": "UT-safe / GAME-harmful",
    "game_safe_ut_harmful": "GAME-safe / UT-harmful",
    "random_pd_150": "Random Prisoner’s Dilemma",
    "random_chicken_150": "Random Chicken",
}
REQUIRED_MODELS = ("base", "ut", "game")
DEFAULT_SUBSETS = ",".join(SUBSET_SPECS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create side-by-side adapter-delta logit-lens heatmaps from saved "
            "layer_margin_summary.csv files. This is a pure post-processing step "
            "and does not run model inference."
        )
    )
    parser.add_argument("--base-dir", type=Path, default=Path("outputs/logit_lens_fixed"))
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("outputs/figures_adapter_delta"),
    )
    parser.add_argument("--table-dir", type=Path, default=Path("outputs/tables_full"))
    parser.add_argument("--subsets", default=DEFAULT_SUBSETS)
    parser.add_argument(
        "--vlim",
        type=float,
        default=None,
        help="Optional symmetric heatmap color limit. Uses [-vlim, +vlim].",
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
    subsets = [subset.strip() for subset in raw.split(",") if subset.strip()]
    if not subsets:
        raise ValueError("At least one subset is required.")
    return subsets


def subset_label(subset_name: str) -> str:
    return SUBSET_SPECS.get(subset_name, subset_name)


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


def compute_delta_by_layer(summary: pd.DataFrame, *, subset_name: str) -> pd.DataFrame:
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

    delta = pd.DataFrame(
        {
            "layer": means.index.astype(int),
            "ut_minus_base": means["ut"].to_numpy(dtype=float)
            - means["base"].to_numpy(dtype=float),
            "game_minus_base": means["game"].to_numpy(dtype=float)
            - means["base"].to_numpy(dtype=float),
        }
    )
    return delta.sort_values("layer").reset_index(drop=True)


def collect_deltas(
    base_dir: Path,
    subsets: list[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, int], list[int]]:
    deltas: dict[str, pd.DataFrame] = {}
    n_by_subset: dict[str, int] = {}
    layer_set: set[int] = set()
    for subset_name in subsets:
        summary = read_layer_summary(base_dir, subset_name)
        deltas[subset_name] = compute_delta_by_layer(summary, subset_name=subset_name)
        n_by_subset[subset_name] = int(summary["n"].max())
        layer_set.update(int(layer) for layer in summary["layer"].unique())

    if not layer_set:
        raise ValueError("No layer values were found in the requested summaries.")
    return deltas, n_by_subset, sorted(layer_set)


def build_heatmap_table(
    deltas: dict[str, pd.DataFrame],
    *,
    subsets: list[str],
    layers: list[int],
    value_column: str,
) -> pd.DataFrame:
    pd = load_pandas()
    rows: list[dict[str, object]] = []
    for subset_name in subsets:
        delta = deltas[subset_name].set_index("layer")
        row: dict[str, object] = {
            "subset": subset_label(subset_name),
            "subset_name": subset_name,
        }
        for layer in layers:
            row[f"layer_{layer}"] = (
                float(delta.loc[layer, value_column]) if layer in delta.index else pd.NA
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_heatmap_tables(
    table_dir: Path,
    ut_table: pd.DataFrame,
    game_table: pd.DataFrame,
) -> tuple[Path, Path]:
    table_dir.mkdir(parents=True, exist_ok=True)
    ut_path = table_dir / "adapter_delta_heatmap_ut_minus_base.csv"
    game_path = table_dir / "adapter_delta_heatmap_game_minus_base.csv"
    ut_table.to_csv(ut_path, index=False)
    game_table.to_csv(game_path, index=False)
    return ut_path, game_path


def table_to_matrix(table: pd.DataFrame, layers: list[int]):
    layer_columns = [f"layer_{layer}" for layer in layers]
    return table[layer_columns].to_numpy(dtype=float)


def smooth_matrix_by_layer(
    matrix,
    layers: list[int],
    *,
    points_per_layer: int = 10,
):
    import numpy as np

    if points_per_layer < 1:
        raise ValueError("points_per_layer must be at least 1.")
    layer_array = np.asarray(layers, dtype=float)
    display_layers = np.linspace(
        float(layer_array.min()),
        float(layer_array.max()),
        int((layer_array.max() - layer_array.min()) * points_per_layer) + 1,
    )
    smoothed = np.vstack(
        [
            np.interp(display_layers, layer_array, row.astype(float))
            for row in matrix
        ]
    )
    return smoothed, display_layers


def symmetric_vlim(*matrices, requested_vlim: float | None = None) -> float:
    import numpy as np

    if requested_vlim is not None:
        if requested_vlim <= 0:
            raise ValueError("--vlim must be positive when provided.")
        return float(requested_vlim)

    finite_values = []
    for matrix in matrices:
        values = matrix[np.isfinite(matrix)]
        if values.size:
            finite_values.append(values)
    if not finite_values:
        return 1.0

    max_abs = float(np.max(np.abs(np.concatenate(finite_values))))
    return max(max_abs, 1e-6)


def layer_to_heatmap_x(layer: float, display_layers) -> float:
    import numpy as np

    return float(np.interp(layer, display_layers, np.arange(len(display_layers))) + 0.5)


def add_late_layer_box(
    ax,
    *,
    display_layers,
    late_start: int,
    late_end: int,
) -> None:
    from matplotlib.patches import Rectangle

    if late_start < min(display_layers) or late_end > max(display_layers):
        return
    left = layer_to_heatmap_x(late_start, display_layers) - 0.5
    right = layer_to_heatmap_x(late_end, display_layers) + 0.5
    ax.add_patch(
        Rectangle(
            (left, 0),
            right - left,
            len(ax.get_yticklabels()),
            fill=False,
            edgecolor="#2A2A2A",
            linewidth=0.75,
            linestyle=(0, (3, 2)),
            alpha=0.75,
        )
    )


def set_layer_ticks(ax, *, display_layers) -> None:
    desired_ticks = [0, 5, 10, 15, 20, 25, 30, 32]
    ticks = [
        layer_to_heatmap_x(layer, display_layers)
        for layer in desired_ticks
        if min(display_layers) <= layer <= max(display_layers)
    ]
    labels = [
        str(layer)
        for layer in desired_ticks
        if min(display_layers) <= layer <= max(display_layers)
    ]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=0)


def plot_heatmap(
    ut_table: pd.DataFrame,
    game_table: pd.DataFrame,
    *,
    layers: list[int],
    n_by_subset: dict[str, int],
    figure_dir: Path,
    vlim: float,
    late_start: int,
    late_end: int,
) -> list[Path]:
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap

    from moral_mechinterp.plot_style import apply_paper_style, save_figure

    sns.set_theme(context="paper", style="white")
    apply_paper_style(font_family="serif")

    ut_matrix = table_to_matrix(ut_table, layers)
    game_matrix = table_to_matrix(game_table, layers)
    ut_display_matrix, display_layers = smooth_matrix_by_layer(ut_matrix, layers)
    game_display_matrix, _ = smooth_matrix_by_layer(game_matrix, layers)
    row_labels = [
        f"{row.subset} (n={n_by_subset[str(row.subset_name)]})"
        for row in ut_table.itertuples(index=False)
    ]
    late_positions = [
        idx for idx, layer in enumerate(layers) if late_start <= layer <= late_end
    ]
    if not late_positions:
        raise ValueError(f"No heatmap columns found for late layers {late_start}-{late_end}.")
    ut_late = np.nanmean(ut_matrix[:, late_positions], axis=1, keepdims=True)
    game_late = np.nanmean(game_matrix[:, late_positions], axis=1, keepdims=True)
    ut_late_annot = np.array([[f"{value:+.2f}"] for value in ut_late[:, 0]])
    game_late_annot = np.array([[f"{value:+.2f}"] for value in game_late[:, 0]])
    cmap = LinearSegmentedColormap.from_list(
        "adapter_delta_icml",
        ["#173B7A", "#19A7B8", "#F8F7F1", "#F3A11B", "#B51F35"],
        N=256,
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11.8, 4.25))
    grid = fig.add_gridspec(
        1,
        5,
        width_ratios=[1.0, 0.13, 1.0, 0.13, 0.045],
        left=0.255,
        right=0.955,
        top=0.82,
        bottom=0.24,
        wspace=0.12,
    )
    ax_ut = fig.add_subplot(grid[0, 0])
    ax_ut_late = fig.add_subplot(grid[0, 1])
    ax_game = fig.add_subplot(grid[0, 2])
    ax_game_late = fig.add_subplot(grid[0, 3])
    cbar_ax = fig.add_subplot(grid[0, 4])

    heatmap_kwargs = {
        "cmap": cmap,
        "vmin": -vlim,
        "vmax": vlim,
        "center": 0.0,
        "linewidths": 0.0,
        "linecolor": "#FFFFFF",
        "square": False,
    }
    late_heatmap_kwargs = {
        **heatmap_kwargs,
        "linewidths": 0.35,
    }
    sns.heatmap(
        ut_display_matrix,
        ax=ax_ut,
        cbar=False,
        yticklabels=row_labels,
        xticklabels=False,
        **heatmap_kwargs,
    )
    sns.heatmap(
        game_display_matrix,
        ax=ax_game,
        cbar=True,
        cbar_ax=cbar_ax,
        yticklabels=False,
        xticklabels=False,
        cbar_kws={"label": "Safe-margin shift relative to Base"},
        **heatmap_kwargs,
    )
    sns.heatmap(
        ut_late,
        ax=ax_ut_late,
        cbar=False,
        yticklabels=False,
        xticklabels=False,
        annot=ut_late_annot,
        fmt="",
        annot_kws={"fontsize": 7.4, "fontweight": "bold"},
        **late_heatmap_kwargs,
    )
    sns.heatmap(
        game_late,
        ax=ax_game_late,
        cbar=False,
        yticklabels=False,
        xticklabels=False,
        annot=game_late_annot,
        fmt="",
        annot_kws={"fontsize": 7.4, "fontweight": "bold"},
        **late_heatmap_kwargs,
    )
    ax_game.set_ylim(ax_ut.get_ylim())
    ax_ut_late.set_ylim(ax_ut.get_ylim())
    ax_game_late.set_ylim(ax_ut.get_ylim())

    for ax, title in ((ax_ut, "UT − Base"), (ax_game, "GAME − Base")):
        ax.set_title(title, pad=8)
        ax.set_xlabel("Layer")
        ax.tick_params(axis="both", length=0)
        set_layer_ticks(ax, display_layers=display_layers)
        add_late_layer_box(
            ax,
            display_layers=display_layers,
            late_start=late_start,
            late_end=late_end,
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.65)
            spine.set_color("#2A2A2A")

    for ax in (ax_ut_late, ax_game_late):
        ax.set_title(f"Late mean\n{late_start}–{late_end}", pad=6)
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.65)
            spine.set_color("#2A2A2A")

    for ax in (ax_ut, ax_ut_late, ax_game, ax_game_late):
        ax.hlines(
            4,
            *ax.get_xlim(),
            color="#2A2A2A",
            linewidth=0.95,
            linestyle=(0, (2, 2)),
            alpha=0.72,
        )

    ax_ut.set_yticks([idx + 0.5 for idx in range(len(row_labels))])
    ax_ut.set_yticklabels(row_labels, rotation=0)
    ax_ut.set_ylabel("Subset")
    ax_ut.tick_params(axis="y", pad=4)
    ax_ut_late.set_ylabel("")
    ax_ut_late.set_yticks([idx + 0.5 for idx in range(len(row_labels))])
    ax_ut_late.set_yticklabels([])
    ax_game.set_ylabel("")
    ax_game.set_yticks([idx + 0.5 for idx in range(len(row_labels))])
    ax_game.set_yticklabels([])
    ax_game_late.set_ylabel("")
    ax_game_late.set_yticks([idx + 0.5 for idx in range(len(row_labels))])
    ax_game_late.set_yticklabels([])
    ax_game.tick_params(labelleft=False)
    ax_game_late.tick_params(labelleft=False)
    cbar_ax.tick_params(length=2.5, width=0.7)
    cbar_ax.yaxis.label.set_size(8.5)

    fig.text(
        0.255,
        0.13,
        (
            "Positive values indicate increased safe-action evidence relative to Base. "
            f"Dashed boxes mark late layers {late_start}–{late_end}; side cells show "
            "the late-layer mean."
        ),
        ha="left",
        va="center",
        fontsize=8.2,
        color="#2A2A2A",
    )
    return save_figure(fig, figure_dir / "adapter_delta_logit_lens_heatmap")


def adapter_delta_winner(ut_minus_base: float, game_minus_base: float) -> str:
    if abs(ut_minus_base - game_minus_base) <= 1e-12:
        return "tie"
    return "UT" if ut_minus_base > game_minus_base else "GAME"


def summarize_late_layers(
    deltas: dict[str, pd.DataFrame],
    *,
    subsets: list[str],
    n_by_subset: dict[str, int],
    late_start: int,
    late_end: int,
) -> pd.DataFrame:
    pd = load_pandas()
    rows: list[dict[str, object]] = []
    for subset_name in subsets:
        delta = deltas[subset_name]
        late = delta[(delta["layer"] >= late_start) & (delta["layer"] <= late_end)]
        if late.empty:
            raise ValueError(f"No rows found for {subset_name} layers {late_start}-{late_end}.")
        ut_minus_base = float(late["ut_minus_base"].mean())
        game_minus_base = float(late["game_minus_base"].mean())
        rows.append(
            {
                "subset": subset_label(subset_name),
                "subset_name": subset_name,
                "n": n_by_subset[subset_name],
                "ut_minus_base_late": ut_minus_base,
                "game_minus_base_late": game_minus_base,
                "winner": adapter_delta_winner(ut_minus_base, game_minus_base),
                "late_layers": f"{late_start}-{late_end}",
            }
        )
    return pd.DataFrame(rows)


def format_signed(value: float) -> str:
    return f"{value:+.3f}"


def format_late_summary_markdown(summary: pd.DataFrame, *, late_start: int, late_end: int) -> str:
    layers = f"{late_start}–{late_end}"
    lines = [
        "| Subset | n | UT−Base late | GAME−Base late | Winner | Layers |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            "| "
            f"{row.subset} | "
            f"{row.n} | "
            f"{format_signed(row.ut_minus_base_late)} | "
            f"{format_signed(row.game_minus_base_late)} | "
            f"{row.winner} | "
            f"{layers} |"
        )
    return "\n".join(lines) + "\n"


def write_late_summary(
    table_dir: Path,
    summary: pd.DataFrame,
    *,
    late_start: int,
    late_end: int,
) -> tuple[Path, Path]:
    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / "adapter_delta_heatmap_late_summary.csv"
    md_path = table_dir / "adapter_delta_heatmap_late_summary.md"
    summary.to_csv(csv_path, index=False)
    md_path.write_text(
        format_late_summary_markdown(
            summary,
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

    subsets = parse_subset_names(args.subsets)
    deltas, n_by_subset, layers = collect_deltas(args.base_dir, subsets)
    ut_table = build_heatmap_table(
        deltas,
        subsets=subsets,
        layers=layers,
        value_column="ut_minus_base",
    )
    game_table = build_heatmap_table(
        deltas,
        subsets=subsets,
        layers=layers,
        value_column="game_minus_base",
    )
    ut_path, game_path = write_heatmap_tables(args.table_dir, ut_table, game_table)

    ut_matrix = table_to_matrix(ut_table, layers)
    game_matrix = table_to_matrix(game_table, layers)
    vlim = symmetric_vlim(ut_matrix, game_matrix, requested_vlim=args.vlim)
    figure_paths = plot_heatmap(
        ut_table,
        game_table,
        layers=layers,
        n_by_subset=n_by_subset,
        figure_dir=args.figure_dir,
        vlim=vlim,
        late_start=args.late_start,
        late_end=args.late_end,
    )

    late_summary = summarize_late_layers(
        deltas,
        subsets=subsets,
        n_by_subset=n_by_subset,
        late_start=args.late_start,
        late_end=args.late_end,
    )
    summary_csv, summary_md = write_late_summary(
        args.table_dir,
        late_summary,
        late_start=args.late_start,
        late_end=args.late_end,
    )

    print("Adapter-delta heatmap late-layer summary:")
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
    print(f"\nWrote UT heatmap table: {ut_path}")
    print(f"Wrote GAME heatmap table: {game_path}")
    print(f"Wrote late summary CSV: {summary_csv}")
    print(f"Wrote late summary Markdown: {summary_md}")
    print("Wrote heatmap figures:")
    for path in figure_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
