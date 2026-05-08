from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


SUBSET_ORDER = (
    "top_ut_margin_shift",
    "top_game_margin_shift",
    "ut_safe_game_harmful",
    "game_safe_ut_harmful",
    "random_pd_150",
    "random_chicken_150",
)
SUBSET_LABELS = {
    "top_ut_margin_shift": "UT-favored margin shifts",
    "top_game_margin_shift": "GAME-favored margin shifts",
    "ut_safe_game_harmful": "UT-safe / GAME-harmful",
    "game_safe_ut_harmful": "GAME-safe / UT-harmful",
    "random_pd_150": "Random Prisoner’s Dilemma",
    "random_chicken_150": "Random Chicken",
}
LABEL_TO_SUBSET = {label: subset_name for subset_name, label in SUBSET_LABELS.items()}
LABEL_TO_SUBSET["Random Prisoner's Dilemma"] = "random_pd_150"
MARGIN_COLUMNS = (
    "ut_minus_base_late",
    "game_minus_base_late",
    "game_minus_ut_late",
)
DRIFT_COLUMNS = (
    "base_ut_late_drift",
    "base_game_late_drift",
    "ut_game_late_drift",
)
OUTPUT_COLUMNS = (
    "subset",
    "subset_name",
    "n",
    *MARGIN_COLUMNS,
    *DRIFT_COLUMNS,
    "late_layers",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a two-panel late-layer effect summary heatmap from saved "
            "adapter-delta and representation-drift summary tables. This only "
            "reads existing CSV outputs and does not run model inference."
        )
    )
    parser.add_argument("--table-dir", type=Path, default=Path("outputs/tables_full"))
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("outputs/figures_adapter_delta"),
    )
    parser.add_argument("--late-layers-label", default="21–31")
    parser.add_argument(
        "--margin-vlim",
        type=float,
        default=None,
        help="Optional symmetric margin color limit. Uses [-vlim, +vlim].",
    )
    parser.add_argument(
        "--drift-vmax",
        type=float,
        default=None,
        help="Optional upper color limit for representation drift.",
    )
    parser.add_argument("--output-prefix", default="late_layer_effect_summary_heatmap")
    return parser.parse_args()


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def load_pandas():
    import pandas as pd

    return pd


def read_csv(path: Path) -> pd.DataFrame:
    pd = load_pandas()
    if not path.exists():
        raise FileNotFoundError(f"Missing required table: {path}")
    return pd.read_csv(path)


def first_existing(columns: set[str], candidates: tuple[str, ...], *, label: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not find a column for {label}; tried {list(candidates)}.")


def read_margin_summary(table_dir: Path) -> pd.DataFrame:
    preferred = table_dir / "adapter_delta_heatmap_late_summary.csv"
    fallback = table_dir / "adapter_delta_late_layer_summary.csv"
    path = preferred if preferred.exists() else fallback
    margin = read_csv(path).copy()
    columns = set(margin.columns)

    subset_key = first_existing(columns, ("subset_name", "subset"), label="subset key")
    subset_label_key = "subset" if "subset" in columns else subset_key
    n_key = first_existing(columns, ("n",), label="n")
    late_key = "late_layers" if "late_layers" in columns else None
    ut_key = first_existing(
        columns,
        ("ut_minus_base_late", "ut_minus_base"),
        label="UT minus Base late margin",
    )
    game_key = first_existing(
        columns,
        ("game_minus_base_late", "game_minus_base"),
        label="GAME minus Base late margin",
    )

    standardized = margin.rename(
        columns={
            subset_label_key: "subset",
            n_key: "n",
            ut_key: "ut_minus_base_late",
            game_key: "game_minus_base_late",
        }
    )
    if subset_key == "subset_name":
        standardized["subset_name"] = margin[subset_key].astype(str)
    else:
        standardized["subset_name"] = (
            margin[subset_key].astype(str).map(LABEL_TO_SUBSET).fillna(margin[subset_key].astype(str))
        )
    if late_key is not None and late_key != "late_layers":
        standardized = standardized.rename(columns={late_key: "late_layers"})

    if "game_minus_ut_late" not in standardized.columns:
        game_minus_ut_key = next(
            (
                column
                for column in ("game_minus_ut_late", "game_minus_ut")
                if column in columns
            ),
            None,
        )
        if game_minus_ut_key is not None:
            standardized["game_minus_ut_late"] = margin[game_minus_ut_key]
        else:
            standardized["game_minus_ut_late"] = (
                standardized["game_minus_base_late"]
                - standardized["ut_minus_base_late"]
            )

    required = {"subset_name", "subset", "n", *MARGIN_COLUMNS}
    missing = sorted(required - set(standardized.columns))
    if missing:
        raise ValueError(f"{path} is missing required margin fields after normalization: {missing}")
    keep = ["subset", "subset_name", "n", *MARGIN_COLUMNS]
    if "late_layers" in standardized.columns:
        keep.append("late_layers")
    return standardized[keep]


def read_drift_summary(table_dir: Path) -> pd.DataFrame:
    path = table_dir / "late_layer_representation_drift.csv"
    drift = read_csv(path).copy()
    columns = set(drift.columns)
    subset_key = first_existing(columns, ("subset_name", "subset"), label="subset key")
    subset_label_key = "subset" if "subset" in columns else subset_key
    n_key = first_existing(columns, ("n",), label="n")
    required = {
        "base_ut_late_drift",
        "base_game_late_drift",
        "ut_game_late_drift",
    }
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"{path} is missing required drift fields: {missing}")

    rename_map = {
        subset_label_key: "drift_subset_label",
        n_key: "drift_n",
    }
    if "late_layers" in columns:
        rename_map["late_layers"] = "drift_late_layers"
    standardized = drift.rename(columns=rename_map)
    if subset_key == "subset_name":
        standardized["subset_name"] = drift[subset_key].astype(str)
    else:
        standardized["subset_name"] = (
            drift[subset_key].astype(str).map(LABEL_TO_SUBSET).fillna(drift[subset_key].astype(str))
        )
    keep = [
        "subset_name",
        "drift_subset_label",
        "drift_n",
        *DRIFT_COLUMNS,
    ]
    if "drift_late_layers" in standardized.columns:
        keep.append("drift_late_layers")
    return standardized[keep]


def ordered_combined_table(table_dir: Path, *, late_layers_label: str) -> pd.DataFrame:
    pd = load_pandas()
    margin = read_margin_summary(table_dir)
    drift = read_drift_summary(table_dir)
    merge_key = (
        "subset_name"
        if "subset_name" in margin.columns and "subset_name" in drift.columns
        else "subset"
    )
    combined = margin.merge(drift, on=merge_key, how="outer", validate="one_to_one")

    if "subset_name" not in combined.columns:
        combined["subset_name"] = combined["subset"]
    combined["subset_name"] = combined["subset_name"].astype(str)
    combined["subset"] = combined["subset_name"].map(SUBSET_LABELS).fillna(combined["subset"])
    combined["n"] = combined["n"].fillna(combined.get("drift_n")).astype(int)
    if "late_layers" not in combined.columns:
        combined["late_layers"] = combined.get("drift_late_layers", late_layers_label)
    combined["late_layers"] = combined["late_layers"].fillna(late_layers_label)

    missing_subsets = [
        subset_name
        for subset_name in SUBSET_ORDER
        if subset_name not in set(combined["subset_name"])
    ]
    if missing_subsets:
        raise ValueError(f"Missing required subset rows after merge: {missing_subsets}")

    combined["_order"] = pd.Categorical(
        combined["subset_name"],
        categories=list(SUBSET_ORDER),
        ordered=True,
    )
    combined = combined[combined["subset_name"].isin(SUBSET_ORDER)]
    combined = combined.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    for column in (*MARGIN_COLUMNS, *DRIFT_COLUMNS):
        combined[column] = pd.to_numeric(combined[column], errors="raise")
    return combined[list(OUTPUT_COLUMNS)]


def write_combined_outputs(
    combined: pd.DataFrame,
    *,
    table_dir: Path,
    output_prefix: str,
) -> tuple[Path, Path]:
    table_dir.mkdir(parents=True, exist_ok=True)
    csv_path = table_dir / f"{output_prefix}.csv"
    md_path = table_dir / f"{output_prefix}.md"
    combined.to_csv(csv_path, index=False)
    md_path.write_text(format_markdown(combined), encoding="utf-8")
    return csv_path, md_path


def format_signed(value: float) -> str:
    return f"{value:+.3f}"


def format_drift(value: float) -> str:
    return f"{value:.5f}"


def format_markdown(combined: pd.DataFrame) -> str:
    lines = [
        (
            "| Subset | n | UT−Base margin | GAME−Base margin | GAME−UT margin | "
            "Base–UT drift | Base–GAME drift | UT–GAME drift | Layers |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in combined.itertuples(index=False):
        lines.append(
            "| "
            f"{row.subset} | "
            f"{row.n} | "
            f"{format_signed(row.ut_minus_base_late)} | "
            f"{format_signed(row.game_minus_base_late)} | "
            f"{format_signed(row.game_minus_ut_late)} | "
            f"{format_drift(row.base_ut_late_drift)} | "
            f"{format_drift(row.base_game_late_drift)} | "
            f"{format_drift(row.ut_game_late_drift)} | "
            f"{row.late_layers} |"
        )
    return "\n".join(lines) + "\n"


def matrix_values(combined: pd.DataFrame, columns: tuple[str, ...]):
    return combined[list(columns)].to_numpy(dtype=float)


def annotation_matrix(combined: pd.DataFrame, columns: tuple[str, ...], *, kind: str):
    import numpy as np

    values = matrix_values(combined, columns)
    if kind == "margin":
        return np.vectorize(format_signed)(values)
    if kind == "drift":
        return np.vectorize(format_drift)(values)
    raise ValueError("kind must be 'margin' or 'drift'")


def symmetric_limit(values, requested: float | None) -> float:
    import numpy as np

    if requested is not None:
        if requested <= 0:
            raise ValueError("--margin-vlim must be positive when provided.")
        return float(requested)
    max_abs = float(np.nanmax(np.abs(values)))
    return max(max_abs, 1e-6)


def positive_limit(values, requested: float | None) -> float:
    import numpy as np

    if requested is not None:
        if requested <= 0:
            raise ValueError("--drift-vmax must be positive when provided.")
        return float(requested)
    max_value = float(np.nanmax(values))
    return max(max_value, 1e-9)


def draw_effect_heatmap(
    *,
    ax,
    cbar_ax,
    values,
    annotations,
    row_labels: list[str],
    column_labels: list[str],
    cmap,
    vmin: float,
    vmax: float,
    center: float | None,
    title: str,
    colorbar_label: str,
    annotation_size: float,
) -> None:
    import seaborn as sns

    heatmap_kwargs = {
        "ax": ax,
        "cmap": cmap,
        "vmin": vmin,
        "vmax": vmax,
        "linewidths": 0.6,
        "linecolor": "#FFFFFF",
        "annot": annotations,
        "fmt": "",
        "annot_kws": {"fontsize": annotation_size, "fontweight": "bold"},
        "xticklabels": column_labels,
        "yticklabels": row_labels,
        "cbar": True,
        "cbar_ax": cbar_ax,
        "cbar_kws": {"label": colorbar_label},
    }
    if center is not None:
        heatmap_kwargs["center"] = center
    sns.heatmap(values, **heatmap_kwargs)

    ax.set_title(title, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("Subset")
    ax.tick_params(axis="both", length=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.hlines(
        4,
        *ax.get_xlim(),
        color="#2A2A2A",
        linewidth=0.95,
        linestyle=(0, (2, 2)),
        alpha=0.72,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
        spine.set_color("#2A2A2A")
    cbar_ax.tick_params(length=2.5, width=0.7)
    cbar_ax.yaxis.label.set_size(8.5)


def plot_single_effect_summary(
    combined: pd.DataFrame,
    *,
    figure_dir: Path,
    output_prefix: str,
    late_layers_label: str,
    values,
    annotations,
    row_labels: list[str],
    column_labels: list[str],
    cmap,
    vmin: float,
    vmax: float,
    center: float | None,
    title: str,
    colorbar_label: str,
    annotation_size: float,
    colorbar_format: str,
) -> list[Path]:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import FormatStrFormatter

    from moral_mechinterp.plot_style import apply_paper_style, save_figure

    sns.set_theme(context="paper", style="white")
    apply_paper_style(font_family="serif")

    figure_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6.9, 4.25))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 0.055],
        left=0.36,
        right=0.94,
        top=0.82,
        bottom=0.24,
        wspace=0.16,
    )
    ax = fig.add_subplot(grid[0, 0])
    cbar_ax = fig.add_subplot(grid[0, 1])
    draw_effect_heatmap(
        ax=ax,
        cbar_ax=cbar_ax,
        values=values,
        annotations=annotations,
        row_labels=row_labels,
        column_labels=column_labels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=center,
        title=title,
        colorbar_label=colorbar_label,
        annotation_size=annotation_size,
    )
    cbar_ax.yaxis.set_major_formatter(FormatStrFormatter(colorbar_format))
    fig.text(
        0.36,
        0.12,
        f"Late-layer means average layers {late_layers_label}; layer 32 is excluded.",
        ha="left",
        va="center",
        fontsize=8.2,
        color="#2A2A2A",
    )
    return save_figure(fig, figure_dir / output_prefix)


def plot_effect_summary(
    combined: pd.DataFrame,
    *,
    figure_dir: Path,
    output_prefix: str,
    late_layers_label: str,
    margin_vlim: float | None,
    drift_vmax: float | None,
) -> list[Path]:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.ticker import FormatStrFormatter

    from moral_mechinterp.plot_style import apply_paper_style, save_figure

    sns.set_theme(context="paper", style="white")
    apply_paper_style(font_family="serif")

    margin_values = matrix_values(combined, MARGIN_COLUMNS)
    drift_values = matrix_values(combined, DRIFT_COLUMNS)
    margin_limit = symmetric_limit(margin_values, margin_vlim)
    drift_limit = positive_limit(drift_values, drift_vmax)
    row_labels = [f"{row.subset} (n={row.n})" for row in combined.itertuples(index=False)]

    margin_cmap = LinearSegmentedColormap.from_list(
        "late_layer_margin_shift",
        ["#173B7A", "#19A7B8", "#F8F7F1", "#F3A11B", "#B51F35"],
        N=256,
    )
    drift_cmap = LinearSegmentedColormap.from_list(
        "late_layer_cosine_drift",
        ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"],
        N=256,
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    separate_paths: list[Path] = []
    separate_paths.extend(
        plot_single_effect_summary(
            combined,
            figure_dir=figure_dir,
            output_prefix=f"{output_prefix}_safe_action_margin_shifts",
            late_layers_label=late_layers_label,
            values=margin_values,
            annotations=annotation_matrix(combined, MARGIN_COLUMNS, kind="margin"),
            row_labels=row_labels,
            column_labels=["UT-Base", "GAME-Base", "GAME-UT"],
            cmap=margin_cmap,
            vmin=-margin_limit,
            vmax=margin_limit,
            center=0.0,
            title="Safe-action margin shifts",
            colorbar_label="Late-layer safe-margin shift",
            annotation_size=8.4,
            colorbar_format="%.1f",
        )
    )
    separate_paths.extend(
        plot_single_effect_summary(
            combined,
            figure_dir=figure_dir,
            output_prefix=f"{output_prefix}_representation_drift",
            late_layers_label=late_layers_label,
            values=drift_values,
            annotations=annotation_matrix(combined, DRIFT_COLUMNS, kind="drift"),
            row_labels=row_labels,
            column_labels=["Base-UT", "Base-GAME", "UT-GAME"],
            cmap=drift_cmap,
            vmin=0.0,
            vmax=drift_limit,
            center=None,
            title="Representation drift",
            colorbar_label="Late-layer cosine drift",
            annotation_size=8.0,
            colorbar_format="%.5f",
        )
    )

    fig = plt.figure(figsize=(9.6, 3.85))
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=[1.0, 0.055, 1.0, 0.055],
        left=0.285,
        right=0.965,
        top=0.83,
        bottom=0.24,
        wspace=0.22,
    )
    ax_margin = fig.add_subplot(grid[0, 0])
    cbar_margin = fig.add_subplot(grid[0, 1])
    ax_drift = fig.add_subplot(grid[0, 2])
    cbar_drift = fig.add_subplot(grid[0, 3])

    sns.heatmap(
        margin_values,
        ax=ax_margin,
        cmap=margin_cmap,
        vmin=-margin_limit,
        vmax=margin_limit,
        center=0.0,
        linewidths=0.6,
        linecolor="#FFFFFF",
        annot=annotation_matrix(combined, MARGIN_COLUMNS, kind="margin"),
        fmt="",
        annot_kws={"fontsize": 7.6, "fontweight": "bold"},
        xticklabels=["UT−Base", "GAME−Base", "GAME−UT"],
        yticklabels=row_labels,
        cbar=True,
        cbar_ax=cbar_margin,
        cbar_kws={"label": "Late-layer safe-margin shift"},
    )
    sns.heatmap(
        drift_values,
        ax=ax_drift,
        cmap=drift_cmap,
        vmin=0.0,
        vmax=drift_limit,
        linewidths=0.6,
        linecolor="#FFFFFF",
        annot=annotation_matrix(combined, DRIFT_COLUMNS, kind="drift"),
        fmt="",
        annot_kws={"fontsize": 7.3, "fontweight": "bold"},
        xticklabels=["Base–UT", "Base–GAME", "UT–GAME"],
        yticklabels=False,
        cbar=True,
        cbar_ax=cbar_drift,
        cbar_kws={"label": "Late-layer cosine drift"},
    )

    ax_margin.set_title("A. Safe-action margin shifts", pad=8)
    ax_drift.set_title("B. Representation drift", pad=8)
    ax_margin.set_ylabel("Subset")
    ax_drift.set_ylabel("")
    ax_drift.tick_params(labelleft=False)

    for ax in (ax_margin, ax_drift):
        ax.set_xlabel("")
        ax.tick_params(axis="both", length=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.hlines(
            4,
            *ax.get_xlim(),
            color="#2A2A2A",
            linewidth=0.95,
            linestyle=(0, (2, 2)),
            alpha=0.72,
        )
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.65)
            spine.set_color("#2A2A2A")

    for cbar_ax in (cbar_margin, cbar_drift):
        cbar_ax.tick_params(length=2.5, width=0.7)
        cbar_ax.yaxis.label.set_size(8.5)
    cbar_margin.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cbar_drift.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))

    fig.text(
        0.285,
        0.13,
        (
            f"Late-layer effect summary, layers {late_layers_label}. "
            "Layer 32 is excluded."
        ),
        ha="left",
        va="center",
        fontsize=8.2,
        color="#2A2A2A",
    )
    combined_paths = save_figure(fig, figure_dir / output_prefix)
    return combined_paths + separate_paths


def main() -> None:
    configure_stdout()
    args = parse_args()
    combined = ordered_combined_table(args.table_dir, late_layers_label=args.late_layers_label)
    csv_path, md_path = write_combined_outputs(
        combined,
        table_dir=args.table_dir,
        output_prefix=args.output_prefix,
    )
    figure_paths = plot_effect_summary(
        combined,
        figure_dir=args.figure_dir,
        output_prefix=args.output_prefix,
        late_layers_label=args.late_layers_label,
        margin_vlim=args.margin_vlim,
        drift_vmax=args.drift_vmax,
    )

    print("Late-layer effect summary:")
    print(combined.to_string(index=False))
    print("\nCompact Markdown table:")
    print(format_markdown(combined), end="")
    print(f"\nWrote combined CSV: {csv_path}")
    print(f"Wrote Markdown: {md_path}")
    print("Wrote effect-summary figures:")
    for path in figure_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
