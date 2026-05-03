"""Polished static plotting for behavior summaries."""

from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from moral_mechinterp.constants import MODEL_COLORS, MODEL_LABELS, MODEL_MARKERS, MODEL_ORDER
from moral_mechinterp.metrics import bootstrap_ci, coerce_bool_series
from moral_mechinterp.plot_style import apply_paper_style, despine, panel_label, save_figure


def _available_prefixes(df: pd.DataFrame) -> list[str]:
    return [prefix for prefix in MODEL_ORDER if f"{prefix}_safe" in df.columns]


def plot_model_safe_rates(
    df: pd.DataFrame,
    outdir: str | Path,
    *,
    font_family: str = "serif",
) -> list[Path]:
    apply_paper_style(font_family=font_family)
    prefixes = _available_prefixes(df)
    fig, ax = plt.subplots(figsize=(3.35, 2.45))

    x = np.arange(len(prefixes))
    means: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for prefix in prefixes:
        values = coerce_bool_series(df[f"{prefix}_safe"]).astype(float).to_numpy()
        mean = float(np.nanmean(values))
        low, high = bootstrap_ci(values)
        means.append(mean)
        lows.append(mean - low)
        highs.append(high - mean)

    for idx, prefix in enumerate(prefixes):
        ax.errorbar(
            x[idx],
            means[idx],
            yerr=np.array([[lows[idx]], [highs[idx]]]),
            color=MODEL_COLORS[prefix],
            marker=MODEL_MARKERS[prefix],
            markersize=5,
            capsize=3,
            linewidth=1.2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[prefix] for prefix in prefixes])
    for tick, prefix in zip(ax.get_xticklabels(), prefixes, strict=True):
        tick.set_color(MODEL_COLORS[prefix])
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Safe-action rate")
    ax.set_title("Model safety preference")
    ax.yaxis.grid(True)
    despine(ax)
    return save_figure(fig, Path(outdir) / "safe_action_rates")


def plot_safe_margin_distributions(
    df: pd.DataFrame,
    outdir: str | Path,
    *,
    font_family: str = "serif",
) -> list[Path]:
    apply_paper_style(font_family=font_family)
    prefixes = _available_prefixes(df)
    data = [
        pd.to_numeric(df[f"{prefix}_safe_margin"], errors="coerce").dropna().to_numpy()
        for prefix in prefixes
    ]
    fig, ax = plt.subplots(figsize=(3.6, 2.55))
    box = ax.boxplot(
        data,
        positions=np.arange(len(prefixes)),
        widths=0.46,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1F1F1F", "linewidth": 1.1},
        whiskerprops={"color": "#505050", "linewidth": 0.9},
        capprops={"color": "#505050", "linewidth": 0.9},
    )
    for patch, prefix in zip(box["boxes"], prefixes, strict=True):
        patch.set_facecolor(MODEL_COLORS[prefix])
        patch.set_alpha(0.2)
        patch.set_edgecolor(MODEL_COLORS[prefix])
        patch.set_linewidth(1.0)

    ax.axhline(0, color="#2A2A2A", linewidth=0.75, linestyle=(0, (3, 2)))
    ax.set_xticks(np.arange(len(prefixes)))
    ax.set_xticklabels([MODEL_LABELS[prefix] for prefix in prefixes])
    for tick, prefix in zip(ax.get_xticklabels(), prefixes, strict=True):
        tick.set_color(MODEL_COLORS[prefix])
    ax.set_ylabel("Safe-action logit margin")
    ax.set_title("Decision-margin distribution")
    ax.yaxis.grid(True)
    despine(ax)
    return save_figure(fig, Path(outdir) / "safe_margin_distributions")


def _format_disagreement_label(label: str) -> str:
    replacements = {
        "all_safe": "All safe",
        "all_harmful": "All harmful",
        "base_harmful_ut_safe_game_safe": "Base harmful / UT safe / GAME safe",
        "base_harmful_ut_safe_game_harmful": "Base harmful / UT safe / GAME harmful",
        "base_harmful_ut_harmful_game_safe": "Base harmful / UT harmful / GAME safe",
        "base_safe_ut_harmful_game_safe": "Base safe / UT harmful / GAME safe",
        "base_safe_ut_safe_game_harmful": "Base safe / UT safe / GAME harmful",
        "base_safe_trained_harmful": "Base safe / trained harmful",
    }
    return fill(replacements.get(label, label.replace("_", " ")), width=31)


def plot_disagreement_counts(
    disagreement_df: pd.DataFrame,
    outdir: str | Path,
    *,
    font_family: str = "serif",
) -> list[Path]:
    apply_paper_style(font_family=font_family)
    table = disagreement_df.sort_values("count", ascending=True).copy()
    fig_height = max(2.6, 0.34 * len(table) + 0.7)
    fig, ax = plt.subplots(figsize=(5.2, fig_height))
    y = np.arange(len(table))
    ax.barh(y, table["count"], color="#6E6E6E", edgecolor="#303030", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([_format_disagreement_label(v) for v in table["disagreement_type"]])
    ax.set_xlabel("Examples")
    ax.set_title("Disagreement-set counts")
    ax.xaxis.grid(True)
    for idx, (_, row) in enumerate(table.iterrows()):
        ax.text(
            row["count"],
            idx,
            f"  {int(row['count'])} ({100 * row['percent']:.1f}%)",
            va="center",
            ha="left",
            fontsize=7.5,
            color="#303030",
        )
    despine(ax)
    return save_figure(fig, Path(outdir) / "disagreement_counts")


def plot_paired_improvements(
    paired_df: pd.DataFrame,
    outdir: str | Path,
    *,
    font_family: str = "serif",
) -> list[Path]:
    apply_paper_style(font_family=font_family)
    if paired_df.empty:
        return []
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    x = np.arange(len(paired_df))
    for idx, row in paired_df.reset_index(drop=True).iterrows():
        prefix = row["model"]
        mean = row["mean_paired_improvement"]
        low = row["ci_low"]
        high = row["ci_high"]
        ax.errorbar(
            x[idx],
            mean,
            yerr=np.array([[mean - low], [high - mean]]),
            color=MODEL_COLORS[prefix],
            marker=MODEL_MARKERS[prefix],
            markersize=5,
            capsize=3,
            linewidth=1.2,
        )
    ax.axhline(0, color="#2A2A2A", linewidth=0.75, linestyle=(0, (3, 2)))
    ax.set_xticks(x)
    ax.set_xticklabels(list(paired_df["model_label"]))
    for tick, prefix in zip(ax.get_xticklabels(), paired_df["model"], strict=True):
        tick.set_color(MODEL_COLORS[prefix])
    ax.set_ylabel("Paired improvement vs Base")
    ax.set_title("Training effect on safe choices")
    ax.yaxis.grid(True)
    despine(ax)
    return save_figure(fig, Path(outdir) / "paired_improvements")


def plot_game_type_safe_rates(
    game_metrics_df: pd.DataFrame,
    outdir: str | Path,
    *,
    font_family: str = "serif",
) -> list[Path]:
    apply_paper_style(font_family=font_family)
    if game_metrics_df.empty:
        return []

    pivot = game_metrics_df.pivot(index="game_type", columns="model", values="safe_rate")
    prefixes = [prefix for prefix in MODEL_ORDER if prefix in pivot.columns]
    pivot = pivot[prefixes].sort_index()
    fig_height = max(2.6, 0.28 * len(pivot) + 0.9)
    fig, ax = plt.subplots(figsize=(3.9, fig_height))
    matrix = pivot.to_numpy(dtype=float)
    image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="Greys")
    ax.set_xticks(np.arange(len(prefixes)))
    ax.set_xticklabels([MODEL_LABELS[prefix] for prefix in prefixes])
    for tick, prefix in zip(ax.get_xticklabels(), prefixes, strict=True):
        tick.set_color(MODEL_COLORS[prefix])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([fill(str(label), width=28) for label in pivot.index])
    ax.set_title("Safe-action rate by game type")

    if matrix.size <= 90:
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                if np.isnan(value):
                    continue
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > 0.55 else "#1F1F1F",
                )
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Safe-action rate")
    return save_figure(fig, Path(outdir) / "game_type_safe_rates")


def plot_behavior_overview(
    behavior_df: pd.DataFrame,
    disagreement_df: pd.DataFrame,
    paired_df: pd.DataFrame,
    outdir: str | Path,
    *,
    font_family: str = "serif",
) -> list[Path]:
    """Create a compact four-panel overview suitable for draft papers."""

    apply_paper_style(font_family=font_family)
    prefixes = _available_prefixes(behavior_df)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1))

    ax = axes[0, 0]
    x = np.arange(len(prefixes))
    for idx, prefix in enumerate(prefixes):
        values = coerce_bool_series(behavior_df[f"{prefix}_safe"]).astype(float).to_numpy()
        mean = float(np.nanmean(values))
        low, high = bootstrap_ci(values)
        ax.errorbar(
            x[idx],
            mean,
            yerr=np.array([[mean - low], [high - mean]]),
            color=MODEL_COLORS[prefix],
            marker=MODEL_MARKERS[prefix],
            markersize=5,
            capsize=3,
            linewidth=1.1,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[prefix] for prefix in prefixes])
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Safe-action rate")
    ax.yaxis.grid(True)
    despine(ax)
    panel_label(ax, "A")

    ax = axes[0, 1]
    data = [
        pd.to_numeric(behavior_df[f"{prefix}_safe_margin"], errors="coerce").dropna().to_numpy()
        for prefix in prefixes
    ]
    box = ax.boxplot(
        data,
        positions=np.arange(len(prefixes)),
        widths=0.46,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1F1F1F", "linewidth": 1.0},
    )
    for patch, prefix in zip(box["boxes"], prefixes, strict=True):
        patch.set_facecolor(MODEL_COLORS[prefix])
        patch.set_alpha(0.2)
        patch.set_edgecolor(MODEL_COLORS[prefix])
    ax.axhline(0, color="#2A2A2A", linewidth=0.75, linestyle=(0, (3, 2)))
    ax.set_xticks(np.arange(len(prefixes)))
    ax.set_xticklabels([MODEL_LABELS[prefix] for prefix in prefixes])
    ax.set_ylabel("Safe margin")
    ax.yaxis.grid(True)
    despine(ax)
    panel_label(ax, "B")

    ax = axes[1, 0]
    if paired_df.empty:
        ax.axis("off")
    else:
        x = np.arange(len(paired_df))
        for idx, row in paired_df.reset_index(drop=True).iterrows():
            prefix = row["model"]
            mean = row["mean_paired_improvement"]
            low = row["ci_low"]
            high = row["ci_high"]
            ax.errorbar(
                x[idx],
                mean,
                yerr=np.array([[mean - low], [high - mean]]),
                color=MODEL_COLORS[prefix],
                marker=MODEL_MARKERS[prefix],
                markersize=5,
                capsize=3,
                linewidth=1.1,
            )
        ax.axhline(0, color="#2A2A2A", linewidth=0.75, linestyle=(0, (3, 2)))
        ax.set_xticks(x)
        ax.set_xticklabels(list(paired_df["model_label"]))
        ax.set_ylabel("Improvement vs Base")
        ax.yaxis.grid(True)
        despine(ax)
    panel_label(ax, "C")

    ax = axes[1, 1]
    top = disagreement_df.sort_values("count", ascending=False).head(5).sort_values(
        "count", ascending=True
    )
    y = np.arange(len(top))
    ax.barh(y, top["count"], color="#6E6E6E", edgecolor="#303030", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([_format_disagreement_label(v) for v in top["disagreement_type"]])
    ax.set_xlabel("Examples")
    ax.xaxis.grid(True)
    despine(ax)
    panel_label(ax, "D")

    fig.subplots_adjust(wspace=0.45, hspace=0.45)
    return save_figure(fig, Path(outdir) / "behavior_overview")
