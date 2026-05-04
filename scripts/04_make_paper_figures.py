from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from moral_mechinterp.constants import MODEL_COLORS, MODEL_LABELS, MODEL_MARKERS, MODEL_ORDER
from moral_mechinterp.plot_style import apply_paper_style, despine, save_figure

PANEL_SPECS = {
    "top_ut_margin_shift": {
        "figure_title": "UT-favored margin shifts",
        "table_label": "UT-favored margin shifts",
    },
    "top_game_margin_shift": {
        "figure_title": "GAME-favored margin shifts",
        "table_label": "GAME-favored margin shifts",
    },
    "ut_safe_game_harmful": {
        "figure_title": "UT-safe / GAME-harmful",
        "table_label": "UT-safe / GAME-harmful disagreements",
    },
    "game_safe_ut_harmful": {
        "figure_title": "GAME-safe / UT-harmful",
        "table_label": "GAME-safe / UT-harmful disagreements",
    },
}
PANEL_LETTERS = ("A", "B", "C", "D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create paper-ready logit-lens figures and late-layer tables from saved CSVs. "
            "This script does not run model inference."
        )
    )
    parser.add_argument("--logit-lens-dir", type=Path, default=Path("outputs/logit_lens"))
    parser.add_argument(
        "--behavior-csv",
        type=Path,
        default=Path("outputs/behavior_full/model_choices.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures_logit_lens"),
    )
    parser.add_argument(
        "--table-path",
        type=Path,
        default=Path("outputs/tables_full/logit_lens_late_layer_separation.csv"),
    )
    parser.add_argument(
        "--paper-table-path",
        type=Path,
        default=Path("outputs/tables_full/logit_lens_late_layer_paper_table.csv"),
    )
    parser.add_argument(
        "--control-output-dir",
        type=Path,
        default=Path("outputs/behavior_full/subsets"),
    )
    parser.add_argument("--late-layer-start", type=int, default=21)
    parser.add_argument("--late-layer-end", type=int, default=31)
    parser.add_argument("--control-n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--font-family", default="serif", choices=["serif", "sans"])
    parser.add_argument("--no-control-subsets", action="store_true")
    return parser.parse_args()


def read_summary(logit_lens_dir: Path, subset_name: str) -> pd.DataFrame:
    path = logit_lens_dir / subset_name / "layer_margin_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {path}")
    return pd.read_csv(path)


def read_layer_margins(logit_lens_dir: Path, subset_name: str) -> pd.DataFrame:
    path = logit_lens_dir / subset_name / "layer_margins.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing layer margins CSV: {path}")
    return pd.read_csv(path)


def load_all_summaries(logit_lens_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        subset_name: read_summary(logit_lens_dir, subset_name)
        for subset_name in PANEL_SPECS
    }


def panel_n_examples(summary: pd.DataFrame) -> int | None:
    if "n" not in summary.columns:
        return None
    values = pd.to_numeric(summary["n"], errors="coerce").dropna()
    if values.empty:
        return None
    return int(values.max())


def plot_combined_logit_lens(
    summaries: dict[str, pd.DataFrame],
    *,
    output_dir: Path,
    font_family: str,
) -> list[Path]:
    apply_paper_style(font_family=font_family)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_values: list[float] = []
    for summary in summaries.values():
        for column in ("ci_low", "ci_high", "mean_safe_margin"):
            all_values.extend(pd.to_numeric(summary[column], errors="coerce").dropna().tolist())
    y_low = float(np.nanmin(all_values))
    y_high = float(np.nanmax(all_values))
    pad = 0.08 * max(1e-6, y_high - y_low)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.25, 5.35),
        sharex=False,
        sharey=False,
    )
    axes_flat = axes.ravel()

    handles = []
    labels = []
    for panel_idx, (subset_name, panel_spec) in enumerate(PANEL_SPECS.items()):
        ax = axes_flat[panel_idx]
        summary = summaries[subset_name]
        for model_key in MODEL_ORDER:
            model_df = summary[summary["model"] == model_key].sort_values("layer")
            if model_df.empty:
                continue
            x = model_df["layer"].to_numpy(dtype=float)
            y = model_df["mean_safe_margin"].to_numpy(dtype=float)
            low = model_df["ci_low"].to_numpy(dtype=float)
            high = model_df["ci_high"].to_numpy(dtype=float)
            (line,) = ax.plot(
                x,
                y,
                color=MODEL_COLORS[model_key],
                marker=MODEL_MARKERS[model_key],
                markevery=max(1, len(x) // 6),
                linewidth=1.25,
                markersize=3.5,
                label=MODEL_LABELS[model_key],
            )
            ax.fill_between(
                x,
                low,
                high,
                color=MODEL_COLORS[model_key],
                alpha=0.14,
                linewidth=0,
            )
            if panel_idx == 0:
                handles.append(line)
                labels.append(MODEL_LABELS[model_key])

        ax.axhline(0, color="#2A2A2A", linewidth=0.75, linestyle=(0, (3, 2)))
        title = f"{PANEL_LETTERS[panel_idx]}. {panel_spec['figure_title']}"
        n_examples = panel_n_examples(summary)
        if n_examples is not None:
            title = f"{title} (n={n_examples})"
        ax.set_title(title)
        ax.set_ylim(y_low - pad, y_high + pad)
        ax.set_xlim(-1.5, 33.5)
        ax.set_xticks([0, 10, 20, 30])
        ax.yaxis.grid(True)
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelleft=True)
        despine(ax)

    for ax in axes[1, :]:
        ax.set_xlabel("Layer")

    fig.supylabel("Mean safe-action logit-lens margin", x=0.015, fontsize=9)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=False,
    )
    fig.text(
        0.5,
        0.025,
        (
            "Panels C–D show categorical disagreement subsets. Layer 0 is the embedding output; "
            "layer 32 recovers the behavioral A/B margin."
        ),
        ha="center",
        va="bottom",
        fontsize=8.2,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        top=0.885,
        bottom=0.13,
        wspace=0.24,
        hspace=0.4,
    )
    return save_figure(fig, output_dir / "logit_lens_combined_2x2")


def dominant_game_type(layer_df: pd.DataFrame) -> str:
    examples = layer_df[["id", "game_type"]].drop_duplicates()
    if examples.empty:
        return "unknown"
    return str(examples["game_type"].value_counts().idxmax())


def make_late_layer_table(
    *,
    logit_lens_dir: Path,
    late_layer_start: int,
    late_layer_end: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subset_name, panel_spec in PANEL_SPECS.items():
        layer_df = read_layer_margins(logit_lens_dir, subset_name)
        late = layer_df[
            (layer_df["layer"] >= late_layer_start)
            & (layer_df["layer"] <= late_layer_end)
        ]
        by_model = late.groupby("model")["safe_margin"].mean()
        base = float(by_model.get("base", np.nan))
        ut = float(by_model.get("ut", np.nan))
        game = float(by_model.get("game", np.nan))
        model_margins = {"Base": base, "UT": ut, "GAME": game}
        winner = max(model_margins, key=lambda key: model_margins[key])
        rows.append(
            {
                "subset": panel_spec["table_label"],
                "subset_name": subset_name,
                "n": int(layer_df["id"].nunique()),
                "dominant_game_type": dominant_game_type(layer_df),
                "base_late_margin": base,
                "ut_late_margin": ut,
                "game_late_margin": game,
                "ut_minus_base": ut - base,
                "game_minus_base": game - base,
                "winner": winner,
                "late_layers": f"{late_layer_start}-{late_layer_end}",
            }
        )
    return pd.DataFrame(rows)


def make_paper_late_layer_table(detailed_table: pd.DataFrame) -> pd.DataFrame:
    paper = detailed_table[
        ["subset_name", "n", "ut_minus_base", "game_minus_base", "winner", "late_layers"]
    ].copy()
    paper["subset"] = paper["subset_name"].map(
        {
            subset_name: str(panel_spec["figure_title"])
            for subset_name, panel_spec in PANEL_SPECS.items()
        }
    )
    paper = paper[
        ["subset", "n", "ut_minus_base", "game_minus_base", "winner", "late_layers"]
    ]
    return paper.rename(
        columns={
            "ut_minus_base": "ut_minus_base_late",
            "game_minus_base": "game_minus_base_late",
        }
    )


def write_markdown_table(table: pd.DataFrame, path: Path) -> None:
    lines = [
        "| Subset | n | UT-Base late | GAME-Base late | Winner | Layers |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            "| "
            f"{row.subset} | "
            f"{row.n} | "
            f"{row.ut_minus_base_late:+.3f} | "
            f"{row.game_minus_base_late:+.3f} | "
            f"{row.winner} | "
            f"{row.late_layers} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sample_control_subsets(
    *,
    behavior_csv: Path,
    output_dir: Path,
    n: int,
    seed: int,
) -> list[Path]:
    behavior = pd.read_csv(behavior_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        f"random_pd_{n}.csv": "Prisoner's Dilemma",
        f"random_chicken_{n}.csv": "Chicken",
    }
    written: list[Path] = []
    for filename, game_type in specs.items():
        game_df = behavior[behavior["game_type"] == game_type].copy()
        if len(game_df) < n:
            raise ValueError(f"Only {len(game_df)} rows available for {game_type}; need {n}.")
        sampled = game_df.sample(n=n, random_state=seed).sort_index()
        path = output_dir / filename
        sampled.to_csv(path, index=False)
        written.append(path)
    return written


def main() -> None:
    args = parse_args()
    summaries = load_all_summaries(args.logit_lens_dir)
    figure_paths = plot_combined_logit_lens(
        summaries,
        output_dir=args.output_dir,
        font_family=args.font_family,
    )

    table = make_late_layer_table(
        logit_lens_dir=args.logit_lens_dir,
        late_layer_start=args.late_layer_start,
        late_layer_end=args.late_layer_end,
    )
    args.table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.table_path, index=False)

    paper_table = make_paper_late_layer_table(table)
    args.paper_table_path.parent.mkdir(parents=True, exist_ok=True)
    paper_table.to_csv(args.paper_table_path, index=False)
    write_markdown_table(paper_table, args.paper_table_path.with_suffix(".md"))

    control_paths: list[Path] = []
    if not args.no_control_subsets:
        control_paths = sample_control_subsets(
            behavior_csv=args.behavior_csv,
            output_dir=args.control_output_dir,
            n=args.control_n,
            seed=args.seed,
        )

    print("Wrote combined logit-lens figure:")
    for path in figure_paths:
        print(f"  {path}")
    print(f"Wrote late-layer separation table: {args.table_path}")
    print(f"Wrote paper late-layer table: {args.paper_table_path}")
    if control_paths:
        print("Wrote random control subset CSVs:")
        for path in control_paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
