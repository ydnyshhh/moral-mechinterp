"""CLI for summary tables and paper-style behavior figures."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from moral_mechinterp.config import load_eval_config
from moral_mechinterp.io import ensure_dir, read_behavior_csv, write_dataframe
from moral_mechinterp.metrics import (
    disagreement_counts,
    game_type_metrics,
    overall_metrics,
    paired_improvements,
    strong_flip_counts,
)
from moral_mechinterp.plotting import (
    plot_behavior_overview,
    plot_disagreement_counts,
    plot_game_type_safe_rates,
    plot_model_safe_rates,
    plot_paired_improvements,
    plot_safe_margin_distributions,
)

app = typer.Typer(help="Summarize behavior CSVs and create polished static figures.")
console = Console()


@app.command()
def main(
    behavior_csv: Annotated[
        Path,
        typer.Argument(
            exists=True,
            readable=True,
            help="Behavior CSV produced by evaluate_behavior.py.",
        ),
    ] = Path("outputs/behavior/model_choices.csv"),
    config: Annotated[Path, typer.Option("--config", "-c", help="Eval YAML.")] = Path(
        "configs/eval.yaml"
    ),
    tables_dir: Annotated[
        Path | None,
        typer.Option("--tables-dir", help="Output directory for summary CSV files."),
    ] = None,
    figures_dir: Annotated[
        Path | None,
        typer.Option("--figures-dir", help="Output directory for PNG/PDF/SVG figures."),
    ] = None,
    bootstrap_samples: Annotated[
        int,
        typer.Option(
            "--bootstrap-samples",
            help="Bootstrap samples for paired improvement confidence intervals.",
        ),
    ] = 10_000,
) -> None:
    cfg = load_eval_config(config)
    table_out = ensure_dir(tables_dir or cfg.outputs.tables_dir)
    figure_out = ensure_dir(figures_dir or cfg.outputs.figures_dir)

    df = read_behavior_csv(behavior_csv)
    overall = overall_metrics(df)
    by_game = game_type_metrics(df)
    disagreements = disagreement_counts(df)
    strong = strong_flip_counts(df)
    paired = paired_improvements(
        df,
        n_bootstrap=bootstrap_samples,
        seed=cfg.seed,
    )

    written_tables = [
        write_dataframe(overall, table_out / "overall_metrics.csv"),
        write_dataframe(by_game, table_out / "game_type_metrics.csv"),
        write_dataframe(disagreements, table_out / "disagreement_counts.csv"),
        write_dataframe(strong, table_out / "strong_flip_counts.csv"),
        write_dataframe(paired, table_out / "paired_improvements.csv"),
    ]

    written_figures: list[Path] = []
    figure_kwargs = {"font_family": cfg.plot_font_family}
    written_figures.extend(plot_model_safe_rates(df, figure_out, **figure_kwargs))
    written_figures.extend(plot_safe_margin_distributions(df, figure_out, **figure_kwargs))
    written_figures.extend(plot_disagreement_counts(disagreements, figure_out, **figure_kwargs))
    written_figures.extend(plot_paired_improvements(paired, figure_out, **figure_kwargs))
    written_figures.extend(plot_game_type_safe_rates(by_game, figure_out, **figure_kwargs))
    written_figures.extend(
        plot_behavior_overview(df, disagreements, paired, figure_out, **figure_kwargs)
    )

    console.print("[bold green]Summary complete[/bold green]")
    console.print("Tables:")
    for path in written_tables:
        console.print(f"  {path}")
    console.print("Figures:")
    for path in written_figures:
        console.print(f"  {path}")


if __name__ == "__main__":
    app()
