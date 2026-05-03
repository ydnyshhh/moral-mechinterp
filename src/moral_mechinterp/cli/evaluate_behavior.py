"""CLI for behavioral A/B logit evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console

from moral_mechinterp.config import EvalConfig, load_eval_config
from moral_mechinterp.constants import MODEL_LABELS, MODEL_ORDER
from moral_mechinterp.data import load_jsonl_examples
from moral_mechinterp.disagreement import finalize_behavior_rows
from moral_mechinterp.io import ensure_dir, write_csv_records, write_jsonl_records
from moral_mechinterp.metrics import overall_metrics
from moral_mechinterp.models import load_tokenizer_and_model, unload_model
from moral_mechinterp.scoring import ABLogitScore, score_examples_for_model
from moral_mechinterp.utils import set_seed

app = typer.Typer(help="Evaluate A/B moral/game-theoretic behavior with next-token logits.")
console = Console()


def _selected_model_keys(config: EvalConfig, requested: str | None) -> list[str]:
    if requested is None:
        keys = [key for key in MODEL_ORDER if key in config.models]
        extras = [key for key in config.models if key not in keys]
        return keys + extras
    keys = [key.strip() for key in requested.split(",") if key.strip()]
    missing = [key for key in keys if key not in config.models]
    if missing:
        raise typer.BadParameter(f"Unknown model key(s): {', '.join(missing)}")
    return keys


def _score_to_row(prefix: str, score: ABLogitScore) -> dict[str, object]:
    return {
        f"{prefix}_choice": score.choice,
        f"{prefix}_safe": score.safe,
        f"{prefix}_logit_A": score.logit_A,
        f"{prefix}_logit_B": score.logit_B,
        f"{prefix}_safe_margin": score.safe_margin,
    }


def _save_behavior(rows: list[dict[str, object]], output_dir: Path) -> tuple[Path, Path]:
    csv_path = write_csv_records(rows, output_dir / "model_choices.csv")
    jsonl_path = write_jsonl_records(rows, output_dir / "model_choices.jsonl")
    return csv_path, jsonl_path


def _init_wandb(config: EvalConfig, data_path: Path):
    if not config.wandb.enabled:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B tracking is enabled, but wandb is not installed. "
            "Install it with `uv sync --extra tracking` or set wandb.enabled=false."
        ) from exc

    return wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.wandb.run_name,
        config={**config.to_dict(), "data_path": str(data_path)},
    )


@app.command()
def main(
    data_path: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Input JSONL data file."),
    ],
    config: Annotated[Path, typer.Option("--config", "-c", help="Eval YAML.")] = Path(
        "configs/eval.yaml"
    ),
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            help="Behavior output directory. Defaults to config outputs.behavior_dir.",
        ),
    ] = None,
    models: Annotated[
        str | None,
        typer.Option(
            "--models",
            help="Comma-separated model keys to evaluate, e.g. base,ut,game. Defaults to all.",
        ),
    ] = None,
) -> None:
    cfg = load_eval_config(config)
    set_seed(cfg.seed)

    behavior_dir = ensure_dir(output_dir or cfg.outputs.behavior_dir)
    examples = load_jsonl_examples(
        data_path,
        max_examples=cfg.max_examples,
        shuffle=cfg.shuffle,
        seed=cfg.seed,
    )
    rows: list[dict[str, object]] = [example.to_record() for example in examples]
    selected_keys = _selected_model_keys(cfg, models)

    console.print(f"[bold]Loaded {len(examples)} examples[/bold] from {data_path}")
    console.print(f"Scoring models: {', '.join(MODEL_LABELS.get(k, k) for k in selected_keys)}")

    run = _init_wandb(cfg, data_path)

    try:
        for prefix in selected_keys:
            model_name = cfg.models[prefix]
            label = MODEL_LABELS.get(prefix, prefix)
            console.rule(f"Loading {label}")
            tokenizer, model = load_tokenizer_and_model(
                model_name,
                torch_dtype=cfg.torch_dtype,
                device_map=cfg.device_map,
                load_in_4bit=cfg.load_in_4bit,
                load_in_8bit=cfg.load_in_8bit,
                trust_remote_code=cfg.trust_remote_code,
            )

            scored_count = 0

            def checkpoint(
                example_idx: int,
                score: ABLogitScore,
                model_prefix: str = prefix,
            ) -> None:
                nonlocal scored_count
                rows[example_idx].update(_score_to_row(model_prefix, score))
                scored_count += 1
                if cfg.save_every > 0 and scored_count % cfg.save_every == 0:
                    _save_behavior(rows, behavior_dir)

            scores = score_examples_for_model(
                model=model,
                tokenizer=tokenizer,
                examples=examples,
                config=cfg,
                description=f"Scoring {label}",
                checkpoint_callback=checkpoint,
            )
            for idx, score in enumerate(scores):
                rows[idx].update(_score_to_row(prefix, score))

            model_df = overall_metrics(pd.DataFrame(rows), prefixes=[prefix])
            if not model_df.empty:
                metrics_row = model_df.iloc[0].to_dict()
                console.print(
                    f"{label}: safe_rate={metrics_row['safe_rate']:.3f}, "
                    f"mean_margin={metrics_row['mean_safe_margin']:.3f}"
                )
                if run is not None:
                    run.log(
                        {
                            f"{prefix}/safe_rate": metrics_row["safe_rate"],
                            f"{prefix}/mean_safe_margin": metrics_row["mean_safe_margin"],
                            f"{prefix}/median_safe_margin": metrics_row["median_safe_margin"],
                        }
                    )

            unload_model(model)
            del model, tokenizer
            _save_behavior(rows, behavior_dir)

        finalize_behavior_rows(rows, tau=cfg.margin_threshold_for_strong_flips)
        csv_path, jsonl_path = _save_behavior(rows, behavior_dir)
        console.print(f"[green]Wrote[/green] {csv_path}")
        console.print(f"[green]Wrote[/green] {jsonl_path}")

        if run is not None:
            run.log({"n_examples": len(rows)})
            run.save(str(csv_path))
            run.save(str(jsonl_path))
            run.finish()
    except Exception:
        if run is not None:
            run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    app()
