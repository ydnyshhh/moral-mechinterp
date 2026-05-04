from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from moral_mechinterp.config import EvalConfig, load_eval_config
from moral_mechinterp.constants import MODEL_LABELS
from moral_mechinterp.data import NormalizedExample
from moral_mechinterp.logit_lens import (
    compute_batch_layer_margins,
    get_final_norm,
    get_lm_head,
    ordered_layer_margin_columns,
    plot_layer_margin_summary,
    summarize_layer_margins,
)
from moral_mechinterp.models import load_tokenizer_and_model, unload_model
from moral_mechinterp.prompts import build_ab_prompt
from moral_mechinterp.scoring import apply_chat_template_if_needed, resolve_score_token_ids
from moral_mechinterp.utils import batched, set_seed

REQUIRED_COLUMNS = {"id", "game_type", "scenario", "option_a", "option_b", "safe_label"}
HIDDEN_STATE_INTERPRETATION = (
    "Layer indices are the hidden_state tuple indices returned by Hugging Face models: "
    "layer 0 is the embedding output, and layers 1..L are transformer block outputs. "
    "For each intermediate residual stream, this script applies the model's final "
    "normalization layer, typically RMSNorm for Qwen/LLaMA-style models, before "
    "projecting through the LM head. The final hidden-state entry is assumed to "
    "already be final-normalized for Qwen/LLaMA-style HF decoder models, so final_norm "
    "is not applied again at the last layer. The last-layer logit-lens margin should "
    "therefore closely match the behavioral final-logit safe margin when prompts and "
    "A/B token ids match. This is a normed logit lens, not a tuned lens."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute layerwise logit-lens safe-action margins for Base/UT/GAME models. "
            "This is a diagnostic of decision-evidence trajectories, not a causal "
            "intervention; it is useful when adapter differences are small margin shifts "
            "rather than strong binary flips."
        )
    )
    parser.add_argument("--subset-csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", default="base,ut,game")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def parse_model_keys(raw: str, config: EvalConfig) -> list[str]:
    keys = [key.strip() for key in raw.split(",") if key.strip()]
    if not keys:
        raise ValueError("--models must contain at least one model key")
    unknown = [key for key in keys if key not in config.models]
    if unknown:
        raise ValueError(
            f"Unknown model key(s): {unknown}; available keys: {sorted(config.models)}"
        )
    return keys


def load_subset(path: Path, *, max_examples: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Subset CSV is missing required columns: {sorted(missing)}")
    bad_labels = sorted(set(df["safe_label"].astype(str)) - {"A", "B"})
    if bad_labels:
        raise ValueError(f"safe_label must be A or B; found invalid labels: {bad_labels}")
    if max_examples is not None:
        df = df.head(max_examples).copy()
    return df


def row_to_example(row: pd.Series) -> NormalizedExample:
    return NormalizedExample(
        id=str(row["id"]),
        game_type=str(row["game_type"]),
        scenario=str(row["scenario"]),
        option_a=str(row["option_a"]),
        option_b=str(row["option_b"]),
        safe_label=str(row["safe_label"]),
    )


def maybe_value(row: pd.Series, column: str) -> Any | None:
    if column not in row.index:
        return None
    value = row[column]
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    return value


def build_base_record(
    *,
    subset_name: str,
    row: pd.Series,
    model_key: str,
    safe_token: str,
    harmful_token: str,
) -> dict[str, Any]:
    return {
        "subset_name": subset_name,
        "id": row["id"],
        "game_type": row["game_type"],
        "model": model_key,
        "model_label": MODEL_LABELS.get(model_key, model_key),
        "safe_label": row["safe_label"],
        "safe_token": safe_token,
        "harmful_token": harmful_token,
        "final_behavior_safe_margin": maybe_value(row, f"{model_key}_safe_margin"),
        "final_behavior_choice": maybe_value(row, f"{model_key}_choice"),
        "final_behavior_safe": maybe_value(row, f"{model_key}_safe"),
    }


def save_outputs(
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
    config: EvalConfig,
    write_summary: bool = True,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_df = pd.DataFrame.from_records(records)
    if not layer_df.empty:
        ordered = ordered_layer_margin_columns()
        remaining = [column for column in layer_df.columns if column not in ordered]
        layer_df = layer_df[ordered + remaining]
    layer_df.to_csv(output_dir / "layer_margins.csv", index=False)
    if write_summary and not layer_df.empty:
        summary_df = summarize_layer_margins(layer_df, seed=config.seed)
        summary_df.to_csv(output_dir / "layer_margin_summary.csv", index=False)
    elif write_summary:
        summary_df = pd.DataFrame()
        summary_df.to_csv(output_dir / "layer_margin_summary.csv", index=False)
    else:
        summary_df = pd.DataFrame()
    return summary_df


def write_metadata(
    *,
    output_dir: Path,
    subset_csv: Path,
    subset_name: str,
    n_examples: int,
    model_keys: list[str],
    config_path: Path,
    config: EvalConfig,
) -> None:
    metadata = {
        "subset_csv": str(subset_csv),
        "subset_name": subset_name,
        "n_examples": n_examples,
        "models": model_keys,
        "config_path": str(config_path),
        "torch_dtype": config.torch_dtype,
        "device_map": config.device_map,
        "load_in_4bit": config.load_in_4bit,
        "load_in_8bit": config.load_in_8bit,
        "use_chat_template": config.use_chat_template,
        "score_tokens": config.score_tokens,
        "allow_multitoken_score_labels": config.allow_multitoken_score_labels,
        "hidden_state_interpretation": HIDDEN_STATE_INTERPRETATION,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def score_model(
    *,
    model_key: str,
    subset_name: str,
    subset_df: pd.DataFrame,
    config: EvalConfig,
    batch_size: int,
    save_every: int,
    output_dir: Path,
    records: list[dict[str, Any]],
) -> None:
    model_id = config.models[model_key]
    print(f"Loading {MODEL_LABELS.get(model_key, model_key)}: {model_id}")
    try:
        tokenizer, model = load_tokenizer_and_model(
            model_id,
            torch_dtype=config.torch_dtype,
            device_map=config.device_map,
            load_in_4bit=config.load_in_4bit,
            load_in_8bit=config.load_in_8bit,
            trust_remote_code=config.trust_remote_code,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load model {model_key} ({model_id})") from exc

    try:
        token_ids = resolve_score_token_ids(
            tokenizer,
            config.score_tokens,
            allow_multitoken_score_labels=config.allow_multitoken_score_labels,
        )
        lm_head = get_lm_head(model)
        final_norm = get_final_norm(model)
        model.eval()

        row_indices = list(range(len(subset_df)))
        progress = tqdm(
            list(batched(row_indices, batch_size)),
            desc=f"Logit lens {MODEL_LABELS.get(model_key, model_key)}",
            unit="batch",
        )
        scored_examples = 0
        for batch_indices in progress:
            batch_rows = [subset_df.iloc[idx] for idx in batch_indices]
            examples = [row_to_example(row) for row in batch_rows]
            prompts = [
                apply_chat_template_if_needed(
                    tokenizer,
                    build_ab_prompt(example),
                    use_chat_template=config.use_chat_template,
                )
                for example in examples
            ]
            safe_labels = [example.safe_label for example in examples]
            batch_layer_outputs = compute_batch_layer_margins(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                safe_labels=safe_labels,
                token_ids=token_ids,
                final_norm=final_norm,
                lm_head=lm_head,
            )

            for row, layer_outputs in zip(batch_rows, batch_layer_outputs, strict=True):
                safe_label = str(row["safe_label"])
                harmful_label = "B" if safe_label == "A" else "A"
                base_record = build_base_record(
                    subset_name=subset_name,
                    row=row,
                    model_key=model_key,
                    safe_token=config.score_tokens[safe_label],
                    harmful_token=config.score_tokens[harmful_label],
                )
                for layer_record in layer_outputs:
                    records.append({**base_record, **layer_record})

                scored_examples += 1
                if save_every > 0 and scored_examples % save_every == 0:
                    save_outputs(
                        records,
                        output_dir=output_dir,
                        config=config,
                        write_summary=False,
                    )

        save_outputs(records, output_dir=output_dir, config=config)
    finally:
        unload_model(model)
        del model, tokenizer


def main() -> None:
    args = parse_args()
    config = load_eval_config(args.config)
    set_seed(config.seed)

    subset_df = load_subset(args.subset_csv, max_examples=args.max_examples)
    subset_name = args.subset_csv.stem
    model_keys = parse_model_keys(args.models, config)
    batch_size = args.batch_size if args.batch_size is not None else config.batch_size
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(
        output_dir=args.output_dir,
        subset_csv=args.subset_csv,
        subset_name=subset_name,
        n_examples=len(subset_df),
        model_keys=model_keys,
        config_path=args.config,
        config=config,
    )

    records: list[dict[str, Any]] = []
    for model_key in model_keys:
        score_model(
            model_key=model_key,
            subset_name=subset_name,
            subset_df=subset_df,
            config=config,
            batch_size=batch_size,
            save_every=args.save_every,
            output_dir=args.output_dir,
            records=records,
        )

    summary_df = save_outputs(records, output_dir=args.output_dir, config=config)
    if not args.no_plot and not summary_df.empty:
        plot_layer_margin_summary(
            summary_df,
            subset_name=subset_name,
            output_dir=str(args.output_dir),
            font_family=config.plot_font_family,
        )

    print(f"Wrote layer logit-lens outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
