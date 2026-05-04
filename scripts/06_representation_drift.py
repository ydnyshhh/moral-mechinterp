from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from moral_mechinterp.constants import MODEL_COLORS, MODEL_LABELS, MODEL_MARKERS

if TYPE_CHECKING:
    import pandas as pd
    import torch


REQUIRED_COLUMNS = {"id", "game_type", "scenario", "option_a", "option_b", "safe_label"}
PAIR_SPECS = (
    ("base", "ut", "Base–UT"),
    ("base", "game", "Base–GAME"),
    ("ut", "game", "UT–GAME"),
)
PAIR_COLORS = {
    "Base–UT": MODEL_COLORS["ut"],
    "Base–GAME": MODEL_COLORS["game"],
    "UT–GAME": "#555555",
}
PAIR_MARKERS = {
    "Base–UT": MODEL_MARKERS["ut"],
    "Base–GAME": MODEL_MARKERS["game"],
    "UT–GAME": "D",
}
HIDDEN_STATE_INTERPRETATION = (
    "Layer indices are the hidden_state tuple indices returned by Hugging Face models: "
    "layer 0 is the embedding output, and layers 1..L are transformer block outputs. "
    "This experiment stores only the final prompt-token hidden state for each layer."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract final-token hidden states and compute layerwise pairwise cosine "
            "representation drift across Base, UT, and GAME. This script does not modify "
            "behavioral evaluation or logit-lens computation."
        )
    )
    parser.add_argument("--subset-csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", default="base,ut,game")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--layers",
        default=None,
        help="Optional comma-separated layer list and/or inclusive ranges, e.g. 0,8,21-31.",
    )
    parser.add_argument(
        "--normalize-hidden",
        action="store_true",
        help=(
            "Optionally L2-normalize hidden states before saving. Off by default because "
            "cosine_similarity already normalizes internally."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_pandas():
    import pandas as pd

    return pd


def load_torch():
    import torch

    return torch


def parse_model_keys(raw: str, config: Any) -> list[str]:
    keys = [key.strip() for key in raw.split(",") if key.strip()]
    if not keys:
        raise ValueError("--models must contain at least one model key")
    unknown = [key for key in keys if key not in config.models]
    if unknown:
        raise ValueError(
            f"Unknown model key(s): {unknown}; available keys: {sorted(config.models)}"
        )
    return keys


def load_subset(path: Path, *, max_examples: int | None) -> pd.DataFrame:
    pd = load_pandas()
    if not path.exists():
        raise FileNotFoundError(f"Missing subset CSV: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Subset CSV is missing required columns: {sorted(missing)}")
    df["safe_label"] = df["safe_label"].astype(str).str.strip().str.upper()
    bad_labels = sorted(set(df["safe_label"]) - {"A", "B"})
    if bad_labels:
        raise ValueError(f"safe_label must be A or B; found invalid labels: {bad_labels}")
    if max_examples is not None:
        df = df.head(max_examples).copy()
    return df


def row_to_example(row: pd.Series) -> Any:
    from moral_mechinterp.data import NormalizedExample

    return NormalizedExample(
        id=str(row["id"]),
        game_type=str(row["game_type"]),
        scenario=str(row["scenario"]),
        option_a=str(row["option_a"]),
        option_b=str(row["option_b"]),
        safe_label=str(row["safe_label"]),
    )


def parse_layer_selection(raw: str | None, num_layers: int) -> list[int]:
    if raw is None:
        return list(range(num_layers))

    selected: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        separator = "-" if "-" in item else ":" if ":" in item else None
        if separator is None:
            selected.add(int(item))
            continue
        start_text, end_text = item.split(separator, maxsplit=1)
        start = int(start_text)
        end = int(end_text)
        if end < start:
            raise ValueError(f"Layer range must be increasing, got {item!r}")
        selected.update(range(start, end + 1))

    if not selected:
        raise ValueError("--layers did not contain any layer indices")
    bad_layers = [layer for layer in selected if layer < 0 or layer >= num_layers]
    if bad_layers:
        raise ValueError(
            f"Layer indices out of range for {num_layers} hidden states: {bad_layers}"
        )
    return sorted(selected)


def infer_input_device(model: Any) -> Any:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return getattr(model, "device", "cpu")


def torch_load(path: Path) -> dict[str, Any]:
    torch = load_torch()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def hidden_state_path(output_dir: Path, model_key: str) -> Path:
    return output_dir / f"hidden_states_{model_key}.pt"


def validate_hidden_payload(
    payload: dict[str, Any],
    *,
    model_key: str,
    expected_example_ids: list[str],
    expected_layers: list[int] | None,
) -> None:
    if payload.get("model") != model_key:
        raise ValueError(f"Hidden-state file model mismatch: expected {model_key}")
    example_ids = list(payload.get("example_ids") or [])
    if example_ids != expected_example_ids[: len(example_ids)]:
        raise ValueError("Existing hidden-state checkpoint example_ids do not match subset order.")
    if expected_layers is not None and list(payload.get("layers") or []) != expected_layers:
        raise ValueError("Existing hidden-state checkpoint layers do not match --layers.")


def save_hidden_payload(
    path: Path,
    *,
    model_key: str,
    example_ids: list[str],
    layers: list[int],
    hidden_states: torch.Tensor,
    subset_name: str,
    complete: bool,
    normalize_hidden: bool,
) -> None:
    torch = load_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model_key,
            "model_label": MODEL_LABELS.get(model_key, model_key),
            "example_ids": example_ids,
            "layers": layers,
            "hidden_states": hidden_states.cpu().float(),
            "subset_name": subset_name,
            "complete": complete,
            "normalize_hidden": normalize_hidden,
            "hidden_state_interpretation": HIDDEN_STATE_INTERPRETATION,
        },
        path,
    )


def prepare_prompts(
    *,
    tokenizer: Any,
    rows: list[pd.Series],
    use_chat_template: bool,
) -> list[str]:
    from moral_mechinterp.prompts import build_ab_prompt
    from moral_mechinterp.scoring import apply_chat_template_if_needed

    prompts: list[str] = []
    for row in rows:
        example = row_to_example(row)
        prompt = build_ab_prompt(example)
        prompts.append(
            apply_chat_template_if_needed(
                tokenizer,
                prompt,
                use_chat_template=use_chat_template,
            )
        )
    return prompts


def extract_batch_hidden_states(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    layer_selection: list[int] | None,
    layer_arg: str | None,
    normalize_hidden: bool,
) -> tuple[torch.Tensor, list[int]]:
    torch = load_torch()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    device = infer_input_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise ValueError("Model did not return hidden_states despite output_hidden_states=True.")

    layers = layer_selection or parse_layer_selection(layer_arg, len(hidden_states))
    attention_mask = inputs["attention_mask"]
    last_indices = attention_mask.shape[1] - 1 - attention_mask.flip(dims=[1]).argmax(dim=1)
    batch_indices = torch.arange(attention_mask.shape[0], device=attention_mask.device)

    per_layer: list[torch.Tensor] = []
    for layer_idx in layers:
        layer_hidden = hidden_states[layer_idx]
        final_token_hidden = layer_hidden[
            batch_indices.to(layer_hidden.device),
            last_indices.to(layer_hidden.device),
            :,
        ].detach().float()
        if normalize_hidden:
            final_token_hidden = torch.nn.functional.normalize(
                final_token_hidden,
                p=2,
                dim=-1,
            )
        per_layer.append(final_token_hidden.cpu())
    return torch.stack(per_layer, dim=1), layers


def existing_expected_layers(path: Path, layer_arg: str | None) -> list[int] | None:
    if layer_arg is None or not path.exists():
        return None
    payload = torch_load(path)
    layers = list(payload.get("layers") or [])
    if not layers:
        return None
    return parse_layer_selection(layer_arg, max(layers) + 1)


def load_existing_hidden_state(
    path: Path,
    *,
    model_key: str,
    expected_example_ids: list[str],
    layer_arg: str | None,
    normalize_hidden: bool,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    expected_layers = existing_expected_layers(path, layer_arg)
    payload = torch_load(path)
    validate_hidden_payload(
        payload,
        model_key=model_key,
        expected_example_ids=expected_example_ids,
        expected_layers=expected_layers,
    )
    if bool(payload.get("normalize_hidden", False)) != normalize_hidden:
        raise ValueError(
            f"Existing hidden-state file {path} was created with "
            f"normalize_hidden={payload.get('normalize_hidden')}; rerun with --overwrite."
        )
    return payload


def extract_hidden_states_for_model(
    *,
    model_key: str,
    subset_name: str,
    subset_df: pd.DataFrame,
    config: Any,
    batch_size: int,
    save_every: int,
    output_dir: Path,
    layer_arg: str | None,
    normalize_hidden: bool,
    overwrite: bool,
) -> Path:
    from tqdm.auto import tqdm

    from moral_mechinterp.models import load_tokenizer_and_model, unload_model
    from moral_mechinterp.utils import batched

    torch = load_torch()
    output_path = hidden_state_path(output_dir, model_key)
    expected_example_ids = [str(value) for value in subset_df["id"].tolist()]

    existing = None if overwrite else load_existing_hidden_state(
        output_path,
        model_key=model_key,
        expected_example_ids=expected_example_ids,
        layer_arg=layer_arg,
        normalize_hidden=normalize_hidden,
    )
    if existing is not None and bool(existing.get("complete")):
        if list(existing["example_ids"]) != expected_example_ids:
            raise ValueError(f"{output_path} is complete but has the wrong example IDs.")
        print(f"Using existing hidden states for {model_key}: {output_path}")
        return output_path

    chunks: list[torch.Tensor] = []
    completed_ids: list[str] = []
    layer_selection: list[int] | None = None
    if existing is not None:
        existing_hidden = existing["hidden_states"].cpu().float()
        chunks.append(existing_hidden)
        completed_ids = list(existing["example_ids"])
        layer_selection = list(existing["layers"])
        print(f"Resuming partial hidden-state extraction for {model_key} at n={len(completed_ids)}")

    model_id = config.models[model_key]
    print(f"Loading {MODEL_LABELS.get(model_key, model_key)} for representation drift: {model_id}")
    model = None
    tokenizer = None
    try:
        tokenizer, model = load_tokenizer_and_model(
            model_id,
            torch_dtype=config.torch_dtype,
            device_map=config.device_map,
            load_in_4bit=config.load_in_4bit,
            load_in_8bit=config.load_in_8bit,
            trust_remote_code=config.trust_remote_code,
        )
        model.eval()

        start_idx = len(completed_ids)
        row_indices = list(range(start_idx, len(subset_df)))
        progress = tqdm(
            list(batched(row_indices, batch_size)),
            desc=f"Hidden states {MODEL_LABELS.get(model_key, model_key)}",
            unit="batch",
        )
        examples_since_save = 0
        for batch_indices in progress:
            rows = [subset_df.iloc[idx] for idx in batch_indices]
            prompts = prepare_prompts(
                tokenizer=tokenizer,
                rows=rows,
                use_chat_template=config.use_chat_template,
            )
            batch_hidden, layer_selection = extract_batch_hidden_states(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                layer_selection=layer_selection,
                layer_arg=layer_arg,
                normalize_hidden=normalize_hidden,
            )
            chunks.append(batch_hidden)
            completed_ids.extend(str(subset_df.iloc[idx]["id"]) for idx in batch_indices)
            examples_since_save += len(batch_indices)

            if save_every > 0 and examples_since_save >= save_every:
                all_hidden = torch.cat(chunks, dim=0)
                save_hidden_payload(
                    output_path,
                    model_key=model_key,
                    example_ids=completed_ids,
                    layers=layer_selection,
                    hidden_states=all_hidden,
                    subset_name=subset_name,
                    complete=False,
                    normalize_hidden=normalize_hidden,
                )
                examples_since_save = 0

        if layer_selection is None:
            raise ValueError("No hidden states were extracted; subset may be empty.")
        all_hidden = torch.cat(chunks, dim=0)
        if completed_ids != expected_example_ids:
            raise ValueError("Extracted example IDs do not match the subset order.")
        save_hidden_payload(
            output_path,
            model_key=model_key,
            example_ids=completed_ids,
            layers=layer_selection,
            hidden_states=all_hidden,
            subset_name=subset_name,
            complete=True,
            normalize_hidden=normalize_hidden,
        )
        return output_path
    except Exception as exc:
        message = f"Failed hidden-state extraction for model {model_key} ({model_id})"
        raise RuntimeError(message) from exc
    finally:
        unload_model(model)
        del model, tokenizer


def load_hidden_payloads(output_dir: Path, model_keys: list[str]) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for model_key in model_keys:
        path = hidden_state_path(output_dir, model_key)
        if not path.exists():
            raise FileNotFoundError(f"Missing hidden-state file for {model_key}: {path}")
        payload = torch_load(path)
        if not bool(payload.get("complete")):
            raise ValueError(f"Hidden-state file is incomplete: {path}")
        payloads[model_key] = payload
    return payloads


def validate_pair_payloads(payload_a: dict[str, Any], payload_b: dict[str, Any]) -> None:
    if list(payload_a["example_ids"]) != list(payload_b["example_ids"]):
        raise ValueError("Example ID ordering differs between hidden-state files.")
    if list(payload_a["layers"]) != list(payload_b["layers"]):
        raise ValueError("Layer indices differ between hidden-state files.")
    hidden_a = payload_a["hidden_states"]
    hidden_b = payload_b["hidden_states"]
    if hidden_a.shape != hidden_b.shape:
        raise ValueError(f"Hidden-state shapes differ: {hidden_a.shape} vs {hidden_b.shape}")


def pair_specs_for_models(model_keys: list[str]) -> list[tuple[str, str, str]]:
    model_set = set(model_keys)
    return [pair for pair in PAIR_SPECS if pair[0] in model_set and pair[1] in model_set]


def compute_cosine_drift_outputs(
    *,
    output_dir: Path,
    subset_name: str,
    model_keys: list[str],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import numpy as np
    import pandas as pd
    import torch

    from moral_mechinterp.metrics import bootstrap_ci

    payloads = load_hidden_payloads(output_dir, model_keys)
    value_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for model_a, model_b, pair_label in pair_specs_for_models(model_keys):
        payload_a = payloads[model_a]
        payload_b = payloads[model_b]
        validate_pair_payloads(payload_a, payload_b)

        hidden_a = payload_a["hidden_states"].float()
        hidden_b = payload_b["hidden_states"].float()
        example_ids = list(payload_a["example_ids"])
        layers = list(payload_a["layers"])
        for layer_pos, layer_idx in enumerate(layers):
            h_a = hidden_a[:, layer_pos, :]
            h_b = hidden_b[:, layer_pos, :]
            similarities = torch.nn.functional.cosine_similarity(h_a, h_b, dim=-1)
            drifts = 1.0 - similarities
            sim_np = similarities.numpy()
            drift_np = drifts.numpy()
            ci_low, ci_high = bootstrap_ci(drift_np, seed=seed)

            for example_id, similarity, drift in zip(example_ids, sim_np, drift_np, strict=True):
                value_rows.append(
                    {
                        "subset_name": subset_name,
                        "pair": pair_label,
                        "model_a": model_a,
                        "model_b": model_b,
                        "layer": int(layer_idx),
                        "id": example_id,
                        "cosine_similarity": float(similarity),
                        "cosine_drift": float(drift),
                    }
                )

            summary_rows.append(
                {
                    "subset_name": subset_name,
                    "pair": pair_label,
                    "model_a": model_a,
                    "model_b": model_b,
                    "layer": int(layer_idx),
                    "mean_cosine_similarity": float(np.mean(sim_np)),
                    "mean_cosine_drift": float(np.mean(drift_np)),
                    "median_cosine_drift": float(np.median(drift_np)),
                    "std_cosine_drift": float(np.std(drift_np, ddof=1)),
                    "sem_cosine_drift": float(np.std(drift_np, ddof=1) / np.sqrt(len(drift_np))),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n": int(len(drift_np)),
                }
            )

    value_df = pd.DataFrame.from_records(value_rows)
    summary_df = pd.DataFrame.from_records(summary_rows)
    value_df.to_csv(output_dir / "cosine_drift_values.csv", index=False)
    summary_df.to_csv(output_dir / "cosine_drift_summary.csv", index=False)
    return value_df, summary_df


def plot_cosine_drift_summary(
    summary_df: pd.DataFrame,
    *,
    subset_name: str,
    font_family: str,
) -> list[Path]:
    import matplotlib.pyplot as plt

    from moral_mechinterp.plot_style import apply_paper_style, despine, save_figure

    figure_dir = Path("outputs/figures_repdrift")
    figure_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style(font_family=font_family)
    fig, ax = plt.subplots(figsize=(4.7, 3.0))

    for _, _, pair_label in PAIR_SPECS:
        pair_df = summary_df[summary_df["pair"] == pair_label].sort_values("layer")
        if pair_df.empty:
            continue
        x = pair_df["layer"].to_numpy(dtype=float)
        y = pair_df["mean_cosine_drift"].to_numpy(dtype=float)
        low = pair_df["ci_low"].to_numpy(dtype=float)
        high = pair_df["ci_high"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            color=PAIR_COLORS[pair_label],
            marker=PAIR_MARKERS[pair_label],
            markevery=max(1, len(x) // 8),
            linewidth=1.35,
            markersize=3.6,
            label=pair_label,
        )
        ax.fill_between(x, low, high, color=PAIR_COLORS[pair_label], alpha=0.15, linewidth=0)

    ax.axhline(0, color="#2A2A2A", linewidth=0.75, linestyle=(0, (3, 2)))
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean cosine drift")
    ax.set_title(f"Representation drift: {subset_name}")
    ax.legend(frameon=False)
    ax.yaxis.grid(True)
    despine(ax)
    return save_figure(fig, figure_dir / f"{subset_name}_cosine_drift")


def write_metadata(
    *,
    output_dir: Path,
    subset_csv: Path,
    subset_name: str,
    n_examples: int,
    model_keys: list[str],
    config_path: Path,
    config: Any,
    layer_arg: str | None,
    normalize_hidden: bool,
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
        "layers": layer_arg or "all",
        "normalize_hidden": normalize_hidden,
        "hidden_state_interpretation": HIDDEN_STATE_INTERPRETATION,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    args = parse_args()

    from moral_mechinterp.config import load_eval_config
    from moral_mechinterp.utils import set_seed

    config = load_eval_config(args.config)
    set_seed(config.seed)

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    subset_df = load_subset(args.subset_csv, max_examples=args.max_examples)
    subset_name = args.subset_csv.stem
    model_keys = parse_model_keys(args.models, config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(
        output_dir=args.output_dir,
        subset_csv=args.subset_csv,
        subset_name=subset_name,
        n_examples=len(subset_df),
        model_keys=model_keys,
        config_path=args.config,
        config=config,
        layer_arg=args.layers,
        normalize_hidden=args.normalize_hidden,
    )

    for model_key in model_keys:
        extract_hidden_states_for_model(
            model_key=model_key,
            subset_name=subset_name,
            subset_df=subset_df,
            config=config,
            batch_size=args.batch_size,
            save_every=args.save_every,
            output_dir=args.output_dir,
            layer_arg=args.layers,
            normalize_hidden=args.normalize_hidden,
            overwrite=args.overwrite,
        )

    _, summary_df = compute_cosine_drift_outputs(
        output_dir=args.output_dir,
        subset_name=subset_name,
        model_keys=model_keys,
        seed=config.seed,
    )
    if not args.no_plot and not summary_df.empty:
        figure_paths = plot_cosine_drift_summary(
            summary_df,
            subset_name=subset_name,
            font_family=config.plot_font_family,
        )
        print("Wrote representation drift figure:")
        for path in figure_paths:
            print(f"  {path}")

    print(f"Wrote representation drift outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
