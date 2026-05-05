from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DEFAULT_LAYERS = "0,4,8,12,16,20,21,22,23,24,25,26,27,28,29,30,31,32"
DEFAULT_SUBSETS = (
    "random_pd_150",
    "random_chicken_150",
    "top_ut_margin_shift",
    "top_game_margin_shift",
    "ut_safe_game_harmful",
    "game_safe_ut_harmful",
)
SUBSET_LABELS = {
    "full": "Full GT-HarmBench",
    "top_ut_margin_shift": "UT-favored margin shifts",
    "top_game_margin_shift": "GAME-favored margin shifts",
    "ut_safe_game_harmful": "UT-safe / GAME-harmful",
    "game_safe_ut_harmful": "GAME-safe / UT-harmful",
    "random_pd_150": "Random Prisoner's Dilemma",
    "random_chicken_150": "Random Chicken",
}
ADAPTER_LABELS = {
    "ut": "UT",
    "game": "GAME",
}
PATCH_DIRECTIONS = ("A_to_Base", "Base_to_A")

PATCHING_PAPER_PARAGRAPH = (
    "To test whether the late-layer adapter-delta signals causally mediate final "
    "safe-action margins, we patched final-token residual-stream states between "
    "Base and each adapter. For adapter A, A-to-Base patching measures whether "
    "adapter states are sufficient to move the Base model toward the adapter, "
    "while Base-to-A patching measures whether replacing adapter states with Base "
    "states removes the adapter effect. We report recovered and removed fractions "
    "of the clean adapter margin shift across layers."
)
DIRECTION_VECTOR_CAUTION = (
    "This activation-patching experiment copies each source example's own "
    "final-token residual state into the matched target example. It does not "
    "estimate an averaged adapter direction v or sweep an intervention scale "
    "alpha. For a future direction-vector experiment, estimate v on one split "
    "(for example random PD train) and test on a held-out split to avoid "
    "circularity, then sweep alpha values around the natural adapter-delta "
    "scale to test local linearity and saturation."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run final-token late-layer activation patching between Base and "
            "UT/GAME PEFT adapters. This is a GPU experiment and does not use "
            "free-form generation."
        )
    )
    parser.add_argument("--data-jsonl", type=Path, default=Path("data/gtharmbench_balanced.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/patching"))
    parser.add_argument("--subset-dir", type=Path, default=Path("outputs/behavior_full/subsets"))
    parser.add_argument("--subsets", default=",".join(DEFAULT_SUBSETS))
    parser.add_argument("--adapters", default="ut,game")
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-examples-per-subset", type=int, default=None)
    parser.add_argument("--full-sample-size", type=int, default=0)
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument("--delta-threshold", type=float, default=0.05)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def load_pandas():
    import pandas as pd

    return pd


def parse_csv_list(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated value.")
    return values


def parse_layers(raw: str, *, n_transformer_layers: int) -> list[int]:
    if raw.strip().lower() == "all":
        return list(range(n_transformer_layers + 1))

    layers: set[int] = set()
    for part in parse_csv_list(raw):
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid layer range {part!r}: end < start.")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))

    invalid = [layer for layer in sorted(layers) if layer < 0 or layer > n_transformer_layers]
    if invalid:
        raise ValueError(
            f"Requested layer(s) {invalid} are outside valid range 0-{n_transformer_layers}."
        )
    return sorted(layers)


def row_to_example(row: Any) -> Any:
    from moral_mechinterp.data import NormalizedExample

    safe_label = str(row["safe_label"]).strip().upper()
    if safe_label not in {"A", "B"}:
        raise ValueError(f"safe_label must be A or B, got {safe_label!r}")
    return NormalizedExample(
        id=str(row["id"]),
        game_type=str(row["game_type"]),
        scenario=str(row["scenario"]),
        option_a=str(row["option_a"]),
        option_b=str(row["option_b"]),
        safe_label=safe_label,
    )


def load_subset_csv(path: Path, *, max_examples: int | None) -> list[Any]:
    pd = load_pandas()
    if not path.exists():
        raise FileNotFoundError(f"Missing subset CSV: {path}")
    df = pd.read_csv(path)
    required = {"id", "game_type", "scenario", "option_a", "option_b", "safe_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if max_examples is not None:
        df = df.head(max_examples).copy()
    return [row_to_example(row) for _, row in df.iterrows()]


def load_subsets(
    *,
    data_jsonl: Path,
    subset_dir: Path,
    subset_names: list[str],
    max_examples_per_subset: int | None,
    full_sample_size: int,
    include_full: bool,
    seed: int,
) -> dict[str, list[Any]]:
    from random import Random

    from moral_mechinterp.data import load_jsonl_examples

    subsets: dict[str, list[Any]] = {}
    for subset_name in subset_names:
        subset_path = subset_dir / f"{subset_name}.csv"
        subsets[subset_name] = load_subset_csv(
            subset_path,
            max_examples=max_examples_per_subset,
        )

    if include_full or full_sample_size > 0:
        examples = load_jsonl_examples(data_jsonl)
        if full_sample_size > 0:
            rng = Random(seed)
            if len(examples) < full_sample_size:
                raise ValueError(
                    f"Requested full sample of {full_sample_size}, but only "
                    f"{len(examples)} examples are available."
                )
            examples = rng.sample(examples, full_sample_size)
            subset_name = f"full_sample_{full_sample_size}"
            SUBSET_LABELS[subset_name] = f"Full random sample (n={full_sample_size})"
        else:
            subset_name = "full"
        if max_examples_per_subset is not None:
            examples = examples[:max_examples_per_subset]
        subsets[subset_name] = examples

    if not subsets:
        raise ValueError("No subsets were selected.")
    return subsets


def unique_examples(subsets: dict[str, list[Any]]) -> list[Any]:
    examples_by_id: dict[str, Any] = {}
    for examples in subsets.values():
        for example in examples:
            existing = examples_by_id.get(example.id)
            if existing is not None and existing.to_record() != example.to_record():
                raise ValueError(f"Example id {example.id!r} appears with conflicting contents.")
            examples_by_id[example.id] = example
    return list(examples_by_id.values())


def build_prompt(tokenizer: Any, example: Any, config: Any) -> str:
    from moral_mechinterp.prompts import build_ab_prompt
    from moral_mechinterp.scoring import apply_chat_template_if_needed

    return apply_chat_template_if_needed(
        tokenizer,
        build_ab_prompt(example),
        use_chat_template=config.use_chat_template,
    )


def tokenize_prompts(tokenizer: Any, prompts: list[str], model: Any) -> dict[str, Any]:
    from moral_mechinterp.logit_lens import infer_input_device

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    device = infer_input_device(model)
    return {key: value.to(device) for key, value in inputs.items()}


def final_token_indices(attention_mask: Any) -> Any:
    return attention_mask.shape[1] - 1 - attention_mask.flip(dims=[1]).argmax(dim=1)


def choice_from_logits(logit_a: float, logit_b: float) -> str:
    return "A" if logit_a >= logit_b else "B"


def score_from_next_token_logits(
    next_token_logits: Any,
    *,
    token_ids: dict[str, int],
    safe_labels: list[str],
) -> list[dict[str, Any]]:
    from moral_mechinterp.scoring import safe_margin_from_logits

    a_logits = next_token_logits[:, token_ids["A"]].detach().float().cpu().tolist()
    b_logits = next_token_logits[:, token_ids["B"]].detach().float().cpu().tolist()
    scores: list[dict[str, Any]] = []
    for logit_a, logit_b, safe_label in zip(a_logits, b_logits, safe_labels, strict=True):
        choice = choice_from_logits(float(logit_a), float(logit_b))
        scores.append(
            {
                "logit_A": float(logit_a),
                "logit_B": float(logit_b),
                "safe_margin": float(safe_margin_from_logits(logit_a, logit_b, safe_label)),
                "choice": choice,
                "safe_choice": choice == safe_label,
            }
        )
    return scores


@contextmanager
def model_mode(model: Any, mode: str):
    """Switch a PEFT-wrapped model into Base, UT, or GAME mode."""

    if mode == "base":
        if hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                yield
        else:
            yield
        return

    if not hasattr(model, "set_adapter"):
        raise ValueError(f"Model does not support PEFT adapter switching for mode {mode!r}.")
    model.set_adapter(mode)
    yield


def get_transformer_layers(model: Any) -> Any:
    from moral_mechinterp.logit_lens import get_by_path

    candidate_paths = (
        "model.layers",
        "model.model.layers",
        "base_model.model.model.layers",
        "base_model.model.model.model.layers",
    )
    for path in candidate_paths:
        layers = get_by_path(model, path)
        if layers is not None:
            return layers
    raise ValueError("Could not locate transformer layers on the model.")


def get_embedding_module(model: Any) -> Any:
    from moral_mechinterp.logit_lens import get_by_path

    candidate_paths = (
        "model.embed_tokens",
        "model.model.embed_tokens",
        "base_model.model.model.embed_tokens",
        "base_model.model.model.model.embed_tokens",
    )
    for path in candidate_paths:
        module = get_by_path(model, path)
        if module is not None:
            return module
    if hasattr(model, "get_input_embeddings"):
        module = model.get_input_embeddings()
        if module is not None:
            return module
    raise ValueError("Could not locate input embedding module on the model.")


def get_final_norm_module(model: Any) -> Any:
    from moral_mechinterp.logit_lens import get_by_path

    candidate_paths = (
        "model.norm",
        "model.model.norm",
        "base_model.model.model.norm",
        "base_model.model.model.model.norm",
    )
    for path in candidate_paths:
        module = get_by_path(model, path)
        if module is not None:
            return module
    raise ValueError("Could not locate final norm module on the model.")


def get_patch_module(model: Any, *, layer: int, n_transformer_layers: int) -> Any:
    if layer == 0:
        return get_embedding_module(model)
    if layer == n_transformer_layers:
        return get_final_norm_module(model)
    layers = get_transformer_layers(model)
    return layers[layer - 1]


def output_tensor_and_rebuilder(output: Any):
    if hasattr(output, "shape"):
        return output, lambda new_tensor: new_tensor
    if isinstance(output, tuple):
        tensor = output[0]
        return tensor, lambda new_tensor: (new_tensor, *output[1:])
    if isinstance(output, list):
        tensor = output[0]
        return tensor, lambda new_tensor: [new_tensor, *output[1:]]
    raise TypeError(f"Unsupported hook output type: {type(output)!r}")


def capture_layer_activation(
    *,
    model: Any,
    inputs: dict[str, Any],
    mode: str,
    layer: int,
    n_transformer_layers: int,
) -> Any:
    import torch

    final_indices = final_token_indices(inputs["attention_mask"])
    store: dict[str, Any] = {}
    module = get_patch_module(model, layer=layer, n_transformer_layers=n_transformer_layers)

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        tensor, _ = output_tensor_and_rebuilder(output)
        indices = final_indices.to(tensor.device)
        batch = torch.arange(tensor.shape[0], device=tensor.device)
        store["activation"] = tensor[batch, indices, :].detach()
        return None

    handle = module.register_forward_hook(hook)
    try:
        with model_mode(model, mode), torch.inference_mode():
            model(**inputs, use_cache=False)
    finally:
        handle.remove()

    if "activation" not in store:
        raise RuntimeError(f"Failed to capture activation for mode={mode}, layer={layer}.")
    return store["activation"]


def forward_ab_scores(
    *,
    model: Any,
    inputs: dict[str, Any],
    mode: str,
    token_ids: dict[str, int],
    safe_labels: list[str],
) -> list[dict[str, Any]]:
    import torch

    with model_mode(model, mode), torch.inference_mode():
        outputs = model(**inputs, use_cache=False)
    logits = outputs.logits
    last_indices = final_token_indices(inputs["attention_mask"])
    batch = torch.arange(logits.shape[0], device=logits.device)
    next_token_logits = logits[batch, last_indices.to(logits.device), :]
    return score_from_next_token_logits(
        next_token_logits,
        token_ids=token_ids,
        safe_labels=safe_labels,
    )


def patched_forward_ab_scores(
    *,
    model: Any,
    inputs: dict[str, Any],
    target_mode: str,
    layer: int,
    n_transformer_layers: int,
    source_activation: Any,
    token_ids: dict[str, int],
    safe_labels: list[str],
) -> list[dict[str, Any]]:
    import torch

    final_indices = final_token_indices(inputs["attention_mask"])
    module = get_patch_module(model, layer=layer, n_transformer_layers=n_transformer_layers)

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        tensor, rebuild = output_tensor_and_rebuilder(output)
        indices = final_indices.to(tensor.device)
        batch = torch.arange(tensor.shape[0], device=tensor.device)
        patched = tensor.clone()
        patched[batch, indices, :] = source_activation.to(
            device=tensor.device,
            dtype=tensor.dtype,
        )
        return rebuild(patched)

    handle = module.register_forward_hook(hook)
    try:
        with model_mode(model, target_mode), torch.inference_mode():
            outputs = model(**inputs, use_cache=False)
    finally:
        handle.remove()

    logits = outputs.logits
    last_indices = final_token_indices(inputs["attention_mask"])
    batch = torch.arange(logits.shape[0], device=logits.device)
    next_token_logits = logits[batch, last_indices.to(logits.device), :]
    return score_from_next_token_logits(
        next_token_logits,
        token_ids=token_ids,
        safe_labels=safe_labels,
    )


def load_base_with_adapters(config: Any) -> tuple[Any, Any]:
    from peft import PeftModel

    from moral_mechinterp.models import load_tokenizer_and_model

    tokenizer, base_model = load_tokenizer_and_model(
        config.models["base"],
        torch_dtype=config.torch_dtype,
        device_map=config.device_map,
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=config.load_in_8bit,
        trust_remote_code=config.trust_remote_code,
    )
    model = PeftModel.from_pretrained(
        base_model,
        config.models["ut"],
        adapter_name="ut",
        is_trainable=False,
    )
    model.load_adapter(
        config.models["game"],
        adapter_name="game",
        is_trainable=False,
    )
    model.eval()
    return tokenizer, model


def write_jsonl(records: list[dict[str, Any]], path: Path) -> Path:
    from moral_mechinterp.io import ensure_dir

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def write_clean_margins(records: list[dict[str, Any]], output_dir: Path) -> None:
    pd = load_pandas()
    write_jsonl(records, output_dir / "clean_margins.jsonl")
    pd.DataFrame.from_records(records).to_csv(output_dir / "clean_margins.csv", index=False)


def clean_margin_records(
    *,
    model: Any,
    tokenizer: Any,
    examples: list[Any],
    config: Any,
    token_ids: dict[str, int],
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    from tqdm.auto import tqdm

    from moral_mechinterp.utils import batched

    records: dict[str, dict[str, Any]] = {}
    batches = list(batched(examples, batch_size))
    for batch_examples in tqdm(batches, desc="Clean margins", unit="batch"):
        prompts = [build_prompt(tokenizer, example, config) for example in batch_examples]
        safe_labels = [example.safe_label for example in batch_examples]
        inputs = tokenize_prompts(tokenizer, prompts, model)

        batch_scores = {
            "base": forward_ab_scores(
                model=model,
                inputs=inputs,
                mode="base",
                token_ids=token_ids,
                safe_labels=safe_labels,
            ),
            "ut": forward_ab_scores(
                model=model,
                inputs=inputs,
                mode="ut",
                token_ids=token_ids,
                safe_labels=safe_labels,
            ),
            "game": forward_ab_scores(
                model=model,
                inputs=inputs,
                mode="game",
                token_ids=token_ids,
                safe_labels=safe_labels,
            ),
        }

        for row_idx, example in enumerate(batch_examples):
            record: dict[str, Any] = {
                "example_id": example.id,
                "game_type": example.game_type,
                "safe_label": example.safe_label,
            }
            for mode in ("base", "ut", "game"):
                score = batch_scores[mode][row_idx]
                record[f"m_{mode}"] = score["safe_margin"]
                record[f"{mode}_choice"] = score["choice"]
                record[f"{mode}_safe_choice"] = score["safe_choice"]
                record[f"{mode}_logit_A"] = score["logit_A"]
                record[f"{mode}_logit_B"] = score["logit_B"]
            record["delta_ut"] = record["m_ut"] - record["m_base"]
            record["delta_game"] = record["m_game"] - record["m_base"]
            records[example.id] = record
    return records


def patch_result_record(
    *,
    example: Any,
    subset_name: str,
    adapter_key: str,
    patch_direction: str,
    layer: int,
    clean: dict[str, Any],
    patched_score: dict[str, Any],
    delta_threshold: float,
    eps: float,
) -> dict[str, Any]:
    adapter_label = ADAPTER_LABELS[adapter_key]
    m_base = float(clean["m_base"])
    m_adapter = float(clean[f"m_{adapter_key}"])
    m_patched = float(patched_score["safe_margin"])
    delta_adapter = m_adapter - m_base
    denominator_ok = abs(delta_adapter) > delta_threshold

    if patch_direction == "A_to_Base":
        raw_patch_effect = m_patched - m_base
    elif patch_direction == "Base_to_A":
        raw_patch_effect = m_adapter - m_patched
    else:
        raise ValueError(f"Unknown patch direction: {patch_direction!r}")

    fraction = None
    if denominator_ok:
        fraction = raw_patch_effect / (delta_adapter + eps)

    return {
        "example_id": example.id,
        "subset_name": subset_name,
        "subset_label": SUBSET_LABELS.get(subset_name, subset_name),
        "game_type": example.game_type,
        "safe_label": example.safe_label,
        "adapter_name": adapter_label,
        "adapter_key": adapter_key,
        "patch_direction": patch_direction,
        "layer": layer,
        "m_base": m_base,
        "m_adapter": m_adapter,
        "m_patched": m_patched,
        "delta_adapter": delta_adapter,
        "raw_patch_effect": raw_patch_effect,
        "recovered_or_removed_fraction": fraction,
        "denominator_ok": denominator_ok,
        "safe_choice_base": bool(clean["base_safe_choice"]),
        "safe_choice_adapter": bool(clean[f"{adapter_key}_safe_choice"]),
        "safe_choice_patched": bool(patched_score["safe_choice"]),
        "choice_base": clean["base_choice"],
        "choice_adapter": clean[f"{adapter_key}_choice"],
        "choice_patched": patched_score["choice"],
    }


def save_patch_outputs(
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
    seed: int,
) -> Any:
    pd = load_pandas()
    write_jsonl(records, output_dir / "activation_patching_results.jsonl")
    result_df = pd.DataFrame.from_records(records)
    result_df.to_csv(output_dir / "activation_patching_results.csv", index=False)
    summary_df = summarize_patch_results(result_df, seed=seed)
    summary_df.to_csv(output_dir / "activation_patching_summary.csv", index=False)
    return summary_df


def summarize_patch_results(result_df: Any, *, seed: int) -> Any:
    pd = load_pandas()
    from moral_mechinterp.metrics import bootstrap_ci

    if result_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    group_cols = ["adapter_name", "patch_direction", "subset_name", "subset_label", "layer"]
    for group_key, group in result_df.groupby(group_cols, sort=True, dropna=False):
        raw = pd.to_numeric(group["raw_patch_effect"], errors="coerce").dropna()
        denom_ok = group[group["denominator_ok"].astype(bool)]
        frac = pd.to_numeric(
            denom_ok["recovered_or_removed_fraction"],
            errors="coerce",
        ).dropna()
        raw_low, raw_high = bootstrap_ci(raw, seed=seed)
        frac_low, frac_high = bootstrap_ci(frac, seed=seed)
        rows.append(
            {
                "adapter_name": group_key[0],
                "patch_direction": group_key[1],
                "subset_name": group_key[2],
                "subset_label": group_key[3],
                "layer": int(group_key[4]),
                "n": int(len(group)),
                "n_denominator_ok": int(len(frac)),
                "mean_m_base": pd.to_numeric(group["m_base"], errors="coerce").mean(),
                "mean_m_adapter": pd.to_numeric(group["m_adapter"], errors="coerce").mean(),
                "mean_m_patched": pd.to_numeric(group["m_patched"], errors="coerce").mean(),
                "mean_delta_adapter": pd.to_numeric(
                    group["delta_adapter"],
                    errors="coerce",
                ).mean(),
                "mean_raw_patch_effect": raw.mean(),
                "mean_recovered_or_removed_fraction": frac.mean(),
                "median_recovered_or_removed_fraction": frac.median(),
                "raw_patch_effect_ci_low": raw_low,
                "raw_patch_effect_ci_high": raw_high,
                "fraction_ci_low": frac_low,
                "fraction_ci_high": frac_high,
            }
        )
    return pd.DataFrame(rows)


def run_activation_patching(
    *,
    model: Any,
    tokenizer: Any,
    subsets: dict[str, list[Any]],
    clean_by_id: dict[str, dict[str, Any]],
    config: Any,
    token_ids: dict[str, int],
    adapters: list[str],
    layers: list[int],
    n_transformer_layers: int,
    batch_size: int,
    delta_threshold: float,
    eps: float,
    output_dir: Path,
    save_every: int,
    seed: int,
) -> Any:
    from tqdm.auto import tqdm

    from moral_mechinterp.utils import batched

    records: list[dict[str, Any]] = []
    completed_batches = 0
    total_steps = sum(
        math.ceil(len(examples) / batch_size) * len(adapters) * len(PATCH_DIRECTIONS) * len(layers)
        for examples in subsets.values()
    )
    progress = tqdm(total=total_steps, desc="Activation patching", unit="batch")

    for subset_name, examples in subsets.items():
        for layer in layers:
            for adapter_key in adapters:
                for batch_examples in batched(examples, batch_size):
                    prompts = [
                        build_prompt(tokenizer, example, config)
                        for example in batch_examples
                    ]
                    safe_labels = [example.safe_label for example in batch_examples]
                    inputs = tokenize_prompts(tokenizer, prompts, model)

                    adapter_activation = capture_layer_activation(
                        model=model,
                        inputs=inputs,
                        mode=adapter_key,
                        layer=layer,
                        n_transformer_layers=n_transformer_layers,
                    )
                    patched_base_scores = patched_forward_ab_scores(
                        model=model,
                        inputs=inputs,
                        target_mode="base",
                        layer=layer,
                        n_transformer_layers=n_transformer_layers,
                        source_activation=adapter_activation,
                        token_ids=token_ids,
                        safe_labels=safe_labels,
                    )
                    for example, patched_score in zip(
                        batch_examples,
                        patched_base_scores,
                        strict=True,
                    ):
                        records.append(
                            patch_result_record(
                                example=example,
                                subset_name=subset_name,
                                adapter_key=adapter_key,
                                patch_direction="A_to_Base",
                                layer=layer,
                                clean=clean_by_id[example.id],
                                patched_score=patched_score,
                                delta_threshold=delta_threshold,
                                eps=eps,
                            )
                        )
                    completed_batches += 1
                    progress.update(1)

                    base_activation = capture_layer_activation(
                        model=model,
                        inputs=inputs,
                        mode="base",
                        layer=layer,
                        n_transformer_layers=n_transformer_layers,
                    )
                    patched_adapter_scores = patched_forward_ab_scores(
                        model=model,
                        inputs=inputs,
                        target_mode=adapter_key,
                        layer=layer,
                        n_transformer_layers=n_transformer_layers,
                        source_activation=base_activation,
                        token_ids=token_ids,
                        safe_labels=safe_labels,
                    )
                    for example, patched_score in zip(
                        batch_examples,
                        patched_adapter_scores,
                        strict=True,
                    ):
                        records.append(
                            patch_result_record(
                                example=example,
                                subset_name=subset_name,
                                adapter_key=adapter_key,
                                patch_direction="Base_to_A",
                                layer=layer,
                                clean=clean_by_id[example.id],
                                patched_score=patched_score,
                                delta_threshold=delta_threshold,
                                eps=eps,
                            )
                        )
                    completed_batches += 1
                    progress.update(1)

                    if save_every > 0 and completed_batches % save_every == 0:
                        save_patch_outputs(records, output_dir=output_dir, seed=seed)

    progress.close()
    return save_patch_outputs(records, output_dir=output_dir, seed=seed)


def line_label(row: Any) -> str:
    adapter = row.adapter_name
    if row.patch_direction == "A_to_Base":
        return f"{adapter}→Base"
    return f"Base→{adapter}"


def plot_summary(summary_df: Any, *, output_dir: Path, font_family: str) -> None:
    if summary_df.empty:
        return
    plot_fraction(
        summary_df[summary_df["patch_direction"] == "A_to_Base"],
        output_dir=output_dir,
        filename="fig_recovered_fraction_by_layer",
        ylabel="Recovered fraction",
        title="Adapter-to-Base recovered fraction",
        font_family=font_family,
    )
    plot_fraction(
        summary_df[summary_df["patch_direction"] == "Base_to_A"],
        output_dir=output_dir,
        filename="fig_removed_fraction_by_layer",
        ylabel="Removed fraction",
        title="Base-to-adapter removed fraction",
        font_family=font_family,
    )
    plot_raw_effects(summary_df, output_dir=output_dir, font_family=font_family)


def subplot_grid(n_panels: int) -> tuple[int, int]:
    n_cols = 2 if n_panels > 1 else 1
    n_rows = math.ceil(n_panels / n_cols)
    return n_rows, n_cols


def plot_fraction(
    df: Any,
    *,
    output_dir: Path,
    filename: str,
    ylabel: str,
    title: str,
    font_family: str,
) -> None:
    import matplotlib.pyplot as plt

    from moral_mechinterp.constants import MODEL_COLORS, MODEL_MARKERS
    from moral_mechinterp.plot_style import apply_paper_style, despine, save_figure

    if df.empty:
        return
    apply_paper_style(font_family=font_family)
    subset_names = list(dict.fromkeys(df["subset_name"].tolist()))
    n_rows, n_cols = subplot_grid(len(subset_names))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.7 * n_cols, 2.7 * n_rows),
        squeeze=False,
    )
    for ax, subset_name in zip(axes.ravel(), subset_names, strict=False):
        subset_df = df[df["subset_name"] == subset_name]
        for adapter_name, adapter_df in subset_df.groupby("adapter_name", sort=False):
            adapter_key = adapter_name.lower()
            ordered = adapter_df.sort_values("layer")
            ax.plot(
                ordered["layer"],
                ordered["mean_recovered_or_removed_fraction"],
                color=MODEL_COLORS.get(adapter_key, "#252525"),
                marker=MODEL_MARKERS.get(adapter_key, "o"),
                linewidth=1.35,
                markersize=3.3,
                label=line_label(ordered.iloc[0]),
            )
            ax.fill_between(
                ordered["layer"],
                ordered["fraction_ci_low"],
                ordered["fraction_ci_high"],
                color=MODEL_COLORS.get(adapter_key, "#252525"),
                alpha=0.14,
                linewidth=0,
            )
        ax.axhline(0, color="#2A2A2A", linewidth=0.8, linestyle=(0, (3, 2)))
        ax.axvspan(21, 31, color="#E8E8E8", alpha=0.42, linewidth=0)
        ax.axvline(32, color="#2A2A2A", linewidth=0.85, linestyle=(0, (2, 2)))
        ax.set_title(str(subset_df["subset_label"].iloc[0]))
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True)
        despine(ax)
    for ax in axes.ravel()[len(subset_names):]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(title, y=0.995)
    fig.subplots_adjust(top=0.88, hspace=0.48, wspace=0.28)
    save_figure(fig, output_dir / filename)


def plot_raw_effects(summary_df: Any, *, output_dir: Path, font_family: str) -> None:
    import matplotlib.pyplot as plt

    from moral_mechinterp.constants import MODEL_COLORS, MODEL_MARKERS
    from moral_mechinterp.plot_style import apply_paper_style, despine, save_figure

    apply_paper_style(font_family=font_family)
    subset_names = list(dict.fromkeys(summary_df["subset_name"].tolist()))
    n_rows, n_cols = subplot_grid(len(subset_names))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.7 * n_cols, 2.7 * n_rows),
        squeeze=False,
    )
    line_styles = {"A_to_Base": "-", "Base_to_A": (0, (3, 2))}
    for ax, subset_name in zip(axes.ravel(), subset_names, strict=False):
        subset_df = summary_df[summary_df["subset_name"] == subset_name]
        for group_key, group in subset_df.groupby(
            ["adapter_name", "patch_direction"],
            sort=False,
        ):
            adapter_name, patch_direction = group_key
            adapter_key = adapter_name.lower()
            ordered = group.sort_values("layer")
            ax.plot(
                ordered["layer"],
                ordered["mean_raw_patch_effect"],
                color=MODEL_COLORS.get(adapter_key, "#252525"),
                marker=MODEL_MARKERS.get(adapter_key, "o"),
                linestyle=line_styles[patch_direction],
                linewidth=1.3,
                markersize=3.2,
                label=line_label(ordered.iloc[0]),
            )
            ax.fill_between(
                ordered["layer"],
                ordered["raw_patch_effect_ci_low"],
                ordered["raw_patch_effect_ci_high"],
                color=MODEL_COLORS.get(adapter_key, "#252525"),
                alpha=0.12,
                linewidth=0,
            )
        ax.axhline(0, color="#2A2A2A", linewidth=0.8, linestyle=(0, (3, 2)))
        ax.axvspan(21, 31, color="#E8E8E8", alpha=0.42, linewidth=0)
        ax.axvline(32, color="#2A2A2A", linewidth=0.85, linestyle=(0, (2, 2)))
        ax.set_title(str(subset_df["subset_label"].iloc[0]))
        ax.set_xlabel("Layer")
        ax.set_ylabel("Raw safe-margin change")
        ax.yaxis.grid(True)
        despine(ax)
    for ax in axes.ravel()[len(subset_names):]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Raw activation-patching margin effects", y=0.995)
    fig.subplots_adjust(top=0.88, hspace=0.48, wspace=0.28)
    save_figure(fig, output_dir / "fig_raw_margin_change_by_layer")


def write_metadata(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    adapters: list[str],
    layers: list[int],
    subsets: dict[str, list[Any]],
    n_transformer_layers: int,
) -> None:
    metadata = {
        "paper_paragraph": PATCHING_PAPER_PARAGRAPH,
        "direction_vector_caution": DIRECTION_VECTOR_CAUTION,
        "data_jsonl": str(args.data_jsonl),
        "config": str(args.config),
        "subsets": {name: len(examples) for name, examples in subsets.items()},
        "adapters": adapters,
        "layers": layers,
        "n_transformer_layers": n_transformer_layers,
        "delta_threshold": args.delta_threshold,
        "eps": args.eps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "token_position": "final non-padding token: attention_mask.sum(dim=-1) - 1",
        "layer_interpretation": (
            "Layer 0 patches the embedding output. Layers 1..L patch transformer "
            "block outputs. Layer L patches the final normalized hidden state before "
            "the LM head and should be interpreted separately from layers 21..31."
        ),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def main() -> None:
    configure_stdout()
    args = parse_args()

    from moral_mechinterp.config import load_eval_config
    from moral_mechinterp.io import ensure_dir
    from moral_mechinterp.models import unload_model
    from moral_mechinterp.scoring import resolve_score_token_ids
    from moral_mechinterp.utils import set_seed

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    config = load_eval_config(args.config)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.delta_threshold < 0:
        raise ValueError("--delta-threshold must be nonnegative.")
    if args.eps <= 0:
        raise ValueError("--eps must be positive.")

    adapters = parse_csv_list(args.adapters)
    unknown_adapters = [adapter for adapter in adapters if adapter not in {"ut", "game"}]
    if unknown_adapters:
        raise ValueError(f"Unknown adapter(s): {unknown_adapters}; expected ut and/or game.")

    subsets = load_subsets(
        data_jsonl=args.data_jsonl,
        subset_dir=args.subset_dir,
        subset_names=parse_csv_list(args.subsets),
        max_examples_per_subset=args.max_examples_per_subset,
        full_sample_size=args.full_sample_size,
        include_full=args.include_full,
        seed=args.seed,
    )
    examples = unique_examples(subsets)
    print(f"Loaded {sum(len(v) for v in subsets.values())} subset rows.")
    print(f"Unique examples requiring clean margins: {len(examples)}")

    tokenizer, model = load_base_with_adapters(config)
    try:
        n_transformer_layers = len(get_transformer_layers(model))
        layers = parse_layers(args.layers, n_transformer_layers=n_transformer_layers)
        token_ids = resolve_score_token_ids(
            tokenizer,
            config.score_tokens,
            allow_multitoken_score_labels=config.allow_multitoken_score_labels,
        )
        clean_by_id = clean_margin_records(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            config=config,
            token_ids=token_ids,
            batch_size=args.batch_size,
        )
        clean_records = [clean_by_id[example.id] for example in examples]
        write_clean_margins(clean_records, output_dir)

        summary_df = run_activation_patching(
            model=model,
            tokenizer=tokenizer,
            subsets=subsets,
            clean_by_id=clean_by_id,
            config=config,
            token_ids=token_ids,
            adapters=adapters,
            layers=layers,
            n_transformer_layers=n_transformer_layers,
            batch_size=args.batch_size,
            delta_threshold=args.delta_threshold,
            eps=args.eps,
            output_dir=output_dir,
            save_every=args.save_every,
            seed=args.seed,
        )
        if not args.no_plot:
            plot_summary(summary_df, output_dir=output_dir, font_family=config.plot_font_family)
        write_metadata(
            output_dir=output_dir,
            args=args,
            adapters=adapters,
            layers=layers,
            subsets=subsets,
            n_transformer_layers=n_transformer_layers,
        )
        print(PATCHING_PAPER_PARAGRAPH)
        print(f"Wrote activation patching artifacts to: {output_dir}")
    finally:
        unload_model(model)


if __name__ == "__main__":
    main()
