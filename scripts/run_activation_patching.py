from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DEFAULT_LAYERS = "0,4,8,12,16,20,21,22,23,24,25,26,27,28,29,30,31,32,final_norm"
DEFAULT_SANITY_EXAMPLES = 8
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
    parser.add_argument(
        "--adapter-effect-threshold",
        type=float,
        default=1e-6,
        help=(
            "Abort if every clean adapter-vs-Base margin delta is at or below this "
            "absolute value for any requested adapter. This catches inactive PEFT adapters."
        ),
    )
    parser.add_argument(
        "--allow-zero-adapter-deltas",
        action="store_true",
        help="Do not abort when clean adapter-vs-Base margins are all zero.",
    )
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help=(
            "Run a 5-10 example hook sanity check over embedding, late, final-block, "
            "and final-norm sites before launching a full experiment."
        ),
    )
    parser.add_argument("--sanity-examples", type=int, default=DEFAULT_SANITY_EXAMPLES)
    parser.add_argument(
        "--include-null-controls",
        action="store_true",
        help=(
            "Also run Base-to-Base, UT-to-UT, and GAME-to-GAME null patch controls. "
            "Sanity-check mode always enables these controls."
        ),
    )
    parser.add_argument(
        "--include-shuffled-control",
        action="store_true",
        help=(
            "Also patch adapter activations from a different example into Base. "
            "This is a reviewer-facing mismatch control and increases runtime."
        ),
    )
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
    final_norm_layer = n_transformer_layers + 1
    if raw.strip().lower() == "all":
        return list(range(final_norm_layer + 1))

    layers: set[int] = set()
    for part in parse_csv_list(raw):
        if part.lower() in {"final_norm", "final-norm", "norm", "readout"}:
            layers.add(final_norm_layer)
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid layer range {part!r}: end < start.")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))

    invalid = [layer for layer in sorted(layers) if layer < 0 or layer > final_norm_layer]
    if invalid:
        raise ValueError(
            f"Requested layer(s) {invalid} are outside valid range 0-{final_norm_layer}. "
            f"For this model, layer {final_norm_layer} is the final norm/readout site."
        )
    return sorted(layers)


def active_adapter_state(model: Any) -> str:
    """Return a compact PEFT active-adapter debug string."""

    candidates = (
        getattr(model, "active_adapter", None),
        getattr(model, "active_adapters", None),
    )
    for value in candidates:
        if value is None:
            continue
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        return str(value)
    return "unavailable"


def sanity_layers(*, n_transformer_layers: int) -> list[int]:
    """Small diagnostic layer set under the explicit indexing convention.

    0 is embedding output. 1..n are transformer block outputs. n+1 is final norm.
    For 32-block Qwen models this returns 0, 21, 31, 32, 33.
    """

    requested = {0, 21, 31, n_transformer_layers, n_transformer_layers + 1}
    return sorted(layer for layer in requested if 0 <= layer <= n_transformer_layers + 1)


def layer_site(layer: int, *, n_transformer_layers: int) -> str:
    if layer == 0:
        return "embedding_output"
    if 1 <= layer <= n_transformer_layers:
        return f"block_{layer - 1}_output"
    if layer == n_transformer_layers + 1:
        return "final_norm_output"
    return "unknown"


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
    model.set_adapter(resolve_peft_adapter_name(model, mode))
    yield


def resolve_peft_adapter_name(model: Any, mode: str) -> str:
    """Map logical model keys to actual PEFT adapter names on the shared model."""

    peft_config = getattr(model, "peft_config", {})
    adapter_names = set(peft_config.keys()) if isinstance(peft_config, dict) else set()
    if mode in adapter_names:
        return mode
    if mode != "base" and adapter_names == {"default"}:
        return "default"
    raise ValueError(
        f"Could not resolve adapter mode {mode!r}. Available PEFT adapters: "
        f"{sorted(adapter_names)}"
    )


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
    if 1 <= layer <= n_transformer_layers:
        layers = get_transformer_layers(model)
        return layers[layer - 1]
    if layer == n_transformer_layers + 1:
        return get_final_norm_module(model)
    raise ValueError(
        f"Invalid patch layer {layer}; expected 0..{n_transformer_layers + 1}."
    )


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


def load_base_with_adapter(config: Any, adapter_key: str) -> tuple[Any, Any]:
    from moral_mechinterp.models import load_tokenizer_and_model

    tokenizer, model = load_tokenizer_and_model(
        config.models[adapter_key],
        torch_dtype=config.torch_dtype,
        device_map=config.device_map,
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=config.load_in_8bit,
        trust_remote_code=config.trust_remote_code,
    )
    if not hasattr(model, "load_adapter"):
        raise ValueError(
            f"Expected {adapter_key!r} to load as a PEFT model with adapter switching."
        )
    model.eval()
    print(f"Loaded PEFT adapters: {sorted(getattr(model, 'peft_config', {}).keys())}")
    print(
        f"Logical {ADAPTER_LABELS[adapter_key]} adapter resolves to: "
        f"{resolve_peft_adapter_name(model, adapter_key)}"
    )
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
    adapters: list[str],
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
            )
        }
        for adapter_key in adapters:
            batch_scores[adapter_key] = forward_ab_scores(
                model=model,
                inputs=inputs,
                mode=adapter_key,
                token_ids=token_ids,
                safe_labels=safe_labels,
            )

        for row_idx, example in enumerate(batch_examples):
            record: dict[str, Any] = {
                "example_id": example.id,
                "game_type": example.game_type,
                "safe_label": example.safe_label,
            }
            for mode in ("base", *adapters):
                score = batch_scores[mode][row_idx]
                record[f"m_{mode}"] = score["safe_margin"]
                record[f"{mode}_choice"] = score["choice"]
                record[f"{mode}_safe_choice"] = score["safe_choice"]
                record[f"{mode}_logit_A"] = score["logit_A"]
                record[f"{mode}_logit_B"] = score["logit_B"]
            for adapter_key in adapters:
                record[f"delta_{adapter_key}"] = record[f"m_{adapter_key}"] - record["m_base"]
            records[example.id] = record
    return records


def validate_clean_adapter_effects(
    clean_by_id: dict[str, dict[str, Any]],
    *,
    adapters: list[str],
    threshold: float,
    allow_zero_adapter_deltas: bool,
) -> None:
    """Fail fast if requested adapters produce Base-identical clean margins."""

    rows = list(clean_by_id.values())
    if not rows:
        raise ValueError("No clean margin records were computed.")

    summaries: list[str] = []
    inactive_adapters: list[str] = []
    for adapter_key in adapters:
        delta_col = f"delta_{adapter_key}"
        deltas = [abs(float(row[delta_col])) for row in rows]
        max_abs_delta = max(deltas)
        mean_abs_delta = sum(deltas) / len(deltas)
        summaries.append(
            f"{adapter_key}: max|delta|={max_abs_delta:.6g}, "
            f"mean|delta|={mean_abs_delta:.6g}"
        )
        if max_abs_delta <= threshold:
            inactive_adapters.append(adapter_key)

    message = "Clean adapter-vs-Base margin check: " + "; ".join(summaries)
    print(message)
    if inactive_adapters and not allow_zero_adapter_deltas:
        raise RuntimeError(
            f"Adapter(s) {inactive_adapters} produced Base-identical clean margins "
            f"for all {len(rows)} examples (threshold={threshold}). This usually "
            "means the PEFT adapters are inactive or were loaded onto the wrong base "
            "model. Aborting before the expensive patching loop. Use "
            "--allow-zero-adapter-deltas only for debugging."
        )


def merge_clean_margin_records(
    combined: dict[str, dict[str, Any]],
    new_records: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    for example_id, new_record in new_records.items():
        if example_id not in combined:
            combined[example_id] = dict(new_record)
            continue
        record = combined[example_id]
        if abs(float(record["m_base"]) - float(new_record["m_base"])) > 1e-4:
            raise RuntimeError(
                f"Base clean margin mismatch for example {example_id!r} across "
                "adapter-specific PEFT models. Check adapter base_model_name_or_path."
            )
        for key, value in new_record.items():
            if key in {"example_id", "game_type", "safe_label"}:
                continue
            if key.startswith(("m_", "delta_", "ut_", "game_")):
                record[key] = value
    return combined


def patch_result_record(
    *,
    example: Any,
    subset_name: str,
    adapter_key: str,
    patch_direction: str,
    layer: int,
    layer_site_name: str,
    clean: dict[str, Any],
    patched_score: dict[str, Any],
    delta_threshold: float,
    eps: float,
    source_example_id: str | None = None,
    is_mismatched_control: bool = False,
) -> dict[str, Any]:
    adapter_label = ADAPTER_LABELS[adapter_key]
    m_base = float(clean["m_base"])
    m_adapter = float(clean[f"m_{adapter_key}"])
    m_patched = float(patched_score["safe_margin"])
    delta_adapter = m_adapter - m_base
    denominator_ok = abs(delta_adapter) > delta_threshold
    if delta_adapter > 0:
        adapter_effect_sign = "positive"
    elif delta_adapter < 0:
        adapter_effect_sign = "negative"
    else:
        adapter_effect_sign = "zero"

    if patch_direction in {"A_to_Base", "A_shuffled_to_Base"}:
        raw_patch_effect = m_patched - m_base
    elif patch_direction == "Base_to_A":
        raw_patch_effect = m_adapter - m_patched
    else:
        raise ValueError(f"Unknown patch direction: {patch_direction!r}")

    fraction = None
    if denominator_ok:
        fraction = raw_patch_effect / (delta_adapter + math.copysign(eps, delta_adapter))
    expected_direction_recovered = raw_patch_effect * delta_adapter > 0
    source_recovery_error = m_patched - m_adapter
    base_recovery_error = m_patched - m_base

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
        "layer_site": layer_site_name,
        "is_null_control": False,
        "is_mismatched_control": is_mismatched_control,
        "source_example_id": source_example_id or example.id,
        "m_base": m_base,
        "m_adapter": m_adapter,
        "m_patched": m_patched,
        "delta_adapter": delta_adapter,
        "source_target_gap_before": delta_adapter,
        "raw_patch_effect": raw_patch_effect,
        "recovered_or_removed_fraction": fraction,
        "adapter_effect_sign": adapter_effect_sign,
        "expected_direction_recovered": expected_direction_recovered,
        "source_recovery_error": source_recovery_error,
        "base_recovery_error": base_recovery_error,
        "denominator_ok": denominator_ok,
        "safe_choice_base": bool(clean["base_safe_choice"]),
        "safe_choice_adapter": bool(clean[f"{adapter_key}_safe_choice"]),
        "safe_choice_patched": bool(patched_score["safe_choice"]),
        "choice_base": clean["base_choice"],
        "choice_adapter": clean[f"{adapter_key}_choice"],
        "choice_patched": patched_score["choice"],
    }


def null_patch_result_record(
    *,
    example: Any,
    subset_name: str,
    mode: str,
    layer: int,
    layer_site_name: str,
    clean: dict[str, Any],
    patched_score: dict[str, Any],
) -> dict[str, Any]:
    mode_label = "Base" if mode == "base" else ADAPTER_LABELS[mode]
    m_base = float(clean["m_base"])
    m_clean = float(clean[f"m_{mode}"])
    m_patched = float(patched_score["safe_margin"])
    patch_error = m_patched - m_clean
    return {
        "example_id": example.id,
        "subset_name": subset_name,
        "subset_label": SUBSET_LABELS.get(subset_name, subset_name),
        "game_type": example.game_type,
        "safe_label": example.safe_label,
        "adapter_name": mode_label,
        "adapter_key": mode,
        "patch_direction": f"{mode_label}_to_{mode_label}",
        "layer": layer,
        "layer_site": layer_site_name,
        "is_null_control": True,
        "is_mismatched_control": False,
        "source_example_id": example.id,
        "m_base": m_base,
        "m_adapter": m_clean,
        "m_patched": m_patched,
        "delta_adapter": m_clean - m_base,
        "source_target_gap_before": 0.0,
        "raw_patch_effect": patch_error,
        "recovered_or_removed_fraction": None,
        "adapter_effect_sign": "null_control",
        "expected_direction_recovered": None,
        "source_recovery_error": patch_error,
        "base_recovery_error": m_patched - m_base,
        "denominator_ok": False,
        "safe_choice_base": bool(clean["base_safe_choice"]),
        "safe_choice_adapter": bool(clean[f"{mode}_safe_choice"]),
        "safe_choice_patched": bool(patched_score["safe_choice"]),
        "choice_base": clean["base_choice"],
        "choice_adapter": clean[f"{mode}_choice"],
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
    group_cols = [
        "adapter_name",
        "patch_direction",
        "subset_name",
        "subset_label",
        "layer",
        "layer_site",
        "is_null_control",
        "is_mismatched_control",
    ]
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
                "layer_site": group_key[5],
                "is_null_control": bool(group_key[6]),
                "is_mismatched_control": bool(group_key[7]),
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
                "mean_source_recovery_error": pd.to_numeric(
                    group["source_recovery_error"],
                    errors="coerce",
                ).mean(),
                "mean_base_recovery_error": pd.to_numeric(
                    group["base_recovery_error"],
                    errors="coerce",
                ).mean(),
                "expected_direction_recovered_rate": pd.to_numeric(
                    denom_ok["expected_direction_recovered"],
                    errors="coerce",
                ).mean(),
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
    include_null_controls: bool,
    include_shuffled_control: bool,
    initial_records: list[dict[str, Any]] | None = None,
) -> Any:
    from tqdm.auto import tqdm

    from moral_mechinterp.utils import batched

    records: list[dict[str, Any]] = list(initial_records or [])
    completed_batches = 0
    total_steps = sum(
        math.ceil(len(examples) / batch_size) * len(adapters) * len(PATCH_DIRECTIONS) * len(layers)
        for examples in subsets.values()
    )
    if include_null_controls:
        total_steps += sum(
            math.ceil(len(examples) / batch_size) * 3 * len(layers)
            for examples in subsets.values()
        )
    if include_shuffled_control:
        total_steps += sum(
            math.ceil(len(examples) / batch_size) * len(adapters) * len(layers)
            for examples in subsets.values()
            if len(examples) > 1
        )
    progress = tqdm(total=total_steps, desc="Activation patching", unit="batch")

    for subset_name, examples in subsets.items():
        for layer in layers:
            layer_site_name = layer_site(layer, n_transformer_layers=n_transformer_layers)
            batches = list(batched(examples, batch_size))
            for batch_index, batch_examples in enumerate(batches):
                prompts = [
                    build_prompt(tokenizer, example, config)
                    for example in batch_examples
                ]
                safe_labels = [example.safe_label for example in batch_examples]
                inputs = tokenize_prompts(tokenizer, prompts, model)

                base_activation = capture_layer_activation(
                    model=model,
                    inputs=inputs,
                    mode="base",
                    layer=layer,
                    n_transformer_layers=n_transformer_layers,
                )
                if include_null_controls:
                    for mode in ("base", *adapters):
                        clean_activation = base_activation
                        if mode != "base":
                            clean_activation = capture_layer_activation(
                                model=model,
                                inputs=inputs,
                                mode=mode,
                                layer=layer,
                                n_transformer_layers=n_transformer_layers,
                            )
                        patched_clean_scores = patched_forward_ab_scores(
                            model=model,
                            inputs=inputs,
                            target_mode=mode,
                            layer=layer,
                            n_transformer_layers=n_transformer_layers,
                            source_activation=clean_activation,
                            token_ids=token_ids,
                            safe_labels=safe_labels,
                        )
                        for example, patched_score in zip(
                            batch_examples,
                            patched_clean_scores,
                            strict=True,
                        ):
                            records.append(
                                null_patch_result_record(
                                    example=example,
                                    subset_name=subset_name,
                                    mode=mode,
                                    layer=layer,
                                    layer_site_name=layer_site_name,
                                    clean=clean_by_id[example.id],
                                    patched_score=patched_score,
                                )
                            )
                        completed_batches += 1
                        progress.update(1)

                for adapter_key in adapters:
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
                                layer_site_name=layer_site_name,
                                clean=clean_by_id[example.id],
                                patched_score=patched_score,
                                delta_threshold=delta_threshold,
                                eps=eps,
                            )
                        )
                    completed_batches += 1
                    progress.update(1)

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
                                layer_site_name=layer_site_name,
                                clean=clean_by_id[example.id],
                                patched_score=patched_score,
                                delta_threshold=delta_threshold,
                                eps=eps,
                            )
                        )
                    completed_batches += 1
                    progress.update(1)

                    if include_shuffled_control and len(examples) > 1:
                        start = batch_index * batch_size
                        source_examples = [
                            examples[(start + offset + 1) % len(examples)]
                            for offset in range(len(batch_examples))
                        ]
                        source_prompts = [
                            build_prompt(tokenizer, example, config)
                            for example in source_examples
                        ]
                        source_inputs = tokenize_prompts(tokenizer, source_prompts, model)
                        shuffled_activation = capture_layer_activation(
                            model=model,
                            inputs=source_inputs,
                            mode=adapter_key,
                            layer=layer,
                            n_transformer_layers=n_transformer_layers,
                        )
                        patched_shuffled_scores = patched_forward_ab_scores(
                            model=model,
                            inputs=inputs,
                            target_mode="base",
                            layer=layer,
                            n_transformer_layers=n_transformer_layers,
                            source_activation=shuffled_activation,
                            token_ids=token_ids,
                            safe_labels=safe_labels,
                        )
                        for example, source_example, patched_score in zip(
                            batch_examples,
                            source_examples,
                            patched_shuffled_scores,
                            strict=True,
                        ):
                            records.append(
                                patch_result_record(
                                    example=example,
                                    subset_name=subset_name,
                                    adapter_key=adapter_key,
                                    patch_direction="A_shuffled_to_Base",
                                    layer=layer,
                                    layer_site_name=layer_site_name,
                                    clean=clean_by_id[example.id],
                                    patched_score=patched_score,
                                    delta_threshold=delta_threshold,
                                    eps=eps,
                                    source_example_id=source_example.id,
                                    is_mismatched_control=True,
                                )
                            )
                        completed_batches += 1
                        progress.update(1)

                    if save_every > 0 and completed_batches % save_every == 0:
                        save_patch_outputs(records, output_dir=output_dir, seed=seed)

    progress.close()
    summary_df = save_patch_outputs(records, output_dir=output_dir, seed=seed)
    return summary_df, records


def line_label(row: Any) -> str:
    adapter = row.adapter_name
    if row.patch_direction == "A_shuffled_to_Base":
        return f"{adapter} shuffled->Base"
    if row.patch_direction == "Base_to_A":
        return f"Base->{adapter}"
    if row.patch_direction == "A_to_Base":
        return f"{adapter}→Base"
    return f"Base→{adapter}"


def plot_summary(
    summary_df: Any,
    *,
    output_dir: Path,
    font_family: str,
    final_norm_layer: int,
) -> None:
    if summary_df.empty:
        return
    primary_df = summary_df[
        (~summary_df["is_null_control"].astype(bool))
        & (~summary_df["is_mismatched_control"].astype(bool))
    ]
    plot_fraction(
        primary_df[primary_df["patch_direction"] == "A_to_Base"],
        output_dir=output_dir,
        filename="fig_recovered_fraction_by_layer",
        ylabel="Recovered fraction",
        title="Adapter-to-Base recovered fraction",
        font_family=font_family,
        final_norm_layer=final_norm_layer,
    )
    plot_fraction(
        primary_df[primary_df["patch_direction"] == "Base_to_A"],
        output_dir=output_dir,
        filename="fig_removed_fraction_by_layer",
        ylabel="Removed fraction",
        title="Base-to-adapter removed fraction",
        font_family=font_family,
        final_norm_layer=final_norm_layer,
    )
    plot_raw_effects(
        primary_df,
        output_dir=output_dir,
        font_family=font_family,
        final_norm_layer=final_norm_layer,
    )


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
    final_norm_layer: int,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

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
            fraction = np.clip(
                ordered["mean_recovered_or_removed_fraction"].to_numpy(dtype=float),
                -2.0,
                2.0,
            )
            ci_low = np.clip(
                ordered["fraction_ci_low"].to_numpy(dtype=float),
                -2.0,
                2.0,
            )
            ci_high = np.clip(
                ordered["fraction_ci_high"].to_numpy(dtype=float),
                -2.0,
                2.0,
            )
            ax.plot(
                ordered["layer"],
                fraction,
                color=MODEL_COLORS.get(adapter_key, "#252525"),
                marker=MODEL_MARKERS.get(adapter_key, "o"),
                linewidth=1.35,
                markersize=3.3,
                label=line_label(ordered.iloc[0]),
            )
            ax.fill_between(
                ordered["layer"],
                ci_low,
                ci_high,
                color=MODEL_COLORS.get(adapter_key, "#252525"),
                alpha=0.14,
                linewidth=0,
            )
        ax.axhline(0, color="#2A2A2A", linewidth=0.8, linestyle=(0, (3, 2)))
        ax.axvspan(22, final_norm_layer - 1, color="#E8E8E8", alpha=0.42, linewidth=0)
        ax.axvline(final_norm_layer, color="#2A2A2A", linewidth=0.85, linestyle=(0, (2, 2)))
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


def plot_raw_effects(
    summary_df: Any,
    *,
    output_dir: Path,
    font_family: str,
    final_norm_layer: int,
) -> None:
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
        ax.axvspan(22, final_norm_layer - 1, color="#E8E8E8", alpha=0.42, linewidth=0)
        ax.axvline(final_norm_layer, color="#2A2A2A", linewidth=0.85, linestyle=(0, (2, 2)))
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


def mean_abs_column(df: Any, column: str) -> float | None:
    pd = load_pandas()

    if df.empty or column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna().abs()
    if values.empty:
        return None
    return float(values.mean())


def write_sanity_report(
    *,
    output_dir: Path,
    result_df: Any,
    final_norm_layer: int,
    threshold: float = 1e-3,
) -> None:
    pd = load_pandas()

    checks = [
        {
            "check": "UT->Base final norm recovers UT margin",
            "filter": (
                (result_df["adapter_key"] == "ut")
                & (result_df["patch_direction"] == "A_to_Base")
                & (result_df["layer"] == final_norm_layer)
            ),
            "column": "source_recovery_error",
        },
        {
            "check": "GAME->Base final norm recovers GAME margin",
            "filter": (
                (result_df["adapter_key"] == "game")
                & (result_df["patch_direction"] == "A_to_Base")
                & (result_df["layer"] == final_norm_layer)
            ),
            "column": "source_recovery_error",
        },
        {
            "check": "Base->UT final norm recovers Base margin",
            "filter": (
                (result_df["adapter_key"] == "ut")
                & (result_df["patch_direction"] == "Base_to_A")
                & (result_df["layer"] == final_norm_layer)
            ),
            "column": "base_recovery_error",
        },
        {
            "check": "Base->GAME final norm recovers Base margin",
            "filter": (
                (result_df["adapter_key"] == "game")
                & (result_df["patch_direction"] == "Base_to_A")
                & (result_df["layer"] == final_norm_layer)
            ),
            "column": "base_recovery_error",
        },
        {
            "check": "Layer 0 UT->Base has near-zero effect",
            "filter": (
                (result_df["adapter_key"] == "ut")
                & (result_df["patch_direction"] == "A_to_Base")
                & (result_df["layer"] == 0)
            ),
            "column": "base_recovery_error",
        },
        {
            "check": "Layer 0 GAME->Base has near-zero effect",
            "filter": (
                (result_df["adapter_key"] == "game")
                & (result_df["patch_direction"] == "A_to_Base")
                & (result_df["layer"] == 0)
            ),
            "column": "base_recovery_error",
        },
        {
            "check": "Layer 0 Base->UT has near-zero effect",
            "filter": (
                (result_df["adapter_key"] == "ut")
                & (result_df["patch_direction"] == "Base_to_A")
                & (result_df["layer"] == 0)
            ),
            "column": "raw_patch_effect",
        },
        {
            "check": "Layer 0 Base->GAME has near-zero effect",
            "filter": (
                (result_df["adapter_key"] == "game")
                & (result_df["patch_direction"] == "Base_to_A")
                & (result_df["layer"] == 0)
            ),
            "column": "raw_patch_effect",
        },
        {
            "check": "Null controls reproduce clean margins",
            "filter": result_df["is_null_control"].astype(bool),
            "column": "source_recovery_error",
        },
    ]
    rows: list[dict[str, Any]] = []
    for item in checks:
        subset = result_df[item["filter"]]
        mean_abs_error = mean_abs_column(subset, str(item["column"]))
        rows.append(
            {
                "check": item["check"],
                "metric_column": item["column"],
                "n": int(len(subset)),
                "mean_abs_error": mean_abs_error,
                "threshold": threshold,
                "passed": bool(mean_abs_error is not None and mean_abs_error < threshold),
            }
        )

    report_df = pd.DataFrame(rows)
    report_df.to_csv(output_dir / "sanity_check_report.csv", index=False)
    markdown_lines = [
        "| Check | n | Mean abs error | Threshold | Passed |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        value = row["mean_abs_error"]
        formatted = "NA" if value is None else f"{value:.6g}"
        markdown_lines.append(
            "| {check} | {n} | {mean_abs_error} | {threshold:.1e} | {passed} |".format(
                check=row["check"],
                n=row["n"],
                mean_abs_error=formatted,
                threshold=row["threshold"],
                passed=row["passed"],
            )
        )
    (output_dir / "sanity_check_report.md").write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )
    print("Sanity-check report:")
    print(report_df.to_string(index=False))


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
        "adapter_loading": (
            "Adapters are processed one at a time in isolated PEFT models. "
            "Each adapter is loaded through the same adapter-loader path used by "
            "behavioral evaluation, then Base mode uses disable_adapter(). This "
            "avoids fragile multi-adapter loading where a second adapter can be "
            "silently inactive."
        ),
        "layers": layers,
        "n_transformer_layers": n_transformer_layers,
        "final_norm_layer": n_transformer_layers + 1,
        "delta_threshold": args.delta_threshold,
        "adapter_effect_threshold": args.adapter_effect_threshold,
        "allow_zero_adapter_deltas": bool(args.allow_zero_adapter_deltas),
        "eps": args.eps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "sanity_check": bool(args.sanity_check),
        "sanity_examples": args.sanity_examples,
        "include_null_controls": bool(args.include_null_controls or args.sanity_check),
        "include_shuffled_control": bool(args.include_shuffled_control),
        "token_position": "final non-padding token: attention_mask.sum(dim=-1) - 1",
        "layer_interpretation": (
            "Layer 0 patches the embedding output. Layers 1..L patch transformer "
            "block outputs, where layer ell replaces the output of block ell-1. "
            "Layer L+1 patches the final normalized hidden state immediately before "
            "the LM head and is a readout sanity check, not a transformer block. "
            "For a 32-block Qwen model, layer 32 is block 31 output and layer 33 "
            "is final norm/readout."
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
    if args.sanity_check and args.output_dir == Path("artifacts/patching"):
        args.output_dir = Path("artifacts/patching_sanity")
    output_dir = ensure_dir(args.output_dir)
    config = load_eval_config(args.config)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.delta_threshold < 0:
        raise ValueError("--delta-threshold must be nonnegative.")
    if args.adapter_effect_threshold < 0:
        raise ValueError("--adapter-effect-threshold must be nonnegative.")
    if args.eps <= 0:
        raise ValueError("--eps must be positive.")

    adapters = parse_csv_list(args.adapters)
    unknown_adapters = [adapter for adapter in adapters if adapter not in {"ut", "game"}]
    if unknown_adapters:
        raise ValueError(f"Unknown adapter(s): {unknown_adapters}; expected ut and/or game.")

    subset_names = parse_csv_list(args.subsets)
    if args.sanity_check:
        subset_names = subset_names[:1]
        print(
            f"Sanity-check mode: using subset {subset_names[0]!r}, "
            f"{args.sanity_examples} examples, and diagnostic layers."
        )

    subsets = load_subsets(
        data_jsonl=args.data_jsonl,
        subset_dir=args.subset_dir,
        subset_names=subset_names,
        max_examples_per_subset=(
            args.sanity_examples if args.sanity_check else args.max_examples_per_subset
        ),
        full_sample_size=args.full_sample_size,
        include_full=args.include_full,
        seed=args.seed,
    )
    examples = unique_examples(subsets)
    print(f"Loaded {sum(len(v) for v in subsets.values())} subset rows.")
    print(f"Unique examples requiring clean margins: {len(examples)}")

    combined_clean_by_id: dict[str, dict[str, Any]] = {}
    patch_records: list[dict[str, Any]] = []
    summary_df = None
    layers: list[int] | None = None
    n_transformer_layers: int | None = None
    final_norm_layer: int | None = None

    for adapter_key in adapters:
        print(f"=== Adapter-specific patching run: {ADAPTER_LABELS[adapter_key]} ===")
        tokenizer, model = load_base_with_adapter(config, adapter_key)
        try:
            print(f"PEFT active adapter after load: {active_adapter_state(model)}")
            current_n_layers = len(get_transformer_layers(model))
            if n_transformer_layers is None:
                n_transformer_layers = current_n_layers
                final_norm_layer = n_transformer_layers + 1
                if args.sanity_check:
                    layers = sanity_layers(n_transformer_layers=n_transformer_layers)
                else:
                    layers = parse_layers(args.layers, n_transformer_layers=n_transformer_layers)
                print(
                    "Layer convention: 0=embedding, "
                    f"1-{n_transformer_layers}=block outputs, "
                    f"{final_norm_layer}=final norm/readout."
                )
            elif current_n_layers != n_transformer_layers:
                raise RuntimeError(
                    f"Adapter {adapter_key!r} has {current_n_layers} transformer layers, "
                    f"but previous adapter had {n_transformer_layers}."
                )

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
                adapters=[adapter_key],
                batch_size=args.batch_size,
            )
            validate_clean_adapter_effects(
                clean_by_id,
                adapters=[adapter_key],
                threshold=args.adapter_effect_threshold,
                allow_zero_adapter_deltas=args.allow_zero_adapter_deltas,
            )
            combined_clean_by_id = merge_clean_margin_records(combined_clean_by_id, clean_by_id)
            clean_records = [combined_clean_by_id[example.id] for example in examples]
            write_clean_margins(clean_records, output_dir)

            summary_df, patch_records = run_activation_patching(
                model=model,
                tokenizer=tokenizer,
                subsets=subsets,
                clean_by_id=clean_by_id,
                config=config,
                token_ids=token_ids,
                adapters=[adapter_key],
                layers=layers or [],
                n_transformer_layers=n_transformer_layers,
                batch_size=args.batch_size,
                delta_threshold=args.delta_threshold,
                eps=args.eps,
                output_dir=output_dir,
                save_every=args.save_every,
                seed=args.seed,
                include_null_controls=args.include_null_controls or args.sanity_check,
                include_shuffled_control=args.include_shuffled_control,
                initial_records=patch_records,
            )
        finally:
            unload_model(model)

    if (
        summary_df is None
        or n_transformer_layers is None
        or final_norm_layer is None
        or layers is None
    ):
        raise RuntimeError("No adapter-specific patching runs completed.")

    result_df = load_pandas().read_csv(output_dir / "activation_patching_results.csv")
    if args.sanity_check:
        write_sanity_report(
            output_dir=output_dir,
            result_df=result_df,
            final_norm_layer=final_norm_layer,
        )
    if not args.no_plot and not args.sanity_check:
        plot_summary(
            summary_df,
            output_dir=output_dir,
            font_family=config.plot_font_family,
            final_norm_layer=final_norm_layer,
        )
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


if __name__ == "__main__":
    main()
