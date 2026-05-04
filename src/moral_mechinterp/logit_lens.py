"""Layerwise logit-lens utilities for safe-action margin trajectories.

Layerwise logit-lens margins measure how strongly each intermediate layer's
final-token hidden state linearly supports the safe option over the harmful
option. Intermediate residual streams are passed through the model's final
normalization before unembedding, but the last hidden-state entry is treated as
already final-normalized for Qwen/LLaMA-style Hugging Face decoder models. This
is a diagnostic of decision-evidence trajectories, not a causal intervention.
"""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd

from moral_mechinterp.constants import MODEL_COLORS, MODEL_LABELS, MODEL_MARKERS, MODEL_ORDER
from moral_mechinterp.metrics import bootstrap_ci
from moral_mechinterp.plot_style import apply_paper_style, despine, save_figure


def infer_input_device(model: Any) -> Any:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return getattr(model, "device", "cpu")


def get_by_path(obj: Any, path: str) -> Any | None:
    current = obj
    for part in path.split("."):
        if current is None or not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def get_lm_head(model: Any) -> Any:
    """Return the LM head for full HF models and PEFT-wrapped models."""

    if hasattr(model, "get_output_embeddings"):
        head = model.get_output_embeddings()
        if head is not None:
            return head

    candidate_paths = (
        "lm_head",
        "model.lm_head",
        "base_model.model.lm_head",
        "base_model.model.model.lm_head",
    )
    for path in candidate_paths:
        head = get_by_path(model, path)
        if head is not None:
            return head

    if hasattr(model, "get_base_model"):
        base_model = model.get_base_model()
        for path in ("lm_head", "model.lm_head"):
            head = get_by_path(base_model, path)
            if head is not None:
                return head

    raise ValueError("Could not find an LM head on the model or PEFT-wrapped base model.")


def get_final_norm(model: Any) -> Any | None:
    """Return the final transformer norm when available."""

    candidate_paths = (
        "model.norm",
        "norm",
        "base_model.model.model.norm",
        "base_model.model.norm",
        "base_model.model.model.model.norm",
    )
    for path in candidate_paths:
        norm = get_by_path(model, path)
        if norm is not None:
            return norm

    if hasattr(model, "get_base_model"):
        base_model = model.get_base_model()
        for path in ("model.norm", "norm"):
            norm = get_by_path(base_model, path)
            if norm is not None:
                return norm

    warnings.warn(
        "Could not find a final model norm; continuing logit lens with unnormalized hidden states.",
        stacklevel=2,
    )
    return None


def project_hidden_states_to_ab_logits(
    hidden_state_batch: Any,
    *,
    final_norm: Any | None,
    lm_head: Any,
    token_ids: dict[str, int],
    apply_final_norm: bool = True,
) -> list[tuple[float, float]]:
    """Project a batch of final-token hidden states to only A/B logits."""

    import torch

    hidden = hidden_state_batch
    if apply_final_norm and final_norm is not None:
        norm_params = list(final_norm.parameters())
        if norm_params:
            hidden = hidden.to(norm_params[0].device)
        hidden = final_norm(hidden)

    weight = lm_head.weight
    head_device = weight.device
    ids = torch.tensor([token_ids["A"], token_ids["B"]], device=head_device)
    selected_weight = weight.index_select(0, ids)
    if not selected_weight.is_floating_point():
        selected_weight = selected_weight.float()

    hidden = hidden.to(device=head_device, dtype=selected_weight.dtype)
    logits = hidden @ selected_weight.transpose(0, 1)

    bias = getattr(lm_head, "bias", None)
    if bias is not None:
        selected_bias = bias.index_select(0, ids).to(device=head_device, dtype=logits.dtype)
        logits = logits + selected_bias

    logits = logits.detach().float().cpu()
    return [(float(row[0].item()), float(row[1].item())) for row in logits]


def compute_batch_layer_margins(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    safe_labels: list[str],
    token_ids: dict[str, int],
    final_norm: Any | None,
    lm_head: Any,
) -> list[list[dict[str, float | int | str]]]:
    """Compute layerwise A/B logits and safe margins for each prompt in a batch.

    Layer 0 is the embedding output, layers 1..L are transformer block outputs,
    and the last hidden-state entry is assumed to already include the model's
    final normalization for Qwen/LLaMA-style HF decoder models. Therefore the
    last layer skips the extra final norm. Its margin should closely match the
    behavioral final-logit safe margin when prompts and A/B token ids match.
    """

    import torch

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    device = infer_input_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
        )

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise ValueError("Model did not return hidden_states despite output_hidden_states=True.")

    attention_mask = inputs["attention_mask"]
    last_indices = attention_mask.shape[1] - 1 - attention_mask.flip(dims=[1]).argmax(dim=1)
    batch_indices = torch.arange(attention_mask.shape[0], device=attention_mask.device)

    per_example: list[list[dict[str, float | int | str]]] = [
        [] for _ in range(len(prompts))
    ]
    num_layers = len(hidden_states)
    for layer_idx, layer_hidden in enumerate(hidden_states):
        final_token_hidden = layer_hidden[
            batch_indices.to(layer_hidden.device),
            last_indices.to(layer_hidden.device),
            :,
        ]
        apply_norm = final_norm is not None and layer_idx != num_layers - 1
        ab_logits = project_hidden_states_to_ab_logits(
            final_token_hidden,
            final_norm=final_norm,
            lm_head=lm_head,
            token_ids=token_ids,
            apply_final_norm=apply_norm,
        )
        for batch_idx, (logit_a, logit_b) in enumerate(ab_logits):
            safe_label = safe_labels[batch_idx]
            if safe_label == "A":
                safe_margin = logit_a - logit_b
            elif safe_label == "B":
                safe_margin = logit_b - logit_a
            else:
                raise ValueError(f"safe_label must be A or B, got {safe_label!r}")

            per_example[batch_idx].append(
                {
                    "layer": layer_idx,
                    "logit_A": logit_a,
                    "logit_B": logit_b,
                    "safe_margin": safe_margin,
                }
            )

    return per_example


def summarize_layer_margins(
    layer_df: pd.DataFrame,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["subset_name", "model", "model_label", "layer"]
    for group_key, group in layer_df.groupby(group_cols, sort=True, dropna=False):
        values = pd.to_numeric(group["safe_margin"], errors="coerce").dropna()
        ci_low, ci_high = bootstrap_ci(values, seed=seed)
        rows.append(
            {
                "subset_name": group_key[0],
                "model": group_key[1],
                "model_label": group_key[2],
                "layer": int(group_key[3]),
                "mean_safe_margin": values.mean(),
                "median_safe_margin": values.median(),
                "std_safe_margin": values.std(ddof=1),
                "sem_safe_margin": values.sem(ddof=1),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n": int(values.count()),
            }
        )
    return pd.DataFrame(rows)


def ordered_layer_margin_columns() -> list[str]:
    return [
        "subset_name",
        "id",
        "game_type",
        "model",
        "model_label",
        "layer",
        "safe_label",
        "safe_token",
        "harmful_token",
        "logit_A",
        "logit_B",
        "safe_margin",
        "final_behavior_safe_margin",
        "final_behavior_choice",
        "final_behavior_safe",
    ]


def plot_layer_margin_summary(
    summary_df: pd.DataFrame,
    *,
    subset_name: str,
    output_dir: str,
    font_family: str = "serif",
) -> list[Any]:
    import matplotlib.pyplot as plt

    apply_paper_style(font_family=font_family)
    fig, ax = plt.subplots(figsize=(4.7, 3.0))

    for model_key in MODEL_ORDER:
        model_df = summary_df[summary_df["model"] == model_key].sort_values("layer")
        if model_df.empty:
            continue
        x = model_df["layer"].to_numpy(dtype=float)
        y = model_df["mean_safe_margin"].to_numpy(dtype=float)
        low = model_df["ci_low"].to_numpy(dtype=float)
        high = model_df["ci_high"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            color=MODEL_COLORS[model_key],
            marker=MODEL_MARKERS[model_key],
            markevery=max(1, len(x) // 8),
            linewidth=1.4,
            markersize=3.8,
            label=MODEL_LABELS[model_key],
        )
        ax.fill_between(
            x,
            low,
            high,
            color=MODEL_COLORS[model_key],
            alpha=0.16,
            linewidth=0,
        )

    ax.axhline(0, color="#2A2A2A", linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean safe-action logit-lens margin")
    ax.set_title(f"Layerwise safe evidence: {subset_name}")
    ax.legend(frameon=False)
    ax.yaxis.grid(True)
    despine(ax)
    return save_figure(fig, f"{output_dir}/logit_lens_{subset_name}")
