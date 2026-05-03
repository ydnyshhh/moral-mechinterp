"""Model and tokenizer loading for Hugging Face causal language models."""

from __future__ import annotations

import gc
from typing import Any


def resolve_torch_dtype(dtype_name: str) -> Any:
    import torch

    name = dtype_name.lower()
    if name in {"auto"}:
        return "auto"
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float16", "fp16", "half"}:
        return torch.float16
    if name in {"float32", "fp32", "full"}:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {dtype_name!r}")


def load_tokenizer_and_model(
    model_name_or_path: str,
    *,
    torch_dtype: str = "bfloat16",
    device_map: str | dict[str, Any] | None = "auto",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    trust_remote_code: bool = True,
) -> tuple[Any, Any]:
    """Load a tokenizer and causal LM, optionally with bitsandbytes quantization."""

    if load_in_4bit and load_in_8bit:
        raise ValueError("Choose only one of load_in_4bit or load_in_8bit")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }

    if load_in_4bit or load_in_8bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_compute_dtype=resolve_torch_dtype(torch_dtype),
        )
    else:
        model_kwargs["torch_dtype"] = resolve_torch_dtype(torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.eval()
    return tokenizer, model


def unload_model(model: Any | None) -> None:
    """Best-effort cleanup between large 9B model evaluations."""

    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass
