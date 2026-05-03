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
    """Load a full causal LM or a PEFT adapter-wrapped causal LM."""

    if load_in_4bit and load_in_8bit:
        raise ValueError("Choose only one of load_in_4bit or load_in_8bit")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_kwargs = _build_model_kwargs(
        torch_dtype=torch_dtype,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        trust_remote_code=trust_remote_code,
    )

    try:
        tokenizer = _load_tokenizer(
            AutoTokenizer,
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    except ValueError as exc:
        if not _looks_like_missing_model_type_error(exc):
            raise
        tokenizer, model = _load_peft_adapter_model(
            adapter_name_or_path=model_name_or_path,
            model_kwargs=model_kwargs,
            trust_remote_code=trust_remote_code,
            auto_tokenizer_cls=AutoTokenizer,
            auto_model_cls=AutoModelForCausalLM,
        )

    model.eval()
    return tokenizer, model


def _build_model_kwargs(
    *,
    torch_dtype: str,
    device_map: str | dict[str, Any] | None,
    load_in_4bit: bool,
    load_in_8bit: bool,
    trust_remote_code: bool,
) -> dict[str, Any]:
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

    return model_kwargs


def _load_tokenizer(
    auto_tokenizer_cls: Any,
    model_name_or_path: str,
    *,
    trust_remote_code: bool,
) -> Any:
    tokenizer = auto_tokenizer_cls.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _looks_like_missing_model_type_error(exc: ValueError) -> bool:
    message = str(exc).lower()
    return "unrecognized model" in message or "model_type" in message


def _load_peft_adapter_model(
    *,
    adapter_name_or_path: str,
    model_kwargs: dict[str, Any],
    trust_remote_code: bool,
    auto_tokenizer_cls: Any,
    auto_model_cls: Any,
) -> tuple[Any, Any]:
    from peft import PeftConfig, PeftModel

    peft_config = PeftConfig.from_pretrained(adapter_name_or_path)
    base_model_name_or_path = peft_config.base_model_name_or_path
    if not base_model_name_or_path:
        raise ValueError(
            f"PEFT adapter {adapter_name_or_path!r} does not declare "
            "base_model_name_or_path in adapter_config.json."
        )

    print(
        "Detected PEFT adapter checkpoint; loading base model "
        f"{base_model_name_or_path!r} and adapter {adapter_name_or_path!r}."
    )
    tokenizer = _load_tokenizer(
        auto_tokenizer_cls,
        base_model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    base_model = auto_model_cls.from_pretrained(base_model_name_or_path, **model_kwargs)
    model = PeftModel.from_pretrained(
        base_model,
        adapter_name_or_path,
        is_trainable=False,
    )
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
