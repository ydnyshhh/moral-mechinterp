"""Configuration loading for behavioral evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = False
    project: str = "moral-mechinterp"
    entity: str | None = None
    run_name: str | None = None


@dataclass(frozen=True)
class OutputConfig:
    behavior_dir: Path = Path("outputs/behavior")
    figures_dir: Path = Path("outputs/figures")
    tables_dir: Path = Path("outputs/tables")


@dataclass(frozen=True)
class EvalConfig:
    seed: int = 42
    max_examples: int | None = None
    shuffle: bool = False
    use_chat_template: bool = False
    torch_dtype: str = "bfloat16"
    device_map: str | dict[str, Any] | None = "auto"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    trust_remote_code: bool = True
    score_tokens: dict[str, str] = field(default_factory=lambda: {"A": " A", "B": " B"})
    allow_multitoken_score_labels: bool = False
    plot_font_family: str = "serif"
    batch_size: int = 1
    save_every: int = 25
    margin_threshold_for_strong_flips: float = 0.5
    models: dict[str, str] = field(default_factory=dict)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["outputs"] = {key: str(value) for key, value in data["outputs"].items()}
        return data


def _as_path(value: str | Path | None, default: Path) -> Path:
    if value is None:
        return default
    return Path(value)


def load_eval_config(path: str | Path) -> EvalConfig:
    """Load an eval config from YAML with typed defaults."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    wandb_raw = raw.get("wandb") or {}
    outputs_raw = raw.get("outputs") or {}

    outputs = OutputConfig(
        behavior_dir=_as_path(outputs_raw.get("behavior_dir"), OutputConfig.behavior_dir),
        figures_dir=_as_path(outputs_raw.get("figures_dir"), OutputConfig.figures_dir),
        tables_dir=_as_path(outputs_raw.get("tables_dir"), OutputConfig.tables_dir),
    )

    wandb = WandbConfig(
        enabled=bool(wandb_raw.get("enabled", WandbConfig.enabled)),
        project=str(wandb_raw.get("project", WandbConfig.project)),
        entity=wandb_raw.get("entity"),
        run_name=wandb_raw.get("run_name"),
    )

    return EvalConfig(
        seed=int(raw.get("seed", EvalConfig.seed)),
        max_examples=raw.get("max_examples", EvalConfig.max_examples),
        shuffle=bool(raw.get("shuffle", EvalConfig.shuffle)),
        use_chat_template=bool(raw.get("use_chat_template", EvalConfig.use_chat_template)),
        torch_dtype=str(raw.get("torch_dtype", EvalConfig.torch_dtype)),
        device_map=raw.get("device_map", EvalConfig.device_map),
        load_in_4bit=bool(raw.get("load_in_4bit", EvalConfig.load_in_4bit)),
        load_in_8bit=bool(raw.get("load_in_8bit", EvalConfig.load_in_8bit)),
        trust_remote_code=bool(raw.get("trust_remote_code", EvalConfig.trust_remote_code)),
        score_tokens=dict(raw.get("score_tokens") or {"A": " A", "B": " B"}),
        allow_multitoken_score_labels=bool(
            raw.get(
                "allow_multitoken_score_labels",
                EvalConfig.allow_multitoken_score_labels,
            )
        ),
        plot_font_family=str(raw.get("plot_font_family", EvalConfig.plot_font_family)),
        batch_size=int(raw.get("batch_size", EvalConfig.batch_size)),
        save_every=int(raw.get("save_every", EvalConfig.save_every)),
        margin_threshold_for_strong_flips=float(
            raw.get(
                "margin_threshold_for_strong_flips",
                EvalConfig.margin_threshold_for_strong_flips,
            )
        ),
        models=dict(raw.get("models") or {}),
        wandb=wandb,
        outputs=outputs,
    )
