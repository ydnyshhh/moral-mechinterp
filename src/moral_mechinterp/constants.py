"""Shared model identities and plotting semantics."""

from __future__ import annotations

MODEL_ORDER: tuple[str, ...] = ("base", "ut", "game")

MODEL_LABELS: dict[str, str] = {
    "base": "Base",
    "ut": "UT",
    "game": "GAME",
}

MODEL_DESCRIPTIONS: dict[str, str] = {
    "base": "Reference base model",
    "ut": "Utilitarian / harm-minimization trained model",
    "game": "Game-theoretic / strategic trained model",
}

MODEL_COLORS: dict[str, str] = {
    "base": "#252525",
    "ut": "#B85C38",
    "game": "#287D8E",
}

MODEL_MARKERS: dict[str, str] = {
    "base": "o",
    "ut": "s",
    "game": "^",
}
