"""Small shared utilities."""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from typing import TypeVar

import numpy as np

T = TypeVar("T")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def batched(items: list[T], batch_size: int) -> Iterator[list[T]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def as_list(items: Iterable[T]) -> list[T]:
    return list(items)
