"""Use tqdm when installed; otherwise fall back to no-op iterators."""

from __future__ import annotations

from typing import Any, Iterable, TypeVar

T = TypeVar("T")


def try_tqdm(
    iterable: Iterable[T],
    *,
    total: int | None = None,
    desc: str | None = None,
    unit: str = "it",
    leave: bool = True,
    disable: bool = False,
    **kwargs: Any,
) -> Iterable[T]:
    if disable:
        return iterable
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc, unit=unit, leave=leave, **kwargs)
    except ImportError:
        return iterable
