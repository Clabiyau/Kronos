from __future__ import annotations

import random
from typing import Literal


def build_val_symbol_sets(
    calendar: list[str],
    symbols: list[str],
    *,
    backtest_start: str,
    block_days: int = 100,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> list[tuple[str, str, frozenset[str]]]:
    """Build rotating validation symbol sets over calendar blocks.

    Callers must pass only pre-backtest trading dates in ``calendar``.
    ``backtest_start`` is part of the public API for downstream callers.
    """
    _ = backtest_start

    if not calendar:
        return []

    sorted_dates = sorted(calendar)
    val_k = 0 if not symbols else max(1, int(round(len(symbols) * val_ratio)))
    sorted_symbols = sorted(symbols)

    blocks: list[tuple[str, str, frozenset[str]]] = []
    block_index = 0
    for start in range(0, len(sorted_dates), block_days):
        chunk = sorted_dates[start : start + block_days]
        rng = random.Random(seed + block_index)
        val_symbols = frozenset(rng.sample(sorted_symbols, k=val_k)) if symbols else frozenset()
        blocks.append((chunk[0], chunk[-1], val_symbols))
        block_index += 1

    return blocks


def assign_split(
    asof: str,
    symbol: str,
    blocks: list[tuple[str, str, frozenset[str]]],
    backtest_start: str,
) -> Literal["train", "val", "backtest"]:
    if asof >= backtest_start:
        return "backtest"

    for block_start, block_end, val_symbols in blocks:
        if block_start <= asof <= block_end:
            return "val" if symbol in val_symbols else "train"

    raise ValueError(f"asof {asof!r} not covered by any block")
