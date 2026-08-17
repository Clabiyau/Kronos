"""Load mainboard daily bars from SQLite into memory."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_FEATURE_COLS = ("open", "high", "low", "close", "volume", "amount")


@dataclass
class SymbolSeries:
    symbol: str
    dates: np.ndarray
    features: np.ndarray
    suspend: np.ndarray


@dataclass
class MarketMemory:
    symbols: list[str]
    by_symbol: dict[str, SymbolSeries]
    calendar: list[str]


def load_market_memory(
    db_path: str | Path, *, lookback: int, predict: int
) -> MarketMemory:
    path = Path(db_path)
    min_valid = lookback + predict
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(
            """
            SELECT symbol, date, open, high, low, close, volume, amount,
                   COALESCE(suspendFlag, 0) AS suspendFlag
            FROM kline_daily
            WHERE symbol LIKE '60%.SH' OR symbol LIKE '00%.SZ'
            ORDER BY symbol, date
            """,
            conn,
        )
    finally:
        conn.close()

    by_symbol: dict[str, SymbolSeries] = {}
    for symbol, g in df.groupby("symbol", sort=True):
        g = g.sort_values("date")
        suspend = g["suspendFlag"].to_numpy(dtype=np.int64)
        n_valid = int(np.sum(suspend != 1))
        if n_valid < min_valid:
            continue
        dates = g["date"].to_numpy()
        features = g.loc[:, list(_FEATURE_COLS)].to_numpy(dtype=np.float64)
        by_symbol[str(symbol)] = SymbolSeries(
            symbol=str(symbol),
            dates=dates,
            features=features,
            suspend=suspend,
        )

    symbols = list(by_symbol.keys())
    if by_symbol:
        all_dates = np.concatenate([s.dates for s in by_symbol.values()])
        calendar = sorted({str(d) for d in all_dates})
    else:
        calendar = []

    return MarketMemory(symbols=symbols, by_symbol=by_symbol, calendar=calendar)
