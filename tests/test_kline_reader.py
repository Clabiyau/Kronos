"""Tests for kline.db reader."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from inference.kline_reader import KlineDbReader, normalize_symbol


def _seed_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE kline_daily ("
        "symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, "
        "volume REAL, amount REAL, suspendFlag INTEGER)"
    )
    rows = []
    dates = pd.bdate_range("2024-01-01", periods=420)
    for i, dt in enumerate(dates):
        px = 10.0 + i * 0.01
        rows.append(
            (
                "600519.SH",
                dt.strftime("%Y-%m-%d"),
                px,
                px + 0.5,
                px - 0.5,
                px + 0.1,
                1000.0,
                10000.0,
                0,
            )
        )
    conn.executemany(
        "INSERT INTO kline_daily VALUES (?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


def test_normalize_symbol():
    assert normalize_symbol("600519") == "600519.SH"
    assert normalize_symbol("600519.SH") == "600519.SH"


def test_load_recent_daily(tmp_path: Path):
    db_path = tmp_path / "kline.db"
    _seed_db(db_path)
    reader = KlineDbReader(db_path)

    df = reader.load_recent_daily("600519", lookback=400)
    assert len(df) == 400
    assert list(df.columns[:6]) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    assert df["date"].is_monotonic_increasing


def test_load_recent_daily_insufficient(tmp_path: Path):
    db_path = tmp_path / "kline.db"
    _seed_db(db_path)
    reader = KlineDbReader(db_path)

    with pytest.raises(ValueError, match="need 100 rows"):
        reader.load_recent_daily("600519", lookback=100, end_date="2024-01-15")
