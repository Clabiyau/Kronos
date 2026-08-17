"""Read daily K-lines from stock_qmt ``kline.db`` (read-only)."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_KLINE_DB = _PROJECT_ROOT.parent / "stock_qmt" / "storage" / "kline.db"

_PURE_DIGIT = re.compile(r"^\d{6}$")
_SELECT_COLS = (
    "date, open, high, low, close, volume, amount, "
    "COALESCE(suspendFlag, 0) AS suspendFlag"
)


def normalize_symbol(symbol: str) -> str:
    """Convert common code formats to QMT ``code.market``."""
    s = symbol.strip().upper()
    if "." in s:
        return s
    if not _PURE_DIGIT.match(s):
        raise ValueError(f"Unsupported symbol: {symbol!r}")

    if s.startswith(("60", "68", "11", "13")):
        return f"{s}.SH"
    if s.startswith(("00", "30", "12", "15")):
        return f"{s}.SZ"
    if s.startswith(("8", "4")):
        return f"{s}.BJ"
    raise ValueError(f"Unrecognized symbol: {symbol!r}")


def _fmt_date(value: object) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def open_kline_db(db_path: Path | str) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"kline.db not found: {path}")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    return conn


class KlineDbReader:
    """Minimal read-only accessor for ``kline_daily``."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._db_path = Path(db_path or _DEFAULT_KLINE_DB)

    @property
    def db_path(self) -> Path:
        return self._db_path

    def load_recent_daily(
        self,
        symbol: str,
        *,
        lookback: int,
        end_date: str | pd.Timestamp | None = None,
        include_suspended: bool = False,
    ) -> pd.DataFrame:
        """Load ascending daily bars ending at ``end_date`` (latest if omitted)."""
        if lookback < 1:
            raise ValueError("lookback must be >= 1")

        sym = normalize_symbol(symbol)
        end_str = _fmt_date(end_date) if end_date is not None else None
        suspend_clause = (
            "" if include_suspended else " AND COALESCE(suspendFlag, 0) != 1"
        )

        with open_kline_db(self._db_path) as conn:
            if end_str is None:
                row = conn.execute(
                    "SELECT MAX(date) FROM kline_daily WHERE symbol=?",
                    (sym,),
                ).fetchone()
                if row is None or row[0] is None:
                    raise ValueError(f"No daily data for symbol: {sym}")
                end_str = row[0]

            rows = conn.execute(
                f"SELECT {_SELECT_COLS} FROM kline_daily "
                f"WHERE symbol=? AND date<=?{suspend_clause} "
                "ORDER BY date DESC LIMIT ?",
                (sym, end_str, lookback),
            ).fetchall()

        if not rows:
            raise ValueError(
                f"Insufficient history for {sym} at end_date={end_str}, "
                f"lookback={lookback}"
            )
        if len(rows) < lookback:
            raise ValueError(
                f"Insufficient history for {sym} at end_date={end_str}: "
                f"need {lookback} rows, got {len(rows)}"
            )

        df = pd.DataFrame(
            [dict(r) for r in reversed(rows)],
        )
        df["date"] = pd.to_datetime(df["date"])
        return df.reset_index(drop=True)

    def last_date(self, symbol: str) -> Optional[pd.Timestamp]:
        sym = normalize_symbol(symbol)
        with open_kline_db(self._db_path) as conn:
            row = conn.execute(
                "SELECT MAX(date) FROM kline_daily WHERE symbol=?",
                (sym,),
            ).fetchone()
        if row is None or row[0] is None:
            return None
        return pd.Timestamp(row[0])
