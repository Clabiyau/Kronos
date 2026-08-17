"""Daily K-line prediction wrapper with batch parallel support."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import pandas as pd
import torch

from model import Kronos, KronosPredictor, KronosTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "pretrained" / "Kronos-Tokenizer-base"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "pretrained" / "Kronos-base"

PRICE_COLS = ["open", "high", "low", "close"]
FEATURE_COLS = ["open", "high", "low", "close", "volume", "amount"]
TIME_COL_CANDIDATES = ("timestamps", "timestamp", "date", "datetime")


def _resolve_device(device: Optional[str]) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_daily_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a daily OHLCV DataFrame for Kronos inference."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")

    out = df.copy()

    time_col = next((col for col in TIME_COL_CANDIDATES if col in out.columns), None)
    if time_col is None:
        raise ValueError(
            f"Missing time column. Expected one of: {', '.join(TIME_COL_CANDIDATES)}"
        )
    if time_col != "timestamps":
        out = out.rename(columns={time_col: "timestamps"})

    out["timestamps"] = pd.to_datetime(out["timestamps"])
    if out["timestamps"].duplicated().any():
        out = out.drop_duplicates(subset=["timestamps"], keep="last")

    missing_price_cols = [col for col in PRICE_COLS if col not in out.columns]
    if missing_price_cols:
        raise ValueError(f"Missing required price columns: {missing_price_cols}")

    for col in PRICE_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")
    if "amount" in out.columns:
        out["amount"] = pd.to_numeric(out["amount"], errors="coerce")

    out = out.sort_values("timestamps").reset_index(drop=True)
    out = out.dropna(subset=PRICE_COLS)

    if out.empty:
        raise ValueError("Input DataFrame is empty after cleaning.")

    return out


def build_daily_inputs(
    df: pd.DataFrame,
    pred_days: int,
    lookback: int = 400,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, int]:
    """Build predictor inputs from daily data."""
    if pred_days < 1:
        raise ValueError("pred_days must be >= 1.")
    if lookback < 1:
        raise ValueError("lookback must be >= 1.")
    if len(df) < lookback:
        raise ValueError(
            f"Insufficient history: need at least {lookback} rows, got {len(df)}."
        )

    hist = df.iloc[-lookback:].copy()
    x_df = hist[FEATURE_COLS] if all(col in hist.columns for col in FEATURE_COLS) else hist[PRICE_COLS]
    x_timestamp = hist["timestamps"].reset_index(drop=True)

    last_date = pd.Timestamp(hist["timestamps"].iloc[-1]).normalize()
    y_timestamp = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=pred_days)

    return x_df.reset_index(drop=True), x_timestamp, pd.Series(y_timestamp), pred_days


class DailyKronosPredictor:
    """High-level daily prediction API built on top of KronosPredictor."""

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        tokenizer_path: Union[str, Path] = DEFAULT_TOKENIZER_PATH,
        device: Optional[str] = None,
        lookback: int = 400,
        max_context: int = 512,
    ):
        if lookback > max_context:
            raise ValueError(
                f"lookback ({lookback}) cannot exceed max_context ({max_context})."
            )

        self.lookback = lookback
        self.max_context = max_context
        self.device = _resolve_device(device)

        tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path))
        model = Kronos.from_pretrained(str(model_path))
        tokenizer.eval()
        model.eval()

        self._predictor = KronosPredictor(
            model,
            tokenizer,
            device=self.device,
            max_context=max_context,
        )

    def predict(
        self,
        df: pd.DataFrame,
        pred_days: int,
        *,
        lookback: int | None = None,
        T: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
        sample_count: int = 1,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Predict future daily bars for one symbol.

        Args:
            df: Daily OHLCV DataFrame.
            pred_days: Number of future trading days to predict.
            lookback: Override instance lookback for this call.
        """
        effective_lookback = self.lookback if lookback is None else lookback
        if effective_lookback > self.max_context:
            raise ValueError(
                f"lookback ({effective_lookback}) cannot exceed max_context ({self.max_context})."
            )

        normalized = normalize_daily_df(df)
        x_df, x_timestamp, y_timestamp, pred_len = build_daily_inputs(
            normalized, pred_days=pred_days, lookback=effective_lookback
        )

        pred_df = self._predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=T,
            top_p=top_p,
            top_k=top_k,
            sample_count=sample_count,
            verbose=verbose,
        )
        return pred_df.reset_index(names="date")

    def predict_batch(
        self,
        df_list: Sequence[pd.DataFrame],
        pred_days: int,
        *,
        lookback: int | None = None,
        batch_size: int = 8,
        T: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
        sample_count: int = 1,
        verbose: bool = False,
    ) -> List[pd.DataFrame]:
        """
        Predict multiple daily series in parallel using GPU batching.

        Args:
            df_list: List of daily OHLCV DataFrames.
            pred_days: Number of future trading days to predict for each series.
            batch_size: Number of series processed per GPU batch.
            lookback: Override instance lookback for this call.
        """
        if not df_list:
            return []
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")

        effective_lookback = self.lookback if lookback is None else lookback
        if effective_lookback > self.max_context:
            raise ValueError(
                f"lookback ({effective_lookback}) cannot exceed max_context ({self.max_context})."
            )

        prepared = []
        for i, df in enumerate(df_list):
            try:
                normalized = normalize_daily_df(df)
                x_df, x_timestamp, y_timestamp, pred_len = build_daily_inputs(
                    normalized,
                    pred_days=pred_days,
                    lookback=effective_lookback,
                )
            except ValueError as exc:
                raise ValueError(f"Failed to prepare series at index {i}: {exc}") from exc
            prepared.append((x_df, x_timestamp, y_timestamp, pred_len))

        pred_lens = {item[3] for item in prepared}
        if len(pred_lens) != 1:
            raise ValueError(
                f"All series must share the same pred_len, got: {sorted(pred_lens)}"
            )

        results: List[pd.DataFrame] = []
        for start in range(0, len(prepared), batch_size):
            chunk = prepared[start : start + batch_size]
            pred_df_list = self._predictor.predict_batch(
                df_list=[item[0] for item in chunk],
                x_timestamp_list=[item[1] for item in chunk],
                y_timestamp_list=[item[2] for item in chunk],
                pred_len=chunk[0][3],
                T=T,
                top_p=top_p,
                top_k=top_k,
                sample_count=sample_count,
                verbose=verbose,
            )
            results.extend(pred.reset_index(names="date") for pred in pred_df_list)

        return results
