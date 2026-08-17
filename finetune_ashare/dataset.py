"""Sample index and PyTorch Dataset for A-share daily finetune."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from finetune_ashare.data_loader import MarketMemory, SymbolSeries
from finetune_ashare.split import assign_split

_FEATURE_COLS = ("open", "high", "low", "close", "volume", "amount")


@dataclass(frozen=True)
class SampleRef:
    symbol: str
    asof: str
    asof_pos: int  # index into that symbol's non-suspend (valid) subsequence


def _valid_views(series: SymbolSeries) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_idx = np.where(series.suspend != 1)[0]
    valid_dates = series.dates[valid_idx]
    valid_feat = series.features[valid_idx]
    return valid_idx, valid_dates, valid_feat


def _time_stamps_from_dates(dates: np.ndarray) -> np.ndarray:
    ts = pd.to_datetime(pd.Series(dates))
    stamp = np.stack(
        [
            ts.dt.minute.to_numpy(),
            ts.dt.hour.to_numpy(),
            ts.dt.weekday.to_numpy(),
            ts.dt.day.to_numpy(),
            ts.dt.month.to_numpy(),
        ],
        axis=1,
    ).astype(np.float32)
    return stamp


def build_sample_refs(
    mem: MarketMemory,
    blocks: list[tuple[str, str, frozenset[str]]],
    *,
    lookback: int,
    predict: int,
    backtest_start: str,
) -> dict[str, list[SampleRef]]:
    """Build train/val/backtest SampleRef lists from market memory.

    Indexing scheme: for each symbol, work on the non-suspend subsequence.
    A valid asof at valid-position ``i`` requires ``i >= lookback - 1`` and
    ``i + predict < len(valid)`` so the window covers
    ``i - lookback + 1 .. i`` (history) and ``i + 1 .. i + predict`` (future).
    """
    out: dict[str, list[SampleRef]] = {"train": [], "val": [], "backtest": []}
    for symbol in mem.symbols:
        series = mem.by_symbol[symbol]
        _, valid_dates, _ = _valid_views(series)
        n_valid = len(valid_dates)
        for i in range(lookback - 1, n_valid - predict):
            asof = str(valid_dates[i])
            split = assign_split(asof, symbol, blocks, backtest_start)
            out[split].append(SampleRef(symbol=symbol, asof=asof, asof_pos=i))
    return out


class AshareKlineDataset(Dataset):
    """Sliding-window dataset over SampleRefs with per-window normalization."""

    def __init__(
        self,
        mem: MarketMemory,
        refs: list[SampleRef],
        lookback: int,
        predict: int,
        clip: float,
        seed: int,
        mode: Literal["train", "eval"],
        n_steps: int | None = None,
    ):
        self.mem = mem
        self.refs = list(refs)
        self.lookback = lookback
        self.predict = predict
        self.clip = clip
        self.seed = seed
        self.mode = mode
        self.n_steps = n_steps
        self.py_rng = random.Random(seed)
        self.current_epoch = 0

        # Cache valid subsequences per symbol for fast __getitem__.
        self._valid_feat: dict[str, np.ndarray] = {}
        self._valid_stamp: dict[str, np.ndarray] = {}
        symbols_needed = {r.symbol for r in self.refs}
        for symbol in symbols_needed:
            series = mem.by_symbol[symbol]
            _, valid_dates, valid_feat = _valid_views(series)
            self._valid_feat[symbol] = valid_feat.astype(np.float32, copy=False)
            self._valid_stamp[symbol] = _time_stamps_from_dates(valid_dates)

    def set_epoch_seed(self, epoch: int) -> None:
        epoch_seed = self.seed + epoch
        self.py_rng.seed(epoch_seed)
        self.current_epoch = epoch

    def __len__(self) -> int:
        if self.mode == "train" and self.n_steps is not None:
            return self.n_steps
        return len(self.refs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.refs:
            raise IndexError("empty SampleRef list")

        if self.mode == "train":
            epoch = getattr(self, "current_epoch", 0)
            ref_idx = (idx * 9973 + (epoch + 1) * 104729) % len(self.refs)
        else:
            ref_idx = idx % len(self.refs)

        ref = self.refs[ref_idx]
        i = ref.asof_pos
        start = i - self.lookback + 1
        end = i + self.predict + 1  # exclusive; includes future predict rows
        feat = self._valid_feat[ref.symbol]
        stamp = self._valid_stamp[ref.symbol]

        x = feat[start:end].astype(np.float32, copy=True)
        x_stamp = stamp[start:end]

        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)

        return torch.from_numpy(x), torch.from_numpy(x_stamp)


def get_day1_price_context(
    mem: MarketMemory,
    ref: SampleRef,
    *,
    lookback: int,
    predict: int,
) -> tuple[float, list[float], pd.DataFrame, pd.Series, pd.Series]:
    """Unnormalized multi-horizon eval context for KronosPredictor.

    Returns ``(base, true_closes, hist_df, x_ts, y_ts)`` where:
    - ``base`` is asof close
    - ``true_closes`` length ``predict``: closes for asof+1 .. asof+predict
    - ``hist_df`` has ``lookback`` rows ending at asof (OHLCV + amount)
    - ``x_ts`` / ``y_ts`` are ``pd.Series`` datetimes (KronosPredictor needs ``.dt``)
    """
    series = mem.by_symbol[ref.symbol]
    _, valid_dates, valid_feat = _valid_views(series)
    i = ref.asof_pos
    if i < lookback - 1 or i + predict >= len(valid_dates):
        raise ValueError(
            f"ref asof_pos={i} out of range for lookback={lookback}, predict={predict}, "
            f"n_valid={len(valid_dates)}"
        )

    hist_start = i - lookback + 1
    hist_feat = valid_feat[hist_start : i + 1]
    hist_dates = valid_dates[hist_start : i + 1]
    future_dates = valid_dates[i + 1 : i + 1 + predict]

    base = float(valid_feat[i, 3])
    true_closes = [float(valid_feat[i + 1 + h, 3]) for h in range(predict)]

    hist_df = pd.DataFrame(hist_feat, columns=list(_FEATURE_COLS))
    # Series (not DatetimeIndex): model.calc_time_stamps uses x_timestamp.dt.*
    x_ts = pd.Series(pd.to_datetime(hist_dates)).reset_index(drop=True)
    y_ts = pd.Series(pd.to_datetime(future_dates)).reset_index(drop=True)
    return base, true_closes, hist_df, x_ts, y_ts
