"""Multi-horizon close evaluation against KronosPredictor outputs."""
from __future__ import annotations

import math
import random
from typing import Any

from finetune_ashare.dataset import SampleRef, get_day1_price_context
from finetune_ashare.metrics import day1_scores, relative_close_error


def _empty_metrics(pred_len: int) -> dict:
    out: dict = {"dir_acc": 0.0, "n": 0}
    for h in range(1, pred_len + 1):
        out[f"e{h}_mean"] = float("inf")
    return out


def evaluate_day1(
    predictor: Any,
    mem,
    refs: list[SampleRef],
    *,
    max_samples: int,
    seed: int,
    pred_len: int,
    T,
    top_p,
    sample_count,
    lookback: int | None = None,
) -> dict:
    """Score relative close error for horizons 1..pred_len, plus day-1 direction.

    Always calls ``predictor.predict`` with ``pred_len``. For each horizon ``h``
    (1-based), ``e{h}_mean`` is the mean of
    ``|pred_close_h - true_close_h| / base`` where ``base`` is asof close.

    ``dir_acc`` remains day-1 only (flat counts as up).

    Empty / unscored runs return ``e{h}_mean=inf`` (never 0.0) so callers do not
    treat a missing eval as a perfect score.
    """
    if lookback is None:
        raise ValueError("lookback is required for get_day1_price_context")
    if pred_len < 1:
        raise ValueError("pred_len must be >= 1")

    if not refs or max_samples <= 0:
        return _empty_metrics(pred_len)

    if len(refs) > max_samples:
        selected = random.Random(seed).sample(refs, max_samples)
    else:
        selected = list(refs)

    if not selected:
        return _empty_metrics(pred_len)

    e_vals: dict[int, list[float]] = {h: [] for h in range(1, pred_len + 1)}
    correct_vals: list[int] = []

    for ref in selected:
        base, true_closes, hist_df, x_ts, y_ts = get_day1_price_context(
            mem, ref, lookback=lookback, predict=pred_len
        )
        if len(true_closes) < pred_len:
            continue

        pred_df = predictor.predict(
            hist_df,
            x_ts,
            y_ts,
            pred_len=pred_len,
            T=T,
            top_p=top_p,
            sample_count=sample_count,
            verbose=False,
        )
        if len(pred_df) < pred_len:
            continue

        for h in range(1, pred_len + 1):
            pred_close = float(pred_df["close"].iloc[h - 1])
            true_close = float(true_closes[h - 1])
            e_vals[h].append(relative_close_error(pred_close, true_close, base))

        # Direction accuracy: day-1 only
        scores = day1_scores(
            float(pred_df["close"].iloc[0]),
            float(true_closes[0]),
            base,
        )
        correct_vals.append(scores["correct"])

    n = len(correct_vals)
    if n == 0:
        return _empty_metrics(pred_len)

    out: dict = {
        "dir_acc": float(sum(correct_vals) / n),
        "n": n,
    }
    for h in range(1, pred_len + 1):
        vals = e_vals[h]
        out[f"e{h}_mean"] = float(sum(vals) / len(vals)) if vals else float("inf")
    return out


def format_e_means(metrics: dict, pred_len: int = 5) -> str:
    """Compact log fragment: ``e1=... e2=...`` (skips non-finite)."""
    parts: list[str] = []
    for h in range(1, pred_len + 1):
        key = f"e{h}_mean"
        val = metrics.get(key)
        if val is None or not math.isfinite(float(val)):
            parts.append(f"e{h}=inf")
        else:
            parts.append(f"e{h}={float(val):.6f}")
    return " ".join(parts)
