from __future__ import annotations


def direction(delta: float) -> int:
    return 1 if delta >= 0 else 0


def relative_close_error(pred_close: float, true_close: float, base: float) -> float:
    if base == 0:
        return 0.0
    return abs(pred_close - true_close) / abs(base)


def day1_scores(pred_close: float, true_close: float, base: float) -> dict:
    e1 = relative_close_error(pred_close, true_close, base)
    pred_dir = direction(pred_close - base)
    true_dir = direction(true_close - base)
    return {
        "e1": e1,
        "pred_dir": pred_dir,
        "true_dir": true_dir,
        "correct": int(pred_dir == true_dir),
    }
