"""Smoke test for DailyKronosPredictor (single + batch)."""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from inference import DailyKronosPredictor

INPUT_PATH = PROJECT_ROOT / "tests" / "data" / "regression_input.csv"


def main():
    df = pd.read_csv(INPUT_PATH, parse_dates=["timestamps"])

    # Resample 5-min bars to pseudo-daily bars for API shape testing.
    daily = (
        df.set_index("timestamps")
        .resample("1D")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "amount": "sum",
            }
        )
        .dropna()
        .reset_index()
    )
    print(f"Daily rows: {len(daily)}")

    predictor = DailyKronosPredictor(lookback=min(400, len(daily) - 5))

    single = predictor.predict(daily, pred_days=5, sample_count=1, verbose=False)
    print("\nSingle prediction:")
    print(single)

    batch = predictor.predict_batch([daily, daily], pred_days=5, batch_size=2, verbose=False)
    print("\nBatch prediction count:", len(batch))
    print(batch[0].head())


if __name__ == "__main__":
    main()
