"""Smoke test: load local Kronos-base and run a short prediction."""
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model import Kronos, KronosPredictor, KronosTokenizer
TOKENIZER_PATH = PROJECT_ROOT / "pretrained" / "Kronos-Tokenizer-base"
MODEL_PATH = PROJECT_ROOT / "pretrained" / "Kronos-base"
INPUT_PATH = PROJECT_ROOT / "tests" / "data" / "regression_input.csv"

LOOKBACK = 400
PRED_LEN = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print(f"Device: {DEVICE}")
    print(f"Tokenizer: {TOKENIZER_PATH}")
    print(f"Model: {MODEL_PATH}")

    df = pd.read_csv(INPUT_PATH, parse_dates=["timestamps"])
    print(f"Input rows: {len(df)}")

    x_df = df.loc[: LOOKBACK - 1, ["open", "high", "low", "close", "volume", "amount"]]
    x_timestamp = df.loc[: LOOKBACK - 1, "timestamps"]
    y_timestamp = df.loc[LOOKBACK : LOOKBACK + PRED_LEN - 1, "timestamps"]

    tokenizer = KronosTokenizer.from_pretrained(str(TOKENIZER_PATH))
    model = Kronos.from_pretrained(str(MODEL_PATH))
    tokenizer.eval()
    model.eval()

    predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=512)

    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=PRED_LEN,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=True,
    )

    print("\nPrediction succeeded.")
    print(f"Output shape: {pred_df.shape}")
    print("\nForecast head:")
    print(pred_df.head())
    print("\nForecast tail:")
    print(pred_df.tail())


if __name__ == "__main__":
    main()
