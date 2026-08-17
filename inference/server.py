"""Minimal local HTTP API for daily K-line prediction."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference.daily_predictor import DailyKronosPredictor
from inference.kline_reader import KlineDbReader, _DEFAULT_KLINE_DB, normalize_symbol

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_predictor: DailyKronosPredictor | None = None
_reader: KlineDbReader | None = None


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    return Path(raw) if raw else default


def _get_reader() -> KlineDbReader:
    if _reader is None:
        raise RuntimeError("KlineDbReader is not initialized")
    return _reader


def _get_predictor() -> DailyKronosPredictor:
    if _predictor is None:
        raise RuntimeError("DailyKronosPredictor is not initialized")
    return _predictor


class PredictRequest(BaseModel):
    symbol: str
    pred_days: int = Field(default=5, ge=1, le=120)
    lookback: int = Field(default=400, ge=1, le=512)
    end_date: Optional[str] = None
    include_suspended: bool = False
    sample_count: int = Field(default=1, ge=1, le=8)


class PredictBatchRequest(BaseModel):
    symbols: List[str] = Field(min_length=1)
    pred_days: int = Field(default=5, ge=1, le=120)
    lookback: int = Field(default=400, ge=1, le=512)
    end_date: Optional[str] = None
    include_suspended: bool = False
    sample_count: int = Field(default=1, ge=1, le=8)
    batch_size: int = Field(default=8, ge=1, le=64)


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor, _reader

    db_path = _env_path("KRONOS_KLINE_DB_PATH", _DEFAULT_KLINE_DB)
    model_path = _env_path(
        "KRONOS_MODEL_PATH",
        PROJECT_ROOT / "pretrained" / "Kronos-base",
    )
    tokenizer_path = _env_path(
        "KRONOS_TOKENIZER_PATH",
        PROJECT_ROOT / "pretrained" / "Kronos-Tokenizer-base",
    )
    device = os.environ.get("KRONOS_DEVICE", "").strip() or None
    max_lookback = int(os.environ.get("KRONOS_MAX_LOOKBACK", "512"))

    _reader = KlineDbReader(db_path)
    _predictor = DailyKronosPredictor(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        device=device,
        lookback=max_lookback,
        max_context=max_lookback,
    )
    yield
    _predictor = None
    _reader = None


app = FastAPI(title="Kronos Inference", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    reader = _get_reader()
    predictor = _get_predictor()
    return {
        "status": "ok",
        "device": predictor.device,
        "kline_db": str(reader.db_path),
        "lookback": predictor.lookback,
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    reader = _get_reader()
    predictor = _get_predictor()
    if req.lookback > predictor.max_context:
        raise HTTPException(
            status_code=400,
            detail=f"lookback ({req.lookback}) exceeds max_context ({predictor.max_context})",
        )

    try:
        hist = reader.load_recent_daily(
            req.symbol,
            lookback=req.lookback,
            end_date=req.end_date,
            include_suspended=req.include_suspended,
        )
        pred_df = predictor.predict(
            hist,
            pred_days=req.pred_days,
            lookback=req.lookback,
            sample_count=req.sample_count,
            verbose=False,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    end_used = hist["date"].iloc[-1].strftime("%Y-%m-%d")
    return {
        "symbol": normalize_symbol(req.symbol),
        "pred_days": req.pred_days,
        "lookback": req.lookback,
        "end_date": end_used,
        "rows": _df_to_records(pred_df),
    }


@app.post("/predict_batch")
def predict_batch(req: PredictBatchRequest) -> dict[str, Any]:
    reader = _get_reader()
    predictor = _get_predictor()
    if req.lookback > predictor.max_context:
        raise HTTPException(
            status_code=400,
            detail=f"lookback ({req.lookback}) exceeds max_context ({predictor.max_context})",
        )

    histories: list[pd.DataFrame] = []
    symbols: list[str] = []
    end_used: str | None = None

    try:
        for raw in req.symbols:
            sym = normalize_symbol(raw)
            hist = reader.load_recent_daily(
                sym,
                lookback=req.lookback,
                end_date=req.end_date,
                include_suspended=req.include_suspended,
            )
            histories.append(hist)
            symbols.append(sym)
            end_used = hist["date"].iloc[-1].strftime("%Y-%m-%d")

        pred_list = predictor.predict_batch(
            histories,
            pred_days=req.pred_days,
            lookback=req.lookback,
            batch_size=req.batch_size,
            sample_count=req.sample_count,
            verbose=False,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "pred_days": req.pred_days,
        "lookback": req.lookback,
        "end_date": end_used,
        "results": [
            {"symbol": sym, "rows": _df_to_records(pred_df)}
            for sym, pred_df in zip(symbols, pred_list)
        ],
    }


def main() -> None:
    host = os.environ.get("KRONOS_HOST", "127.0.0.1")
    port = int(os.environ.get("KRONOS_PORT", "8765"))
    uvicorn.run(
        "inference.server:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
