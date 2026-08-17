"""Tests for inference HTTP API (mocked predictor)."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import inference.server as server


@asynccontextmanager
async def _noop_lifespan(_app):
    yield


@pytest.fixture
def client():
    mock_reader = MagicMock()
    mock_reader.db_path = Path("tests/data/kline.db")
    mock_reader.load_recent_daily.return_value = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=400),
            "open": [10.0] * 400,
            "high": [10.5] * 400,
            "low": [9.5] * 400,
            "close": [10.1] * 400,
            "volume": [1000.0] * 400,
            "amount": [10000.0] * 400,
        }
    )

    mock_predictor = MagicMock()
    mock_predictor.device = "cpu"
    mock_predictor.max_context = 512
    mock_predictor.lookback = 512
    mock_predictor.predict.return_value = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-03-01", periods=3),
            "open": [10.2, 10.3, 10.4],
            "high": [10.6, 10.7, 10.8],
            "low": [9.8, 9.9, 10.0],
            "close": [10.3, 10.4, 10.5],
            "volume": [1100.0, 1200.0, 1300.0],
            "amount": [11000.0, 12000.0, 13000.0],
        }
    )
    mock_predictor.predict_batch.return_value = [
        mock_predictor.predict.return_value,
        mock_predictor.predict.return_value,
    ]

    server._reader = mock_reader
    server._predictor = mock_predictor
    server.app.router.lifespan_context = _noop_lifespan
    with TestClient(server.app) as test_client:
        yield test_client
    server._reader = None
    server._predictor = None


def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["device"] == "cpu"


def test_predict(client: TestClient):
    resp = client.post(
        "/predict",
        json={"symbol": "600519", "pred_days": 3, "lookback": 400},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "600519.SH"
    assert body["pred_days"] == 3
    assert len(body["rows"]) == 3


def test_predict_batch(client: TestClient):
    resp = client.post(
        "/predict_batch",
        json={"symbols": ["600519", "000001"], "pred_days": 3, "lookback": 400},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 2
