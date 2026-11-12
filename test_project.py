"""Unit and API tests (run by the DVC `test` stage). The MLflow model server is replaced by a stub."""
import json
import math

import numpy as np
import pandas as pd
import pytest
import requests
from fastapi.testclient import TestClient

import backend
import pipeline
from pipeline import FEATURES, build_features, check, scores, spark_session, to_vol


# ---------- pipeline: Spark features and metrics ----------
@pytest.fixture(scope="module")
def spark():
    s = spark_session()
    yield s
    s.stop()


def test_har_features_and_target_match_pandas(spark):
    rng = np.random.default_rng(0)
    n = 40
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    pdf = pd.DataFrame({"ticker": "AAA", "date": pd.bdate_range("2020-01-01", periods=n), "open": close * 0.999,
                        "high": close * 1.01, "low": close * 0.985, "close": close, "volume": 1000.0})
    vix = pd.DataFrame({"date": pdf["date"], "vix": 20.0})
    out = build_features(spark.createDataFrame(pdf), spark.createDataFrame(vix), [1, 5, 10], 3).toPandas()
    out = out.sort_values("date").reset_index(drop=True)
    gk = 0.5 * np.log(pdf["high"] / pdf["low"]) ** 2 - (2 * math.log(2) - 1) * np.log(pdf["close"] / pdf["open"]) ** 2
    log_rv_5 = np.log(np.sqrt(252 * gk.rolling(5).mean()))
    target = np.log(np.sqrt(252 * gk[::-1].rolling(3).mean()[::-1].shift(-1)))
    assert len(out) == n - 10
    np.testing.assert_allclose(out["log_rv_5"], log_rv_5.iloc[10:], rtol=1e-9)
    np.testing.assert_allclose(out["target"].iloc[:-3], target.iloc[10:-3], rtol=1e-9)
    assert out["target"].iloc[-3:].isna().all()
    assert (out["log_vix"] == math.log(20)).all() and (out["vix_change_5"] == 0).all()


def test_scores_and_bias_correction():
    perfect = scores([0.2, 0.3], [0.2, 0.3])
    assert perfect == {"rmse": 0.0, "rmspe": 0.0, "qlike": 0.0}
    assert scores([0.2], [0.1])["qlike"] > scores([0.2], [0.4])["qlike"]      # QLIKE punishes under-forecasts more
    assert to_vol(np.log(0.2), 0.02)[()] == pytest.approx(0.2 * math.exp(0.01))


# ---------- pipeline: data-quality checks ----------
def frame(n=600):
    dates = pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d")
    df = pd.DataFrame({"ticker": "AAA", "date": dates, "open": 100.0, "high": 102.0, "low": 98.0, "close": 101.0,
                       "volume": 1000.0})
    return df, pd.DataFrame({"date": dates, "vix": 20.0})


def test_clean_data_passes():
    df, vix = frame()
    assert all(check(df, vix, ["AAA"]).values())


def test_detects_bad_rows():
    df, vix = frame()
    df.loc[0, "high"] = 90.0           # high below close
    df.loc[1, "volume"] = None
    df = pd.concat([df, df.tail(1)])
    r = check(df, vix, ["AAA"])
    assert not r["high_is_highest"] and not r["no_nulls"] and not r["unique_ticker_date"]


def test_missing_column_ticker_history_and_vix():
    df, vix = frame()
    assert check(df.drop(columns="volume"), vix, ["AAA"]) == {"required_columns": False}
    short, _ = frame(10)
    r = check(short, vix.head(5), ["AAA", "BBB"])
    assert not r["enough_history"] and not r["all_tickers_present"] and not r["vix_covers_dates"]


def test_tracked_records_failed_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "REPORTS", tmp_path)
    monkeypatch.setattr(pipeline, "push_metrics", lambda info: None)
    with pytest.raises(ValueError), pipeline.tracked("demo"):
        raise ValueError("boom")
    run = json.loads((tmp_path / "pipeline_runs.jsonl").read_text())
    assert run["stage"] == "demo" and run["status"] == "failed" and "boom" in run["error"]


# ---------- backend API ----------
@pytest.fixture
def client(tmp_path, monkeypatch):
    (tmp_path / "features").mkdir()
    df = pd.DataFrame({"ticker": ["AAA"] * 3, "date": pd.bdate_range("2024-01-01", periods=3),
                       "close": [100.0, 101.0, 99.0], "vix": 15.0, "target": [np.log(0.25), np.log(0.3), None],
                       **{f: [-1.5, -1.4, -1.3] for f in FEATURES}})
    df.to_parquet(tmp_path / "features" / "part.parquet")
    ref = {"run_id": "test", "features": FEATURES, "mean": dict.fromkeys(FEATURES, 0.02),
           "std": dict.fromkeys(FEATURES, 0.01), "bucket_edges": [20.0, 30.0], "resid_var": 0.0}
    (tmp_path / "reference.json").write_text(json.dumps(ref))
    monkeypatch.setattr(backend, "DATA_DIR", tmp_path)
    monkeypatch.setattr(backend, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(backend, "REPORTS_DIR", tmp_path)
    backend.state.update(features=None, reference=None)
    return TestClient(backend.app)


class Reply:
    def __init__(self, body):
        self.body, self.ok = body, True

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


def test_forecast_converts_log_vol_to_percent_and_bucket(client, monkeypatch):
    monkeypatch.setattr(backend.requests, "post", lambda *a, **k: Reply({"predictions": [np.log(0.35)]}))
    r = client.get("/forecast/aaa")
    assert r.status_code == 200
    body = r.json()
    assert body["forecast_vol"] == 35.0 and body["bucket"] == "high" and body["realized_vol"] is None
    assert len(body["history"]) == 3 and body["as_of"] == "2024-01-03"


def test_as_of_picks_earlier_day_with_realized_value(client, monkeypatch):
    monkeypatch.setattr(backend.requests, "post", lambda *a, **k: Reply({"predictions": [np.log(0.1)]}))
    body = client.get("/forecast/AAA?as_of=2024-01-01").json()
    assert body["as_of"] == "2024-01-01" and body["realized_vol"] == 25.0 and body["bucket"] == "low"
    assert client.get("/forecast/AAA?as_of=2023-01-01").status_code == 404
    assert client.get("/forecast/AAA?as_of=not-a-date").status_code == 422
    assert "backend_forecast_rel_error_count 1.0" in client.get("/metrics").text


def test_unknown_ticker_is_404(client):
    assert client.get("/forecast/NOPE").status_code == 404


def test_model_server_down_is_503(client, monkeypatch):
    def down(*a, **k):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(backend.requests, "post", down)
    monkeypatch.setattr(backend.requests, "get", down)
    assert client.get("/forecast/AAA").status_code == 503
    assert client.get("/health").status_code == 503


def test_tickers_and_metrics(client):
    assert client.get("/tickers").json() == {"tickers": ["AAA"], "first_date": "2024-01-01", "last_date": "2024-01-03"}
    text = client.get("/metrics").text
    assert "backend_requests_total" in text and "backend_feature_drift_z" in text
