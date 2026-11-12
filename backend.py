"""
Endpoints: GET /health, GET /tickers, GET /forecast/{ticker}?as_of=YYYY-MM-DD, GET /pipeline, GET /metrics.
Used to predict log annualised volatility; this service turns it into % with the bias correction.
Configuration (environment): MODEL_URL, DATA_DIR, MODEL_DIR, REPORTS_DIR.
"""
import json
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, Info, generate_latest
from pydantic import BaseModel

MODEL_URL = os.getenv("MODEL_URL", "http://localhost:5001/invocations")
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "serving"))     # the promoted model and its reference.json
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "reports"))
HISTORY_DAYS = 120

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("backend")

REQUESTS = Counter("backend_requests_total", "HTTP requests by route and status code", ["route", "status"])
LATENCY = Histogram("backend_request_seconds", "Request latency by route", ["route"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
MODEL_LATENCY = Histogram("backend_model_call_seconds", "Latency of the call to the model server",
                          buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1))
ERRORS = Counter("backend_errors_total", "Errors by type", ["type"])
PREDICTION = Histogram("backend_prediction_value", "Forecast annualised volatility in %",
                       buckets=(10, 15, 20, 25, 30, 40, 60, 100))
REL_ERROR = Histogram("backend_forecast_rel_error", "|forecast - realised| / realised, when known",
                      buckets=(0.05, 0.1, 0.2, 0.3, 0.5, 1))
BUCKET = Counter("backend_risk_bucket_total", "Forecasts by risk bucket", ["bucket"])
DRIFT = Gauge("backend_feature_drift_z", "Mean z-score of the last 50 requests against training data", ["feature"])
MODEL_UP = Gauge("backend_model_server_up", "1 if the last call to the model server succeeded")
MODEL_INFO = Info("backend_model", "Model served behind this backend")
for b in ("low", "medium", "high"):
    BUCKET.labels(b)
for t in ("model_server", "unknown_ticker", "no_data_for_date", "unexpected"):   # pre-created: panels show 0
    ERRORS.labels(t)

state = {"features": None, "reference": None}
recent = deque(maxlen=50)


class Forecast(BaseModel):
    ticker: str
    as_of: str
    forecast_vol: float                     # annualised volatility over the next 5 trading days, %
    bucket: str
    realized_vol: float | None              # what actually happened, known only for past dates
    vix: float
    history: list[dict]


class Tickers(BaseModel):
    tickers: list[str]
    first_date: str
    last_date: str


def load_state():
    """Load the latest features and the training reference statistics; return True when both exist."""
    try:
        state["features"] = pd.read_parquet(DATA_DIR / "features").sort_values("date")
        state["reference"] = json.loads((MODEL_DIR / "reference.json").read_text())
        for f in state["reference"]["features"]:
            DRIFT.labels(f)
        MODEL_INFO.info({"run_id": state["reference"]["run_id"]})
        return True
    except (OSError, ValueError, KeyError) as e:
        log.error("could not load data or reference: %s", e)
        return False


@asynccontextmanager
async def lifespan(_app):
    load_state()
    yield


app = FastAPI(title="Volatility forecast backend", version="1.0", lifespan=lifespan)


@app.middleware("http")
async def observe(request: Request, call_next):
    """Count and time every request by its route template (not the raw path, to keep labels bounded)."""
    t0 = time.perf_counter()
    response = await call_next(request)
    route = getattr(request.scope.get("route"), "path", "unmatched")
    if route != "/metrics":
        REQUESTS.labels(route, response.status_code).inc()
        LATENCY.labels(route).observe(time.perf_counter() - t0)
    return response


@app.exception_handler(Exception)
async def unexpected(request: Request, exc: Exception):
    """Log unexpected errors and return a short 500 instead of a stack trace."""
    ERRORS.labels("unexpected").inc()
    log.exception("unexpected error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal error"})


def predict(row):
    """Send one feature row to the MLflow model server; return the forecast as annualised volatility in %."""
    t0 = time.perf_counter()
    try:
        r = requests.post(MODEL_URL, json={"dataframe_records": [row]}, timeout=5)
        r.raise_for_status()
        MODEL_UP.set(1)
        log_vol = float(r.json()["predictions"][0])
        return round(100 * float(np.exp(log_vol + state["reference"]["resid_var"] / 2)), 2)
    except (requests.RequestException, KeyError, ValueError) as e:
        MODEL_UP.set(0)
        ERRORS.labels("model_server").inc()
        log.error("model server call failed: %s", e)
        raise HTTPException(status_code=503, detail="model server unavailable")
    finally:
        MODEL_LATENCY.observe(time.perf_counter() - t0)


def bucket(value):
    """Map a forecast (%) to low / medium / high using the tertiles of training volatility."""
    low, high = state["reference"]["bucket_edges"]
    return "low" if value < low else "medium" if value < high else "high"


def update_drift(row):
    """Keep the last 50 request rows and publish their mean z-score per feature."""
    ref = state["reference"]
    recent.append(row)
    df = pd.DataFrame(recent)
    for f in ref["features"]:
        DRIFT.labels(f).set(float((df[f].mean() - ref["mean"][f]) / ref["std"][f]))


@app.get("/health")
def health():
    """Readiness: 200 only when the data is loaded and the model server answers /ping."""
    ready = state["features"] is not None or load_state()
    try:
        model_ok = requests.get(MODEL_URL.rsplit("/", 1)[0] + "/ping", timeout=2).ok
    except requests.RequestException:
        model_ok = False
    MODEL_UP.set(int(model_ok))
    body = {"status": "ok" if ready and model_ok else "unavailable", "data_loaded": ready, "model_server": model_ok,
            "run_id": (state["reference"] or {}).get("run_id")}
    return JSONResponse(status_code=200 if ready and model_ok else 503, content=body)


@app.get("/tickers", response_model=Tickers)
def tickers():
    """List the tickers with features and the date range a forecast can be asked for."""
    if state["features"] is None and not load_state():
        raise HTTPException(status_code=503, detail="data not loaded")
    df = state["features"]
    return Tickers(tickers=sorted(df["ticker"].unique().tolist()), first_date=str(df["date"].min())[:10],
                   last_date=str(df["date"].max())[:10])


@app.get("/forecast/{ticker}", response_model=Forecast)
def forecast(ticker: str, as_of: date | None = None):
    """Forecast from the last trading day on or before `as_of` (default: latest), its risk bucket and history."""
    if state["features"] is None and not load_state():
        raise HTTPException(status_code=503, detail="data not loaded")
    rows = state["features"][state["features"]["ticker"] == ticker.upper()]
    if rows.empty:
        ERRORS.labels("unknown_ticker").inc()
        raise HTTPException(status_code=404, detail=f"unknown ticker '{ticker}'")
    if as_of is not None:
        rows = rows[pd.to_datetime(rows["date"]) <= pd.Timestamp(as_of)]
        if rows.empty:
            ERRORS.labels("no_data_for_date").inc()
            raise HTTPException(status_code=404, detail=f"no data for {ticker} on or before {as_of}")
    latest = rows.iloc[-1]
    row = {f: float(latest[f]) for f in state["reference"]["features"]}
    value = predict(row)
    PREDICTION.observe(value)
    level = bucket(value)
    BUCKET.labels(level).inc()
    update_drift(row)
    hist = rows.tail(HISTORY_DAYS)
    history = [{"date": str(d.date()), "close": float(c), "rv_5": round(100 * float(np.exp(v)), 2)}
               for d, c, v in zip(pd.to_datetime(hist["date"]), hist["close"], hist["log_rv_5"])]
    realized = None if pd.isna(latest["target"]) else round(100 * float(np.exp(latest["target"])), 2)
    if realized:
        REL_ERROR.observe(abs(value - realized) / realized)      # model performance whenever the outcome is known
    return Forecast(ticker=ticker.upper(), as_of=str(pd.to_datetime(latest["date"]).date()), forecast_vol=value,
                    bucket=level, realized_vol=realized, vix=round(float(latest["vix"]), 2), history=history)


@app.get("/pipeline")
def pipeline():
    """Pipeline console data: the last stage runs and the latest report of each stage."""
    def read(name):
        p = REPORTS_DIR / name
        return p.read_text() if p.exists() else ""
    runs = [json.loads(line) for line in read("pipeline_runs.jsonl").splitlines()[-50:] if line.strip()]
    return {"runs": runs, "reports": {n: json.loads(read(n) or "{}")
                                      for n in ("validation.json", "features.json", "train.json", "metrics.json")}}


@app.get("/metrics")
def metrics():
    """Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
