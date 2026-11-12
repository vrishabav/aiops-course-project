"""Web UI (Streamlit) which knows the backend's REST API, set by BACKEND_URL"""
import logging
import os
import time
from datetime import date

import pandas as pd
import requests
import streamlit as st
from prometheus_client import Counter, Histogram, start_http_server

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")
LINKS = {"MLflow (experiments, registry)": os.getenv("MLFLOW_UI", "http://localhost:5000"),
         "Grafana (live monitoring)": os.getenv("GRAFANA_UI", "http://localhost:3000"),
         "Prometheus (targets, alerts)": os.getenv("PROMETHEUS_UI", "http://localhost:9090"),
         "Backend API docs": os.getenv("API_DOCS", "http://localhost:8000/docs")}
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("frontend")
COLORS = {"low": "#1a7f37", "medium": "#b35900", "high": "#c62828"}
STAGES = ["test", "ingest", "validate", "features", "train", "evaluate"]
STATUS_COLORS = {"success": "#b7e4c7", "failed": "#f4a6a6"}


@st.cache_resource
def ui_metrics():
    """Start the UI's own /metrics server once per process (Streamlit reruns this script on every click)."""
    start_http_server(int(os.getenv("UI_METRICS_PORT", "9101")))
    return {"views": Counter("ui_page_views_total", "Page renders by tab", ["tab"]),
            "calls": Histogram("ui_backend_call_seconds", "Backend call latency seen by the UI"),
            "errors": Counter("ui_backend_errors_total", "Failed backend calls")}


M = ui_metrics()


def api(path):
    """GET a backend path; return the JSON body or None after showing a friendly message."""
    t0 = time.perf_counter()
    try:
        r = requests.get(BACKEND + path, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        M["errors"].inc()
        log.error("backend call %s failed: %s", path, e)
        st.error("The forecasting service is not reachable right now. Please try again in a minute.")
        return None
    finally:
        M["calls"].observe(time.perf_counter() - t0)


st.set_page_config(page_title="Volatility forecast", page_icon="📈", layout="wide")
st.title("Next-week stock volatility")
forecast_tab, pipeline_tab, help_tab = st.tabs(["Forecast", "Pipeline", "Help"])

with forecast_tab:
    M["views"].labels("forecast").inc()
    meta = api("/tickers")          # list of stocks and the date range for the slider
    if meta:
        left, right = st.columns([1, 2])
        ticker = left.selectbox("Stock", meta["tickers"])
        first, last = (date.fromisoformat(meta[k]) for k in ("first_date", "last_date"))
        as_of = right.slider("Forecast as of", first, last, last, format="YYYY-MM-DD",
                             help="Move back in time to see what the model would have said then, e.g. March 2020.")
        result = api(f"/forecast/{ticker}?as_of={as_of}")
        if result:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Forecast volatility, next 5 days", f"{result['forecast_vol']:.1f} %",
                      help="Annualised: typical yearly price swing if the next week's pace continued.")
            c2.markdown(f"**Risk level**<br><span style='font-size:1.8rem;font-weight:600;"
                        f"color:{COLORS[result['bucket']]}'>{result['bucket'].upper()}</span>", unsafe_allow_html=True)
            actual = result["realized_vol"]
            c3.metric("What actually happened", f"{actual:.1f} %" if actual is not None else "not known yet")
            c4.metric("Market fear index (VIX)", f"{result['vix']:.1f}")
            st.caption(f"Based on data up to {result['as_of']}.")
            hist = pd.DataFrame(result["history"]).set_index("date")
            hist = hist.rename(columns={"rv_5": "last 5 days", "close": "closing price"})
            hist["forecast"] = result["forecast_vol"]
            st.markdown("**Volatility over the last 5 days (%) and the forecast for the next 5 days**")
            st.line_chart(hist[["last 5 days", "forecast"]], color=["#1f77b4", COLORS[result["bucket"]]])
            st.markdown("**Closing price**")
            st.line_chart(hist[["closing price"]])

with pipeline_tab:
    M["views"].labels("pipeline").inc()
    info = api("/pipeline")
    if info:
        runs = pd.DataFrame(info["runs"])
        last = runs.groupby("stage").last() if not runs.empty else pd.DataFrame()
        st.subheader("Pipeline (DVC DAG): colour = last run, green ok, red failed")
        nodes = []
        for name in STAGES:
            ok = last["status"].get(name) if name in last.index else None
            label = f"{name}\\n{last['seconds'][name]} s" if ok else name      # DOT newline between name and time
            nodes.append(f'{name} [label="{label}", style=filled, fillcolor="{STATUS_COLORS.get(ok, "#dddddd")}"]')
        st.graphviz_chart("digraph { rankdir=LR; node [shape=box, fontname=sans]; " + "; ".join(nodes)
                          + "; test -> ingest -> validate -> features -> train -> evaluate }")
        r = info["reports"]
        f, t, m = (r.get(k, {}) for k in ("features.json", "train.json", "metrics.json"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Spark throughput (rows/s)", f.get("rows_per_s", "-"))
        c2.metric("Ray speed-up vs sequential", f"{t.get('ray_speedup', '-')}x")
        c3.metric("Test QLIKE vs naive (lower is better)", f"{m.get('qlike_ratio', 0):.2f}")
        c4.metric("Promotion gate", "passed" if m.get("passed") else "not passed")
        if not runs.empty:
            ok = (runs["status"] == "success").sum()
            st.subheader(f"Run history: {ok} succeeded, {len(runs) - ok} failed (last {len(runs)} stage runs)")
            st.dataframe(runs.iloc[::-1], hide_index=True, width="stretch")
        if t.get("validation"):
            st.subheader(f"Model comparison on the validation year (champion: {t['champion'].upper()})")
            st.dataframe(pd.DataFrame(t["validation"]).T.round(4), width="stretch")
    with st.expander("Run or re-run the pipeline (on the host)"):
        st.code("dvc repro                    # run what changed\ndvc repro -f -s train        # force one stage\n"
                "dvc dag                      # show the DAG", language="bash")
    st.subheader("Tool consoles")
    for name, url in LINKS.items():
        st.markdown(f"- [{name}]({url})")

with help_tab:
    M["views"].labels("help").inc()
    st.markdown("""
**How to use:** pick a stock in *Forecast*. The page shows how much the price is expected to move over the
next five trading days, a risk level and the recent history. Move the date slider back to see what the
model would have forecast on an earlier day, and compare it with what actually happened.

**Volatility** is shown annualised in %: 30 % means price swings typical of a stock that moves about 30 % a
year. **Risk levels** split the training period into three equal groups: *high* means next week is expected
to be among the most volatile third of 2015-2022.

**If something looks wrong:** a red message means the service is down; wait a minute and reload.
The *Pipeline* tab shows when the data and model were last updated.
""")
