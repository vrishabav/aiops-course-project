"""Volatility pipeline, one function per DVC stage: `python pipeline.py test|ingest|validate|features|train|evaluate`.

Every stage records status, duration and rows/s (reports/pipeline_runs.jsonl and the Pushgateway); `evaluate` copies
the candidate model to serving/ only if it beats the naive forecast. An interrupted `dvc repro` resumes at the first
unfinished stage.
"""
import contextlib
import hashlib
import itertools
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

ROOT = Path(__file__).resolve().parent
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())
RAW, VIX, FEATURES_DIR = ROOT / "data/raw.csv", ROOT / "data/vix.csv", ROOT / "data/features"
MODEL_DIR, SERVING_DIR = ROOT / "model", ROOT / "serving"
REPORTS, CACHE = ROOT / "reports", ROOT / ".cache"
HAR = ["log_rv_1", "log_rv_5", "log_rv_22"]
FEATURES = HAR + ["ret_5", "log_volume_ratio", "log_vix", "vix_change_5"]
TARGET = "target"                      # log of annualised volatility over the next 5 trading days
MODEL_NAME = "vol-forecaster"
PUSHGATEWAY = os.getenv("PUSHGATEWAY", "localhost:9091")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("pipeline")


# ---------- shared helpers ----------
def md5(path):
    """Hash a file or every file under a directory, so an MLflow run names the exact data it used."""
    h = hashlib.md5()
    for f in sorted(Path(path).rglob("*")) if Path(path).is_dir() else [Path(path)]:
        if f.is_file() and not f.name.startswith("."):
            h.update(f.read_bytes())
    return h.hexdigest()


def load_split():
    """(train, validation, test) split by date, so later periods never leak into training."""
    p = PARAMS["train"]
    df = pd.read_parquet(FEATURES_DIR).dropna(subset=[TARGET]).sort_values(["date", "ticker"])
    d = df["date"].astype(str)
    return df[d <= p["train_end"]], df[(d > p["train_end"]) & (d <= p["val_end"])], df[d > p["val_end"]]


def to_vol(log_pred, resid_var=0.0):
    """Log-volatility forecast -> volatility; exp(x + var/2) removes the downward bias of exp(x)."""
    return np.exp(np.asarray(log_pred) + resid_var / 2)


def scores(y, p):
    """RMSE and RMSPE on volatility, QLIKE on variance (y = realised, p = forecast)."""
    y, p = np.asarray(y), np.asarray(p)
    ratio = y**2 / p**2
    return {"rmse": float(np.sqrt(np.mean((y - p) ** 2))), "rmspe": float(np.sqrt(np.mean(((y - p) / y) ** 2))),
            "qlike": float(np.mean(ratio - np.log(ratio) - 1))}


def push_metrics(info):
    """Send a stage's outcome to the Pushgateway; skip with a warning when it is not running."""
    reg = CollectorRegistry()
    for name, value in {"pipeline_stage_duration_seconds": info["seconds"], "pipeline_stage_rows": info["rows"],
                        "pipeline_stage_rows_per_second": info["rows_per_s"],
                        "pipeline_stage_success": int(info["status"] == "success"),
                        "pipeline_stage_last_run_timestamp_seconds": time.time()}.items():
        Gauge(name, name.replace("_", " "), registry=reg).set(value)
    try:
        push_to_gateway(PUSHGATEWAY, job="pipeline", grouping_key={"stage": info["stage"]}, registry=reg, timeout=2)
    except OSError as e:
        log.warning("metrics not pushed (%s)", e)


@contextlib.contextmanager
def tracked(name):
    """Time a stage and record success or failure (with the error) whatever happens inside it."""
    info = {"stage": name, "start": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": 0, "status": "failed"}
    t0 = time.perf_counter()
    try:
        yield info
        info["status"] = "success"
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        log.exception("stage %s failed", name)
        raise
    finally:
        info["seconds"] = round(time.perf_counter() - t0, 2)
        info["rows_per_s"] = round(info["rows"] / info["seconds"], 1) if info["seconds"] else 0.0
        REPORTS.mkdir(exist_ok=True)
        with open(REPORTS / "pipeline_runs.jsonl", "a") as f:
            f.write(json.dumps(info) + "\n")
        push_metrics(info)
        log.info("%s %s in %.2f s (%s rows/s)", name, info["status"], info["seconds"], info["rows_per_s"])


def write_report(name, report):
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / name).write_text(json.dumps(report, indent=2))


# ---------- test ----------
def run_tests(info):
    """Run the unit tests as the first stage, so a failing test stops the pipeline before any data is touched."""
    junit = REPORTS / "junit.xml"
    REPORTS.mkdir(exist_ok=True)
    code = subprocess.call([sys.executable, "-m", "pytest", "-q", "test_project.py", f"--junitxml={junit}"], cwd=ROOT)
    info["rows"] = int(ET.parse(junit).getroot().find("testsuite").get("tests"))     # "rows" = tests run
    if code:
        raise RuntimeError(f"pytest failed (exit code {code})")


# ---------- ingest ----------
def download(tickers, start, end, attempts=3):
    """Adjusted daily OHLCV from Yahoo Finance in long format; retries tickers Yahoo skipped (rate limits)."""
    import yfinance as yf
    CACHE.mkdir(exist_ok=True)
    path = {t: CACHE / f"{t}_{start}_{end}.csv" for t in tickers}
    frames = [pd.read_csv(path[t]) for t in tickers if path[t].exists()]
    missing = [t for t in tickers if not path[t].exists()]
    for attempt in range(attempts):
        if not missing:
            break
        wide = yf.download(missing, start=start, end=end, auto_adjust=True, group_by="ticker", progress=False)
        for t in list(missing):
            if t in wide.columns.get_level_values(0) and wide[t]["Close"].notna().any():
                df = wide[t].rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(how="all")
                df = df.reset_index().rename(columns={"Date": "date"}).assign(ticker=t)
                df.to_csv(path[t], index=False)                 # checkpoint
                frames.append(df)
                missing.remove(t)
        if missing:
            log.warning("no data for %s, retrying", missing)
            time.sleep(5 * (attempt + 1))
    if missing:
        raise RuntimeError(f"Yahoo returned no data for {missing}")
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    return out[["ticker", "date", "open", "high", "low", "close", "volume"]].sort_values(["ticker", "date"])


def ingest(info):
    """Write data/raw.csv (30 stocks) and data/vix.csv for the fixed dates in params.yaml."""
    p = PARAMS["data"]
    raw = download(p["tickers"], p["start"], p["end"])
    vix = download([p["vix"]], p["start"], p["end"])[["date", "close"]].rename(columns={"close": "vix"})
    RAW.parent.mkdir(exist_ok=True)
    raw.to_csv(RAW, index=False)
    vix.to_csv(VIX, index=False)
    info["rows"] = len(raw) + len(vix)


# ---------- validate ----------
def check(df, vix, tickers, min_rows=500, tol=1e-4):
    """Data-quality rules -> {rule: passed}. tol allows rounding in Yahoo's adjusted prices."""
    cols = ["ticker", "date", "open", "high", "low", "close", "volume"]
    if not set(cols) <= set(df.columns):
        return {"required_columns": False}
    prices = df[["open", "high", "low", "close"]]
    return {
        "required_columns": True,
        "no_nulls": bool(df[cols].notna().all().all()),
        "prices_positive": bool((prices > 0).all().all()),
        "high_is_highest": bool((df["high"] * (1 + tol) >= prices.max(axis=1)).all()),
        "low_is_lowest": bool((df["low"] * (1 - tol) <= prices.min(axis=1)).all()),
        "volume_positive": bool((df["volume"] > 0).all()),
        "unique_ticker_date": not df.duplicated(["ticker", "date"]).any(),
        "all_tickers_present": set(tickers) <= set(df["ticker"]),
        "enough_history": bool(df.groupby("ticker").size().min() >= min_rows),
        "vix_covers_dates": bool(df["date"].isin(vix["date"]).mean() >= 0.99 and (vix["vix"] > 0).all()),
    }


def validate(info):
    """Stop the pipeline when any data-quality rule fails."""
    df, vix = pd.read_csv(RAW), pd.read_csv(VIX)
    info["rows"] = len(df)
    checks = check(df, vix, PARAMS["data"]["tickers"])
    write_report("validation.json", {"rows": len(df), "tickers": int(df["ticker"].nunique()),
                                     "first_date": df["date"].min(), "last_date": df["date"].max(),
                                     "checks": checks, "passed": all(checks.values())})
    if not all(checks.values()):
        raise ValueError(f"data checks failed: {[k for k, ok in checks.items() if not ok]}")


# ---------- features (Spark) ----------
def spark_session():
    """Local Spark on all cores; temp files go to the project disk because /tmp is RAM on this machine."""
    from pyspark.sql import SparkSession
    os.environ.setdefault("SPARK_LOCAL_DIRS", str(ROOT / ".spark_tmp"))
    spark = (SparkSession.builder.master("local[*]").appName("features")
             .config("spark.sql.shuffle.partitions", "8").config("spark.driver.memory", "2g").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    return spark


def build_features(df, vix, windows, horizon):
    """HAR features and the target per ticker with Spark window functions.

    Daily variance = Garman-Klass 0.5*ln(H/L)^2 - (2ln2-1)*ln(C/O)^2 (uses the day's range, far less noisy
    than a squared return). Volatility over k days = sqrt(252 * mean daily variance), annualised; we model its
    log. target = the same over the next `horizon` days, empty when that window is incomplete (the latest
    rows, which the service forecasts from). The first max(windows) rows per ticker are dropped (partial windows).
    """
    from pyspark.sql import Window
    from pyspark.sql import functions as F  # noqa: N812 (the usual Spark alias)

    def log_vol(var):
        return F.log(F.sqrt(F.greatest(var, F.lit(1e-8)) * 252))     # floor: days with high == low

    w = Window.partitionBy("ticker").orderBy("date")
    df = df.withColumn("gk", 0.5 * F.log(F.col("high") / F.col("low")) ** 2
                       - (2 * math.log(2) - 1) * F.log(F.col("close") / F.col("open")) ** 2)
    for k in windows:
        df = df.withColumn(f"log_rv_{k}", log_vol(F.avg("gk").over(w.rowsBetween(-(k - 1), 0))))
    ahead = w.rowsBetween(1, horizon)
    df = (df.withColumn("ret_5", F.log(F.col("close") / F.lag("close", 5).over(w)))
            .withColumn("log_volume_ratio", F.log(F.col("volume") / F.avg("volume").over(w.rowsBetween(-21, 0))))
            .withColumn(TARGET, F.when(F.count("gk").over(ahead) == horizon, log_vol(F.avg("gk").over(ahead))))
            .withColumn("n", F.row_number().over(w)))
    vix = (vix.withColumn("log_vix", F.log("vix"))
              .withColumn("vix_change_5", F.log(F.col("vix") / F.lag("vix", 5).over(Window.orderBy("date")))))
    df = df.join(F.broadcast(vix), "date")       # small VIX table is copied to every task: no shuffle
    return df.filter(F.col("n") > max(windows)).drop("n").dropna(subset=["vix_change_5"])


def features(info):
    """Read the CSVs, build features in Spark, write data/features (parquet) and the measured throughput."""
    from pyspark.sql import functions as F  # noqa: N812
    spark = spark_session()
    t0 = time.perf_counter()
    raw = spark.read.csv(str(RAW), header=True, inferSchema=True).withColumn("date", F.to_date("date"))
    vix = spark.read.csv(str(VIX), header=True, inferSchema=True).withColumn("date", F.to_date("date"))
    shutil.rmtree(FEATURES_DIR, ignore_errors=True)
    build_features(raw, vix, PARAMS["features"]["windows"], PARAMS["features"]["horizon"]) \
        .write.parquet(str(FEATURES_DIR))
    seconds = time.perf_counter() - t0
    rows_in, rows_out = raw.count(), spark.read.parquet(str(FEATURES_DIR)).count()
    spark.stop()
    info["rows"] = rows_in
    write_report("features.json", {"rows_in": rows_in, "rows_out": rows_out, "spark_seconds": round(seconds, 2),
                                   "rows_per_s": round(rows_in / seconds, 1)})


# ---------- train (Ray + MLflow) ----------
def har_model():
    """HAR: linear regression of log volatility on its daily, weekly and monthly values."""
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    return Pipeline([("har", ColumnTransformer([("lags", "passthrough", HAR)])), ("ols", LinearRegression())])


def gbm_model(cfg, seed=42):
    """Histogram gradient boosting: the LightGBM algorithm, built into scikit-learn."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(random_state=seed, **cfg)


def fit_score(model, train, val):
    """Fit on train; return (model, validation scores, variance of the validation log residuals)."""
    model.fit(train[FEATURES], train[TARGET])
    pred = model.predict(val[FEATURES])
    var = float(np.var(val[TARGET] - pred))
    return model, scores(np.exp(val[TARGET]), to_vol(pred, var)), var


def grid_point(cfg, train, val):
    """One boosting configuration on one CPU core (so sequential and Ray runs compare like with like)."""
    from threadpoolctl import threadpool_limits
    t0 = time.perf_counter()
    with threadpool_limits(1):
        _, sc, _ = fit_score(gbm_model(cfg), train, val)
    return cfg, sc, time.perf_counter() - t0


def git_commit():
    """Commit of the code behind a run, with -dirty when there are uncommitted changes."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        # dvc.lock and reports/ are rewritten by the running pipeline itself, so they do not count as edits
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", ".", ":!dvc.lock", ":!reports"])
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def train(info):
    """Compare naive, HAR and a boosting grid (sequential vs Ray) on 2022; save and log the winner."""
    import mlflow.sklearn
    import ray
    from mlflow.models import infer_signature
    from sklearn.inspection import permutation_importance

    p = PARAMS["train"]
    tr, va, _ = load_split()
    info["rows"] = len(tr) + len(va)
    grid = [dict(zip(p["grid"], v)) for v in itertools.product(*p["grid"].values())]

    t0 = time.perf_counter()
    results = [grid_point(cfg, tr, va) for cfg in grid]
    seq_s = time.perf_counter() - t0
    ray.init(num_cpus=p["ray_cpus"], num_gpus=0, include_dashboard=False, logging_level="ERROR")  # CPU only
    remote = ray.remote(grid_point)
    ray.get([remote.remote(grid[0], tr.head(100), va.head(100)) for _ in range(p["ray_cpus"])])  # start workers
    t0 = time.perf_counter()
    tr_ref, va_ref = ray.put(tr), ray.put(va)             # data goes to the object store once, not per task
    ray.get([remote.remote(cfg, tr_ref, va_ref) for cfg in grid])
    par_s = time.perf_counter() - t0
    ray.shutdown()

    best_cfg, gbm_val, _ = min(results, key=lambda r: r[1]["qlike"])
    har, har_val, _ = fit_score(har_model(), tr, va)
    comparison = {"naive": scores(np.exp(va[TARGET]), np.exp(va["log_rv_5"])), "har": har_val, "gbm": gbm_val}
    champion = "har" if har_val["qlike"] <= gbm_val["qlike"] else "gbm"
    make = har_model if champion == "har" else lambda: gbm_model(best_cfg)
    resid_var = fit_score(make(), tr, va)[2]              # bias term from the validation residuals
    ols = har.named_steps["ols"]
    har_coef = dict(zip(["intercept", *HAR], np.round([ols.intercept_, *ols.coef_], 4).tolist()))
    timing = {"grid_size": len(grid), "sequential_s": round(seq_s, 2), "ray_parallel_s": round(par_s, 2),
              "ray_speedup": round(seq_s / par_s, 2)}

    from importlib.metadata import version
    reqs = [f"{n}=={version(n)}" for n in ("mlflow-skinny", "scikit-learn", "skops", "pandas", "numpy", "scipy")]
    full = pd.concat([tr, va])
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    if "MLFLOW_RUN_ID" not in os.environ:                 # `mlflow run` has already chosen the experiment
        mlflow.set_experiment("volatility")
    with mlflow.start_run(run_name=f"train-{champion}") as run:
        # autolog records every estimator parameter and the training scores of the final fit
        mlflow.sklearn.autolog(log_models=False, log_datasets=False, silent=True)
        model = make().fit(full[FEATURES], full[TARGET])  # final model uses train + validation
        mlflow.sklearn.autolog(disable=True)
        perm = permutation_importance(model, va[FEATURES], va[TARGET], n_repeats=3, random_state=p["seed"])
        importance = dict(zip(FEATURES, perm.importances_mean.round(4).tolist()))
        signature = infer_signature(full[FEATURES].head(), model.predict(full[FEATURES].head()))
        mlflow.set_tags({"git_commit": git_commit(), "data_md5": md5(FEATURES_DIR), "champion": champion})
        mlflow.log_params({f"best_{k}": v for k, v in best_cfg.items()} | {
            "features": ",".join(FEATURES), "train_end": p["train_end"], "val_end": p["val_end"],
            "ray_cpus": p["ray_cpus"], "train_rows": len(tr), "val_rows": len(va)})
        mlflow.log_metrics({f"val_{m}_{k}": v for m, sc in comparison.items() for k, v in sc.items()}
                           | timing | {"resid_var": resid_var})
        for cfg, sc, secs in results:                     # one child run per grid point
            with mlflow.start_run(run_name="grid", nested=True):
                mlflow.log_params(cfg)
                mlflow.log_metrics({f"val_{k}": v for k, v in sc.items()} | {"fit_s": secs})
        mlflow.log_dict(importance, "permutation_importance.json")
        mlflow.log_dict(har_coef, "har_coefficients.json")
        mlflow.sklearn.log_model(model, name="model", signature=signature, pip_requirements=reqs,
                                 input_example=full[FEATURES].head(2), registered_model_name=MODEL_NAME)

    shutil.rmtree(MODEL_DIR, ignore_errors=True)
    mlflow.sklearn.save_model(model, str(MODEL_DIR), signature=signature, pip_requirements=reqs,
                              input_example=full[FEATURES].head(2))
    vol = 100 * np.exp(full[TARGET])                      # annualised volatility in %
    (MODEL_DIR / "reference.json").write_text(json.dumps({
        "run_id": run.info.run_id, "champion": champion, "features": FEATURES, "resid_var": resid_var,
        "mean": full[FEATURES].mean().round(6).to_dict(), "std": full[FEATURES].std().round(6).to_dict(),
        "bucket_edges": vol.quantile([1 / 3, 2 / 3]).round(2).tolist()}, indent=2))
    write_report("train.json", {"run_id": run.info.run_id, "champion": champion, "best_gbm": best_cfg,
                                "validation": comparison, **timing, "har_coefficients": har_coef,
                                "permutation_importance": importance})


# ---------- evaluate ----------
def evaluate(info):
    """Score the candidate on the test years; promote it (serving/ + `champion` alias) only if it beats naive."""
    import mlflow
    _, _, te = load_split()
    info["rows"] = len(te)
    ref = json.loads((MODEL_DIR / "reference.json").read_text())
    pred = to_vol(mlflow.pyfunc.load_model(str(MODEL_DIR)).predict(te[FEATURES]), ref["resid_var"])
    model, naive = scores(np.exp(te[TARGET]), pred), scores(np.exp(te[TARGET]), np.exp(te["log_rv_5"]))
    m = {f"test_{k}": v for k, v in model.items()} | {f"naive_{k}": v for k, v in naive.items()}
    m |= {"qlike_ratio": model["qlike"] / naive["qlike"], "rmse_ratio": model["rmse"] / naive["rmse"]}
    lim = PARAMS["evaluate"]
    m["passed"] = m["qlike_ratio"] <= lim["max_qlike_ratio"] and m["rmse_ratio"] <= lim["max_rmse_ratio"]
    write_report("metrics.json", m | {"test_rows": len(te), "champion": ref["champion"]})

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    client = mlflow.MlflowClient()
    for k, v in m.items():
        client.log_metric(ref["run_id"], k, float(v))
    if not m["passed"]:
        raise ValueError(f"QLIKE ratio {m['qlike_ratio']:.3f}, RMSE ratio {m['rmse_ratio']:.3f}: above the limit")
    version = client.search_model_versions(f"run_id='{ref['run_id']}'")[0].version
    client.set_registered_model_alias(MODEL_NAME, "champion", version)
    shutil.rmtree(SERVING_DIR, ignore_errors=True)
    shutil.copytree(MODEL_DIR, SERVING_DIR)
    log.info("promoted run %s (registry version %s) to serving/", ref["run_id"], version)


STAGES = {"test": run_tests, "ingest": ingest, "validate": validate, "features": features, "train": train,
          "evaluate": evaluate}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in STAGES:
        sys.exit(f"usage: python pipeline.py {{{'|'.join(STAGES)}}}")
    with tracked(sys.argv[1]) as stage_info:
        STAGES[sys.argv[1]](stage_info)
