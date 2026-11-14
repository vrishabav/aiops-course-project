# Instructions for TAs

Next-week stock volatility forecaster. Background, architecture, design decisions (HLD), API (LLD), user manual
and AI disclosure are in [README.md](README.md) and the documents it links. This file: what to run and what to check; the test plan is in [TEST_PLAN.md](TEST_PLAN.md).
Allow about 15 minutes.

## 1. Mandatory documents
| Document | Where |
|---|---|
| Architecture diagram with block explanations | [docs/hld.pdf](docs/hld.pdf) §2 (also README §1) |
| High-level design (choices and rationale) | [docs/hld.pdf](docs/hld.pdf) |
| Low-level design (endpoints, I/O) | [docs/lld.pdf](docs/lld.pdf) (live version at http://localhost:8000/docs) |
| Test plan and cases | [TEST_PLAN.md](TEST_PLAN.md) |
| User manual | [docs/user_manual.pdf](docs/user_manual.pdf) (also the UI *Help* tab) |

## 2. Run it
Needs Python 3.14, Java 17, Docker with Compose v2, uv, git-lfs. Run from the repository root.
```bash
uv venv --python 3.14 .venv && source .venv/bin/activate
UV_LINK_MODE=copy uv pip install -r requirements.txt
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
dvc remote modify --local local url "$PWD/../dvcstore"      # the committed remote path is the author's machine

docker compose up -d prometheus pushgateway node-exporter grafana   # first: it receives the pipeline's metrics
dvc repro                       # test → ingest → validate → features → train → evaluate (~2 min, downloads from Yahoo)
dvc dag && dvc metrics show
dvc repro                       # no change: every stage "didn't change, skipping"
dvc push                        # optional: data and models to the DVC remote

docker compose up -d --build    # model-server, backend, frontend
docker compose ps               # model-server and backend "healthy"
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000       # in a second terminal
```
If Yahoo rate-limits, ingest retries 3 times and keeps each finished ticker in `.cache/`; re-run `dvc repro` if it still fails.
Open http://localhost:8501 once (it starts the UI's metrics endpoint). Stop with `docker compose down`.

## 3. What to check, by rubric item
| Rubric item | Check | Expected |
|---|---|---|
| UI/UX, responsive | http://localhost:8501: pick **JPM**, move the slider to **2020-03-16**; narrow the window | HIGH (red), forecast about 82-84 % (varies slightly per training run), actual 112.5 %, VIX 82.7; layout stacks on narrow width |
| Pipeline visualisation, console, speed | UI tab **Pipeline** | DAG (test → … → evaluate) with green stages and durations; a command hint for re-running; Spark rows/s; Ray speed-up (about 4x); run history with status and errors; model comparison (boosting < HAR < naive QLIKE) |
| Design docs, loose coupling | `docker-compose.yml`; `docs/hld.pdf`, `docs/lld.pdf`; http://localhost:8000/docs | UI knows only `BACKEND_URL`, backend only `MODEL_URL`; `/docs` matches the LLD |
| Implementation (style, logging, exceptions, docstrings, unit tests) | `uvx ruff check .` and `python -m pytest -q test_project.py` | ruff clean; 11 passed |
| Data engineering | `cat reports/features.json`; Spark UI http://localhost:4040 during `dvc repro -f -s features` | about 16-20k rows/s on 82,980 rows; broadcast join in the SQL tab |
| Distributed training | `cat reports/train.json` | 32 grid fits, `ray_speedup` about 4.2-4.4 |
| CI with DVC; Git, Git LFS, DVC | `dvc dag`, `dvc.lock`, `git lfs ls-files`, `dvc push` | one chain test → … → evaluate; hashes in `dvc.lock`; `outputs/` screenshots in LFS |
| Experiment tracking | MLflow http://localhost:5000 → *Model training* → Experiments → `volatility` → run `train-gbm`; Models → `vol-forecaster` | autologged params plus 32 child `grid` runs, val/test metrics, Ray timings, tags `git_commit` and `data_md5`, `permutation_importance.json`, `har_coefficients.json`, model with signature, alias `champion` |
| Prometheus and Grafana | http://localhost:9090/targets; http://localhost:3000 (no login; dashboard is the home page, refresh 5 s) | 5 targets UP (frontend after the UI was opened); every panel has data; alerts listed at :9090/alerts |
| Packaging | `docker compose ps`; `mlflow run . -e test --env-manager uv --experiment-name volatility` | 7 services, separate frontend and backend containers; MLproject builds a fresh pinned env and tests pass |

## 4. Failure drills (about 3 minutes)
```bash
# E2: model server down → backend 503, UI message, ModelServerDown fires after ~15 s
docker compose stop model-server && curl -s -w ' %{http_code}\n' localhost:8000/health
docker compose start model-server

# E3: drift → FeatureDrift fires after ~1 min (30 stocks as of the March 2020 crash)
for r in 1 2; do for t in $(curl -s localhost:8000/tickers | python3 -c "import json,sys;print(*json.load(sys.stdin)['tickers'])"); do
  curl -s -o /dev/null "localhost:8000/forecast/$t?as_of=2020-03-16"; done; done   # twice: fills the 50-request buffer
curl -s localhost:8000/metrics | grep ^backend_feature_drift_z

# E4: promotion gate blocks a worse model; serving/ keeps the previous one
sed -i 's/max_qlike_ratio: 1.0/max_qlike_ratio: 0.5/' params.yaml && dvc repro
grep run_id serving/reference.json            # unchanged
sed -i 's/max_qlike_ratio: 0.5/max_qlike_ratio: 1.0/' params.yaml && dvc repro
```
E4 shows a red *evaluate* node and the error in the UI Pipeline tab, and `PipelineStageFailed` in Prometheus until the next success.

## 5. Test plan and report
The test approach, acceptance criteria, test-case list and test report are in [TEST_PLAN.md](TEST_PLAN.md).

## 6. Evidence and notes
- Screenshots and recordings of runs are in `outputs/` (Git LFS); the pipeline numbers are in `reports/*.json` after `dvc repro`.
- Results are reproducible: dates and seeds are fixed in `params.yaml`, so model scores match.
