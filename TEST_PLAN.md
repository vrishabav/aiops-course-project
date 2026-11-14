# Test plan and report
**Approach.** Unit tests (Spark features against pandas, metrics, data rules), API tests (`test_project.py`, run by `python pipeline.py test` as the first DVC stage and shown in the Pipeline tab; report in `reports/junit.xml`), and end-to-end checks of the running stack.

**Acceptance criteria**
1. All automated tests pass.
2. The pipeline finishes with `reports/metrics.json` `passed: true`: on 2023-2025 the model's QLIKE and RMSE are below the naive forecast's.
3. With the stack up every service answers its health URL and the UI shows a forecast in under 1 s.
4. Stopping the model server gives 503 on `/health` and `/forecast`, a friendly UI message and a `ModelServerDown` alert; starting it restores service.
5. Requests for March 2020 push the drift z-score above 2 and fire `FeatureDrift`.

| ID | Type | Case | Expected |
|---|---|---|---|
| T1 | unit | Spark HAR features, target and VIX join | equal to pandas (1e-9); last 3 targets empty; partial windows dropped |
| T2 | unit | RMSE, RMSPE, QLIKE, bias correction | 0 for a perfect forecast; QLIKE punishes under-forecasts more |
| T3 | unit | clean OHLCV data | every data rule passes |
| T4 | unit | high below close, null volume, duplicate row | matching rules fail |
| T5 | unit | missing column, short history, missing ticker or VIX dates | reported as failed |
| T11 | unit | a failing stage inside `tracked` | run history records status `failed` and the error text |
| T6 | API | forecast for the latest date | 200; log forecast converted to %; bucket; no realised value |
| T7 | API | `as_of` in the past / before the data / invalid | earlier row with realised value and forecast-error metric / 404 / 422 |
| T8 | API | unknown ticker | 404 |
| T9 | API | model server down | 503 on `/forecast` and `/health` |
| T10 | API | `/tickers`, `/metrics` | date range; metric names present |
| E1 | e2e | `docker compose up`, health URLs | all 200; Prometheus targets UP |
| E2 | e2e | stop model server | criterion 4 |
| E3 | e2e | March 2020 requests | criterion 5 |
| E4 | e2e | gate set to 0.5 | evaluate fails, red node, `serving/` unchanged |

**Report** (author's runs, 2026-10-30)
| Run | Total | Passed | Failed | Criteria met |
|---|---|---|---|---|
| pytest (`reports/junit.xml`) | 11 | 11 | 0 | 1 |
| evaluate (`reports/metrics.json`: QLIKE ratio 0.73, RMSE ratio 0.85) | 1 | 1 | 0 | 2 |
| E1-E3 | 3 | 3 | 0 | 3, 4, 5 |
| E4 | 1 | 1 | 0 | gate blocks a worse model |
