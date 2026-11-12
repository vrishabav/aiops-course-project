# Three images; docker compose builds each service with its own `target`
FROM python:3.14-slim AS base
WORKDIR /app
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore

FROM base AS model-server
RUN --mount=type=cache,target=/root/.cache/pip pip install mlflow-skinny==3.15.1 scikit-learn==1.9.0 \
    skops==0.14.0 pandas==2.3.3 numpy==2.5.3 scipy==1.18.1
EXPOSE 5001
CMD ["mlflow", "models", "serve", "-m", "/model", "--env-manager", "local", "-h", "0.0.0.0", "-p", "5001"]

FROM base AS backend
RUN --mount=type=cache,target=/root/.cache/pip pip install fastapi==0.141.1 uvicorn==0.52.3 requests==2.34.2 \
    pandas==2.3.3 pyarrow==25.0.1 prometheus-client==0.26.0
COPY backend.py .
EXPOSE 8000
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS frontend
RUN --mount=type=cache,target=/root/.cache/pip pip install streamlit==1.66.0 requests==2.34.2 pandas==2.3.3 \
    prometheus-client==0.26.0
COPY frontend.py .
EXPOSE 8501 9101
CMD ["streamlit", "run", "frontend.py", "--server.port", "8501", "--server.address", "0.0.0.0", \
     "--server.headless", "true", "--browser.gatherUsageStats", "false", "--client.toolbarMode", "viewer"]
