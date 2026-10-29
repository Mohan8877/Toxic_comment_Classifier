FROM python:3.10-slim AS backend-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5

WORKDIR /app

COPY requirements-backend.txt ./
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install --index-url https://download.pytorch.org/whl/cpu --timeout 180 --retries 5 --no-cache-dir torch==2.6.0 && \
    python -m pip install --timeout 180 --retries 5 --no-cache-dir -r requirements-backend.txt

COPY . /app

RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /app/data /app/models && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${API_PORT:-8000}"]

FROM python:3.10-slim AS frontend-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5

WORKDIR /app

COPY requirements-frontend.txt ./
RUN python -m pip install --timeout 180 --retries 5 --no-cache-dir -r requirements-frontend.txt

COPY . /app

RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
