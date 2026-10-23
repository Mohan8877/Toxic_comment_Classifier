# Toxic Comment Classifier

Production-oriented toxic comment classification system built with a Bi-LSTM deep learning model, FastAPI backend, SQLAlchemy persistence, and a Streamlit moderation UI.

## Architecture overview

- API layer: FastAPI service exposing `/health` and `/predict` endpoints.
- ML layer: Bi-LSTM model for six toxicity labels: toxic, severe_toxic, obscene, threat, insult, identity_hate.
- Persistence layer: SQLAlchemy models for moderation records and predictions.
- Frontend layer: Streamlit dashboard for live inference and human moderation review.
- Deployment layer: Docker-ready, containerized service architecture with environment-based configuration.

## Features

- Deep learning classifier trained on comment text
- Probability output across six toxicity categories
- Deterministic moderation action based on score thresholds
- Optional API key protection for secure production deployment
- Health check endpoint and production logging
- SQLAlchemy model persistence for auditability
- Docker configuration for containerized deployment
- Automated test coverage for API behaviors

## Project layout

```text
api/
  dependencies.py
  main.py
  schemas.py
database/
  crud.py
  models.py
  session.py
frontend/
  app.py
src/
  config.py
  metrics.py
  model.py
  predict.py
  preprocessing.py
  tokenizer.py
  train.py
tests/
  test_api.py
  test_model.py
Dockerfile
requirements.txt
README.md
.env.example
```

## Quick start

1. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure environment variables:

```bash
cp .env.example .env
```

3. Train the model:

```bash
python -m src.train --epochs 5 --batch-size 32
```

4. Run the API:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

5. Run the frontend:

```bash
streamlit run frontend/app.py
```

## Production API usage

### Health check

```bash
curl http://localhost:8000/health
```

### Prediction request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"comment": "You are a terrible person and I hate you"}'
```

If `API_KEY` is configured, include:

```bash
-H "X-API-Key: your-key"
```

## Docker

Build and run with Docker:

```bash
docker build -t toxic-comment-classifier .
docker run -p 8000:8000 --env-file .env toxic-comment-classifier
```

## Testing

```bash
pytest -q
```

## Security model

- API key can be enforced through `X-API-Key`.
- CORS is restricted via environment configuration.
- Sensitive deployment values are supplied through environment variables rather than committed source code.
- The application is structured around explicit service boundaries to keep ML, storage, and API logic isolated.

## Deployment topology

- API service runs behind a reverse proxy / load balancer in production.
- Streamlit frontend is deployed as a separate web client or embedded dashboard.
- Model artifacts are stored in `models/` and mounted as persistent volumes if used in production.
- Database can be switched from SQLite for local development to PostgreSQL in production by setting `DATABASE_URL`.

## CI/CD recommendations

- Run `pytest` on every PR.
- Build Docker image in CI and scan for vulnerabilities.
- Deploy tagged image to a container registry and rollout with versioned image tags.
- Use health endpoints and a staged rollout strategy for production reliability.

## Observability

- Structured logging is enabled via Python `logging` with configurable `LOG_LEVEL`.
- `/health` endpoint supports health probes.
- Prediction requests are logged and persisted to the database for auditability.
- Metrics for uptime and moderation decisions can be integrated with Prometheus or a monitoring stack.
