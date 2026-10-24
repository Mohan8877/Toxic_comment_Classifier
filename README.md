# Toxic Comment Classifier

Production-oriented toxic comment classification system built with a DistilBERT-based multi-label moderation pipeline, FastAPI backend, SQLAlchemy persistence, and a Streamlit moderation UI.

## Architecture overview

- API layer: FastAPI service exposing `/health` and `/predict` endpoints.
- ML layer: DistilBERT multi-label classifier for six toxicity labels: toxic, severe_toxic, obscene, threat, insult, identity_hate.
- Persistence layer: SQLAlchemy models for moderation records and predictions.
- Frontend layer: Streamlit dashboard for live inference and human moderation review.
- Deployment layer: Docker-ready, containerized service architecture with environment-based configuration.

## Model status

The active runtime checkpoint is the tuned local model located at:

- `models/distilbert_experiments/exp_v1_tuned/`

This checkpoint has been validated locally for loading and six-label inference. The original debug checkpoint remains preserved at:

- `models/distilbert_experiments/exp_v0/`

This is an experimental local checkpoint and should not be considered production-ready without additional validation and deployment review.

## Features

- Multi-label toxicity probability output across six categories
- Probabilistic moderation decisions using the documented threshold logic
- Optional API key protection for secure local deployment
- Health check endpoint and production logging
- SQLAlchemy model persistence for auditability
- Streamlit frontend for local inspection of prediction outputs
- Automated test coverage for API and integration behavior

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
models/
  distilbert_experiments/
    exp_v0/
  distilbert_model/
  distilbert_tokenizer/
tests/
  test_api.py
  test_inference.py
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

3. Start the API:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

4. Run the frontend:

```bash
streamlit run frontend/app.py
```

5. Run the test suite:

```bash
pytest -q
```

## API usage

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

The repository includes a Dockerfile, but Docker was not verified in this environment. The current local runtime configuration is:

```bash
docker build -t toxic-comment-classifier .
docker run -p 8000:8000 --env-file .env toxic-comment-classifier
```

This should be treated as a local deployment template, not as a verified build result for this host.

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
- Model artifacts are stored under `models/` and should be versioned and mounted as persistent volumes when used in deployment.
- Database can remain SQLite for local development or be switched to PostgreSQL for production by setting `DATABASE_URL`.

## Observability

- Structured logging is enabled via Python `logging` with configurable `LOG_LEVEL`.
- `/health` endpoint supports health probes.
- Prediction requests are logged and persisted to the database for auditability.
- Metrics for uptime and moderation decisions can be integrated with Prometheus or a monitoring stack.
streamlit run yourscript.py