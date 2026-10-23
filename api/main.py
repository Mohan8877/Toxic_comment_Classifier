from __future__ import annotations

import logging

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from api.dependencies import require_api_key
from api.schemas import HealthResponse, PredictRequest, PredictResponse
from database.crud import create_prediction_record
from database.session import SessionLocal, init_db
from src.config import ALLOWED_ORIGINS, DEFAULT_API_HOST, DEFAULT_API_PORT, LOG_LEVEL
from src.metrics import calculate_toxicity_score, determine_moderation_action, uptime_seconds
from src.predict import load_model, predict_comment

init_db()

logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger("toxic_api")

app = FastAPI(
    title="Toxic Comment Classifier",
    version="1.0.0",
    description="Production-ready toxic comment moderation service using a Bi-LSTM classification model.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS and ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event() -> None:
    load_model()
    logger.info("API startup complete on %s:%s", DEFAULT_API_HOST, DEFAULT_API_PORT)


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"service": "toxic-comment-classifier", "status": "ok"}


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    model_loaded = load_model() is not None
    return HealthResponse(
        status="ok",
        service="toxic_comment_classifier_api",
        uptime_seconds=uptime_seconds(),
        model_loaded=model_loaded,
    )


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
def predict(req: PredictRequest) -> PredictResponse:
    categories = predict_comment(req.comment)
    toxicity_score = calculate_toxicity_score(categories)
    moderation_action = determine_moderation_action(toxicity_score)
    is_toxic = toxicity_score >= 0.5

    db: Session = SessionLocal()
    try:
        try:
            create_prediction_record(
                db,
                comment_text=req.comment,
                toxicity_score=toxicity_score,
                categories=categories,
                moderation_status=moderation_action,
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Prediction persistence failed: %s", exc)
    finally:
        db.close()

    return PredictResponse(
        is_toxic=is_toxic,
        toxicity_score=toxicity_score,
        categories=categories,
        moderation_action=moderation_action,
        model_version="bi_lstm_v1",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host=DEFAULT_API_HOST, port=DEFAULT_API_PORT, reload=True)
