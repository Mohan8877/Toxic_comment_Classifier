from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, HTTPException, status
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
    description="Experimental toxic comment moderation service backed by the tuned DistilBERT checkpoint in models/distilbert_experiments/exp_v1_tuned.",
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
    try:
        load_model()
        logger.info("API startup complete on %s:%s", DEFAULT_API_HOST, DEFAULT_API_PORT)
    except Exception as exc:
        logger.warning("Model not ready at startup: %s", exc)


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"service": "toxic-comment-classifier", "status": "ok"}


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    try:
        model_loaded = load_model() is not None
    except Exception:
        model_loaded = False
    return HealthResponse(
        status="ok",
        service="toxic_comment_classifier_api",
        uptime_seconds=uptime_seconds(),
        model_loaded=model_loaded,
    )


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
def predict(req: PredictRequest) -> PredictResponse:
    try:
        categories = predict_comment(req.comment)
    except (TypeError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference failed: {exc}") from exc

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
        model_version="distilbert_tuned_experimental",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host=DEFAULT_API_HOST, port=DEFAULT_API_PORT, reload=True)
