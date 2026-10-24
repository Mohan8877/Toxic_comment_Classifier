from __future__ import annotations

from typing import Dict

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    uptime_seconds: float
    model_loaded: bool


class PredictRequest(BaseModel):
    comment: str = Field(..., min_length=1, max_length=5000, description="Comment text to analyze")


class PredictResponse(BaseModel):
    is_toxic: bool
    toxicity_score: float
    categories: Dict[str, float]
    moderation_action: str
    model_version: str = "unknown"
