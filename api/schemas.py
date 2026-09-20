from pydantic import BaseModel
from typing import Dict


class PredictRequest(BaseModel):
    comment: str


class PredictResponse(BaseModel):
    is_toxic: bool
    toxicity_score: float
    categories: Dict[str, float]
    moderation_action: str
