from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict

from src import preprocessing
from src import predict as predict_module

app = FastAPI(title='Toxic Comment Classifier')


class PredictRequest(BaseModel):
    comment: str


class PredictResponse(BaseModel):
    is_toxic: bool
    toxicity_score: float
    categories: Dict[str, float]
    moderation_action: str


@app.post('/predict', response_model=PredictResponse)
def predict(req: PredictRequest):
    text = preprocessing.preprocess_text(req.comment)
    # NOTE: tokenization/padding pipeline should go here; currently stubbed
    # For now pass an all-zero sequence placeholder
    seq = [0] * 200
    categories = predict_module.predict_comment(seq)
    toxicity_score = max(categories.values()) if categories else 0.0
    is_toxic = toxicity_score >= 0.5
    if toxicity_score < 0.3:
        action = 'allow'
    elif toxicity_score < 0.75:
        action = 'flag'
    else:
        action = 'hide_and_flag'
    return PredictResponse(
        is_toxic=is_toxic,
        toxicity_score=toxicity_score,
        categories=categories,
        moderation_action=action,
    )
