from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from src.config import LABELS, MODEL_INPUT_LENGTH, MODEL_PATH, TOKENIZER_PATH
from src.preprocessing import preprocess_text

_model = None


def load_model() -> Any:
    global _model
    if _model is None and os.path.exists(MODEL_PATH):
        _model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    return _model


def _fallback_categories() -> dict[str, float]:
    return {label: 0.0 for label in LABELS}


def _encode_comment(comment: str):
    text = preprocess_text(comment)
    tokenizer = DistilBertTokenizerFast.from_pretrained(TOKENIZER_PATH)
    return tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MODEL_INPUT_LENGTH,
        return_tensors="pt",
    )


def predict_comment(comment: str) -> dict[str, float]:
    model = load_model()
    if model is None:
        return _fallback_categories()

    model.eval()
    encodings = _encode_comment(comment)
    with torch.no_grad():
        logits = model(**encodings).logits[0]
    probs = torch.sigmoid(logits).tolist()
    if len(probs) != len(LABELS):
        return _fallback_categories()
    return {label: float(prob) for label, prob in zip(LABELS, probs)}
