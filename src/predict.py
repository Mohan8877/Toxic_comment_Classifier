from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from src.config import LABELS, MODEL_INPUT_LENGTH, MODEL_PATH, TOKENIZER_PATH
from src.preprocessing import preprocess_text

_model = None
_tokenizer = None


def _artifact_dir_label(path: str) -> str:
    return os.path.basename(os.path.normpath(path))


def load_model() -> Any:
    global _model
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(
            f"Model artifacts are missing at '{MODEL_PATH}'. Expected a versioned DistilBERT checkpoint directory."
        )
    if not os.path.isdir(TOKENIZER_PATH):
        raise FileNotFoundError(
            f"Tokenizer artifacts are missing at '{TOKENIZER_PATH}'. Expected the tokenizer directory for the same checkpoint."
        )

    if _model is None:
        try:
            _model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        except Exception as exc:  # pragma: no cover - defensive wrapper for real runtime errors
            raise ValueError(
                f"Unable to load model from '{MODEL_PATH}'. Confirm the directory contains a compatible DistilBERT checkpoint."
            ) from exc
    return _model


def load_tokenizer() -> Any:
    global _tokenizer
    if _tokenizer is None:
        try:
            _tokenizer = DistilBertTokenizerFast.from_pretrained(TOKENIZER_PATH)
        except Exception as exc:  # pragma: no cover - defensive wrapper for real runtime errors
            raise ValueError(
                f"Unable to load tokenizer from '{TOKENIZER_PATH}'. Confirm the tokenizer files match the model checkpoint."
            ) from exc
    return _tokenizer


def _validate_comment(comment: str) -> str:
    if not isinstance(comment, str):
        raise TypeError("Comment must be a string.")
    text = comment.strip()
    if not text:
        raise ValueError("Comment text cannot be empty.")
    if len(text) > 5000:
        raise ValueError("Comment text exceeds the maximum supported length of 5000 characters.")
    return text


def _encode_comment(comment: str):
    text = preprocess_text(_validate_comment(comment))
    tokenizer = load_tokenizer()
    return tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MODEL_INPUT_LENGTH,
        return_tensors="pt",
    )


def predict_comment(comment: str) -> dict[str, float]:
    model = load_model()
    model.eval()
    encodings = _encode_comment(comment)
    with torch.no_grad():
        logits = model(**encodings).logits[0]
    probs = torch.sigmoid(logits)

    if probs.numel() != len(LABELS):
        raise ValueError(f"Expected {len(LABELS)} probabilities but received {probs.numel()}.")
    if not torch.isfinite(probs).all():
        raise ValueError("Model produced non-finite probabilities; inference failed.")

    values = probs.detach().cpu().tolist()
    return {label: float(prob) for label, prob in zip(LABELS, values)}
