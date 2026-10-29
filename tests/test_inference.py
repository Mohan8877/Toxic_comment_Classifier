import os

import pytest

from src.config import LABELS, MODEL_PATH, TOKENIZER_PATH
from src.predict import load_model, load_tokenizer, predict_comment


def test_inference_shape_and_label_order():
    comment = "This is a test comment about toxic behavior."
    probs = predict_comment(comment)

    assert list(probs.keys()) == LABELS
    assert len(probs) == len(LABELS)
    assert all(isinstance(value, float) for value in probs.values())
    assert all(0.0 <= value <= 1.0 for value in probs.values())
    assert all((value == value) for value in probs.values())


def test_invalid_input_raises():
    with pytest.raises(ValueError):
        predict_comment("")
    with pytest.raises(ValueError):
        predict_comment(" " * 5001)
    with pytest.raises(TypeError):
        predict_comment(None)


def test_artifact_loading_uses_expected_paths():
    assert os.path.isdir(MODEL_PATH)
    assert os.path.isdir(TOKENIZER_PATH)
    model = load_model()
    tokenizer = load_tokenizer()
    assert model is not None
    assert tokenizer is not None


def test_missing_artifact_directory_raises(monkeypatch):
    import src.predict as predict_module

    monkeypatch.setattr(predict_module, "MODEL_PATH", "/definitely/missing/model_dir")
    with pytest.raises(FileNotFoundError):
        predict_module.load_model()
