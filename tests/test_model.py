from src.model import build_model
from src.config import MODEL_INPUT_LENGTH


def test_build_model_returns_tensorflow_model():
    model = build_model(vocab_size=30000, max_len=MODEL_INPUT_LENGTH, num_labels=6)
    assert model is not None
    assert model.num_labels == 6
