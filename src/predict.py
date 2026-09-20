import os
from typing import Dict

import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = os.environ.get('MODEL_PATH', 'models/toxic_model.keras')
TOKENIZER_PATH = os.environ.get('TOKENIZER_PATH', 'models/tokenizer.pkl')

_model = None
_tokenizer = None


def load():
    global _model, _tokenizer
    if _model is None:
        if os.path.exists(MODEL_PATH):
            _model = load_model(MODEL_PATH)
    return _model


def predict_comment(tokenized_sequence) -> Dict[str, float]:
    """Expect a padded token sequence (numpy array). Returns label probs."""
    model = load()
    if model is None:
        # stub: return zeros if model missing
        return {
            'toxic': 0.0,
            'severe_toxic': 0.0,
            'obscene': 0.0,
            'threat': 0.0,
            'insult': 0.0,
            'identity_hate': 0.0,
        }
    probs = model.predict(np.array([tokenized_sequence]))[0]
    labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    return dict(zip(labels, probs.tolist()))
