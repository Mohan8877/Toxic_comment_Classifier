import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
MODEL_EXPERIMENTS_DIR = BASE_DIR / "models" / "distilbert_experiments"

DEFAULT_EXPERIMENT_DIR = MODEL_EXPERIMENTS_DIR / "exp_v0"
MODEL_PATH = os.getenv("MODEL_PATH", str(DEFAULT_EXPERIMENT_DIR))
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", str(DEFAULT_EXPERIMENT_DIR))

PREPROCESSING_VERSION = "preprocess_v1"
PREPROCESSING_DESCRIPTION = (
    "Current preprocessing preserves text content and punctuation while normalizing whitespace; "
    "no aggressive cleaning is applied before tokenization."
)

DEFAULT_API_HOST = os.getenv("API_HOST", "0.0.0.0")
DEFAULT_API_PORT = int(os.getenv("API_PORT", "8000"))
DEFAULT_FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "8501"))
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./toxicity.db")
ALLOWED_ORIGINS = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",") if origin.strip()]
API_KEY = os.getenv("API_KEY")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]
MODEL_INPUT_LENGTH = 200

MODELS_DIR.mkdir(parents=True, exist_ok=True)
