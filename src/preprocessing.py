import re
import unicodedata


def clean_text(text: str) -> str:
    """Basic text cleaning: normalize, lowercase, remove excessive whitespace."""
    if not text:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    # replace URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # remove non-alphanumeric (keep basic punctuation)
    text = re.sub(r'[^a-z0-9\s\.,!?\'"-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def preprocess_text(text: str) -> str:
    """Full pipeline entrypoint for a single comment."""
    return clean_text(text)
