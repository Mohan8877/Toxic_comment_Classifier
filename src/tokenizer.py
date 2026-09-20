import pickle
from typing import Optional


def save_tokenizer(tokenizer, path: str):
    with open(path, 'wb') as f:
        pickle.dump(tokenizer, f)


def load_tokenizer(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)
