from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from src.config import LABELS, MODEL_INPUT_LENGTH, MODELS_DIR, MODEL_PATH, PREPROCESSING_VERSION, TOKENIZER_PATH
from src.preprocessing import preprocess_text

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _collect_package_versions() -> Dict[str, str]:
    versions = {}
    for name in ["torch", "transformers", "pandas", "numpy", "scikit-learn", "pytest"]:
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = "not-installed"
    return versions


def _validate_split_files(train_csv: str, val_csv: str, test_csv: str | None = None) -> None:
    for path in [train_csv, val_csv]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required dataset: {path}")
    if test_csv and os.path.exists(test_csv):
        logger.info("Test split exists at %s but will not be used during training or validation selection.", test_csv)


class MultiLabelDataset(Dataset):
    def __init__(self, texts: List[str], labels: np.ndarray, tokenizer: DistilBertTokenizerFast, max_length: int):
        self.texts = texts
        self.labels = labels.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = preprocess_text(self.texts[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item


def compute_metrics(pred) -> Dict[str, float]:
    logits = pred.predictions
    labels = pred.label_ids
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    preds = (probs >= 0.5).astype(int)

    metrics = {}
    metrics["macro_f1"] = float(f1_score(labels, preds, average="macro", zero_division=0))
    metrics["micro_f1"] = float(f1_score(labels, preds, average="micro", zero_division=0))
    metrics["macro_precision"] = float(precision_score(labels, preds, average="macro", zero_division=0))
    metrics["macro_recall"] = float(recall_score(labels, preds, average="macro", zero_division=0))
    return metrics


def load_dataframe(path: str, limit: int | None = None) -> Tuple[List[str], np.ndarray]:
    df = pd.read_csv(path)
    if limit is not None:
        df = df.head(limit)
    text_col = None
    for c in ["comment", "comment_text", "text"]:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        raise ValueError(f"No text column found in dataset: {path}")
    missing = [label for label in LABELS if label not in df.columns]
    if missing:
        raise ValueError(f"Missing label columns in {path}: {missing}")
    labels = df[[l for l in LABELS]].fillna(0).values.astype(np.float32)
    texts = df[text_col].astype(str).tolist()
    return texts, labels


def _save_runtime_metadata(output_dir: Path, config: Dict[str, object], metrics: Dict[str, float] | None = None):
    metadata = {
        "model_identifier": "distilbert-base-uncased",
        "label_order": LABELS,
        "preprocessing_version": PREPROCESSING_VERSION,
        "package_versions": _collect_package_versions(),
        "system": {
            "platform": platform.platform(),
            "python_version": sys.version.replace("\n", " "),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
        },
        "training_config": config,
    }
    if metrics is not None:
        metadata["validation_metrics"] = metrics
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def train(
    train_csv: str,
    val_csv: str,
    base_model: str = "distilbert-base-uncased",
    output_dir: str | Path = MODEL_PATH,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    weight_decay: float = 0.0,
    max_length: int = MODEL_INPUT_LENGTH,
    seed: int = 42,
    train_limit: int | None = None,
    val_limit: int | None = None,
    test_csv: str | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _validate_split_files(train_csv, val_csv, test_csv=test_csv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    tokenizer = DistilBertTokenizerFast.from_pretrained(base_model)

    train_texts, train_labels = load_dataframe(train_csv, limit=train_limit)
    val_texts, val_labels = load_dataframe(val_csv, limit=val_limit)

    if train_labels.shape[1] != len(LABELS):
        raise ValueError(f"Unexpected label count: expected {len(LABELS)}, got {train_labels.shape[1]}")
    if val_labels.shape[1] != len(LABELS):
        raise ValueError(f"Unexpected label count: expected {len(LABELS)}, got {val_labels.shape[1]}")

    train_ds = MultiLabelDataset(train_texts, train_labels, tokenizer, max_length=max_length)
    val_ds = MultiLabelDataset(val_texts, val_labels, tokenizer, max_length=max_length)

    try:
        model = DistilBertForSequenceClassification.from_pretrained(str(output_dir), local_files_only=False)
        logger.info("Loaded existing model from %s", output_dir)
    except Exception:
        model = DistilBertForSequenceClassification.from_pretrained(base_model, num_labels=len(LABELS))
        logger.info("Loaded base model %s", base_model)

    model.config.problem_type = "multi_label_classification"
    model.to(device)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "training"),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=seed,
        fp16=torch.cuda.is_available(),
        logging_strategy="epoch",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    start = monotonic()
    trainer.train()
    end = monotonic()

    metrics = trainer.evaluate()
    tokenizer.save_pretrained(str(output_dir))
    trainer.save_model(str(output_dir))

    config = {
        "model_identifier": base_model,
        "output_dir": str(output_dir),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "max_length": max_length,
        "seed": seed,
        "train_limit": train_limit,
        "val_limit": val_limit,
        "train_time_seconds": end - start,
        "device": str(device),
        "loss": "BCEWithLogitsLoss",
        "label_order": LABELS,
        "preprocessing_version": PREPROCESSING_VERSION,
        "uses_test_split": False,
    }

    _save_runtime_metadata(output_dir, config, metrics={k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()})

    with open(output_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    logger.info("Training complete. Artifacts saved to %s", output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for multi-label toxic classification")
    parser.add_argument("--train-csv", type=str, default="data/processed/train.csv")
    parser.add_argument("--val-csv", type=str, default="data/processed/val.csv")
    parser.add_argument("--test-csv", type=str, default="data/processed/test.csv")
    parser.add_argument("--base-model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output-dir", type=str, default=str(MODEL_PATH))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=MODEL_INPUT_LENGTH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        base_model=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        seed=args.seed,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_csv=args.test_csv,
    )
