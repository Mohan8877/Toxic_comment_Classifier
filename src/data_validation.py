from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.config import BASE_DIR, LABELS


def detect_text_column(cols: List[str]) -> str | None:
    candidates = ["comment", "comment_text", "text", "body", "commentText"]
    for c in candidates:
        if c in cols:
            return c
    # fallback: choose first string-like column
    for c in cols:
        if c.lower().find("comment") >= 0 or c.lower().find("text") >= 0:
            return c
    return None


def validate_labels(df: pd.DataFrame, label_cols: List[str]) -> dict:
    results = {}
    for col in label_cols:
        if col not in df.columns:
            results[col] = {"present": False}
            continue
        series = df[col]
        non_null = series.dropna()
        unique_vals = sorted(non_null.unique().tolist())
        counts = int(series.notna().sum())
        zeros = int((series == 0).sum())
        ones = int((series == 1).sum())
        results[col] = {
            "present": True,
            "unique_values": unique_vals,
            "count_non_null": counts,
            "zeros": zeros,
            "ones": ones,
            "prevalence": ones / counts if counts else None,
        }
    return results


def report_duplicates_and_missing(df: pd.DataFrame, text_col: str) -> dict:
    missing_text = int(df[text_col].isna().sum())
    empty_text = int((df[text_col].astype(str).str.strip() == "").sum())
    dup_count = int(df.duplicated(subset=[text_col]).sum())
    long_comments = int((df[text_col].astype(str).str.len() > 5000).sum())
    return {
        "missing_text": missing_text,
        "empty_text": empty_text,
        "duplicates": dup_count,
        "very_long": long_comments,
    }


def make_splits(df: pd.DataFrame, label_cols: List[str], seed: int = 42):
    from sklearn.model_selection import train_test_split

    # try to stratify by multilabel combination; if too many unique combos, bucket by label-sum
    df = df.reset_index(drop=True)
    combo = df[label_cols].fillna(0).astype(int).astype(str).agg("".join, axis=1)
    n_unique = combo.nunique()
    if n_unique < 1000:
        stratify = combo
    else:
        # bucket by number of positive labels
        stratify = df[label_cols].fillna(0).astype(int).sum(axis=1)

    try:
        train_val_idx, test_idx = train_test_split(
            df.index,
            test_size=0.1,
            random_state=seed,
            stratify=stratify,
        )
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.111111,  # 0.11111 * 0.9 ~= 0.1 => val ~10%, train ~80%
            random_state=seed,
            stratify=(stratify.loc[train_val_idx] if hasattr(stratify, 'loc') else stratify[train_val_idx]),
        )
    except Exception:
        # fallback to non-stratified splits when stratification is not possible
        train_val_idx, test_idx = train_test_split(
            df.index,
            test_size=0.1,
            random_state=seed,
        )
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=0.111111,
            random_state=seed,
        )
    return list(train_idx), list(val_idx), list(test_idx)


def main(path: str, out_dir: str, seed: int = 42):
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {path}")
    df = pd.read_csv(path)
    cols = df.columns.tolist()
    text_col = detect_text_column(cols)
    if text_col is None:
        raise SystemExit("No text column found. Please provide a column named 'comment' or 'comment_text'.")

    print(f"Detected text column: {text_col}")

    label_cols = [c for c in LABELS if c in df.columns]
    if len(label_cols) != len(LABELS):
        missing = [c for c in LABELS if c not in df.columns]
        raise SystemExit(f"Missing required label columns: {missing}")

    print("Validating label columns...")
    label_report = validate_labels(df, label_cols)

    dup_report = report_duplicates_and_missing(df, text_col)

    print("Creating train/val/test splits (seed=%s)" % seed)
    train_idx, val_idx, test_idx = make_splits(df, label_cols, seed=seed)

    # Save splits as CSVs
    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    splits_meta = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "test_path": str(test_path),
    }

    report = {
        "input_path": str(path),
        "text_column": text_col,
        "label_columns": label_cols,
        "label_report": label_report,
        "duplicates_report": dup_report,
        "splits": splits_meta,
    }

    report_path = out_dir / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # also save indices for reproducibility
    indices = {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}
    with open(out_dir / "split_indices.json", "w", encoding="utf-8") as f:
        json.dump(indices, f)

    print(f"Validation complete. Report saved to {report_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate dataset and create reproducible splits")
    parser.add_argument("--input", type=str, default=str(Path(BASE_DIR) / "data" / "raw" / "train.csv"))
    parser.add_argument("--out", type=str, default=str(Path(BASE_DIR) / "data" / "processed"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.input, args.out, seed=args.seed)
