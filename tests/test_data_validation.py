import pandas as pd

from src.config import LABELS
from src.data_validation import detect_text_column, make_splits, main, validate_labels


def test_detect_text_column_matches_comment_text():
    cols = ["id", "comment_text", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    assert detect_text_column(cols) == "comment_text"


def test_validate_labels_and_order():
    df = pd.DataFrame(
        {
            "comment_text": ["a", "b", "c", "d"],
            "toxic": [0, 1, 0, 1],
            "severe_toxic": [0, 0, 1, 0],
            "obscene": [0, 1, 0, 0],
            "threat": [0, 0, 0, 1],
            "insult": [0, 0, 1, 0],
            "identity_hate": [1, 0, 0, 0],
        }
    )
    report = validate_labels(df, LABELS)
    assert list(report.keys()) == LABELS
    assert report["toxic"]["unique_values"] == [0, 1]
    assert report["threat"]["ones"] == 1


def test_make_splits_are_disjoint_and_cover_dataset():
    rows = []
    labels = [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 1],
    ]
    for i, label in enumerate(labels):
        rows.append({"comment_text": f"comment {i}", **dict(zip(LABELS, label))})
    df = pd.DataFrame(rows)

    train_idx, val_idx, test_idx = make_splits(df, LABELS, seed=42)
    all_idx = set(train_idx) | set(val_idx) | set(test_idx)
    assert len(all_idx) == len(df)
    assert set(train_idx).isdisjoint(val_idx)
    assert set(train_idx).isdisjoint(test_idx)
    assert set(val_idx).isdisjoint(test_idx)


def test_main_generates_split_csvs(tmp_path):
    rows = []
    for i in range(24):
        label_row = [1 if (i + j) % 3 == 0 else 0 for j in range(len(LABELS))]
        rows.append({"comment_text": f"comment {i}", **dict(zip(LABELS, label_row))})
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "synthetic.csv"
    df.to_csv(csv_path, index=False)
    out_dir = tmp_path / "processed"

    report = main(str(csv_path), str(out_dir), seed=42)

    assert report["text_column"] == "comment_text"
    assert (out_dir / "train.csv").exists()
    assert (out_dir / "val.csv").exists()
    assert (out_dir / "test.csv").exists()
    assert (out_dir / "validation_report.json").exists()
