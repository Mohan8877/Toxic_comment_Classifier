
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

RAW = Path("data/raw/train.csv")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

labels = [
    "toxic", "severe_toxic", "obscene",
    "threat", "insult", "identity_hate"
]

df = pd.read_csv(RAW)
required = ["comment_text", *labels]

missing = set(required) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=required)
df = df[df["comment_text"].str.strip().ne("")]
df = df.drop_duplicates(subset=["comment_text"])

for label in labels:
    if not df[label].isin([0, 1]).all():
        raise ValueError(f"Invalid values in {label}")

train, temp = train_test_split(
    df, test_size=0.20, random_state=42
)
val, test = train_test_split(
    temp, test_size=0.50, random_state=42
)

for name, data in [
    ("train", train),
    ("val", val),
    ("test", test)
]:
    data.to_csv(OUT / f"{name}.csv", index=False)
    print(f"{name}: {len(data)} rows")
    print(data[labels].sum().to_dict())
