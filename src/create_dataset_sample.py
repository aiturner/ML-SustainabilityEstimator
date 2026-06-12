from itertools import islice
from pathlib import Path
import os

import pandas as pd
from datasets import load_dataset

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

print("Loading dataset stream...")

train_stream = load_dataset(
    "fancyzhx/amazon_polarity",
    split="train",
    streaming=True,
)

test_stream = load_dataset(
    "fancyzhx/amazon_polarity",
    split="test",
    streaming=True,
)

print("Sampling data...")

train_sample = list(islice(train_stream, 250000))
test_sample = list(islice(test_stream, 500000))

pd.DataFrame(train_sample).to_csv(
    DATA_DIR / "amazon_train_sample.csv",
    index=False,
)

pd.DataFrame(test_sample).to_csv(
    DATA_DIR / "amazon_test_sample.csv",
    index=False,
)

print("Dataset samples saved.")

os._exit(0)
