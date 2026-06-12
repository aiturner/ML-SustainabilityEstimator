from pathlib import Path
from itertools import islice

import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from measure_emissions import add_emissions


def main():
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    results_dir.mkdir(exist_ok=True)

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

    train_sample = list(islice(train_stream, 20000))
    test_sample = list(islice(test_stream, 5000))

    train_texts = [f"{x['title']} {x['content']}" for x in train_sample]
    train_labels = [x["label"] for x in train_sample]

    test_texts = [f"{x['title']} {x['content']}" for x in test_sample]
    test_labels = [x["label"] for x in test_sample]

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train_labels)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(test_labels, predictions)

    print(f"Accuracy: {accuracy:.4f}")

    # After running turbostat i added these values in
    energy_j = 81.99 
    runtime_s = 7.17 

    results = pd.DataFrame([
        {
            "model": "Logistic Regression",
            "accuracy": round(accuracy, 4),
            "energy_j": energy_j,
            "runtime_s": runtime_s,
        }
    ])

    results = add_emissions(results)

    print(results)

    results.to_csv(results_dir / "model_comparison.csv", index=False)
    print(f"Results saved to {results_dir / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
    
