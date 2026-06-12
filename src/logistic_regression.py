from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from measure_emissions import add_emissions


def main():
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    data_dir = repo_root / "data"
    results_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    train_df = pd.read_csv(data_dir / "amazon_train_sample.csv")
    test_df = pd.read_csv(data_dir / "amazon_test_sample.csv")

    train_texts = (
        train_df["title"].fillna("").astype(str)
        + " "
        + train_df["content"].fillna("").astype(str)
    )
    train_labels = train_df["label"]

    test_texts = (
        test_df["title"].fillna("").astype(str)
        + " "
        + test_df["content"].fillna("").astype(str)
    )
    test_labels = test_df["label"]

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, train_labels)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(test_labels, predictions)

    print(f"Accuracy: {accuracy:.4f}")

    # Using averaged values from turbostat
    energy_j = 309.02
    runtime_s = 16.28 

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
