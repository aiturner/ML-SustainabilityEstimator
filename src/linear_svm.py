from pathlib import Path
import os

import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

from measure_emissions import add_emissions


def main():
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / "results" / "model_comparison.csv"
    csv_path.parent.mkdir(exist_ok=True)

    dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")

    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    model = LinearSVC()
    model.fit(X_train, train_labels)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(test_labels, predictions)

    print(f"Accuracy: {accuracy:.4f}")

    # These value was found from an average of running turbostat like in assignment 3
    energy_j = 43.03 
    runtime_s = 3.43

    new_row = pd.DataFrame([
        {
            "model": "Linear SVM",
            "accuracy": round(accuracy, 4),
            "energy_j": energy_j,
            "runtime_s": runtime_s,
        }
    ])

    if csv_path.exists():
        results = pd.read_csv(csv_path)
        results = results[results["model"] != "Linear SVM"]
    else:
        results = pd.DataFrame(columns=[
            "model", "accuracy", "energy_j", "runtime_s", "energy_kwh", "emissions_kgco2e"
        ])

    new_row = add_emissions(new_row)
    results = pd.concat([results, new_row], ignore_index=True)
    results.to_csv(csv_path, index=False)

    print(results)


if __name__ == "__main__":
    main()
