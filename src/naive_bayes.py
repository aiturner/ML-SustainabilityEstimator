from pathlib import Path
from itertools import islice

import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

from measure_emissions import add_emissions


def main():
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "model_comparison.csv"

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

    model = MultinomialNB()
    model.fit(X_train, train_labels)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(test_labels, predictions)

    print(f"Accuracy: {accuracy:.4f}")

    energy_j = 80.62  #value from averaging after turbostat
    runtime_s = 7.18 

    new_row = pd.DataFrame([
        {
            "model": "Multinomial Naive Bayes",
            "accuracy": round(accuracy, 4),
            "energy_j": energy_j,
            "runtime_s": runtime_s,
        }
    ])

    new_row = add_emissions(new_row)

    if csv_path.exists():
        results = pd.read_csv(csv_path)
        results = results[results["model"] != "Multinomial Naive Bayes"]
    else:
        results = pd.DataFrame(columns=[
            "model", "accuracy", "energy_j", "runtime_s", "energy_kwh", "emissions_kgco2e"
        ])

    results = pd.concat([results, new_row], ignore_index=True)
    results.to_csv(csv_path, index=False)

    print(results)


if __name__ == "__main__":
    main()
