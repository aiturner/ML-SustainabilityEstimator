from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from measure_emissions import add_emissions
import pandas as pd

dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")

train_texts = dataset["train"]["text"]
train_labels = dataset["train"]["label"]

test_texts = dataset["test"]["text"]
test_labels = dataset["test"]["label"]

vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, train_labels)

predictions = model.predict(X_test)
accuracy = accuracy_score(test_labels, predictions)

#print(f"Accuracy: {accuracy:.4f}")

# This value was found from an average of running sudo turbostat -q --Joules --show Pkg_J
energy_j = 49.55 

results = pd.DataFrame([
    {
        "model": "Logistic Regression",
        "accuracy": round(accuracy, 4),
        "energy_j": energy_j,
        "runtime_s": 3.82
    }
])


results = add_emissions(results)

results.to_csv("results/model_comparison.csv", index=False)

print("Results saved to results/model_comparison.csv")
