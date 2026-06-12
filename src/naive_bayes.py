from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd

from measure_emissions import add_emissions

# Load dataset
dataset = load_dataset("fancyzhx/amazon_polarity")

train_texts = dataset["train"].shuffle(seed=42).select(range(100000))["content"]
train_labels = dataset["train"].shuffle(seed=42).select(range(100000))["label"]

test_texts = dataset["test"].shuffle(seed=42).select(range(10000))["content"]
test_labels = dataset["test"].shuffle(seed=42).select(range(10000))["label"]

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)

X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

# Train Naive Bayes
model = MultinomialNB()
model.fit(X_train, train_labels)

# Predict
predictions = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(test_labels, predictions)

print(f"Accuracy: {accuracy:.4f}")

energy_j = 40.95  # These value was found from an average of running turbostat like in assignment 3
runtime_s = 3.433  

results = pd.read_csv("results/model_comparison.csv")

# Remove any existing Naive Bayes row
results = results[results["model"] != "Multinomial Naive Bayes"]

new_row = pd.DataFrame([
    {
        "model": "Multinomial Naive Bayes",
        "accuracy": round(accuracy, 4),
        "energy_j": energy_j,
        "runtime_s": runtime_s
    }
])

new_row = add_emissions(new_row)

results = pd.concat([results, new_row], ignore_index=True)

print(results)

results.to_csv("results/model_comparison.csv", index=False)

