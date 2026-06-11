from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd

from measure_emissions import add_emissions

# Load dataset
dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")

train_texts = dataset["train"]["text"]
train_labels = dataset["train"]["label"]

test_texts = dataset["test"]["text"]
test_labels = dataset["test"]["label"]

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

energy_j = 40.95  ## These value was found from an average of running turbostat like in assignment 3
runtime_s = 3.433  

results = pd.read_csv("results/model_comparison.csv")

new_row = pd.DataFrame([
    {
        "model": "Multinomial Naive Bayes",
        "accuracy": round(accuracy, 4),
        "energy_j": energy_j,
        "runtime_s": runtime_s
    }
])

results = pd.concat([results, new_row], ignore_index=True) 
results = add_emissions(results)

print(results)

results.to_csv("results/model_comparison.csv", index=False)

