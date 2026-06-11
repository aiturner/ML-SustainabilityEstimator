from datasets import load_dataset

dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")
print(dataset)
print(dataset["train"][0])
