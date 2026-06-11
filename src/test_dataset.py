from datasets import load_dataset

dataset = load_dataset("rotten_tomatoes")

print(dataset)
print(dataset["train"][0])
