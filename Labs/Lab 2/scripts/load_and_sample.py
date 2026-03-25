import json
from datasets import load_from_disk
import random

# Load the dataset
data_path = "data/amazon_reviews_small_4k"
print(f"Loading dataset from {data_path}")

# Load train, validation, test
train_dataset = load_from_disk(f"{data_path}/train")
val_dataset = load_from_disk(f"{data_path}/validation")
test_dataset = load_from_disk(f"{data_path}/test")

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# Check the column names
print(f"\nDataset columns: {train_dataset.column_names}")

# Show first sample
print("\nFirst sample:")
print(train_dataset[0])

# Convert to list for sampling
train_list = list(train_dataset)

# Shuffle
random.seed(42)
random.shuffle(train_list)

# Create different sample sizes
sample_sizes = [1000, 2000, 5000]

for size in sample_sizes:
    if size <= len(train_list):
        sampled = train_list[:size]
        output_file = f"data/train_{size}.jsonl"
        with open(output_file, 'w') as f:
            for item in sampled:
                f.write(json.dumps(item) + '\n')
        print(f"Created {output_file} with {size} samples")

# Save validation set
val_list = list(val_dataset)
with open("data/val.jsonl", 'w') as f:
    for item in val_list:
        f.write(json.dumps(item) + '\n')
print(f"Created data/val.jsonl with {len(val_list)} samples")

# Save test set
test_list = list(test_dataset)
with open("data/test.jsonl", 'w') as f:
    for item in test_list:
        f.write(json.dumps(item) + '\n')
print(f"Created data/test.jsonl with {len(test_list)} samples")

print("\n✅ Data preparation complete!")
