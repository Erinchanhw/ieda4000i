import json
import random

# Load full training data
train_data = []
with open("../data/train.jsonl", "r") as f:
    for line in f:
        train_data.append(json.loads(line))

print(f"Total training samples: {len(train_data)}")

# Shuffle
random.seed(42)
random.shuffle(train_data)

# Create different sample sizes
sample_sizes = [1000, 2000, 5000]

for size in sample_sizes:
    if size <= len(train_data):
        sampled = train_data[:size]
        output_file = f"../data/train_{size}.jsonl"
        with open(output_file, "w") as f:
            for item in sampled:
                f.write(json.dumps(item) + "\n")
        print(f"Created {output_file} with {len(sampled)} samples")

# Create validation set (if not exists)
if not os.path.exists("../data/val.jsonl"):
    val_data = train_data[-500:]
    with open("../data/val.jsonl", "w") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")
    print(f"Created ../data/val.jsonl with {len(val_data)} samples")
