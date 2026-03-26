import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

print("Loading model...")
model_path = "/project/ugiedahpc4/ieda4000i/models/Qwen3-1.7B"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

print("Loading evaluation data...")
eval_data = []
with open("../data/test.jsonl", "r") as f:
    for line in f:
        eval_data.append(json.loads(line))

print(f"Evaluating on {len(eval_data)} samples")

results = []
for item in tqdm(eval_data[:50]):
    prompt = item["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    prediction = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    results.append({
        "id": item.get("id", len(results)),
        "prompt": prompt,
        "true_stars": item.get("stars", "N/A"),
        "prediction": prediction.strip()
    })

with open("../results/baseline_results.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

print(f"Saved to ../results/baseline_results.jsonl")
