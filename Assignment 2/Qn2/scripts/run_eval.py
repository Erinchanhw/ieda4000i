import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    if args.adapter_path:
        print(f"Loading adapter from {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load test data
    test_data = []
    with open(args.test_file, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    print(f"Evaluating on {len(test_data)} samples")
    
    results = []
    for item in tqdm(test_data):
        prompt = item["text"]
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=0.2,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        prediction = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        results.append({
            "id": item.get("id", len(results)),
            "prompt": prompt[:100],
            "true_label": item.get("label", "N/A"),
            "prediction": prediction.strip()
        })
    
    with open(args.output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    
    print(f"Saved to {args.output_file}")

if __name__ == "__main__":
    main()
