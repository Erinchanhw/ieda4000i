import argparse
import json
import os
import re
import sys
import random
from vllm import LLM, SamplingParams
from tqdm import tqdm

TEMPLATE_q2mc_en = r"""
Below is an operations research question. Build a mathematical model and corresponding python code using `coptpy` that appropriately addresses the question.

# Question:
{Question}

# Response:
"""

def extract_python_code(text):
    """Extract Python code blocks from model response"""
    # Look for code blocks with python or without language spec
    pattern = r'```(?:python)?\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    return None

def main(args):
    # Load dataset
    print(f"Loading dataset from: {args.dataset_path}")
    with open(args.dataset_path, 'r') as f:
        if args.dataset_path.endswith('.json'):
            dataset = json.load(f)
        elif args.dataset_path.endswith('.jsonl'):
            dataset = [json.loads(line) for line in f]
        else:
            # Assume it's a text file with one question per line
            dataset = [{"question": line.strip()} for line in f if line.strip()]
    
    print(f"Loaded {len(dataset)} total questions")
    
    # Randomly select N samples
    if len(dataset) > args.num_samples:
        selected_samples = random.sample(dataset, args.num_samples)
        print(f"Randomly selected {args.num_samples} samples")
    else:
        selected_samples = dataset
        print(f"Using all {len(dataset)} samples")
    
    # Initialize model
    print(f"Loading model from: {args.model_name_or_path}")
    model = LLM(model=args.model_name_or_path, tensor_parallel_size=args.tensor_parallel_size)
    print("Model loaded.")
    
    # Set sampling parameters
    stop_tokens = ["</s>"]
    if args.decoding_method == "greedy":
        sampling_params = SamplingParams(
            n=args.topk, 
            temperature=0, 
            top_p=1, 
            max_tokens=args.max_tokens,
            stop=stop_tokens
        )
    elif args.decoding_method == "sampling":
        sampling_params = SamplingParams(
            n=args.topk, 
            temperature=0.7, 
            top_p=0.95, 
            max_tokens=args.max_tokens,
            stop=stop_tokens
        )
    else:
        raise ValueError(f"Unknown decoding method: {args.decoding_method}")
    
    # Process each sample
    results = []
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, sample in enumerate(tqdm(selected_samples, desc="Processing samples")):
        question = sample.get('question', sample.get('text', str(sample)))
        
        # Create prompt
        prompt = TEMPLATE_q2mc_en.replace("{Question}", question.strip()).strip()
        
        # Generate response
        generations = model.generate([prompt], sampling_params)
        response = generations[0].outputs[0].text
        
        # Extract Python code
        code = extract_python_code(response)
        
        # Save results
        result = {
            "instance_id": idx + 1,
            "question": question,
            "response": response,
            "extracted_code": code,
            "ground_truth": sample.get('ground_truth', sample.get('answer', 'Not provided'))
        }
        results.append(result)
        
        # Save individual code file
        if code:
            code_filename = os.path.join(output_dir, f"extracted_code_{idx+1:03d}.py")
            with open(code_filename, 'w') as f:
                f.write(f'"""\nExtracted code from inference.py - Instance {idx+1:03d}\n"""\n\n')
                f.write(code)
            print(f"Saved code to: {code_filename}")
        else:
            print(f"Warning: No Python code found in response for instance {idx+1}")
        
        # Save full response for report
        response_filename = os.path.join(output_dir, f"full_response_{idx+1:03d}.txt")
        with open(response_filename, 'w') as f:
            f.write(f"Question:\n{question}\n\n")
            f.write(f"Response:\n{response}\n\n")
            if code:
                f.write(f"Extracted Code:\n{code}\n")
    
    # Save all results
    results_filename = os.path.join(output_dir, "all_results.json")
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Processed {len(results)} instances")
    print(f"✓ Results saved to: {results_filename}")
    print(f"✓ Code files saved to: {output_dir}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, 
                        help="Path to the model")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to dataset file (json, jsonl, or txt)")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save outputs")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of random samples to process")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs")
    parser.add_argument("--topk", type=int, default=1,
                        help="Number of generations per prompt")
    parser.add_argument("--decoding_method", type=str, default="greedy",
                        choices=["greedy", "sampling"],
                        help="Decoding method")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Maximum tokens to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    main(args)
