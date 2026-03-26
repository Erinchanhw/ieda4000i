#!/usr/bin/env python3
"""
Simple script to run all LoRA experiments
"""
import subprocess
import os
import sys

def run_command(cmd, description):
    """Run a command and print output"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout[-2000:])  # Show last 2000 chars
    if result.stderr:
        print("STDERR:")
        print(result.stderr[-2000:])
    
    if result.returncode == 0:
        print(f"\n✓ SUCCESS: {description}")
    else:
        print(f"\n✗ FAILED: {description} (exit code: {result.returncode})")
    
    return result.returncode == 0

def main():
    # Change to the project directory
    os.chdir(os.path.expanduser("~/ieda4000i"))
    
    # Create necessary directories
    os.makedirs("Assignment 2/logs", exist_ok=True)
    os.makedirs("Assignment 2/outputs", exist_ok=True)
    os.makedirs("Assignment 2/results", exist_ok=True)
    
    # List of experiments
    experiments = [
        # 1. Baseline evaluation
        {
            "desc": "Baseline Evaluation (no training)",
            "cmd": "python 'Assignment 2/scripts/run_baseline.py'"
        },
        
        # 2. LoRA r=8 with 1000 samples
        {
            "desc": "LoRA r=8, 1000 samples",
            "cmd": "python 'Assignment 2/scripts/train_lora.py' --train_data 'Assignment 2/data/train_1000.jsonl' --val_data 'Assignment 2/data/val.jsonl' --lora_r 8 --output_dir 'Assignment 2/outputs/lora_r8_1k' --num_epochs 3"
        },
        
        # 3. LoRA r=8 with 2000 samples
        {
            "desc": "LoRA r=8, 2000 samples",
            "cmd": "python 'Assignment 2/scripts/train_lora.py' --train_data 'Assignment 2/data/train_2000.jsonl' --val_data 'Assignment 2/data/val.jsonl' --lora_r 8 --output_dir 'Assignment 2/outputs/lora_r8_2k' --num_epochs 3"
        },
        
        # 4. LoRA r=8 with 5000 samples (if exists)
        # {
        #     "desc": "LoRA r=8, 5000 samples",
        #     "cmd": "python 'Assignment 2/scripts/train_lora.py' --train_data 'Assignment 2/data/train_5000.jsonl' --val_data 'Assignment 2/data/val.jsonl' --lora_r 8 --output_dir 'Assignment 2/outputs/lora_r8_5k' --num_epochs 3"
        # },
        
        # 5. LoRA r=32 with 1000 samples
        {
            "desc": "LoRA r=32, 1000 samples",
            "cmd": "python 'Assignment 2/scripts/train_lora.py' --train_data 'Assignment 2/data/train_1000.jsonl' --val_data 'Assignment 2/data/val.jsonl' --lora_r 32 --output_dir 'Assignment 2/outputs/lora_r32_1k' --num_epochs 3"
        },
    ]
    
    # Run each experiment
    results = []
    for exp in experiments:
        success = run_command(exp["cmd"], exp["desc"])
        results.append((exp["desc"], success))
        
        # Ask if user wants to continue after failure
        if not success:
            response = input("\nExperiment failed. Continue? (y/n): ")
            if response.lower() != 'y':
                break
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    for desc, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {desc}")
    
    # After all experiments, run evaluation
    print("\n" + "="*60)
    print("Running evaluation and generating results...")
    print("="*60)
    
    # Extract losses and create plots
    run_command("python 'Assignment 2/scripts/extract_loss.py'", "Extract loss values")
    run_command("python 'Assignment 2/scripts/compare_results.py'", "Compare model results")
    run_command("python 'Assignment 2/scripts/run_eval.py'", "Run evaluation on test set")

if __name__ == "__main__":
    main()
