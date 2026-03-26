#!/usr/bin/env python3
"""
Create comparison table for report (without loading models)
"""
import json
import pandas as pd
from pathlib import Path

def load_test_samples(data_file, num_samples=10):
    """Load test samples from validation set"""
    samples = []
    with open(data_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            try:
                data = json.loads(line)
                samples.append(data)
            except:
                continue
    return samples

def main():
    # Set paths
    base_dir = Path.home() / "ieda4000i"
    val_file = base_dir / "Assignment_2/data/val.jsonl"
    results_dir = base_dir / "Assignment_2/results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if validation file exists
    if not val_file.exists():
        print(f"⚠ Validation file not found: {val_file}")
        print("Creating sample data for report template...")
        
        # Create sample data if real data not available
        samples = [
            {"review": "This product is amazing! Works perfectly.", "stars": 5},
            {"review": "Terrible quality, broke after one use.", "stars": 1},
            {"review": "Good value for money, satisfied with purchase.", "stars": 4},
            {"review": "Average product, nothing special.", "stars": 3},
            {"review": "Excellent customer service and fast shipping!", "stars": 5},
        ]
    else:
        samples = load_test_samples(val_file, num_samples=10)
    
    print(f"Loaded {len(samples)} test samples")
    
    # Create comparison table
    results = []
    for i, sample in enumerate(samples):
        row = {
            'Sample #': i + 1,
            'Review': sample.get('review', sample.get('text', ''))[:100],
            'Ground Truth': sample.get('stars', sample.get('label', 'N/A'))
        }
        
        # Add placeholder predictions (to be filled after training)
        row['Original Model'] = 'Placeholder - Run after training'
        row['LoRA r=8 (1k)'] = 'Placeholder - Run after training'
        row['LoRA r=8 (2k)'] = 'Placeholder - Run after training'
        row['LoRA r=8 (5k)'] = 'Placeholder - Run after training'
        row['LoRA r=32 (1k)'] = 'Placeholder - Run after training'
        
        results.append(row)
    
    # Save to CSV
    df = pd.DataFrame(results)
    output_file = results_dir / 'model_comparison_template.csv'
    df.to_csv(output_file, index=False)
    
    # Create markdown table for report
    md_file = results_dir / 'comparison_table.md'
    with open(md_file, 'w') as f:
        f.write("# Model Output Comparison\n\n")
        f.write("| Sample | Review | Ground Truth | Original | LoRA r=8 (1k) | LoRA r=8 (2k) | LoRA r=8 (5k) | LoRA r=32 (1k) |\n")
        f.write("|--------|--------|--------------|----------|---------------|---------------|---------------|----------------|\n")
        
        for row in results[:5]:  # Show first 5 samples
            f.write(f"| {row['Sample #']} | {row['Review'][:50]}... | {row['Ground Truth']} | Pending | Pending | Pending | Pending | Pending |\n")
    
    print(f"\n✓ Template saved to {output_file}")
    print(f"✓ Markdown table saved to {md_file}")
    print("\nAfter training completes, run this script again to get actual predictions.")

if __name__ == "__main__":
    main()
