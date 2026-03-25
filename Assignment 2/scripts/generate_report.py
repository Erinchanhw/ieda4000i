#!/usr/bin/env python3
"""
Generate all content needed for the report
"""
from pathlib import Path

def main():
    base_dir = Path.home() / "ieda4000i"
    results_dir = base_dir / "Assignment_2/results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("PROBLEM 2 - REPORT CONTENT")
    print("="*60)
    
    # 1. Parameters Used
    print("\n" + "="*60)
    print("1. DATA SCALING DESIGN PARAMETERS")
    print("="*60)
    print("\n| Experiment | Sample Size | LoRA Rank | Description |")
    print("|------------|-------------|-----------|-------------|")
    print("| Baseline | 0 | N/A | Original model without fine-tuning |")
    print("| Experiment 1 | 1,000 | r=8 | Small dataset, low rank |")
    print("| Experiment 2 | 2,000 | r=8 | Medium dataset, low rank |")
    print("| Experiment 3 | 5,000 | r=8 | Large dataset, low rank |")
    print("| Experiment 4 | 1,000 | r=32 | Small dataset, high rank |")
    
    # 2. Training Configuration
    print("\n" + "="*60)
    print("2. TRAINING CONFIGURATION")
    print("="*60)
    print("""
- Base Model: Qwen2.5-1.7B
- LoRA Target Modules: q_proj, v_proj
- LoRA Alpha: 32
- LoRA Dropout: 0.1
- Optimizer: AdamW
- Learning Rate: 2e-4
- Batch Size: 4
- Epochs: 3
- Evaluation Strategy: per epoch
    """)
    
    # 3. Expected Results (to be filled after training)
    print("\n" + "="*60)
    print("3. EVALUATION RESULTS")
    print("="*60)
    print("\nAfter training completes, run the following commands to get results:")
    print("  python extract_all_results.py")
    print("  python compare_model_outputs.py")
    print("\nExpected metrics to report:")
    print("  - Final eval_loss for each experiment")
    print("  - Training time per experiment")
    print("  - Comparison of predictions vs ground truth")
    
    # 4. Analysis Template
    print("\n" + "="*60)
    print("4. ANALYSIS TEMPLATE")
    print("="*60)
    print("""
Based on the results, the following analysis should be included:

a) Effect of Increasing Data Size:
   - Compare eval_loss between 1k, 2k, and 5k samples (all with r=8)
   - Discuss whether more data leads to better performance
   - Analyze if improvements diminish after certain point

b) Effect of Higher Rank:
   - Compare r=8 vs r=32 with same data size (1k samples)
   - Discuss if higher rank leads to better adaptation
   - Consider trade-offs between model capacity and overfitting

c) Qualitative Analysis:
   - Compare sample outputs from different models
   - Identify cases where fine-tuning improves predictions
   - Note any failure cases or hallucinations
    """)
    
    # Save to file
    report_file = results_dir / 'report_content.txt'
    with open(report_file, 'w') as f:
        f.write("PROBLEM 2 REPORT CONTENT\n")
        f.write("="*60 + "\n\n")
        f.write("1. DATA SCALING DESIGN PARAMETERS\n")
        f.write("-"*40 + "\n")
        f.write("Experiments conducted:\n")
        f.write("- Baseline: Original Qwen2.5-1.7B model\n")
        f.write("- LoRA r=8 with 1,000 training samples\n")
        f.write("- LoRA r=8 with 2,000 training samples\n")
        f.write("- LoRA r=8 with 5,000 training samples\n")
        f.write("- LoRA r=32 with 1,000 training samples\n\n")
        f.write("2. LOSS VISUALIZATION\n")
        f.write("-"*40 + "\n")
        f.write("Eval_loss comparison plot will be saved to Assignment_2/figures/eval_loss_comparison.png\n\n")
        f.write("3. MODEL OUTPUT COMPARISON\n")
        f.write("-"*40 + "\n")
        f.write("Comparison table will be saved to Assignment_2/results/model_comparison.csv\n")
    
    print(f"\n✓ Report template saved to {report_file}")
    print("\nTo complete the report:")
    print("  1. After training finishes, run: python extract_all_results.py")
    print("  2. Run: python compare_model_outputs.py")
    print("  3. Add the loss plot and comparison table to your report")

if __name__ == "__main__":
    main()
