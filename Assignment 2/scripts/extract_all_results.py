#!/usr/bin/env python3
"""
Extract all results from training logs and create report visualizations
"""
import json
import re
import glob
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

def extract_loss_from_log(log_file, model_name):
    """Extract eval_loss values from log file"""
    losses = []
    epochs = []
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            # Look for eval_loss pattern
            pattern = r'eval_loss["\s:]+([0-9.]+)'
            matches = re.findall(pattern, content)
            
            for i, match in enumerate(matches):
                losses.append(float(match))
                epochs.append(i + 1)
                
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
    
    return epochs, losses

def plot_loss_comparison(all_losses, output_dir):
    """Create comparison plot of all models"""
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, (model_name, (epochs, losses)) in enumerate(all_losses.items()):
        if losses:
            plt.plot(epochs, losses, 
                    marker=markers[i % len(markers)], 
                    label=model_name,
                    color=colors[i % len(colors)],
                    linewidth=2,
                    markersize=6)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Evaluation Loss', fontsize=12)
    plt.title('LoRA Fine-tuning: Evaluation Loss Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(output_dir / 'eval_loss_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {output_dir / 'eval_loss_comparison.png'}")
    plt.close()

def main():
    # Set paths
    base_dir = Path.home() / "ieda4000i"
    log_dir = base_dir / "Assignment_2/logs"
    output_dir = base_dir / "Assignment_2/figures"
    results_dir = base_dir / "Assignment_2/results"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all log files
    log_files = glob.glob(str(log_dir / "*.out")) + glob.glob(str(log_dir / "*.log"))
    
    if not log_files:
        print("No log files found.")
        print("Check: ls -la ~/ieda4000i/Assignment_2/logs/")
        return
    
    print(f"Found {len(log_files)} log files")
    
    # Extract losses for each model
    all_losses = {}
    all_loss_data = []
    
    for log_file in log_files:
        log_file = str(log_file)
        
        # Determine model name from filename
        if 'lora_r8_1k' in log_file or 'r8_1k' in log_file:
            model_name = 'LoRA r=8 (1k samples)'
        elif 'lora_r8_2k' in log_file or 'r8_2k' in log_file:
            model_name = 'LoRA r=8 (2k samples)'
        elif 'lora_r8_5k' in log_file or 'r8_5k' in log_file:
            model_name = 'LoRA r=8 (5k samples)'
        elif 'lora_r32_1k' in log_file or 'r32_1k' in log_file:
            model_name = 'LoRA r=32 (1k samples)'
        elif 'baseline' in log_file:
            model_name = 'Baseline'
        else:
            model_name = Path(log_file).stem.replace('_', ' ')
        
        print(f"\nProcessing: {model_name}")
        epochs, losses = extract_loss_from_log(log_file, model_name)
        
        if losses:
            all_losses[model_name] = (epochs, losses)
            print(f"  ✓ Found {len(losses)} eval_loss values")
            print(f"  Final eval_loss: {losses[-1]:.4f}")
            if len(losses) > 1:
                print(f"  Best eval_loss: {min(losses):.4f}")
            
            # Save to CSV
            df = pd.DataFrame({'epoch': epochs, 'eval_loss': losses})
            csv_file = results_dir / f'{model_name.replace(" ", "_").replace("=", "").replace("(", "").replace(")", "")}_losses.csv'
            df.to_csv(csv_file, index=False)
            all_loss_data.append({'model': model_name, 'final_loss': losses[-1], 'best_loss': min(losses)})
        else:
            print(f"  ✗ No eval_loss found in this log")
    
    # Create loss comparison plot
    if all_losses:
        plot_loss_comparison(all_losses, output_dir)
        
        # Save summary to CSV
        summary_df = pd.DataFrame(all_loss_data)
        summary_df.to_csv(results_dir / 'loss_summary.csv', index=False)
        print(f"\n✓ Loss summary saved to {results_dir / 'loss_summary.csv'}")
        
        # Print summary table
        print("\n" + "="*60)
        print("LOSS SUMMARY")
        print("="*60)
        print(summary_df.to_string(index=False))
    else:
        print("\n⚠ No eval_loss data found. Check if training completed successfully.")
    
    print(f"\n✓ Results saved to {results_dir}")
    print(f"✓ Figures saved to {output_dir}")

if __name__ == "__main__":
    main()
