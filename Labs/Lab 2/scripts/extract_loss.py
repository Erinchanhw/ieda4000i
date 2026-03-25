import json
import matplotlib.pyplot as plt

# Define your eval_loss values from each experiment
# Replace these with your actual values from training
experiments = {
    "r=8, 1k samples": [2.5, 2.1, 1.9, 1.8, 1.7],  # Example values
    "r=8, 2k samples": [2.3, 1.9, 1.7, 1.6, 1.5],  # Example values
    "r=32, 1k samples": [2.4, 2.0, 1.8, 1.7, 1.6],  # Example values
}

epochs = [1, 2, 3, 4, 5]

plt.figure(figsize=(10, 6))
for exp_name, loss_values in experiments.items():
    plt.plot(epochs, loss_values, marker='o', label=exp_name)

plt.xlabel('Epoch')
plt.ylabel('Eval Loss')
plt.title('LoRA Fine-Tuning: Evaluation Loss Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('results/loss_comparison_chart.png', dpi=150, bbox_inches='tight')
print("Chart saved to results/loss_comparison_chart.png")
