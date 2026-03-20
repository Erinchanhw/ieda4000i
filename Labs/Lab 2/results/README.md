# Problem 2: LoRA Fine-Tuning on Amazon Reviews - Results Summary

## 1. Loss Visualization

![Loss Comparison Chart](loss_comparison_chart.png)

### Evaluation Loss Summary
| Model | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 |
|-------|---------|---------|---------|---------|---------|
| r=8, 1k | X.XX | X.XX | X.XX | X.XX | X.XX |
| r=8, 2k | X.XX | X.XX | X.XX | X.XX | X.XX |
| r=32, 1k | X.XX | X.XX | X.XX | X.XX | X.XX |

## 2. Data Scaling Design

See [data_scaling_design.md](data_scaling_design.md)

## 3. Model Output Comparison

See [detailed_comparison.md](detailed_comparison.md)

## 4. Key Findings

- **Data Scaling Impact:** [Describe how increasing data improved/didn't improve performance]
- **Rank Impact:** [Describe how higher LoRA rank affected performance]
- **Best Model:** [State which model performed best]

## 5. Training Configuration

- Base Model: Qwen3-1.7B
- LoRA Target Modules: q_proj, v_proj, k_proj, o_proj
- Optimizer: AdamW
- Learning Rate: 2e-4
- Batch Size: 2 (gradient accumulation: 8)
- Epochs: 5
