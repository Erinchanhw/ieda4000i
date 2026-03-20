# Data Scaling Design for LoRA Fine-Tuning

## Introduction
To evaluate the impact of dataset size on LoRA fine-tuning performance, we conducted experiments with varying numbers of training samples while keeping other parameters constant.

## Experiment Parameters

### Fixed Parameters:
- **Model:** Qwen3-1.7B
- **LoRA Rank:** 8
- **Learning Rate:** 2e-4
- **Epochs:** 5
- **Batch Size:** 2 (with gradient accumulation steps: 8)
- **Validation Set:** 100 samples (from original validation split)
- **Test Set:** 100 samples

### Variable Parameters:
| Experiment | Sample Size | LoRA Rank | Description |
|------------|-------------|-----------|-------------|
| Baseline | 0 (no training) | N/A | Original model without fine-tuning |
| Small Dataset | 1,000 | 8 | Standard LoRA with limited data |
| Medium Dataset | 2,000 | 8 | 2x data scaling experiment |
| High Rank | 1,000 | 32 | Higher rank LoRA with same data size |

## Data Sampling Method
- Training data was randomly shuffled using fixed seed (42) for reproducibility
- Stratified sampling was not used; data was sampled randomly to maintain natural distribution
- Validation and test sets remained constant across all experiments

## Expected Impact
- Larger dataset should lead to lower eval_loss (better performance)
- Higher LoRA rank may capture more complex patterns but risks overfitting
