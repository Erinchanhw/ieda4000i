# Lab 2: LoRA Fine-Tuning on Amazon Reviews

## Objective
Fine-tune Qwen3-1.7B using LoRA to predict sentiment stars from Amazon reviews.

## Experiments
1. **Baseline**: Evaluate untrained model
2. **LoRA r=8, 1,000 samples**: Standard LoRA with small dataset
3. **LoRA r=8, 2,000 samples**: Data scaling experiment
4. **LoRA r=32, 1,000 samples**: Higher rank experiment

## Dataset
- Source: Amazon Reviews
- Pre-processed by TA
- Train/validation/test splits

## Results
- eval_loss values for each experiment
- Before vs after comparison
- Performance analysis
