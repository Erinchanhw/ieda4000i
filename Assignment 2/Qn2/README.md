# Assignment 2: LoRA Fine-tuning for Sentiment Analysis

## Student Information
- **Name**: Chan Hei Wun
- **SIS ID**: 20950005

## Problem 2: Amazon Reviews Sentiment Analysis

This directory contains the complete implementation for fine-tuning Qwen models on Amazon reviews dataset using LoRA.

## Experiments Conducted

### 1. Baseline Evaluation (Section 2.1)
- Evaluated Qwen3-1.7B directly on test set
- No training, just inference

### 2. LoRA Fine-tuning (r=8, 1k samples) (Section 2.2)
- Trained with LoRA rank=8
- Used 1,000 random samples from training set

### 3. Data Scaling (Section 2.3)
- Increased training samples to 2,000
- Same LoRA rank=8

### 4. Rank Analysis (Section 2.3)
- Used LoRA rank=32 with 1,000 samples
- Compare with rank=8 results

## How to Run

### Run the complete experiment:
```bash
cd "Assignment 2/scripts"
python problem2_lora_v2.py
```

### Submit to HPC4 using SLURM:
```bash
cd "Assignment 2"
sbatch submit_job.slurm
```
