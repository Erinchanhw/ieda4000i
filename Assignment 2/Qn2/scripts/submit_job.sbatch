#!/bin/bash
#SBATCH --job-name=lora_exp
#SBATCH --output=../../Assignment_2/logs/job_%j.out
#SBATCH --error=../../Assignment_2/logs/job_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-rtx40+
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --account=ugiedahpc4

# Load modules
module load python/3.9 cuda/11.8

# Install required packages
pip install --user transformers datasets peft accelerate torch pandas matplotlib scikit-learn

# Create necessary directories
mkdir -p ~/ieda4000i/Assignment_2/logs
mkdir -p ~/ieda4000i/Assignment_2/outputs
mkdir -p ~/ieda4000i/Assignment_2/results
mkdir -p ~/ieda4000i/Assignment_2/figures

# Run experiments
cd ~/ieda4000i
python "Assignment 2/scripts/run_experiments.py"

echo "Job completed at $(date)"

