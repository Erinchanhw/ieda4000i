#!/bin/bash
#SBATCH --job-name=orlm_inference
#SBATCH --output=inference_%j.out
#SBATCH --error=inference_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=erinchanhw
# Load modules (adjust based on HPC4 setup)
module load python/3.9
module load cuda/11.7

# Activate virtual environment if you have one
# source ~/venv/bin/activate

# Install required packages if needed
# pip install vllm tqdm

# Set variables
MODEL_PATH="/path/to/your/model"  # Update this path
DATASET_PATH="/path/to/orlm/dataset"  # Update this path
OUTPUT_DIR="~/ieda4000i/Assignment\ 2/Qn3/inference_output"

# Run inference
python inference_problem3.py \
    --model_name_or_path $MODEL_PATH \
    --dataset_path $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --num_samples 10 \
    --tensor_parallel_size 1 \
    --decoding_method greedy \
    --seed 42

echo "Inference completed!"
