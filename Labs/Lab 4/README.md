touch Labs/Lab\ 4/data/.gitkeep
touch Labs/Lab\ 4/scripts/.gitkeep
ls -la Labs/Lab\ 4/
# Create the folder structure
mkdir -p Labs/Lab\ 4/data
mkdir -p Labs/Lab\ 4/scripts

# Create a README file
cat > Labs/Lab\ 4/README.md << 'EOF'
# Lab 4: LoRA Fine-Tuning

This lab covers LoRA fine-tuning on HPC4.

## Contents
- `data/` - Training and evaluation data
- `scripts/` - Python scripts for training and inference
- `run_eval.py` - Evaluation script
- `train_lora.py` - LoRA training script
- `train_lora.sbatch` - SLURM job submission script
