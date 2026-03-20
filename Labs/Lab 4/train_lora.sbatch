#!/bin/bash
#SBATCH --job-name=lab4_lora
#SBATCH --account=ugiedahpc4
#SBATCH --partition=gpu-a30
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=lab4_lora_%j.out
#SBATCH --error=lab4_lora_%j.err

set -euo pipefail

# Use SLURM_SUBMIT_DIR (the directory where sbatch was called) to locate scripts and data.
# BASH_SOURCE does not work reliably in sbatch because SLURM copies the script to a spool directory.
LAB4_DIR="${SLURM_SUBMIT_DIR}"
SCRIPT_DIR="${LAB4_DIR}/scripts"

COURSE_ENV="${COURSE_ENV:-ieda4000i}"
BASE_MODEL="${BASE_MODEL:-/project/ugiedahpc4/ieda4000i/models/Qwen3-1.7B}"
TRAIN_FILE="${TRAIN_FILE:-${LAB4_DIR}/data/train_lora.jsonl}"
VAL_FILE="${VAL_FILE:-${LAB4_DIR}/data/val_lora.jsonl}"

RUN_TAG="${RUN_TAG:-${USER}_${SLURM_JOB_ID}}"
LAB4_RUN_ROOT="${LAB4_RUN_ROOT:-$HOME/lab4_runs/${RUN_TAG}}"
OUTPUT_DIR="${OUTPUT_DIR:-${LAB4_RUN_ROOT}/adapter}"
LOG_DIR="${LOG_DIR:-${LAB4_RUN_ROOT}/logs}"
HF_HOME="${HF_HOME:-${LAB4_RUN_ROOT}/hf_cache}"

MAX_LENGTH="${MAX_LENGTH:-512}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-5}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
TARGET_MODULES="${TARGET_MODULES:-q_proj,v_proj}"
USE_4BIT="${USE_4BIT:-0}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${HF_HOME}"

source "/opt/shared/.spack-edge/dist/bin/setup-env.sh" -y

echo "[INFO] Job id: ${SLURM_JOB_ID:-unknown}"
echo "[INFO] Hostname: $(hostname)"
echo "[INFO] Environment: ${COURSE_ENV}"
echo "[INFO] Base model: ${BASE_MODEL}"
echo "[INFO] Train file: ${TRAIN_FILE}"
echo "[INFO] Val file: ${VAL_FILE}"
echo "[INFO] Output dir: ${OUTPUT_DIR}"
echo "[INFO] Log dir: ${LOG_DIR}"
echo "[INFO] HF_HOME: ${HF_HOME}"

nvidia-smi || true

PYTHON_BIN="${HOME}/.conda/envs/${COURSE_ENV}/bin/python"

CMD=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/train_lora.py"
  --base_model "${BASE_MODEL}"
  --train_file "${TRAIN_FILE}"
  --val_file "${VAL_FILE}"
  --output_dir "${OUTPUT_DIR}"
  --logging_dir "${LOG_DIR}"
  --max_length "${MAX_LENGTH}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --lora_r "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --lora_dropout "${LORA_DROPOUT}"
  --target_modules "${TARGET_MODULES}"
)

if [[ "${USE_4BIT}" == "1" ]]; then
  CMD+=(--use_4bit)
fi

CMD+=(--bf16)

"${CMD[@]}"

echo "[INFO] Training completed successfully."
