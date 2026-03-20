# Lab 4 Scripts (LoRA Finetuning)

## Files

- `train_lora.py`: trains a LoRA adapter from JSONL instruction data.
- `train_lora.sbatch`: submits one SLURM job with student-isolated output/cache/log paths.
- `run_eval.py`: runs prompt-file inference for before/after comparison.

## Expected data format

`train_lora.py` supports either of these per-line JSON formats:

1. Prompt-response:

```json
{"prompt": "...", "response": "..."}
```

2. Instruction format:

```json
{"instruction": "...", "input": "...", "output": "..."}
```

`input` is optional.

## Quick start

From your course repo's `Labs/Lab 4` directory:

```bash
export COURSE_ENV=ieda4000i
export BASE_MODEL="/project/ugiedahpc4/ieda4000i/models/Qwen3-1.7B"
export RUN_TAG="${USER}_$(date +%Y%m%d_%H%M%S)"
export LAB4_RUN_ROOT="$HOME/lab4_runs/${RUN_TAG}"
export OUTPUT_DIR="${LAB4_RUN_ROOT}/adapter"
export LOG_DIR="${LAB4_RUN_ROOT}/logs"
export HF_HOME="${LAB4_RUN_ROOT}/hf_cache"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${HF_HOME}"

python scripts/run_eval.py \
  --model_path "${BASE_MODEL}" \
  --prompt_file data/eval_prompts.jsonl \
  --output_file "${LAB4_RUN_ROOT}/before.jsonl" \
  --device cuda

sbatch scripts/train_lora.sbatch

python scripts/run_eval.py \
  --model_path "${BASE_MODEL}" \
  --adapter_path "${OUTPUT_DIR}" \
  --prompt_file data/eval_prompts.jsonl \
  --output_file "${LAB4_RUN_ROOT}/after.jsonl" \
  --device cuda
```
