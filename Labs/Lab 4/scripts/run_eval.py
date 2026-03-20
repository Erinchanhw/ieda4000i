#!/usr/bin/env python3
"""Run batch prompt evaluation for base model or base+LoRA adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run before/after inference for Lab 4")
    parser.add_argument("--model_path", required=True, help="Base model path or HF model id")
    parser.add_argument("--adapter_path", default="", help="Optional LoRA adapter path")
    parser.add_argument("--prompt_file", required=True, help="JSONL prompt file")
    parser.add_argument("--output_file", required=True, help="JSONL output file")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    cuda_available = torch.cuda.is_available()

    if device_arg == "cuda" and not cuda_available:
        print("ERROR: CUDA is not available in this session.")
        print("Request GPU with srun/sbatch first.")
        sys.exit(2)

    if device_arg == "cpu":
        return "cpu"
    if device_arg == "auto":
        return "cuda" if cuda_available else "cpu"
    return "cuda"


def load_prompts(path: str) -> List[Dict]:
    prompts: List[Dict] = []
    with open(path, "r", encoding="utf-8") as file:
        for line_no, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc

            if "prompt" not in item:
                raise ValueError(f"Missing 'prompt' field at {path}:{line_no}")
            prompts.append(item)

    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("=== Lab 4 Evaluation Runner ===")
    print(f"model_path: {args.model_path}")
    print(f"adapter_path: {args.adapter_path or '[none]'}")
    print(f"prompt_file: {args.prompt_file}")
    print(f"output_file: {args.output_file}")
    print(f"device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )

    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model = model.to(device)
    model.eval()

    prompts = load_prompts(args.prompt_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out_file:
        for idx, item in enumerate(prompts, start=1):
            prompt = str(item["prompt"])
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[-1]

            generation_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 0.0,
                "temperature": max(args.temperature, 1e-5),
                "top_p": args.top_p,
            }
            if tokenizer.eos_token_id is not None:
                generation_kwargs["pad_token_id"] = tokenizer.eos_token_id

            with torch.no_grad():
                output_ids = model.generate(**inputs, **generation_kwargs)

            generated_ids = output_ids[0][input_len:]
            prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            record = {
                "id": item.get("id", idx),
                "prompt": prompt,
                "prediction": prediction,
            }
            if "reference" in item:
                record["reference"] = item["reference"]

            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[INFO] Wrote {len(prompts)} predictions to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
