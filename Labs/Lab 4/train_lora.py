#!/usr/bin/env python3
"""Train a LoRA adapter on instruction data for Lab 4."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoRA adapter on HPC4")
    parser.add_argument("--base_model", required=True, help="Shared model path or HF model id")
    parser.add_argument("--train_file", required=True, help="Path to train jsonl")
    parser.add_argument("--val_file", default="", help="Path to val jsonl (optional)")
    parser.add_argument("--output_dir", required=True, help="Output dir for adapter")
    parser.add_argument("--logging_dir", required=True, help="Logging dir")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", default="q_proj,v_proj", help="Comma-separated target module names")
    parser.add_argument("--use_4bit", action="store_true", help="Enable 4-bit QLoRA loading")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 training when supported")
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as file:
        for line_no, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def to_text(example: Dict) -> str:
    if "prompt" in example and "response" in example:
        prompt = str(example["prompt"]).strip()
        response = str(example["response"]).strip()
        return f"### Instruction:\n{prompt}\n\n### Response:\n{response}"

    if "instruction" in example and "output" in example:
        instruction = str(example["instruction"]).strip()
        input_text = str(example.get("input", "")).strip()
        output = str(example["output"]).strip()
        if input_text:
            return (
                "### Instruction:\n"
                f"{instruction}\n\n"
                "### Input:\n"
                f"{input_text}\n\n"
                "### Response:\n"
                f"{output}"
            )
        return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    raise ValueError(
        "Each record must contain either {prompt,response} or {instruction,output[,input]}"
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    logging_dir = Path(args.logging_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)

    print("=== Lab 4 LoRA Training Runner ===")
    print(f"base_model: {args.base_model}")
    print(f"train_file: {args.train_file}")
    print(f"val_file: {args.val_file or '[none]'}")
    print(f"output_dir: {output_dir}")
    print(f"logging_dir: {logging_dir}")
    print(f"use_4bit: {args.use_4bit}")

    train_records = load_jsonl(args.train_file)
    val_records = load_jsonl(args.val_file) if args.val_file else []

    train_texts = [to_text(item) for item in train_records]
    val_texts = [to_text(item) for item in val_records]

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )
        return tokenized

    train_ds = Dataset.from_dict({"text": train_texts}).map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
    )
    eval_ds = None
    if val_texts:
        eval_ds = Dataset.from_dict({"text": val_texts}).map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"],
        )

    quantization_config = None
    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
    }

    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["dtype"] = torch.bfloat16 if args.bf16 else torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False

    target_modules = [name.strip() for name in args.target_modules.split(",") if name.strip()]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        logging_dir=str(logging_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        fp16=not args.bf16,
        bf16=args.bf16,
        report_to=[],
        remove_unused_columns=False,
        save_total_limit=args.save_total_limit,
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = train_result.metrics
    metrics_path = output_dir / "train_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    run_meta = {
        "base_model": args.base_model,
        "train_file": args.train_file,
        "val_file": args.val_file,
        "output_dir": str(output_dir),
        "logging_dir": str(logging_dir),
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "use_4bit": args.use_4bit,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": target_modules,
        },
        "training": {
            "epochs": args.num_train_epochs,
            "batch_size": args.per_device_train_batch_size,
            "grad_accum": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "seed": args.seed,
        },
    }
    with (output_dir / "run_meta.json").open("w", encoding="utf-8") as file:
        json.dump(run_meta, file, indent=2)

    print("[INFO] Training complete.")
    print(f"[INFO] Adapter saved to: {output_dir}")
    print(f"[INFO] Metrics saved to: {metrics_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise
