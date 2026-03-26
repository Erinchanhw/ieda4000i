#!/usr/bin/env python3
"""
Problem 2: LoRA Fine-tuning for Amazon Reviews Sentiment Analysis
Author: Chan Hei Wun (Erinchanhw)
SIS ID: [YOUR SIS ID HERE]
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from datasets import Dataset, load_from_disk, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import warnings
warnings.filterwarnings('ignore')

class Config:
    # Paths - Update this to your actual dataset path
    DATA_PATH = os.path.expanduser("~/Assignment2_datasets/ieda4000i_Assignment2_dataset/amazon_reviews_small_4k")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    FIGURES_DIR = os.path.join(BASE_DIR, "figures")
    
    # Model - using smaller model for faster training
    MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Training parameters
    NUM_EPOCHS = 3
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION = 2
    LEARNING_RATE = 2e-4
    SEED = 42
    MAX_LENGTH = 512
    
    # Create directories
    for dir_path in [OUTPUT_DIR, RESULTS_DIR, FIGURES_DIR]:
        os.makedirs(dir_path, exist_ok=True)

def setup_environment():
    """Setup and print environment info"""
    print("="*60)
    print("Problem 2: LoRA Fine-tuning for Sentiment Analysis")
    print("Author: Chan Hei Wun (Erinchanhw)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    return device

def load_dataset():
    """Load the Amazon reviews dataset from HuggingFace format"""
    print("\n=== Loading Dataset ===")
    
    # Check if dataset exists
    if not os.path.exists(Config.DATA_PATH):
        print(f"Error: Dataset not found at {Config.DATA_PATH}")
        sys.exit(1)
    
    # Load the dataset from disk
    try:
        dataset_dict = load_from_disk(Config.DATA_PATH)
        print(f"Dataset loaded successfully!")
        print(f"Available splits: {list(dataset_dict.keys())}")
        
        # Convert to pandas DataFrames for easier manipulation
        train_df = pd.DataFrame(dataset_dict['train'])
        test_df = pd.DataFrame(dataset_dict['test'])
        val_df = pd.DataFrame(dataset_dict['validation']) if 'validation' in dataset_dict else None
        
        print(f"\nTrain samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
        if val_df is not None:
            print(f"Validation samples: {len(val_df)}")
        
        # Check column names
        print(f"\nColumn names: {train_df.columns.tolist()}")
        
        # Print first few rows to understand the structure
        print("\nFirst few rows of training data:")
        print(train_df.head())
        
        # Check rating distribution
        if 'rating' in train_df.columns:
            print(f"\nRating distribution in train:")
            print(train_df['rating'].value_counts().sort_index())
        elif 'label' in train_df.columns:
            print(f"\nLabel distribution in train:")
            print(train_df['label'].value_counts().sort_index())
        
        # Shuffle training data
        train_df = train_df.sample(frac=1, random_state=Config.SEED).reset_index(drop=True)
        
        return train_df, test_df
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTrying alternative loading method...")
        
        # Alternative: load directly from the folder
        from datasets import load_dataset
        dataset = load_dataset('json', data_files={
            'train': f"{Config.DATA_PATH}/train/*.json",
            'test': f"{Config.DATA_PATH}/test/*.json"
        })
        
        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
        
        print(f"Train samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
        
        return train_df, test_df

def format_prompt(review_text):
    """Format the prompt for the model"""
    return f"""Predict the sentiment star rating (1-5) for this Amazon review.

Review: {review_text}

Sentiment rating (1-5):"""

def extract_rating(text):
    """Extract rating from model output"""
    import re
    numbers = re.findall(r'\b[1-5]\b', text)
    if numbers:
        return int(numbers[0])
    return None

def evaluate_model(model, tokenizer, test_df, num_samples=50, model_name="Model"):
    """Evaluate model on test samples"""
    print(f"\n=== Evaluating {model_name} ===")
    
    model.eval()
    correct = 0
    valid_predictions = 0
    results = []
    
    # Determine the rating column name
    rating_col = 'rating' if 'rating' in test_df.columns else 'label'
    
    for idx, row in test_df.head(num_samples).iterrows():
        # Get the review text
        review_text = row.get('review', row.get('text', row.get('content', '')))
        ground_truth = row[rating_col]
        
        prompt = format_prompt(review_text)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=Config.MAX_LENGTH)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted = extract_rating(response)
        
        if predicted is not None:
            valid_predictions += 1
            if predicted == ground_truth:
                correct += 1
        
        results.append({
            'review': review_text[:150],
            'ground_truth': int(ground_truth),
            'predicted': predicted,
            'correct': predicted == ground_truth if predicted else False
        })
    
    accuracy = correct / num_samples if num_samples > 0 else 0
    valid_rate = valid_predictions / num_samples
    
    print(f"Accuracy: {accuracy:.2%} ({correct}/{num_samples})")
    print(f"Valid predictions: {valid_rate:.2%} ({valid_predictions}/{num_samples})")
    
    return results, accuracy

def train_lora_model(base_model, tokenizer, train_data, eval_data, lora_r, sample_size, output_dir):
    """Train a LoRA model"""
    print(f"\n=== Training LoRA (r={lora_r}, samples={sample_size}) ===")
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=2 * lora_r,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none"
    )
    
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION,
        warmup_steps=50,
        learning_rate=Config.LEARNING_RATE,
        fp16=torch.cuda.is_available(),
        logging_steps=20,
        eval_steps=50,
        save_steps=100,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to="none",
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # Train
    train_result = trainer.train()
    
    # Save training logs
    with open(os.path.join(output_dir, "training_logs.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    
    # Extract eval loss
    eval_losses = []
    for log in trainer.state.log_history:
        if 'eval_loss' in log:
            eval_losses.append(log['eval_loss'])
    
    final_eval_loss = eval_losses[-1] if eval_losses else None
    
    metrics = {
        "lora_r": lora_r,
        "sample_size": sample_size,
        "train_loss": train_result.training_loss,
        "final_eval_loss": final_eval_loss
    }
    
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    return lora_model, trainer, metrics

def prepare_training_data(train_df, tokenizer, sample_size):
    """Prepare training dataset"""
    samples = train_df.head(sample_size)
    
    # Determine the rating column
    rating_col = 'rating' if 'rating' in train_df.columns else 'label'
    
    train_data = []
    for _, row in samples.iterrows():
        review_text = row.get('review', row.get('text', row.get('content', '')))
        prompt = format_prompt(review_text)
        response = str(row[rating_col])
        train_data.append({
            "text": f"{prompt}\n{response}"
        })
    
    dataset = Dataset.from_list(train_data)
    
    # Split into train and eval
    split_dataset = dataset.train_test_split(test_size=0.1, seed=Config.SEED)
    
    return split_dataset["train"], split_dataset["test"]

def plot_loss_curves():
    """Plot eval_loss curves for all models"""
    plt.figure(figsize=(12, 8))
    
    models = [
        ("lora_r8_1000", "LoRA r=8, 1k samples", "blue"),
        ("lora_r8_2000", "LoRA r=8, 2k samples", "green"),
        ("lora_r32_1000", "LoRA r=32, 1k samples", "red"),
    ]
    
    for model_dir, label, color in models:
        log_file = os.path.join(Config.OUTPUT_DIR, model_dir, "training_logs.json")
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
            
            steps = []
            losses = []
            for log in logs:
                if 'eval_loss' in log and 'step' in log:
                    steps.append(log['step'])
                    losses.append(log['eval_loss'])
            
            if steps:
                plt.plot(steps, losses, label=label, marker='o', color=color, linewidth=2, markersize=4)
    
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Evaluation Loss', fontsize=12)
    plt.title('LoRA Fine-tuning Loss Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(Config.FIGURES_DIR, "loss_curves.png")
    plt.savefig(save_path, dpi=150)
    print(f"\nLoss curves saved to: {save_path}")

def main():
    """Main execution"""
    device = setup_environment()
    
    # Load data
    train_df, test_df = load_dataset()
    
    # Load model and tokenizer
    print(f"\n=== Loading Model: {Config.MODEL_NAME} ===")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    
    # 2.1 Baseline Evaluation
    baseline_results, baseline_acc = evaluate_model(
        base_model, tokenizer, test_df, num_samples=50, model_name="Baseline"
    )
    
    with open(os.path.join(Config.RESULTS_DIR, "baseline_results.json"), "w") as f:
        json.dump({"accuracy": baseline_acc, "results": baseline_results}, f, indent=2)
    
    # 2.2 LoRA r=8 with 1000 samples
    if len(train_df) >= 1000:
        train_data_1k, eval_data_1k = prepare_training_data(train_df, tokenizer, 1000)
        lora_model_1k, trainer_1k, metrics_1k = train_lora_model(
            base_model, tokenizer, train_data_1k, eval_data_1k,
            lora_r=8, sample_size=1000,
            output_dir=os.path.join(Config.OUTPUT_DIR, "lora_r8_1000")
        )
        
        lora_results_1k, lora_acc_1k = evaluate_model(
            lora_model_1k, tokenizer, test_df, num_samples=50, model_name="LoRA r=8 (1k)"
        )
    else:
        print(f"Not enough training data! Only {len(train_df)} samples available.")
        return
    
    # 2.3 Higher Rank - r=32 with 1000 samples
    train_data_32, eval_data_32 = prepare_training_data(train_df, tokenizer, 1000)
    lora_model_32, trainer_32, metrics_32 = train_lora_model(
        base_model, tokenizer, train_data_32, eval_data_32,
        lora_r=32, sample_size=1000,
        output_dir=os.path.join(Config.OUTPUT_DIR, "lora_r32_1000")
    )
    
    lora_results_32, lora_acc_32 = evaluate_model(
        lora_model_32, tokenizer, test_df, num_samples=50, model_name="LoRA r=32 (1k)"
    )
    
    # Save all results
    all_results = {
        "baseline": {"accuracy": baseline_acc, "results": baseline_results},
        "lora_r8_1000": {"accuracy": lora_acc_1k, "results": lora_results_1k, "metrics": metrics_1k},
        "lora_r32_1000": {"accuracy": lora_acc_32, "results": lora_results_32, "metrics": metrics_32},
    }
    
    with open(os.path.join(Config.RESULTS_DIR, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Plot loss curves
    plot_loss_curves()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Baseline Accuracy: {baseline_acc:.2%}")
    print(f"LoRA r=8 (1k) Accuracy: {lora_acc_1k:.2%}")
    print(f"LoRA r=32 (1k) Accuracy: {lora_acc_32:.2%}")
    print("\nResults saved to:", Config.RESULTS_DIR)
    print("Figures saved to:", Config.FIGURES_DIR)

if __name__ == "__main__":
    main()
