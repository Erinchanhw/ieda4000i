
#!/usr/bin/env python3
"""Test script to verify setup"""

import os
import sys

print("="*60)
print("Testing Assignment 2 Setup")
print("="*60)

# Check Python version
print(f"\nPython version: {sys.version}")

# Check if required packages are installed
packages = ['torch', 'transformers', 'datasets', 'peft', 'pandas', 'numpy']
missing = []

for pkg in packages:
    try:
        __import__(pkg)
        print(f"✅ {pkg} is installed")
    except ImportError:
        print(f"❌ {pkg} is NOT installed")
        missing.append(pkg)

if missing:
    print(f"\nPlease install missing packages: pip install {' '.join(missing)}")

# Check dataset
dataset_path = os.path.expanduser("~/Assignment2_datasets/ieda4000i_Assignment2_dataset/amazon_reviews_small_4k")
print(f"\nChecking dataset at: {dataset_path}")

if os.path.exists(dataset_path):
    print("✅ Dataset folder exists")
    
    try:
        from datasets import load_from_disk
        dataset = load_from_disk(dataset_path)
        print(f"✅ Dataset loaded successfully!")
        print(f"   Splits: {list(dataset.keys())}")
        print(f"   Train samples: {len(dataset['train'])}")
        print(f"   Test samples: {len(dataset['test'])}")
        print(f"   Columns: {dataset['train'].column_names}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
else:
    print("❌ Dataset folder not found!")

print("\n" + "="*60)
print("Setup check complete!")
print("="*60)
