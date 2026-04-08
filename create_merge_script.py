#!/usr/bin/env python3
"""
Merge LoRA adapters into base model and save for deployment.
Supports both full merge and 16-bit merged output.
"""

import os
os.environ['HF_HOME'] = '/home/claude/.hf_home'
os.environ['TRANSFORMERS_CACHE'] = '/home/claude/.transformers_cache'
os.environ['HF_DATASETS_CACHE'] = '/home/claude/.datasets_cache'

import torch
from unsloth import FastLanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# Configuration
# =============================================================================

BASE_MODEL = "0xSero/gemma-4-21b-a4b-it-REAP"
LORA_ADAPTERS_PATH = "./outputs"  # Change to your checkpoint path
OUTPUT_DIR = "./merged_model"

# Options: "merged_16bit", "merged_4bit", "lora_only"
SAVE_METHOD = "merged_16bit"

print("=" * 80)
print("Merging LoRA Adapters")
print("=" * 80)
print(f"Base model: {BASE_MODEL}")
print(f"LoRA adapters: {LORA_ADAPTERS_PATH}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Save method: {SAVE_METHOD}")

# =============================================================================
# Load Model with LoRA
# =============================================================================

print("\nLoading model with LoRA adapters...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Load LoRA adapters
model.load_adapter(LORA_ADAPTERS_PATH)
print("✓ LoRA adapters loaded")

# =============================================================================
# Merge and Save
# =============================================================================

print(f"\nMerging and saving with method: {SAVE_METHOD}")

model.save_pretrained_merged(
    OUTPUT_DIR,
    tokenizer,
    save_method=SAVE_METHOD,
)

print(f"\n✓ Model saved to {OUTPUT_DIR}")

# =============================================================================
# Verification
# =============================================================================

print("\n" + "=" * 80)
print("Verification")
print("=" * 80)

# Check output directory
import os
if os.path.exists(OUTPUT_DIR):
    files = os.listdir(OUTPUT_DIR)
    print(f"Files in {OUTPUT_DIR}:")
    for f in files:
        file_path = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        print(f"  - {f}: {size:.2f} MB")
else:
    print(f"✗ Output directory not found: {OUTPUT_DIR}")

print("\n✓ Merge complete!")
print("\nNext steps:")
print("  1. Test the merged model with inference")
print("  2. Upload to HuggingFace if desired")
print("  3. Deploy to your application")
