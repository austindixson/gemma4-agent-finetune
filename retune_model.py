#!/usr/bin/env python3
"""
Light fine-tuning script for post-pruning recovery.
Uses fewer steps and conservative hyperparameters.
"""

import os
os.environ['HF_HOME'] = '/home/claude/.hf_home'
os.environ['TRANSFORMERS_CACHE'] = '/home/claude/.transformers_cache'
os.environ['HF_DATASETS_CACHE'] = '/home/claude/.datasets_cache'

import torch
from datasets import load_dataset
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt

# =============================================================================
# Configuration
# =============================================================================

# Pruned model path (if retraining pruned model)
MODEL_PATH = "./outputs"  # or "./outputs_pruned" for pruned model
BASE_MODEL = "0xSero/gemma-4-21b-a4b-it-REAP"

# Conservative training for recovery
MAX_STEPS = 150  # Light fine-tuning
BATCH_SIZE_PER_DEVICE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4  # Lower than initial training
WARMUP_STEPS = 20

OUTPUT_DIR = "./outputs_recovered"
CHECKPOINT_DIR = "./checkpoints_recovered"

print("=" * 80)
print("Light Fine-Tuning for Post-Pruning Recovery")
print("=" * 80)
print(f"Model: {MODEL_PATH}")
print(f"Steps: {MAX_STEPS} (light fine-tuning)")
print(f"Learning rate: {LEARNING_RATE}")

# =============================================================================
# Model Loading
# =============================================================================

print("\nLoading model...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Load adapters (either original or pruned)
model.load_adapter(MODEL_PATH)
print(f"✓ Loaded adapters from {MODEL_PATH}")

# Configure tokenizer
tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# =============================================================================
# Dataset Loading
# =============================================================================

print("\nLoading dataset...")

dataset = load_dataset(
    "json",
    data_files={
        "train": "agent-dataset-unsloth/train.jsonl",
        "valid": "agent-dataset-unsloth/valid.jsonl",
    }
)

dataset = standardize_sharegpt(dataset)
dataset = dataset.map(lambda x: {
    "text": tokenizer.apply_chat_template(x["conversations"], tokenize=False)
})

print(f"✓ Dataset loaded: {len(dataset['train'])} train, {len(dataset['valid'])} valid")

# =============================================================================
# Training Setup
# =============================================================================

print("\nSetting up training...")

training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    per_device_train_batch_size=BATCH_SIZE_PER_DEVICE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    max_steps=MAX_STEPS,
    num_train_epochs=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    logging_steps=5,
    save_steps=50,
    save_total_limit=2,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    gradient_checkpointing=True,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    load_best_model_at_end=True,
    report_to=["tensorboard"],
    run_name="gemma-recovery-finutune",
    dataloader_num_workers=4,
    remove_unused_columns=False,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["valid"],
    dataset_text_field="text",
    args=training_args,
    data_collator=data_collator,
    max_seq_length=2048,
)

print("✓ Trainer configured")

# =============================================================================
# Training
# =============================================================================

print("\n" + "=" * 80)
print("Starting Light Fine-Tuning")
print("=" * 80)

trainer.train()

# =============================================================================
# Save
# =============================================================================

print("\nSaving recovered model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✓ Recovered model saved to {OUTPUT_DIR}")
print("\nRecovery complete!")
