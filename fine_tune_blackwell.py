#!/usr/bin/env python3
"""
Fine-tuning script for Gemma-4-21B-A4B-IT-REAP with 4-bit quantization using Unsloth.
"""

import os

# Set cache directories before importing torch/transformers
os.environ['HF_HOME'] = '/home/claude/.hf_home'
os.environ['TRANSFORMERS_CACHE'] = '/home/claude/.transformers_cache'
os.environ['HF_DATASETS_CACHE'] = '/home/claude/.datasets_cache'

import torch
from datasets import load_dataset
from pathlib import Path
from transformers import (
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import shutil

# =============================================================================
# Configuration
# =============================================================================

# Model Configuration
MODEL_NAME = "0xSero/gemma-4-21b-a4b-it-REAP"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

# LoRA Configuration
# Unsloth recommends alpha >= r, with 2x being better for 26B-A4B
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Training Configuration
BATCH_SIZE_PER_DEVICE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 0.0002
MAX_GRAD_NORM = 1.0  # Gradient clipping per Unsloth best practices
WARMUP_STEPS = 50  # Will be overridden by warmup_ratio in training args
MAX_STEPS = 500
OPTIMIZER = "adamw_8bit"
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "cosine"

# Output Configuration
OUTPUT_DIR = "./outputs"
CHECKPOINT_DIR = "./checkpoints"
LOGGING_STEPS = 5
SAVE_STEPS = 25

# =============================================================================
# Model Loading
# =============================================================================

print("=" * 80)
print("Loading Model with 4-bit Quantization")
print("=" * 80)

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # Auto-detect dtype
        load_in_4bit=LOAD_IN_4BIT,
        trust_remote_code=True,
    )
    print(f"✓ Model loaded successfully: {MODEL_NAME}")
    print(f"✓ Max sequence length: {MAX_SEQ_LENGTH}")
    print(f"✓ 4-bit quantization: {LOAD_IN_4BIT}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    raise

# =============================================================================
# LoRA Configuration
# =============================================================================

print("\n" + "=" * 80)
print("Configuring LoRA Adapters")
print("=" * 80)

try:
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    print(f"✓ LoRA adapters configured")
    print(f"  - Rank (r): {LORA_R}")
    print(f"  - Alpha: {LORA_ALPHA}")
    print(f"  - Dropout: {LORA_DROPOUT}")
    print(f"  - Target modules: {', '.join(LORA_TARGET_MODULES)}")
except Exception as e:
    print(f"✗ Error configuring LoRA: {e}")
    raise

# =============================================================================
# Tokenizer Configuration
# =============================================================================

print("\n" + "=" * 80)
print("Configuring Tokenizer")
print("=" * 80)

try:
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-3",
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("✓ Tokenizer configured with Gemma-3 chat template")
except Exception as e:
    print(f"✗ Error configuring tokenizer: {e}")
    raise

# =============================================================================
# Dataset Loading
# =============================================================================

print("\n" + "=" * 80)
print("Loading Dataset")
print("=" * 80)

# Dataset paths
DATASET_DIR = "agent-dataset-unsloth"

try:
    # Load ShareGPT format dataset
    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{DATASET_DIR}/train.jsonl",
            "valid": f"{DATASET_DIR}/valid.jsonl",
            "test": f"{DATASET_DIR}/test.jsonl",
        }
    )

    print(f"✓ Dataset loaded from {DATASET_DIR}/")
    print(f"  - Train: {len(dataset['train'])} samples")
    print(f"  - Valid: {len(dataset['valid'])} samples")
    print(f"  - Test: {len(dataset['test'])} samples")

    # Map roles and format to text in one pass
    print("\n  Processing and formatting conversations...")
    def process_and_format(example):
        """Map roles and apply chat template in one pass"""
        conversations = example.get("conversations", [])
        # Map to user/assistant format
        mapped = []
        for turn in conversations:
            role = turn.get("from", "")
            content = turn.get("value", "")
            if role == "human":
                mapped.append({"role": "user", "content": content})
            elif role == "gpt":
                mapped.append({"role": "assistant", "content": content})
            else:
                mapped.append({"role": role, "content": content})

        # Apply chat template to get formatted text
        try:
            # Get the actual tokenizer from the processor if needed
            actual_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
            text = actual_tokenizer.apply_chat_template(mapped, tokenize=False, add_generation_prompt=False)
        except Exception as e:
            print(f"  Warning: Template error on sample: {e}")
            text = ""

        return {"text": text}

    dataset = dataset.map(
        process_and_format,
        remove_columns=["conversations"],
        desc="Processing and formatting"
    )
    print("  ✓ Conversations processed and formatted")

    # Tokenize the dataset
    print("\n  Tokenizing dataset...")
    def tokenize_function(example):
        # Get the actual tokenizer from the processor if needed
        actual_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer

        # Tokenize the text
        result = actual_tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
            return_tensors=None,
        )
        return result

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing dataset",
    )
    print("  ✓ Dataset tokenized")

    # Update dataset reference
    dataset = tokenized_dataset

    print(f"✓ Dataset processed successfully")

except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    raise

# =============================================================================
# Training Setup
# =============================================================================

print("\n" + "=" * 80)
print("Setting Up Training")
print("=" * 80)

# Training Arguments
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    per_device_train_batch_size=BATCH_SIZE_PER_DEVICE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_ratio=0.03,  # Unsloth recommends 3% warmup for Gemma-4
    max_grad_norm=MAX_GRAD_NORM,  # Gradient clipping for stability
    max_steps=MAX_STEPS,
    num_train_epochs=1,  # Set to 1 for max_steps-based training
    optim=OPTIMIZER,
    weight_decay=WEIGHT_DECAY,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=1,  # Keep only most recent checkpoint to save disk space
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    gradient_checkpointing=True,
    eval_strategy="no",
    save_strategy="steps",
    load_best_model_at_end=False,
    report_to=["tensorboard"],
    run_name=f"gemma-4-21b-reap-finutune",
    dataloader_num_workers=4,
    remove_unused_columns=False,
)

print(f"✓ Training arguments configured:")
print(f"  - Batch size per device: {BATCH_SIZE_PER_DEVICE}")
print(f"  - Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
print(f"  - Effective batch size: {BATCH_SIZE_PER_DEVICE * GRADIENT_ACCUMULATION_STEPS}")
print(f"  - Learning rate: {LEARNING_RATE}")
print(f"  - Warmup steps: {WARMUP_STEPS}")
print(f"  - Max steps: {MAX_STEPS}")
print(f"  - Optimizer: {OPTIMIZER}")
print(f"  - Scheduler: {LR_SCHEDULER_TYPE}")
print(f"  - Precision: {'bf16' if torch.cuda.is_bf16_supported() else 'fp16'}")

# =============================================================================
# Checkpoint Cleanup Callback
# =============================================================================

class CheckpointCleanupCallback(TrainerCallback):
    """Delete old checkpoints when new ones are saved - keeps only most recent"""

    def on_save(self, args, state, control, **kwargs):
        """Called after checkpoint is saved"""
        checkpoint_dir = Path(args.output_dir)
        if not checkpoint_dir.exists():
            return control

        checkpoints = sorted(checkpoint_dir.glob("checkpoint-*"),
                           key=lambda x: x.stat().st_mtime, reverse=True)

        # Keep only the most recent checkpoint
        for old_checkpoint in checkpoints[1:]:
            try:
                shutil.rmtree(old_checkpoint)
                print(f"🧹 Deleted old checkpoint: {old_checkpoint.name}")
            except Exception as e:
                print(f"⚠ Could not delete {old_checkpoint}: {e}")

        return control

# =============================================================================
# Trainer Setup
# =============================================================================

print("\n" + "=" * 80)
print("Initializing Trainer")
print("=" * 80)

try:
    from transformers import Trainer, DataCollatorForLanguageModeling

    # Get the actual tokenizer from the processor if needed
    actual_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer

    # Add checkpoint cleanup callback to save disk space
    checkpoint_cleanup = CheckpointCleanupCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=actual_tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        ),
        callbacks=[checkpoint_cleanup],
    )
    print("✓ Trainer initialized successfully")
except Exception as e:
    print(f"✗ Error initializing trainer: {e}")
    raise

# =============================================================================
# Training
# =============================================================================

print("\n" + "=" * 80)
print("Starting Training")
print("=" * 80)

try:
    # Print model info before training
    print(f"\nModel parameters: {model.num_parameters():,}")
    trainable = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable[0]:,}")

    # Start training
    trainer.train()

    print("\n✓ Training completed successfully!")

except Exception as e:
    print(f"\n✗ Error during training: {e}")
    raise

# =============================================================================
# Save Final Model
# =============================================================================

print("\n" + "=" * 80)
print("Saving Model")
print("=" * 80)

try:
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save LoRA adapters
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"✓ Model saved to {OUTPUT_DIR}")
    print(f"✓ Tokenizer saved to {OUTPUT_DIR}")

    # Optional: Merge and save as full model (commented out by default)
    # print("\nMerging LoRA adapters...")
    # merged_model = model.merge_and_unload()
    # merged_model.save_pretrained(f"{OUTPUT_DIR}_merged")
    # tokenizer.save_pretrained(f"{OUTPUT_DIR}_merged")
    # print(f"✓ Merged model saved to {OUTPUT_DIR}_merged")

except Exception as e:
    print(f"✗ Error saving model: {e}")
    raise

# =============================================================================
# Evaluation Metrics
# =============================================================================

print("\n" + "=" * 80)
print("Training Metrics")
print("=" * 80)

try:
    # Get training metrics
    if trainer.state.log_history:
        final_loss = trainer.state.log_history[-1].get("train_loss", "N/A")
        print(f"Final training loss: {final_loss}")

        # Print all available metrics
        print("\nTraining history:")
        for entry in trainer.state.log_history[-5:]:  # Last 5 entries
            print(f"  Step {entry.get('step', 'N/A')}: {entry}")

except Exception as e:
    print(f"⚠ Warning: Could not retrieve metrics: {e}")

# =============================================================================
# Inference Example
# =============================================================================

print("\n" + "=" * 80)
print("Inference Example")
print("=" * 80)

try:
    # Load the fine-tuned model for inference
    print("\nLoading fine-tuned model for inference...")
    model_inference, tokenizer_inference = FastLanguageModel.from_pretrained(
        model_name=OUTPUT_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    # Enable native 2x faster inference
    FastLanguageModel.for_inference(model_inference)

    # Example inference
    messages = [
        {"role": "user", "content": "What is the capital of France?"}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    outputs = model_inference.generate(
        **inputs,
        max_new_tokens=64,
        use_cache=True,
        temperature=0.7,
        top_p=0.9,
    )

    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print("\nSample inference response:")
    print(response)

except Exception as e:
    print(f"⚠ Warning: Inference example failed: {e}")

print("\n" + "=" * 80)
print("Script Completed Successfully!")
print("=" * 80)
print(f"\nModel saved at: {OUTPUT_DIR}")
print(f"Checkpoints saved at: {CHECKPOINT_DIR}")