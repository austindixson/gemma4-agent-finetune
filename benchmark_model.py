#!/usr/bin/env python3
"""
Benchmarking script for fine-tuned Gemma model.
Evaluates perplexity, inference speed, and generates samples.
"""

import os
os.environ['HF_HOME'] = '/home/claude/.hf_home'
os.environ['TRANSFORMERS_CACHE'] = '/home/claude/.transformers_cache'
os.environ['HF_DATASETS_CACHE'] = '/home/claude/.datasets_cache'

import torch
import time
import math
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# =============================================================================
# Configuration
# =============================================================================

MODEL_PATH = "./outputs"  # Path to trained LoRA adapters
BASE_MODEL = "0xSero/gemma-4-21b-a4b-it-REAP"
DATASET_PATH = "agent-dataset-unsloth"
MAX_SEQ_LENGTH = 2048
NUM_SAMPLES = 10  # Number of inference samples to generate

# =============================================================================
# Model Loading
# =============================================================================

print("=" * 80)
print("Loading Fine-tuned Model for Benchmarking")
print("=" * 80)

# Load base model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

# Load LoRA adapters
print(f"\nLoading LoRA adapters from {MODEL_PATH}...")
model.load_adapter(MODEL_PATH)
print("✓ Adapters loaded successfully")

# Configure tokenizer
tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
tokenizer.pad_token = tokenizer.eos_token

# =============================================================================
# Benchmark 1: Validation Perplexity
# =============================================================================

print("\n" + "=" * 80)
print("Benchmark 1: Validation Perplexity")
print("=" * 80)

try:
    dataset = load_dataset(
        "json",
        data_files={"valid": f"{DATASET_PATH}/valid.jsonl"}
    )

    print(f"Loaded {len(dataset['valid'])} validation samples")

    # Evaluate perplexity on a subset
    eval_samples = 100
    total_loss = 0.0
    model.eval()

    from unsloth.chat_templates import standardize_sharegpt
    dataset = standardize_sharegpt(dataset)

    with torch.no_grad():
        for i in range(min(eval_samples, len(dataset['valid']))):
            sample = dataset['valid'][i]

            # Format conversation
            text = tokenizer.apply_chat_template(
                sample["conversations"],
                tokenize=False,
                add_generation_prompt=False
            )

            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()

    avg_loss = total_loss / eval_samples
    perplexity = math.exp(avg_loss)

    print(f"\n✓ Validation Results:")
    print(f"  - Average loss: {avg_loss:.4f}")
    print(f"  - Perplexity: {perplexity:.2f}")

except Exception as e:
    print(f"✗ Error computing perplexity: {e}")

# =============================================================================
# Benchmark 2: Inference Speed
# =============================================================================

print("\n" + "=" * 80)
print("Benchmark 2: Inference Speed")
print("=" * 80)

try:
    # Sample conversation
    test_conversation = [
        {"from": "human", "value": "I need to debug this Python script that's failing with a segmentation fault. Can you help me identify the issue?"},
    ]

    # Format input
    prompt = tokenizer.apply_chat_template(
        test_conversation,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Benchmark inference
    print("Running inference benchmark...")
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    torch.cuda.synchronize()
    end_time = time.time()

    generated_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    inference_time = end_time - start_time
    tokens_per_second = generated_tokens / inference_time

    print(f"\n✓ Inference Results:")
    print(f"  - Generated tokens: {generated_tokens}")
    print(f"  - Time: {inference_time:.2f}s")
    print(f"  - Speed: {tokens_per_second:.2f} tokens/second")

    # Decode output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n  Sample response:\n{response[:500]}...")

except Exception as e:
    print(f"✗ Error benchmarking inference: {e}")

# =============================================================================
# Benchmark 3: GPU Memory Usage
# =============================================================================

print("\n" + "=" * 80)
print("Benchmark 3: GPU Memory Usage")
print("=" * 80)

memory_allocated = torch.cuda.memory_allocated() / 1e9
memory_reserved = torch.cuda.memory_reserved() / 1e9
memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9

print(f"✓ Memory Usage:")
print(f"  - Allocated: {memory_allocated:.2f} GB")
print(f"  - Reserved: {memory_reserved:.2f} GB")
print(f"  - Total GPU: {memory_total:.2f} GB")
print(f"  - Free: {memory_total - memory_reserved:.2f} GB")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 80)
print("Benchmark Summary")
print("=" * 80)
print("All benchmarks completed. Results above.")
print("\nNext steps:")
print("  1. Compare perplexity across training checkpoints")
print("  2. Run dead-head analysis using coherence-guided-dead-head-identification/")
print("  3. Generate test set samples for manual evaluation")
