#!/usr/bin/env python3
"""
Resume training from iteration 4
"""
import os
import json
import subprocess
import re
from pathlib import Path
from datetime import datetime

ITERATIONS = 10
START_ITERATION = 4  # Resume from here
SUMMARY_DIR = "./iterations_summary"

# Read iteration 3 config
with open(f"{SUMMARY_DIR}/iter_3.json") as f:
    iter_3_data = json.load(f)

# Start with iter 3's config, augmented to iter 4 dataset
current_config = {
    "learning_rate": 2e-4,
    "max_steps": 500,
    "batch_size_per_device": 2,
    "lora_r": 16,
    "lora_alpha": 32,  # Updated per Unsloth best practices
    "dataset_path": "./agent-dataset-unsloth/train_iter_3.jsonl"
}

def run_training(config, iteration):
    print(f"\n{'='*80}")
    print(f"Iteration {iteration}/{ITERATIONS}")
    print(f"{'='*80}")
    
    with open("fine_tune_blackwell.py", "r") as f:
        content = f.read()
    
    content = re.sub(r"LEARNING_RATE = .*", f"LEARNING_RATE = {config['learning_rate']}", content)
    content = re.sub(r"MAX_STEPS = .*", f"MAX_STEPS = {config['max_steps']}", content)
    content = re.sub(r"LORA_R = .*", f"LORA_R = {config['lora_r']}", content)
    content = re.sub(r"LORA_ALPHA = .*", f"LORA_ALPHA = {config['lora_alpha']}", content)
    
    if "dataset_path" in config:
        content = re.sub(r'"train": "agent-dataset-unsloth/train\.jsonl"',
                       f'"train": "{config["dataset_path"]}"', content)
    
    with open("fine_tune_blackwell.py", "w") as f:
        f.write(content)
    
    # Cleanup before training
    subprocess.run(["bash", "cleanup.sh"], capture_output=True)
    
    result = subprocess.run(["bash", "run_training.sh"], capture_output=True, text=True, timeout=7200)
    
    if result.returncode != 0:
        print(f"✗ Training failed")
        return None
    
    return "./outputs"

def extract_loss(model_path):
    log_files = sorted([(f.stat().st_mtime, f) for f in Path(".").glob("training_*.log")])
    if not log_files:
        return None
    
    import math
    latest_log = log_files[-1][1]
    losses = []
    
    with open(latest_log, 'r') as f:
        for line in f:
            match = re.search(r"'loss':\s*'([\d.]+)'", line)
            if match:
                losses.append(float(match.group(1)))
    
    if losses:
        return {
            "loss": losses[-1],
            "avg_loss": sum(losses[-10:]) / min(10, len(losses)),
            "perplexity": math.exp(losses[-1]),
        }
    return None

print(f"\n{'='*80}")
print(f"RESUMING FROM ITERATION {START_ITERATION}")
print(f"{'='*80}")
print(f"Previous best: Iteration 3, loss: {iter_3_data['train_metrics']['loss']}")

for iteration in range(START_ITERATION, ITERATIONS + 1):
    model_path = run_training(current_config, iteration)
    if not model_path:
        break
    
    metrics = extract_loss(model_path)
    
    # Save summary
    summary = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "config": current_config,
        "metrics": metrics,
    }
    
    with open(f"{SUMMARY_DIR}/iter_{iteration}.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Iteration {iteration} complete: loss {metrics['loss']:.4f}")
    
    # Update for next iteration
    if iteration < ITERATIONS:
        current_config["dataset_path"] = f"./agent-dataset-unsloth/train_iter_{iteration}.jsonl"

print(f"\n{'='*80}")
print("RESUME TRAINING COMPLETE")
print(f"{'='*80}")
