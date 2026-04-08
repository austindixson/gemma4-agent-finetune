#!/usr/bin/env python3
"""
Fixed 10x Training Iteration Workflow

This script:
1. Runs training (500 steps)
2. Extracts training loss from logs (NOT broken benchmark)
3. Autoresearch (adjust hyperparameters based on training loss)
4. Repeats 10x
5. Selects best model
6. Proceeds to pruning
"""

import os
import json
import subprocess
import re
from pathlib import Path
from datetime import datetime

# Configuration
ITERATIONS = 10
SUMMARY_DIR = "./iterations_summary"

# Initial hyperparameters
initial_config = {
    "learning_rate": 2e-4,
    "max_steps": 500,
    "batch_size_per_device": 2,
    "lora_r": 16,
    "lora_alpha": 16,
}

def run_training(config, iteration):
    """Run training with current config"""
    print(f"\n{'='*80}")
    print(f"Iteration {iteration}/{ITERATIONS}: Training")
    print(f"{'='*80}")
    print(f"Config: lr={config['learning_rate']}, steps={config['max_steps']}, r={config['lora_r']}")

    # Update fine_tune_blackwell.py with current config
    try:
        import re
        with open("fine_tune_blackwell.py", "r") as f:
            content = f.read()

        # Update learning rate
        content = re.sub(r"LEARNING_RATE = .*", f"LEARNING_RATE = {config['learning_rate']}", content)
        # Update max steps
        content = re.sub(r"MAX_STEPS = .*", f"MAX_STEPS = {config['max_steps']}", content)
        # Update LoRA rank
        content = re.sub(r"LORA_R = .*", f"LORA_R = {config['lora_r']}", content)
        content = re.sub(r"LORA_ALPHA = .*", f"LORA_ALPHA = {config['lora_alpha']}", content)

        # Update dataset path if provided
        if "dataset_path" in config:
            content = re.sub(r'"train": "agent-dataset-unsloth/train\.jsonl"',
                           f'"train": "{config["dataset_path"]}"', content)

        with open("fine_tune_blackwell.py", "w") as f:
            f.write(content)

        print("✓ Config updated in fine_tune_blackwell.py")

    except Exception as e:
        print(f"⚠ Failed to update config: {e}")

    # Run training
    cmd = ["bash", "run_training.sh"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout

    if result.returncode != 0:
        print(f"✗ Training failed")
        return None

    print("✓ Training completed")
    return "./outputs"

def extract_training_loss(model_path, iteration):
    """Extract training loss from logs instead of broken benchmark"""
    print(f"\n{'='*80}")
    print(f"Iteration {iteration}/{ITERATIONS}: Extracting Training Loss")
    print(f"{'='*80}")

    # Check if model exists
    if not os.path.exists(model_path):
        print("✗ Model not found")
        return None

    # Find the most recent training log
    log_files = []
    for file in Path(".").glob("training_*.log"):
        log_files.append((file.stat().st_mtime, file))

    if not log_files:
        print("✗ No training logs found")
        return {"loss": float('inf'), "perplexity": float('inf')}

    # Get the most recent log
    latest_log = sorted(log_files)[-1][1]
    print(f"Reading training log: {latest_log}")

    try:
        # Parse training loss from log
        final_loss = None
        losses = []

        with open(latest_log, 'r') as f:
            for line in f:
                # Look for loss values in training output
                # Format: {'loss': '4.606', 'grad_norm': '2.5', 'learning_rate': '1.6e-05', 'epoch': '0.42'}
                loss_match = re.search(r"'loss':\s*'([\d.]+)'", line)
                if loss_match:
                    loss_value = float(loss_match.group(1))
                    losses.append(loss_value)

        if losses:
            final_loss = losses[-1]  # Get the final loss value
            avg_loss = sum(losses[-10:]) / min(10, len(losses))  # Average of last 10 steps

            # Convert loss to perplexity approximation
            perplexity = math.exp(final_loss)

            print(f"✓ Extracted {len(losses)} loss values from training log")
            print(f"  Final loss: {final_loss:.4f}")
            print(f"  Avg loss (last 10): {avg_loss:.4f}")
            print(f"  Approx perplexity: {perplexity:.2f}")

            return {
                "loss": final_loss,
                "avg_loss": avg_loss,
                "perplexity": perplexity,
                "num_steps": len(losses),
            }
        else:
            print("✗ No loss values found in training log")
            return {"loss": float('inf'), "perplexity": float('inf')}

    except Exception as e:
        print(f"⚠ Error extracting loss: {e}")
        return {"loss": float('inf'), "perplexity": float('inf')}

def autoresearch(current_config, metrics, iteration):
    """Adjust hyperparameters and augment dataset based on training loss"""
    print(f"\n{'='*80}")
    print(f"Iteration {iteration}/{ITERATIONS}: Autoresearch")
    print(f"{'='*80}")

    if not metrics:
        print("No metrics available, keeping current config")
        return current_config

    loss = metrics.get("loss", float('inf'))
    perplexity = metrics.get("perplexity", float('inf'))
    print(f"Current loss: {loss:.4f}")
    print(f"Current perplexity: {perplexity:.2f}")

    new_config = current_config.copy()

    # Simple heuristic adjustment based on loss
    if loss > 6.0:
        print("High loss - reducing learning rate")
        if new_config["learning_rate"] > 5e-5:
            new_config["learning_rate"] = max(5e-5, new_config["learning_rate"] / 2)
    elif loss < 3.0:
        print("Low loss - can increase LoRA rank")
        if iteration % 3 == 0 and new_config["lora_r"] < 32:
            new_config["lora_r"] *= 2

    # Periodic adjustments
    if iteration % 3 == 0:
        print("Increasing training steps")
        new_config["max_steps"] = min(new_config["max_steps"] + 250, 1500)

    print(f"New config: lr={new_config['learning_rate']}, steps={new_config['max_steps']}, r={new_config['lora_r']}")

    # Dataset augmentation with synthetic examples
    print(f"\n{'='*80}")
    print("Dataset Augmentation")
    print(f"{'='*80}")

    try:
        from synthetic_data_generator import SyntheticDataGenerator

        generator = SyntheticDataGenerator()

        # Determine augmentation strategy based on iteration and loss
        if iteration == 1:
            num_synthetic = 200
            strategy = "balanced"
            print(f"Iteration {iteration}: Adding {num_synthetic} balanced synthetic examples")
        elif loss > 5.0:
            num_synthetic = 300
            strategy = "balanced"
            print(f"High loss ({loss:.4f}): Adding {num_synthetic} synthetic examples focusing on edge cases")
        elif iteration % 3 == 0:
            num_synthetic = 500
            strategy = "balanced"
            print(f"Iteration {iteration}: Adding {num_synthetic} synthetic examples (major augmentation)")
        else:
            num_synthetic = 150
            strategy = "balanced"
            print(f"Iteration {iteration}: Adding {num_synthetic} synthetic examples")

        # Paths for dataset augmentation
        base_dataset = "./agent-dataset-unsloth/train.jsonl"
        current_dataset = f"./agent-dataset-unsloth/train_iter_{iteration-1}.jsonl" if iteration > 1 else base_dataset
        augmented_dataset = f"./agent-dataset-unsloth/train_iter_{iteration}.jsonl"

        # Check if current dataset exists
        if not os.path.exists(current_dataset):
            print(f"Warning: {current_dataset} not found, using base dataset")
            current_dataset = base_dataset

        # Perform augmentation
        if os.path.exists(current_dataset):
            generator.augment_dataset(
                current_dataset_path=current_dataset,
                output_path=augmented_dataset,
                num_synthetic=num_synthetic,
                strategy=strategy
            )

            # Update training dataset path for next iteration
            new_config["dataset_path"] = augmented_dataset
            print(f"✓ Dataset augmented: {augmented_dataset}")
        else:
            print(f"✗ Dataset not found: {current_dataset}, skipping augmentation")

    except Exception as e:
        print(f"⚠ Dataset augmentation failed: {e}")
        print("Continuing with current dataset...")

    return new_config

def save_summary(iteration, config, metrics):
    """Save iteration summary"""
    summary = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "metrics": metrics,
    }

    Path(SUMMARY_DIR).mkdir(exist_ok=True)
    summary_file = Path(SUMMARY_DIR) / f"iter_{iteration}.json"

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary to {summary_file}")

def main():
    import math
    print("\n" + "="*80)
    print("10x Training Iteration Workflow (Fixed)")
    print("="*80)
    print(f"Iterations: {ITERATIONS}")
    print(f"Output directory: {SUMMARY_DIR}")
    print("Using training loss from logs (NOT broken benchmark)")

    current_config = initial_config.copy()
    all_results = []

    for iteration in range(1, ITERATIONS + 1):
        print(f"\n\n{'#'*80}")
        print(f"# ITERATION {iteration}/{ITERATIONS}")
        print(f"{'#'*80}")

        # Train
        model_path = run_training(current_config, iteration)
        if not model_path:
            print("Training failed - stopping")
            break

        # Extract training loss
        metrics = extract_training_loss(model_path, iteration)
        all_results.append({
            "iteration": iteration,
            "config": current_config.copy(),
            "metrics": metrics,
            "model_path": model_path,
        })

        # Save summary
        save_summary(iteration, current_config, metrics)

        # Autoresearch for next iteration
        if iteration < ITERATIONS:
            current_config = autoresearch(current_config, metrics, iteration)

    # Find best model
    print(f"\n\n{'='*80}")
    print("Finding Best Model")
    print(f"{'='*80}")

    best = min(all_results, key=lambda x: x["metrics"].get("loss", float('inf')))

    print(f"\n✓ Best model: Iteration {best['iteration']}")
    print(f"  Loss: {best['metrics']['loss']:.4f}")
    print(f"  Perplexity: {best['metrics']['perplexity']:.2f}")
    print(f"  Config: {best['config']}")
    print(f"  Model path: {best['model_path']}")

    # Copy best model
    best_dir = Path("./outputs_best")
    best_dir.mkdir(exist_ok=True)

    if os.path.exists(best["model_path"]):
        import shutil
        for file in Path(best["model_path"]).glob("*"):
            if file.is_file():
                shutil.copy(file, best_dir / file.name)
        print(f"✓ Best model copied to {best_dir}")

    print(f"\n\n{'='*80}")
    print("10x Iteration Workflow Complete!")
    print(f"{'='*80}")
    print("\nNext steps:")
    print("  1. Apply dead-head pruning: cd coherence-guided-dead-head-identification && python scripts/prune_model.py")
    print("  2. Run agent benchmarks: python evaluate_finetuned.py")
    print("  3. Deploy to production")

if __name__ == "__main__":
    main()
