#!/usr/bin/env python3
"""
Enhanced 10x Training Iteration Workflow with Validation & Early Stopping

Features:
1. Runs training (500 steps)
2. Extracts training loss from logs
3. Validation loss testing every 3 iterations
4. Overfitting detection & mitigation
5. Synthetic general instruction data if capability degrades
6. Early stopping if no improvement
7. Selects best model
8. Proceeds to pruning
"""

import os
import json
import subprocess
import re
import math
from pathlib import Path
from datetime import datetime

# Configuration
ITERATIONS = 10
SUMMARY_DIR = "./iterations_summary"
VALIDATION_INTERVAL = 3  # Validate every 3 iterations
EARLY_STOPPING_PATIENCE = 2  # Stop after 2 validations without improvement
OVERFITTING_THRESHOLD = 1.5  # Val loss must be within 1.5x of train loss

# Initial hyperparameters
initial_config = {
    "learning_rate": 2e-4,
    "max_steps": 500,
    "batch_size_per_device": 2,
    "lora_r": 16,
    "lora_alpha": 32,  # Updated per Unsloth Gemma-4 best practices
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
    """Extract training loss from logs"""
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

def compute_validation_loss(model_path, iteration):
    """Compute validation loss on held-out set"""
    print(f"\n{'='*80}")
    print(f"Iteration {iteration}/{ITERATIONS}: Validation Loss Testing")
    print(f"{'='*80}")
    print("Computing loss on held-out validation set...")

    try:
        import torch
        from datasets import load_dataset
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template, standardize_sharegpt

        # Load base model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="0xSero/gemma-4-21b-a4b-it-REAP",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )

        # Load LoRA adapters
        model.load_adapter(model_path)
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
        tokenizer.pad_token = tokenizer.eos_token

        # Load validation set
        dataset = load_dataset("json", data_files={"valid": "agent-dataset-unsloth/valid.jsonl"})
        dataset = standardize_sharegpt(dataset)

        print(f"Loaded {len(dataset['valid'])} validation samples")

        # Compute loss on subset
        eval_samples = 100  # Increased from 50 for better reliability
        total_loss = 0.0
        model.eval()

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
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                # Forward pass
                outputs = model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item()

        avg_val_loss = total_loss / eval_samples
        val_perplexity = math.exp(avg_val_loss)

        print(f"✓ Validation Results:")
        print(f"  - Avg val loss: {avg_val_loss:.4f}")
        print(f"  - Val perplexity: {val_perplexity:.2f}")

        return {
            "val_loss": avg_val_loss,
            "val_perplexity": val_perplexity,
            "num_samples": eval_samples,
        }

    except Exception as e:
        print(f"⚠ Validation error: {e}")
        return {"val_loss": float('inf'), "val_perplexity": float('inf')}

def check_overfitting(train_metrics, val_metrics):
    """Check for overfitting by comparing train vs validation loss"""
    train_loss = train_metrics.get("loss", float('inf'))
    val_loss = val_metrics.get("val_loss", float('inf'))

    if train_loss == float('inf') or val_loss == float('inf'):
        return False, "Cannot determine - missing metrics"

    # Calculate ratio
    ratio = val_loss / train_loss

    print(f"\n{'='*80}")
    print("Overfitting Check")
    print(f"{'='*80}")
    print(f"Train loss: {train_loss:.4f}")
    print(f"Val loss: {val_loss:.4f}")
    print(f"Ratio (val/train): {ratio:.2f}")

    if ratio > OVERFITTING_THRESHOLD:
        print(f"⚠ OVERFITTING DETECTED: Val loss is {ratio:.1f}x train loss")
        return True, f"Overfitting: Val loss {ratio:.1f}x train loss"
    else:
        print(f"✓ No significant overfitting")
        return False, "OK"

def generate_general_instruction_examples(num_examples=100):
    """Generate general instruction-following examples to maintain broad capability"""
    print(f"\n{'='*80}")
    print("Generating General Instruction Examples")
    print(f"{'='*80}")
    print(f"Creating {num_examples} broad instruction-following examples...")

    examples = []

    # General instruction categories
    categories = {
        "explanation": [
            ("Explain what recursion is in programming", "Recursion is a programming technique where a function calls itself to solve smaller instances of the same problem. It consists of a base case that stops the recursion and a recursive case that breaks down the problem."),
            ("What is the difference between HTTP and HTTPS?", "HTTPS (HTTP Secure) encrypts the data being transmitted, while HTTP sends it in plain text. HTTPS uses TLS/SSL encryption to protect sensitive information from eavesdropping."),
            ("Explain the concept of object-oriented programming", "Object-oriented programming (OOP) is a paradigm based on 'objects' that contain data and code. Key principles include encapsulation, inheritance, and polymorphism."),
        ],
        "writing": [
            ("Write a haiku about programming", "Code flows like water\nBugs emerge from the deep dark\nDebug, test, deploy now."),
            ("Write a professional email to request a meeting", "Subject: Meeting Request - [Topic]\n\nHi [Name],\n\nI hope this email finds you well. I would like to request a brief meeting to discuss [topic]. Are you available sometime [time frame]?\n\nBest regards,\n[Your name]"),
        ],
        "analysis": [
            ("Anze the pros and cons of microservices architecture", "Pros: Scalability, independent deployment, technology diversity. Cons: Increased complexity, network latency, debugging challenges. Best for large teams and complex applications."),
            ("Compare SQL and NoSQL databases", "SQL: Structured data, ACID transactions, fixed schema. NoSQL: Flexible schema, horizontal scaling, high performance. Use SQL for complex relationships, NoSQL for simple data models."),
        ],
        "coding_general": [
            ("Write a function to check if a number is prime", "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True"),
            ("Create a function to reverse a string", "def reverse_string(s):\n    return s[::-1]\n\n# Example: reverse_string('hello') returns 'olleh'"),
        ],
    }

    # Generate examples from each category
    for category, items in categories.items():
        for prompt, response in items:
            examples.append([
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": response}
            ])

    # Add more variety by shuffling and selecting
    random.shuffle(examples)
    return examples[:num_examples]

def autoresearch(current_config, train_metrics, val_metrics, iteration, history):
    """Adjust hyperparameters and augment dataset based on results"""
    print(f"\n{'='*80}")
    print(f"Iteration {iteration}/{ITERATIONS}: Autoresearch")
    print(f"{'='*80}")

    if not train_metrics:
        print("No metrics available, keeping current config")
        return current_config

    train_loss = train_metrics.get("loss", float('inf'))
    val_loss = val_metrics.get("val_loss", None) if val_metrics else None

    print(f"Train loss: {train_loss:.4f}")
    if val_loss:
        print(f"Val loss: {val_loss:.4f}")

    new_config = current_config.copy()

    # Simple heuristic adjustment based on training loss
    if train_loss > 6.0:
        print("High training loss - reducing learning rate")
        if new_config["learning_rate"] > 5e-5:
            new_config["learning_rate"] = max(5e-5, new_config["learning_rate"] / 2)
    elif train_loss < 3.0:
        print("Low training loss - can increase LoRA rank")
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

        # Check if we need general instruction examples (overfitting mitigation)
        needs_general_examples = False
        if val_loss and iteration > 1:
            # Check history for overfitting pattern
            recent = history[-3:] if len(history) >= 3 else history
            overfitting_count = sum(1 for h in recent if h.get("overfitting", False))
            if overfitting_count >= 2:
                needs_general_examples = True
                print("⚠ Overfitting pattern detected - adding general instruction examples")

        # Determine augmentation strategy
        if iteration == 1:
            num_synthetic = 200
            strategy = "balanced"
            print(f"Iteration {iteration}: Adding {num_synthetic} balanced synthetic examples")
        elif train_loss > 5.0:
            num_synthetic = 300
            strategy = "balanced"
            print(f"High loss ({train_loss:.4f}): Adding {num_synthetic} synthetic examples")
        elif needs_general_examples:
            num_synthetic = 250
            strategy = "balanced"
            print(f"Adding {num_synthetic} balanced examples + general instructions")
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

            # Add general instruction examples if needed
            if needs_general_examples:
                general_examples = generate_general_instruction_examples(num_examples=100)
                with open(augmented_dataset, 'a') as f:
                    for example in general_examples:
                        f.write(json.dumps({"conversations": example}) + '\n')
                print(f"✓ Added {len(general_examples)} general instruction examples")

            # Update training dataset path for next iteration
            new_config["dataset_path"] = augmented_dataset
            print(f"✓ Dataset augmented: {augmented_dataset}")
        else:
            print(f"✗ Dataset not found: {current_dataset}, skipping augmentation")

    except Exception as e:
        print(f"⚠ Dataset augmentation failed: {e}")
        print("Continuing with current dataset...")

    return new_config

def check_early_stopping(history):
    """Check if we should stop early based on validation loss"""
    print(f"\n{'='*80}")
    print("Early Stopping Check")
    print(f"{'='*80}")

    if len(history) < 2:
        return False, "Need more history"

    # Get recent validation losses
    val_losses = [h.get("val_loss", float('inf')) for h in history if h.get("val_loss") is not None]

    if len(val_losses) < 2:
        return False, "Need at least 2 validation points"

    # Check if validation loss is improving
    recent_losses = val_losses[-EARLY_STOPPING_PATIENCE:]
    if len(recent_losses) < EARLY_STOPPING_PATIENCE:
        return False, f"Need {EARLY_STOPPING_PATIENCE} validation points"

    # Check if plateaued or getting worse
    best_val_loss = min(val_losses)
    latest_val_loss = recent_losses[-1]

    if latest_val_loss >= best_val_loss:
        print(f"⚠ Early stopping triggered:")
        print(f"  - Best val loss: {best_val_loss:.4f}")
        print(f"  - Latest val loss: {latest_val_loss:.4f}")
        print(f"  - No improvement for {EARLY_STOPPING_PATIENCE} validations")
        return True, f"No improvement for {EARLY_STOPPING_PATIENCE} validations"

    print(f"✓ Validation loss still improving: {val_losses[0]:.4f} → {latest_val_loss:.4f}")
    return False, "Validation loss improving"

def save_summary(iteration, config, train_metrics, val_metrics, overfitting_info):
    """Save iteration summary"""
    summary = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "overfitting": overfitting_info,
    }

    Path(SUMMARY_DIR).mkdir(exist_ok=True)
    summary_file = Path(SUMMARY_DIR) / f"iter_{iteration}.json"

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary to {summary_file}")

def main():
    import random
    print("\n" + "="*80)
    print("Enhanced 10x Training Iteration Workflow")
    print("="*80)
    print(f"Iterations: {ITERATIONS}")
    print(f"Output directory: {SUMMARY_DIR}")
    print(f"Validation interval: Every {VALIDATION_INTERVAL} iterations")
    print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE} validations")

    current_config = initial_config.copy()
    all_results = []
    history = []

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
        train_metrics = extract_training_loss(model_path, iteration)

        # Validation loss testing (every 3 iterations)
        val_metrics = None
        overfitting_info = {"overfitting": False, "message": "No validation this iteration"}

        if iteration % VALIDATION_INTERVAL == 0:
            val_metrics = compute_validation_loss(model_path, iteration)
            is_overfitting, overfit_msg = check_overfitting(train_metrics, val_metrics)
            overfitting_info = {"overfitting": is_overfitting, "message": overfit_msg}

        # Record results
        result = {
            "iteration": iteration,
            "config": current_config.copy(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "overfitting": overfitting_info.get("overfitting", False),
            "model_path": model_path,
        }
        all_results.append(result)
        history.append(result)

        # Save summary
        save_summary(iteration, current_config, train_metrics, val_metrics, overfitting_info)

        # Check early stopping (only if we have validation data)
        should_stop, stop_reason = False, ""
        if val_metrics:
            should_stop, stop_reason = check_early_stopping(history)

        if should_stop:
            print(f"\n{'='*80}")
            print("EARLY STOPPING TRIGGERED")
            print(f"{'='*80}")
            print(f"Reason: {stop_reason}")
            print("Stopping training early to prevent overfitting")
            break

        # Autoresearch for next iteration
        if iteration < ITERATIONS:
            current_config = autoresearch(current_config, train_metrics, val_metrics, iteration, history)

    # Find best model
    print(f"\n\n{'='*80}")
    print("Finding Best Model")
    print(f"{'='*80}")

    # Prefer models with validation data
    models_with_val = [r for r in all_results if r.get("val_metrics")]
    if models_with_val:
        best = min(models_with_val, key=lambda x: x["val_metrics"].get("val_loss", float('inf')))
        metric_used = f"val_loss: {best['val_metrics']['val_loss']:.4f}"
    else:
        best = min(all_results, key=lambda x: x["train_metrics"].get("loss", float('inf')))
        metric_used = f"train_loss: {best['train_metrics']['loss']:.4f}"

    print(f"\n✓ Best model: Iteration {best['iteration']}")
    print(f"  {metric_used}")
    if best.get("val_metrics"):
        print(f"  Val perplexity: {best['val_metrics']['val_perplexity']:.2f}")
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
    print("Enhanced 10x Iteration Workflow Complete!")
    print(f"{'='*80}")
    print("\nNext steps:")
    print("  1. Apply dead-head pruning: cd coherence-guided-dead-head-identification && python scripts/prune_model.py")
    print("  2. Run agent benchmarks: python evaluate_finetuned.py")
    print("  3. Deploy to production")

if __name__ == "__main__":
    main()

# Auto-cleanup function (call before each iteration)
def auto_cleanup():
    import subprocess
    try:
        # Clear temp files
        subprocess.run(['bash', '~/agent-finetune/cleanup.sh'], 
                      capture_output=True, timeout=30)
    except:
        pass
