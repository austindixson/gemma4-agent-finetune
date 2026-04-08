#!/usr/bin/env python3
"""
10x Training Iteration Workflow with Benchmarking and Autoresearch

This script orchestrates:
1. Training iteration
2. Benchmarking (perplexity, samples, metrics)
3. Autoresearch (analyze results, adjust hyperparameters)
4. Repeat 10x
5. Final pruning and deployment
"""

import os
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================

ITERATIONS = 10
BASE_MODEL = "0xSero/gemma-4-21b-a4b-it-REAP"
OUTPUT_BASE = "./outputs"
BENCHMARK_RESULTS = "./benchmark_results"

# Training hyperparameter ranges for autoresearch
HYPERPARAMETER_RANGES = {
    "learning_rate": [1e-4, 2e-4, 3e-4, 5e-4],
    "max_steps": [250, 500, 750, 1000],
    "batch_size_per_device": [1, 2, 4],
    "lora_r": [8, 16, 32],
    "lora_alpha": [8, 16, 32],
}

# =============================================================================
# Iteration Workflow
# =============================================================================

def run_training(iteration, config):
    """Run a single training iteration"""
    print(f"\n{'='*80}")
    print(f"Iteration {iteration}/{ITERATIONS}: Training")
    print(f"{'='*80}")
    print(f"Config: {json.dumps(config, indent=2)}")

    # Update training script with new config
    # (This would involve editing fine_tune_blackwell.py or passing args)

    # Run training
    cmd = ["bash", "run_training.sh"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"✗ Training failed: {result.stderr}")
        return None

    print(f"✓ Training completed")
    return f"{OUTPUT_BASE}_iter{iteration}"

def run_benchmark(model_path, iteration):
    """Run benchmark evaluation"""
    print(f"\n{'='*80}")
    print(f"Iteration {iteration}/{ITERATIONS}: Benchmarking")
    print(f"{'='*80}")

    cmd = [
        "python", "benchmark_model.py",
        "--model_path", model_path,
        "--output_dir", f"{BENCHMARK_RESULTS}/iter_{iteration}"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"✗ Benchmark failed: {result.stderr}")
        return None

    # Load benchmark results
    results_file = Path(f"{BENCHMARK_RESULTS}/iter_{iteration}/metrics.json")
    if results_file.exists():
        with open(results_file) as f:
            metrics = json.load(f)
        print(f"✓ Benchmark completed")
        print(f"  Perplexity: {metrics.get('perplexity', 'N/A')}")
        print(f"  Inference speed: {metrics.get('tokens_per_second', 'N/A')} tok/s")
        print(f"  GPU memory: {metrics.get('gpu_memory_gb', 'N/A')} GB")
        return metrics

    return None

def autoresearch(iteration, current_config, benchmark_metrics):
    """
    Analyze results and adjust hyperparameters for next iteration

    This is a simplified autoresearch - in production, you'd want:
    - More sophisticated analysis
    - Bayesian optimization
    - Multi-objective optimization
    - Early stopping based on trends
    """
    print(f"\n{'='*80}")
    print(f"Iteration {iteration}/{ITERATIONS}: Autoresearch")
    print(f"{'='*80}")

    if not benchmark_metrics:
        print("⚠ No benchmark metrics available, using default config")
        return current_config

    perplexity = benchmark_metrics.get('perplexity', float('inf'))
    tokens_per_second = benchmark_metrics.get('tokens_per_second', 0)

    # Simple heuristic: if perplexity is high, reduce learning rate or increase steps
    new_config = current_config.copy()

    if perplexity > 15:
        print(f"  High perplexity ({perplexity:.2f}) - adjusting hyperparameters")
        if iteration % 2 == 0:
            # Try reducing learning rate
            current_lr = new_config.get('learning_rate', 2e-4)
            lr_options = [x for x in HYPERPARAMETER_RANGES['learning_rate'] if x < current_lr]
            if lr_options:
                new_config['learning_rate'] = max(lr_options)
                print(f"    → Reduced learning rate to {new_config['learning_rate']}")
        else:
            # Try increasing training steps
            current_steps = new_config.get('max_steps', 500)
            steps_options = [x for x in HYPERPARAMETER_RANGES['max_steps'] if x > current_steps]
            if steps_options:
                new_config['max_steps'] = min(steps_options)
                print(f"    → Increased max steps to {new_config['max_steps']}")
    elif perplexity < 10:
        print(f"  Low perplexity ({perplexity:.2f}) - good performance!")
        # Try more aggressive LoRA
        if iteration % 3 == 0:
            current_r = new_config.get('lora_r', 16)
            r_options = [x for x in HYPERPARAMETER_RANGES['lora_r'] if x > current_r]
            if r_options:
                new_config['lora_r'] = min(r_options)
                print(f"    → Increased LoRA rank to {new_config['lora_r']}")

    print(f"✓ Autoresearch completed - new config for next iteration")
    return new_config

def save_iteration_summary(iteration, config, model_path, metrics):
    """Save summary of each iteration"""
    summary = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "model_path": model_path,
        "metrics": metrics,
    }

    summary_file = Path(f"./iterations_summary/iter_{iteration}.json")
    summary_file.parent.mkdir(exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved summary to {summary_file}")

def analyze_best_model():
    """Analyze all iterations to find the best model"""
    print(f"\n{'='*80}")
    print(f"Analyzing Best Model Across {ITERATIONS} Iterations")
    print(f"{'='*80}")

    summaries_dir = Path("./iterations_summary")
    if not summaries_dir.exists():
        print("⚠ No iteration summaries found")
        return None

    best_model = None
    best_perplexity = float('inf')

    for summary_file in sorted(summaries_dir.glob("iter_*.json")):
        with open(summary_file) as f:
            summary = json.load(f)

        metrics = summary.get('metrics', {})
        perplexity = metrics.get('perplexity', float('inf'))

        print(f"  Iteration {summary['iteration']}: perplexity={perplexity:.2f}")

        if perplexity < best_perplexity:
            best_perplexity = perplexity
            best_model = summary

    if best_model:
        print(f"\n✓ Best model: Iteration {best_model['iteration']}")
        print(f"  Perplexity: {best_perplexity:.2f}")
        print(f"  Config: {json.dumps(best_model['config'], indent=2)}")

    return best_model

# =============================================================================
# Main Workflow
# =============================================================================

def main():
    print("\n" + "="*80)
    print("10x Training Iteration Workflow")
    print("="*80)
    print(f"Iterations: {ITERATIONS}")
    print(f"Base model: {BASE_MODEL}")
    print(f"Output base: {OUTPUT_BASE}")

    # Initial config
    current_config = {
        "learning_rate": 2e-4,
        "max_steps": 500,
        "batch_size_per_device": 2,
        "lora_r": 16,
        "lora_alpha": 16,
    }

    # Run iterations
    for iteration in range(1, ITERATIONS + 1):
        print(f"\n\n{'#'*80}")
        print(f"# ITERATION {iteration}/{ITERATIONS}")
        print(f"{'#'*80}")

        # Train
        model_path = run_training(iteration, current_config)
        if not model_path:
            print(f"✗ Iteration {iteration} failed - stopping")
            break

        # Benchmark
        metrics = run_benchmark(model_path, iteration)

        # Save summary
        save_iteration_summary(iteration, current_config, model_path, metrics)

        # Autoresearch for next iteration
        if iteration < ITERATIONS:
            current_config = autoresearch(iteration, current_config, metrics)

    # Analyze best model
    best_model = analyze_best_model()

    print(f"\n\n{'='*80}")
    print("10x Iteration Workflow Completed!")
    print(f"{'='*80}")

    if best_model:
        print(f"\nBest model: Iteration {best_model['iteration']}")
        print(f"Model path: {best_model['model_path']}")
        print(f"\nNext steps:")
        print("  1. Apply dead-head pruning")
        print("  2. Run recovery fine-tune if needed")
        print("  3. Run agent benchmarks (PinchBench/WildClawBench)")
        print("  4. Deploy to production")

if __name__ == "__main__":
    main()
