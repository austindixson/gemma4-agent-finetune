#!/usr/bin/env python3
"""
Evaluate fine-tuned model on PinchBench and WildClawBench.

This script runs agent benchmarks to measure how well the fine-tuned
model performs on real-world tasks compared to the baseline.
"""

import os
import subprocess
import json
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

# Model paths
BASE_MODEL = "0xSero/gemma-4-21b-a4b-it-REAP"
FINETUNED_MODEL = "./merged_model"  # Path to merged fine-tuned model

# Benchmark paths
PINCHBENCH_DIR = "./skill"
WILDCLAW_DIR = "./WildClawBench"

# Output
RESULTS_DIR = "./benchmark_results"

print("=" * 80)
print("Agent Benchmark Evaluation")
print("=" * 80)

# =============================================================================
# Setup
# =============================================================================

os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# PinchBench Evaluation
# =============================================================================

print("\n" + "=" * 80)
print("Phase 1: PinchBench Evaluation (23 tasks)")
print("=" * 80)

print("\nThis will test:")
print("  - Calendar management")
print("  - Email triage and search")
print("  - Stock research")
print("  - Code generation")
print("  - Document analysis")
print("  - Multi-step workflows")
print("\nEstimated time: 30-60 minutes")

choice = input("\nRun PinchBench? (y/n): ").lower()

if choice == 'y':
    print("\nRunning PinchBench with fine-tuned model...")

    try:
        # Run PinchBench with fine-tuned model
        cmd = [
            "uv", "run", "benchmark.py",
            "--model", FINETUNED_MODEL,
            "--output-dir", f"{RESULTS_DIR}/pinchbench_finetuned",
            "--no-upload"
        ]

        result = subprocess.run(
            cmd,
            cwd=PINCHBENCH_DIR,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        print(result.stdout)
        if result.returncode != 0:
            print("STDERR:", result.stderr)

    except subprocess.TimeoutExpired:
        print("PinchBench timed out after 1 hour")
    except Exception as e:
        print(f"Error running PinchBench: {e}")
        print("\nTip: Make sure PinchBench dependencies are installed:")
        print("  cd skill && uv sync")

# =============================================================================
# WildClawBench Evaluation
# =============================================================================

print("\n" + "=" * 80)
print("Phase 2: WildClawBench Evaluation (60 tasks)")
print("=" * 80)

print("\nThis will test:")
print("  - Productivity flows (10 tasks)")
print("  - Code intelligence (12 tasks)")
print("  - Social interaction (6 tasks)")
print("  - Search & retrieval (11 tasks)")
print("  - Creative synthesis (11 tasks)")
print("  - Safety alignment (10 tasks)")
print("\nEstimated time: 2-4 hours")
print("Top score on leaderboard: 51.6% (Claude Opus 4.6)")

choice = input("\nRun WildClawBench? (y/n): ").lower()

if choice == 'y':
    print("\nRunning WildClawBench with fine-tuned model...")

    try:
        # Run WildClawBench with fine-tuned model
        # Note: WildClawBench requires Docker and specific setup
        cmd = [
            "python", "eval/run_benchmark.py",
            "--model", FINETUNED_MODEL,
            "--output", f"{RESULTS_DIR}/wildclaw_finetuned"
        ]

        result = subprocess.run(
            cmd,
            cwd=WILDCLAW_DIR,
            capture_output=True,
            text=True,
            timeout=14400  # 4 hours timeout
        )

        print(result.stdout)
        if result.returncode != 0:
            print("STDERR:", result.stderr)

    except subprocess.TimeoutExpired:
        print("WildClawBench timed out after 4 hours")
    except Exception as e:
        print(f"Error running WildClawBench: {e}")
        print("\nTip: Make sure WildClawBench is properly set up:")
        print("  cd WildClawBench && docker-compose up -d")

# =============================================================================
# Results Summary
# =============================================================================

print("\n" + "=" * 80)
print("Benchmark Results")
print("=" * 80)

print("\nResults saved to:", RESULTS_DIR)
print("\nTo compare with baseline model:")
print("  1. Run benchmarks on base model")
print("  2. Compare scores")
print("  3. Calculate improvement %")

print("\nKey metrics to track:")
print("  - Overall success rate")
print("  - Per-category performance")
print("  - Average time per task")
print("  - Cost per task")

# =============================================================================
# Compare Results (if both baselines exist)
# =============================================================================

print("\n" + "=" * 80)
print("Comparison with Leaderboard")
print("=" * 80)

print("\nWildClawBench Leaderboard (top models):")
print("  1. Claude Opus 4.6:  51.6%")
print("  2. GPT-5.4:          50.3%")
print("  3. GLM 5:            42.6%")
print("  ...")
print("\nYour fine-tuned model will be scored against the same tasks.")

print("\n" + "=" * 80)
print("Next Steps")
print("=" * 80)
print("\n1. If benchmarks pass:")
print("     → Model is ready for deployment")
print("     → Consider submitting to leaderboards")
print("\n2. If benchmarks show weaknesses:")
print("     → Analyze failed task categories")
print("     → Add more training data for weak areas")
print("     → Re-train with adjusted hyperparameters")
print("\n3. For iterative improvement:")
print("     → Run benchmarks after each training iteration")
print("     → Track progress over time")
print("     → Focus on lowest-scoring categories")

print("\n✓ Evaluation script complete!")
