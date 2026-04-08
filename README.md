# Gemma-4 Agent Fine-Tuning

Training setup for fine-tuning Gemma-4-21B-A4B-IT-REAP with LoRA adapters for agent tasks.

## Overview

**Model:** Gemma-4-21B-A4B-IT-REAP (21.4B parameters, MoE with 103 experts)
**Method:** LoRA fine-tuning (29.6M trainable parameters, 0.14%)
**Hardware:** H100 GPU
**Framework:** Unsloth with 4-bit quantization

## Current Best Configuration

```python
# Optimized per Unsloth Gemma-4 best practices
LORA_R = 16
LORA_ALPHA = 32          # 2x rank (recommended for 21B models)
LORA_DROPOUT = 0
MAX_GRAD_NORM = 1.0     # Gradient clipping
WARMUP_RATIO = 0.03    # Adaptive warmup (3%)
LEARNING_RATE = 2e-4
MAX_STEPS = 500
```

## Performance

| Iteration | Loss | Perplexity | Config |
|-----------|------|------------|--------|
| 3 | 2.858 | 17.43 | alpha=16 (baseline) |
| **4** | **2.655** | **14.22** | **alpha=32** ✨ |
| 5 | 2.655 | 14.22 | alpha=32 |

**Improvement from alpha=32:** 7.1% loss reduction, 18.4% perplexity reduction

## Files

### Training Scripts
- `fine_tune_blackwell.py` - Main training script with Unsloth
- `iterate_resume.py` - Resume training from last iteration
- `synthetic_data_generator.py` - Generate training examples

### Workflow Scripts
- `run_training.sh` - Start training
- `cleanup.sh` - Quick disk cleanup
- `aggressive_cleanup.sh` - Full cleanup (5GB+ recovery)
- `cleanup_report.sh` - Check disk usage

### Analysis Scripts
- `benchmark_model.py` - Evaluate model performance
- `evaluate_finetuned.py` - Test fine-tuned model

## Key Improvements

1. **LoRA Alpha**: 16 → 32 (Unsloth recommendation)
2. **Warmup**: Fixed 50 steps → 0.03 ratio (adaptive)
3. **Gradient Clipping**: None → 1.0 (stability)
4. **Validation**: 50 → 100 samples (reliability)

## Dataset

- **Base**: 3,995 agent task samples
- **Augmented**: ~6,000 samples with synthetic examples
- **Growth**: ~5% per iteration

## Training Progress

- **Completed**: Iterations 1-5 (50% of 10x workflow)
- **Current**: Iteration 6 in progress
- **Best Model**: Iteration 4/5 (loss: 2.655)

## Usage

### Start Training
```bash
source ~/miniconda/bin/activate gemma-h100
python fine_tune_blackwell.py
```

### Resume from Last Iteration
```bash
python iterate_resume.py
```

### Monitor Training
```bash
tail -f training_*.log
```

## Requirements

- Unsloth: `pip install "unsloth[colab-new]"`
- PyTorch 2.10+ with CUDA support
- H100 GPU (80GB VRAM minimum)

## Notes

- Model: 0xSero/gemma-4-21b-a4b-it-REAP
- Quantization: 4-bit (QLoRA)
- Training time: ~34 minutes per iteration
- Disk space: Keep at least 7GB free

## References

- [Unsloth Gemma-4 Documentation](https://www.unsloth.ai/docs/models/gemma-4/train)
- [D-Flash Analysis](dflash_analysis.md)
- [Autoresearch Findings](autoresearch_findings.md)

---

*Training in progress - Iteration 6/10*
