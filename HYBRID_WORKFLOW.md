# Hybrid Training + Pruning Workflow

## Overview

This workflow combines **iterative training** with **dead-head pruning** to create an efficient, high-quality fine-tuned model.

## Workflow Summary

```
Train → Evaluate → (Option: Retune) → Dead-Head Analysis → Prune → (Option: Recover) → Deploy
```

## Scripts Available

| Script | Purpose | Phase |
|--------|---------|-------|
| `fine_tune_blackwell.py` | Main training script | 1, 3 |
| `benchmark_model.py` | Evaluate model quality | 2, 6 |
| `create_merge_script.py` | Merge LoRA for deployment | Final |
| `retune_model.py` | Light fine-tuning after pruning | 7 |

## Phase 1: Baseline Training (20-30 min)

```bash
cd ~/agent-finetune
./run_training.sh
```

**What happens**:
- Loads gemma-4-21b-a4b-it-REAP in 4-bit
- Trains on your 3,797 Claude conversations
- Saves checkpoints every 25 steps
- Final output: `./outputs/` with LoRA adapters

**Monitor with**:
```bash
# Terminal 1: Training logs
tail -f training_*.log

# Terminal 2: GPU usage
watch -n 1 nvidia-smi
```

## Phase 2: Baseline Evaluation (5 min)

```bash
source ~/miniconda/bin/activate gemma-h100
python benchmark_model.py
```

**Expected outputs**:
- Perplexity score (target: < 15)
- Inference speed (tokens/sec)
- Sample generations
- GPU memory usage

**Decision point**:
- ✅ Perplexity < 15 + good samples → Continue to Phase 4
- ❌ Perplexity > 20 + poor samples → Go to Phase 3

## Phase 3: Iteration (Optional, 20-30 min)

If baseline isn't good enough, retrain with adjusted parameters:

**Option A**: More steps
```bash
# Edit fine_tune_blackwell.py
MAX_STEPS = 1000  # Instead of 500

./run_training.sh
```

**Option B**: Different learning rate
```bash
# Edit fine_tune_blackwell.py
LEARNING_RATE = 1e-4  # Or 5e-4

./run_training.sh
```

**Option C**: Larger batch size (if GPU memory allows)
```bash
# Edit fine_tune_blackwell.py
BATCH_SIZE_PER_DEVICE = 4  # Instead of 2

./run_training.sh
```

Repeat Phase 2 evaluation after each iteration.

## Phase 4: Dead-Head Analysis (5-10 min)

```bash
cd ~/agent-finetune/coherence-guided-dead-head-identification

# Analyze attention heads
python scripts/analyze_attention_heads.py \
  --model_path ../outputs \
  --output ../dead_head_analysis.json
```

**What happens**:
- Computes coupling strength for each attention head
- Identifies dead heads using: `tau_death = 0.96 / sqrt(d_model)`
- Reports % of heads that can be safely pruned

**Expected**: 10-30% of heads are typically dead

## Phase 5: Pruning (5 min)

```bash
# Prune identified dead heads
python scripts/prune_model.py \
  --model_path ../outputs \
  --analysis ../dead_head_analysis.json \
  --output ../outputs_pruned/
```

**What happens**:
- Removes dead attention heads from model
- Creates pruned model with fewer parameters
- Maintains most of the performance

**Expected benefits**:
- Smaller model size
- Faster inference
- Lower memory usage

## Phase 6: Post-Pruning Evaluation (5 min)

```bash
cd ~/agent-finetune
python benchmark_model.py --model_path ./outputs_pruned/
```

**Compare with baseline**:
- Perplexity change (should be minimal)
- Speed improvement
- Memory reduction

**Decision point**:
- ✅ Degradation < 10% → Skip to Final
- ❌ Degradation > 10% → Go to Phase 7

## Phase 7: Recovery Fine-Tuning (Optional, 10 min)

If pruning caused too much degradation:

```bash
source ~/miniconda/bin/activate gemma-h100
python retune_model.py
```

**What happens**:
- Light fine-tuning (150 steps)
- Conservative learning rate (1e-4)
- Recovers performance while keeping efficiency gains

**Output**: `./outputs_recovered/`

## Final: Merge & Deploy (10 min)

```bash
python create_merge_script.py
```

**What happens**:
- Merges LoRA adapters into base model
- Saves as 16-bit model (or 4-bit if preferred)
- Ready for deployment

**Output**: `./merged_model/`

---

## Quick Reference Commands

### Start training
```bash
cd ~/agent-finetune
./run_training.sh
```

### Monitor training
```bash
tail -f training_*.log
watch -n 1 nvidia-smi
```

### Evaluate model
```bash
python benchmark_model.py
```

### Merge for deployment
```bash
python create_merge_script.py
```

### Test merged model
```bash
python test_inference.py --model ./merged_model
```

---

## Timeline Estimates

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| 1. Baseline Training | 20-30 min | None |
| 2. Baseline Evaluation | 5 min | Phase 1 |
| 3. Iteration (optional) | 20-30 min | Phase 2 |
| 4. Dead-Head Analysis | 5-10 min | Phase 2 |
| 5. Pruning | 5 min | Phase 4 |
| 6. Post-Pruning Eval | 5 min | Phase 5 |
| 7. Recovery (optional) | 10 min | Phase 6 |
| **Total (no iterations)** | **45-55 min** | - |
| **Total (with iterations)** | **1.5-2 hours** | - |

---

## Success Criteria

✅ **Baseline**: Perplexity < 15, coherent samples
✅ **Pruning**: < 10% perplexity increase, > 10% speed gain
✅ **Final**: Model generates quality responses, efficient inference

---

## Troubleshooting

### Training Issues

**Out of Memory**:
- Reduce `BATCH_SIZE_PER_DEVICE` to 1
- Increase `GRADIENT_ACCUMULATION_STEPS` to 8

**Slow Training**:
- Verify GPU utilization is > 80%
- Check `nvidia-smi` for throttling

### Evaluation Issues

**High Perplexity**:
- Train for more steps
- Check data quality
- Verify chat template is applied

**Poor Samples**:
- Review training data
- Adjust temperature in generation
- Train for more steps

### Pruning Issues

**Too Much Degradation**:
- Use less aggressive threshold
- Run recovery fine-tuning
- Consider keeping some "borderline" heads

---

## Ready to Start?

**Phase 1**: Run `./run_training.sh`

Or would you like to review the training parameters first?
