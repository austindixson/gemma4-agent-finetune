# Training Improvements for Remaining Iterations (4-10)

## Status
- **Current**: Iteration 3/10 in progress (~9% complete, step 45/500)
- **First validation check**: End of iteration 3
- **Rate limit**: Unsloth docs blocked until 2026-04-20

## Implemented in Current Run
✓ Validation loss testing (every 3 iterations)
✓ Overfitting detection (val_loss > 1.5x train_loss)
✓ Early stopping (patience=2 validations)
✓ Synthetic data augmentation
✓ General instruction examples for capability drift
✓ Training loss extraction from logs

## Improvements to Apply for Iterations 4-10

### 1. Gradient Clipping (HIGH PRIORITY)
**Why**: Prevents gradient explosions, improves stability
**Implementation**: Add to fine_tune_blackwell.py
```python
max_grad_norm = 1.0  # Clip gradients at norm 1.0
```

### 2. Increased Warmup (HIGH PRIORITY)
**Current**: 50 steps (10% of 500)
**Issue**: May be too short for stable convergence
**Proposed**: 100 steps (20%)
**Benefit**: More stable learning rate ramp-up

### 3. Larger Validation Set (MEDIUM PRIORITY)
**Current**: 50 samples every 3 iterations
**Proposed**: 100 samples
**Benefit**: More reliable overfitting detection

### 4. Save Best Validation Checkpoint (MEDIUM PRIORITY)
**Why**: Currently saves based on training loss only
**Proposed**: Track best val_loss and save checkpoint
**Benefit**: Prevents saving overfitted models

### 5. Learning Rate Schedule Tuning (LOW PRIORITY)
**Current**: cosine with 50-step warmup
**Proposed**: linear warmup (100 steps) + cosine decay
**Benefit**: More stable early training

### 6. LoRA Alpha Adjustment (LOW PRIORITY)
**Current**: alpha=16 (same as r)
**Proposed**: alpha=32 (2x rank) for iterations 7+
**Benefit**: Better scaling when r increases to 32

## Application Strategy
1. Update fine_tune_blackwell.py for iteration 4+
2. Update iterate_10x_enhanced.py validation code
3. Keep current training running
4. Apply changes before iteration 4 starts

## Expected Impact
- More stable convergence with gradient clipping
- Better overfitting detection with larger val set
- Prevent saving degraded models with val-based checkpointing
