# Improvements Applied for Iterations 4-10

## ✅ Applied (2026-04-08)

### fine_tune_blackwell.py
- [x] `LORA_ALPHA = 32` (was 16) - Unsloth recommends alpha ≥ r, 2x better for 26B-A4B
- [x] `warmup_ratio = 0.03` (3% of training, was fixed 50 steps) - More adaptive
- [x] `MAX_GRAD_NORM = 1.0` - Gradient clipping for stability

### iterate_10x_enhanced.py
- [x] Initial `lora_alpha = 32` (was 16)
- [x] Validation samples increased to 100 (was 50) - More reliable estimates

## Impact on Remaining Iterations (4-10)

### Iteration 4+ will have:
1. Better LoRA scaling (alpha=32 instead of 16) - More stable updates
2. Adaptive warmup (3% of steps instead of fixed 50) - Scales with training length
3. Gradient clipping (max_norm=1.0) - Prevents gradient explosions
4. More reliable validation (100 samples vs 50) - Better overfitting detection

## Expected Benefits
- **Stability**: Gradient clipping reduces risk of loss spikes
- **Convergence**: Proper alpha/r ratio improves optimization
- **Reliability**: Larger validation set catches overfitting earlier
- **Adaptability**: warmup_ratio scales with training length

## Training Status (2026-04-08 14:20 UTC)
- **Iteration 3**: 60/500 steps (12%), loss: 3.93-4.606
- **First validation check**: End of iteration 3
- **Improvements active**: Iteration 4 onwards
