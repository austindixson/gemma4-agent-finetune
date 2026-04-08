# Hybrid Training + Pruning Workflow Checklist

## Phase 1: Baseline Training
- [ ] Start training: `./run_training.sh`
- [ ] Monitor GPU usage: `watch -n 1 nvidia-smi`
- [ ] Check training logs: `tail -f training_*.log`

## Phase 2: Baseline Evaluation
- [ ] Run benchmark: `python benchmark_model.py`
- [ ] Record perplexity: _____
- [ ] Review sample outputs
- [ ] Document baseline metrics

## Phase 3: Iteration Decision
- [ ] Is perplexity acceptable? (< 15)
- [ ] Do samples look good?
- [ ] Decision: [ ] Continue to pruning  [ ] Retune with different parameters

## Phase 4: Dead-Head Analysis
- [ ] Run coherence analysis
- [ ] Identify % of dead heads: _____
- [ ] Document which layers have most dead heads

## Phase 5: Pruning
- [ ] Prune identified dead heads
- [ ] Save pruned model

## Phase 6: Post-Pruning Evaluation
- [ ] Benchmark pruned model
- [ ] Compare perplexity: baseline _____ vs pruned _____
- [ ] Measure speed improvement: _____%
- [ ] Measure memory reduction: _____%

## Phase 7: Optional Fine-Tuning
- [ ] Is degradation > 10%?
- [ ] If yes, light fine-tune (100-200 steps)
- [ ] Re-evaluate

## Final: Merge & Deploy
- [ ] Merge LoRA adapters
- [ ] Quantize if needed
- [ ] Test inference
- [ ] Deploy
