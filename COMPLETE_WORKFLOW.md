# Complete Workflow: Training + Pruning + Evaluation

## 🎯 Overview

```
Train → Evaluate → (Iterate) → Prune → (Recover) → Benchmarks → Deploy
```

This workflow combines:
1. **Fine-tuning** on your Claude conversations
2. **Dead-head pruning** for efficiency
3. **Agent benchmarks** (PinchBench + WildClawBench) for quality assurance

---

## 📋 Complete Workflow

### **Phase 1: Baseline Training** (20-30 min)
```bash
cd ~/agent-finetune
./run_training.sh
```

**Output**: `./outputs/` with LoRA adapters

---

### **Phase 2: Quick Evaluation** (5 min)
```bash
source ~/miniconda/bin/activate gemma-h100
python benchmark_model.py
```

**Metrics**:
- Perplexity (target: < 15)
- Inference speed
- GPU memory
- Sample quality

**Decision**: Good enough? → Continue | Poor? → Retune

---

### **Phase 3: Iteration** (Optional, 20-30 min)

If perplexity > 15 or samples are poor:

**Option A**: More steps
```bash
# Edit fine_tune_blackwell.py
MAX_STEPS = 1000  # 2 epochs
./run_training.sh
```

**Option B**: Adjust learning rate
```bash
LEARNING_RATE = 1e-4  # Lower
# or
LEARNING_RATE = 5e-4  # Higher
```

---

### **Phase 4: Dead-Head Analysis** (5-10 min)
```bash
cd ~/agent-finetune/coherence-guided-dead-head-identification

python scripts/analyze_attention_heads.py \
  --model_path ../outputs \
  --output ../dead_head_analysis.json
```

**Output**: `dead_head_analysis.json` with % of dead heads

---

### **Phase 5: Pruning** (5 min)
```bash
python scripts/prune_model.py \
  --model_path ../outputs \
  --analysis ../dead_head_analysis.json \
  --output ../outputs_pruned/
```

**Benefits**:
- Smaller model
- Faster inference
- Lower memory

---

### **Phase 6: Post-Pruning Eval** (5 min)
```bash
cd ~/agent-finetune
python benchmark_model.py --model_path ./outputs_pruned/
```

**Compare**:
- Perplexity change
- Speed improvement
- Memory reduction

**Decision**: Degradation < 10%? → Continue | > 10%? → Recover

---

### **Phase 7: Recovery Fine-Tune** (Optional, 10 min)
```bash
python retune_model.py
```

Light fine-tuning to recover performance after pruning.

---

### **Phase 8: Merge Model** (10 min)
```bash
python create_merge_script.py
```

**Output**: `./merged_model/` with merged weights

---

### **Phase 9: Agent Benchmarks** (30 min - 4 hours)

**Option A: PinchBench** (23 tasks, ~30-60 min)
```bash
cd ~/agent-finetune
python evaluate_finetuned.py
# Choose: Run PinchBench? y
```

Tests: Calendar, email, coding, research, workflows

**Option B: WildClawBench** (60 tasks, ~2-4 hours)
```bash
python evaluate_finetuned.py
# Choose: Run WildClawBench? y
```

Tests: Productivity, code, social, search, creative, safety

**Option C: Both** (Comprehensive evaluation)

---

### **Phase 10: Deploy** 🚀

If benchmarks pass:
- Upload to HuggingFace
- Deploy to production
- Submit to leaderboards

If benchmarks show weaknesses:
- Analyze failed categories
- Add more training data
- Re-train and re-test

---

## 📊 Decision Tree

```
Start Training
     │
     ▼
 Perplexity < 15?
     │
     ├─ NO ──→ Retune (Phase 3) ──→ Loop back
     │
     └─ YES ──→ Prune (Phase 5)
                    │
                    ▼
             Degradation < 10%?
                    │
                    ├─ NO ──→ Recover (Phase 7) ──→ Merge (Phase 8)
                    │
                    └─ YES ──→ Merge (Phase 8)
                                  │
                                  ▼
                          Run Benchmarks (Phase 9)
                                  │
                                  ▼
                            Pass benchmarks?
                                  │
                    ├─ NO ──→ Analyze failures → Add data → Retrain
                    │
                    └─ YES ──→ Deploy 🎉
```

---

## 🎯 Success Criteria

✅ **Training**: Perplexity < 15, coherent samples
✅ **Pruning**: < 10% perplexity increase, > 10% speed gain
✅ **Benchmarks**: Competitive with leaderboard scores
✅ **Deployment**: Model generates quality agent responses

---

## 📁 Quick Reference

| Script | Purpose | Time |
|--------|---------|------|
| `fine_tune_blackwell.py` | Main training | 20-30 min |
| `benchmark_model.py` | Quick eval | 5 min |
| `retune_model.py` | Recovery fine-tune | 10 min |
| `create_merge_script.py` | Merge LoRA | 10 min |
| `evaluate_finetuned.py` | Agent benchmarks | 30 min - 4 hours |

---

## 🚀 Ready to Start?

**Begin Phase 1**:
```bash
cd ~/agent-finetune
./run_training.sh
```

This will start training on your 3,797 Claude conversations and create a model optimized for agent tasks.
