#!/bin/bash
echo "=== DISK CLEANUP REPORT ==="
echo ""
echo "Disk usage:"
df -h /tmp | head -2
echo ""
echo "Largest directories:"
du -sh ~/{.hf_home,.cache,.datasets_cache,agent-finetune} 2>/dev/null | sort -h
echo ""
echo "Training files:"
du -sh ~/agent-finetune/{outputs,checkpoints,agent-dataset-unsloth} 2>/dev/null
echo ""
echo "Temp files:"
find /tmp/claude-100* -name "*.output" 2>/dev/null | wc -l
echo "Training logs:"
ls ~/agent-finetune/training_*.log 2>/dev/null | wc -l
