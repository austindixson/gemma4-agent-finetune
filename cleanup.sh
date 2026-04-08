#!/bin/bash
# Auto-cleanup to prevent disk full issues
rm -rf /tmp/claude-1002/-workspace/*/tasks/*.output 2>/dev/null
rm -rf ~/agent-finetune/checkpoints/checkpoint-* 2>/dev/null
ls -t ~/agent-finetune/training_*.log 2>/dev/null | tail -n +4 | xargs rm -f 2>/dev/null
echo "✓ Cleanup complete"
