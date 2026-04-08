#!/bin/bash

# Activate conda environment
echo "Activating conda environment..."
source ~/miniconda/bin/activate gemma-h100

# Set environment variables
export WANDB_PROJECT="agent-finetune"
export WANDB_RUN_NAME="training-$(date +%Y%m%d-%H%M%S)"

# Display configuration
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "WANDB_RUN_NAME: $WANDB_RUN_NAME"

# Clear GPU cache
echo "Clearing GPU cache..."
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"

# Run training with logging
echo "Starting training..."
python fine_tune_blackwell.py 2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log

echo "Training completed"
