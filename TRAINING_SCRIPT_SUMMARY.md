# Gemma-4-21B-A4B-IT-REAP Training Script Summary

## File Location
`~/agent-finetune/fine_tune_blackwell.py`

## Key Features Implemented

### 1. Model Loading with 4-bit Quantization
- Uses Unsloth `FastLanguageModel.from_pretrained()`
- Model: `0xSero/gemma-4-21b-a4b-it-REAP`
- Max sequence length: 2048
- 4-bit quantization enabled
- Auto-detects dtype (bf16/fp16)

### 2. LoRA Configuration
- Rank (r): 16
- Alpha: 16
- Dropout: 0
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Uses gradient checkpointing (unsloth optimized)
- Random state: 3407

### 3. Training Parameters
- Batch size per device: 2
- Gradient accumulation steps: 4
- Effective batch size: 8
- Learning rate: 2e-4
- Warmup steps: 10
- Max steps: 100 (easily adjustable)
- Optimizer: adamw_8bit
- Weight decay: 0.01
- LR scheduler: cosine
- Precision: Auto-detects bf16 or falls back to fp16

### 4. Checkpointing & Saving
- Checkpoint directory: `./checkpoints/`
- Output directory: `./outputs/`
- Save steps: Every 25 steps
- Save total limit: 3 (keeps last 3 checkpoints)
- Logging steps: Every 5 steps
- TensorBoard logging enabled

### 5. Training Components
- Uses `SFTTrainer` from TRL library
- Data collator with padding to multiple of 8
- Gemma-3 chat template for tokenizer
- Supports both instruction and ShareGPT formats

### 6. Error Handling
- Try-except blocks around all major operations
- Clear error messages with context
- Graceful handling of dataset loading issues
- Warning messages for non-critical failures

### 7. Evaluation Metrics
- Tracks training loss
- Displays model parameter count
- Shows trainable parameters
- Prints training history
- Logs to TensorBoard

### 8. Model Saving
- Saves LoRA adapters to `./outputs/`
- Saves tokenizer separately
- Optional code for merging adapters (commented out)
- Creates output directories automatically

### 9. Inference Example
- Loads fine-tuned model for testing
- Demonstrates inference with sample prompt
- Shows how to use chat template
- Includes generation parameters

## Usage

### Basic Usage
```bash
cd ~/agent-finetune
python fine_tune_blackwell.py
```

### Customization
Edit the configuration section at the top of the script:
- Change `MAX_STEPS` for longer/shorter training
- Adjust `BATCH_SIZE_PER_DEVICE` based on GPU memory
- Modify `LEARNING_RATE` for different convergence behavior
- Update dataset loading code to use your data

### Dataset Integration
Replace the placeholder dataset code:
```python
dataset = load_dataset("yahma/alpaca-cleaned", split="train")
```

With your actual dataset:
```python
# For ShareGPT format
dataset = load_dataset("sharegpt", data_files="your_data.jsonl", split="train")
dataset = standardize_sharegpt(dataset)

# For instruction format
dataset = load_dataset("your_dataset", split="train")
```

## Output Files
- `./outputs/` - Final LoRA adapters and tokenizer
- `./checkpoints/` - Intermediate training checkpoints
- TensorBoard logs - Training metrics and curves

## Requirements
- unsloth
- transformers
- trl
- torch
- datasets
- CUDA-capable GPU (recommended: H100/A100)

## Notes
- Script automatically detects and uses bf16 if supported
- Gradient checkpointing enabled for memory efficiency
- 4-bit quantization significantly reduces memory usage
- Effective batch size = 2 × 4 = 8 with current settings
