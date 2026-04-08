#!/usr/bin/env python3
"""Quick fix to replace SFTTrainer with standard Trainer"""

import re

with open('fine_tune_blackwell.py', 'r') as f:
    content = f.read()

# Replace SFTTrainer import
content = content.replace(
    'from trl import SFTTrainer',
    '# from trl import SFTTrainer  # Using standard Trainer instead'
)

# Replace SFTTrainer with standard Trainer
old_trainer = '''try:
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
        packing=False,
    )
    print("✓ Trainer configured")'''

new_trainer = '''from transformers import Trainer

try:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        ),
    )
    print("✓ Trainer configured with standard Trainer")'''

content = content.replace(old_trainer, new_trainer)

with open('fine_tune_blackwell.py', 'w') as f:
    f.write(content)

print("✓ Fixed SFTTrainer -> standard Trainer")
