#!/usr/bin/env python3
"""
Fine-tuning script for Creative Problem-Solving Dataset

This script demonstrates how to fine-tune a language model on the 
Creative Problem-Solving Patterns dataset using various approaches.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

# Try to import optional dependencies
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Note: PyTorch not available. Showing dataset preparation only.")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Note: Transformers not available. Showing dataset preparation only.")


class CreativeProblemSolvingDataset(Dataset):
    """PyTorch Dataset for Creative Problem-Solving examples."""
    
    def __init__(self, data_path: str, tokenizer: Any, max_length: int = 512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load JSONL data
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line.strip())
                self.examples.append(example)
        
        print(f"Loaded {len(self.examples)} examples from {data_path}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format the example as instruction-response pair
        prompt = self._format_prompt(example)
        
        # Tokenize
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': encoded['input_ids'].squeeze(0).clone()
        }
    
    def _format_prompt(self, example: Dict[str, Any]) -> str:
        """Format example as instruction-following prompt."""
        template = """### Instruction:
Analyze this creative problem-solving case and explain the key insights.

### Context:
Domain: {domain}
Problem Type: {problem_type}
Problem: {problem}
Constraints: {constraints}

### Task:
Describe the solution approach, key insight, and outcomes. Explain how analogical thinking was applied.

### Response:
{response}""".format(
            domain=example['domain'],
            problem_type=example['problem_type'],
            problem=example['problem_description'],
            constraints=', '.join(example['constraint_set']),
            response=self._generate_response(example)
        )
        
        return template
    
    def _generate_response(self, example: Dict[str, Any]) -> str:
        """Generate response text from example."""
        response_parts = [
            f"**Solution Approach:**",
            '\n'.join([f"- {step}" for step in example['solution_approach']]),
            f"\n\n**Key Insight:** {example['key_insight']}",
            f"\n\n**Analogical Sources:** {', '.join(example['analogical_sources'])}",
            f"\n\n**Outcome:** {example['outcome']}",
            f"\n\n**Transferability Score:** {example['transferability_score']}/10",
            f"\n\nThis case from the {example['domain']} domain demonstrates how cross-domain thinking can lead to innovative solutions."
        ]
        return '\n'.join(response_parts)


def prepare_finetuning_data(
    train_path: str,
    val_path: str,
    output_dir: str = "prepared_data",
    model_name: str = "meta-llama/Llama-2-7b-hf"
):
    """Prepare data for fine-tuning."""
    
    print("="*60)
    print("PREPARING FINE-TUNING DATA")
    print("="*60)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if not TRANSFORMERS_AVAILABLE:
        print("\nTransformers library not available.")
        print("Showing data format preview instead...\n")
        
        # Preview formatted examples
        with open(train_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= 2:
                    break
                example = json.loads(line)
                print(f"\n{'='*60}")
                print(f"Example {i+1}:")
                print(f"{'='*60}")
                print(f"Domain: {example['domain']}")
                print(f"Problem: {example['problem_description']}")
                print(f"Key Insight: {example['key_insight']}")
                print(f"Transferability: {example['transferability_score']}/10")
        
        return None
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add special tokens if needed
    special_tokens = ['### Instruction:', '### Context:', '### Task:', '### Response:']
    tokenizer.add_tokens(special_tokens)
    
    # Prepare datasets
    print("\nPreparing training dataset...")
    train_dataset = CreativeProblemSolvingDataset(train_path, tokenizer)
    
    print("Preparing validation dataset...")
    val_dataset = CreativeProblemSolvingDataset(val_path, tokenizer)
    
    print(f"\nData preparation complete!")
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    
    return {
        'train': train_dataset,
        'val': val_dataset,
        'tokenizer': tokenizer
    }


def create_training_config():
    """Create training configuration."""
    
    config = {
        # Model settings
        'model_name': 'meta-llama/Llama-2-7b-hf',
        'max_length': 512,
        
        # Training hyperparameters
        'batch_size': 4,
        'gradient_accumulation_steps': 4,
        'learning_rate': 2e-5,
        'num_train_epochs': 3,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        
        # Optimization
        'fp16': True,
        'gradient_checkpointing': True,
        
        # Logging
        'logging_steps': 10,
        'save_steps': 100,
        'eval_steps': 50,
        
        # Output
        'output_dir': './cps_finetuned_model',
        'save_total_limit': 3
    }
    
    return config


def show_training_preview():
    """Show what training would look like."""
    
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION PREVIEW")
    print("="*60)
    
    config = create_training_config()
    
    print(f"""
Model: {config['model_name']}
Max Sequence Length: {config['max_length']}

Training Hyperparameters:
  - Batch Size: {config['batch_size']}
  - Gradient Accumulation: {config['gradient_accumulation_steps']}
  - Learning Rate: {config['learning_rate']}
  - Epochs: {config['num_train_epochs']}
  - Weight Decay: {config['weight_decay']}
  - Warmup Ratio: {config['warmup_ratio']}

Optimization:
  - Mixed Precision (FP16): {config['fp16']}
  - Gradient Checkpointing: {config['gradient_checkpointing']}

Training Schedule:
  - Logging every {config['logging_steps']} steps
  - Save checkpoint every {config['save_steps']} steps
  - Evaluate every {config['eval_steps']} steps

Output:
  - Directory: {config['output_dir']}
  - Max Checkpoints Kept: {config['save_total_limit']}
""")


def generate_training_examples(num_examples: int = 5):
    """Generate sample training prompts for demonstration."""
    
    dataset_dir = Path(__file__).parent
    
    print("\n" + "="*60)
    print("SAMPLE TRAINING EXAMPLES")
    print("="*60)
    
    # Load some examples
    with open(dataset_dir / 'train.jsonl', 'r') as f:
        examples = [json.loads(f.readline()) for _ in range(num_examples)]
    
    for i, ex in enumerate(examples, 1):
        print(f"\n{'='*60}")
        print(f"SAMPLE {i}")
        print(f"{'='*60}")
        print(f"\nDomain: {ex['domain']}")
        print(f"Problem Type: {ex['problem_type']}")
        print(f"Problem: {ex['problem_description']}")
        print(f"\nConstraints:")
        for c in ex['constraint_set']:
            print(f"  • {c}")
        print(f"\nSolution Approach:")
        for step in ex['solution_approach']:
            print(f"  → {step}")
        print(f"\nKey Insight: {ex['key_insight']}")
        print(f"Analogies Used: {', '.join(ex['analogical_sources'])}")
        print(f"Outcome: {ex['outcome']}")
        print(f"Transferability Score: {ex['transferability_score']}/10")


def main():
    """Main entry point."""
    
    dataset_dir = Path(__file__).parent
    
    print("\n" + "="*60)
    print("CREATIVE PROBLEM-SOLVING DATASET - FINE-TUNING SCRIPT")
    print("="*60)
    
    # Show sample training examples
    generate_training_examples(3)
    
    # Show training configuration
    show_training_preview()
    
    # Attempt data preparation
    train_path = dataset_dir / 'train.jsonl'
    val_path = dataset_dir / 'validation.jsonl'
    
    if train_path.exists() and val_path.exists():
        prepared_data = prepare_finetuning_data(
            str(train_path),
            str(val_path),
            model_name='microsoft/phi-2'  # Smaller model for demo
        )
        
        if prepared_data is not None and TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE:
            print("\n" + "="*60)
            print("READY FOR TRAINING")
            print("="*60)
            print("""
To start fine-tuning, run:

```python
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

# Load model
model = AutoModelForCausalLM.from_pretrained(
    'microsoft/phi-2',
    trust_remote_code=True
)

# Configure training
training_args = TrainingArguments(
    output_dir='./cps_finetuned',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy='steps',
    eval_steps=50,
    save_steps=100,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=prepared_data['train'],
    eval_dataset=prepared_data['val'],
    tokenizer=prepared_data['tokenizer']
)

# Start training
trainer.train()
```
""")
    else:
        print("\nError: Dataset files not found. Run generate_dataset.py first.")


if __name__ == "__main__":
    main()
