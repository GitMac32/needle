# Creative Problem-Solving Patterns Dataset

## Overview

An innovative wildcard dataset containing 1,000 structured examples of creative problem-solving approaches across 8 diverse domains. This dataset is specifically designed for training and fine-tuning AI systems on cross-domain reasoning, creative thinking patterns, and innovative solution generation.

## Key Features

- **Cross-Domain Coverage**: 8 domains including biomimicry, urban planning, software architecture, healthcare, education, sustainability, arts/creativity, and business innovation
- **Rich Structure**: Each example contains problem description, constraints, solution approach, key insights, analogical sources, outcomes, and transferability scores
- **Multiple Formats**: Available in JSONL (for streaming), CSV (for analysis), and pre-split train/validation/test sets
- **Transferability Scores**: Each example rated 1-10 on how applicable the pattern is to other domains
- **Ready for Fine-Tuning**: Includes instruction-formatted prompts for immediate use in model training

## Dataset Statistics

| Split | Examples | Percentage |
|-------|----------|------------|
| Training | 800 | 80% |
| Validation | 100 | 10% |
| Test | 100 | 10% |
| **Total** | **1,000** | **100%** |

### Domain Distribution

Each domain is represented with ~12-14% of the dataset:
- Biomimicry
- Urban Planning
- Software Architecture
- Healthcare
- Education
- Sustainability
- Arts & Creativity
- Business Innovation

## File Structure

```
creative_problem_solving_dataset/
├── README.md                          # This file
├── generate_dataset.py                # Dataset generation script
├── fine_tune.py                       # Fine-tuning demonstration script
├── creative_problem_solving.jsonl     # Complete dataset (JSONL)
├── creative_problem_solving.csv       # Complete dataset (CSV)
├── train.jsonl                        # Training split (800 examples)
├── validation.jsonl                   # Validation split (100 examples)
└── test.jsonl                         # Test split (100 examples)
```

## Data Schema

Each example contains:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (e.g., "CPS-00001") |
| `domain` | string | Domain of expertise |
| `problem_type` | string | Category of problem (optimization, system_design, etc.) |
| `problem_description` | string | Clear statement of the challenge |
| `constraint_set` | array | List of limitations or requirements |
| `solution_approach` | array | Step-by-step methodology |
| `key_insight` | string | The breakthrough moment or realization |
| `analogical_sources` | array | Other domains/sources that inspired the solution |
| `outcome` | string | Result of applying the solution |
| `transferability_score` | integer | Applicability to other domains (1-10) |
| `metadata` | object | Additional context (expert profile, year, complexity, validation status) |

## Usage Examples

### Loading the Dataset

```python
import json

# Load training data
with open('train.jsonl', 'r') as f:
    training_data = [json.loads(line) for line in f]

# Access an example
example = training_data[0]
print(f"Domain: {example['domain']}")
print(f"Problem: {example['problem_description']}")
print(f"Key Insight: {example['key_insight']}")
```

### Fine-Tuning a Language Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2')
model = AutoModelForCausalLM.from_pretrained('microsoft/phi-2')

# Prepare dataset (see fine_tune.py for full implementation)
# ...

# Configure training
training_args = TrainingArguments(
    output_dir='./cps_finetuned',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    evaluation_strategy='steps',
    eval_steps=50,
    save_steps=100,
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)
trainer.train()
```

## Innovative Use Cases

This wildcard dataset enables numerous research and application directions:

### 1. **Creative AI Assistants**
Fine-tune models to provide innovative solutions by drawing analogies across domains.

### 2. **Cross-Domain Reasoning Research**
Study how AI systems can transfer knowledge between seemingly unrelated fields.

### 3. **Educational Tools**
Build systems that teach creative problem-solving by showing patterns across domains.

### 4. **Innovation Consulting**
Create recommendation systems that suggest analogous solutions from other industries.

### 5. **Multi-Task Learning**
Train models on diverse problem types to improve generalization.

### 6. **Transfer Learning Experiments**
Test how well models transfer creative thinking skills between domains.

### 7. **Analogical Reasoning Benchmarks**
Use the transferability scores to evaluate model performance on cross-domain tasks.

### 8. **Constraint-Based Innovation**
Study how AI systems handle multiple constraints while generating creative solutions.

## Generation Process

The dataset was generated using `generate_dataset.py`, which:
1. Defines domain-specific knowledge bases with problems, experts, and analogies
2. Generates realistic constraints for each problem
3. Creates multi-step solution approaches based on best practices
4. Produces key insights demonstrating analogical thinking
5. Assigns transferability scores based on pattern generality
6. Splits data into train/validation/test sets

Re-run the script with different random seeds to generate variations.

## License

This dataset is released under **CC0 (Public Domain)** for maximum flexibility in research and commercial applications.

## Citation

If you use this dataset in your research, please consider citing:

```bibtex
@dataset{creative_problem_solving_2024,
  title = {Creative Problem-Solving Patterns Dataset},
  year = {2024},
  publisher = {Open Source Community},
  author = {AI-Assisted Generation}
}
```

## Getting Started

1. **Explore the data**: Open `creative_problem_solving.csv` in your favorite spreadsheet tool
2. **Run the demo**: Execute `python3 fine_tune.py` to see sample training examples
3. **Generate variations**: Run `python3 generate_dataset.py` to create new dataset instances
4. **Start training**: Use the provided scripts as a starting point for your fine-tuning pipeline

## Future Enhancements

Potential improvements for future versions:
- Expand to 10,000+ examples
- Add more granular domain subcategories
- Include visual representations of solution approaches
- Create evaluation benchmarks for creative thinking
- Add multilingual translations
- Include expert validation ratings
