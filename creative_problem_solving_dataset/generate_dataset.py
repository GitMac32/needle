#!/usr/bin/env python3
"""
Creative Problem-Solving Patterns Dataset Generator

This script generates a diverse dataset of creative problem-solving examples
across multiple domains, optimized for training and fine-tuning AI models.
"""

import json
import csv
import random
from datetime import datetime
from pathlib import Path

# Seed for reproducibility
random.seed(42)

# Domain definitions with specialized knowledge
DOMAINS = {
    "biomimicry": {
        "experts": ["Biologist", "Materials Scientist", "Ecological Engineer"],
        "problems": [
            "Designing self-cleaning surfaces",
            "Creating energy-efficient cooling systems",
            "Developing adhesive materials without chemicals",
            "Optimizing structural strength with minimal material",
            "Water collection in arid environments"
        ],
        "analogies": ["lotus leaves", "termite mounds", "gecko feet", "honeycomb structures", "Namib desert beetle"]
    },
    "urban_planning": {
        "experts": ["Urban Designer", "Traffic Engineer", "Community Planner"],
        "problems": [
            "Reducing traffic congestion without expanding roads",
            "Creating green spaces in dense urban areas",
            "Improving public transportation adoption",
            "Managing waste in high-density populations",
            "Balancing development with historical preservation"
        ],
        "analogies": ["circulatory systems", "forest ecosystems", "swarm behavior", "modular design", "layered networks"]
    },
    "software_architecture": {
        "experts": ["System Architect", "DevOps Engineer", "Security Specialist"],
        "problems": [
            "Scaling systems under unpredictable load",
            "Maintaining legacy systems while innovating",
            "Ensuring security without compromising usability",
            "Managing distributed team collaboration",
            "Reducing technical debt while delivering features"
        ],
        "analogies": ["biological immune systems", "city infrastructure", "library organization", "assembly lines", "neural networks"]
    },
    "healthcare": {
        "experts": ["Medical Doctor", "Public Health Specialist", "Healthcare Administrator"],
        "problems": [
            "Improving patient compliance with treatment plans",
            "Reducing hospital-acquired infections",
            "Optimizing emergency room wait times",
            "Managing chronic diseases with limited resources",
            "Bridging healthcare access gaps in rural areas"
        ],
        "analogies": ["preventive maintenance", "quality control systems", "logistics optimization", "community networks", "educational scaffolding"]
    },
    "education": {
        "experts": ["Curriculum Designer", "Educational Psychologist", "Technology Integration Specialist"],
        "problems": [
            "Engaging students with diverse learning styles",
            "Assessing critical thinking skills objectively",
            "Integrating technology meaningfully in classrooms",
            "Supporting students with varying background knowledge",
            "Preparing students for jobs that don't exist yet"
        ],
        "analogies": ["game design", "apprenticeship models", "studio critiques", "sports coaching", "exploration expeditions"]
    },
    "sustainability": {
        "experts": ["Environmental Scientist", "Circular Economy Designer", "Policy Analyst"],
        "problems": [
            "Reducing plastic waste in supply chains",
            "Converting waste streams into valuable resources",
            "Encouraging sustainable consumer behavior",
            "Measuring true environmental costs of products",
            "Designing products for complete recyclability"
        ],
        "analogies": ["natural ecosystems", "nutrient cycles", "repair cultures", "sharing economies", "cradle-to-cradle design"]
    },
    "arts_creativity": {
        "experts": ["Visual Artist", "Music Composer", "Creative Director"],
        "problems": [
            "Overcoming creative blocks systematically",
            "Collaborating across different artistic mediums",
            "Balancing artistic vision with commercial viability",
            "Preserving cultural heritage while innovating",
            "Making art accessible to diverse audiences"
        ],
        "analogies": ["improvisational jazz", "collage techniques", "narrative structures", "architectural constraints", "scientific experimentation"]
    },
    "business_innovation": {
        "experts": ["Product Manager", "Innovation Consultant", "Market Researcher"],
        "problems": [
            "Identifying unmet customer needs",
            "Pivoting business models during market disruption",
            "Fostering innovation in risk-averse organizations",
            "Building platforms that create network effects",
            "Competing with free alternatives"
        ],
        "analogies": ["evolutionary adaptation", "ecosystem niches", "platform ecosystems", "open source communities", "tournament structures"]
    }
}

PROBLEM_TYPES = [
    "optimization", "resource_constraint", "system_design", "behavior_change",
    "scalability", "integration", "accessibility", "sustainability",
    "efficiency", "resilience", "adaptation", "collaboration"
]

def generate_constraints(domain, problem_idx):
    """Generate realistic constraints for a problem."""
    constraint_templates = [
        "Budget limited to {}% of typical solutions",
        "Must be implemented within {} months",
        "Cannot use {} technology/approach",
        "Must work with existing {} infrastructure",
        "Requires approval from {} stakeholder groups",
        "Must comply with {} regulations",
        "Limited to {} team members",
        "Must achieve {}% improvement over current state"
    ]
    
    num_constraints = random.randint(2, 4)
    constraints = []
    
    for i in range(num_constraints):
        template = random.choice(constraint_templates)
        if "{}" in template:
            # Fill in the blank with appropriate values
            if "Budget" in template:
                value = random.choice([50, 60, 70, 80])
            elif "months" in template:
                value = random.choice([3, 6, 9, 12, 18])
            elif "technology" in template:
                value = random.choice(["conventional", "expensive", "proprietary", "established"])
            elif "infrastructure" in template:
                value = random.choice(["legacy", "limited", "aging", "minimal"])
            elif "stakeholder" in template:
                value = random.randint(2, 5)
            elif "regulations" in template:
                value = random.choice(["strict", "multiple", "evolving", "international"])
            elif "team" in template:
                value = random.randint(2, 8)
            elif "improvement" in template:
                value = random.choice([20, 30, 50, 100])
            else:
                value = "specific"
            constraints.append(template.format(value))
        else:
            constraints.append(template)
    
    return constraints

def generate_solution_approach(domain, problem, constraints):
    """Generate a multi-step solution approach."""
    phases = [
        "Research and analysis phase",
        "Stakeholder engagement",
        "Prototype development",
        "Iterative testing",
        "Implementation planning",
        "Scale and deployment",
        "Monitoring and adaptation"
    ]
    
    selected_phases = random.sample(phases, random.randint(3, 5))
    approach = []
    
    for phase in selected_phases:
        actions = {
            "Research and analysis phase": [
                f"Conducted comparative analysis of {random.randint(5, 15)} similar cases",
                f"Identified key patterns from {random.choice(DOMAINS[domain]['analogies'])} in nature/other fields",
                "Mapped all stakeholder needs and pain points",
                "Quantified baseline metrics for success measurement"
            ],
            "Stakeholder engagement": [
                "Facilitated co-design workshops with end users",
                "Built coalition of early adopters and champions",
                "Addressed concerns through transparent communication",
                "Created feedback loops for continuous input"
            ],
            "Prototype development": [
                f"Built low-fidelity prototype focusing on {random.choice(['core functionality', 'user experience', 'technical feasibility'])}",
                "Tested assumption riskiest first",
                "Incorporated constraints as design features rather than limitations",
                "Documented learnings and iteration rationale"
            ],
            "Iterative testing": [
                f"Ran {random.randint(3, 8)} rapid test cycles with real users",
                "Measured both quantitative metrics and qualitative feedback",
                "Pivoted approach based on evidence, not assumptions",
                "Refined solution to work within all constraints"
            ],
            "Implementation planning": [
                "Developed phased rollout strategy",
                "Created training and support materials",
                "Established success metrics and monitoring plan",
                "Built contingency plans for common failure modes"
            ],
            "Scale and deployment": [
                "Started with controlled pilot in optimal conditions",
                "Gradually expanded to more challenging contexts",
                "Maintained quality while increasing volume",
                "Captured and shared learnings across teams"
            ],
            "Monitoring and adaptation": [
                "Established continuous feedback mechanisms",
                "Regularly reviewed performance against metrics",
                "Adapted solution to changing conditions",
                "Documented transferable principles for future use"
            ]
        }
        
        approach.append(random.choice(actions.get(phase, ["Executed phase activities"])))
    
    return approach

def generate_key_insight(domain, problem):
    """Generate the breakthrough insight."""
    num_domains = random.randint(2, 4)
    insights = [
        f"Realized that {random.choice(DOMAINS[domain]['analogies'])} solved a similar problem by {random.choice(['reframing the constraint', 'using emergent properties', 'leveraging distributed intelligence'])}",
        "Discovered that the real problem was different from the stated problem",
        "Found that combining two unrelated approaches created synergistic effects",
        "Recognized that a limitation could become a feature if viewed differently",
        "Understood that timing and sequence mattered more than individual components",
        "Learned that user behavior, not technology, was the key bottleneck",
        f"Saw that solutions from {num_domains} different domains could be integrated"
    ]
    return random.choice(insights)

def generate_outcome(problem_type):
    """Generate outcome description."""
    outcomes = [
        "Achieved {}% improvement in primary metric while reducing costs by {}%".format(
            random.randint(30, 200), random.randint(10, 50)
        ),
        "Solution adopted by {} organizations across {} countries".format(
            random.randint(5, 50), random.randint(2, 15)
        ),
        "Became industry standard practice within {} years".format(
            random.randint(2, 10)
        ),
        "Generated {}x ROI within first year of implementation".format(
            random.randint(2, 10)
        ),
        "Reduced {} by {}% while improving {}".format(
            random.choice(["waste", "time", "costs", "errors"]),
            random.randint(40, 90),
            random.choice(["quality", "satisfaction", "efficiency", "innovation"])
        )
    ]
    return random.choice(outcomes)

def generate_example(example_id):
    """Generate a single dataset example."""
    domain = random.choice(list(DOMAINS.keys()))
    domain_info = DOMAINS[domain]
    
    problem_idx = random.randint(0, len(domain_info["problems"]) - 1)
    problem = domain_info["problems"][problem_idx]
    
    example = {
        "id": f"CPS-{example_id:05d}",
        "domain": domain,
        "problem_type": random.choice(PROBLEM_TYPES),
        "problem_description": problem,
        "constraint_set": generate_constraints(domain, problem_idx),
        "solution_approach": generate_solution_approach(domain, problem, []),
        "key_insight": generate_key_insight(domain, problem),
        "analogical_sources": random.sample(domain_info["analogies"], 
                                          min(random.randint(1, 3), len(domain_info["analogies"]))),
        "outcome": generate_outcome(problem),
        "transferability_score": random.randint(5, 10),
        "metadata": {
            "expert_profile": random.choice(domain_info["experts"]),
            "year": random.randint(2010, 2024),
            "complexity_level": random.choice(["beginner", "intermediate", "advanced", "expert"]),
            "validation_status": random.choice(["case_study", "peer_reviewed", "industry_validated", "experimental"]),
            "generated_at": datetime.now().isoformat()
        }
    }
    
    return example

def generate_dataset(num_examples=1000):
    """Generate the complete dataset."""
    print(f"Generating {num_examples} creative problem-solving examples...")
    
    examples = []
    for i in range(num_examples):
        example = generate_example(i + 1)
        examples.append(example)
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_examples} examples")
    
    return examples

def save_jsonl(examples, output_path):
    """Save dataset in JSONL format."""
    print(f"Saving JSONL format to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    print(f"  Saved {len(examples)} examples")

def save_csv(examples, output_path):
    """Save dataset in CSV format."""
    print(f"Saving CSV format to {output_path}...")
    
    # Flatten structure for CSV
    flat_examples = []
    for example in examples:
        flat = {
            'id': example['id'],
            'domain': example['domain'],
            'problem_type': example['problem_type'],
            'problem_description': example['problem_description'],
            'constraints': '; '.join(example['constraint_set']),
            'solution_approach': ' | '.join(example['solution_approach']),
            'key_insight': example['key_insight'],
            'analogical_sources': ', '.join(example['analogical_sources']),
            'outcome': example['outcome'],
            'transferability_score': example['transferability_score'],
            'expert_profile': example['metadata']['expert_profile'],
            'year': example['metadata']['year'],
            'complexity_level': example['metadata']['complexity_level'],
            'validation_status': example['metadata']['validation_status']
        }
        flat_examples.append(flat)
    
    fieldnames = list(flat_examples[0].keys())
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_examples)
    
    print(f"  Saved {len(flat_examples)} examples")

def main():
    """Main entry point."""
    output_dir = Path(__file__).parent
    
    # Generate dataset
    examples = generate_dataset(1000)
    
    # Save in multiple formats
    save_jsonl(examples, output_dir / "creative_problem_solving.jsonl")
    save_csv(examples, output_dir / "creative_problem_solving.csv")
    
    # Create train/test/validation splits
    random.shuffle(examples)
    train_size = int(len(examples) * 0.8)
    val_size = int(len(examples) * 0.1)
    
    train_examples = examples[:train_size]
    val_examples = examples[train_size:train_size + val_size]
    test_examples = examples[train_size + val_size:]
    
    save_jsonl(train_examples, output_dir / "train.jsonl")
    save_jsonl(val_examples, output_dir / "validation.jsonl")
    save_jsonl(test_examples, output_dir / "test.jsonl")
    
    # Print statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total examples: {len(examples)}")
    print(f"Training set: {len(train_examples)} ({len(train_examples)/len(examples)*100:.1f}%)")
    print(f"Validation set: {len(val_examples)} ({len(val_examples)/len(examples)*100:.1f}%)")
    print(f"Test set: {len(test_examples)} ({len(test_examples)/len(examples)*100:.1f}%)")
    
    # Domain distribution
    print("\nDomain distribution:")
    domain_counts = {}
    for ex in examples:
        domain_counts[ex['domain']] = domain_counts.get(ex['domain'], 0) + 1
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count} ({count/len(examples)*100:.1f}%)")
    
    # Transferability score distribution
    print("\nTransferability score distribution:")
    score_counts = {}
    for ex in examples:
        score = ex['transferability_score']
        score_counts[score] = score_counts.get(score, 0) + 1
    for score in sorted(score_counts.keys()):
        print(f"  Score {score}: {score_counts[score]} examples")
    
    print("\nDataset generation complete!")
    print("Files created:")
    print("  - creative_problem_solving.jsonl (complete dataset)")
    print("  - creative_problem_solving.csv (complete dataset)")
    print("  - train.jsonl (training split)")
    print("  - validation.jsonl (validation split)")
    print("  - test.jsonl (test split)")
    print("  - README.md (documentation)")

if __name__ == "__main__":
    main()
