# Evaluation Guide: From Simulation to Results

This guide explains how to evaluate your inference system simulation results.

## Overview

Your simulation ran **300 prompts** across **98 batches**. The evaluation process converts your simulation results into a standardized format and grades them against ground truth answers.

## Step-by-Step Process

### Step 1: Convert Simulation Results to Student Outputs

Use the provided script to extract model responses from your simulation:

```bash
cd /Users/ty_cao/Downloads/inference-system/EfficientInference/request

# Convert simulation_summary.json to student_outputs.jsonl
python3 convert_simulation_to_outputs.py batch_results_sample_py/simulation_summary.json student_outputs.jsonl
```

**What this does:**
- Extracts model responses from successful batches (status_code == 200)
- Maps each response to its corresponding prompt index
- Saves in JSONL format: `{"index": <prompt_idx>, "output": "<model_response>"}`
- Shows coverage statistics (how many prompts have responses)

**Output:** `student_outputs.jsonl` with format:
```jsonl
{"index": 3, "output": "{\"name\": \"submit_paths\", \"arguments\": {\"paths\": [[0, 2, 4, 6]], \"weights\": [1040]}}"}
{"index": 4, "output": "{\"name\": \"submit_paths\", \"arguments\": {\"paths\": [[0, 6, 8, 3, 9]], \"weights\": [33]}}"}
```

### Step 2: Generate the Full Combined Dataset

Your simulation used 300 prompts from HuggingFace. You need to create the corresponding ground truth dataset:

```bash
cd /Users/ty_cao/Downloads/inference-system/EfficientInference/eval

python3 << 'EOF'
import json
from datasets import load_dataset

def process_graph_entry(ex, idx):
    return {
        "index": idx,
        "task": "graph",
        "prompt": ex.get('prompt'),
        "gold_answer": ex.get('solution'),
        "meta": {
            "graph_params": ex.get('graph_params'),
            "edges": ex.get('edges'),
            "original_id": ex.get('id')
        }
    }

def process_infobench_entry(ex, idx):
    raw_instruction = ex.get('instruction', '') or ''
    context = ex.get('input', '') or ''
    full_prompt = f"Instruction: {raw_instruction}\nQuestion: {context}\nGeneration:" if context else f"Instruction: {raw_instruction}\nQuestion: \nGeneration:"

    return {
        "index": idx,
        "task": "infobench",
        "prompt": full_prompt,
        "gold_answer": None,
        "meta": {
            "decomposed_questions": ex.get('decomposed_questions'),
            "category": ex.get('category'),
            "subset": ex.get('subset'),
            "original_id": ex.get('id')
        }
    }

def process_mmlu_med_entry(ex, idx):
    q = ex.get("question", "") or ""
    choices = ex.get("choices", []) or []
    formatted_choices = "\n".join([f"{chr(65+i)}. {choices[i]}" for i in range(len(choices))])

    subject = ex.get("subject", "")
    prompt_text = f"The following is a multiple choice question (with answers) about {subject.replace('_', ' ')}.  Output the answer in the format of \"The answer is (X)\" at the end.\n\nQuestion: {q}\n Options:\n{formatted_choices}\nAnswer:"

    answer_idx = ex.get("answer")
    gold_letter = chr(65 + answer_idx) if answer_idx is not None else None

    return {
        "index": idx,
        "task": "mmlu_med",
        "prompt": prompt_text,
        "gold_answer": gold_letter,
        "meta": {
            "subject": ex.get("subject"),
            "answer_index": answer_idx,
            "raw_choices": choices,
            "original_id": None
        }
    }

print("Loading datasets from HuggingFace...")
graph_ds = load_dataset("vashistht/11763_datasets", 'graph_dev', split='dev_test')
infobench_ds = load_dataset("vashistht/11763_datasets", 'infobench', split='dev_test')
mmlu_ds = load_dataset("vashistht/11763_datasets", 'mmlu_med', split='dev_test')

combined_data = []
global_idx = 0

print("Processing Graph (indices 0-99)...")
for ex in graph_ds:
    combined_data.append(process_graph_entry(ex, global_idx))
    global_idx += 1

print("Processing InfoBench (indices 100-199)...")
for ex in infobench_ds:
    combined_data.append(process_infobench_entry(ex, global_idx))
    global_idx += 1

print("Processing MMLU Med (indices 200-299)...")
for ex in mmlu_ds:
    combined_data.append(process_mmlu_med_entry(ex, global_idx))
    global_idx += 1

out_path = "combined_dataset_full.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for item in combined_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✓ Wrote {len(combined_data)} items to {out_path}")
print("\nDataset breakdown:")
print(f"  Graph:     indices 0-99    (100 examples)")
print(f"  InfoBench: indices 100-199 (100 examples)")
print(f"  MMLU Med:  indices 200-299 (100 examples)")
EOF
```

**Output:** `combined_dataset_full.jsonl` (300 lines)

### Step 3: Run Evaluation

Copy your student outputs to the eval directory and run the evaluation:

```bash
cd /Users/ty_cao/Downloads/inference-system/EfficientInference/eval

# Copy student outputs
cp ../request/student_outputs.jsonl .

# Set your OpenAI API key (required for InfoBench evaluation)
export OPENAI_API_KEY="your-openai-api-key"

# Run evaluation
python3 eval.py
```

**You'll need to edit eval.py first** to update these paths:
```python
# Around line 247-252 in eval.py
HIDDEN_TEST_PATH = "combined_dataset_full.jsonl"  # Change from combined_dataset.jsonl
STUDENT_OUTPUT_PATH = "student_outputs.jsonl"
OUTPUT_DIR = "./eval_results"
STUDENT_ID = "my_simulation"  # Change to your preferred ID
EVAL_MODEL = "gpt-5-nano-2025-08-07"
```

### Step 4: Review Results

The evaluation produces two files in `eval_results/`:

**1. `{STUDENT_ID}_results.jsonl`** - Detailed per-question results:
```jsonl
{
  "index": 3,
  "task": "graph",
  "prompt": "You are given a directed graph...",
  "student_output": "{\"name\": \"submit_paths\", ...}",
  "gold_answer": {"paths": [{"path": [0, 8, 9], "weight": 25}]},
  "score": 1.0,
  "eval_details": {"parsed_paths": {...}, "matches": 1, "total": 1}
}
```

**2. `{STUDENT_ID}_metrics.json`** - Aggregate performance metrics:
```json
{
  "student_id": "my_simulation",
  "total_examples": 69,
  "task_metrics": {
    "mmlu_med": {
      "count": 20,
      "accuracy": 0.65,
      "total_score": 13.0
    },
    "graph": {
      "count": 30,
      "accuracy": 0.80,
      "total_score": 24.0
    },
    "infobench": {
      "count": 19,
      "accuracy": 0.75,
      "total_score": 14.25
    }
  },
  "overall_accuracy": 0.74
}
```

## Understanding Your Results

### Current Simulation Performance

From your `simulation_summary.json`:
- **Total batches:** 98
- **Successful batches:** 29 (29.6% success rate)
- **Failed batches:** 69 (70.4% failure rate)
- **Coverage:** Only 69/300 prompts (23%) have outputs

**This low success rate indicates your inference system may have issues with:**
- Request timeouts
- Server capacity/concurrency limits
- Error handling
- Resource constraints

### Evaluation Metrics

The evaluation system measures:

1. **MMLU (Multiple Choice)**: Exact match scoring
   - Looks for "The answer is (X)" or `\boxed{X}` format
   - Score: 1.0 if correct, 0.0 otherwise

2. **Graph (Shortest Path)**: Partial credit scoring
   - Parses JSON or function call format
   - Score = (correct paths) / (total expected paths)
   - Must match both node sequence AND weight

3. **InfoBench (Open-ended)**: LLM-as-judge scoring
   - Uses GPT model to evaluate quality
   - Asks decomposed yes/no questions
   - Score = ratio of "YES" answers

## Tips for Better Results

1. **Improve success rate**: Debug why 70% of batches are failing
2. **Check prompt formats**: Ensure responses match expected formats
3. **Monitor resource usage**: Adjust `max_concurrent_requests` in make_requests.py
4. **Validate outputs**: Check that model generates answers in correct format

## Quick Reference

```bash
# Full workflow
cd /Users/ty_cao/Downloads/inference-system/EfficientInference

# 1. Convert simulation to outputs
cd request
python3 convert_simulation_to_outputs.py batch_results_sample_py/simulation_summary.json student_outputs.jsonl

# 2. Generate ground truth dataset
cd ../eval
python3 generate_combined_dataset.py  # (use script from Step 2)

# 3. Run evaluation
export OPENAI_API_KEY="sk-..."
python3 eval.py

# 4. View results
cat eval_results/my_simulation_metrics.json
```

## File Locations

- **Simulation input:** `request/batch_arrivals.json`
- **Simulation output:** `request/batch_results_sample_py/simulation_summary.json`
- **Student outputs:** `request/student_outputs.jsonl`
- **Ground truth:** `eval/combined_dataset_full.jsonl`
- **Evaluation results:** `eval/eval_results/`
