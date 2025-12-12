#!/usr/bin/env python3
"""
Convert simulation_summary.json to student_outputs.jsonl format for evaluation.

This script extracts the model responses from your simulation results and formats them
for the evaluation pipeline.

Usage:
    python3 convert_simulation_to_outputs.py simulation_summary.json student_outputs.jsonl

Or with default paths:
    python3 convert_simulation_to_outputs.py
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any


def extract_student_outputs(simulation_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract student outputs from simulation_summary.json.

    Args:
        simulation_summary: The loaded simulation summary dictionary

    Returns:
        List of dictionaries with format: {"index": int, "output": str}
    """
    student_outputs = []

    for batch in simulation_summary.get('results', []):
        # Only process successful batches
        if batch.get('status_code') != 200:
            print(f"⚠️  Skipping batch {batch.get('batch_id')} - status code: {batch.get('status_code')}")
            continue

        response = batch.get('response')
        if not response:
            print(f"⚠️  Skipping batch {batch.get('batch_id')} - no response")
            continue

        prompt_idxs = batch.get('prompt_idxs', [])
        choices = response.get('choices', [])

        if not prompt_idxs:
            print(f"⚠️  Skipping batch {batch.get('batch_id')} - no prompt_idxs")
            continue

        # Match each choice with its corresponding prompt_idx
        for choice in choices:
            choice_index = choice.get('index')
            text = choice.get('text', '')

            # Map choice index to prompt index
            if choice_index is not None and choice_index < len(prompt_idxs):
                prompt_idx = prompt_idxs[choice_index]
                student_outputs.append({
                    'index': prompt_idx,
                    'output': text
                })
            else:
                print(f"⚠️  Warning: choice index {choice_index} out of range for batch {batch.get('batch_id')}")

    return student_outputs


def save_jsonl(data: List[Dict[str, Any]], output_path: Path) -> None:
    """Save data as JSONL format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    # Parse command line arguments
    if len(sys.argv) >= 2:
        input_path = Path(sys.argv[1])
    else:
        input_path = Path('batch_results_sample_py/simulation_summary.json')

    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = Path('student_outputs.jsonl')

    # Check if input file exists
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        print(f"\nUsage: python3 {sys.argv[0]} [simulation_summary.json] [student_outputs.jsonl]")
        sys.exit(1)

    # Load simulation summary
    print(f"📖 Loading simulation summary from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        simulation_summary = json.load(f)

    # Extract statistics
    total_batches = simulation_summary.get('total_batches', 0)
    total_prompts = simulation_summary.get('total_prompts', 0)
    successful_batches = simulation_summary.get('successful_batches', 0)
    failed_batches = simulation_summary.get('failed_batches', 0)

    print(f"\n📊 Simulation Statistics:")
    print(f"   Total batches: {total_batches}")
    print(f"   Total prompts: {total_prompts}")
    print(f"   Successful batches: {successful_batches}")
    print(f"   Failed batches: {failed_batches}")

    # Extract student outputs
    print(f"\n🔄 Extracting student outputs...")
    student_outputs = extract_student_outputs(simulation_summary)

    # Sort by index for easier validation
    student_outputs.sort(key=lambda x: x['index'])

    # Save to JSONL
    print(f"💾 Saving to: {output_path}")
    save_jsonl(student_outputs, output_path)

    # Print summary
    print(f"\n✅ Successfully converted!")
    print(f"   Extracted {len(student_outputs)} outputs from {total_prompts} prompts")
    print(f"   Coverage: {len(student_outputs)/total_prompts*100:.1f}%")

    # Check for missing indices
    extracted_indices = set(item['index'] for item in student_outputs)
    expected_indices = set(range(total_prompts))
    missing_indices = expected_indices - extracted_indices

    if missing_indices:
        print(f"\n⚠️  Warning: {len(missing_indices)} prompts have no output")
        if len(missing_indices) <= 10:
            print(f"   Missing indices: {sorted(missing_indices)}")
        else:
            print(f"   Missing indices (first 10): {sorted(list(missing_indices))[:10]}...")

    # Show sample outputs
    print(f"\n📝 Sample outputs:")
    for i, item in enumerate(student_outputs[:3]):
        print(f"\n   [{i+1}] Index {item['index']}:")
        output_preview = item['output'][:100].replace('\n', ' ')
        if len(item['output']) > 100:
            output_preview += "..."
        print(f"       {output_preview}")

    print(f"\n✨ Done! You can now run evaluation with this file.")


if __name__ == "__main__":
    main()
