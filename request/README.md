# Request script

1. Each system will be tested with the same simulation, generated with something like the function in `generate_simulation.py`. The final simulation is not guaranteed to have the same overall request frequency and batch shape distribution, but it will be within the same ballpark and at least not significantly harder. jsons provided for convenience
2. `python make_requests.py` to run a simulation. Pre-loaded with a lightweight 1 minute sample -- in reality the simulation will be longer and most likely denser

---

## Evaluation Conversion

After running your simulation, convert results to evaluation format:

### Quick Start

```bash
# Convert simulation results to student outputs for evaluation
python3 convert_simulation_to_outputs.py batch_results_sample_py/simulation_summary.json student_outputs.jsonl
```

### Output Format

Each line in `student_outputs.jsonl`:
```jsonl
{"index": <prompt_index>, "output": "<model_response_text>"}
```

### Next Steps

See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) for complete evaluation workflow.