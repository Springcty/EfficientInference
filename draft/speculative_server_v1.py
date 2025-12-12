import modal
import asyncio
import time
import json
import re
import math
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

# Modal configuration
ANDREW_ID = "tianyuca"
SYSTEM_ID = "2"

GPU_CONFIG = "A100-80GB:2"
TIMEOUT = 600

# Model configurations
MAIN_MODEL_ID = "Qwen/Qwen3-8B"
DRAFT_MODEL_ID = "Qwen/Qwen3-0.6B"

cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

flash_attn_release = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post1/"
    "flash_attn-2.7.1.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git", "clang")
    .uv_pip_install(
        "torch==2.6.0",
        "transformers==4.57.3",
        "accelerate==1.12.0",
        "sentencepiece==0.2.1",
        "protobuf==6.33.2",
        "huggingface_hub==0.36.0",
        "setuptools==69.0.3",
        "wheel==0.45.1",
        "ninja==1.11.1.4",
        "packaging==25.0",
        "bitsandbytes==0.48.2",
        "fastapi[standard]",
        flash_attn_release,
    )
)

app = modal.App(f"{ANDREW_ID}-{SYSTEM_ID}")

class TaskRouter:
    """Classifies tasks and estimates complexity."""
    TASK_PATTERNS = {
        'algorithmic': [
            r'directed graph', 
            r'shortest path', 
            r'source.*target', 
            r'adjacency',
            r'edge.*weight',
            r'node.*labeled',
            r'submit_paths',
            r'top-?\d+.*path'
        ],
        'mmlu': [
            r'multiple choice', 
            r'The answer is \([A-D]\)', 
            r'Options:\s*\n\s*A\.',
            r'college.*medicine',
            r'professional.*medicine',
            r'\([A-D]\)\s+\w',
        ],
    }
    
    @classmethod
    def classify_task(cls, prompt: str) -> str:
        prompt_lower = prompt.lower()
        for p in cls.TASK_PATTERNS['algorithmic']:
            if re.search(p, prompt_lower): return 'algorithmic'
        for p in cls.TASK_PATTERNS['mmlu']:
            if re.search(p, prompt, re.IGNORECASE): return 'mmlu'
        return 'infobench'

import heapq
class TaskHandlers:
    """Handles prompt formatting and post-processing."""
    @staticmethod
    def get_handler(task_type: str):
        if task_type == 'algorithmic': return TaskHandlers.AlgorithmicHandler
        if task_type == 'mmlu': return TaskHandlers.MMLUHandler
        return TaskHandlers.InfobenchHandler
    
    class AlgorithmicHandler:
        # Prompt modified for JSON extraction
        SYSTEM_PROMPT = """Extract the graph structure from this problem. You do not need to tackle the problem itself. Only extract the graph structure.
Output ONLY a JSON object:
{
  "edges": [[source_node, target_node, weight], ...],   // edges of the graph
  "start_node": int,
  "end_node": int,
  "num_paths": int  // number of paths requested
}"""

        @staticmethod
        def format(prompt: str) -> str:
            return f"Extract graph structure from this problem: {prompt}"
        
        @staticmethod
        def _generalized_dijkstra(edges, start, end, k=1):
            adj = {}
            for u, v, w in edges:
                if u not in adj: adj[u] = []
                adj[u].append((v, w))
            pq = [(0, start, [start])]
            paths = []
            counts = {}
            while pq and len(paths) < k:
                cost, u, path = heapq.heappop(pq)
                if u == end:
                    paths.append((path, cost))
                    continue
                counts[u] = counts.get(u, 0) + 1
                if counts[u] > k: continue
                for v, w in adj.get(u, []):
                    heapq.heappush(pq, (cost + w, v, path + [v]))
            return paths

        @staticmethod
        def postprocess(response: str) -> str:
            try:
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if not match: return response
                data = json.loads(match.group())
                found = TaskHandlers.AlgorithmicHandler._generalized_dijkstra(
                    data.get('edges', []), data.get('start_node'), data.get('end_node'), data.get('num_paths', 1)
                )
                return json.dumps({
                    "name": "submit_paths",
                    "arguments": {"paths": [p[0] for p in found], "weights": [p[1] for p in found]}
                })
            except: return response

    class MMLUHandler:
        SYSTEM_PROMPT = "Answer the multiple choice question. End with: 'The answer is (X)'."
        @staticmethod
        def format(p): return p
        @staticmethod
        def postprocess(r):
            m = re.search(r'[Tt]he answer is \(?([A-D])\)?', r)
            return f"The answer is ({m.group(1).upper()})." if m else r

    class InfobenchHandler:
        SYSTEM_PROMPT = "You are a helpful assistant."
        @staticmethod
        def format(p): return p
        @staticmethod
        def postprocess(r): return r.strip()

# Increase replicas to 4 per GPU (Total 8) to maximize lanes
REPLICAS_PER_GPU = 4

@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    startup_timeout=600,
    scaledown_window=300,
    timeout=600,
)
@modal.concurrent(max_inputs=300)

class SpeculativeInferenceEngine:
    
    class GPUReplica:
        """Manages a single model pipeline. Designed for batch_size=1 execution."""
        def __init__(self, device_id: int, replica_id: int, main_model_id: str, draft_model_id: str):
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            self.device = f"cuda:{device_id}"
            self.id = f"GPU{device_id}-R{replica_id}"
            print(f"[{self.id}] Loading models...")

            self.tokenizer = AutoTokenizer.from_pretrained(main_model_id, trust_remote_code=True, padding_side='left')
            if not self.tokenizer.pad_token: self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load Main (INT4)
            self.main_model = AutoModelForCausalLM.from_pretrained(
                main_model_id,
                dtype=torch.float16,
                device_map={"": device_id},
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            )
            self.main_model.eval()

            # Load Draft (FP16)
            self.draft_model = AutoModelForCausalLM.from_pretrained(
                draft_model_id,
                dtype=torch.float16,
                device_map={"": device_id},
                trust_remote_code=True,
            )
            self.draft_model.eval()
            print(f"[{self.id}] Ready.")

        def process_single_request(self, prompt: str, task_type: str) -> Dict[str, Any]:
            """
            Process exactly one prompt. 
            Maximizes Speculative Decoding (BS=1) and scheduling flexibility.
            """
            import torch
            
            handler = TaskHandlers.get_handler(task_type)
            msgs = [{"role": "system", "content": handler.SYSTEM_PROMPT},
                    {"role": "user", "content": handler.format(prompt)}]
            formatted = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            
            inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
            prompt_len = inputs.input_ids.shape[1]

            # Task-Specific Params
            if task_type == 'algorithmic':
                # Extraction is easy, use low temp + draft model only
                params = {'max_new_tokens': 2048, 'temperature': 0.1}
            elif task_type == 'mmlu':
                # Greedy for MMLU
                params = {'max_new_tokens': 64, 'temperature': 0.0}
            else:
                # Standard generation
                params = {'max_new_tokens': 512, 'temperature': 0.7}

            with torch.inference_mode():
                # STRATEGY 1: Algorithmic -> Draft Model Only (Fastest)
                if task_type == 'algorithmic':
                    outputs = self.draft_model.generate(
                        **inputs,
                        do_sample=params['temperature'] > 0,
                        temperature=params.get('temperature'),
                        max_new_tokens=params['max_new_tokens'],
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                # STRATEGY 2: Everything else -> Main + Speculative Decoding
                else:
                    # Since we force batch_size=1, we ALWAYS use assistant_model
                    outputs = self.main_model.generate(
                        **inputs,
                        assistant_model=self.draft_model,
                        do_sample=params['temperature'] > 0,
                        temperature=params.get('temperature'),
                        max_new_tokens=params['max_new_tokens'],
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

            new_tokens = outputs[0][prompt_len:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            return {
                'text': handler.postprocess(text),
                'prompt_tokens': prompt_len,
                'completion_tokens': len(new_tokens),
                'task_type': task_type
            }

    @modal.enter()
    def load_models(self):
        self.replicas = []
        # Create 10 independent workers (5 per GPU)
        for device_id in [0, 1]:
            for r_id in range(REPLICAS_PER_GPU):
                self.replicas.append(self.GPUReplica(device_id, r_id, MAIN_MODEL_ID, DRAFT_MODEL_ID))
        
        self.executor = ThreadPoolExecutor(max_workers=len(self.replicas))
        print(f"System ready with {len(self.replicas)} parallel workers.")

    def generate_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """
        Splits batch into individual tasks and schedules them on the thread pool.
        This ensures short tasks don't wait for long tasks in the same batch.
        """
        if not prompts: return []
        
        # Submit all prompts individually
        futures = []
        for prompt in prompts:
            task_type = TaskRouter.classify_task(prompt)
            # Pick any replica via the executor queue
            # random selection is handled by whoever picks up the task in ThreadPool? 
            # Actually ThreadPoolExecutor just needs a function. 
            # We need to explicitly round-robin or let them pick.
            # Simplified: Randomly assign to a replica wrapper function? 
            # No, we need to map to a specific replica instance.
            
            # Simple Round-Robin Dispatch or Random Dispatch isn't ideal if one is busy.
            # ThreadPoolExecutor works best if we submit a task that acquires a replica.
            # But here we have stateful replicas.
            
            # Better approach for ThreadPool with stateful objects:
            # We can't easily dynamic load-balance with simple ThreadPoolExecutor over specific objects
            # unless we manage the queue ourselves. 
            # HACK: Just mod assignment for now? No, that causes blocking.
            
            # IMPROVEMENT: Use a managed queue of replicas?
            # For simplicity in this script: We will rely on random/mod assignment 
            # BUT split the batch so at least we are not processing serially.
            pass

        # REVISED DISPATCH: 
        # To truly prevent blocking, we need to find an IDLE replica. 
        # Since implementing a complex scheduler is hard here, we will use
        # a "Chunk Size = 1" approach with the executor, but we need to pass the replica.
        
        # Let's simply submit tasks to the executor where the executor manages the workers.
        # But `self.replicas` are distinct objects.
        
        # Solution: Use a queue of replica indices? 
        # For this implementation, we will perform a 'best effort' static assignment 
        # by distributing the batch across ALL replicas using modulo.
        # While not perfect dynamic scheduling, with 10 replicas and batch size 3, 
        # collisions are rare.
        
        results_map = [None] * len(prompts)
        futures = []
        
        for i, prompt in enumerate(prompts):
            task_type = TaskRouter.classify_task(prompt)
            
            # Modulo assignment to spread load across all 10 workers
            # (i % 10) ensures we use all lanes.
            # Even if batch size is 3, prompt 0 goes to R0, prompt 1 to R1...
            replica = self.replicas[i % len(self.replicas)]
            
            future = self.executor.submit(replica.process_single_request, prompt, task_type)
            futures.append((i, future))
            
        # Collect results
        for idx, future in futures:
            results_map[idx] = future.result()
            
        return results_map

    @modal.fastapi_endpoint(method="POST")
    def completion(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if 'prompt' in request:
            prompts = request['prompt']
            if isinstance(prompts, str): prompts = [prompts]
        elif 'messages' in request:
            prompts = []
            for msg_list in request['messages']:
                if isinstance(msg_list, list):
                    user_msgs = [m['content'] for m in msg_list if m.get('role') == 'user']
                    prompts.append(' '.join(user_msgs))
                elif isinstance(msg_list, dict):
                    prompts.append(msg_list.get('content', ''))
        else:
            prompts = []

        if not prompts: return {'choices': [], 'usage': {}}

        # Process
        results = self.generate_batch(prompts)
        
        choices = []
        total_p, total_c = 0, 0
        for i, res in enumerate(results):
            choices.append({'text': res['text'], 'index': i, 'finish_reason': 'stop'})
            total_p += res['prompt_tokens']
            total_c += res['completion_tokens']
            
        return {
            'choices': choices,
            'model': f'{ANDREW_ID}-{SYSTEM_ID}',
            'usage': {'prompt_tokens': total_p, 'completion_tokens': total_c, 'total_tokens': total_p + total_c}
        }