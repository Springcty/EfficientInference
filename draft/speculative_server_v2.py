"""
Enhanced Inference System with Speculative Decoding
System 2: High-efficiency variant with speculative decoding

Key optimizations:
1. Speculative decoding with Qwen3-0.6B draft model
2. Adaptive model selection based on task complexity
3. INT4 quantization for maximum throughput
4. Aggressive batching strategies
"""

import modal
import asyncio
import time
import json
import re
from typing import List, Dict, Any, Optional, Tuple
import heapq
from dataclasses import dataclass
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

# Modal configuration
ANDREW_ID = "tianyuca"
SYSTEM_ID = "2"

GPU_CONFIG = "A100-80GB:2"
TIMEOUT = 600

# Model configurations
MAIN_MODEL_ID = "Qwen/Qwen3-8B"
DRAFT_MODEL_ID = "Qwen/Qwen3-0.6B"  # Small draft model for speculative decoding

cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"


flash_attn_release = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.1.post1/"
    "flash_attn-2.7.1.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)  # We use a pre-built binary for flash-attn to install it in the image.

# Image with all dependencies
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
        # "verifiers[all]==0.1.1",


app = modal.App(f"{ANDREW_ID}-{SYSTEM_ID}")


class TaskRouter:
    """Enhanced task router with complexity estimation"""
    
    TASK_PATTERNS = {
        'algorithmic': [
            r'directed graph',
            r'shortest path',
            r'source.*target',
            r'adjacency',
            r'edge.*weight',
            r'node.*labeled',
            r'submit_paths',
            r'top-?\d+.*path',
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
        
        for pattern in cls.TASK_PATTERNS['algorithmic']:
            if re.search(pattern, prompt_lower):
                return 'algorithmic'
        
        for pattern in cls.TASK_PATTERNS['mmlu']:
            if re.search(pattern, prompt, re.IGNORECASE):
                return 'mmlu'
        
        return 'infobench'
    
    @classmethod
    def estimate_complexity(cls, prompt: str, task_type: str) -> str:
        """Estimate query complexity: 'simple', 'medium', 'complex'"""
        prompt_len = len(prompt)
        
        if task_type == 'mmlu':
            # TODO
            # MMLU is relatively simple - just answer selection
            return 'complex'
        
        elif task_type == 'algorithmic':
            # Check graph size indicators
            if re.search(r'node.*[5-9]|node.*1\d', prompt.lower()):
                return 'complex'
            elif re.search(r'top-?[3-9]|top-?\d{2}', prompt.lower()):
                return 'complex'
            return 'medium'
        
        else:  # infobench
            if prompt_len > 500:
                return 'complex'
            elif prompt_len > 200:
                return 'medium'
            return 'simple'


class TaskHandlers:
    """Task-specific prompt formatting and postprocessing"""
    
    @staticmethod
    def get_handler(task_type: str):
        handlers = {
            'algorithmic': TaskHandlers.AlgorithmicHandler,
            'mmlu': TaskHandlers.MMLUHandler,
            'infobench': TaskHandlers.InfobenchHandler,
        }
        return handlers.get(task_type, TaskHandlers.InfobenchHandler)
    
    class AlgorithmicHandler:
        # 1. Prompt asks for structured data extraction
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
            """
            Finds K-Shortest paths using a generalized Dijkstra algorithm.
            """
            # Build adjacency list
            adj = {}
            for u, v, w in edges:
                if u not in adj: adj[u] = []
                adj[u].append((v, w))
            
            # Priority Queue: (total_weight, current_node, path_list)
            # We use a path_list to reconstruct the path easily
            pq = [(0, start, [start])]
            
            paths = []
            # Count how many times we've visited a node to prune search for K paths
            counts = {} 
            
            while pq and len(paths) < k:
                cost, u, path = heapq.heappop(pq)
                
                # If we reached the target, record the path
                if u == end:
                    paths.append((path, cost))
                    continue
                
                # Pruning: If we've visited u more than K times, skip
                counts[u] = counts.get(u, 0) + 1
                if counts[u] > k:
                    continue
                
                # Expand neighbors
                if u in adj:
                    for v, w in adj[u]:
                        heapq.heappush(pq, (cost + w, v, path + [v]))
            
            return paths

        @staticmethod
        def postprocess(response: str) -> str:
            """
            Parses the LLM's JSON graph, runs Dijkstra, and formats the output.
            """
            # 1. Parsing: Extract JSON from the response
            try:
                # Find the first JSON-like block
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if not match:
                    return response # Fallback: return raw if parsing fails
                
                data = json.loads(match.group())
                
                edges = data.get('edges', [])
                start = data.get('start_node')
                end = data.get('end_node')
                k = data.get('target_num_paths', 1)
                
                # 2. Execution: Run Python Solver
                if edges and start is not None and end is not None:
                    found_paths = TaskHandlers.AlgorithmicHandler._generalized_dijkstra(
                        edges, start, end, k
                    )
                    
                    # 3. Formatting: Convert to requested output format
                    # The task expects: {"name": "submit_paths", "arguments": ...}
                    formatted_paths = [p[0] for p in found_paths]
                    formatted_weights = [p[1] for p in found_paths]
                    
                    tool_output = {
                        "name": "submit_paths", 
                        "arguments": {
                            "paths": formatted_paths,
                            "weights": formatted_weights
                        }
                    }
                    return json.dumps(tool_output)
                    
            except Exception as e:
                print(f"Error in algorithmic solver: {e}")
                
            return response

    class MMLUHandler:
        SYSTEM_PROMPT = """You are a medical expert. Answer the multiple choice question by selecting the best option. End your response with: "The answer is (X)" where X is A, B, C, or D."""

        @staticmethod
        def format(prompt: str) -> str:
            return prompt
        
        @staticmethod
        def postprocess(response: str) -> str:
            match = re.search(r'[Tt]he answer is \(?([A-D])\)?', response)
            if match:
                return f"The answer is ({match.group(1).upper()})."
            match = re.search(r'\b([A-D])\b', response)
            if match:
                return f"The answer is ({match.group(1).upper()})."
            return response
    
    class InfobenchHandler:
        SYSTEM_PROMPT = """You are a helpful assistant. Provide clear, comprehensive, and accurate responses."""

        @staticmethod
        def format(prompt: str) -> str:
            return prompt
        
        @staticmethod
        def postprocess(response: str) -> str:
            return response.strip()


@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    startup_timeout=300,
    scaledown_window=300,
    timeout=600,
)
@modal.concurrent(max_inputs=300)

class SpeculativeInferenceEngine:
    """
    High-Throughput Inference Engine using Data Parallelism.
    Runs two independent model replicas on GPU 0 and GPU 1.
    """

    class GPUReplica:
        """Manages a complete set of models on a single GPU."""
        def __init__(self, device_id: int, main_model_id: str, draft_model_id: str):
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            self.device = f"cuda:{device_id}"
            self.device_id = device_id
            print(f"Loading Replica on {self.device}...")

            # 1. Load Tokenizer (Shared logic, but good to have local reference)
            self.tokenizer = AutoTokenizer.from_pretrained(
                main_model_id, trust_remote_code=True, padding_side='left'
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 2. Load Main Model (INT8)
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            self.main_model = AutoModelForCausalLM.from_pretrained(
                main_model_id,
                dtype=torch.bfloat16,
                device_map={"": device_id}, # Pin to specific GPU
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                quantization_config=quantization_config,
            )
            self.main_model.eval()

            # 3. Load Draft Model (FP16)
            self.draft_model = AutoModelForCausalLM.from_pretrained(
                draft_model_id,
                dtype=torch.float16,
                device_map={"": device_id}, # Pin to same GPU
                trust_remote_code=True,
            )
            self.draft_model.eval()
            print(f"Replica on {self.device} ready.")

        def generate_batch(self, prompts: List[str], task_type: str) -> List[Dict[str, Any]]:
            """Process a sub-batch on this GPU."""
            import torch
            
            if not prompts:
                return []

            handler = TaskHandlers.get_handler(task_type)

            # Format
            formatted_prompts = []
            for p in prompts:
                msgs = [{"role": "system", "content": handler.SYSTEM_PROMPT},
                        {"role": "user", "content": handler.format(p)}]
                formatted_prompts.append(
                    self.tokenizer.apply_chat_template(
                        msgs, 
                        tokenize=False, 
                        add_generation_prompt=True, 
                        enable_thinking=False
                    )
                )

            # Tokenize & Move to specific device
            inputs = self.tokenizer(
                formatted_prompts, 
                return_tensors="pt", 
                padding=True,
                truncation=True, 
                max_length=2048
            ).to(self.device)

            prompt_len = inputs.input_ids.shape[1]
            batch_size = inputs.input_ids.shape[0]

            # Set Params
            if task_type == 'algorithmic':
                params = {'max_new_tokens': 1024, 'temperature': 0.1}
            elif task_type == 'mmlu':
                # MMLU optimization: greedy, fewer tokens
                params = {'max_new_tokens': 256, 'temperature': 0.0} 
            else: # infobench
                params = {'max_new_tokens': 512, 'temperature': 0.7}

            # Generate with Native Speculative Decoding
            with torch.inference_mode():
                if task_type == "algorithmic":
                    # Use draft model only for simple extractions
                    outputs = self.draft_model.generate(
                        **inputs,
                        do_sample=params['temperature'] > 0,
                        temperature=params['temperature'],
                        max_new_tokens=params['max_new_tokens'],
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                    )
                else:
                    generation_kwargs = {
                        'do_sample': params['temperature'] > 0,
                        'temperature': params['temperature'],
                        'max_new_tokens': params['max_new_tokens'],
                        'pad_token_id': self.tokenizer.pad_token_id,
                        'eos_token_id': self.tokenizer.eos_token_id,
                        'use_cache': True,
                    }
                    
                    # Enable speculative decoding only for single requests
                    if batch_size == 1:
                        generation_kwargs["assistant_model"] = self.draft_model
                    
                    outputs = self.main_model.generate(
                        **inputs,
                        **generation_kwargs,
                    )

            # Decode
            results = []
            for i, output in enumerate(outputs):
                new_tokens = output[prompt_len:]
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                results.append({
                    'text': handler.postprocess(response),
                    'prompt_tokens': prompt_len,
                    'completion_tokens': len(new_tokens),
                    'task_type': task_type
                })
            return results

    @modal.enter()
    def load_models(self):
        # Initialize two replicas
        self.replicas = [
            self.GPUReplica(0, MAIN_MODEL_ID, DRAFT_MODEL_ID),
            self.GPUReplica(1, MAIN_MODEL_ID, DRAFT_MODEL_ID)
        ]
        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=2)

    def generate_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Distribute batch across GPUs"""
        if not prompts:
            return []

        # 1. Group by Task Type
        # We must group by task type first because params differ per task
        task_groups = {}
        for idx, prompt in enumerate(prompts):
            task_type = TaskRouter.classify_task(prompt)
            if task_type not in task_groups:
                task_groups[task_type] = []
            task_groups[task_type].append((idx, prompt))

        final_results = [None] * len(prompts)

        # 2. Process each task group
        # OPTIMIZATION: Process in order of expected complexity
        processing_order = ['algorithmic', 'mmlu', 'infobench']
        for task_type in processing_order:
            if task_type not in task_groups:
                continue
            group = task_groups[task_type]
            
            indices, group_prompts = zip(*group)
            prompt_list = list(group_prompts)
            
            # 3. Split prompts between Replica 0 and Replica 1
            mid_point = len(prompt_list) // 2
            batch_0 = prompt_list[:mid_point]
            batch_1 = prompt_list[mid_point:]
            
            # 4. Execute in parallel
            combined_results = []
            if len(batch_0) > 0:
                future_0 = self.executor.submit(self.replicas[0].generate_batch, batch_0, task_type)
                results_0 = future_0.result()
                combined_results.extend(results_0)
            if len(batch_1) > 0:
                future_1 = self.executor.submit(self.replicas[1].generate_batch, batch_1, task_type)
                results_1 = future_1.result()
                combined_results.extend(results_1)
            
            # 5. Map back to original indices
            for original_idx, result in zip(indices, combined_results):
                final_results[original_idx] = result

        return final_results

    @modal.fastapi_endpoint(method="POST")
    def completion(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI-compatible completion endpoint"""
        # (Same implementation as before)
        if 'prompt' in request:
            prompts = request['prompt']
            if isinstance(prompts, str):
                prompts = [prompts]
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
        
        if not prompts:
            return {'choices': [], 'usage': {}}
        
        # Route to generate_batch (which handles splitting)
        results = self.generate_batch(prompts)
        
        choices = []
        total_prompt = 0
        total_completion = 0
        
        for i, result in enumerate(results):
            choices.append({
                'text': result['text'],
                'index': i,
                'finish_reason': 'stop',
            })
            total_prompt += result['prompt_tokens']
            total_completion += result['completion_tokens']
        
        return {
            'choices': choices,
            'model': f'{ANDREW_ID}-{SYSTEM_ID}',
            'usage': {
                'prompt_tokens': total_prompt,
                'completion_tokens': total_completion,
                'total_tokens': total_prompt + total_completion,
            }
        }


if __name__ == "__main__":
    print(f"Deploying: {ANDREW_ID}-{SYSTEM_ID}")
