"""
Optimized Inference System - Fixed for High Throughput
Key fixes:
1. Regex-based graph parsing + Dijkstra (no LLM for algorithmic tasks)
2. Proper parallel GPU execution with asyncio
3. Reduced token counts for faster inference
4. Better batching strategy
"""

import modal
import json
import re
import heapq
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

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


# =============================================================================
# GRAPH SOLVER - Pure Python, No LLM needed (~500x faster)
# =============================================================================

class GraphSolver:
    """Solves shortest path problems using regex parsing + Dijkstra"""
    
    @staticmethod
    def parse_graph_from_prompt(prompt: str) -> Tuple[int, List[Tuple[int, int, int]], int]:
        """Parse graph from prompt using regex patterns"""
        edges = []
        num_nodes = 0
        num_paths = 1
        
        # Find number of paths requested
        paths_match = re.search(r'top\s*(\d+)', prompt.lower())
        if paths_match:
            num_paths = int(paths_match.group(1))
        
        # Find number of nodes
        nodes_match = re.search(r'(\d+)\s*nodes', prompt.lower())
        if nodes_match:
            num_nodes = int(nodes_match.group(1))
        
        # Parse edges: "X -> Y, weight: Z" format (most common in batch_arrivals)
        edge_pattern = r'(\d+)\s*->\s*(\d+),?\s*weight:?\s*(\d+)'
        matches = re.findall(edge_pattern, prompt)
        
        for match in matches:
            u, v, w = int(match[0]), int(match[1]), int(match[2])
            edges.append((u, v, w))
            num_nodes = max(num_nodes, u + 1, v + 1)
        
        # Fallback patterns
        if not edges:
            patterns = [
                r'[Nn]ode\s*(\d+)\s*(?:to|->)\s*[Nn]ode\s*(\d+)[,:\s]*(?:weight)?[:\s]*(\d+)',
                r'\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)',
            ]
            for pattern in patterns:
                matches = re.findall(pattern, prompt)
                for match in matches:
                    u, v, w = int(match[0]), int(match[1]), int(match[2])
                    if (u, v, w) not in edges:
                        edges.append((u, v, w))
                        num_nodes = max(num_nodes, u + 1, v + 1)
        
        return num_nodes, edges, num_paths
    
    @staticmethod
    def find_k_shortest_paths(
        num_nodes: int,
        edges: List[Tuple[int, int, int]],
        source: int,
        target: int,
        k: int
    ) -> Tuple[List[List[int]], List[int]]:
        """Yen's K-Shortest Paths algorithm"""
        if num_nodes == 0 or not edges:
            return [], []
        
        graph = defaultdict(list)
        for u, v, w in edges:
            graph[u].append((v, w))
        
        def dijkstra(forbidden_edges: Set = None, forbidden_nodes: Set = None):
            if forbidden_edges is None:
                forbidden_edges = set()
            if forbidden_nodes is None:
                forbidden_nodes = set()
            
            if source in forbidden_nodes or target in forbidden_nodes:
                return None, float('inf')
            
            dist = {i: float('inf') for i in range(num_nodes)}
            dist[source] = 0
            parent = {source: None}
            pq = [(0, source)]
            
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue
                if u == target:
                    path = []
                    node = target
                    while node is not None:
                        path.append(node)
                        node = parent.get(node)
                    return path[::-1], dist[target]
                
                for v, w in graph[u]:
                    if v in forbidden_nodes or (u, v) in forbidden_edges:
                        continue
                    new_dist = dist[u] + w
                    if new_dist < dist[v]:
                        dist[v] = new_dist
                        parent[v] = u
                        heapq.heappush(pq, (new_dist, v))
            
            return None, float('inf')
        
        # Find first path
        first_path, first_weight = dijkstra()
        if first_path is None:
            return [], []
        
        paths = [first_path]
        path_weights = [first_weight]
        candidates = []
        
        for i in range(1, k):
            base_path = paths[-1]
            
            for j in range(len(base_path) - 1):
                spur_node = base_path[j]
                root_path = base_path[:j + 1]
                
                root_weight = 0
                for idx in range(len(root_path) - 1):
                    u, v = root_path[idx], root_path[idx + 1]
                    for next_node, w in graph[u]:
                        if next_node == v:
                            root_weight += w
                            break
                
                forbidden_edges = set()
                for prev_path in paths:
                    if len(prev_path) > j and prev_path[:j + 1] == root_path:
                        if j + 1 < len(prev_path):
                            forbidden_edges.add((prev_path[j], prev_path[j + 1]))
                
                forbidden_nodes = set(root_path[:-1])
                spur_path, spur_weight = dijkstra(forbidden_edges, forbidden_nodes)
                
                if spur_path is not None:
                    total_path = root_path[:-1] + spur_path
                    total_weight = root_weight + spur_weight
                    
                    if len(total_path) == len(set(total_path)) and total_path not in paths:
                        heapq.heappush(candidates, (total_weight, tuple(total_path)))
            
            if not candidates:
                break
            
            while candidates:
                next_weight, next_path = heapq.heappop(candidates)
                next_path = list(next_path)
                if next_path not in paths:
                    paths.append(next_path)
                    path_weights.append(next_weight)
                    break
            
            if len(paths) <= i:
                break
        
        return paths, path_weights
    
    @staticmethod
    def solve(prompt: str) -> str:
        """Main solver: returns JSON string"""
        num_nodes, edges, num_paths = GraphSolver.parse_graph_from_prompt(prompt)
        
        if num_nodes > 0 and edges:
            source = 0
            target = num_nodes - 1
            paths, weights = GraphSolver.find_k_shortest_paths(
                num_nodes, edges, source, target, num_paths
            )
        else:
            paths, weights = [], []
        
        # result = {
        #     "name": "submit_paths",
        #     "arguments": {"paths": paths, "weights": weights}
        # }
        result = {
            "paths": [{"path": p, "weight": w} for p, w in zip(paths, weights)]
        }
        return json.dumps(result)


# =============================================================================
# TASK ROUTING
# =============================================================================

class TaskRouter:
    @staticmethod
    def classify(prompt: str) -> str:
        prompt_lower = prompt.lower()
        
        # Algorithmic patterns
        if any(p in prompt_lower for p in ['directed graph', 'shortest path', 'submit_paths', '-> ', 'weight:']):
            return 'algorithmic'
        
        # MMLU patterns
        if re.search(r'multiple choice|Options:\s*\n?\s*A\.|college.*medicine|professional.*medicine', prompt, re.IGNORECASE):
            return 'mmlu'
        
        return 'infobench'


# =============================================================================
# RESPONSE POSTPROCESSING  
# =============================================================================

class PostProcessor:
    @staticmethod
    def process_mmlu(response: str) -> str:
        # Extract answer letter
        match = re.search(r'[Tt]he answer is \(?([A-D])\)?', response)
        if match:
            return f"The answer is ({match.group(1).upper()})."
        match = re.search(r'\b([A-D])\b', response)
        if match:
            return f"The answer is ({match.group(1).upper()})."
        return response
    
    @staticmethod
    def process_infobench(response: str) -> str:
        return response.strip()


# =============================================================================
# MAIN INFERENCE ENGINE
# =============================================================================

@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    timeout=TIMEOUT,
    scaledown_window=300,
)
@modal.concurrent(max_inputs=300)
class OptimizedInferenceEngine:
    """High-throughput inference with hybrid LLM + Algorithm approach"""
    
    class GPUReplica:
        """Single GPU replica"""
        def __init__(self, device_id: int):
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            self.device = f"cuda:{device_id}"
            self.device_id = device_id
            print(f"Loading models on {self.device}...")
            
            # Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                MAIN_MODEL_ID, trust_remote_code=True, padding_side='left'
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Main model with 4-bit quantization
            quant_config = BitsAndBytesConfig(load_in_4bit=True)
            self.main_model = AutoModelForCausalLM.from_pretrained(
                MAIN_MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map={"": device_id},
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                quantization_config=quant_config,
            )
            self.main_model.eval()
            
            # Draft model for speculative decoding (single requests)
            self.draft_model = AutoModelForCausalLM.from_pretrained(
                DRAFT_MODEL_ID,
                torch_dtype=torch.float16,
                device_map={"": device_id},
                trust_remote_code=True,
            )
            self.draft_model.eval()
            print(f"Replica {device_id} ready!")
        
        def generate(self, prompts: List[str], task_type: str) -> List[Dict]:
            """Generate responses for a batch"""
            import torch
            
            if not prompts:
                return []
            
            # System prompts
            if task_type == 'mmlu':
                system = "You are a medical expert. Answer with 'The answer is (X)' where X is A, B, C, or D."
                max_tokens = 256  # MMLU answers are very short!
                temperature = 0.0
            else:  # infobench
                system = "You are a helpful assistant. Provide clear, accurate responses."
                max_tokens = 512
                temperature = 0.7
            
            # Format prompts
            formatted = []
            for p in prompts:
                msgs = [{"role": "system", "content": system}, {"role": "user", "content": p}]
                formatted.append(self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
                ))
            
            # Tokenize
            inputs = self.tokenizer(
                formatted, return_tensors="pt", padding=True,
                truncation=True, max_length=2048
            ).to(self.device)
            
            prompt_len = inputs.input_ids.shape[1]
            batch_size = len(prompts)
            
            # Generate
            with torch.inference_mode():
                gen_kwargs = {
                    'max_new_tokens': max_tokens,
                    'do_sample': temperature > 0,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'use_cache': True,
                }
                if temperature > 0:
                    gen_kwargs['temperature'] = temperature
                
                # Use speculative decoding only for single requests
                if batch_size == 1 and task_type != 'mmlu':
                    gen_kwargs['assistant_model'] = self.draft_model
                
                outputs = self.main_model.generate(**inputs, **gen_kwargs)
            
            # Decode and postprocess
            results = []
            for i, output in enumerate(outputs):
                new_tokens = output[prompt_len:]
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                if task_type == 'mmlu':
                    response = PostProcessor.process_mmlu(response)
                else:
                    response = PostProcessor.process_infobench(response)
                
                results.append({
                    'text': response,
                    'prompt_tokens': prompt_len,
                    'completion_tokens': len(new_tokens),
                    'task_type': task_type,
                })
            
            return results
    
    @modal.enter()
    def load_models(self):
        """Initialize GPU replicas"""
        self.replicas = [
            self.GPUReplica(0),
            self.GPUReplica(1),
        ]
        self.executor = ThreadPoolExecutor(max_workers=2)
        print("All replicas ready!")
    
    def process_batch(self, prompts: List[str]) -> List[Dict]:
        """Process a batch of prompts with task-aware routing"""
        if not prompts:
            return []
        
        results = [None] * len(prompts)
        
        # Classify and separate tasks
        algo_items = []  # (original_idx, prompt)
        llm_items = []   # (original_idx, prompt, task_type)
        
        for idx, prompt in enumerate(prompts):
            task_type = TaskRouter.classify(prompt)
            if task_type == 'algorithmic':
                algo_items.append((idx, prompt))
            else:
                llm_items.append((idx, prompt, task_type))
        
        # ============================================
        # 1. Process ALGORITHMIC tasks (instant, no LLM)
        # ============================================
        for idx, prompt in algo_items:
            result_text = GraphSolver.solve(prompt)
            results[idx] = {
                'text': result_text,
                'prompt_tokens': len(prompt) // 4,  # Approximate
                'completion_tokens': len(result_text) // 4,
                'task_type': 'algorithmic',
            }
        
        # ============================================
        # 2. Process LLM tasks (MMLU + Infobench)
        # ============================================
        if llm_items:
            # Group by task type
            mmlu_items = [(idx, p) for idx, p, t in llm_items if t == 'mmlu']
            info_items = [(idx, p) for idx, p, t in llm_items if t == 'infobench']
            
            # Process MMLU (high priority - fast)
            if mmlu_items:
                indices, prompts_list = zip(*mmlu_items)
                prompts_list = list(prompts_list)
                
                # Split between GPUs
                mid = len(prompts_list) // 2
                batch_0 = prompts_list[:mid] if mid > 0 else []
                batch_1 = prompts_list[mid:] if mid < len(prompts_list) else prompts_list
                
                futures = []
                if batch_0:
                    futures.append(('mmlu', 0, indices[:mid], 
                                   self.executor.submit(self.replicas[0].generate, batch_0, 'mmlu')))
                if batch_1:
                    futures.append(('mmlu', 1, indices[mid:],
                                   self.executor.submit(self.replicas[1].generate, batch_1, 'mmlu')))
                
                # Collect results
                for task_type, gpu_id, idx_list, future in futures:
                    batch_results = future.result()
                    for orig_idx, result in zip(idx_list, batch_results):
                        results[orig_idx] = result
            
            # Process Infobench
            if info_items:
                indices, prompts_list = zip(*info_items)
                prompts_list = list(prompts_list)
                
                mid = len(prompts_list) // 2
                batch_0 = prompts_list[:mid] if mid > 0 else []
                batch_1 = prompts_list[mid:] if mid < len(prompts_list) else prompts_list
                
                futures = []
                if batch_0:
                    futures.append((indices[:mid],
                                   self.executor.submit(self.replicas[0].generate, batch_0, 'infobench')))
                if batch_1:
                    futures.append((indices[mid:],
                                   self.executor.submit(self.replicas[1].generate, batch_1, 'infobench')))
                
                for idx_list, future in futures:
                    batch_results = future.result()
                    for orig_idx, result in zip(idx_list, batch_results):
                        results[orig_idx] = result
        
        return results
    
    @modal.fastapi_endpoint(method="POST")
    def completion(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI-compatible completion endpoint"""
        # Extract prompts
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
            return {'choices': [], 'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}}
        
        # Process
        results = self.process_batch(prompts)
        
        # Format response
        choices = []
        total_prompt = 0
        total_completion = 0
        
        for i, result in enumerate(results):
            if result:
                choices.append({
                    'text': result['text'],
                    'prompt': prompts[i],
                    'index': i,
                    'finish_reason': 'stop',
                })
                total_prompt += result.get('prompt_tokens', 0)
                total_completion += result.get('completion_tokens', 0)
        
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
