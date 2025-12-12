"""
Enhanced Inference System with Speculative Decoding
System 2: High-efficiency variant with speculative decoding

Key optimizations:
1. Speculative decoding with Qwen3-0.5B draft model
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
from dataclasses import dataclass
import threading
from collections import deque

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

class GraphSolver:
    """Solves shortest path problems using Dijkstra's algorithm"""
    
    @staticmethod
    def parse_graph_from_prompt(prompt: str) -> Tuple[int, List[Tuple[int, int, int]], int]:
        """
        Parse graph structure from natural language prompt.
        Returns: (num_nodes, edges as [(u, v, weight), ...], num_paths_requested)
        """
        edges = []
        num_nodes = 0
        num_paths = 1  # Default
        
        # Try to find number of paths requested (top-P)
        paths_match = re.search(r'top-?\s*(\d+)', prompt.lower())
        if paths_match:
            num_paths = int(paths_match.group(1))
        
        # Try to find number of nodes
        nodes_match = re.search(r'(\d+)\s*nodes?', prompt.lower())
        if nodes_match:
            num_nodes = int(nodes_match.group(1))
        
        # Pattern 1: "Node X to Node Y, weight Z" or "X -> Y: Z" or "X to Y with weight Z"
        edge_patterns = [
            # "Node 0 to Node 1, weight 3" or "node 0 to node 1: weight 3"
            r'[Nn]ode\s*(\d+)\s*(?:to|->|→)\s*[Nn]ode\s*(\d+)[,:\s]*(?:weight|w|cost|c)?[:\s]*(\d+)',
            # "0 -> 1: 3" or "0 to 1: 3"
            r'(\d+)\s*(?:->|→|to)\s*(\d+)[:\s]+(\d+)',
            # "edge from 0 to 1 with weight 3"
            r'edge\s+from\s+(\d+)\s+to\s+(\d+)\s+(?:with\s+)?weight\s+(\d+)',
            # "(0, 1, 3)" or "(0, 1): 3"
            r'\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)',
            r'\((\d+)\s*,\s*(\d+)\)[:\s]*(\d+)',
            # "0-1: 3" 
            r'(\d+)\s*-\s*(\d+)[:\s]+(\d+)',
        ]
        
        for pattern in edge_patterns:
            matches = re.findall(pattern, prompt)
            for match in matches:
                u, v, w = int(match[0]), int(match[1]), int(match[2])
                edges.append((u, v, w))
                num_nodes = max(num_nodes, u + 1, v + 1)
        
        # Try adjacency matrix format
        if not edges:
            # Look for matrix-like structure
            matrix_match = re.search(r'adjacency[^:]*:\s*\[\s*\[([\d\s,\[\]]+)\]', prompt, re.IGNORECASE)
            if matrix_match:
                try:
                    matrix_str = '[' + matrix_match.group(1) + ']'
                    matrix = json.loads(matrix_str.replace(' ', ',').replace(',,', ','))
                    num_nodes = len(matrix)
                    for i in range(num_nodes):
                        for j in range(num_nodes):
                            if matrix[i][j] > 0 and matrix[i][j] != float('inf'):
                                edges.append((i, j, matrix[i][j]))
                except:
                    pass
        
        # Try JSON format edges
        if not edges:
            json_match = re.search(r'"edges"\s*:\s*\[(.*?)\]', prompt, re.DOTALL)
            if json_match:
                try:
                    edges_str = '[' + json_match.group(1) + ']'
                    edges_data = json.loads(edges_str)
                    for e in edges_data:
                        if isinstance(e, (list, tuple)) and len(e) >= 3:
                            edges.append((int(e[0]), int(e[1]), int(e[2])))
                            num_nodes = max(num_nodes, int(e[0]) + 1, int(e[1]) + 1)
                except:
                    pass
        
        return num_nodes, edges, num_paths
    
    @staticmethod
    def find_k_shortest_paths(
        num_nodes: int,
        edges: List[Tuple[int, int, int]],
        source: int,
        target: int,
        k: int
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Find k shortest simple paths using Yen's algorithm.
        Returns: (paths, weights)
        """
        if num_nodes == 0 or not edges:
            return [], []
        
        # Build adjacency list
        graph = defaultdict(list)
        for u, v, w in edges:
            graph[u].append((v, w))
        
        def dijkstra_with_path(
            graph: Dict[int, List[Tuple[int, int]]],
            source: int,
            target: int,
            num_nodes: int,
            forbidden_edges: Set[Tuple[int, int]] = None,
            forbidden_nodes: Set[int] = None
        ) -> Tuple[Optional[List[int]], int]:
            """Modified Dijkstra that returns path and avoids forbidden edges/nodes"""
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
                    # Reconstruct path
                    path = []
                    node = target
                    while node is not None:
                        path.append(node)
                        node = parent.get(node)
                    return path[::-1], dist[target]
                
                for v, w in graph[u]:
                    if v in forbidden_nodes:
                        continue
                    if (u, v) in forbidden_edges:
                        continue
                    
                    new_dist = dist[u] + w
                    if new_dist < dist[v]:
                        dist[v] = new_dist
                        parent[v] = u
                        heapq.heappush(pq, (new_dist, v))
            
            return None, float('inf')
        
        # Yen's K-Shortest Paths Algorithm
        paths = []
        path_weights = []
        
        # Find first shortest path
        first_path, first_weight = dijkstra_with_path(graph, source, target, num_nodes)
        
        if first_path is None:
            return [], []
        
        paths.append(first_path)
        path_weights.append(first_weight)
        
        # Candidates for next shortest paths
        candidates = []  # (weight, path)
        
        for i in range(1, k):
            # Use the last found path as base
            base_path = paths[-1]
            
            for j in range(len(base_path) - 1):
                # Spur node
                spur_node = base_path[j]
                root_path = base_path[:j + 1]
                
                # Calculate root path weight
                root_weight = 0
                for idx in range(len(root_path) - 1):
                    u, v = root_path[idx], root_path[idx + 1]
                    for next_node, w in graph[u]:
                        if next_node == v:
                            root_weight += w
                            break
                
                # Find edges to remove (edges that were used by previous paths at this spur)
                forbidden_edges = set()
                for prev_path in paths:
                    if len(prev_path) > j and prev_path[:j + 1] == root_path:
                        if j + 1 < len(prev_path):
                            forbidden_edges.add((prev_path[j], prev_path[j + 1]))
                
                # Forbid nodes in root path except spur node
                forbidden_nodes = set(root_path[:-1])
                
                # Find spur path
                spur_path, spur_weight = dijkstra_with_path(
                    graph, spur_node, target, num_nodes,
                    forbidden_edges, forbidden_nodes
                )
                
                if spur_path is not None:
                    total_path = root_path[:-1] + spur_path
                    total_weight = root_weight + spur_weight
                    
                    # Check if path is simple (no repeated nodes)
                    if len(total_path) == len(set(total_path)):
                        # Check if this path is already found
                        if total_path not in paths:
                            heapq.heappush(candidates, (total_weight, total_path))
            
            if not candidates:
                break
            
            # Get the best candidate
            while candidates:
                next_weight, next_path = heapq.heappop(candidates)
                if next_path not in paths:
                    paths.append(next_path)
                    path_weights.append(next_weight)
                    break
            
            if len(paths) <= i:
                break
        
        return paths, path_weights
    
    @staticmethod
    def solve(prompt: str) -> Dict[str, Any]:
        """
        Main entry point: parse prompt and solve shortest paths problem.
        Returns the result in submit_paths format.
        """
        num_nodes, edges, num_paths = GraphSolver.parse_graph_from_prompt(prompt)
        
        if num_nodes == 0 or not edges:
            # Return empty result if parsing failed
            return {
                "name": "submit_paths",
                "arguments": {"paths": [], "weights": []}
            }
        
        # Source is always 0, target is always N-1
        source = 0
        target = num_nodes - 1
        
        paths, weights = GraphSolver.find_k_shortest_paths(
            num_nodes, edges, source, target, num_paths
        )
        
        return {
            "name": "submit_paths",
            "arguments": {"paths": paths, "weights": weights}
        }


# ============================================================================
# LLM-based Graph Parser (Fallback when regex fails)
# ============================================================================

class LLMGraphParser:
    """Uses LLM to parse graph when regex patterns fail"""
    
    PARSE_PROMPT = """Extract the graph structure from this problem. Output ONLY a JSON object with:
- "num_nodes": number of nodes (integer)
- "edges": list of [source, target, weight] arrays
- "num_paths": number of paths requested (integer)

Problem:
{prompt}

JSON output:"""

    @staticmethod
    def create_parse_prompt(prompt: str) -> str:
        return LLMGraphParser.PARSE_PROMPT.format(prompt=prompt)
    
    @staticmethod
    def parse_llm_response(response: str) -> Tuple[int, List[Tuple[int, int, int]], int]:
        """Parse LLM's JSON response"""
        try:
            # Find JSON in response
            json_match = re.search(r'\{[^{}]*"num_nodes"[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                num_nodes = data.get('num_nodes', 0)
                edges = [tuple(e) for e in data.get('edges', [])]
                num_paths = data.get('num_paths', 1)
                return num_nodes, edges, num_paths
        except:
            pass
        return 0, [], 1


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


class SpeculativeDecoder:
    """Implements speculative decoding for faster generation"""
    
    def __init__(self, main_model, draft_model, tokenizer, num_speculative_tokens: int = 4):
        self.main_model = main_model
        self.draft_model = draft_model
        self.tokenizer = tokenizer
        self.num_speculative_tokens = num_speculative_tokens
        
        self.main_device = main_model.device
        self.draft_device = draft_model.device
    
    def generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ):
        """Generate with speculative decoding"""
        import torch
        
        input_ids = input_ids.to(self.main_device)
        attention_mask = attention_mask.to(self.main_device)
        device = self.main_device
        
        batch_size = input_ids.shape[0]
        
        # For batch size > 1, fall back to regular generation
        # Speculative decoding is most effective for single sequences
        if batch_size > 1:
            return self._regular_generate(
                input_ids, attention_mask, max_new_tokens, temperature, **kwargs
            )
        
        generated_tokens = []
        current_input_ids = input_ids.clone()
        current_attention_mask = attention_mask.clone()
        
        tokens_generated = 0
        
        while tokens_generated < max_new_tokens:
            # Step 1: Generate speculative tokens with draft model
            # Move current input to draft device
            draft_input_ids = current_input_ids.to(self.draft_device)
            draft_attention_mask = current_attention_mask.to(self.draft_device)
            
            with torch.inference_mode():
                draft_outputs = self.draft_model.generate(
                    input_ids=draft_input_ids,
                    attention_mask=draft_attention_mask,
                    max_new_tokens=self.num_speculative_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 0.01),
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            
            # Extract new tokens (on GPU 1) and move back to main device (GPU 0)
            draft_tokens = draft_outputs.sequences[:, current_input_ids.shape[1]:].to(self.main_device)
            
            num_draft = draft_tokens.shape[1]
            if num_draft == 0:
                break
            
            # Step 2: Verify with main model (single forward pass for all draft tokens)
            verify_input = torch.cat([current_input_ids, draft_tokens], dim=1)
            verify_mask = torch.cat([
                current_attention_mask,
                torch.ones(batch_size, num_draft, device=self.main_device)
            ], dim=1)
            
            with torch.inference_mode():
                main_outputs = self.main_model(
                    input_ids=verify_input,
                    attention_mask=verify_mask,
                    use_cache=True,
                )
            
            # Step 3: Accept/reject draft tokens
            main_logits = main_outputs.logits[:, current_input_ids.shape[1]-1:-1, :]
            
            accepted_tokens = []
            for i in range(num_draft):
                draft_token = draft_tokens[0, i].item()
                main_probs = torch.softmax(main_logits[0, i] / max(temperature, 0.01), dim=-1)
                
                # Accept if main model agrees (simplified acceptance)
                main_top_token = main_probs.argmax().item()
                
                if draft_token == main_top_token or main_probs[draft_token] > 0.1:
                    accepted_tokens.append(draft_token)
                else:
                    # Reject and sample from main model
                    if temperature > 0:
                        new_token = torch.multinomial(main_probs, 1).item()
                    else:
                        new_token = main_top_token
                    accepted_tokens.append(new_token)
                    break  # Stop at first rejection
            
            # Update state
            generated_tokens.extend(accepted_tokens)
            tokens_generated += len(accepted_tokens)
            
            new_tokens = torch.tensor([accepted_tokens], device=self.main_device)
            current_input_ids = torch.cat([current_input_ids, new_tokens], dim=1)
            current_attention_mask = torch.cat([
                current_attention_mask,
                torch.ones(batch_size, len(accepted_tokens), device=self.main_device)
            ], dim=1)
            
            # Check for EOS
            if self.tokenizer.eos_token_id in accepted_tokens:
                break
        
        return torch.cat([input_ids, torch.tensor([generated_tokens], device=device)], dim=1)
    
    def _regular_generate(
        self,
        input_ids,
        attention_mask,
        max_new_tokens: int,
        temperature: float,
        **kwargs
    ):
        """Fall back to regular generation for batches"""
        import torch
        input_ids = input_ids.to(self.main_device)
        attention_mask = attention_mask.to(self.main_device)
        
        with torch.inference_mode():
            outputs = self.main_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 0.01) if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        return outputs


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
        SYSTEM_PROMPT = """You solve graph shortest path problems. Given a directed weighted graph, find the top-P shortest paths from source (node 0) to target (node N-1).

Output your answer as a JSON tool call:
{"name": "submit_paths", "arguments": {"paths": [[path1], [path2], ...], "weights": [w1, w2, ...]}}

Where paths are sorted by weight in ascending order."""

        @staticmethod
        def format(prompt: str) -> str:
            return prompt
        
        @staticmethod
        def postprocess(response: str) -> str:
            # Extract JSON
            match = re.search(r'\{[^{}]*"name"[^{}]*"submit_paths"[^{}]*\}', response, re.DOTALL)
            if match:
                try:
                    json.loads(match.group())
                    return match.group()
                except:
                    pass
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
            # Try to find any standalone letter
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
    """Inference engine with speculative decoding"""
    
    @modal.enter()
    def load_models(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        print("Loading models for speculative decoding...")
        
        # Load tokenizer (shared)
        self.tokenizer = AutoTokenizer.from_pretrained(
            MAIN_MODEL_ID,
            trust_remote_code=True,
            padding_side='left',
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load main model on GPU 0
        print("Loading main model...")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        self.main_model = AutoModelForCausalLM.from_pretrained(
            MAIN_MODEL_ID,
            dtype=torch.bfloat16,
            device_map={"": 0},  # GPU 0
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            quantization_config=quantization_config,
        )
        self.main_model.eval()
        
        # Load draft model on GPU 1
        print("Loading draft model...")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            DRAFT_MODEL_ID,
            dtype=torch.float16,
            device_map={"": 1},  # GPU 1
            trust_remote_code=True,
        )
        self.draft_model.eval()
        
        # Initialize speculative decoder
        self.speculative_decoder = SpeculativeDecoder(
            self.main_model,
            self.draft_model,
            self.tokenizer,
            num_speculative_tokens=5,
        )
        
        print("All models loaded!")
    
    def _prepare_input(self, prompt: str, task_type: str) -> Tuple[Any, Any]:
        import torch
        
        handler = TaskHandlers.get_handler(task_type)
        
        # Build messages
        messages = [
            {"role": "system", "content": handler.SYSTEM_PROMPT},
            {"role": "user", "content": handler.format(prompt)},
        ]
        
        # Apply chat template
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Tokenize
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        
        return inputs['input_ids'].cuda(), inputs['attention_mask'].cuda()
    
    def _get_generation_params(self, task_type: str, complexity: str) -> Dict[str, Any]:
        """Get task and complexity-specific generation parameters"""
        
        base_params = {
            'algorithmic': {'max_new_tokens': 1024, 'temperature': 0.2},
            'mmlu': {'max_new_tokens': 512, 'temperature': 0.0},
            'infobench': {'max_new_tokens': 512, 'temperature': 0.7},
        }
        
        params = base_params.get(task_type, base_params['infobench']).copy()
        
        # Adjust based on complexity
        if complexity == 'simple':
            params['max_new_tokens'] = min(params['max_new_tokens'], 256)
        elif complexity == 'complex':
            params['max_new_tokens'] = int(params['max_new_tokens'] * 2)
        
        return params
    
    def generate(self, prompt: str) -> Dict[str, Any]:
        import torch
        
        # Classify and analyze
        task_type = TaskRouter.classify_task(prompt)
        complexity = TaskRouter.estimate_complexity(prompt, task_type)
        
        # Prepare input
        input_ids, attention_mask = self._prepare_input(prompt, task_type)
        prompt_tokens = input_ids.shape[1]
        
        # Get generation params
        params = self._get_generation_params(task_type, complexity)
        
        # Generate (use speculative for single requests, regular for batches)
        if complexity == 'simple' and task_type == 'mmlu':
            # For simple MMLU, use draft model directly (faster)
            with torch.inference_mode():
                outputs = self.draft_model.generate(
                    input_ids=input_ids.to(self.draft_model.device),
                    attention_mask=attention_mask.to(self.draft_model.device),
                    max_new_tokens=params['max_new_tokens'],
                    do_sample=params['temperature'] > 0,
                    temperature=max(params['temperature'], 0.01) if params['temperature'] > 0 else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        else:
            # Use speculative decoding
            outputs = self.speculative_decoder.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **params,
            )
        
        # Decode
        new_tokens = outputs[0, prompt_tokens:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Postprocess
        handler = TaskHandlers.get_handler(task_type)
        response = handler.postprocess(response)
        
        return {
            'text': response,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': len(new_tokens),
            'task_type': task_type,
        }
    
    def generate_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Process batch with task-aware grouping"""
        import torch
        
        if not prompts:
            return []
        
        # Group by task type
        task_groups = {}
        for idx, prompt in enumerate(prompts):
            task_type = TaskRouter.classify_task(prompt)
            if task_type not in task_groups:
                task_groups[task_type] = []
            task_groups[task_type].append((idx, prompt))
        
        results = [None] * len(prompts)
        
        # TODO: Easy first
        for task_type, group in task_groups.items():
            indices, group_prompts = zip(*group)
            
            # For MMLU, batch process with main model (greedy)
            if task_type == 'mmlu':
                batch_results = self._batch_generate_mmlu(list(group_prompts))
            else:
                # Process individually with speculative decoding
                batch_results = [self.generate(p) for p in group_prompts]
            
            for i, result in zip(indices, batch_results):
                results[i] = result
        
        return results
    
    def _batch_generate_mmlu(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Optimized batch generation for MMLU"""
        import torch
        
        handler = TaskHandlers.MMLUHandler
        
        # Prepare all inputs
        formatted_prompts = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": handler.SYSTEM_PROMPT},
                {"role": "user", "content": handler.format(prompt)},
            ]
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted_prompts.append(formatted)
        
        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        
        input_ids = inputs['input_ids'].cuda()
        attention_mask = inputs['attention_mask'].cuda()
        prompt_tokens = input_ids.shape[1]
        
        # Generate
        with torch.inference_mode():
            outputs = self.main_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode and postprocess
        results = []
        for i, output in enumerate(outputs):
            new_tokens = output[prompt_tokens:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            response = handler.postprocess(response)
            
            results.append({
                'text': response,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': len(new_tokens),
                'task_type': 'mmlu',
            })
        
        return results
    
    # @modal.method()
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
            return {
                'choices': [],
                'model': f'{ANDREW_ID}-{SYSTEM_ID}',
                'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
            }
        
        # Generate
        if len(prompts) == 1:
            results = [self.generate(prompts[0])]
        else:
            results = self.generate_batch(prompts)
        
        # Format response
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


# @app.function(image=image, timeout=TIMEOUT)
# @modal.fastapi_endpoint(method="POST")
# @modal.concurrent(max_inputs=300)
# async def completion(request: Dict[str, Any]) -> Dict[str, Any]:
#     engine = SpeculativeInferenceEngine()
#     return engine.completion.remote(request)


if __name__ == "__main__":
    print(f"Deploying: {ANDREW_ID}-{SYSTEM_ID}")
