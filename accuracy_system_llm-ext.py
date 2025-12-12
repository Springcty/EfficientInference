"""
Accuracy System - Continuous Microbatching

- Adds per-GPU worker threads + global request queues (continuous microbatching)
- Each HTTP call enqueues prompts and waits on Futures; workers batch across calls
- Keeps algorithmic tasks in pure Python (no LLM)
- Adds bounded queues + per-prompt overload/timeout handling (reduces TIMEOUTs)

Deploy:
  modal deploy accuracy_system.py
"""

import modal
import json
import re
import heapq
import threading
import queue
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from concurrent.futures import Future, TimeoutError as FutureTimeoutError


# =============================================================================
# Modal configuration
# =============================================================================

ANDREW_ID = "tianyuca"
SYSTEM_ID = "3"

GPU_CONFIG = "A100-80GB:2"
TIMEOUT = 600  # Modal function timeout (seconds)

# Model configurations
MAIN_MODEL_ID = "Qwen/Qwen3-32B"

# Continuous microbatching knobs
MICROBATCH_WAIT_MS = 10          # gather window after first item arrives

MICROBATCH_MAX_BS_EXTRACT = 8
EXTRACT_MAX_NEW_TOKENS = 1024

MICROBATCH_MAX_BS_MMLU = 32      # MMLU tends to be short (safe to batch bigger)
MICROBATCH_MAX_BS_INFO = 8       # InfoBench can be longer; keep smaller
QUEUE_MAXSIZE = 4096             # per-GPU per-task queue capacity
LLM_RESULT_TIMEOUT_S = TIMEOUT - 10  # must be < Modal input timeout


# =============================================================================
# Modal Image
# =============================================================================

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

app = modal.App(f"{ANDREW_ID}-system-{SYSTEM_ID}")


# =============================================================================
# GRAPH SOLVER - Pure Python, No LLM needed
# =============================================================================

class GraphSolver:
    """Solves shortest path problems using regex parsing + Dijkstra"""

    @staticmethod
    def parse_graph_from_prompt(prompt: str) -> Tuple[int, List[Tuple[int, int, int]], int]:
        num_nodes = 0
        num_paths = 1

        # Find number of paths requested
        paths_match = re.search(r"top\s*(\d+)", prompt.lower())
        if paths_match:
            num_paths = int(paths_match.group(1))

        # Find number of nodes
        nodes_match = re.search(r"(\d+)\s*nodes", prompt.lower())
        if nodes_match:
            num_nodes = int(nodes_match.group(1))

        # Parse edges: "X -> Y, weight: Z"
        edge_pattern = r"(\d+)\s*->\s*(\d+),\s*weight:\s*(\d+)"
        edges = [(int(u), int(v), int(w)) for u, v, w in re.findall(edge_pattern, prompt)]

        return num_nodes, edges, num_paths

    @staticmethod
    def dijkstra_k_paths(n: int, edges: List[Tuple[int, int, int]], src: int, dst: int, k: int):
        graph = defaultdict(list)
        for u, v, w in edges:
            graph[u].append((v, w))

        def dijkstra(start: int, banned_edges: set, banned_nodes: set):
            dist = [float("inf")] * n
            prev = [-1] * n
            dist[start] = 0
            pq = [(0, start)]
            while pq:
                d, u = heapq.heappop(pq)
                if d != dist[u]:
                    continue
                if u == dst:
                    break
                if u in banned_nodes:
                    continue
                for v, w in graph[u]:
                    if (u, v) in banned_edges:
                        continue
                    if v in banned_nodes:
                        continue
                    nd = d + w
                    if nd < dist[v]:
                        dist[v] = nd
                        prev[v] = u
                        heapq.heappush(pq, (nd, v))

            if dist[dst] == float("inf"):
                return None, None

            # Reconstruct path
            path = []
            cur = dst
            while cur != -1:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            return path, dist[dst]

        # Yen's algorithm (simple)
        first_path, first_weight = dijkstra(src, set(), set())
        if not first_path:
            return [], []

        paths = [first_path]
        weights = [first_weight]
        candidates = []

        for _ in range(1, k):
            base_path = paths[-1]

            for j in range(len(base_path) - 1):
                spur_node = base_path[j]
                root_path = base_path[: j + 1]

                banned_edges = set()
                banned_nodes = set(root_path[:-1])

                for p in paths:
                    if len(p) > j and p[: j + 1] == root_path:
                        banned_edges.add((p[j], p[j + 1]))

                spur_path, spur_weight = dijkstra(spur_node, banned_edges, banned_nodes)
                if spur_path:
                    total_path = root_path[:-1] + spur_path

                    # compute total weight
                    total_w = 0
                    edge_map = {(u, v): w for u, v, w in edges}
                    for a, b in zip(total_path[:-1], total_path[1:]):
                        total_w += edge_map.get((a, b), 0)

                    heapq.heappush(candidates, (total_w, total_path))

            if not candidates:
                break

            w, p = heapq.heappop(candidates)
            paths.append(p)
            weights.append(w)

        return paths, weights

    @staticmethod
    def solve(prompt: str) -> str:
        """Main solver: returns JSON string compatible with the evaluation."""
        num_nodes, edges, num_paths = GraphSolver.parse_graph_from_prompt(prompt)

        if num_nodes > 0 and edges:
            source = 0
            target = num_nodes - 1
            paths, weights = GraphSolver.dijkstra_k_paths(num_nodes, edges, source, target, num_paths)
        else:
            paths, weights = [], []

        result = {"paths": [{"path": p, "weight": w} for p, w in zip(paths, weights)]}
        return json.dumps(result)

    @staticmethod
    def build_extract_prompt(raw_prompt: str) -> str:
        # Keep it short to maximize batching efficiency
        return (
            "Extract graph info from the text below.\n"
            "Return ONLY JSON with keys: num_nodes,num_paths,source,target,edges.\n"
            "Text:\n"
            + raw_prompt
        )

    @staticmethod
    def parse_extract_json(s: str) -> Optional[Dict[str, Any]]:
        import json
        # Be robust to minor leading/trailing junk: try to grab the first JSON object
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            obj = json.loads(s[start:end+1])
        except Exception:
            return None

        if not isinstance(obj, dict):
            return None
        edges = obj.get("edges", None)
        if edges is None or not isinstance(edges, list) or len(edges) == 0:
            return None

        clean_edges = []
        for e in edges:
            if isinstance(e, (list, tuple)) and len(e) == 3:
                u, v, w = e
            elif isinstance(e, dict) and {"u","v","w"} <= set(e.keys()):
                u, v, w = e["u"], e["v"], e["w"]
            else:
                continue
            try:
                u = int(u); v = int(v); w = int(w)
            except Exception:
                continue
            clean_edges.append((u, v, w))

        if not clean_edges:
            return None

        def _to_int_or_none(x):
            if x is None:
                return None
            try:
                return int(x)
            except Exception:
                return None

        return {
            "num_nodes": _to_int_or_none(obj.get("num_nodes")),
            "num_paths": _to_int_or_none(obj.get("num_paths")),
            "source": _to_int_or_none(obj.get("source")),
            "target": _to_int_or_none(obj.get("target")),
            "edges": clean_edges,
        }

# =============================================================================
# TASK ROUTING + POSTPROCESSING
# =============================================================================

class TaskRouter:
    TASK_PATTERNS = {
        'algorithmic': [
            r'directed graph',
            r'shortest path',
            r'source.*target',
            r'edge.*weight',
            r'node.*labeled',
            r'submit_paths',
            r'top-?\d+.*path',
            r'->\s*\d+',
            r'weight:\s*\d+',
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
    
    @staticmethod
    def classify(prompt: str) -> str:
        prompt_lower = prompt.lower()

        # Algorithmic patterns
        for pattern in TaskRouter.TASK_PATTERNS['algorithmic']:
            if re.search(pattern, prompt_lower):
                return "algorithmic"

        # MMLU-ish patterns
        for pattern in TaskRouter.TASK_PATTERNS['mmlu']:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                return "mmlu"

        return "infobench"


class PostProcessor:
    @staticmethod
    def process_mmlu(response: str) -> str:
        match = re.search(r"[Tt]he answer is \(?([A-D])\)?", response)
        if match:
            return f"The answer is ({match.group(1).upper()})."
        match = re.search(r"\b([A-D])\b", response)
        if match:
            return f"The answer is ({match.group(1).upper()})."
        return response

    @staticmethod
    def process_infobench(response: str) -> str:
        return response.strip()


# =============================================================================
# MAIN INFERENCE ENGINE (continuous microbatching)
# =============================================================================

class OverloadedError(Exception):
    pass


@app.cls(
    image=image,
    gpu=GPU_CONFIG,
    startup_timeout=300,
    scaledown_window=300,
    timeout=TIMEOUT,
    max_containers=1,
)
@modal.concurrent(max_inputs=300)
class Model:
    """High-throughput inference with tool calling + continuous microbatching."""

    class GPUReplica:
        """Single GPU replica."""
        def __init__(self, device_id: int):
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            self.device = f"cuda:{device_id}"
            self.device_id = device_id
            print(f"Loading models on {self.device}...")

            self.tokenizer = AutoTokenizer.from_pretrained(
                MAIN_MODEL_ID, trust_remote_code=True, padding_side="left"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.main_model = AutoModelForCausalLM.from_pretrained(
                MAIN_MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map={"": device_id},
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
            )
            self.main_model.eval()

            # speed knobs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            print(f"Replica {device_id} ready!")

        def generate(self, prompts: List[str], task_type: str) -> List[Dict[str, Any]]:
            """Generate responses for a batch."""
            import torch

            # debug
            if task_type == "algo_extract":
                print(f"GPU{self.device_id} processing algo_extract batch of size {len(prompts)}")
                print(f"Prompts: {prompts}")

            if not prompts:
                return []

            if task_type == "algo_extract":
                system_prompt = (
                    "You extract a directed weighted graph from user text.\n"
                    "Return ONLY valid JSON (no markdown, no commentary) with this schema:\n"
                    "{"
                    "\"num_nodes\": <int or null>, "
                    "\"num_paths\": <int or null>, "
                    "\"source\": <int or null>, "
                    "\"target\": <int or null>, "
                    "\"edges\": [[u,v,w], ...] "
                    "}\n"
                    "u,v are integers node ids; w is positive integer weight.\n"
                    "If something is not specified, set it to null.\n"
                    "If edges are described in natural language, infer best-effort.\n"
                )
                max_tokens = EXTRACT_MAX_NEW_TOKENS
                temperature = 0.0
            elif task_type == "mmlu":
                system_prompt = (
                    "You are a helpful assistant. Answer the multiple-choice question with the letter only.\n"
                    "Format: The answer is (X)."
                )
                max_tokens = 128
                temperature = 0.0
            else:
                system_prompt = "You are a helpful assistant."
                max_tokens = 512
                temperature = 0.7

            formatted = []
            for prompt in prompts:
                msgs = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                formatted.append(formatted_prompt)

            inputs = self.tokenizer(
                formatted,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.device)

            prompt_len = inputs.input_ids.shape[1]

            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature

            with torch.inference_mode():
                outputs = self.main_model.generate(**inputs, **gen_kwargs)

            results: List[Dict[str, Any]] = []
            for i, out in enumerate(outputs):
                decoded = self.tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
                if task_type == "mmlu":
                    decoded = PostProcessor.process_mmlu(decoded)
                else:
                    decoded = PostProcessor.process_infobench(decoded)

                results.append(
                    {
                        "text": decoded,
                        "prompt_tokens": prompt_len,
                        "completion_tokens": max(0, len(out) - prompt_len),
                        "task_type": task_type,
                    }
                )

            return results

    # -----------------------------
    # Continuous microbatching types
    # -----------------------------

    @dataclass
    class _WorkItem:
        prompt: str
        task_type: str
        fut: Future

    @modal.enter()
    def load_models(self):
        """Initialize GPU replicas + start microbatch worker threads."""
        self.replicas = [self.GPUReplica(0), self.GPUReplica(1)]
        self._stop_event = threading.Event()

        # Per-GPU, per-task queue
        self._q = {
            0: {
                "algo_extract": queue.Queue(maxsize=QUEUE_MAXSIZE),
                "mmlu": queue.Queue(maxsize=QUEUE_MAXSIZE), 
                "infobench": queue.Queue(maxsize=QUEUE_MAXSIZE)
            },
            1: {
                "algo_extract": queue.Queue(maxsize=QUEUE_MAXSIZE),
                "mmlu": queue.Queue(maxsize=QUEUE_MAXSIZE), 
                "infobench": queue.Queue(maxsize=QUEUE_MAXSIZE)
            },
        }

        self._workers: List[threading.Thread] = []
        for gpu_id in (0, 1):
            t = threading.Thread(target=self._gpu_worker_loop, args=(gpu_id,), daemon=True)
            t.start()
            self._workers.append(t)

        print("All replicas ready + microbatch workers started!")

    @modal.exit()
    def shutdown(self):
        self._stop_event.set()

    def _choose_gpu(self, task_type: str) -> int:
        if task_type == "algo_extract":
            return 0  # dedicate GPU0 for algorithmic extraction
        
        # Pick GPU with smaller backlog for this task
        q0 = self._q[0][task_type].qsize()
        q1 = self._q[1][task_type].qsize()
        return 0 if q0 <= q1 else 1

    def _enqueue_llm(self, prompt: str, task_type: str) -> Future:
        fut = Future()
        gpu_id = self._choose_gpu(task_type)
        try:
            self._q[gpu_id][task_type].put_nowait(self._WorkItem(prompt=prompt, task_type=task_type, fut=fut))
        except queue.Full:
            raise OverloadedError(f"Queue full on GPU{gpu_id} for {task_type}")
        return fut

    def _drain_microbatch(self, q: "queue.Queue", first: "_WorkItem", max_bs: int) -> List["_WorkItem"]:
        batch = [first]
        deadline = time.time() + (MICROBATCH_WAIT_MS / 1000.0)

        while len(batch) < max_bs:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                item = q.get(timeout=remaining)
                batch.append(item)
            except queue.Empty:
                break
        return batch

    def _gpu_worker_loop(self, gpu_id: int):
        """Single worker loop per GPU; the ONLY place that calls replica.generate for that GPU."""
        replica = self.replicas[gpu_id]
        
        q_ext = self._q[gpu_id]["algo_extract"]
        q_mmlu = self._q[gpu_id]["mmlu"]
        q_info = self._q[gpu_id]["infobench"]

        # Priority: MMLU first (usually shorter; reduces tail latency)
        while not self._stop_event.is_set():
            first = None
            task_type = None
            q = None
            max_bs = None

            try:
                first = q_ext.get(timeout=0.005)
                task_type = "algo_extract"
                q = q_ext
                max_bs = MICROBATCH_MAX_BS_EXTRACT
            except queue.Empty:
                try:
                    first = q_mmlu.get(timeout=0.005)
                    task_type = "mmlu"
                    q = q_mmlu
                    max_bs = MICROBATCH_MAX_BS_MMLU
                except queue.Empty:
                    try:
                        first = q_info.get(timeout=0.005)
                        task_type = "infobench"
                        q = q_info
                        max_bs = MICROBATCH_MAX_BS_INFO
                    except queue.Empty:
                        continue

            batch_items = self._drain_microbatch(q, first, max_bs)
            prompts = [wi.prompt for wi in batch_items]

            try:
                outs = replica.generate(prompts, task_type)
                for wi, out in zip(batch_items, outs):
                    if not wi.fut.done():
                        wi.fut.set_result(out)
            except Exception as e:
                for wi in batch_items:
                    if not wi.fut.done():
                        wi.fut.set_exception(e)

    # -----------------------------
    # Batch processing (routes algo vs LLM)
    # -----------------------------

    def process_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of prompts with task-aware routing."""
        if not prompts:
            return []

        results: List[Optional[Dict[str, Any]]] = [None] * len(prompts)

        # Classify and separate tasks
        algo_items: List[Tuple[int, str]] = []
        llm_items: List[Tuple[int, str, str]] = []

        for idx, prompt in enumerate(prompts):
            task_type = TaskRouter.classify(prompt)
            if task_type == "algorithmic":
                algo_items.append((idx, prompt))
            else:
                llm_items.append((idx, prompt, task_type))

        # 1) Algorithmic (instant)
        algo_need_llm = []
        
        for idx, prompt in algo_items:
            n, edges, k = GraphSolver.parse_graph_from_prompt(prompt)
            if n <= 0 or not edges:
                algo_need_llm.append((idx, prompt))
            else:
                result_text = GraphSolver.solve(prompt)
                results[idx] = {
                    "text": result_text,
                    "prompt_tokens": max(1, len(prompt) // 4),
                    "completion_tokens": max(1, len(result_text) // 4),
                    "task_type": "algorithmic",
                }

        # 2) LLM tasks via global microbatch queues
        if llm_items:
            pending: List[Tuple[int, Future]] = []

            for idx, prompt, task_type in llm_items:
                try:
                    fut = self._enqueue_llm(prompt, task_type)
                    pending.append((idx, fut))
                except OverloadedError:
                    results[idx] = {
                        "text": "OVERLOADED",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "task_type": "overloaded",
                    }

            for idx, fut in pending:
                try:
                    results[idx] = fut.result(timeout=LLM_RESULT_TIMEOUT_S)
                except FutureTimeoutError:
                    results[idx] = {
                        "text": "TIMEOUT",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "task_type": "timeout",
                    }
                except Exception as e:
                    results[idx] = {
                        "text": f"ERROR: {type(e).__name__}: {e}",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "task_type": "error",
                    }


        # Re-route Algorithmic tasks needing LLM
        if algo_need_llm:
            # debug
            print(f"Algorithmic tasks needing LLM extraction: {algo_need_llm}")
            print('-' * 40)
            pending_ext = []
            for idx, prompt in algo_need_llm:
                ext_prompt = GraphSolver.build_extract_prompt(prompt)
                print(f"Extract Prompt [{idx}]: {ext_prompt}")
                try:
                    fut = self._enqueue_llm(ext_prompt, "algo_extract")
                    pending_ext.append((idx, fut, prompt))
                except OverloadedError:
                    results[idx] = {
                        "text": json.dumps({"paths": []}),
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "task_type": "algorithmic_overloaded",
                    }

            for idx, fut, raw_prompt in pending_ext:
                try:
                    out = fut.result(timeout=LLM_RESULT_TIMEOUT_S)
                    print(f"Extract Output [{idx}]: {out.get('text', '')}")
                    print('=' * 40)
                    parsed = GraphSolver.parse_extract_json(out.get("text", ""))
                    if not parsed:
                        # final fallback: return empty
                        results[idx] = {
                            "text": json.dumps({"paths": []}),
                            "prompt_tokens": out.get("prompt_tokens", 0),
                            "completion_tokens": out.get("completion_tokens", 0),
                            "task_type": "algorithmic_extract_failed",
                        }
                        continue

                    n = parsed["num_nodes"]
                    edges = parsed["edges"]
                    k = parsed["num_paths"] or 1

                    # infer num_nodes if missing: max node id + 1
                    if not n:
                        mx = 0
                        for u, v, _ in edges:
                            mx = max(mx, u, v)
                        n = mx + 1

                    src = parsed["source"] if parsed["source"] is not None else 0
                    dst = parsed["target"] if parsed["target"] is not None else (n - 1)

                    paths, weights = GraphSolver.dijkstra_k_paths(n, edges, src, dst, k)
                    result_text = json.dumps({"paths": [{"path": p, "weight": w} for p, w in zip(paths, weights)]})

                    results[idx] = {
                        "text": result_text,
                        "prompt_tokens": out.get("prompt_tokens", 0),
                        "completion_tokens": out.get("completion_tokens", 0),
                        "task_type": "algorithmic_llm_extract",
                    }
                except FutureTimeoutError:
                    results[idx] = {
                        "text": json.dumps({"paths": []}),
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "task_type": "algorithmic_timeout",
                    }
                except Exception as e:
                    results[idx] = {
                        "text": json.dumps({"paths": []}),
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "task_type": f"algorithmic_error:{type(e).__name__}",
                    }

        # Fill any missing
        for i in range(len(results)):
            if results[i] is None:
                results[i] = {
                    "text": "ERROR: missing result",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "task_type": "error",
                }

        return results  # type: ignore[return-value]

    # -----------------------------
    # OpenAI-ish endpoint
    # -----------------------------

    @modal.fastapi_endpoint(method="POST")
    def completion(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI-compatible completion endpoint (batch prompts supported)."""
        # Extract prompts
        if "prompt" in request:
            prompts = request["prompt"]
            if isinstance(prompts, str):
                prompts = [prompts]
        elif "messages" in request:
            prompts = []
            for msg_list in request["messages"]:
                # expecting a list of messages (role/content); take the last content as prompt
                if isinstance(msg_list, list) and msg_list:
                    prompts.append(msg_list[-1].get("content", ""))
                elif isinstance(msg_list, dict):
                    prompts.append(msg_list.get("content", ""))
                else:
                    prompts.append("")
        else:
            prompts = []

        results = self.process_batch(prompts)

        # Build OpenAI-style response
        choices = []
        total_prompt = 0
        total_completion = 0

        for i, result in enumerate(results):
            choices.append(
                {
                    "index": i,
                    "text": result.get("text", ""),
                    "prompt": prompts[i],
                    "finish_reason": "stop",
                }
            )
            total_prompt += int(result.get("prompt_tokens", 0) or 0)
            total_completion += int(result.get("completion_tokens", 0) or 0)

        return {
            "choices": choices,
            "model": f"{ANDREW_ID}-system-{SYSTEM_ID}",
            "usage": {
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion, 
            },
        }


if __name__ == "__main__":
    print(f"Deploying: {ANDREW_ID}-system-{SYSTEM_ID}")
