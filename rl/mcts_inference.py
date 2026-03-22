"""
Centralized GPU inference service for MCTS self-play.

Workers keep tree search local on CPU and ship leaf-evaluation requests here.
The service owns a single model instance, batches requests across workers, and
optionally caches model outputs for repeated states within the same policy
version.
"""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from queue import Empty as QueueEmpty
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from rl.models import GATV2ActorCritic


def _cap_process_threads() -> None:
    """Limit BLAS/OpenMP and PyTorch threads for the inference process."""
    for var in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        os.environ[var] = os.environ.get(var, "1") or "1"

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def state_to_pyg_data(state_dict: Dict[str, Any]) -> Data:
    """Convert a state dict into a PyG graph."""
    return Data(
        x=torch.from_numpy(state_dict["node_features"]).float(),
        edge_index=torch.from_numpy(state_dict["edge_index"]).long(),
        edge_attr=torch.from_numpy(state_dict["edge_features"]).float(),
    )


@torch.inference_mode()
def batch_network_forward(
    model: GATV2ActorCritic,
    state_dicts: List[Dict[str, Any]],
    valid_actions_list: List[List[int]],
    device: str = "cpu",
) -> List[Tuple[Dict[int, float], float]]:
    """
    Batched forward pass for multiple states.

    Returns a list of `(priors_dict, value)` tuples in the same order as the
    input states.
    """
    if not state_dicts:
        return []

    model.eval()
    data_list = [state_to_pyg_data(sd).to(device) for sd in state_dicts]
    batch = Batch.from_data_list(data_list)

    z = model._get_node_embeddings(batch.x, batch.edge_index, batch.edge_attr)
    logits = model.actor_head(z).squeeze(-1)
    g = model.critic_readout(z, batch.batch)
    values = model.critic_head(g).squeeze(-1)

    ptr = batch.ptr
    results: List[Tuple[Dict[int, float], float]] = []
    for i in range(len(state_dicts)):
        start, end = ptr[i].item(), ptr[i + 1].item()
        graph_logits = logits[start:end]
        value = values[i].item()

        valid_actions = valid_actions_list[i]
        if not valid_actions:
            results.append(({}, value))
            continue

        masked_logits = torch.full_like(graph_logits, float("-inf"))
        for action in valid_actions:
            if action < len(graph_logits):
                masked_logits[action] = graph_logits[action]

        probs = F.softmax(masked_logits, dim=0).cpu().numpy()
        priors = {
            action: float(probs[action])
            for action in valid_actions
            if action < len(probs)
        }
        results.append((priors, value))

    return results


class LRUInferenceCache:
    """Simple per-policy-version LRU cache for `(priors, value)` outputs."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self._data: OrderedDict[Any, Tuple[Dict[int, float], float]] = OrderedDict()

    def get(self, key: Any) -> Tuple[Tuple[Dict[int, float], float] | None, bool]:
        if key not in self._data:
            return None, False
        self._data.move_to_end(key)
        return self._data[key], True

    def put(self, key: Any, value: Tuple[Dict[int, float], float]) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self.capacity:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()


def _load_policy_weights(
    model: GATV2ActorCritic,
    shared_model_state: Dict[str, torch.Tensor],
) -> None:
    """Refresh the server-side model from shared CPU tensors."""
    model.load_state_dict(shared_model_state)
    model.eval()


def _drain_request_batch(
    request_queue: Any,
    first_request: Dict[str, Any],
    batch_timeout_s: float,
    max_pending_requests: int,
) -> List[Dict[str, Any]]:
    """Collect a short burst of requests to improve GPU batch size."""
    requests = [first_request]
    deadline = time.monotonic() + batch_timeout_s

    while len(requests) < max_pending_requests and time.monotonic() < deadline:
        try:
            requests.append(request_queue.get_nowait())
        except QueueEmpty:
            time.sleep(0.0005)

    return requests


def inference_server_loop(
    config: Dict[str, Any],
    policy_kwargs: Dict[str, Any],
    shared_model_state: Dict[str, torch.Tensor],
    command_queue: Any,
    ack_queue: Any,
    request_queue: Any,
    response_queues: List[Any],
) -> None:
    """
    Dedicated inference server.

    Commands:
    - `{"type": "set_policy", "policy_version": int}` refresh weights and clear cache
    - `{"type": "stop"}` terminate
    """
    _cap_process_threads()
    device = "cuda" if (config.get("gpu", False) and torch.cuda.is_available()) else "cpu"
    batch_timeout_s = float(config.get("mcts_inference_batch_timeout_s", 0.002))
    max_pending_requests = int(config.get("mcts_inference_max_pending_requests", 32))
    cache = LRUInferenceCache(capacity=int(config.get("mcts_inference_cache_capacity", 20000)))

    model = GATV2ActorCritic(**policy_kwargs).to(device)
    model.eval()

    current_policy_version: int | None = None
    should_stop = False

    while not should_stop:
        try:
            command = command_queue.get_nowait()
        except QueueEmpty:
            command = None

        if command is not None:
            cmd_type = command.get("type") if isinstance(command, dict) else None
            if cmd_type == "stop":
                should_stop = True
                continue
            if cmd_type == "set_policy":
                _load_policy_weights(model, shared_model_state)
                current_policy_version = int(command["policy_version"])
                cache.clear()
                ack_queue.put(
                    {
                        "type": "policy_ready",
                        "policy_version": current_policy_version,
                    }
                )
                continue

        try:
            first_request = request_queue.get(timeout=0.1)
        except QueueEmpty:
            continue

        requests = _drain_request_batch(
            request_queue=request_queue,
            first_request=first_request,
            batch_timeout_s=batch_timeout_s,
            max_pending_requests=max_pending_requests,
        )

        responses: Dict[int, List[Tuple[Dict[int, float], float] | None]] = {}
        misses: List[Tuple[int, int, Any, Dict[str, Any], List[int]]] = []

        for request in requests:
            worker_id = int(request["worker_id"])
            policy_version = int(request["policy_version"])
            if current_policy_version is None:
                response_queues[worker_id].put({"error": "Inference service has no active policy"})
                continue
            if policy_version != current_policy_version:
                response_queues[worker_id].put(
                    {
                        "error": (
                            f"Worker requested policy_version={policy_version}, "
                            f"but service is on version={current_policy_version}"
                        )
                    }
                )
                continue

            payloads = request["payloads"]
            responses[worker_id] = [None] * len(payloads)
            for index, payload in enumerate(payloads):
                cache_value, found = cache.get(payload["state_key"])
                if found:
                    responses[worker_id][index] = cache_value
                else:
                    misses.append(
                        (
                            worker_id,
                            index,
                            payload["state_key"],
                            payload["state_dict"],
                            payload["valid_actions"],
                        )
                    )

        if misses:
            batch_sds = [entry[3] for entry in misses]
            batch_valid_actions = [entry[4] for entry in misses]
            batch_outputs = batch_network_forward(
                model=model,
                state_dicts=batch_sds,
                valid_actions_list=batch_valid_actions,
                device=device,
            )
            for (worker_id, index, state_key, _, _), output in zip(misses, batch_outputs):
                cache.put(state_key, output)
                responses[worker_id][index] = output

        for request in requests:
            worker_id = int(request["worker_id"])
            if worker_id not in responses:
                continue
            outputs = responses[worker_id]
            if any(output is None for output in outputs):
                response_queues[worker_id].put(
                    {"error": f"Incomplete response payload for worker_id={worker_id}"}
                )
                continue
            response_queues[worker_id].put(
                {
                    "policy_version": current_policy_version,
                    "outputs": outputs,
                }
            )
