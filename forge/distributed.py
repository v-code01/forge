"""
Toy DDP profiler — measures gradient allreduce overhead across simulated workers.

On a real multi-node setup (8×A100, AWS EFA/NCCL), gradient synchronisation
is the dominant bottleneck once DataLoader starvation is fixed. This probe
makes that cost measurable on single-machine GLOO, then maps the findings
to the distributed context.

comms_frac = allreduce_time / step_time — the distributed analog of Forge's
gpu_idle_pct. Both metrics measure the fraction of each step where the GPU
is waiting rather than computing; the cause differs (DataLoader starvation
vs. gradient sync) but the diagnosis and fix methodology is identical.

Uses CPU tensors + GLOO backend: runs on any hardware, no GPU required.
"""
import dataclasses
import json
import os
import socket
import tempfile
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


@dataclass
class DDPStepProfile:
    world_size: int
    n_params: int           # number of gradient elements; bytes = n_params * 4
    compute_ms: float       # mean forward + backward time per step (ms)
    allreduce_ms: float     # mean gradient sync time per step (ms)
    comms_frac: float       # allreduce_ms / (compute_ms + allreduce_ms)
    tokens_per_sec: float   # effective throughput accounting for both phases


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _ddp_worker(
    rank: int,
    world_size: int,
    n_steps: int,
    port: int,
    output_path: str,
) -> None:
    """Module-level worker — must be at top level to be picklable under spawn."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    # Bind GLOO to loopback to suppress hostname-resolution warnings on macOS.
    import sys
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Dense model with ~4M params → ~16MB gradient.
    # Small enough for fast CPU steps; large enough for measurable allreduce.
    # On a real GPT-2 Small (117M params), gradients are ~468MB — allreduce
    # cost scales linearly with parameter count over the network.
    hidden = 1024
    model = nn.Sequential(
        nn.Linear(hidden, hidden * 4), nn.GELU(),
        nn.Linear(hidden * 4, hidden * 4), nn.GELU(),
        nn.Linear(hidden * 4, hidden),
    )
    n_params = sum(p.numel() for p in model.parameters())

    batch_size, seq_len = 8, 128
    x = torch.randn(batch_size * seq_len, hidden)

    compute_times: list[float] = []
    allreduce_times: list[float] = []

    for _ in range(n_steps):
        t0 = time.perf_counter()
        out = model(x)
        loss = out.sum()
        loss.backward()
        t1 = time.perf_counter()

        # Explicit allreduce: this is what DDP calls under the hood per bucket.
        # On GLOO/localhost this is shared-memory memcpy; over EFA/InfiniBand
        # it becomes a ring-allreduce over the network fabric.
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        t2 = time.perf_counter()

        for param in model.parameters():
            param.grad = None

        compute_times.append(t1 - t0)
        allreduce_times.append(t2 - t1)

    dist.destroy_process_group()

    if rank == 0:
        avg_compute = (sum(compute_times) / n_steps) * 1000
        avg_allreduce = (sum(allreduce_times) / n_steps) * 1000
        step_total = avg_compute + avg_allreduce
        comms_frac = avg_allreduce / step_total if step_total > 0 else 0.0
        total_s = sum(compute_times) + sum(allreduce_times)
        tps = (batch_size * seq_len * n_steps) / total_s if total_s > 0 else 0.0
        result = DDPStepProfile(
            world_size=world_size,
            n_params=n_params,
            compute_ms=avg_compute,
            allreduce_ms=avg_allreduce,
            comms_frac=comms_frac,
            tokens_per_sec=tps,
        )
        with open(output_path, "w") as f:
            json.dump(dataclasses.asdict(result), f)


def probe_ddp(
    world_sizes: list[int] | None = None,
    n_steps: int = 20,
) -> list[DDPStepProfile]:
    """Spawn GLOO process groups for each world_size; profile compute vs allreduce.

    Returns one DDPStepProfile per entry in world_sizes. Runs sequentially —
    each probe completes and tears down before the next starts.

    Args:
        world_sizes: List of worker counts to probe. Default: [1, 2, 4].
        n_steps: Training steps per probe. 20 is sufficient for stable means.

    Returns:
        List of DDPStepProfile in the same order as world_sizes.
    """
    if world_sizes is None:
        world_sizes = [1, 2, 4]

    profiles: list[DDPStepProfile] = []

    for ws in world_sizes:
        port = _find_free_port()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            output_path = f.name

        try:
            mp.spawn(  # type: ignore[attr-defined]
                _ddp_worker,
                args=(ws, n_steps, port, output_path),
                nprocs=ws,
                join=True,
            )
            with open(output_path) as f:
                profiles.append(DDPStepProfile(**json.load(f)))
        finally:
            try:
                os.unlink(output_path)
            except FileNotFoundError:
                pass

    return profiles
