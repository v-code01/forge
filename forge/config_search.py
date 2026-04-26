import os
from dataclasses import dataclass, replace
from itertools import product

import yaml

from forge.harness import HarnessConfig, BASELINE_CONFIG
from forge.profiler import profile


@dataclass
class SearchResult:
    num_workers: int
    batch_size: int
    tokens_per_sec: float
    memory_used_pct: float


_cpu = os.cpu_count() or 4
NUM_WORKERS_GRID = sorted({1, 2, 4, _cpu // 4, _cpu // 2, _cpu})
BATCH_SIZE_GRID = [8, 16, 32, 64]


def config_search(n_steps: int = 50) -> tuple[HarnessConfig, list[SearchResult]]:
    """Grid search num_workers × batch_size. Returns (optimal_config, all_results).

    Runs with all optimizations active (pin_memory, bfloat16, tokenize_offline) to
    isolate the effect of the DataLoader configuration on throughput.
    Selection criterion: max tokens/sec where memory_used_pct < 80%.
    If no cell qualifies (all OOM or all above memory threshold), returns BASELINE_CONFIG.
    """
    results: list[SearchResult] = []
    best_config = BASELINE_CONFIG
    best_tps = 0.0

    cells = list(product(NUM_WORKERS_GRID, BATCH_SIZE_GRID))
    total = len(cells)

    for i, (nw, bs) in enumerate(cells, start=1):
        print(f"  [{i}/{total}] workers={nw} batch={bs}...", end="\r", flush=True)
        cfg = replace(
            BASELINE_CONFIG,
            num_workers=nw,
            pin_memory=True,
            dtype="bfloat16",
            batch_size=bs,
            tokenize_offline=True,
            n_steps=n_steps,
        )
        try:
            result = profile(cfg)
        except Exception:
            # MPS OOM and device errors surface as RuntimeError, AssertionError,
            # or torch.OutOfMemoryError depending on PyTorch version — catch all.
            results.append(SearchResult(nw, bs, 0.0, 100.0))
            continue

        memory_used_pct = 100.0 - result.memory_headroom_pct
        results.append(SearchResult(nw, bs, result.tokens_per_sec, memory_used_pct))

        if result.tokens_per_sec > best_tps and memory_used_pct < 80.0:
            best_tps = result.tokens_per_sec
            best_config = cfg

    print(" " * 40, end="\r")  # clear progress line
    return best_config, results


def save_optimal_yaml(config: HarnessConfig, path: str = "configs/optimal.yaml") -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    data = {
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "dtype": config.dtype,
        "batch_size": config.batch_size,
        "tokenize_offline": config.tokenize_offline,
    }
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
