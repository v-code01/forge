import os
from dataclasses import dataclass, replace

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
    Selection criterion: max tokens/sec where memory_used < 80%.
    """
    results: list[SearchResult] = []
    best_config = BASELINE_CONFIG
    best_tps = 0.0

    total = len(NUM_WORKERS_GRID) * len(BATCH_SIZE_GRID)
    i = 0

    for nw in NUM_WORKERS_GRID:
        for bs in BATCH_SIZE_GRID:
            i += 1
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
            except RuntimeError:
                # OOM or MPS error for large batch sizes — skip cell
                results.append(SearchResult(nw, bs, 0.0, 100.0))
                continue

            memory_used_pct = 100.0 - result.memory_headroom_pct
            results.append(SearchResult(nw, bs, result.tokens_per_sec, memory_used_pct))

            if result.tokens_per_sec > best_tps and memory_used_pct < 80.0:
                best_tps = result.tokens_per_sec
                best_config = cfg

    print()
    return best_config, results


def save_optimal_yaml(config: HarnessConfig, path: str = "configs/optimal.yaml") -> None:
    data = {
        "num_workers": config.num_workers,
        "pin_memory": config.pin_memory,
        "dtype": config.dtype,
        "batch_size": config.batch_size,
        "tokenize_offline": config.tokenize_offline,
    }
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
