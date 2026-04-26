import numpy as np
import psutil
import torch


# Peak FP32 TFLOPS for M4 Pro 20-core GPU: ~8.1 TFLOPS
# Source: nanoreview.net third-party benchmarks — Apple does not publish official TFLOPS
# Note: MPS lacks dedicated Tensor Cores. This peak is FP32 scalar compute.
# MFU here measures utilization of raw FP32 throughput, not tensor throughput.
# Numbers are relative indicators of efficiency, not comparable to CUDA MFU figures.
PEAK_FLOPS_FP32 = 8.1e12  # 8.1 TFLOPS — M4 Pro 20-core GPU

GPT2_SMALL_PARAMS = 117_000_000


def compute_mfu(
    tokens_per_sec: float,
    n_params: int = GPT2_SMALL_PARAMS,
    peak_flops: float = PEAK_FLOPS_FP32,
) -> float:
    """MFU = (tokens/sec × 6 × N_params) / peak_flops. 6N = 2x fwd + 4x bwd."""
    if tokens_per_sec == 0.0:
        return 0.0
    return (tokens_per_sec * 6 * n_params) / peak_flops


def compute_gpu_idle_pct(
    step_times: list[float],
    load_times: list[float],
) -> float:
    """GPU idle % = mean(load_time / step_time) × 100 across all steps."""
    if not step_times:
        return 0.0
    ratios = [lt / st for lt, st in zip(load_times, step_times) if st > 0]
    if not ratios:
        return 0.0
    return (sum(ratios) / len(ratios)) * 100.0


def compute_percentile(values: list[float], p: int) -> float:
    return float(np.percentile(values, p))


def get_memory_headroom_pct() -> float:
    """Available system RAM as % of total (Apple unified memory proxy)."""
    vm = psutil.virtual_memory()
    return (vm.available / vm.total) * 100.0
