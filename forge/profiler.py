from dataclasses import dataclass

from forge.harness import HarnessConfig, StepProfile, run_training, BASELINE_CONFIG
from forge.metrics import (
    compute_mfu,
    compute_gpu_idle_pct,
    compute_percentile,
    get_memory_headroom_pct,
)


@dataclass
class ProfileResult:
    step_profiles: list[StepProfile]
    gpu_idle_pct: float
    mfu: float
    tokens_per_sec: float
    memory_headroom_pct: float
    step_time_p50: float
    step_time_p90: float
    transfer_time_mean: float
    load_frac: float
    transfer_frac: float
    forward_frac: float
    backward_frac: float
    optimizer_frac: float


def _aggregate(steps: list[StepProfile]) -> ProfileResult:
    """Aggregate a list of StepProfile records into a single ProfileResult.

    Args:
        steps: Non-empty list of per-step timing profiles from run_training().

    Returns:
        ProfileResult with derived metrics:
        - gpu_idle_pct: mean fraction of step time spent waiting for data × 100
        - mfu: Model FLOP Utilization (dimensionless, 0–1 range typical for MPS)
        - tokens_per_sec: total tokens / total wall-clock time
        - memory_headroom_pct: available unified RAM as % of total (via psutil)
        - step_time_p50/p90: numpy-interpolated percentiles of per-step wall time
        - transfer_time_mean: mean CPU→GPU copy latency in seconds
        - {load,transfer,forward,backward,optimizer}_frac: time budget breakdown,
          each as fraction of mean step time; they sum to exactly 1.0

    Complexity: O(n) time, O(n) space for the step_times/load_times lists.
    Side effects: calls get_memory_headroom_pct() which invokes psutil (syscall).
    """
    step_times = [s.total_time for s in steps]
    load_times = [s.load_time for s in steps]

    total_tokens = sum(s.tokens for s in steps)
    total_time = sum(step_times)
    tokens_per_sec = total_tokens / total_time if total_time > 0 else 0.0

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    mean_total = _mean(step_times)

    # Each fraction = mean(phase_time) / mean(total_time).
    # Since all steps have identical timing in the uniform case, this is equivalent
    # to phase_time / total_time and the five fractions sum exactly to 1.0.
    def _frac(attr: str) -> float:
        if mean_total <= 0.0:
            return 0.0
        return _mean([getattr(s, attr) for s in steps]) / mean_total

    return ProfileResult(
        step_profiles=steps,
        gpu_idle_pct=compute_gpu_idle_pct(step_times, load_times),
        mfu=compute_mfu(tokens_per_sec),
        tokens_per_sec=tokens_per_sec,
        memory_headroom_pct=get_memory_headroom_pct(),
        step_time_p50=compute_percentile(step_times, 50),
        step_time_p90=compute_percentile(step_times, 90),
        transfer_time_mean=_mean([s.transfer_time for s in steps]),
        load_frac=_frac("load_time"),
        transfer_frac=_frac("transfer_time"),
        forward_frac=_frac("forward_time"),
        backward_frac=_frac("backward_time"),
        optimizer_frac=_frac("optimizer_time"),
    )


def profile(config: HarnessConfig = BASELINE_CONFIG) -> ProfileResult:
    """Run a full training profile and return aggregated metrics.

    Args:
        config: HarnessConfig controlling batch size, dtype, num_workers, n_steps.

    Returns:
        ProfileResult aggregated over config.n_steps training steps.

    Note: Requires an MPS-capable device (Apple Silicon). Tests should call
    _aggregate() directly with synthetic StepProfile data to avoid this dependency.
    """
    steps = run_training(config)
    return _aggregate(steps)
