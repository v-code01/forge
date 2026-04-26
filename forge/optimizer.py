import os
from dataclasses import dataclass, replace

from forge.harness import HarnessConfig, BASELINE_CONFIG
from forge.profiler import ProfileResult, profile


@dataclass
class FixResult:
    fix_name: str
    tokens_per_sec_before: float
    tokens_per_sec_after: float
    delta_pct: float


@dataclass
class OptimizeResult:
    baseline: ProfileResult
    optimized: ProfileResult
    fixes: list[FixResult]
    largest_gain_fix: str


# Canonical fix application order — each fix builds on the previous config state.
# Order chosen to maximize waterfall legibility: data-loading wins surface first
# (num_workers, pin_memory), then compute wins (bfloat16), then throughput
# amplifiers (batch_size), then pre-processing amortisation (tokenize_offline).
_FIX_ORDER = ["num_workers", "pin_memory", "bfloat16", "batch_size", "tokenize_offline"]


def _apply_fix(config: HarnessConfig, fix_name: str) -> HarnessConfig:
    """Return a new HarnessConfig with the named fix applied.

    Args:
        config: Current cumulative config (previous fixes already applied).
        fix_name: One of the names in _FIX_ORDER.

    Returns:
        New HarnessConfig with exactly one field mutated.

    Raises:
        ValueError: If fix_name is not recognised.
    """
    if fix_name == "num_workers":
        # Use half the logical CPUs — enough to saturate prefetch without
        # competing with the MPS process (which uses the P-cores heavily).
        return replace(config, num_workers=max(1, (os.cpu_count() or 4) // 2))
    if fix_name == "pin_memory":
        # Page-locked host memory enables async DMA to the GPU, eliminating
        # the CPU stall during H→D transfer when num_workers > 0.
        return replace(config, pin_memory=True)
    if fix_name == "bfloat16":
        # BF16 halves bandwidth on weight loads and doubles effective FLOPS
        # on MPS (no fp16 GEMM, but BF16 is natively supported from M2+).
        return replace(config, dtype="bfloat16")
    if fix_name == "batch_size":
        # 32 is 4× the baseline (8) — saturates the MPS queue without OOM
        # on 16 GB unified memory with GPT-2 Small (117 M params).
        return replace(config, batch_size=32)
    if fix_name == "tokenize_offline":
        # Pre-tokenise the dataset once and cache as tensors, eliminating
        # the per-batch tokenizer call that dominates load_time at baseline.
        return replace(config, tokenize_offline=True)
    raise ValueError(f"Unknown fix: {fix_name}")


def optimize(baseline_config: HarnessConfig = BASELINE_CONFIG) -> OptimizeResult:
    """Apply _FIX_ORDER fixes sequentially; measure per-fix token throughput delta.

    Each fix is profiled against the accumulated config (prior fixes remain in
    effect), so delta_pct reflects the marginal gain of that fix alone rather
    than the gain from the full optimised stack.

    Args:
        baseline_config: Starting HarnessConfig. Defaults to BASELINE_CONFIG.

    Returns:
        OptimizeResult with:
        - baseline: ProfileResult from the unmodified baseline_config.
        - optimized: ProfileResult from the fully-patched config (all 5 fixes).
        - fixes: List of 5 FixResult records in _FIX_ORDER sequence.
        - largest_gain_fix: fix_name with the highest delta_pct.

    Complexity: O(|_FIX_ORDER|) profile() calls = 6 total (1 baseline + 5 fixes).
    Side effects: calls profile() which calls run_training() — MPS device required
    in production; tests should mock forge.optimizer.profile.
    """
    baseline = profile(baseline_config)
    fixes: list[FixResult] = []
    current_config = baseline_config
    prev_tps = baseline.tokens_per_sec
    last_profile = baseline

    for fix_name in _FIX_ORDER:
        new_config = _apply_fix(current_config, fix_name)
        new_result = profile(new_config)
        # Marginal delta relative to the immediately preceding state, not baseline.
        # This correctly attributes multi-fix synergies to whichever fix triggers them.
        delta_pct = (
            (new_result.tokens_per_sec - prev_tps) / prev_tps * 100.0
            if prev_tps > 0
            else 0.0
        )
        fixes.append(FixResult(
            fix_name=fix_name,
            tokens_per_sec_before=prev_tps,
            tokens_per_sec_after=new_result.tokens_per_sec,
            delta_pct=delta_pct,
        ))
        current_config = new_config
        prev_tps = new_result.tokens_per_sec
        last_profile = new_result

    largest = max(fixes, key=lambda f: f.delta_pct)
    return OptimizeResult(
        baseline=baseline,
        optimized=last_profile,
        fixes=fixes,
        largest_gain_fix=largest.fix_name,
    )
