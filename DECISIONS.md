# Design Decisions

## Why MPS over CUDA

Built on the hardware available. The profiling methodology — MFU calculation, DataLoader starvation detection, per-fix waterfall attribution — is hardware-agnostic. The same five bottleneck categories (DataLoader workers, memory pinning, dtype precision, batch size, tokenization strategy) exist on CUDA clusters at larger scale. MPS was the constraint, not the scope.

One MPS-specific caveat worth noting: there are no Tensor Cores. `PEAK_FLOPS_FP32 = 8.1e12` is raw FP32 scalar throughput for the M4 Pro 20-core GPU. MFU here is a relative efficiency indicator within a single hardware configuration, not a figure comparable to H100 MFU benchmarks that assume tensor-accelerated throughput.

## Why the 100ms artificial stall

The HuggingFace fast tokenizer (Rust-backed) processes a batch of 8 sequences in ~2–5ms on M4 Pro. A GPU forward+backward pass for GPT-2 Small at `batch_size=8` takes ~250ms. Without an explicit stall, `load_time / step_time ≈ 1–2%` — below noise floor, not diagnosable.

The 100ms sleep in `_OnlineCollate.__call__` raises GPU idle to ~30%, making the DataLoader the clear bottleneck. This is documented in the source as `_BASELINE_COLLATE_SLEEP_S` with an explanation of why it's there. The stall models what actually happens in real workloads at scale: slow storage, large vocabulary tokenization, remote dataset streaming, or a preprocessing pipeline that hasn't been moved offline. The sleep is a controlled stand-in for any of those.

## Why 100 steps is sufficient for diagnosis

Infrastructure bottlenecks don't require model convergence to measure. DataLoader starvation, memory transfer overhead, and compute utilization stabilize within the first 30–50 steps once MPS shader compilation finishes (step 1 is slow; by step 10 the JIT is warm). Running 100 steps gives a clean 50-step warmup window and 50 steps of steady-state data. The per-step variance at steady state is under 5%.

For convergence studies — loss curves, generalization, hyperparameter sensitivity — you'd need more. Forge is not that tool.

## Why pin_memory and bfloat16 show flat or negative deltas

Both are genuine findings, not failures.

`pin_memory=True` on MPS is silently ignored. PyTorch emits a warning: *"pin_memory argument is set as true but not supported on MPS now, device pinned memory won't be used."* Apple Silicon uses unified memory — there is no separate device memory to pin to. The fix is architecturally inapplicable on this hardware. Forge reports the honest zero.

`bfloat16` via `torch.autocast("mps", dtype=torch.bfloat16)` shows a −1% delta at `batch_size=8`. M4 Pro has no dedicated BF16 matrix units; the gain from reduced memory bandwidth is offset by mixed-precision overhead at small batch sizes. At `batch_size=128+` on larger models the picture changes. The tool reporting a negative is the correct output — it means the fix was applied, measured, and found to be non-beneficial on this specific hardware and model scale.
