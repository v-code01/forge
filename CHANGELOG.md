# Changelog

## v0.3.0 — Config search, CLI, full integration

Complete command-line interface with four commands: `--baseline`, `--optimize`, `--config-search`, `--report`. Grid search over `num_workers × batch_size` finds the optimal DataLoader configuration for the hardware. Results persist to JSON and SQLite. Benchmark report rendered as a Rich terminal table with fix-attribution waterfall. README includes real numbers from a 100-step run on M4 Pro.

Measured on M4 Pro (20-core GPU, 8.1 TFLOPS FP32):
- Baseline: 2,860 tok/s · 24.8% MFU · 30% GPU idle
- Optimized: 4,769 tok/s · 41.3% MFU · 0.8% GPU idle
- Speedup: 1.7× · largest gain: `num_workers` +44%

## v0.2.0 — Optimizer with fix attribution

Sequential waterfall optimizer applies five fixes in a defined order, profiles after each one, and measures the marginal delta against the immediately preceding state. This correctly attributes multi-fix synergies to whichever fix triggers them rather than summing from baseline. `OptimizeResult` captures per-fix `tokens_per_sec_before`, `tokens_per_sec_after`, and `delta_pct`, plus the `largest_gain_fix` across the run.

Fix order: `num_workers` → `pin_memory` → `bfloat16` → `batch_size` → `tokenize_offline`. Ordered to surface DataLoader wins first (highest leverage), then compute wins, then preprocessing amortization.

## v0.1.0 — Baseline profiler

Core profiling infrastructure: `HarnessConfig`, `StepProfile`, `run_training()`. Runs GPT-2 Small (117M params) on MPS with per-phase timing fenced by `torch.mps.synchronize()` — required because MPS command submission is asynchronous. Without the fence, timestamps measure submission latency, not execution time.

`ProfileResult` aggregates per-step data into: `gpu_idle_pct`, `mfu`, `tokens_per_sec`, step-time percentiles (p50/p90), phase fractions (load/transfer/forward/backward/optimizer), `memory_headroom_pct`, and `transfer_time_mean`. `compute_mfu` uses the standard 6N FLOPs estimate for training (1× forward + 2× backward + 1× gradient accumulation).

26 unit tests. No MPS hardware required to run the test suite — hardware-dependent calls are mocked at the boundary.
