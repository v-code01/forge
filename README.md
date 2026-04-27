# Forge

Training efficiency profiler and optimizer for GPT-2 Small on Apple Silicon (MPS).

Forge diagnoses GPU starvation in a deliberately broken training loop, applies five sequential fixes, and measures the per-fix throughput delta. The goal is to make the distinction between *GPU utilization %* and *Model FLOP Utilization (MFU)* concrete and measurable — and to show how the same diagnostic methodology extends to multi-node distributed training.

## Results (M4 Pro, 20-core GPU, 100 steps)

| Metric | Baseline | Optimized |
|---|---|---|
| Tokens/sec | 2,860 | 4,769 |
| MFU | 24.8% | 41.3% |
| GPU idle | 29.7% | 0.8% |
| Step time p50 | 356 ms | 820 ms |

**1.7× throughput. GPU idle drops from 30% → 0.8%.**

### Fix attribution

| Fix | Before | After | Δ |
|---|---|---|---|
| `num_workers` (0 → 7) | 2,860 tok/s | 4,127 tok/s | **+44%** |
| `pin_memory` | 4,127 tok/s | 4,121 tok/s | −0% (MPS no-op) |
| `bfloat16` | 4,121 tok/s | 4,100 tok/s | −1% |
| `batch_size` (8 → 32) | 4,100 tok/s | 4,804 tok/s | **+17%** |
| `tokenize_offline` | 4,804 tok/s | 4,769 tok/s | −1% |

The two fixes that matter are `num_workers` and `batch_size`. `pin_memory` is silently ignored by MPS (PyTorch warns). `bfloat16` and `tokenize_offline` are marginal at GPT-2 Small scale — the former because MPS has no dedicated BF16 tensor units, the latter because parallel workers already hide tokenization latency.

## Scaling to multi-node: the `comms_frac` metric

On a single GPU, the dominant bottleneck is DataLoader starvation (`gpu_idle_pct`). On a multi-node cluster, once DataLoader starvation is fixed, the dominant bottleneck becomes gradient synchronisation: the time each GPU waits for an NCCL allreduce to complete before the next forward pass can start.

Forge includes a DDP probe that makes this cost measurable:

```bash
python forge.py --ddp-probe --steps 20
```

```
DDP probe — GLOO backend, 20 steps per world size
Model: ~4M params (~16MB gradient per allreduce)

  Workers    Compute   Allreduce   Comms%      Tok/s
  ----------------------------------------------------
        1        68ms        1.2ms     1.8%      14808
        2       119ms       39.0ms    24.7%       6479
        4       221ms       77.5ms    26.0%       3431

  Note: GLOO/localhost uses shared-memory memcpy — allreduce cost is
  ~10-100x lower than NCCL/EFA over a real network fabric.
  On 8×A100 with 100GbE, comms_frac is typically 15–40% for GPT-2 scale.
  comms_frac is the distributed analog of Forge's gpu_idle_pct.
```

`comms_frac = allreduce_time / step_time` — exactly what `gpu_idle_pct` measures for DataLoader starvation, applied to the distributed bottleneck. The fix is the same class of solution: overlap communication with computation (gradient bucketing, async allreduce, FSDP sharding) just as DataLoader starvation is fixed by prefetching with `num_workers`.

### Mapping single-GPU findings to 8×A100

| Single-GPU | Multi-node |
|---|---|
| `gpu_idle_pct` | `comms_frac` |
| DataLoader starvation | NCCL allreduce wait |
| Fix: `num_workers` → parallel prefetch | Fix: gradient bucketing + async allreduce |
| Fix: `tokenize_offline` → eliminate I/O | Fix: FSDP → reduce per-rank gradient size |
| `load_frac` in step profile | `comms_frac` in step profile |

The profiling infrastructure in `forge/profiler.py` (`ProfileResult`, per-phase timing, percentile aggregation) transfers directly to a distributed context by substituting `allreduce_time` for `load_time` in the step timing loop.

## What Forge measures

**MFU** = (tokens/sec × 6 × N\_params) / peak\_FLOPS

The 6× is the standard training FLOPs estimate: ~1× forward, ~2× backward (chain rule), ~1× gradient accumulation. For GPT-2 Small (117 M params) on M4 Pro (8.1 TFLOPS FP32), MFU is a relative efficiency indicator — not comparable to CUDA figures because MPS has no Tensor Cores.

**GPU idle %** = mean(load\_time / step\_time) × 100

Measures DataLoader starvation: time the GPU waits for the next batch rather than computing. At baseline, 30% of every step is wasted this way.

**comms\_frac** = mean(allreduce\_time / step\_time) × 100

Measures gradient synchronisation overhead in distributed training. The same bottleneck category, different layer of the stack.

## Setup

```bash
git clone https://github.com/v-code01/forge
cd forge
pip install -r requirements.txt
```

Python 3.13, PyTorch 2.4+. `--baseline`, `--optimize`, `--config-search` require Apple Silicon. `--ddp-probe` runs on any hardware (CPU + GLOO, no GPU needed).

## Usage

```bash
# Profile the broken baseline (num_workers=0, fp32, batch=8, live tokenization)
python forge.py --baseline --steps 100

# Apply all 5 fixes sequentially, show before/after table
python forge.py --optimize --steps 100

# Grid search optimal num_workers × batch_size
python forge.py --config-search --steps 50

# Show run history from SQLite
python forge.py --report

# Measure gradient allreduce overhead vs world size (no GPU required)
python forge.py --ddp-probe --steps 20
```

Results are saved to `results/{timestamp}.json` and `results/forge.db`.

## Stack

Python 3.13 · PyTorch 2.4 + MPS · transformers · datasets · rich · SQLite · psutil

## Tests

```bash
pytest tests/ -v   # 30 tests, no MPS hardware required for unit tests
```

Unit tests run without MPS — hardware-dependent calls are mocked at the boundary. `--ddp-probe` tests use GLOO on CPU and run on any machine.
