# Forge

Training efficiency profiler and optimizer for GPT-2 Small on Apple Silicon (MPS).

Forge diagnoses GPU starvation in a deliberately broken training loop, applies five sequential fixes, and measures the per-fix throughput delta. The goal is to make the distinction between *GPU utilization %* and *Model FLOP Utilization (MFU)* concrete and measurable.

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

## What Forge measures

**MFU** = (tokens/sec × 6 × N\_params) / peak\_FLOPS

The 6× comes from the standard training FLOPs estimate: ~1× forward, ~2× backward (chain rule), ~1× gradient accumulation. For GPT-2 Small (117 M params) on M4 Pro (8.1 TFLOPS FP32), MFU is a relative efficiency indicator — not comparable to CUDA figures because MPS has no Tensor Cores.

**GPU idle %** = mean(load\_time / step\_time) × 100

This measures DataLoader starvation: time the GPU is waiting for the next batch rather than computing. At baseline, 30% of every step is wasted this way.

## Setup

```bash
git clone https://github.com/v-code01/forge
cd forge
pip install -r requirements.txt
```

Python 3.13, PyTorch 2.4+, Apple Silicon required.

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
```

Results are saved to `results/{timestamp}.json` and `results/forge.db`.

## Stack

Python 3.13 · PyTorch 2.4 + MPS · transformers · datasets · rich · SQLite · psutil

## Tests

```bash
pytest tests/ -v   # 26 tests, no hardware required
```

All unit tests run without MPS — hardware-dependent calls (`run_training`, `profile`, `optimize`) are mocked at the boundary.
