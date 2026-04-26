import pytest
from forge.metrics import (
    PEAK_FLOPS_FP32,
    GPT2_SMALL_PARAMS,
    compute_mfu,
    compute_gpu_idle_pct,
    compute_percentile,
    get_memory_headroom_pct,
)


def test_peak_flops_constant():
    assert PEAK_FLOPS_FP32 == 8.1e12


def test_gpt2_small_params_constant():
    assert GPT2_SMALL_PARAMS == 117_000_000


def test_compute_mfu_known_value():
    # (1000 * 6 * 117_000_000) / 8.1e12 = 702e9 / 8.1e12 ≈ 0.08666
    result = compute_mfu(tokens_per_sec=1000.0)
    assert abs(result - 0.08666) < 0.0001


def test_compute_mfu_zero_tokens():
    assert compute_mfu(0.0) == 0.0


def test_compute_mfu_custom_params_and_peak():
    # (1.0 * 6 * 1) / 6.0 = 1.0
    assert compute_mfu(tokens_per_sec=1.0, n_params=1, peak_flops=6.0) == 1.0


def test_compute_gpu_idle_pct_basic():
    # load takes 40% of each step → idle = 40%
    step_times = [1.0, 1.0, 1.0]
    load_times = [0.4, 0.4, 0.4]
    assert compute_gpu_idle_pct(step_times, load_times) == pytest.approx(40.0)


def test_compute_gpu_idle_pct_no_idle():
    step_times = [1.0, 1.0]
    load_times = [0.0, 0.0]
    assert compute_gpu_idle_pct(step_times, load_times) == 0.0


def test_compute_gpu_idle_pct_empty_lists():
    assert compute_gpu_idle_pct([], []) == 0.0


def test_compute_percentile_p50():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert compute_percentile(values, 50) == pytest.approx(3.0)


def test_compute_percentile_p90():
    values = [float(i) for i in range(1, 11)]  # 1.0 .. 10.0
    result = compute_percentile(values, 90)
    assert 8.5 < result < 9.5


def test_get_memory_headroom_pct_in_valid_range():
    pct = get_memory_headroom_pct()
    assert 0.0 <= pct <= 100.0
