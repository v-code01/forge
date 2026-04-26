import dataclasses
from unittest.mock import patch

import pytest

from forge.harness import BASELINE_CONFIG
from forge.profiler import ProfileResult
from forge.harness import StepProfile


def _make_profile(tokens_per_sec: float) -> ProfileResult:
    return ProfileResult(
        step_profiles=[],
        gpu_idle_pct=40.0,
        mfu=0.01,
        tokens_per_sec=tokens_per_sec,
        memory_headroom_pct=70.0,
        step_time_p50=0.5,
        step_time_p90=0.8,
        transfer_time_mean=0.01,
        load_frac=0.40,
        transfer_frac=0.05,
        forward_frac=0.30,
        backward_frac=0.20,
        optimizer_frac=0.05,
    )


@patch("forge.optimizer.profile")
def test_optimize_returns_optimized_result(mock_profile):
    from forge.optimizer import optimize, OptimizeResult
    # 6 calls: 1 baseline + 5 fix profiles (last fix profile becomes .optimized)
    mock_profile.side_effect = [_make_profile(t) for t in [100, 400, 450, 500, 550, 600]]
    result = optimize(BASELINE_CONFIG)
    assert isinstance(result, OptimizeResult)
    assert result.baseline.tokens_per_sec == 100.0
    assert result.optimized.tokens_per_sec == 600.0


@patch("forge.optimizer.profile")
def test_optimize_has_five_fixes(mock_profile):
    from forge.optimizer import optimize
    mock_profile.side_effect = [_make_profile(t) for t in [100, 400, 450, 500, 550, 600]]
    result = optimize(BASELINE_CONFIG)
    assert len(result.fixes) == 5


@patch("forge.optimizer.profile")
def test_optimize_fix_names_in_order(mock_profile):
    from forge.optimizer import optimize
    mock_profile.side_effect = [_make_profile(t) for t in [100, 400, 450, 500, 550, 600]]
    result = optimize(BASELINE_CONFIG)
    assert [f.fix_name for f in result.fixes] == [
        "num_workers", "pin_memory", "bfloat16", "batch_size", "tokenize_offline"
    ]


@patch("forge.optimizer.profile")
def test_optimize_first_fix_delta(mock_profile):
    from forge.optimizer import optimize
    mock_profile.side_effect = [_make_profile(t) for t in [100, 400, 450, 500, 550, 600]]
    result = optimize(BASELINE_CONFIG)
    # Fix 1: 100 → 400, delta = (400-100)/100 * 100 = 300%
    assert result.fixes[0].delta_pct == pytest.approx(300.0)
    assert result.fixes[0].fix_name == "num_workers"


@patch("forge.optimizer.profile")
def test_optimize_largest_gain_identified(mock_profile):
    from forge.optimizer import optimize
    mock_profile.side_effect = [_make_profile(t) for t in [100, 400, 450, 500, 550, 600]]
    result = optimize(BASELINE_CONFIG)
    # Fix 1: +300%, Fix 2: +12.5%, Fix 3: +11.1%, Fix 4: +10%, Fix 5: +9%
    assert result.largest_gain_fix == "num_workers"
