import pytest
from forge.harness import StepProfile
from forge.profiler import ProfileResult, _aggregate


def _step(i, load, transfer, forward, backward, opt):
    total = load + transfer + forward + backward + opt
    return StepProfile(
        step_idx=i,
        load_time=load,
        transfer_time=transfer,
        forward_time=forward,
        backward_time=backward,
        optimizer_time=opt,
        total_time=total,
        tokens=128 * 8,  # seq_len=128, batch=8 → 1024 tok
    )


def test_aggregate_returns_profile_result():
    steps = [_step(i, 0.4, 0.05, 0.3, 0.2, 0.05) for i in range(3)]
    assert isinstance(_aggregate(steps), ProfileResult)


def test_aggregate_gpu_idle_pct():
    steps = [_step(i, 0.4, 0.05, 0.3, 0.2, 0.05) for i in range(5)]
    result = _aggregate(steps)
    # load=0.4, total=1.0 → idle=40%
    assert result.gpu_idle_pct == pytest.approx(40.0, abs=0.1)


def test_aggregate_tokens_per_sec():
    # 1024 tok per step, 1.0s per step → 1024 tok/s
    steps = [_step(i, 0.4, 0.05, 0.3, 0.2, 0.05) for i in range(5)]
    result = _aggregate(steps)
    assert result.tokens_per_sec == pytest.approx(1024.0, abs=1.0)


def test_aggregate_step_fractions_sum_to_one():
    steps = [_step(i, 0.4, 0.05, 0.3, 0.2, 0.05) for i in range(5)]
    result = _aggregate(steps)
    total = (result.load_frac + result.transfer_frac + result.forward_frac
             + result.backward_frac + result.optimizer_frac)
    assert total == pytest.approx(1.0, abs=0.001)


def test_aggregate_mfu_is_positive():
    steps = [_step(i, 0.4, 0.05, 0.3, 0.2, 0.05) for i in range(5)]
    result = _aggregate(steps)
    assert result.mfu > 0.0


def test_aggregate_step_time_percentiles_uniform():
    # All identical steps → p50 == p90 == 1.0
    steps = [_step(i, 0.4, 0.05, 0.3, 0.2, 0.05) for i in range(10)]
    result = _aggregate(steps)
    assert result.step_time_p50 == pytest.approx(1.0, abs=0.001)
    assert result.step_time_p90 == pytest.approx(1.0, abs=0.001)


def test_aggregate_memory_headroom_in_range():
    steps = [_step(i, 0.4, 0.05, 0.3, 0.2, 0.05) for i in range(3)]
    result = _aggregate(steps)
    assert 0.0 <= result.memory_headroom_pct <= 100.0


def test_aggregate_transfer_time_mean():
    steps = [_step(i, 0.4, 0.05, 0.3, 0.2, 0.05) for i in range(5)]
    result = _aggregate(steps)
    assert result.transfer_time_mean == pytest.approx(0.05, abs=0.001)
