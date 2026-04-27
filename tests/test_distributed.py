import pytest
from forge.distributed import DDPStepProfile, probe_ddp


def test_ddp_step_profile_fields():
    p = DDPStepProfile(
        world_size=2,
        n_params=4_198_400,
        compute_ms=150.0,
        allreduce_ms=18.0,
        comms_frac=0.107,
        tokens_per_sec=800.0,
    )
    assert p.world_size == 2
    assert p.n_params == 4_198_400
    assert p.comms_frac == pytest.approx(0.107)


def test_ddp_probe_single_worker():
    results = probe_ddp(world_sizes=[1], n_steps=5)
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, DDPStepProfile)
    assert r.world_size == 1
    assert r.n_params > 0
    assert r.compute_ms > 0
    assert r.allreduce_ms >= 0
    assert 0.0 <= r.comms_frac <= 1.0
    assert r.tokens_per_sec > 0


def test_ddp_probe_returns_one_profile_per_world_size():
    results = probe_ddp(world_sizes=[1, 2], n_steps=3)
    assert len(results) == 2
    assert results[0].world_size == 1
    assert results[1].world_size == 2


def test_ddp_probe_comms_frac_increases_with_world_size():
    # With more workers, allreduce coordinates more processes → higher comms_frac.
    # On GLOO/localhost the gap is small; on real networks it dominates.
    results = probe_ddp(world_sizes=[1, 4], n_steps=5)
    assert results[1].comms_frac >= results[0].comms_frac
