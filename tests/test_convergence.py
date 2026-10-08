from dataclasses import asdict

import pytest

from autoosu.ml.convergence import Convergence, attribute_nll


def test_plateau_reduces_lr_before_stopping_at_floor():
    p = Convergence()
    assert p.observe(.7, 1., 0) == "improved"
    actions = [p.observe(.7, 1., step * 1000) for step in range(1, 21)]
    assert actions.count("reduce_lr") == 4
    assert actions[-1] == "converged"
    assert "converged" not in actions[:-1]
    assert p.lr == p.min_lr


def test_learning_either_task_prevents_premature_stop():
    p = Convergence(lr=3.125e-6)
    p.observe(.7, 1., 0)
    for step in range(1, 8):
        assert p.observe(.7, 1., 10000 + step) != "converged"
    assert p.observe(.704, 1., 11000) == "improved"
    assert p.floor_stale == 0
    assert p.observe(.7, .99, 12000) == "improved"


def test_metric_noise_and_state_roundtrip():
    p = Convergence()
    p.observe(.7, 1., 0)
    assert p.observe(.701, .999, 1000) == "continue"
    restored = Convergence(**asdict(p))
    for i in range(2, 22):
        assert restored.observe(.7, 1., i * 1000) == p.observe(.7, 1., i * 1000)
    assert asdict(restored) == asdict(p)


def test_stop_requires_minimum_training():
    p = Convergence(lr=3.125e-6)
    p.observe(.7, 1., 0)
    for i in range(1, 10):
        assert p.observe(.7, 1., i) != "converged"


def test_invalid_metrics_and_missing_supervision_fail():
    with pytest.raises(FloatingPointError):
        Convergence().observe(float("nan"), 1., 0)
    with pytest.raises(ValueError):
        attribute_nll({"attributes": {}})
    assert attribute_nll({"attributes": {"a": {"reference_count": 2, "reference_nll": 4.},
                                        "b": {"reference_count": 4, "reference_nll": 4.}}}) == 1.5


def test_full_precision_adam_state_preserves_next_update(tmp_path):
    import torch
    from autoosu.ml.train_convergence import atomic_save

    torch.manual_seed(7)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    x = torch.randn(4, 3)
    def update(net, opt):
        opt.zero_grad()
        net(x).square().sum().backward()
        opt.step()
    update(model, optimizer)
    path = tmp_path / "state.pt"
    atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), path)
    saved = torch.load(path, weights_only=True)
    restored = torch.nn.Linear(3, 2)
    restored.load_state_dict(saved["model"])
    restored_optimizer = torch.optim.AdamW(restored.parameters())
    restored_optimizer.load_state_dict(saved["optimizer"])
    update(model, optimizer)
    update(restored, restored_optimizer)
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
