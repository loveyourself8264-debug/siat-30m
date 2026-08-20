"""Tests for warmup + cosine learning-rate schedule."""

from __future__ import annotations

import math

import pytest

from config import TrainConfig
from train.trainer import get_learning_rate


def _cfg(**kwargs) -> TrainConfig:
    defaults = dict(
        batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        min_learning_rate=1e-4,
        weight_decay=0.1,
        warmup_steps=10,
        max_steps=110,
        max_grad_norm=1.0,
        seed=42,
    )
    defaults.update(kwargs)
    return TrainConfig(**defaults)


def test_first_step_lr():
    cfg = _cfg(learning_rate=1e-3, warmup_steps=10)
    # step 0 → peak * 1 / 10
    assert abs(get_learning_rate(0, cfg) - 1e-4) < 1e-12


def test_warmup_final_reaches_peak():
    cfg = _cfg(learning_rate=1e-3, warmup_steps=10)
    assert abs(get_learning_rate(9, cfg) - 1e-3) < 1e-12


def test_cosine_middle():
    cfg = _cfg(learning_rate=1e-3, min_learning_rate=1e-4, warmup_steps=10, max_steps=110)
    # Midpoint of cosine: progress=0.5 → cos(pi/2)=0 → lr = min + 0.5*(peak-min)
    mid_step = 10 + (110 - 10) // 2  # 60
    expected = 1e-4 + 0.5 * (1e-3 - 1e-4)
    assert abs(get_learning_rate(mid_step, cfg) - expected) < 1e-9


def test_final_lr_near_min():
    cfg = _cfg(learning_rate=1e-3, min_learning_rate=1e-4, warmup_steps=10, max_steps=110)
    # Last training step index is max_steps - 1
    lr = get_learning_rate(109, cfg)
    progress = (109 - 10) / (110 - 10)
    expected = 1e-4 + 0.5 * (1.0 + math.cos(math.pi * progress)) * (1e-3 - 1e-4)
    assert abs(lr - expected) < 1e-12
    assert lr >= cfg.min_learning_rate - 1e-15


def test_clamp_beyond_max_steps():
    cfg = _cfg(min_learning_rate=1e-4, max_steps=110, warmup_steps=10)
    assert get_learning_rate(110, cfg) == 1e-4
    assert get_learning_rate(999, cfg) == 1e-4


def test_cosine_never_below_min():
    cfg = _cfg(learning_rate=1e-3, min_learning_rate=1e-4, warmup_steps=5, max_steps=50)
    for step in range(cfg.max_steps + 5):
        assert get_learning_rate(step, cfg) >= cfg.min_learning_rate - 1e-15


def test_invalid_step():
    cfg = _cfg()
    with pytest.raises(ValueError, match="step"):
        get_learning_rate(-1, cfg)


def test_invalid_config_warmup_vs_max():
    with pytest.raises(ValueError, match="warmup_steps"):
        _cfg(warmup_steps=100, max_steps=50)
