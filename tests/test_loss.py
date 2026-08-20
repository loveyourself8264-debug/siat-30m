"""Tests for causal_lm_loss."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from train.loss import causal_lm_loss


def test_valid_loss_is_scalar():
    logits = torch.randn(2, 4, 16)
    labels = torch.randint(0, 16, (2, 4))
    loss = causal_lm_loss(logits, labels)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_perfect_prediction_near_zero():
    # V=4, all positions predict class 2 with logit 10
    logits = torch.zeros(1, 3, 4)
    logits[..., 2] = 10.0
    labels = torch.full((1, 3), 2, dtype=torch.long)
    loss = causal_lm_loss(logits, labels)
    assert loss.item() < 0.01


def test_uniform_logits_approx_ln_v():
    v = 8
    logits = torch.zeros(2, 4, v)
    labels = torch.randint(0, v, (2, 4))
    loss = causal_lm_loss(logits, labels)
    assert abs(loss.item() - math.log(v)) < 1e-5


def test_shape_mismatch_raises():
    logits = torch.randn(2, 4, 8)
    labels = torch.randint(0, 8, (2, 3))
    with pytest.raises(ValueError, match="must match"):
        causal_lm_loss(logits, labels)


def test_wrong_rank_raises():
    with pytest.raises(ValueError, match="rank 3"):
        causal_lm_loss(torch.randn(4, 8), torch.randint(0, 8, (4,)))
    with pytest.raises(ValueError, match="rank 2"):
        causal_lm_loss(torch.randn(2, 4, 8), torch.randint(0, 8, (2, 4, 1)))


def test_labels_must_be_long():
    logits = torch.randn(1, 2, 4)
    labels = torch.zeros(1, 2)  # float
    with pytest.raises(ValueError, match="long"):
        causal_lm_loss(logits, labels)


def test_gradient_flow():
    logits = torch.randn(2, 4, 16, requires_grad=True)
    labels = torch.randint(0, 16, (2, 4))
    loss = causal_lm_loss(logits, labels)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert not torch.isnan(logits.grad).any()


def test_no_double_shift_matches_direct_ce():
    """Loss must equal CE on full [B,S] without slicing logits/labels."""
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 10)
    labels = torch.randint(0, 10, (2, 5))
    ours = causal_lm_loss(logits, labels)
    direct = F.cross_entropy(
        logits.reshape(-1, 10),
        labels.reshape(-1),
    )
    assert torch.allclose(ours, direct)

    # Wrong double-shift would differ on this tensor
    wrong = F.cross_entropy(
        logits[:, :-1].reshape(-1, 10),
        labels[:, 1:].reshape(-1),
    )
    assert not torch.allclose(ours, wrong)
