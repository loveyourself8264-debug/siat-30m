"""Tests for Siat causal attention mask (no production softmax)."""

from __future__ import annotations

import pytest
import torch

from config import ModelConfig
from model.attention import (
    SiatQKVProjection,
    apply_causal_mask,
    build_causal_mask,
    scaled_dot_product_scores,
)
from model.rope import SiatRoPE


def test_build_causal_mask_shape_and_pattern():
    allow = build_causal_mask(4)
    assert allow.shape == (1, 1, 4, 4)
    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    assert torch.equal(allow[0, 0], expected)


def test_apply_preserves_shape():
    scores = torch.randn(2, 4, 8, 8)
    masked = apply_causal_mask(scores)
    assert masked.shape == scores.shape


def test_zeros_future_filled():
    scores = torch.zeros(1, 1, 4, 4)
    masked = apply_causal_mask(scores)
    fill = torch.finfo(scores.dtype).min
    allow = build_causal_mask(4)[0, 0]
    assert torch.all(masked[0, 0][allow] == 0)
    assert torch.all(masked[0, 0][~allow] == fill)


def test_s4_deterministic_positions():
    scores = torch.arange(16, dtype=torch.float32).view(1, 1, 4, 4)
    masked = apply_causal_mask(scores)[0, 0]
    fill = torch.finfo(torch.float32).min

    allowed = {
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (2, 2),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
    }
    blocked = {
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    }
    for i, j in allowed:
        assert masked[i, j] == scores[0, 0, i, j]
    for i, j in blocked:
        assert masked[i, j] == fill


def test_diagonal_and_past_allowed_future_blocked():
    scores = torch.tensor(
        [[[ [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0] ]]],
        dtype=torch.float32,
    )
    masked = apply_causal_mask(scores)[0, 0]
    fill = torch.finfo(torch.float32).min
    assert masked[0, 0] == 1.0
    assert masked[1, 0] == 4.0 and masked[1, 1] == 5.0
    assert masked[2, 0] == 7.0 and masked[2, 1] == 8.0 and masked[2, 2] == 9.0
    assert masked[0, 1] == fill and masked[0, 2] == fill and masked[1, 2] == fill


def test_allowed_values_preserved():
    scores = torch.tensor(
        [[[ [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0] ]]],
    )
    masked = apply_causal_mask(scores)
    assert masked[0, 0, 0, 0] == 1.0
    assert masked[0, 0, 1, 0] == 4.0
    assert masked[0, 0, 1, 1] == 5.0
    assert masked[0, 0, 2, 0] == 7.0
    assert masked[0, 0, 2, 1] == 8.0
    assert masked[0, 0, 2, 2] == 9.0


def test_batch_independence():
    scores = torch.zeros(2, 1, 3, 3)
    scores[0] = 1.0
    scores[1] = 2.0
    masked = apply_causal_mask(scores)
    allow = build_causal_mask(3)[0, 0]
    assert torch.all(masked[0, 0][allow] == 1.0)
    assert torch.all(masked[1, 0][allow] == 2.0)


def test_head_independence():
    scores = torch.zeros(1, 2, 3, 3)
    scores[0, 0] = 1.0
    scores[0, 1] = 3.0
    masked = apply_causal_mask(scores)
    allow = build_causal_mask(3)[0, 0]
    assert torch.all(masked[0, 0][allow] == 1.0)
    assert torch.all(masked[0, 1][allow] == 3.0)


def test_square_validation():
    with pytest.raises(ValueError, match="square"):
        apply_causal_mask(torch.randn(1, 1, 4, 3))


def test_gradient_flow_on_allowed():
    scores = torch.randn(2, 4, 8, 8, requires_grad=True)
    masked = apply_causal_mask(scores)
    allow = build_causal_mask(8, device=scores.device)[0, 0]
    # Sum only finite (allowed) positions so loss is not -inf.
    loss = masked[:, :, allow].sum()
    loss.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad[:, :, allow]).all()
    assert not torch.isnan(scores.grad).any()
    # Allowed positions should receive non-zero grads from the sum.
    assert scores.grad[:, :, allow].abs().sum() > 0


def test_softmax_compatibility_in_tests_only():
    """Production code does not softmax; tests verify mask semantics."""
    scores = torch.randn(1, 1, 4, 4)
    masked = apply_causal_mask(scores)
    probs = torch.softmax(masked, dim=-1)
    allow = build_causal_mask(4)[0, 0]
    assert torch.allclose(probs[0, 0][~allow], torch.zeros(6), atol=1e-6)
    assert torch.allclose(probs[0, 0].sum(dim=-1), torch.ones(4), atol=1e-5)
    # query 0 → only key 0
    assert torch.allclose(probs[0, 0, 0], torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-5)


def test_scaled_qkt_integration():
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    scores = scaled_dot_product_scores(q, k)
    masked = apply_causal_mask(scores)
    assert masked.shape == (2, 4, 8, 8)


def test_qkv_rope_integration():
    proj = SiatQKVProjection(d_model=64, n_heads=4)
    rope = SiatRoPE(head_dim=16, max_seq_len=32, theta=10000.0)
    x = torch.randn(2, 8, 64)
    q, k, v = proj(x)
    q, k = rope(q, k)
    masked = apply_causal_mask(scaled_dot_product_scores(q, k))
    assert masked.shape == (2, 4, 8, 8)
    assert v.shape == (2, 4, 8, 16)


def test_tiny_config():
    config = ModelConfig.tiny()
    proj = SiatQKVProjection(config.d_model, config.n_heads)
    rope = SiatRoPE(config.head_dim, config.max_seq_len, config.rope_theta)
    x = torch.randn(2, 16, config.d_model)
    q, k, _ = proj(x)
    q, k = rope(q, k)
    masked = apply_causal_mask(scaled_dot_product_scores(q, k))
    assert masked.shape == (2, config.n_heads, 16, 16)


def test_siat_30m_compatibility():
    # Shape-only with S=16 (not full max_seq_len=1024).
    scores = torch.randn(2, 8, 16, 16)
    masked = apply_causal_mask(scores)
    assert masked.shape == (2, 8, 16, 16)
    fill = torch.finfo(scores.dtype).min
    assert masked[0, 0, 0, 1] == fill
    assert masked[0, 0, 0, 0] == scores[0, 0, 0, 0]
