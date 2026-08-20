"""Tests for attention softmax / attention weights."""

from __future__ import annotations

import math

import pytest
import torch

from config import ModelConfig
from model.attention import (
    SiatQKVProjection,
    apply_causal_mask,
    build_causal_mask,
    compute_attention_weights,
    scaled_dot_product_scores,
)
from model.rmsnorm import SiatRMSNorm
from model.rope import SiatRoPE


def test_output_shape():
    scores = apply_causal_mask(torch.randn(2, 4, 16, 16))
    weights = compute_attention_weights(scores)
    assert weights.shape == (2, 4, 16, 16)


def test_row_sum_and_nonnegativity():
    scores = apply_causal_mask(torch.randn(2, 4, 8, 8))
    weights = compute_attention_weights(scores)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4, 8), atol=1e-5)
    assert (weights >= 0).all()
    assert (weights <= 1 + 1e-5).all()


def test_future_positions_zero():
    scores = apply_causal_mask(torch.randn(1, 1, 4, 4))
    weights = compute_attention_weights(scores)[0, 0]
    allow = build_causal_mask(4)[0, 0]
    assert torch.allclose(weights[~allow], torch.zeros(weights[~allow].numel()), atol=1e-6)
    assert torch.allclose(weights[0], torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-5)


def test_deterministic_equal_scores():
    # query0: [0,-inf,-inf] → [1,0,0]
    # query1: [0,0,-inf] → [0.5,0.5,0]
    # query2: [0,0,0] → [1/3,1/3,1/3]
    raw = torch.zeros(1, 1, 3, 3)
    masked = apply_causal_mask(raw)
    weights = compute_attention_weights(masked)[0, 0]
    assert torch.allclose(weights[0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-5)
    assert torch.allclose(weights[1], torch.tensor([0.5, 0.5, 0.0]), atol=1e-5)
    third = 1.0 / 3.0
    assert torch.allclose(
        weights[2], torch.tensor([third, third, third]), atol=1e-5
    )


def test_unequal_score_prefers_higher():
    # query row: [1, 2, -inf] after mask on last dim of S=3 for query index 1
    # Use full matrix: only query 1 matters for this check
    raw = torch.full((1, 1, 3, 3), float("-inf"))
    raw[0, 0, 0, 0] = 0.0
    raw[0, 0, 1, 0] = 1.0
    raw[0, 0, 1, 1] = 2.0
    raw[0, 0, 2, 0] = 0.0
    raw[0, 0, 2, 1] = 0.0
    raw[0, 0, 2, 2] = 0.0
    # Already causal-shaped; still run through apply for consistency
    masked = apply_causal_mask(
        torch.tensor([[[[0.0, 9.0, 9.0], [1.0, 2.0, 9.0], [0.0, 0.0, 0.0]]]])
    )
    weights = compute_attention_weights(masked)[0, 0]
    # query 1: softmax([1,2]) with future 0
    e1, e2 = math.exp(1.0), math.exp(2.0)
    expected = torch.tensor([e1 / (e1 + e2), e2 / (e1 + e2), 0.0])
    assert torch.allclose(weights[1], expected, atol=1e-5)
    assert weights[1, 1] > weights[1, 0]


def test_batch_independence():
    a = torch.zeros(1, 1, 2, 2)
    b = torch.tensor([[[[0.0, 0.0], [3.0, 0.0]]]])
    # Build batch of two different score matrices via stacking after mask
    scores = torch.zeros(2, 1, 2, 2)
    scores[0] = 0.0
    scores[1, 0, 1, 0] = 3.0
    scores[1, 0, 1, 1] = 0.0
    weights = compute_attention_weights(apply_causal_mask(scores))
    assert torch.allclose(weights[0, 0, 1], torch.tensor([0.5, 0.5]), atol=1e-5)
    e3, e0 = math.exp(3.0), math.exp(0.0)
    expected = torch.tensor([e3 / (e3 + e0), e0 / (e3 + e0)])
    assert torch.allclose(weights[1, 0, 1], expected, atol=1e-5)


def test_head_independence():
    scores = torch.zeros(1, 2, 2, 2)
    scores[0, 0] = 0.0
    scores[0, 1, 1, 0] = 2.0
    scores[0, 1, 1, 1] = 0.0
    weights = compute_attention_weights(apply_causal_mask(scores))
    assert torch.allclose(weights[0, 0, 1], torch.tensor([0.5, 0.5]), atol=1e-5)
    e2, e0 = math.exp(2.0), math.exp(0.0)
    assert torch.allclose(
        weights[0, 1, 1],
        torch.tensor([e2 / (e2 + e0), e0 / (e2 + e0)]),
        atol=1e-5,
    )


def test_query_row_independence():
    scores = torch.zeros(1, 1, 3, 3)
    scores[0, 0, 2, 0] = 5.0  # only affects query 2
    weights = compute_attention_weights(apply_causal_mask(scores))
    assert torch.allclose(weights[0, 0, 0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-5)
    assert torch.allclose(weights[0, 0, 1], torch.tensor([0.5, 0.5, 0.0]), atol=1e-5)
    assert weights[0, 0, 2, 0] > weights[0, 0, 2, 1]


def test_gradient_flow():
    scores = torch.randn(2, 4, 8, 8, requires_grad=True)
    masked = apply_causal_mask(scores)
    weights = compute_attention_weights(masked)
    coeff = torch.randn_like(weights)
    loss = (weights * coeff).sum()
    loss.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert not torch.isnan(scores.grad).any()
    assert not torch.isinf(scores.grad).any()


def test_masked_future_gradient_near_zero():
    scores = torch.randn(1, 1, 4, 4, requires_grad=True)
    masked = apply_causal_mask(scores)
    weights = compute_attention_weights(masked)
    coeff = torch.ones_like(weights)
    (weights * coeff).sum().backward()
    allow = build_causal_mask(4)[0, 0]
    # Future positions should get ~0 grad through softmax.
    assert torch.allclose(
        scores.grad[0, 0][~allow],
        torch.zeros(scores.grad[0, 0][~allow].numel()),
        atol=1e-5,
    )


def test_square_validation():
    with pytest.raises(ValueError, match="square"):
        compute_attention_weights(torch.randn(1, 1, 4, 3))


def test_scaled_mask_integration():
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    weights = compute_attention_weights(
        apply_causal_mask(scaled_dot_product_scores(q, k))
    )
    assert weights.shape == (2, 4, 8, 8)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4, 8), atol=1e-5)


def test_qkv_rope_rmsnorm_pipeline():
    d_model, n_heads, s = 32, 4, 8
    norm = SiatRMSNorm(d_model)
    proj = SiatQKVProjection(d_model, n_heads)
    rope = SiatRoPE(d_model // n_heads, max_seq_len=32, theta=10000.0)
    x = torch.randn(2, s, d_model)
    q, k, v = proj(norm(x))
    q, k = rope(q, k)
    weights = compute_attention_weights(
        apply_causal_mask(scaled_dot_product_scores(q, k))
    )
    assert weights.shape == (2, n_heads, s, s)
    assert v.shape == (2, n_heads, s, d_model // n_heads)
    assert torch.allclose(
        weights.sum(dim=-1), torch.ones(2, n_heads, s), atol=1e-5
    )


def test_tiny_config():
    config = ModelConfig.tiny()
    proj = SiatQKVProjection(config.d_model, config.n_heads)
    rope = SiatRoPE(config.head_dim, config.max_seq_len, config.rope_theta)
    x = torch.randn(2, 16, config.d_model)
    q, k, _ = proj(x)
    q, k = rope(q, k)
    weights = compute_attention_weights(
        apply_causal_mask(scaled_dot_product_scores(q, k))
    )
    assert weights.shape == (2, config.n_heads, 16, 16)
    assert torch.allclose(
        weights.sum(dim=-1), torch.ones(2, config.n_heads, 16), atol=1e-5
    )


def test_siat_30m_shape():
    # B=2, H=8, S=16, Hd=64 → weights [2,8,16,16]
    q = torch.randn(2, 8, 16, 64)
    k = torch.randn(2, 8, 16, 64)
    weights = compute_attention_weights(
        apply_causal_mask(scaled_dot_product_scores(q, k))
    )
    assert weights.shape == (2, 8, 16, 16)
