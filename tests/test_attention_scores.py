"""Tests for scaled QKᵀ attention scores (no mask / softmax)."""

from __future__ import annotations

import math

import pytest
import torch

from config import ModelConfig
from model.attention import SiatQKVProjection, scaled_dot_product_scores
from model.rmsnorm import SiatRMSNorm
from model.rope import SiatRoPE


def test_manual_2x2_example():
    # B=1, H=1, S=2, Hd=2; Q=K=I → QKᵀ=I → scale 1/sqrt(2)
    q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    scores = scaled_dot_product_scores(q, k)
    expected = torch.eye(2) / math.sqrt(2)
    assert scores.shape == (1, 1, 2, 2)
    assert torch.allclose(scores[0, 0], expected, atol=1e-6)


def test_unscaled_vs_scaled():
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    raw = torch.matmul(q, k.transpose(-2, -1))
    scaled = scaled_dot_product_scores(q, k)
    expected = raw / math.sqrt(q.size(-1))
    assert torch.allclose(scaled, expected, atol=1e-5, rtol=1e-5)


def test_output_shape():
    q = torch.randn(2, 8, 32, 64)
    k = torch.randn(2, 8, 32, 64)
    scores = scaled_dot_product_scores(q, k)
    assert scores.shape == (2, 8, 32, 32)


def test_per_head_independence():
    # H=2, Hd=2, S=2 — different content per head
    q = torch.zeros(1, 2, 2, 2)
    k = torch.zeros(1, 2, 2, 2)
    q[0, 0] = torch.eye(2)
    k[0, 0] = torch.eye(2)
    q[0, 1] = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    k[0, 1] = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    scores = scaled_dot_product_scores(q, k)
    # Head 0: I / sqrt(2)
    assert torch.allclose(scores[0, 0], torch.eye(2) / math.sqrt(2), atol=1e-5)
    # Head 1: (4*I) / sqrt(2) = 2*sqrt(2)*I on diagonal of QKᵀ=[[4,0],[0,4]]
    assert torch.allclose(
        scores[0, 1],
        torch.tensor([[4.0, 0.0], [0.0, 4.0]]) / math.sqrt(2),
        atol=1e-5,
    )
    assert not torch.allclose(scores[0, 0], scores[0, 1])


def test_batch_independence():
    q = torch.zeros(2, 1, 2, 2)
    k = torch.zeros(2, 1, 2, 2)
    q[0, 0] = torch.eye(2)
    k[0, 0] = torch.eye(2)
    q[1, 0] = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    k[1, 0] = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    scores = scaled_dot_product_scores(q, k)
    assert torch.allclose(scores[0, 0], torch.eye(2) / math.sqrt(2), atol=1e-5)
    assert torch.allclose(
        scores[1, 0],
        torch.tensor([[9.0, 0.0], [0.0, 9.0]]) / math.sqrt(2),
        atol=1e-5,
    )


def test_gradient_flow():
    q = torch.randn(2, 4, 8, 32, requires_grad=True)
    k = torch.randn(2, 4, 8, 32, requires_grad=True)
    scores = scaled_dot_product_scores(q, k)
    scores.sum().backward()
    assert q.grad is not None and k.grad is not None
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()
    assert not torch.isnan(q.grad).any()
    assert not torch.isinf(k.grad).any()


def test_mismatched_shapes():
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 7, 16)
    with pytest.raises(ValueError, match="share"):
        scaled_dot_product_scores(q, k)
    with pytest.raises(ValueError, match="rank 4"):
        scaled_dot_product_scores(torch.randn(2, 8, 16), torch.randn(2, 8, 16))


def test_rope_integration():
    proj = SiatQKVProjection(d_model=64, n_heads=4)
    rope = SiatRoPE(head_dim=16, max_seq_len=32, theta=10000.0)
    x = torch.randn(2, 8, 64)
    q, k, v = proj(x)
    q, k = rope(q, k)
    scores = scaled_dot_product_scores(q, k)
    assert scores.shape == (2, 4, 8, 8)
    # V unused this stage
    assert v.shape == (2, 4, 8, 16)


def test_rmsnorm_qkv_rope_scores():
    d_model, n_heads, s = 32, 4, 8
    norm = SiatRMSNorm(d_model)
    proj = SiatQKVProjection(d_model, n_heads)
    rope = SiatRoPE(d_model // n_heads, max_seq_len=32, theta=10000.0)
    x = torch.randn(2, s, d_model)
    q, k, _ = proj(norm(x))
    q, k = rope(q, k)
    scores = scaled_dot_product_scores(q, k)
    assert scores.shape == (2, n_heads, s, s)


def test_tiny_config():
    config = ModelConfig.tiny()
    proj = SiatQKVProjection(config.d_model, config.n_heads)
    rope = SiatRoPE(config.head_dim, config.max_seq_len, config.rope_theta)
    x = torch.randn(2, 16, config.d_model)
    q, k, _ = proj(x)
    q, k = rope(q, k)
    scores = scaled_dot_product_scores(q, k)
    assert scores.shape == (2, config.n_heads, 16, 16)


def test_siat_30m_shape():
    q = torch.randn(2, 8, 16, 64)
    k = torch.randn(2, 8, 16, 64)
    scores = scaled_dot_product_scores(q, k)
    assert scores.shape == (2, 8, 16, 16)
