"""Tests for AttentionWeights @ V → per-head context."""

from __future__ import annotations

import pytest
import torch

from config import ModelConfig
from model.attention import (
    SiatQKVProjection,
    aggregate_values,
    apply_causal_mask,
    compute_attention_weights,
    scaled_dot_product_scores,
)
from model.rope import SiatRoPE


def test_output_shape():
    weights = torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)
    v = torch.randn(2, 4, 8, 32)
    context = aggregate_values(weights, v)
    assert context.shape == (2, 4, 8, 32)


def test_deterministic_weighted_sum():
    # V0=[1,2], V1=[3,4]; q0→[1,0]→[1,2]; q1→[0.25,0.75]→[2.5,3.5]
    v = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # [1,1,2,2]
    weights = torch.tensor([[[[1.0, 0.0], [0.25, 0.75]]]])
    context = aggregate_values(weights, v)
    assert torch.allclose(context[0, 0, 0], torch.tensor([1.0, 2.0]))
    assert torch.allclose(context[0, 0, 1], torch.tensor([2.5, 3.5]))


def test_identity_attention():
    v = torch.randn(1, 1, 3, 4)
    weights = torch.eye(3).view(1, 1, 3, 3)
    context = aggregate_values(weights, v)
    assert torch.allclose(context, v)


def test_uniform_attention():
    v = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    weights = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]])
    context = aggregate_values(weights, v)
    expected = (v[0, 0, 0] + v[0, 0, 1]) / 2
    assert torch.allclose(context[0, 0, 0], expected)
    assert torch.allclose(context[0, 0, 1], expected)


def test_causal_query0_equals_first_v():
    proj = SiatQKVProjection(d_model=32, n_heads=4)
    rope = SiatRoPE(head_dim=8, max_seq_len=16, theta=10000.0)
    x = torch.randn(1, 4, 32)
    q, k, v = proj(x)
    q, k = rope(q, k)
    weights = compute_attention_weights(
        apply_causal_mask(scaled_dot_product_scores(q, k))
    )
    context = aggregate_values(weights, v)
    # Causal: query 0 attends only to key 0 → context == V[:, :, 0, :]
    assert torch.allclose(context[:, :, 0, :], v[:, :, 0, :], atol=1e-5)


def test_batch_independence():
    weights = torch.zeros(2, 1, 2, 2)
    weights[0, 0] = torch.eye(2)
    weights[1, 0] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    v = torch.zeros(2, 1, 2, 2)
    v[0, 0] = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    v[1, 0] = torch.tensor([[10.0, 10.0], [20.0, 20.0]])
    context = aggregate_values(weights, v)
    assert torch.allclose(context[0], v[0])
    assert torch.allclose(context[1], v[1])


def test_head_independence():
    weights = torch.zeros(1, 2, 2, 2)
    weights[0, 0] = torch.eye(2)
    weights[0, 1] = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    v = torch.zeros(1, 2, 2, 2)
    v[0, 0] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    v[0, 1] = torch.tensor([[2.0, 2.0], [4.0, 4.0]])
    context = aggregate_values(weights, v)
    assert torch.allclose(context[0, 0], v[0, 0])
    assert torch.allclose(context[0, 1, 0], torch.tensor([3.0, 3.0]))


def test_query_independence():
    v = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]])
    weights_a = torch.zeros(1, 1, 3, 3)
    weights_a[0, 0, 0] = torch.tensor([1.0, 0.0, 0.0])
    weights_a[0, 0, 1] = torch.tensor([0.0, 1.0, 0.0])
    weights_a[0, 0, 2] = torch.tensor([0.0, 0.0, 1.0])
    weights_b = weights_a.clone()
    weights_b[0, 0, 0] = torch.tensor([0.0, 1.0, 0.0])  # only query 0 changes
    ctx_a = aggregate_values(weights_a, v)
    ctx_b = aggregate_values(weights_b, v)
    assert torch.allclose(ctx_a[0, 0, 1], ctx_b[0, 0, 1])
    assert torch.allclose(ctx_a[0, 0, 2], ctx_b[0, 0, 2])
    assert not torch.allclose(ctx_a[0, 0, 0], ctx_b[0, 0, 0])


def test_invalid_rank():
    with pytest.raises(ValueError, match="rank-4"):
        aggregate_values(torch.randn(2, 8, 8), torch.randn(2, 4, 8, 8))


def test_incompatible_sequence():
    with pytest.raises(ValueError, match="key length"):
        aggregate_values(torch.randn(1, 1, 4, 3), torch.randn(1, 1, 4, 8))
    # key matches V (4) but query S (3) != V S (4)
    with pytest.raises(ValueError, match="query S"):
        aggregate_values(torch.randn(1, 1, 3, 4), torch.randn(1, 1, 4, 2))


def test_v_gradient_flow():
    weights = torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)
    v = torch.randn(2, 4, 8, 32, requires_grad=True)
    context = aggregate_values(weights, v)
    context.sum().backward()
    assert v.grad is not None
    assert torch.isfinite(v.grad).all()
    assert not torch.isnan(v.grad).any()


def test_weight_gradient_flow():
    raw = torch.randn(2, 4, 8, 8, requires_grad=True)
    weights = torch.softmax(raw, dim=-1)
    v = torch.randn(2, 4, 8, 16)
    context = aggregate_values(weights, v)
    context.sum().backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


def test_full_pipeline_gradients():
    proj = SiatQKVProjection(d_model=32, n_heads=4)
    rope = SiatRoPE(head_dim=8, max_seq_len=16, theta=10000.0)
    x = torch.randn(2, 8, 32, requires_grad=True)
    q, k, v = proj(x)
    q, k = rope(q, k)
    weights = compute_attention_weights(
        apply_causal_mask(scaled_dot_product_scores(q, k))
    )
    context = aggregate_values(weights, v)
    context.sum().backward()
    assert x.grad is not None
    assert proj.q_proj.weight.grad is not None
    assert proj.k_proj.weight.grad is not None
    assert proj.v_proj.weight.grad is not None
    assert torch.isfinite(x.grad).all()
    assert not torch.isnan(proj.q_proj.weight.grad).any()


def test_tiny_config():
    config = ModelConfig.tiny()
    proj = SiatQKVProjection(config.d_model, config.n_heads)
    rope = SiatRoPE(config.head_dim, config.max_seq_len, config.rope_theta)
    x = torch.randn(2, 16, config.d_model)
    q, k, v = proj(x)
    q, k = rope(q, k)
    weights = compute_attention_weights(
        apply_causal_mask(scaled_dot_product_scores(q, k))
    )
    context = aggregate_values(weights, v)
    assert weights.shape == (2, config.n_heads, 16, 16)
    assert context.shape == (2, config.n_heads, 16, config.head_dim)


def test_siat_30m_shape():
    weights = torch.softmax(torch.randn(2, 8, 16, 16), dim=-1)
    v = torch.randn(2, 8, 16, 64)
    context = aggregate_values(weights, v)
    assert context.shape == (2, 8, 16, 64)
