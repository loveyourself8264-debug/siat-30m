"""Tests for head merge and attention output projection."""

from __future__ import annotations

import pytest
import torch

from config import ModelConfig
from model.attention import (
    SiatAttentionOutput,
    SiatQKVProjection,
    aggregate_values,
    apply_causal_mask,
    compute_attention_weights,
    merge_heads,
    scaled_dot_product_scores,
)
from model.rope import SiatRoPE


def test_merge_output_shape():
    context = torch.randn(2, 4, 16, 32)
    merged = merge_heads(context)
    assert merged.shape == (2, 16, 128)


def test_split_merge_inverse():
    # B=1, S=2, D=8, H=2, Hd=4
    original = torch.arange(16, dtype=torch.float32).view(1, 2, 8)
    proj = SiatQKVProjection(d_model=8, n_heads=2)
    # Use identity-like path via private split helper on a copy
    split = proj._split_heads(original)
    assert split.shape == (1, 2, 2, 4)
    merged = merge_heads(split)
    assert torch.equal(merged, original)


def test_merge_dtype_preserved():
    context = torch.randn(1, 2, 4, 8, dtype=torch.float32)
    assert merge_heads(context).dtype == torch.float32


def test_merge_gradient_flow():
    context = torch.randn(2, 4, 8, 32, requires_grad=True)
    merged = merge_heads(context)
    merged.sum().backward()
    assert context.grad is not None
    assert torch.isfinite(context.grad).all()
    assert not torch.isnan(context.grad).any()


def test_merge_invalid_rank():
    with pytest.raises(ValueError, match="rank 4"):
        merge_heads(torch.randn(2, 16, 128))


def test_output_projection_shape_and_params():
    d_model = 128
    out = SiatAttentionOutput(d_model=d_model)
    x = torch.randn(2, 8, d_model)
    y = out(x)
    assert y.shape == (2, 8, d_model)
    assert out.o_proj.bias is None
    assert sum(p.numel() for p in out.parameters()) == d_model * d_model


def test_qkv_plus_output_param_count():
    d_model = 512
    qkv = SiatQKVProjection(d_model=d_model, n_heads=8)
    o = SiatAttentionOutput(d_model=d_model)
    total = sum(p.numel() for p in qkv.parameters()) + sum(
        p.numel() for p in o.parameters()
    )
    assert total == 4 * d_model * d_model
    assert total == 1_048_576


def test_identity_output_projection():
    d_model = 4
    out = SiatAttentionOutput(d_model=d_model)
    with torch.no_grad():
        out.o_proj.weight.copy_(torch.eye(d_model))
    x = torch.randn(2, 3, d_model)
    assert torch.allclose(out(x), x)


def test_output_gradient_flow():
    out = SiatAttentionOutput(d_model=128)
    x = torch.randn(2, 8, 128, requires_grad=True)
    y = out(x)
    y.sum().backward()
    assert x.grad is not None
    assert out.o_proj.weight.grad is not None
    assert torch.isfinite(x.grad).all()
    assert not torch.isnan(out.o_proj.weight.grad).any()


def test_output_wrong_last_dim():
    out = SiatAttentionOutput(d_model=64)
    with pytest.raises(ValueError, match="last dim"):
        out(torch.randn(2, 8, 32))


def test_full_pipeline_and_gradients():
    d_model, n_heads, s = 32, 4, 8
    proj = SiatQKVProjection(d_model, n_heads)
    o_proj = SiatAttentionOutput(d_model)
    rope = SiatRoPE(d_model // n_heads, max_seq_len=16, theta=10000.0)
    x = torch.randn(2, s, d_model, requires_grad=True)
    q, k, v = proj(x)
    q, k = rope(q, k)
    weights = compute_attention_weights(
        apply_causal_mask(scaled_dot_product_scores(q, k))
    )
    context = aggregate_values(weights, v)
    merged = merge_heads(context)
    assert merged.shape == (2, s, d_model)
    output = o_proj(merged)
    assert output.shape == (2, s, d_model)
    output.sum().backward()
    assert x.grad is not None
    assert proj.q_proj.weight.grad is not None
    assert proj.k_proj.weight.grad is not None
    assert proj.v_proj.weight.grad is not None
    assert o_proj.o_proj.weight.grad is not None
    assert torch.isfinite(x.grad).all()


def test_tiny_config():
    config = ModelConfig.tiny()
    proj = SiatQKVProjection(config.d_model, config.n_heads)
    o_proj = SiatAttentionOutput(config.d_model)
    rope = SiatRoPE(config.head_dim, config.max_seq_len, config.rope_theta)
    x = torch.randn(2, 16, config.d_model)
    q, k, v = proj(x)
    q, k = rope(q, k)
    context = aggregate_values(
        compute_attention_weights(
            apply_causal_mask(scaled_dot_product_scores(q, k))
        ),
        v,
    )
    merged = merge_heads(context)
    output = o_proj(merged)
    assert merged.shape == (2, 16, config.d_model)
    assert output.shape == (2, 16, config.d_model)


def test_siat_30m_shape():
    context = torch.randn(2, 8, 16, 64)
    merged = merge_heads(context)
    assert merged.shape == (2, 16, 512)
    out = SiatAttentionOutput(d_model=512)
    assert out(merged).shape == (2, 16, 512)
