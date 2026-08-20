"""Tests for SiatSelfAttention assembly."""

from __future__ import annotations

import pytest
import torch

from config import ModelConfig
from model.attention import (
    SiatSelfAttention,
    aggregate_values,
    apply_causal_mask,
    compute_attention_weights,
    merge_heads,
    scaled_dot_product_scores,
)


def test_construct():
    attn = SiatSelfAttention(d_model=128, n_heads=4, max_seq_len=256)
    assert attn.d_model == 128
    assert attn.n_heads == 4
    assert attn.head_dim == 32
    assert attn.max_seq_len == 256


def test_invalid_d_model_n_heads():
    with pytest.raises(ValueError, match="d_model"):
        SiatSelfAttention(d_model=0, n_heads=4, max_seq_len=16)
    with pytest.raises(ValueError, match="n_heads"):
        SiatSelfAttention(d_model=128, n_heads=0, max_seq_len=16)
    with pytest.raises(ValueError, match="divisible"):
        SiatSelfAttention(d_model=130, n_heads=4, max_seq_len=16)


def test_invalid_dropout_and_max_seq():
    with pytest.raises(ValueError, match="dropout"):
        SiatSelfAttention(d_model=128, n_heads=4, max_seq_len=16, dropout=1.5)
    with pytest.raises(ValueError, match="max_seq_len"):
        SiatSelfAttention(d_model=128, n_heads=4, max_seq_len=0)


def test_sequence_length_validation():
    attn = SiatSelfAttention(d_model=32, n_heads=4, max_seq_len=8)
    x = torch.randn(1, 9, 32)
    with pytest.raises(ValueError, match="max_seq_len"):
        attn(x)


def test_hidden_dim_validation():
    attn = SiatSelfAttention(d_model=32, n_heads=4, max_seq_len=16)
    with pytest.raises(ValueError, match="d_model"):
        attn(torch.randn(1, 4, 16))
    with pytest.raises(ValueError, match="rank 3"):
        attn(torch.randn(1, 4, 4, 8))


def test_output_shape():
    attn = SiatSelfAttention(d_model=128, n_heads=4, max_seq_len=256)
    x = torch.randn(2, 16, 128)
    assert attn(x).shape == (2, 16, 128)


def test_parameter_count_tiny_and_30m():
    tiny = SiatSelfAttention(d_model=128, n_heads=4, max_seq_len=256)
    assert sum(p.numel() for p in tiny.parameters()) == 4 * 128 * 128
    assert sum(p.numel() for p in tiny.parameters()) == 65_536

    big = SiatSelfAttention(d_model=512, n_heads=8, max_seq_len=1024)
    assert sum(p.numel() for p in big.parameters()) == 4 * 512 * 512
    assert sum(p.numel() for p in big.parameters()) == 1_048_576


def test_rope_has_no_learnable_parameters():
    attn = SiatSelfAttention(d_model=64, n_heads=4, max_seq_len=32)
    assert sum(p.numel() for p in attn.rope.parameters()) == 0
    # Buffers exist but are not trainable.
    assert attn.rope.cos_cached.requires_grad is False


def test_tiny_config_integration():
    config = ModelConfig.tiny()
    attn = SiatSelfAttention(
        d_model=config.d_model,
        n_heads=config.n_heads,
        max_seq_len=config.max_seq_len,
        rope_theta=config.rope_theta,
        dropout=config.dropout,
    )
    x = torch.randn(2, 16, config.d_model)
    assert attn(x).shape == (2, 16, config.d_model)


def test_siat_30m_compatibility():
    config = ModelConfig.siat_30m()
    attn = SiatSelfAttention(
        d_model=config.d_model,
        n_heads=config.n_heads,
        max_seq_len=config.max_seq_len,
        rope_theta=config.rope_theta,
        dropout=config.dropout,
    )
    x = torch.randn(1, 8, config.d_model)
    assert attn(x).shape == (1, 8, 512)


def test_matches_primitive_pipeline():
    attn = SiatSelfAttention(d_model=32, n_heads=4, max_seq_len=16)
    attn.eval()
    x = torch.randn(2, 8, 32)

    with torch.no_grad():
        assembled = attn(x)

        q, k, v = attn.qkv(x)
        q, k = attn.rope(q, k)
        weights = compute_attention_weights(
            apply_causal_mask(scaled_dot_product_scores(q, k))
        )
        context = aggregate_values(weights, v)
        manual = attn.output(merge_heads(context))

    assert torch.allclose(assembled, manual, atol=1e-6, rtol=1e-5)


def test_position_zero_and_future_weights_zero():
    attn = SiatSelfAttention(d_model=32, n_heads=4, max_seq_len=16, dropout=0.0)
    attn.eval()
    x = torch.randn(1, 4, 32)
    _, weights = attn(x, return_attention_weights=True)

    # Position 0 attends only to itself.
    expected0 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    assert torch.allclose(weights[0, 0, 0], expected0, atol=1e-6)

    # Future keys j > i must be ~0.
    s = weights.shape[-1]
    for i in range(s):
        for j in range(i + 1, s):
            assert torch.allclose(
                weights[..., i, j],
                torch.zeros_like(weights[..., i, j]),
                atol=1e-6,
            )


def test_attention_row_sum():
    attn = SiatSelfAttention(d_model=32, n_heads=4, max_seq_len=16, dropout=0.0)
    attn.eval()
    _, weights = attn(torch.randn(2, 5, 32), return_attention_weights=True)
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_causal_invariance():
    attn = SiatSelfAttention(d_model=32, n_heads=4, max_seq_len=16, dropout=0.0)
    attn.eval()
    torch.manual_seed(0)
    a = torch.randn(2, 4, 32)
    b = a.clone()
    b[:, 2:, :] = torch.randn(2, 2, 32)

    with torch.no_grad():
        out_a = attn(a)
        out_b = attn(b)

    assert torch.allclose(out_a[:, 0, :], out_b[:, 0, :], atol=1e-5)
    assert torch.allclose(out_a[:, 1, :], out_b[:, 1, :], atol=1e-5)
    # Later positions may differ.
    assert not torch.allclose(out_a[:, 3, :], out_b[:, 3, :], atol=1e-5)


def test_batch_independence():
    attn = SiatSelfAttention(d_model=32, n_heads=4, max_seq_len=16, dropout=0.0)
    attn.eval()
    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        out_full = attn(x)
        out_b1 = attn(x[1:2])
    assert torch.allclose(out_full[1:2], out_b1, atol=1e-5)

    x2 = x.clone()
    x2[0] = torch.randn_like(x2[0])
    with torch.no_grad():
        out2 = attn(x2)
    assert torch.allclose(out_full[1], out2[1], atol=1e-5)


def test_gradient_flow_qkvo():
    attn = SiatSelfAttention(d_model=128, n_heads=4, max_seq_len=256)
    x = torch.randn(2, 8, 128, requires_grad=True)
    output = attn(x)
    coeff = torch.randn_like(output)
    loss = (output * coeff).sum()
    loss.backward()

    assert x.grad is not None
    assert attn.qkv.q_proj.weight.grad is not None
    assert attn.qkv.k_proj.weight.grad is not None
    assert attn.qkv.v_proj.weight.grad is not None
    assert attn.output.o_proj.weight.grad is not None
    assert torch.isfinite(x.grad).all()
    assert not torch.isnan(x.grad).any()
    assert torch.isfinite(attn.qkv.q_proj.weight.grad).all()
    assert torch.isfinite(attn.output.o_proj.weight.grad).all()


def test_dropout_zero_deterministic():
    attn = SiatSelfAttention(
        d_model=32, n_heads=4, max_seq_len=16, dropout=0.0
    )
    x = torch.randn(2, 8, 32)
    attn.train()
    y1 = attn(x)
    attn.eval()
    y2 = attn(x)
    y3 = attn(x)
    assert torch.allclose(y1, y2, atol=1e-6)
    assert torch.allclose(y2, y3, atol=1e-6)


def test_dropout_positive_smoke():
    attn = SiatSelfAttention(
        d_model=32, n_heads=4, max_seq_len=16, dropout=0.5
    )
    x = torch.randn(2, 8, 32)
    attn.train()
    y_train = attn(x)
    attn.eval()
    y_eval = attn(x)
    assert y_train.shape == y_eval.shape == (2, 8, 32)
