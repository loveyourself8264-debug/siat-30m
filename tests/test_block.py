"""Tests for SiatTransformerBlock (Pre-Norm)."""

from __future__ import annotations

import pytest
import torch

from config import ModelConfig
from model.block import SiatTransformerBlock


def _tiny_block(**kwargs) -> SiatTransformerBlock:
    defaults = dict(
        d_model=32,
        n_heads=4,
        ffn_dim=64,
        max_seq_len=64,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        dropout=0.0,
    )
    defaults.update(kwargs)
    return SiatTransformerBlock(**defaults)


def test_construct():
    block = _tiny_block()
    assert block.d_model == 32
    assert block.attn_norm is not block.ffn_norm


def test_output_shape():
    block = _tiny_block()
    x = torch.randn(2, 8, 32)
    assert block(x).shape == (2, 8, 32)


def test_parameter_count_siat_30m():
    d_model, ffn_dim = 512, 1536
    block = SiatTransformerBlock(
        d_model=d_model,
        n_heads=8,
        ffn_dim=ffn_dim,
        max_seq_len=1024,
    )
    total = sum(p.numel() for p in block.parameters())
    expected = 4 * d_model * d_model + 3 * d_model * ffn_dim + 2 * d_model
    assert total == expected
    assert total == 3_408_896


def test_rmsnorm_parameter_independence():
    block = _tiny_block()
    assert block.attn_norm.weight is not block.ffn_norm.weight
    assert (
        block.attn_norm.weight.data_ptr()
        != block.ffn_norm.weight.data_ptr()
    )


def test_prenorm_matches_manual_composition():
    block = _tiny_block()
    block.eval()
    x = torch.randn(2, 8, 32)
    with torch.no_grad():
        h = x + block.attention(block.attn_norm(x))
        expected = h + block.ffn(block.ffn_norm(h))
        actual = block(x)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_zero_sublayers_identity():
    """Zeroing o_proj and down_proj makes both sublayers emit 0 → Block(x)==x."""
    block = _tiny_block()
    with torch.no_grad():
        block.attention.output.o_proj.weight.zero_()
        block.ffn.down_proj.weight.zero_()
    block.eval()
    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        y = block(x)
    assert torch.allclose(y, x, atol=1e-6)


def test_causal_invariance():
    block = _tiny_block()
    block.eval()
    torch.manual_seed(0)
    a = torch.randn(2, 4, 32)
    b = a.clone()
    b[:, 2:, :] = torch.randn(2, 2, 32)
    with torch.no_grad():
        out_a = block(a)
        out_b = block(b)
    assert torch.allclose(out_a[:, 0, :], out_b[:, 0, :], atol=1e-5)
    assert torch.allclose(out_a[:, 1, :], out_b[:, 1, :], atol=1e-5)
    assert not torch.allclose(out_a[:, 3, :], out_b[:, 3, :], atol=1e-5)


def test_batch_independence():
    block = _tiny_block()
    block.eval()
    x = torch.randn(2, 6, 32)
    with torch.no_grad():
        out_full = block(x)
        out_b1 = block(x[1:2])
    assert torch.allclose(out_full[1:2], out_b1, atol=1e-5)

    x2 = x.clone()
    x2[0] = torch.randn_like(x2[0])
    with torch.no_grad():
        out2 = block(x2)
    assert torch.allclose(out_full[1], out2[1], atol=1e-5)


def test_gradient_flow():
    block = _tiny_block(d_model=64, n_heads=4, ffn_dim=128)
    x = torch.randn(2, 8, 64, requires_grad=True)
    output = block(x)
    loss = (output * torch.randn_like(output)).sum()
    loss.backward()

    assert x.grad is not None
    assert block.attn_norm.weight.grad is not None
    assert block.ffn_norm.weight.grad is not None
    assert block.attention.qkv.q_proj.weight.grad is not None
    assert block.attention.qkv.k_proj.weight.grad is not None
    assert block.attention.qkv.v_proj.weight.grad is not None
    assert block.attention.output.o_proj.weight.grad is not None
    assert block.ffn.gate_proj.weight.grad is not None
    assert block.ffn.up_proj.weight.grad is not None
    assert block.ffn.down_proj.weight.grad is not None
    assert torch.isfinite(x.grad).all()
    assert not torch.isnan(x.grad).any()
    assert torch.isfinite(block.attn_norm.weight.grad).all()
    assert torch.isfinite(block.ffn.down_proj.weight.grad).all()


def test_dropout_zero_deterministic():
    block = _tiny_block(dropout=0.0)
    x = torch.randn(2, 8, 32)
    block.train()
    y1 = block(x)
    block.eval()
    y2 = block(x)
    y3 = block(x)
    assert torch.allclose(y1, y2, atol=1e-6)
    assert torch.allclose(y2, y3, atol=1e-6)


def test_tiny_config():
    config = ModelConfig.tiny()
    block = SiatTransformerBlock(
        d_model=config.d_model,
        n_heads=config.n_heads,
        ffn_dim=config.ffn_dim,
        max_seq_len=config.max_seq_len,
        rope_theta=config.rope_theta,
        rms_norm_eps=config.rms_norm_eps,
        dropout=config.dropout,
    )
    x = torch.randn(2, 16, config.d_model)
    assert block(x).shape == (2, 16, config.d_model)


def test_siat_30m_compatibility():
    config = ModelConfig.siat_30m()
    block = SiatTransformerBlock(
        d_model=config.d_model,
        n_heads=config.n_heads,
        ffn_dim=config.ffn_dim,
        max_seq_len=config.max_seq_len,
        rope_theta=config.rope_theta,
        rms_norm_eps=config.rms_norm_eps,
        dropout=config.dropout,
    )
    x = torch.randn(1, 8, config.d_model)
    assert block(x).shape == (1, 8, 512)


def test_wrong_last_dim():
    block = _tiny_block()
    with pytest.raises(ValueError, match="d_model"):
        block(torch.randn(2, 4, 16))
