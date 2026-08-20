"""Tests for SiatSwiGLU feed-forward network."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from config import ModelConfig
from model.ffn import SiatSwiGLU


def test_construct():
    ffn = SiatSwiGLU(d_model=32, ffn_dim=64)
    assert ffn.d_model == 32
    assert ffn.ffn_dim == 64


def test_invalid_d_model_ffn_dim():
    with pytest.raises(ValueError, match="d_model"):
        SiatSwiGLU(d_model=0, ffn_dim=64)
    with pytest.raises(ValueError, match="ffn_dim"):
        SiatSwiGLU(d_model=32, ffn_dim=0)
    with pytest.raises(ValueError, match="dropout"):
        SiatSwiGLU(d_model=32, ffn_dim=64, dropout=1.5)


def test_output_shape():
    ffn = SiatSwiGLU(d_model=128, ffn_dim=256)
    x = torch.randn(2, 16, 128)
    assert ffn(x).shape == (2, 16, 128)


def test_gate_up_intermediate_shape():
    ffn = SiatSwiGLU(d_model=32, ffn_dim=48)
    x = torch.randn(2, 4, 32)
    assert ffn.gate_proj(x).shape == (2, 4, 48)
    assert ffn.up_proj(x).shape == (2, 4, 48)


def test_parameter_count():
    d_model, ffn_dim = 512, 1536
    ffn = SiatSwiGLU(d_model=d_model, ffn_dim=ffn_dim)
    total = sum(p.numel() for p in ffn.parameters())
    assert total == 3 * d_model * ffn_dim
    assert total == 2_359_296


def test_no_bias():
    ffn = SiatSwiGLU(d_model=32, ffn_dim=64)
    assert ffn.gate_proj.bias is None
    assert ffn.up_proj.bias is None
    assert ffn.down_proj.bias is None


def test_gate_up_weight_independence():
    ffn = SiatSwiGLU(d_model=32, ffn_dim=64)
    assert ffn.gate_proj.weight is not ffn.up_proj.weight
    # Distinct storage even if values collide by chance after init.
    assert ffn.gate_proj.weight.data_ptr() != ffn.up_proj.weight.data_ptr()


def test_forward_matches_reference():
    ffn = SiatSwiGLU(d_model=4, ffn_dim=8, dropout=0.0)
    ffn.eval()
    x = torch.randn(2, 3, 4)
    with torch.no_grad():
        gate = F.silu(ffn.gate_proj(x))
        up = ffn.up_proj(x)
        expected = ffn.down_proj(gate * up)
        actual = ffn(x)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_gradient_flow():
    ffn = SiatSwiGLU(d_model=32, ffn_dim=64)
    x = torch.randn(2, 8, 32, requires_grad=True)
    output = ffn(x)
    loss = (output * torch.randn_like(output)).sum()
    loss.backward()
    assert x.grad is not None
    assert ffn.gate_proj.weight.grad is not None
    assert ffn.up_proj.weight.grad is not None
    assert ffn.down_proj.weight.grad is not None
    assert torch.isfinite(x.grad).all()
    assert not torch.isnan(x.grad).any()
    assert torch.isfinite(ffn.gate_proj.weight.grad).all()
    assert torch.isfinite(ffn.down_proj.weight.grad).all()


def test_dropout_zero_deterministic():
    ffn = SiatSwiGLU(d_model=32, ffn_dim=64, dropout=0.0)
    x = torch.randn(2, 8, 32)
    ffn.train()
    y1 = ffn(x)
    ffn.eval()
    y2 = ffn(x)
    y3 = ffn(x)
    assert torch.allclose(y1, y2, atol=1e-6)
    assert torch.allclose(y2, y3, atol=1e-6)


def test_token_independence():
    """FFN is position-wise: changing one token does not change another's FFN out."""
    ffn = SiatSwiGLU(d_model=16, ffn_dim=32, dropout=0.0)
    ffn.eval()
    a = torch.randn(1, 4, 16)
    b = a.clone()
    b[:, 2, :] = torch.randn(1, 16)
    with torch.no_grad():
        out_a = ffn(a)
        out_b = ffn(b)
    assert torch.allclose(out_a[:, 0, :], out_b[:, 0, :], atol=1e-6)
    assert torch.allclose(out_a[:, 1, :], out_b[:, 1, :], atol=1e-6)
    assert torch.allclose(out_a[:, 3, :], out_b[:, 3, :], atol=1e-6)
    assert not torch.allclose(out_a[:, 2, :], out_b[:, 2, :], atol=1e-6)


def test_tiny_config():
    config = ModelConfig.tiny()
    ffn = SiatSwiGLU(
        d_model=config.d_model,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
    )
    x = torch.randn(2, 16, config.d_model)
    assert ffn(x).shape == (2, 16, config.d_model)
    assert sum(p.numel() for p in ffn.parameters()) == (
        3 * config.d_model * config.ffn_dim
    )


def test_siat_30m_compatibility():
    config = ModelConfig.siat_30m()
    ffn = SiatSwiGLU(
        d_model=config.d_model,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
    )
    x = torch.randn(1, 8, config.d_model)
    assert ffn(x).shape == (1, 8, 512)
    assert sum(p.numel() for p in ffn.parameters()) == 2_359_296


def test_wrong_last_dim():
    ffn = SiatSwiGLU(d_model=32, ffn_dim=64)
    with pytest.raises(ValueError, match="d_model"):
        ffn(torch.randn(2, 4, 16))
