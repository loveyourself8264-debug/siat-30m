"""Tests for Siat RMSNorm."""

from __future__ import annotations

import pytest
import torch

from config import ModelConfig
from model.embedding import SiatEmbedding
from model.rmsnorm import SiatRMSNorm


def _reference_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Independent reference: y = x / RMS(x) * weight (no mean centering)."""
    x_f32 = x.float()
    rms_inv = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_f32 * rms_inv * weight.float()).to(dtype=x.dtype)


def test_output_shape():
    norm = SiatRMSNorm(d_model=32, eps=1e-6)
    x = torch.randn(2, 8, 32)
    out = norm(x)
    assert out.shape == (2, 8, 32)


def test_parameter_count_and_init():
    norm = SiatRMSNorm(d_model=32, eps=1e-6)
    assert sum(p.numel() for p in norm.parameters()) == 32
    assert torch.allclose(norm.weight, torch.ones_like(norm.weight))
    # No bias parameter.
    assert len(list(norm.parameters())) == 1


def test_matches_reference():
    norm = SiatRMSNorm(d_model=32, eps=1e-6)
    x = torch.randn(2, 8, 32)
    out = norm(x)
    expected = _reference_rmsnorm(x, norm.weight, eps=1e-6)
    assert torch.allclose(out, expected, rtol=1e-5, atol=1e-6)


def test_no_mean_centering_differs_from_layernorm_style():
    """RMSNorm must not subtract the mean (LayerNorm-style centering)."""
    norm = SiatRMSNorm(d_model=32, eps=1e-6)
    x = torch.randn(1, 4, 32)
    out = norm(x)
    # If we wrongly centered, result would differ from reference RMSNorm.
    centered = x - x.mean(dim=-1, keepdim=True)
    wrong = _reference_rmsnorm(centered, norm.weight, 1e-6)
    # For non-zero-mean inputs, centering changes the result.
    assert not torch.allclose(out, wrong, rtol=1e-3, atol=1e-4)


def test_gradient_flow():
    norm = SiatRMSNorm(d_model=32, eps=1e-6)
    x = torch.randn(2, 8, 32, requires_grad=True)
    out = norm(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert norm.weight.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(norm.weight.grad).all()
    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(norm.weight.grad).any()


def test_zero_input_finite():
    norm = SiatRMSNorm(d_model=32, eps=1e-6)
    x = torch.zeros(2, 8, 32)
    out = norm(x)
    assert torch.isfinite(out).all()
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_small_and_large_values_finite():
    norm = SiatRMSNorm(d_model=32, eps=1e-6)
    small = torch.full((2, 4, 32), 1e-8)
    large = torch.full((2, 4, 32), 1e3)
    for x in (small, large):
        out = norm(x)
        assert torch.isfinite(out).all()


def test_invalid_constructor():
    with pytest.raises(ValueError, match="d_model"):
        SiatRMSNorm(d_model=0)
    with pytest.raises(ValueError, match="eps"):
        SiatRMSNorm(d_model=32, eps=0.0)


def test_wrong_last_dim():
    norm = SiatRMSNorm(d_model=32)
    with pytest.raises(ValueError, match="last dimension"):
        norm(torch.randn(2, 8, 16))


def test_tiny_config_integration():
    config = ModelConfig.tiny()
    norm = SiatRMSNorm(d_model=config.d_model, eps=config.rms_norm_eps)
    x = torch.randn(2, 16, config.d_model)
    out = norm(x)
    assert out.shape == (2, 16, config.d_model)


def test_embedding_then_rmsnorm():
    emb = SiatEmbedding(vocab_size=100, d_model=32)
    norm = SiatRMSNorm(d_model=32, eps=1e-6)
    input_ids = torch.randint(0, 100, (2, 8), dtype=torch.long)
    hidden = emb(input_ids)
    out = norm(hidden)
    assert hidden.shape == (2, 8, 32)
    assert out.shape == (2, 8, 32)
    assert torch.isfinite(out).all()
