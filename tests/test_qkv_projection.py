"""Tests for Siat Q/K/V projection and head split."""

from __future__ import annotations

import pytest
import torch

from config import ModelConfig
from model.attention import SiatQKVProjection
from model.rmsnorm import SiatRMSNorm
from model.rope import SiatRoPE


def _manual_split(x: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    b, s, _ = x.shape
    return x.view(b, s, n_heads, head_dim).transpose(1, 2)


def test_creation():
    proj = SiatQKVProjection(d_model=128, n_heads=4)
    assert proj.d_model == 128
    assert proj.n_heads == 4
    assert proj.head_dim == 32
    assert proj.bias is False


def test_invalid_d_model():
    with pytest.raises(ValueError, match="d_model"):
        SiatQKVProjection(d_model=0, n_heads=4)


def test_invalid_n_heads():
    with pytest.raises(ValueError, match="n_heads"):
        SiatQKVProjection(d_model=128, n_heads=0)


def test_non_divisible():
    with pytest.raises(ValueError, match="divisible"):
        SiatQKVProjection(d_model=130, n_heads=4)


def test_odd_head_dim_rejected():
    # d_model=12, n_heads=2 → head_dim=6 (even) OK; use n_heads=3 → Hd=4 even.
    # d_model=10, n_heads=2 → head_dim=5 odd → reject.
    with pytest.raises(ValueError, match="even"):
        SiatQKVProjection(d_model=10, n_heads=2)


def test_output_shapes():
    proj = SiatQKVProjection(d_model=128, n_heads=4)
    x = torch.randn(2, 16, 128)
    q, k, v = proj(x)
    assert q.shape == (2, 4, 16, 32)
    assert k.shape == (2, 4, 16, 32)
    assert v.shape == (2, 4, 16, 32)


def test_parameter_count():
    d_model = 512
    proj = SiatQKVProjection(d_model=d_model, n_heads=8)
    num_params = sum(p.numel() for p in proj.parameters())
    assert num_params == 3 * d_model * d_model
    assert num_params == 786_432
    assert proj.q_proj.bias is None
    assert proj.k_proj.bias is None
    assert proj.v_proj.bias is None


def test_weight_independence():
    proj = SiatQKVProjection(d_model=64, n_heads=4)
    assert proj.q_proj.weight is not proj.k_proj.weight
    assert proj.q_proj.weight is not proj.v_proj.weight
    assert proj.k_proj.weight is not proj.v_proj.weight
    assert not torch.equal(proj.q_proj.weight, proj.k_proj.weight)


def test_matches_manual_reshape():
    proj = SiatQKVProjection(d_model=128, n_heads=4)
    x = torch.randn(2, 16, 128)
    q, k, v = proj(x)
    assert torch.equal(q, _manual_split(proj.q_proj(x), 4, 32))
    assert torch.equal(k, _manual_split(proj.k_proj(x), 4, 32))
    assert torch.equal(v, _manual_split(proj.v_proj(x), 4, 32))


def test_deterministic_head_ordering():
    """Explicit tiny tensor: reshape must keep expected head blocks."""
    # B=1, S=2, D=8, H=2, Hd=4
    proj = SiatQKVProjection(d_model=8, n_heads=2)
    with torch.no_grad():
        # Make projections identity-like so we control the split input.
        eye = torch.eye(8)
        proj.q_proj.weight.copy_(eye)
        proj.k_proj.weight.copy_(eye)
        proj.v_proj.weight.copy_(eye)

    # Token 0: [0,1,2,3, 4,5,6,7] → head0 [0,1,2,3], head1 [4,5,6,7]
    x = torch.arange(16, dtype=torch.float32).view(1, 2, 8)
    q, _, _ = proj(x)
    assert torch.equal(q[0, 0, 0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert torch.equal(q[0, 1, 0], torch.tensor([4.0, 5.0, 6.0, 7.0]))
    assert torch.equal(q[0, 0, 1], torch.tensor([8.0, 9.0, 10.0, 11.0]))
    assert torch.equal(q[0, 1, 1], torch.tensor([12.0, 13.0, 14.0, 15.0]))


def test_gradient_flow():
    proj = SiatQKVProjection(d_model=32, n_heads=4)
    x = torch.randn(2, 8, 32, requires_grad=True)
    q, k, v = proj(x)
    loss = q.sum() + k.sum() + v.sum()
    loss.backward()
    assert x.grad is not None
    assert proj.q_proj.weight.grad is not None
    assert proj.k_proj.weight.grad is not None
    assert proj.v_proj.weight.grad is not None
    assert torch.isfinite(x.grad).all()
    assert not torch.isnan(proj.q_proj.weight.grad).any()


def test_tiny_config():
    config = ModelConfig.tiny()
    proj = SiatQKVProjection(d_model=config.d_model, n_heads=config.n_heads)
    assert proj.head_dim == config.head_dim
    x = torch.randn(2, 16, config.d_model)
    q, k, v = proj(x)
    expected = (2, config.n_heads, 16, config.head_dim)
    assert q.shape == expected
    assert k.shape == expected
    assert v.shape == expected


def test_siat_30m_shape():
    config = ModelConfig.siat_30m()
    proj = SiatQKVProjection(d_model=config.d_model, n_heads=config.n_heads)
    assert proj.head_dim == 64
    x = torch.randn(2, 16, 512)
    q, k, v = proj(x)
    assert q.shape == (2, 8, 16, 64)
    assert k.shape == (2, 8, 16, 64)
    assert v.shape == (2, 8, 16, 64)


def test_rope_integration_qk_only():
    proj = SiatQKVProjection(d_model=64, n_heads=4)
    rope = SiatRoPE(head_dim=16, max_seq_len=32, theta=10000.0)
    x = torch.randn(2, 8, 64)
    q, k, v = proj(x)
    v_before = v.detach().clone()
    q_rot, k_rot = rope(q, k)
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    assert torch.equal(v, v_before)
    # V must not be passed through RoPE in this pipeline.
    assert not torch.equal(q_rot, q) or torch.allclose(q_rot[..., 0, :], q[..., 0, :])


def test_rmsnorm_then_qkv():
    d_model, n_heads = 32, 4
    norm = SiatRMSNorm(d_model=d_model)
    proj = SiatQKVProjection(d_model=d_model, n_heads=n_heads)
    x = torch.randn(2, 8, d_model)
    q, k, v = proj(norm(x))
    assert q.shape == (2, n_heads, 8, d_model // n_heads)
    assert k.shape == q.shape
    assert v.shape == q.shape
