"""Tests for Siat RoPE."""

from __future__ import annotations

import math

import pytest
import torch

from config import ModelConfig
from model.rope import SiatRoPE, apply_rotary_pos_emb, rotate_half


def test_creation():
    rope = SiatRoPE(head_dim=64, max_seq_len=128, theta=10000.0)
    assert rope.head_dim == 64
    assert rope.cos_cached.shape == (128, 64)
    assert rope.sin_cached.shape == (128, 64)


def test_odd_head_dim_rejected():
    with pytest.raises(ValueError, match="even"):
        SiatRoPE(head_dim=63, max_seq_len=32, theta=10000.0)


def test_invalid_max_seq_len():
    with pytest.raises(ValueError, match="max_seq_len"):
        SiatRoPE(head_dim=32, max_seq_len=0, theta=10000.0)


def test_invalid_theta():
    with pytest.raises(ValueError, match="theta"):
        SiatRoPE(head_dim=32, max_seq_len=16, theta=0.0)


def test_output_shape_preserved():
    rope = SiatRoPE(head_dim=32, max_seq_len=64, theta=10000.0)
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    q_rot, k_rot = rope(q, k)
    assert q_rot.shape == (2, 4, 16, 32)
    assert k_rot.shape == (2, 4, 16, 32)


def test_position_zero_unchanged():
    rope = SiatRoPE(head_dim=32, max_seq_len=16, theta=10000.0)
    q = torch.randn(1, 2, 8, 32)
    k = torch.randn(1, 2, 8, 32)
    q_rot, k_rot = rope(q, k)
    assert torch.allclose(q_rot[..., 0, :], q[..., 0, :], atol=1e-5, rtol=1e-5)
    assert torch.allclose(k_rot[..., 0, :], k[..., 0, :], atol=1e-5, rtol=1e-5)


def test_position_differs_from_zero():
    rope = SiatRoPE(head_dim=4, max_seq_len=8, theta=10000.0)
    # Same vector at every position; only position 0 should stay identical.
    vec = torch.tensor([1.0, 2.0, 3.0, 4.0])
    q = vec.view(1, 1, 1, 4).expand(1, 1, 2, 4).clone()
    k = q.clone()
    q_rot, _ = rope(q, k)
    assert torch.allclose(q_rot[0, 0, 0], vec, atol=1e-5)
    assert not torch.allclose(q_rot[0, 0, 1], vec, atol=1e-5)


def test_manual_2d_rotation():
    """head_dim=2, position=1: compare against hand-computed rotation."""
    theta = 10000.0
    head_dim = 2
    rope = SiatRoPE(head_dim=head_dim, max_seq_len=4, theta=theta)
    # inv_freq[0] = 1 / theta^(0) = 1 → angle at p=1 is 1.0
    angle = 1.0
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    x0, x1 = 3.0, -1.5
    q = torch.tensor([[[[x0, x1], [x0, x1]]]], dtype=torch.float32)  # [1,1,2,2]
    k = q.clone()
    q_rot, k_rot = rope(q, k)

    # Position 0 unchanged
    assert torch.allclose(q_rot[0, 0, 0], torch.tensor([x0, x1]), atol=1e-5)

    expected0 = x0 * cos_a - x1 * sin_a
    expected1 = x0 * sin_a + x1 * cos_a
    assert torch.allclose(
        q_rot[0, 0, 1],
        torch.tensor([expected0, expected1]),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(q_rot, k_rot)


def test_norm_preservation():
    rope = SiatRoPE(head_dim=32, max_seq_len=64, theta=10000.0)
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    q_rot, k_rot = rope(q, k)
    assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5, rtol=1e-5)
    assert torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_qk_same_input_same_output():
    rope = SiatRoPE(head_dim=16, max_seq_len=32, theta=10000.0)
    x = torch.randn(1, 2, 8, 16)
    q_rot, k_rot = rope(x, x.clone())
    assert torch.allclose(q_rot, k_rot)


def test_gradient_flow():
    rope = SiatRoPE(head_dim=16, max_seq_len=32, theta=10000.0)
    q = torch.randn(1, 2, 8, 16, requires_grad=True)
    k = torch.randn(1, 2, 8, 16, requires_grad=True)
    q_rot, k_rot = rope(q, k)
    loss = q_rot.sum() + k_rot.sum()
    loss.backward()
    assert q.grad is not None
    assert k.grad is not None
    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()
    assert not torch.isnan(q.grad).any()


def test_no_learnable_parameters():
    rope = SiatRoPE(head_dim=32, max_seq_len=64, theta=10000.0)
    assert sum(p.numel() for p in rope.parameters()) == 0
    # Buffers exist but are not parameters.
    assert "cos_cached" in dict(rope.named_buffers())
    assert "sin_cached" in dict(rope.named_buffers())


def test_seq_len_exceeds_max():
    rope = SiatRoPE(head_dim=8, max_seq_len=4, theta=10000.0)
    q = torch.randn(1, 1, 5, 8)
    k = torch.randn(1, 1, 5, 8)
    with pytest.raises(ValueError, match="max_seq_len"):
        rope(q, k)


def test_position_offset_exceeds():
    rope = SiatRoPE(head_dim=8, max_seq_len=8, theta=10000.0)
    q = torch.randn(1, 1, 4, 8)
    k = torch.randn(1, 1, 4, 8)
    with pytest.raises(ValueError, match="max_seq_len"):
        rope(q, k, position_offset=5)


def test_tiny_config_integration():
    config = ModelConfig.tiny()
    rope = SiatRoPE(
        head_dim=config.head_dim,
        max_seq_len=config.max_seq_len,
        theta=config.rope_theta,
    )
    b, h, s = 2, config.n_heads, 16
    q = torch.randn(b, h, s, config.head_dim)
    k = torch.randn(b, h, s, config.head_dim)
    q_rot, k_rot = rope(q, k)
    assert q_rot.shape == (b, h, s, config.head_dim)
    assert k_rot.shape == (b, h, s, config.head_dim)


def test_rotate_half_helper():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    # (-x1, x0, -x3, x2) = (-2, 1, -4, 3)
    assert torch.equal(rotate_half(x), torch.tensor([-2.0, 1.0, -4.0, 3.0]))


def test_apply_rotary_pos_emb_standalone():
    q = torch.randn(1, 1, 2, 4)
    k = torch.randn(1, 1, 2, 4)
    cos = torch.ones(1, 1, 2, 4)
    sin = torch.zeros(1, 1, 2, 4)
    q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
    assert torch.allclose(q_out, q)
    assert torch.allclose(k_out, k)
