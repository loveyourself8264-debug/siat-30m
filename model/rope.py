"""Siat RoPE (Rotary Positional Embedding).

Applies 2D rotations to adjacent feature pairs of Query/Key tensors.
Not added to token embeddings; intended for Attention Q/K after head split.

Pairing convention (adjacent pairs)::

    (x0, x1), (x2, x3), ...

    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos

Inverse frequencies::

    inv_freq[i] = 1 / theta^(2i / head_dim),  i = 0 .. head_dim/2 - 1

No learnable parameters — cos/sin are registered buffers.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Map ``(x0, x1, x2, x3, ...)`` → ``(-x1, x0, -x3, x2, ...)`` (adjacent pairs).

    Used so that ``x * cos + rotate_half(x) * sin`` equals the 2D rotation above.
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    # Stack as [..., Hd/2, 2] then flatten to interleave (-odd, even).
    rotated = torch.stack((-x_odd, x_even), dim=-1)
    return rotated.flatten(-2)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q and K with shared cos/sin.

    Parameters
    ----------
    q, k:
        Shape ``[B, H, S, Hd]``.
    cos, sin:
        Broadcastable to ``[1, 1, S, Hd]`` (or already that shape).
    """
    # Broadcast over batch and heads: cos/sin are [S, Hd] or [1, 1, S, Hd].
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


class SiatRoPE(nn.Module):
    """Precomputed cos/sin rotary embeddings for attention head features.

    Parameters
    ----------
    head_dim:
        Per-head size (must be even). Matches ``ModelConfig.head_dim``.
    max_seq_len:
        Maximum sequence length cached (matches ``ModelConfig.max_seq_len``).
    theta:
        RoPE base frequency (matches ``ModelConfig.rope_theta``).
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        if head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {head_dim}.")
        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim must be even for RoPE adjacent pairs, got {head_dim}."
            )
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}.")
        if theta <= 0:
            raise ValueError(f"theta must be > 0, got {theta}.")

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = float(theta)

        # Build caches in float32 for numerical stability; cast at apply time.
        inv_freq, cos, sin = self._build_cache(head_dim, max_seq_len, self.theta)
        # persistent=True: move with .to(device) and appear in state_dict;
        # not optimizer parameters.
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        self.register_buffer("cos_cached", cos, persistent=True)
        self.register_buffer("sin_cached", sin, persistent=True)

    @staticmethod
    def _build_cache(
        head_dim: int,
        max_seq_len: int,
        theta: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # inv_freq[i] = 1 / theta^(2i / head_dim)
        freqs_index = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (freqs_index / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        # angles[p, i] = p * inv_freq[i]  → [max_seq_len, head_dim/2]
        angles = torch.outer(positions, inv_freq)
        # Duplicate each angle for the two dims of an adjacent pair → [S, Hd]
        cos = torch.cos(angles).repeat_interleave(2, dim=-1)
        sin = torch.sin(angles).repeat_interleave(2, dim=-1)
        return inv_freq, cos, sin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate Q and K.

        Parameters
        ----------
        q, k:
            ``[B, H, S, Hd]`` with ``Hd == head_dim``.
        position_offset:
            Starting position index (default 0). Positions used are
            ``offset .. offset+S-1``. Must satisfy ``offset + S <= max_seq_len``.
        """
        if q.ndim != 4 or k.ndim != 4:
            raise ValueError(
                f"Expected q/k shape [B, H, S, Hd], got q={tuple(q.shape)}, "
                f"k={tuple(k.shape)}."
            )
        if q.shape != k.shape:
            raise ValueError(
                f"q and k must have the same shape, got {tuple(q.shape)} vs "
                f"{tuple(k.shape)}."
            )
        if q.size(-1) != self.head_dim:
            raise ValueError(
                f"Expected head_dim={self.head_dim}, got {q.size(-1)}."
            )
        if position_offset < 0:
            raise ValueError(
                f"position_offset must be >= 0, got {position_offset}."
            )

        seq_len = q.size(-2)
        end = position_offset + seq_len
        if end > self.max_seq_len:
            raise ValueError(
                f"Sequence end position {end} exceeds max_seq_len "
                f"{self.max_seq_len} (offset={position_offset}, S={seq_len})."
            )

        cos = self.cos_cached[position_offset:end]  # [S, Hd]
        sin = self.sin_cached[position_offset:end]
        # [1, 1, S, Hd] for broadcast over B, H
        cos = cos.view(1, 1, seq_len, self.head_dim).to(dtype=q.dtype)
        sin = sin.view(1, 1, seq_len, self.head_dim).to(dtype=q.dtype)
        return apply_rotary_pos_emb(q, k, cos, sin)
