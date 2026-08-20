"""Siat Transformer block (Pre-Norm decoder style).

::

    h = x + SelfAttention(RMSNorm(x))
    y = h + SwiGLU(RMSNorm(h))

Residual adds to the **pre-norm** branch input, not the normalized tensor.
No final RMSNorm, LM head, or layer stack here — those belong to Full LM.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.attention import SiatSelfAttention
from model.ffn import SiatSwiGLU
from model.rmsnorm import SiatRMSNorm


class SiatTransformerBlock(nn.Module):
    """One Pre-Norm causal Transformer block.

    Parameters match ``ModelConfig`` fields but Config is not imported here.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}.")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be > 0, got {n_heads}.")
        if ffn_dim <= 0:
            raise ValueError(f"ffn_dim must be > 0, got {ffn_dim}.")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}.")
        if rms_norm_eps <= 0:
            raise ValueError(f"rms_norm_eps must be > 0, got {rms_norm_eps}.")
        if rope_theta <= 0:
            raise ValueError(f"rope_theta must be > 0, got {rope_theta}.")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(
                f"dropout must be in [0.0, 1.0], got {dropout}."
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.max_seq_len = max_seq_len
        self.rope_theta = float(rope_theta)
        self.rms_norm_eps = float(rms_norm_eps)
        self.dropout = float(dropout)

        # Independent norms — do not share weights.
        self.attn_norm = SiatRMSNorm(d_model, eps=rms_norm_eps)
        self.attention = SiatSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            dropout=dropout,
        )
        self.ffn_norm = SiatRMSNorm(d_model, eps=rms_norm_eps)
        self.ffn = SiatSwiGLU(
            d_model=d_model,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-Norm block: ``[B, S, D]`` → ``[B, S, D]``."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor.")
        if x.ndim != 3:
            raise ValueError(
                f"Expected x rank 3 [B, S, D], got ndim={x.ndim}."
            )
        if x.size(-1) != self.d_model:
            raise ValueError(
                f"Expected last dim d_model={self.d_model}, got {x.size(-1)}."
            )

        # Attention branch (residual = original x)
        residual = x
        x = residual + self.attention(self.attn_norm(x))

        # FFN branch (residual = post-attention x)
        residual = x
        x = residual + self.ffn(self.ffn_norm(x))
        return x
