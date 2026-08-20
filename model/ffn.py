"""Siat SwiGLU feed-forward network.

Structure::

    x [B, S, D]
      ├── gate_proj → SiLU ──┐
      │                      × → dropout → down_proj → [B, S, D]
      └── up_proj ───────────┘

``hidden = SiLU(gate_proj(x)) * up_proj(x)`` (element-wise), then down-project.
No bias on Linear layers. Residual / RMSNorm belong to the Transformer Block.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiatSwiGLU(nn.Module):
    """SwiGLU FFN: ``down(dropout(SiLU(gate(x)) * up(x)))``.

    Parameters
    ----------
    d_model:
        Input/output hidden size ``D``.
    ffn_dim:
        Intermediate width ``F``.
    dropout:
        Applied after the gated product, before ``down_proj``. ``0.0`` is identity.
    """

    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}.")
        if ffn_dim <= 0:
            raise ValueError(f"ffn_dim must be > 0, got {ffn_dim}.")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(
                f"dropout must be in [0.0, 1.0], got {dropout}."
            )

        self.d_model = d_model
        self.ffn_dim = ffn_dim
        self.dropout = float(dropout)

        # Independent W_gate, W_up, W_down. Default bias=False for Siat.
        # Weight init: leave nn.Linear defaults; Full LM may unify init later.
        self.gate_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.up_proj = nn.Linear(d_model, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, d_model, bias=False)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU: ``[B, S, D]`` → ``[B, S, D]``."""
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

        # [B, S, F]
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = self.ffn_dropout(gate * up)
        # [B, S, D]
        return self.down_proj(hidden)
