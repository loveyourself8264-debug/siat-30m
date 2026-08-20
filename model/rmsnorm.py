"""Siat RMSNorm (root-mean-square normalization).

Unlike LayerNorm, RMSNorm does **not** subtract the mean:

    LayerNorm:  (x - mean) / std
    RMSNorm:    x / RMS(x) * weight

where ``RMS(x) = sqrt(mean(x²) + eps)`` over the last (hidden) dimension.

Implemented with plain tensor ops — not ``nn.RMSNorm`` / ``nn.LayerNorm``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SiatRMSNorm(nn.Module):
    """Normalize hidden features by RMS, then apply a learnable scale.

    Parameters
    ----------
    d_model:
        Hidden size (last dimension of the input).
    eps:
        Added under the square root for numerical stability (default ``1e-6``).
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}.")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}.")

        self.d_model = d_model
        self.eps = eps
        # Learnable scale only; no bias.
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm over the last dimension.

        Expects ``x[..., d_model]``. Typical Siat shape is ``[B, S, D]``.
        """
        if x.size(-1) != self.d_model:
            raise ValueError(
                f"Expected last dimension {self.d_model}, got {x.size(-1)}."
            )

        # Compute RMS in float32 for stability under future FP16/BF16 training,
        # then cast back to the input dtype.
        x_f32 = x.float()
        # mean(x²) over D; no mean centering (that would be LayerNorm).
        variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f32 * torch.rsqrt(variance + self.eps)
        return (x_norm * self.weight.float()).to(dtype=x.dtype)
