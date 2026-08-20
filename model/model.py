"""Siat decoder-only causal language model.

Assembles Embedding → Transformer Blocks × N → Final RMSNorm → LM Head.

No learned positional embeddings (RoPE lives inside attention).
No loss / generation / KV cache in this module.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import ModelConfig
from model.block import SiatTransformerBlock
from model.embedding import SiatEmbedding
from model.rmsnorm import SiatRMSNorm


def analytical_param_count(config: ModelConfig) -> int:
    """Expected parameter count for bias-free Siat (RoPE has no params).

    Tied: ``V*D + N*(4*D² + 3*D*F + 2*D) + D``
    Untied: same + ``V*D`` for a separate LM head.
    """
    v, d, n, f = (
        config.vocab_size,
        config.d_model,
        config.n_layers,
        config.ffn_dim,
    )
    blocks = n * (4 * d * d + 3 * d * f + 2 * d)
    total = v * d + blocks + d
    if not config.tie_embeddings:
        total += v * d
    return total


class SiatForCausalLM(nn.Module):
    """Decoder-only LM: ``input_ids [B,S]`` → ``logits [B,S,V]``.

    Parameters
    ----------
    config:
        ``ModelConfig`` (tiny / siat_30m / custom). Not mutated.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if not isinstance(config, ModelConfig):
            raise TypeError(
                f"config must be ModelConfig, got {type(config).__name__}."
            )
        # Re-validate in case a caller bypassed __post_init__.
        config.validate()

        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self.max_seq_len = config.max_seq_len
        self.tie_embeddings = config.tie_embeddings

        self.embed = SiatEmbedding(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
        )
        self.layers = nn.ModuleList(
            [
                SiatTransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    ffn_dim=config.ffn_dim,
                    max_seq_len=config.max_seq_len,
                    rope_theta=config.rope_theta,
                    rms_norm_eps=config.rms_norm_eps,
                    dropout=config.dropout,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.norm = SiatRMSNorm(
            d_model=config.d_model,
            eps=config.rms_norm_eps,
        )
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        if config.tie_embeddings:
            # Share the same Parameter object (not a data copy).
            self.lm_head.weight = self.embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map token IDs to vocabulary logits (no softmax).

        Parameters
        ----------
        input_ids:
            Long tensor ``[B, S]`` with ``S <= max_seq_len``.

        Returns
        -------
        logits:
            ``[B, S, vocab_size]``.
        """
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids must be a torch.Tensor.")
        if input_ids.ndim != 2:
            raise ValueError(
                f"Expected input_ids rank 2 [B, S], got ndim={input_ids.ndim}."
            )
        seq_len = input_ids.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}."
            )

        # [B, S] → [B, S, D]
        hidden_states = self.embed(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        # Final RMSNorm (independent of per-block norms)
        hidden_states = self.norm(hidden_states)
        # [B, S, V] — raw logits, no softmax
        return self.lm_head(hidden_states)
