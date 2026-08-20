"""Token embedding for Siat (input IDs → hidden states).

Maps token IDs ``[B, S]`` to dense vectors ``[B, S, D]`` via ``nn.Embedding``.
No learned positional embeddings (RoPE will be applied later on Attention Q/K).
No ``sqrt(d_model)`` scaling in this first version.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SiatEmbedding(nn.Module):
    """Lookup table from vocabulary indices to ``d_model`` vectors.

    Parameters
    ----------
    vocab_size:
        Number of tokens (must match tokenizer / ``ModelConfig.vocab_size``).
    d_model:
        Hidden size (must match ``ModelConfig.d_model``).
    padding_idx:
        Optional index whose embedding stays zero under PyTorch's rules.
        Causal LM pretraining usually does not need padding; default is None.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {vocab_size}.")
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}.")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )
        # Temporary init; Full Language Model stage may unify init policy.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[padding_idx].fill_(0.0)

    @property
    def weight(self) -> torch.Tensor:
        """Embedding matrix ``[vocab_size, d_model]`` for future LM-head tying."""
        return self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs.

        Parameters
        ----------
        input_ids:
            Long tensor of shape ``[B, S]``.

        Returns
        -------
        Tensor of shape ``[B, S, d_model]`` (same dtype as embedding weights).
        """
        return self.embedding(input_ids)
