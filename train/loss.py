"""Next-token causal language modeling loss for Siat.

Dataset already provides one-token-shifted labels::

    input_ids = tokens[i : i+S]
    labels    = tokens[i+1 : i+S+1]

Do **not** shift logits/labels again here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Mean cross-entropy over next-token predictions.

    Parameters
    ----------
    logits:
        Model outputs ``[B, S, V]`` (raw logits, not softmax).
    labels:
        Target token IDs ``[B, S]`` (``torch.long``), already shifted by Dataset.

    Returns
    -------
    Scalar mean loss over ``B * S`` positions.
    """
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("logits and labels must be torch.Tensor.")
    if logits.ndim != 3:
        raise ValueError(
            f"Expected logits rank 3 [B, S, V], got ndim={logits.ndim}."
        )
    if labels.ndim != 2:
        raise ValueError(
            f"Expected labels rank 2 [B, S], got ndim={labels.ndim}."
        )
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"logits batch/seq {tuple(logits.shape[:2])} must match "
            f"labels shape {tuple(labels.shape)}."
        )
    if labels.dtype != torch.long:
        raise ValueError(
            f"labels dtype must be torch.long, got {labels.dtype}."
        )

    vocab_size = logits.size(-1)
    # Flatten: [B, S, V] → [B*S, V], [B, S] → [B*S]. No second shift.
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
    )
