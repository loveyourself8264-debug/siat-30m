"""Siat attention building blocks.

Implemented so far:
  - Q/K/V linear projections + multi-head reshape (``SiatQKVProjection``)
  - Scaled QKᵀ scores (``scaled_dot_product_scores``)
  - Causal mask (``build_causal_mask`` / ``apply_causal_mask``)
  - Softmax → attention weights (``compute_attention_weights``)
  - V aggregation (``aggregate_values``)
  - Head merge (``merge_heads``) + output projection (``SiatAttentionOutput``)
  - Full causal self-attention assembly (``SiatSelfAttention``)

Next: SwiGLU → Transformer Block (RMSNorm / residual live outside this module).

Do not use ``nn.MultiheadAttention`` / ``F.scaled_dot_product_attention``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from model.rope import SiatRoPE


def scaled_dot_product_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Compute scaled attention logits ``(Q @ Kᵀ) / sqrt(Hd)``.

    Parameters
    ----------
    q, k:
        Shape ``[B, H, S, Hd]``.

    Returns
    -------
    scores:
        Shape ``[B, H, S, S]`` — **not** softmaxed; no causal mask applied.

    Scaling by ``1/sqrt(Hd)`` keeps dot-product magnitudes from growing with
    head dimension (which would later make softmax overly peaked). Softmax and
    masking are left to later stages.
    """
    if not isinstance(q, torch.Tensor) or not isinstance(k, torch.Tensor):
        raise TypeError("q and k must be torch.Tensor.")
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError(
            f"Expected q/k rank 4 [B, H, S, Hd], got q.ndim={q.ndim}, k.ndim={k.ndim}."
        )
    if q.shape[:3] != k.shape[:3] or q.size(-1) != k.size(-1):
        raise ValueError(
            f"q and k must share [B, H, S, Hd], got q={tuple(q.shape)}, "
            f"k={tuple(k.shape)}."
        )
    head_dim = q.size(-1)
    if head_dim <= 0:
        raise ValueError(f"head_dim must be > 0, got {head_dim}.")

    # Kᵀ: [B, H, S, Hd] → [B, H, Hd, S]
    # Q @ Kᵀ: [B, H, S, S]
    scale = 1.0 / math.sqrt(head_dim)
    return torch.matmul(q, k.transpose(-2, -1)) * scale


def build_causal_mask(
    seq_len: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """Build a lower-triangular allow-mask including the diagonal.

    Shape ``[1, 1, S, S]`` for broadcasting over ``[B, H, S, S]`` scores.

    ``True`` means attend (key_pos <= query_pos); ``False`` means future (blocked).
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}.")
    # tril includes diagonal: query i may see keys 0..i
    allow = torch.tril(
        torch.ones(seq_len, seq_len, device=device, dtype=dtype)
    )
    return allow.view(1, 1, seq_len, seq_len)


def apply_causal_mask(scores: torch.Tensor) -> torch.Tensor:
    """Mask future key positions in attention logits (non-inplace).

    Parameters
    ----------
    scores:
        Attention logits ``[..., S, S]`` (typically ``[B, H, S, S]``).

    Returns
    -------
    masked_scores:
        Same shape; allowed positions unchanged; future positions filled with
        ``torch.finfo(scores.dtype).min``.

    We use ``finfo(...).min`` instead of hard-coding ``-inf`` so FP16/BF16
    stay within a representable range for that dtype when mixed precision is
    used later. Softmax is **not** applied here.
    """
    if not isinstance(scores, torch.Tensor):
        raise TypeError("scores must be a torch.Tensor.")
    if scores.ndim < 2:
        raise ValueError(
            f"Expected scores with at least 2 dims [..., S, S], got ndim={scores.ndim}."
        )
    seq_q, seq_k = scores.shape[-2], scores.shape[-1]
    if seq_q != seq_k:
        raise ValueError(
            f"Last two dims must form a square [S, S], got {seq_q} x {seq_k}."
        )

    allow = build_causal_mask(
        seq_q, device=scores.device, dtype=torch.bool
    )
    # finfo.min is dtype-safe for future FP16/BF16; future → blocked for softmax.
    fill = torch.finfo(scores.dtype).min
    return scores.masked_fill(~allow, fill)


def compute_attention_weights(masked_scores: torch.Tensor) -> torch.Tensor:
    """Softmax over the key dimension → attention probabilities.

    Parameters
    ----------
    masked_scores:
        Causal-masked logits ``[..., S, S]`` (typically ``[B, H, S, S]``).

    Returns
    -------
    attention_weights:
        Same shape; each query row sums to 1 along ``dim=-1``.
        Future (masked) keys should have probability ~0.

    Softmax runs in float32 when the input is FP16/BF16 for numerical
    stability under future mixed precision, then casts back. Float32 inputs
    use ``torch.softmax`` directly. Does not mutate ``masked_scores``.
    """
    if not isinstance(masked_scores, torch.Tensor):
        raise TypeError("masked_scores must be a torch.Tensor.")
    if masked_scores.ndim < 2:
        raise ValueError(
            f"Expected masked_scores with at least 2 dims [..., S, S], "
            f"got ndim={masked_scores.ndim}."
        )
    seq_q, seq_k = masked_scores.shape[-2], masked_scores.shape[-1]
    if seq_q != seq_k:
        raise ValueError(
            f"Last two dims must form a square [S, S], got {seq_q} x {seq_k}."
        )

    original_dtype = masked_scores.dtype
    # FP16/BF16: compute softmax in FP32, then cast back.
    if original_dtype in (torch.float16, torch.bfloat16):
        weights = torch.softmax(masked_scores.float(), dim=-1)
        return weights.to(dtype=original_dtype)
    return torch.softmax(masked_scores, dim=-1)


def aggregate_values(
    attention_weights: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Weighted sum of Value vectors: ``Context = AttentionWeights @ V``.

    Parameters
    ----------
    attention_weights:
        Softmax probabilities ``[B, H, S, S]`` (query × key). Already causal;
        do not re-mask or re-softmax here.
    v:
        Values ``[B, H, S, Hd]``.

    Returns
    -------
    context:
        Per-head context ``[B, H, S, Hd]``.

    Shape math: ``[S, S] @ [S, Hd] → [S, Hd]`` per batch/head.
    No learnable parameters; head merge / ``o_proj`` are later stages.
    """
    if not isinstance(attention_weights, torch.Tensor) or not isinstance(
        v, torch.Tensor
    ):
        raise TypeError("attention_weights and v must be torch.Tensor.")
    if attention_weights.ndim != 4 or v.ndim != 4:
        raise ValueError(
            f"Expected rank-4 weights and v, got weights.ndim={attention_weights.ndim}, "
            f"v.ndim={v.ndim}."
        )
    if attention_weights.shape[0] != v.shape[0] or attention_weights.shape[1] != v.shape[1]:
        raise ValueError(
            f"Batch/head mismatch: weights={tuple(attention_weights.shape)}, "
            f"v={tuple(v.shape)}."
        )
    # key length of weights must match V sequence length
    if attention_weights.shape[-1] != v.shape[-2]:
        raise ValueError(
            f"weights key length {attention_weights.shape[-1]} must equal "
            f"v sequence length {v.shape[-2]}."
        )
    # Self-attention: query S == key/value S
    if attention_weights.shape[-2] != v.shape[-2]:
        raise ValueError(
            f"Self-attention requires query S == V S, got "
            f"query_S={attention_weights.shape[-2]}, V_S={v.shape[-2]}."
        )

    # [B, H, S, S] @ [B, H, S, Hd] → [B, H, S, Hd]
    return torch.matmul(attention_weights, v)


class SiatQKVProjection(nn.Module):
    """Project hidden states to multi-head Q, K, V tensors.

    Standard Multi-Head Attention layout (no GQA/MQA)::

        x [B, S, D]
          → q/k/v_proj (separate Linear, bias=False by default)
          → [B, S, D]
          → split heads
          → q, k, v  each [B, H, S, Hd]

    where ``Hd = d_model // n_heads``.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}.")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be > 0, got {n_heads}.")
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})."
            )

        head_dim = d_model // n_heads
        # RoPE requires even head_dim (adjacent pairs).
        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim (= d_model // n_heads = {head_dim}) must be even "
                f"for RoPE, but d_model={d_model}, n_heads={n_heads}."
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.bias = bias

        # Independent Wq, Wk, Wv. Default bias=False for Siat.
        # Weight init: leave nn.Linear defaults; Full LM may unify init later.
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape ``[B, S, D]`` → ``[B, H, S, Hd]``."""
        batch, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(
                f"Expected last dim {self.d_model}, got {d_model}."
            )
        # [B, S, D] → [B, S, H, Hd] → [B, H, S, Hd]
        x = x.view(batch, seq_len, self.n_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return multi-head Q, K, V.

        Parameters
        ----------
        hidden_states:
            ``[B, S, D]``

        Returns
        -------
        q, k, v:
            Each ``[B, H, S, Hd]``. No attention scores are computed.
        """
        # [B, S, D]
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        # [B, H, S, Hd]
        return self._split_heads(q), self._split_heads(k), self._split_heads(v)


def merge_heads(context: torch.Tensor) -> torch.Tensor:
    """Inverse of head split: ``[B, H, S, Hd]`` → ``[B, S, D]`` with ``D = H × Hd``.

    Matches ``SiatQKVProjection._split_heads`` (view → transpose) in reverse:
    transpose heads/seq, then contiguous view to merge ``H`` and ``Hd``.
    """
    if not isinstance(context, torch.Tensor):
        raise TypeError("context must be a torch.Tensor.")
    if context.ndim != 4:
        raise ValueError(
            f"Expected context rank 4 [B, H, S, Hd], got ndim={context.ndim}."
        )
    batch, n_heads, seq_len, head_dim = context.shape
    if n_heads <= 0 or seq_len <= 0 or head_dim <= 0:
        raise ValueError(
            f"H, S, Hd must be > 0, got H={n_heads}, S={seq_len}, Hd={head_dim}."
        )
    d_model = n_heads * head_dim
    # [B, H, S, Hd] → [B, S, H, Hd] → [B, S, D]
    return (
        context.transpose(1, 2)
        .contiguous()
        .view(batch, seq_len, d_model)
    )


class SiatAttentionOutput(nn.Module):
    """Output projection ``W_O`` on already-merged attention context.

    Expects ``[B, S, D]`` (after ``merge_heads``). Does not merge heads, add
    residual, or apply RMSNorm/dropout — those belong to later stages.
    """

    def __init__(self, d_model: int, bias: bool = False) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}.")
        self.d_model = d_model
        self.bias = bias
        # Default nn.Linear init; Full LM may unify init later.
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, merged_context: torch.Tensor) -> torch.Tensor:
        """Project ``[B, S, D]`` → ``[B, S, D]``."""
        if merged_context.size(-1) != self.d_model:
            raise ValueError(
                f"Expected last dim {self.d_model}, got {merged_context.size(-1)}."
            )
        return self.o_proj(merged_context)


class SiatSelfAttention(nn.Module):
    """Causal multi-head self-attention assembled from Siat primitives.

    Pipeline::

        x [B, S, D]
          → QKV → RoPE(Q, K)
          → scaled QKᵀ → causal mask → softmax
          → attn dropout → weights @ V
          → merge heads → W_O
          → [B, S, D]

    No residual, no RMSNorm, no KV cache. Standard MHA (H_q = H_k = H_v).
    Learnable params are Q/K/V/O only (``4 * D²`` when bias=False).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {d_model}.")
        if n_heads <= 0:
            raise ValueError(f"n_heads must be > 0, got {n_heads}.")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}.")
        if rope_theta <= 0:
            raise ValueError(f"rope_theta must be > 0, got {rope_theta}.")
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(
                f"dropout must be in [0.0, 1.0], got {dropout}."
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.rope_theta = float(rope_theta)
        self.dropout = float(dropout)

        # QKV validates d_model % n_heads and even head_dim for RoPE.
        self.qkv = SiatQKVProjection(d_model, n_heads, bias=False)
        self.head_dim = self.qkv.head_dim
        self.rope = SiatRoPE(
            head_dim=self.head_dim,
            max_seq_len=max_seq_len,
            theta=rope_theta,
        )
        self.output = SiatAttentionOutput(d_model, bias=False)
        # Softmax → dropout → @V; dropout=0.0 is identity.
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run causal self-attention.

        Parameters
        ----------
        hidden_states:
            ``[B, S, D]`` with ``D == d_model`` and ``S <= max_seq_len``.
        return_attention_weights:
            If True, also return pre-dropout softmax weights ``[B, H, S, S]``.

        Returns
        -------
        output:
            ``[B, S, D]`` attention output (no residual).
        weights (optional):
            Softmax attention weights before dropout.
        """
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("hidden_states must be a torch.Tensor.")
        if hidden_states.ndim != 3:
            raise ValueError(
                f"Expected hidden_states rank 3 [B, S, D], "
                f"got ndim={hidden_states.ndim}."
            )
        batch, seq_len, d_model = hidden_states.shape
        if d_model != self.d_model:
            raise ValueError(
                f"Expected last dim d_model={self.d_model}, got {d_model}."
            )
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}."
            )

        # 1. Q/K/V Projection → [B, H, S, Hd]
        q, k, v = self.qkv(hidden_states)
        # 2. RoPE on Q and K only
        q, k = self.rope(q, k)
        # 3–5. Scaled QKᵀ → causal mask → softmax
        scores = scaled_dot_product_scores(q, k)
        masked = apply_causal_mask(scores)
        weights = compute_attention_weights(masked)
        # 6. Dropout then AttentionWeights @ V → context [B, H, S, Hd]
        dropped = self.attn_dropout(weights)
        context = aggregate_values(dropped, v)
        # 7–8. Head merge → output projection → [B, S, D]
        merged = merge_heads(context)
        output = self.output(merged)

        if return_attention_weights:
            return output, weights
        return output
