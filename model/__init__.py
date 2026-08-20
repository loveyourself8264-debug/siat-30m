"""Siat model package (decoder-only Transformer components).

Implemented: Embedding, RMSNorm, RoPE, Attention, SwiGLU, Transformer Block,
Full Causal LM (``SiatForCausalLM``).
Next: Tiny Overfit / Training → …
완성된 nn.Transformer / Hugging Face LLM 구현체에는 의존하지 않는다.
"""

from model.attention import (
    SiatAttentionOutput,
    SiatQKVProjection,
    SiatSelfAttention,
    aggregate_values,
    apply_causal_mask,
    build_causal_mask,
    compute_attention_weights,
    merge_heads,
    scaled_dot_product_scores,
)
from model.block import SiatTransformerBlock
from model.embedding import SiatEmbedding
from model.ffn import SiatSwiGLU
from model.model import SiatForCausalLM, analytical_param_count
from model.rmsnorm import SiatRMSNorm
from model.rope import SiatRoPE, apply_rotary_pos_emb

__all__ = [
    "SiatEmbedding",
    "SiatRMSNorm",
    "SiatRoPE",
    "apply_rotary_pos_emb",
    "SiatQKVProjection",
    "scaled_dot_product_scores",
    "build_causal_mask",
    "apply_causal_mask",
    "compute_attention_weights",
    "aggregate_values",
    "merge_heads",
    "SiatAttentionOutput",
    "SiatSelfAttention",
    "SiatSwiGLU",
    "SiatTransformerBlock",
    "SiatForCausalLM",
    "analytical_param_count",
]
