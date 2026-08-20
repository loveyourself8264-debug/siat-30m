"""Siat training package.

Loss, AdamW trainer, validation, checkpoint/resume, basic logging,
optional BF16 autocast.
"""

from train.loss import causal_lm_loss
from train.trainer import (
    SiatTrainer,
    build_adamw_param_groups,
    get_learning_rate,
    is_bf16_supported,
    require_bf16_support,
)

__all__ = [
    "causal_lm_loss",
    "SiatTrainer",
    "get_learning_rate",
    "build_adamw_param_groups",
    "is_bf16_supported",
    "require_bf16_support",
]
