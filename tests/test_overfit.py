"""Tiny overfit smoke tests for the full Siat train path."""

from __future__ import annotations

import torch

from model.model import SiatForCausalLM
from train.loss import causal_lm_loss
from train.tiny_overfit import (
    OVERFIT_LR,
    OVERFIT_SEED,
    OVERFIT_WEIGHT_DECAY,
    build_synthetic_batch,
    make_overfit_config,
    next_token_accuracy,
    run_tiny_overfit,
)


def test_tiny_model_forward_loss_finite():
    torch.manual_seed(OVERFIT_SEED)
    config = make_overfit_config()
    model = SiatForCausalLM(config)
    model.train()
    input_ids, labels = build_synthetic_batch(
        config.vocab_size, seq_len=8, batch_size=2, seed=OVERFIT_SEED
    )
    logits = model(input_ids)
    loss = causal_lm_loss(logits, labels)
    assert logits.shape == (2, 8, config.vocab_size)
    assert torch.isfinite(loss)


def test_one_step_backward_and_optimizer():
    torch.manual_seed(OVERFIT_SEED)
    config = make_overfit_config()
    model = SiatForCausalLM(config)
    model.train()
    input_ids, labels = build_synthetic_batch(
        config.vocab_size, seq_len=8, batch_size=2, seed=OVERFIT_SEED
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=OVERFIT_LR, weight_decay=OVERFIT_WEIGHT_DECAY
    )
    optimizer.zero_grad(set_to_none=True)
    loss = causal_lm_loss(model(input_ids), labels)
    loss.backward()
    assert model.embed.weight.grad is not None
    assert model.layers[0].attention.qkv.q_proj.weight.grad is not None
    assert model.layers[0].ffn.gate_proj.weight.grad is not None
    assert model.norm.weight.grad is not None
    assert torch.isfinite(model.embed.weight.grad).all()
    optimizer.step()


def test_overfit_loss_reduces_significantly():
    result = run_tiny_overfit(
        steps=60,
        lr=OVERFIT_LR,
        seed=OVERFIT_SEED,
        seq_len=16,
        batch_size=4,
        device="cpu",
        log_every=1000,
        verbose=False,
    )
    assert torch.isfinite(torch.tensor(result.initial_loss))
    assert torch.isfinite(torch.tensor(result.final_loss))
    assert result.final_loss < result.initial_loss * 0.5
    # Accuracy should improve on memorized batch
    assert result.final_accuracy >= result.initial_accuracy
