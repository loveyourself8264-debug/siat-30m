"""Tests for Siat token embedding."""

from __future__ import annotations

import pytest
import torch

from config import ModelConfig
from model.embedding import SiatEmbedding


def test_siat_embedding_creation():
    emb = SiatEmbedding(vocab_size=100, d_model=32)
    assert emb.vocab_size == 100
    assert emb.d_model == 32
    assert emb.weight.shape == (100, 32)


def test_output_shape_and_dtype():
    emb = SiatEmbedding(vocab_size=100, d_model=32)
    input_ids = torch.randint(0, 100, (2, 8), dtype=torch.long)
    out = emb(input_ids)
    assert out.shape == (2, 8, 32)
    assert out.dtype == emb.weight.dtype
    assert input_ids.dtype == torch.long


def test_parameter_count_small():
    emb = SiatEmbedding(vocab_size=100, d_model=32)
    num_params = sum(p.numel() for p in emb.parameters())
    assert num_params == 100 * 32


def test_parameter_count_siat_30m():
    emb = SiatEmbedding(vocab_size=32_000, d_model=512)
    num_params = sum(p.numel() for p in emb.parameters())
    assert num_params == 32_000 * 512
    assert num_params == 16_384_000


def test_deterministic_lookup():
    emb = SiatEmbedding(vocab_size=50, d_model=16)
    input_ids = torch.tensor([[5, 5]], dtype=torch.long)
    out = emb(input_ids)
    assert torch.equal(out[0, 0], out[0, 1])


def test_different_token_rows():
    emb = SiatEmbedding(vocab_size=50, d_model=16)
    # Same module weights: different indices select different rows of the table.
    assert not torch.equal(emb.weight[3], emb.weight[7])


def test_gradient_flow():
    emb = SiatEmbedding(vocab_size=50, d_model=16)
    input_ids = torch.tensor([[3, 9, 3]], dtype=torch.long)
    out = emb(input_ids)
    loss = out.sum()
    loss.backward()
    assert emb.weight.grad is not None
    assert emb.weight.grad[3].abs().sum() > 0
    assert emb.weight.grad[9].abs().sum() > 0
    # Unused token row should have zero grad.
    assert emb.weight.grad[0].abs().sum() == 0


def test_invalid_constructor():
    with pytest.raises(ValueError, match="vocab_size"):
        SiatEmbedding(vocab_size=0, d_model=32)
    with pytest.raises(ValueError, match="d_model"):
        SiatEmbedding(vocab_size=100, d_model=-1)


def test_tiny_config_integration():
    config = ModelConfig.tiny()
    emb = SiatEmbedding(vocab_size=config.vocab_size, d_model=config.d_model)
    batch, seq = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch, seq), dtype=torch.long)
    out = emb(input_ids)
    assert out.shape == (batch, seq, config.d_model)


def test_dataloader_style_input_ids():
    """batch['input_ids'] from Dataset is long [B, S] — same contract here."""
    emb = SiatEmbedding(vocab_size=100, d_model=32)
    batch = {"input_ids": torch.randint(0, 100, (4, 8), dtype=torch.long)}
    hidden = emb(batch["input_ids"])
    assert hidden.shape == (4, 8, 32)
