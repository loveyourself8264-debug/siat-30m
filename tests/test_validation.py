"""Tests for SiatTrainer.validate."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from config import ModelConfig, TrainConfig
from model.model import SiatForCausalLM
from train.trainer import SiatTrainer


def _tiny_model() -> SiatForCausalLM:
    return SiatForCausalLM(
        ModelConfig(
            model_name="Siat",
            vocab_size=64,
            d_model=32,
            n_layers=2,
            n_heads=4,
            ffn_dim=64,
            max_seq_len=32,
            dropout=0.0,
            tie_embeddings=True,
        )
    )


def _train_cfg(**kwargs) -> TrainConfig:
    defaults = dict(
        batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=3e-3,
        min_learning_rate=3e-4,
        weight_decay=0.0,
        warmup_steps=2,
        max_steps=20,
        max_grad_norm=1.0,
        seed=42,
    )
    defaults.update(kwargs)
    return TrainConfig(**defaults)


def _loader(ids: torch.Tensor, labels: torch.Tensor, batch_size: int) -> DataLoader:
    ds = TensorDataset(ids, labels)

    def collate(samples):
        return {
            "input_ids": torch.stack([s[0] for s in samples], dim=0),
            "labels": torch.stack([s[1] for s in samples], dim=0),
        }

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)


def test_validation_loss_finite_and_ppl():
    torch.manual_seed(0)
    model = _tiny_model()
    trainer = SiatTrainer(model, _train_cfg(), device="cpu")
    ids = torch.randint(0, 64, (8, 8))
    labels = torch.randint(0, 64, (8, 8))
    metrics = trainer.validate(_loader(ids, labels, 2))
    assert torch.isfinite(torch.tensor(metrics["val_loss"]))
    assert metrics["perplexity"] > 1.0
    assert torch.isfinite(torch.tensor(metrics["perplexity"]))


def test_validation_no_grad_and_restores_train_mode():
    model = _tiny_model()
    trainer = SiatTrainer(model, _train_cfg(), device="cpu")
    model.train()
    ids = torch.randint(0, 64, (4, 8))
    labels = torch.randint(0, 64, (4, 8))
    trainer.validate(_loader(ids, labels, 2))
    assert model.training is True

    # No parameter grads from validate
    for p in model.parameters():
        assert p.grad is None


def test_validation_max_batches():
    model = _tiny_model()
    trainer = SiatTrainer(model, _train_cfg(), device="cpu")
    ids = torch.randint(0, 64, (16, 8))
    labels = torch.randint(0, 64, (16, 8))
    m1 = trainer.validate(_loader(ids, labels, 2), max_batches=1)
    m2 = trainer.validate(_loader(ids, labels, 2), max_batches=2)
    assert torch.isfinite(torch.tensor(m1["val_loss"]))
    assert torch.isfinite(torch.tensor(m2["val_loss"]))


def test_validation_empty_loader_raises():
    model = _tiny_model()
    trainer = SiatTrainer(model, _train_cfg(), device="cpu")
    ids = torch.randint(0, 64, (2, 8))
    labels = torch.randint(0, 64, (2, 8))
    loader = _loader(ids, labels, 2)
    try:
        trainer.validate(loader, max_batches=0)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "zero" in str(e).lower() or "empty" in str(e).lower()


def test_validation_deterministic():
    torch.manual_seed(1)
    model = _tiny_model()
    trainer = SiatTrainer(model, _train_cfg(), device="cpu")
    ids = torch.randint(0, 64, (8, 8))
    labels = torch.randint(0, 64, (8, 8))
    loader = _loader(ids, labels, 2)
    a = trainer.validate(loader)
    b = trainer.validate(loader)
    assert abs(a["val_loss"] - b["val_loss"]) < 1e-6
