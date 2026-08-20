"""Tests for checkpoint save/load and resume equivalence."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import ModelConfig, TrainConfig
from model.model import SiatForCausalLM
from train.trainer import SiatTrainer, get_learning_rate


def _tiny_cfg() -> ModelConfig:
    return ModelConfig(
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


def _train_cfg(**kwargs) -> TrainConfig:
    defaults = dict(
        batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=3e-3,
        min_learning_rate=3e-4,
        weight_decay=0.0,
        warmup_steps=3,
        max_steps=20,
        max_grad_norm=0.0,
        seed=42,
    )
    defaults.update(kwargs)
    return TrainConfig(**defaults)


def _fixed_loader(
    ids: torch.Tensor, labels: torch.Tensor, batch_size: int
) -> DataLoader:
    ds = TensorDataset(ids, labels)

    def collate(samples):
        return {
            "input_ids": torch.stack([s[0] for s in samples], dim=0),
            "labels": torch.stack([s[1] for s in samples], dim=0),
        }

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)


def test_checkpoint_save_contents(tmp_path: Path):
    model = SiatForCausalLM(_tiny_cfg())
    trainer = SiatTrainer(
        model, _train_cfg(), device="cpu", model_config=_tiny_cfg()
    )
    trainer.optimizer_step = 7
    trainer.tokens_processed = 1234
    path = trainer.save_checkpoint(tmp_path / "ckpt.pt")
    assert path.is_file()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert "model" in ckpt
    assert "optimizer" in ckpt
    assert ckpt["optimizer_step"] == 7
    assert ckpt["tokens_processed"] == 1234
    assert "train_config" in ckpt
    assert "model_config" in ckpt
    assert "rng_state" in ckpt


def test_checkpoint_load_restores(tmp_path: Path):
    cfg = _tiny_cfg()
    model_a = SiatForCausalLM(cfg)
    trainer_a = SiatTrainer(model_a, _train_cfg(), device="cpu", model_config=cfg)
    ids = torch.randint(0, 64, (8, 8))
    labels = torch.randint(0, 64, (8, 8))
    trainer_a.train(_fixed_loader(ids, labels, 2), log_interval=0, max_steps=3)
    path = trainer_a.save_checkpoint(tmp_path / "a.pt")

    model_b = SiatForCausalLM(cfg)
    trainer_b = SiatTrainer(model_b, _train_cfg(), device="cpu", model_config=cfg)
    trainer_b.load_checkpoint(path)

    assert trainer_b.optimizer_step == trainer_a.optimizer_step
    assert trainer_b.tokens_processed == trainer_a.tokens_processed
    for (n1, p1), (n2, p2) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        assert n1 == n2
        assert torch.equal(p1, p2)


def test_checkpoint_missing_file():
    model = SiatForCausalLM(_tiny_cfg())
    trainer = SiatTrainer(model, _train_cfg(), device="cpu")
    with pytest.raises(FileNotFoundError):
        trainer.load_checkpoint("does_not_exist_ckpt.pt")


def test_weight_tying_after_resume(tmp_path: Path):
    cfg = _tiny_cfg()
    assert cfg.tie_embeddings
    model = SiatForCausalLM(cfg)
    trainer = SiatTrainer(model, _train_cfg(), device="cpu", model_config=cfg)
    path = trainer.save_checkpoint(tmp_path / "tie.pt")

    model2 = SiatForCausalLM(cfg)
    trainer2 = SiatTrainer(model2, _train_cfg(), device="cpu", model_config=cfg)
    trainer2.load_checkpoint(path)
    assert model2.lm_head.weight is model2.embed.weight


def test_logging_tokens_increase():
    model = SiatForCausalLM(_tiny_cfg())
    trainer = SiatTrainer(model, _train_cfg(), device="cpu")
    ids = torch.randint(0, 64, (8, 8))
    labels = torch.randint(0, 64, (8, 8))
    hist = trainer.train(
        _fixed_loader(ids, labels, 2), log_interval=1, max_steps=3
    )
    assert trainer.tokens_processed > 0
    assert hist[-1]["tokens_processed"] >= hist[0]["tokens_processed"]
    assert "learning_rate" in hist[0]
    assert "grad_norm" in hist[0]


def test_resume_equivalence(tmp_path: Path):
    """10 continuous steps ≈ 5 + save + load + 5 (deterministic batches)."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    train_cfg = _train_cfg(max_steps=20, warmup_steps=3, weight_decay=0.0)

    g = torch.Generator().manual_seed(123)
    # Enough samples so cycling doesn't reshuffle order issues with fixed loader
    ids = torch.randint(0, 64, (40, 8), generator=g)
    labels = torch.randint(0, 64, (40, 8), generator=g)

    # Path A: uninterrupted 10 steps
    torch.manual_seed(99)
    model_a = SiatForCausalLM(cfg)
    trainer_a = SiatTrainer(
        model_a, copy.deepcopy(train_cfg), device="cpu", model_config=cfg
    )
    loader_a = _fixed_loader(ids.clone(), labels.clone(), batch_size=2)
    trainer_a.train(loader_a, log_interval=0, max_steps=10)
    params_a = {n: p.detach().clone() for n, p in model_a.named_parameters()}
    lr_a_next = get_learning_rate(10, train_cfg)  # LR that would be used at step index 10

    # Path B: 5 steps → save → new trainer → load → 5 more
    torch.manual_seed(99)
    model_b1 = SiatForCausalLM(cfg)
    trainer_b1 = SiatTrainer(
        model_b1, copy.deepcopy(train_cfg), device="cpu", model_config=cfg
    )
    loader_b1 = _fixed_loader(ids.clone(), labels.clone(), batch_size=2)
    trainer_b1.train(loader_b1, log_interval=0, max_steps=5)
    ckpt = trainer_b1.save_checkpoint(tmp_path / "mid.pt")
    assert trainer_b1.optimizer_step == 5

    model_b2 = SiatForCausalLM(cfg)
    trainer_b2 = SiatTrainer(
        model_b2, copy.deepcopy(train_cfg), device="cpu", model_config=cfg
    )
    trainer_b2.load_checkpoint(ckpt)
    assert trainer_b2.optimizer_step == 5
    # LR for next update (optimizer_step index 5) must match uninterrupted path
    lr_b_next = get_learning_rate(trainer_b2.optimizer_step, train_cfg)
    assert abs(lr_b_next - get_learning_rate(5, train_cfg)) < 1e-15

    # Continue with same data stream position: after 5 steps with B=2, used 10 samples
    # Infinite cycle from start — for equivalence we need the same batch sequence.
    # After 5 opt steps with accum=1 and B=2, each step consumes 1 batch of 2.
    # So 5 batches from the start were used. Resume must continue from batch index 5.
    # Our infinite loader always restarts from 0 — that breaks equivalence!
    #
    # Fix: use a custom repeating iterator via training only on a Dataset large enough
    # and the same cyclic order. Both paths cycle from start each `train()` call.
    # For path B second half, we must feed batches starting at index 5.
    #
    # Simpler approach: pass the same full loader but start train from step 5 with
    # a sliced dataset that begins where path A would be after 5 steps.
    # With shuffle=False and cycling: batch order is 0,1,2,... then wrap.
    # After 5 steps, next batch index is 5. Dataset has 20 batches (40/2).
    remaining_ids = ids[10:]  # skip first 5 batches * 2 samples
    remaining_labels = labels[10:]
    # Also need wrap-around samples that path A would see if it wraps — for 10 steps
    # only 10 batches needed, so remaining 15 batches from index 5 is enough if we
    # concat wrap: ids[10:] + ids[:10]
    cont_ids = torch.cat([ids[10:], ids[:10]], dim=0)
    cont_labels = torch.cat([labels[10:], labels[:10]], dim=0)
    loader_b2 = _fixed_loader(cont_ids, cont_labels, batch_size=2)
    trainer_b2.train(loader_b2, log_interval=0, max_steps=10)

    params_b = {n: p.detach().clone() for n, p in model_b2.named_parameters()}

    max_diff = 0.0
    for name in params_a:
        diff = (params_a[name] - params_b[name]).abs().max().item()
        max_diff = max(max_diff, diff)
        assert torch.allclose(params_a[name], params_b[name], atol=1e-5, rtol=1e-4), (
            f"{name} max_diff={diff}"
        )

    # Optimizer state restored: Adam exp_avg should exist after resume+train
    assert trainer_b2.optimizer.state
    # LR continuity for the 6th update was checked via get_learning_rate(5)
    assert abs(get_learning_rate(5, train_cfg) - lr_b_next) < 1e-15
    _ = lr_a_next  # documented: uninterrupted would use get_learning_rate(10) later
    assert max_diff < 1e-4


def test_arch_mismatch_raises(tmp_path: Path):
    cfg = _tiny_cfg()
    model = SiatForCausalLM(cfg)
    trainer = SiatTrainer(model, _train_cfg(), device="cpu", model_config=cfg)
    path = trainer.save_checkpoint(tmp_path / "m.pt")

    other = ModelConfig(
        model_name="Siat",
        vocab_size=64,
        d_model=32,
        n_layers=1,  # mismatch
        n_heads=4,
        ffn_dim=64,
        max_seq_len=32,
        dropout=0.0,
        tie_embeddings=True,
    )
    model2 = SiatForCausalLM(other)
    trainer2 = SiatTrainer(model2, _train_cfg(), device="cpu", model_config=other)
    with pytest.raises(ValueError, match="n_layers"):
        trainer2.load_checkpoint(path)
